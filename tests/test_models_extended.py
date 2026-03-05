"""
Extended tests for adsorblab_pro/models.py - targeting 80%+ coverage.

Covers:
- Model registry (register_model, get_model_by_name)
- Multicomponent models (extended_langmuir, extended_freundlich)
- Selectivity coefficient (array inputs, edge cases)
- rPSO and rPSO_fixed_conditions
- 3D surface generators
- _fit_model_core edge cases (singular covariance, insufficient data, dof)
- _fit_model_cached_impl
- fit_model_with_ci (cache and non-cache paths)
- calculate_initial_rate / identify_equilibrium_time edge cases
- get_model_info completeness
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# ---------------------------------------------------------------------------
# Imports from the package under test
# ---------------------------------------------------------------------------
from adsorblab_pro.models import (
    _MODEL_REGISTRY,
    _fit_model_core,
    _fit_model_cached_impl,
    calculate_biot_number,
    calculate_initial_rate,
    calculate_selectivity_coefficient,
    elovich_model,
    extended_freundlich_multicomponent,
    extended_langmuir_multicomponent,
    fit_model_with_ci,
    freundlich_model,
    get_model_by_name,
    get_model_info,
    identify_equilibrium_time,
    identify_rate_limiting_step,
    ipd_model,
    langmuir_3d_surface,
    langmuir_model,
    parameter_space_visualization,
    pfo_model,
    ph_temperature_response_surface,
    pso_model,
    register_model,
    revised_pso_model,
    revised_pso_model_fixed_conditions,
    sips_model,
    temkin_model,
)


# =============================================================================
# MODEL REGISTRY
# =============================================================================
class TestModelRegistry:
    def test_get_model_by_name_langmuir(self):
        func = get_model_by_name("Langmuir")
        assert func is langmuir_model

    def test_get_model_by_name_freundlich(self):
        func = get_model_by_name("Freundlich")
        assert func is freundlich_model

    def test_get_model_by_name_temkin(self):
        func = get_model_by_name("Temkin")
        assert func is temkin_model

    def test_get_model_by_name_sips(self):
        func = get_model_by_name("Sips")
        assert func is sips_model

    def test_get_model_by_name_pfo(self):
        func = get_model_by_name("PFO")
        assert func is pfo_model

    def test_get_model_by_name_pso(self):
        func = get_model_by_name("PSO")
        assert func is pso_model

    def test_get_model_by_name_rpso(self):
        func = get_model_by_name("rPSO")
        assert func is revised_pso_model

    def test_get_model_by_name_elovich(self):
        func = get_model_by_name("Elovich")
        assert func is elovich_model

    def test_get_model_by_name_ipd(self):
        func = get_model_by_name("IPD")
        assert func is ipd_model

    def test_get_model_by_name_unknown_returns_none(self):
        assert get_model_by_name("NonExistent") is None

    def test_register_model_decorator(self):
        @register_model("_test_model_xyz")
        def dummy(x, a):
            return a * x

        assert "_test_model_xyz" in _MODEL_REGISTRY
        assert _MODEL_REGISTRY["_test_model_xyz"] is dummy
        # Cleanup
        del _MODEL_REGISTRY["_test_model_xyz"]


# =============================================================================
# MULTICOMPONENT MODELS
# =============================================================================
class TestExtendedLangmuirMulticomponent:
    def test_basic_two_component(self):
        Ce_dye = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        Ce_metal = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        qm_dye, KL_dye = 100.0, 0.05
        KL_metal = 0.1

        qe = extended_langmuir_multicomponent(
            Ce_i=Ce_dye,
            qm_i=qm_dye,
            KL_i=KL_dye,
            Ce_all=[Ce_dye, Ce_metal],
            KL_all=[KL_dye, KL_metal],
        )
        assert qe.shape == Ce_dye.shape
        assert np.all(qe > 0)
        assert np.all(qe <= qm_dye)

    def test_competition_reduces_capacity(self):
        Ce = np.array([10.0, 20.0, 30.0])
        qm, KL = 100.0, 0.05

        # Single component (no competition)
        qe_single = extended_langmuir_multicomponent(
            Ce_i=Ce,
            qm_i=qm,
            KL_i=KL,
            Ce_all=[Ce],
            KL_all=[KL],
        )
        # With competitor
        Ce_comp = np.array([5.0, 10.0, 15.0])
        qe_comp = extended_langmuir_multicomponent(
            Ce_i=Ce,
            qm_i=qm,
            KL_i=KL,
            Ce_all=[Ce, Ce_comp],
            KL_all=[KL, 0.1],
        )
        # Competition should reduce capacity
        assert np.all(qe_comp <= qe_single + 1e-10)

    def test_zero_competitor_concentration(self):
        Ce = np.array([10.0, 20.0])
        Ce_zero = np.array([0.0, 0.0])
        qe = extended_langmuir_multicomponent(
            Ce_i=Ce,
            qm_i=100.0,
            KL_i=0.05,
            Ce_all=[Ce, Ce_zero],
            KL_all=[0.05, 0.1],
        )
        assert np.all(np.isfinite(qe))

    def test_scalar_like_input(self):
        qe = extended_langmuir_multicomponent(
            Ce_i=np.array([10.0]),
            qm_i=50.0,
            KL_i=0.05,
            Ce_all=[np.array([10.0])],
            KL_all=[0.05],
        )
        assert qe.shape == (1,)


class TestExtendedFreundlichMulticomponent:
    def test_basic_two_component(self):
        Ce_1 = np.array([10.0, 20.0, 30.0, 40.0])
        Ce_2 = np.array([5.0, 10.0, 15.0, 20.0])
        Kf_1, n_1 = 5.0, 0.5
        Kf_2, n_2 = 3.0, 0.6

        qe = extended_freundlich_multicomponent(
            Ce_i=Ce_1,
            Kf_i=Kf_1,
            n_i=n_1,
            Ce_all=[Ce_1, Ce_2],
            Kf_all=[Kf_1, Kf_2],
            n_all=[n_1, n_2],
        )
        assert qe.shape == Ce_1.shape
        assert np.all(qe > 0)
        assert np.all(np.isfinite(qe))

    def test_single_component(self):
        Ce = np.array([10.0, 20.0, 30.0])
        qe = extended_freundlich_multicomponent(
            Ce_i=Ce,
            Kf_i=5.0,
            n_i=0.5,
            Ce_all=[Ce],
            Kf_all=[5.0],
            n_all=[0.5],
        )
        assert np.all(np.isfinite(qe))
        assert np.all(qe > 0)

    def test_zero_concentration(self):
        Ce = np.array([0.0, 10.0, 20.0])
        qe = extended_freundlich_multicomponent(
            Ce_i=Ce,
            Kf_i=5.0,
            n_i=0.5,
            Ce_all=[Ce],
            Kf_all=[5.0],
            n_all=[0.5],
        )
        assert np.all(np.isfinite(qe))


# =============================================================================
# SELECTIVITY COEFFICIENT
# =============================================================================
class TestSelectivityCoefficientExtended:
    def test_scalar_inputs(self):
        alpha = calculate_selectivity_coefficient(
            qe_i=80.0,
            Ce_i=10.0,
            qe_j=20.0,
            Ce_j=20.0,
        )
        assert isinstance(alpha, float)
        assert alpha == pytest.approx(8.0)

    def test_array_inputs(self):
        qe_i = np.array([80.0, 60.0])
        Ce_i = np.array([10.0, 15.0])
        qe_j = np.array([20.0, 30.0])
        Ce_j = np.array([20.0, 25.0])
        alpha = calculate_selectivity_coefficient(qe_i, Ce_i, qe_j, Ce_j)
        assert alpha.shape == (2,)
        assert np.all(np.isfinite(alpha))

    def test_zero_concentration_returns_nan(self):
        alpha = calculate_selectivity_coefficient(
            qe_i=80.0,
            Ce_i=0.0,
            qe_j=20.0,
            Ce_j=20.0,
        )
        assert np.isnan(alpha)

    def test_zero_qe_j_returns_nan(self):
        alpha = calculate_selectivity_coefficient(
            qe_i=80.0,
            Ce_i=10.0,
            qe_j=0.0,
            Ce_j=20.0,
        )
        assert np.isnan(alpha)

    def test_equal_preference(self):
        alpha = calculate_selectivity_coefficient(
            qe_i=50.0,
            Ce_i=10.0,
            qe_j=50.0,
            Ce_j=10.0,
        )
        assert alpha == pytest.approx(1.0)


# =============================================================================
# REVISED PSO MODELS
# =============================================================================
class TestRevisedPSOModel:
    def test_basic_rpso(self):
        t = np.array([0.0, 5.0, 10.0, 30.0, 60.0, 120.0])
        qt = revised_pso_model(t, qe=50.0, k2=0.01, C0=100.0, m=0.5, V=0.1)
        assert qt.shape == t.shape
        assert qt[0] == pytest.approx(0.0, abs=1e-6)
        assert np.all(np.diff(qt) >= 0)  # monotonic increasing

    def test_rpso_approaches_equilibrium(self):
        t = np.array([0.0, 100.0, 1000.0, 10000.0])
        qt = revised_pso_model(t, qe=50.0, k2=0.01, C0=100.0, m=0.5, V=0.1)
        # Should approach but not exceed qe
        assert qt[-1] < 50.0

    def test_rpso_epsilon_protection(self):
        t = np.array([0.0, 10.0])
        # All zeros should be protected by EPSILON
        qt = revised_pso_model(t, qe=0.0, k2=0.0, C0=0.0, m=0.0, V=0.0)
        assert np.all(np.isfinite(qt))


class TestRevisedPSOFixedConditions:
    def test_creates_callable(self):
        model = revised_pso_model_fixed_conditions(C0=100.0, m=0.5, V=0.1)
        assert callable(model)

    def test_fixed_conditions_output(self):
        model = revised_pso_model_fixed_conditions(C0=100.0, m=0.5, V=0.1)
        t = np.array([0.0, 5.0, 10.0, 30.0, 60.0])
        qt = model(t, qe=50.0, k2=0.01)
        assert qt.shape == t.shape
        assert np.all(np.isfinite(qt))

    def test_registered_in_registry(self):
        revised_pso_model_fixed_conditions(C0=100.0, m=0.5, V=0.1)
        # Check that the registry key exists
        found = any(k.startswith("rPSO_C0=") for k in _MODEL_REGISTRY)
        assert found

    def test_matches_full_rpso(self):
        model = revised_pso_model_fixed_conditions(C0=100.0, m=0.5, V=0.1)
        t = np.array([0.0, 5.0, 10.0, 30.0])
        qt_fixed = model(t, qe=50.0, k2=0.01)
        qt_full = revised_pso_model(t, qe=50.0, k2=0.01, C0=100.0, m=0.5, V=0.1)
        assert_allclose(qt_fixed, qt_full)


# =============================================================================
# 3D SURFACE GENERATORS
# =============================================================================
class TestLangmuir3DSurface:
    def test_basic_output_shape(self):
        Ce_grid, T_grid, qe_grid = langmuir_3d_surface(
            Ce_range=(1.0, 100.0),
            temp_range=(20.0, 60.0),
            qm=100.0,
            KL=0.05,
        )
        assert Ce_grid.shape == (30, 30)
        assert T_grid.shape == (30, 30)
        assert qe_grid.shape == (30, 30)

    def test_values_positive(self):
        _, _, qe_grid = langmuir_3d_surface(
            Ce_range=(1.0, 100.0),
            temp_range=(20.0, 60.0),
            qm=100.0,
            KL=0.05,
        )
        assert np.all(qe_grid > 0)
        assert np.all(np.isfinite(qe_grid))

    def test_exothermic_process(self):
        # Exothermic: capacity should decrease with temperature
        _, T_grid, qe_grid = langmuir_3d_surface(
            Ce_range=(50.0, 50.0),
            temp_range=(20.0, 60.0),
            qm=100.0,
            KL=0.05,
            delta_H=-25000,
        )
        # At fixed Ce, qe should decrease as T increases (exothermic)
        qe_col = qe_grid[:, 0]
        assert qe_col[0] > qe_col[-1]

    def test_endothermic_process(self):
        _, T_grid, qe_grid = langmuir_3d_surface(
            Ce_range=(50.0, 50.0),
            temp_range=(20.0, 60.0),
            qm=100.0,
            KL=0.05,
            delta_H=25000,
        )
        qe_col = qe_grid[:, 0]
        assert qe_col[-1] > qe_col[0]


class TestPHTemperatureResponseSurface:
    def test_basic_output_shape(self):
        pH_grid, T_grid, response = ph_temperature_response_surface(
            pH_range=(2.0, 12.0),
            temp_range=(20.0, 60.0),
        )
        assert pH_grid.shape == (25, 25)
        assert T_grid.shape == (25, 25)
        assert response.shape == (25, 25)

    def test_positive_values(self):
        _, _, response = ph_temperature_response_surface(
            pH_range=(2.0, 12.0),
            temp_range=(20.0, 60.0),
        )
        assert np.all(response >= 0)

    def test_max_at_optimal(self):
        pH_grid, T_grid, response = ph_temperature_response_surface(
            pH_range=(2.0, 12.0),
            temp_range=(20.0, 60.0),
            optimal_pH=6.0,
            optimal_temp=40.0,
            max_capacity=100.0,
        )
        max_idx = np.unravel_index(np.argmax(response), response.shape)
        assert response[max_idx] <= 100.0


class TestParameterSpaceVisualization:
    def test_with_langmuir(self):
        p1_grid, p2_grid, qe_grid = parameter_space_visualization(
            model_func=langmuir_model,
            param1_range=(10.0, 200.0),
            param2_range=(0.01, 0.2),
            Ce_fixed=50.0,
        )
        assert p1_grid.shape == (25, 25)
        assert np.all(np.isfinite(qe_grid))

    def test_fallback_for_non_vectorizable(self):
        def bad_model(Ce, a, b):
            if isinstance(a, np.ndarray):
                raise ValueError("not vectorizable")
            return a * Ce / (1 + b * Ce)

        p1, p2, qe = parameter_space_visualization(
            model_func=bad_model,
            param1_range=(10.0, 100.0),
            param2_range=(0.01, 0.1),
            Ce_fixed=50.0,
        )
        assert qe.shape == (25, 25)


# =============================================================================
# FIT MODEL CORE
# =============================================================================
class TestFitModelCore:
    @pytest.fixture
    def langmuir_data(self):
        Ce = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0, 150.0])
        qm_true, KL_true = 100.0, 0.05
        qe = langmuir_model(Ce, qm_true, KL_true)
        noise = np.random.RandomState(42).normal(0, 1, len(Ce))
        return Ce, qe + noise

    def test_successful_fit(self, langmuir_data):
        Ce, qe = langmuir_data
        result = _fit_model_core(
            langmuir_model,
            Ce,
            qe,
            p0=[80.0, 0.03],
            param_names=["qm", "KL"],
        )
        assert result is not None
        assert result["converged"] is True
        assert result["r_squared"] > 0.95
        assert "qm" in result["params"]
        assert "KL" in result["params"]

    def test_insufficient_data(self):
        Ce = np.array([1.0, 5.0])
        qe = np.array([4.5, 17.5])
        result = _fit_model_core(langmuir_model, Ce, qe, p0=[100.0, 0.05])
        assert result is None

    def test_with_bounds(self, langmuir_data):
        Ce, qe = langmuir_data
        result = _fit_model_core(
            langmuir_model,
            Ce,
            qe,
            p0=[80.0, 0.03],
            bounds=((0, 0), (500, 1)),
            param_names=["qm", "KL"],
        )
        assert result is not None
        assert result["converged"] is True

    def test_without_param_names(self, langmuir_data):
        Ce, qe = langmuir_data
        result = _fit_model_core(langmuir_model, Ce, qe, p0=[80.0, 0.03])
        assert result is not None
        assert "p0" in result["params"]
        assert "p1" in result["params"]

    def test_result_keys(self, langmuir_data):
        Ce, qe = langmuir_data
        result = _fit_model_core(
            langmuir_model,
            Ce,
            qe,
            p0=[80.0, 0.03],
            param_names=["qm", "KL"],
        )
        expected_keys = {
            "params",
            "popt",
            "pcov",
            "perr",
            "ci_95",
            "y_pred",
            "y_data",
            "x_data",
            "residuals",
            "r_squared",
            "adj_r_squared",
            "rmse",
            "chi_squared",
            "aic",
            "aicc",
            "bic",
            "sse",
            "sst",
            "n_points",
            "num_params",
            "dof",
            "converged",
        }
        assert expected_keys.issubset(result.keys())

    def test_bad_model_returns_converged_false(self):
        def always_fails(x, a, b):
            raise RuntimeError("boom")

        Ce = np.array([1.0, 5.0, 10.0, 20.0, 50.0])
        qe = np.array([5.0, 17.0, 25.0, 35.0, 42.0])
        result = _fit_model_core(always_fails, Ce, qe, p0=[1.0, 1.0])
        assert result is not None
        assert result["converged"] is False
        assert "error" in result

    def test_ci_95_contains_param_names(self, langmuir_data):
        Ce, qe = langmuir_data
        result = _fit_model_core(
            langmuir_model,
            Ce,
            qe,
            p0=[80.0, 0.03],
            param_names=["qm", "KL"],
        )
        assert "qm" in result["ci_95"]
        assert "KL" in result["ci_95"]
        # CI should bracket the fitted value
        qm_ci = result["ci_95"]["qm"]
        assert qm_ci[0] < result["params"]["qm"] < qm_ci[1]


class TestFitModelCachedImpl:
    def test_unknown_model_name(self):
        result = _fit_model_cached_impl(
            "NonExistentModel",
            (1.0, 5.0, 10.0, 20.0, 50.0),
            (5.0, 17.0, 25.0, 35.0, 42.0),
            (100.0, 0.05),
        )
        assert result is not None
        assert result["converged"] is False
        assert "Unknown model" in result["error"]

    def test_langmuir_cached(self):
        Ce = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0])
        qm_true, KL_true = 100.0, 0.05
        qe = langmuir_model(Ce, qm_true, KL_true)

        result = _fit_model_cached_impl(
            "Langmuir",
            tuple(Ce.tolist()),
            tuple(qe.tolist()),
            (80.0, 0.03),
            bounds_lower=(0.0, 0.0),
            bounds_upper=(500.0, 1.0),
            param_names_tuple=("qm", "KL"),
        )
        assert result is not None
        assert result["converged"] is True

    def test_without_bounds_or_param_names(self):
        Ce = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0])
        qe = langmuir_model(Ce, 100.0, 0.05)
        result = _fit_model_cached_impl(
            "Langmuir",
            tuple(Ce.tolist()),
            tuple(qe.tolist()),
            (80.0, 0.03),
        )
        assert result is not None
        assert result["converged"] is True


class TestFitModelWithCI:
    def test_unregistered_model_direct(self):
        """Unregistered model should fall back to direct _fit_model_core."""

        def custom_model(x, a, b):
            return a * x / (1 + b * x)

        Ce = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0])
        qe = custom_model(Ce, 100.0, 0.05)

        result = fit_model_with_ci(
            custom_model,
            Ce,
            qe,
            p0=[80.0, 0.03],
            param_names=["a", "b"],
            use_cache=False,
        )
        assert result is not None
        assert result["converged"] is True

    def test_registered_model(self):
        Ce = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0])
        qe = langmuir_model(Ce, 100.0, 0.05) + np.random.RandomState(42).normal(0, 0.5, 7)

        result = fit_model_with_ci(
            langmuir_model,
            Ce,
            qe,
            p0=[80.0, 0.03],
            param_names=["qm", "KL"],
        )
        assert result is not None
        assert result["converged"] is True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
class TestCalculateInitialRateExtended:
    def test_too_few_points(self):
        rate = calculate_initial_rate(np.array([0.0, 1.0]), np.array([0.0, 5.0]))
        assert rate == 0.0

    def test_constant_time(self):
        rate = calculate_initial_rate(np.array([0.0, 0.0, 0.0]), np.array([0.0, 5.0, 10.0]))
        assert rate == 0.0

    def test_low_correlation(self):
        # Random data → low R
        t = np.array([0.0, 1.0, 2.0, 3.0])
        qt = np.array([0.0, 10.0, 2.0, 8.0])
        rate = calculate_initial_rate(t, qt)
        # Could be 0 due to low r_value
        assert isinstance(rate, float)

    def test_good_linear_region(self):
        t = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
        qt = np.array([0.0, 5.0, 10.0, 15.0, 40.0])
        rate = calculate_initial_rate(t, qt)
        assert rate > 0


class TestIdentifyEquilibriumTimeExtended:
    def test_empty_data(self):
        assert identify_equilibrium_time(np.array([]), np.array([])) == 0.0

    def test_all_zero_qt(self):
        t = np.array([0.0, 1.0, 2.0])
        qt = np.array([0.0, 0.0, 0.0])
        result = identify_equilibrium_time(t, qt)
        assert result == 2.0  # returns t[-1]

    def test_quick_equilibrium(self):
        t = np.array([0.0, 1.0, 5.0, 10.0, 30.0])
        qt = np.array([0.0, 48.0, 49.0, 49.5, 50.0])
        teq = identify_equilibrium_time(t, qt, threshold=0.95)
        assert teq == 1.0

    def test_slow_equilibrium(self):
        t = np.array([0.0, 5.0, 10.0, 30.0, 60.0, 120.0])
        qt = np.array([0.0, 10.0, 20.0, 35.0, 45.0, 50.0])
        teq = identify_equilibrium_time(t, qt, threshold=0.95)
        assert teq == 120.0


class TestGetModelInfoExtended:
    def test_has_isotherms_and_kinetics(self):
        info = get_model_info()
        assert "isotherms" in info
        assert "kinetics" in info

    def test_isotherm_models_present(self):
        info = get_model_info()
        for model in ["Langmuir", "Freundlich", "Temkin", "Sips"]:
            assert model in info["isotherms"]

    def test_kinetic_models_present(self):
        info = get_model_info()
        for model in ["PFO", "PSO", "rPSO", "Elovich", "IPD"]:
            assert model in info["kinetics"]

    def test_model_has_required_fields(self):
        info = get_model_info()
        for model_type in ["isotherms", "kinetics"]:
            for name, details in info[model_type].items():
                assert "equation" in details, f"{name} missing equation"
                assert "params" in details, f"{name} missing params"
                assert "description" in details, f"{name} missing description"


# =============================================================================
# RATE LIMITING STEP IDENTIFICATION
# =============================================================================
class TestIdentifyRateLimitingStepExtended:
    def test_with_particle_radius(self):
        t = np.array([0, 5, 10, 20, 30, 60, 90, 120, 180, 240], dtype=float)
        qt = pso_model(t, qe=50.0, k2=0.01)
        result = identify_rate_limiting_step(t, qt, particle_radius=0.5e-3)
        assert isinstance(result, dict)
        assert "mechanism" in result

    def test_with_ipd_stages(self):
        t = np.array([1, 4, 9, 16, 25, 36, 49, 64, 100, 144], dtype=float)
        qt = np.array([5, 10, 14, 17, 19, 20, 20.5, 21, 21.3, 21.5])
        result = identify_rate_limiting_step(t, qt)
        assert "ipd_stages" in result or "mechanism" in result

    def test_insufficient_data(self):
        t = np.array([0.0, 1.0])
        qt = np.array([0.0, 5.0])
        result = identify_rate_limiting_step(t, qt)
        assert isinstance(result, dict)


# =============================================================================
# BIOT NUMBER
# =============================================================================
class TestBiotNumberExtended:
    def test_high_biot(self):
        # High kf relative to Dp/r → film diffusion not limiting
        Bi = calculate_biot_number(kf=0.01, Dp=1e-10, r=1e-3)
        assert Bi > 1

    def test_low_biot(self):
        # Low kf relative to Dp/r → film diffusion limiting
        Bi = calculate_biot_number(kf=1e-6, Dp=1e-8, r=1e-3)
        assert isinstance(Bi, float)
        assert np.isfinite(Bi)


# =============================================================================
# NUMERICAL STABILITY EXTENSIONS
# =============================================================================
class TestNumericalStabilityExtended:
    def test_langmuir_very_large_Ce(self):
        Ce = np.array([1e6, 1e8, 1e10])
        qe = langmuir_model(Ce, qm=100.0, KL=0.05)
        assert np.all(np.isfinite(qe))
        assert np.all(qe <= 100.0 + 1e-6)

    def test_freundlich_very_small_Ce(self):
        Ce = np.array([1e-10, 1e-8, 1e-6])
        qe = freundlich_model(Ce, KF=5.0, n_inv=0.5)
        assert np.all(np.isfinite(qe))

    def test_sips_n_equals_1_reduces_to_langmuir(self):
        Ce = np.array([5.0, 10.0, 50.0, 100.0])
        qe_sips = sips_model(Ce, qm=100.0, Ks=0.05, ns=1.0)
        qe_lang = langmuir_model(Ce, qm=100.0, KL=0.05)
        assert_allclose(qe_sips, qe_lang, rtol=0.01)

    def test_temkin_with_very_small_KT(self):
        Ce = np.array([10.0, 50.0, 100.0])
        qe = temkin_model(Ce, B1=20.0, KT=1e-10)
        assert np.all(np.isfinite(qe))

    def test_elovich_monotonic(self):
        t = np.array([1.0, 5.0, 10.0, 30.0, 60.0, 120.0])
        qt = elovich_model(t, alpha=10.0, beta=0.1)
        assert np.all(np.diff(qt) >= 0)

    def test_ipd_sqrt_dependency(self):
        t = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
        qt = ipd_model(t, kid=3.0, C=5.0)
        # qt = kid * sqrt(t) + C
        expected = 3.0 * np.sqrt(t) + 5.0
        assert_allclose(qt, expected, rtol=1e-6)

    def test_pfo_negative_time_protection(self):
        t = np.array([-1.0, 0.0, 5.0])
        qt = pfo_model(t, qe=50.0, k1=0.1)
        assert np.all(np.isfinite(qt))

    def test_pso_large_time(self):
        t = np.array([0.0, 1e6])
        qt = pso_model(t, qe=50.0, k2=0.01)
        assert qt[-1] == pytest.approx(50.0, rel=0.01)
