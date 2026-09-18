# tests/test_scientific_corrections.py
"""
Regression tests for the scientific-correctness fixes.
=======================================================

Each test here pins a specific defect that produced wrong numbers or wrong
labels in reported results. They are grouped by the defect rather than by the
module, so a failure names the science that broke.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adsorblab_pro.models import (  # noqa: E402
    fit_model_with_ci,
    langmuir_model,
    mass_balance_capacity,
    pso_model,
    revised_pso_equilibrium_capacity,
    revised_pso_model,
    revised_pso_model_fixed_conditions,
    revised_pso_qe_parameter,
    sips_model,
)
from adsorblab_pro.utils import check_mechanism_consistency, sign_label  # noqa: E402


# =============================================================================
# AICc PARAMETER COUNT
# =============================================================================
class TestAiccParameterCount:
    """k must be n_params + 1: the residual variance is an estimated parameter."""

    @staticmethod
    def _fit(model, x, y, p0, bounds):
        return fit_model_with_ci(model, x, y, p0=p0, bounds=bounds)

    def test_aicc_uses_k_equals_nparams_plus_one(self):
        Ce = np.array([2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 90.0, 120.0])
        qe = langmuir_model(Ce, 80.0, 0.05)
        rng = np.random.default_rng(3)
        qe = qe + rng.normal(0, 0.4, len(Ce))

        result = self._fit(langmuir_model, Ce, qe, [70.0, 0.05], ([0, 0], [300, 5]))
        assert result["converged"]

        n = result["n_points"]
        k = result["num_params"] + 1
        expected_correction = (2 * k * (k + 1)) / (n - k - 1)
        assert result["aicc"] - result["aic"] == pytest.approx(expected_correction)

    def test_penalty_terms_include_the_variance_parameter(self):
        Ce = np.array([2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 90.0, 120.0])
        qe = langmuir_model(Ce, 80.0, 0.05) + 0.3

        result = self._fit(langmuir_model, Ce, qe, [70.0, 0.05], ([0, 0], [300, 5]))
        n, p = result["n_points"], result["num_params"]
        k = p + 1

        log_lik = -n / 2 * np.log(2 * np.pi) - n / 2 * np.log(result["sse"] / n) - n / 2
        assert result["aic"] == pytest.approx(-2 * log_lik + 2 * k)
        assert result["bic"] == pytest.approx(-2 * log_lik + k * np.log(n))

    def test_extra_parameter_is_penalised_by_the_correct_margin(self):
        """At n=8, the p=2 -> p=3 AICc correction gap must be 7.33, not 3.60."""
        n = 8
        gap = {}
        for p in (2, 3):
            k = p + 1
            gap[p] = (2 * k * (k + 1)) / (n - k - 1)
        assert gap[3] - gap[2] == pytest.approx(7.333, abs=1e-3)

    def test_aicc_is_inf_when_undefined_rather_than_falling_back_to_aic(self):
        """n = p + 2 leaves the AICc denominator at zero; report inf, not AIC."""
        Ce = np.array([2.0, 10.0, 40.0, 90.0, 150.0])
        qe = sips_model(Ce, 80.0, 0.05, 0.9)

        result = fit_model_with_ci(
            sips_model, Ce, qe, p0=[70.0, 0.05, 1.0], bounds=([0, 0, 0.1], [300, 5, 5])
        )
        if result.get("converged"):
            assert result["num_params"] == 3
            assert result["n_points"] == 5
            assert np.isinf(result["aicc"])
            assert np.isfinite(result["aic"])


# =============================================================================
# rPSO: PLATEAU, BOUNDS, AND BOUND DETECTION
# =============================================================================
class TestRevisedPsoCapacity:
    C0, m, V = 100.0, 0.1, 0.1  # -> Q = 100 mg/g

    def test_curve_plateaus_at_qe_over_phi_not_at_qe(self):
        qe, k2 = 100.0, 0.01
        phi = 1 + (qe * self.m) / (self.C0 * self.V)
        late = revised_pso_model(np.array([1e9]), qe, k2, self.C0, self.m, self.V)
        assert late[0] == pytest.approx(qe / phi, rel=1e-6)
        assert late[0] < qe

    def test_equilibrium_capacity_helper_matches_the_curve(self):
        qe, k2 = 250.0, 0.005
        predicted = revised_pso_equilibrium_capacity(qe, self.C0, self.m, self.V)
        late = revised_pso_model(np.array([1e9]), qe, k2, self.C0, self.m, self.V)
        assert predicted == pytest.approx(late[0], rel=1e-6)

    def test_equilibrium_capacity_never_exceeds_mass_balance_ceiling(self):
        Q = mass_balance_capacity(self.C0, self.m, self.V)
        for qe in (1.0, 50.0, 500.0, 1e6):
            assert revised_pso_equilibrium_capacity(qe, self.C0, self.m, self.V) < Q

    def test_qe_parameter_round_trips_through_the_capacity(self):
        for target in (10.0, 50.0, 88.0, 99.0):
            param = revised_pso_qe_parameter(target, self.C0, self.m, self.V)
            back = revised_pso_equilibrium_capacity(param, self.C0, self.m, self.V)
            assert back == pytest.approx(target, rel=1e-9)

    def test_qe_parameter_is_infinite_at_or_above_the_ceiling(self):
        Q = mass_balance_capacity(self.C0, self.m, self.V)
        assert np.isinf(revised_pso_qe_parameter(Q, self.C0, self.m, self.V))
        assert np.isinf(revised_pso_qe_parameter(Q * 1.5, self.C0, self.m, self.V))

    def test_old_bound_of_three_times_qe_exp_could_not_reach_the_plateau(self):
        """The defect: above 66.7% removal, 3*qe_exp caps the plateau too low."""
        Q = mass_balance_capacity(self.C0, self.m, self.V)
        qe_exp = 0.88 * Q  # 88% removal
        reachable = revised_pso_equilibrium_capacity(3 * qe_exp, self.C0, self.m, self.V)
        assert reachable < qe_exp

        # ...and the threshold is exactly two-thirds of the ceiling.
        at_threshold = revised_pso_equilibrium_capacity((2 / 3) * Q * 3, self.C0, self.m, self.V)
        assert at_threshold == pytest.approx((2 / 3) * Q, rel=1e-9)

    def test_mass_balance_bounds_recover_the_fit_on_high_removal_data(self):
        """With capacity-derived bounds, rPSO matches PSO instead of pinning."""
        rng = np.random.default_rng(0)
        t = np.array([5, 10, 20, 30, 45, 60, 90, 120, 180, 240.0])
        qt = pso_model(t, 90.0, 0.002) + rng.normal(0, 0.5, len(t))
        qe_exp = qt.max()
        Q = mass_balance_capacity(self.C0, self.m, self.V)
        assert qe_exp / Q > 0.667, "fixture must exceed the 66.7% removal threshold"

        model = revised_pso_model_fixed_conditions(self.C0, self.m, self.V)
        upper = revised_pso_qe_parameter(0.99 * Q, self.C0, self.m, self.V)
        start = revised_pso_qe_parameter(qe_exp, self.C0, self.m, self.V)

        result = fit_model_with_ci(
            model, t, qt, p0=[start, 0.01], bounds=([0, 0], [upper, 10]), param_names=["qe", "k2"]
        )
        assert result["converged"]
        assert result["bounds_hit"] == []
        assert result["r_squared"] > 0.99

        plateau = revised_pso_equilibrium_capacity(result["params"]["qe"], self.C0, self.m, self.V)
        assert plateau == pytest.approx(qe_exp, rel=0.05)
        assert plateau < Q


class TestBoundsHitDetection:
    def test_bound_pinned_fit_is_reported(self):
        t = np.array([5, 10, 20, 30, 45, 60, 90, 120.0])
        qt = pso_model(t, 90.0, 0.002)
        # Upper bound far below the real qe forces the optimiser onto it.
        result = fit_model_with_ci(
            pso_model, t, qt, p0=[10.0, 0.01], bounds=([0, 0], [20.0, 10]), param_names=["qe", "k2"]
        )
        assert result["converged"]
        assert any("qe" in entry for entry in result["bounds_hit"])

    def test_unconstrained_good_fit_reports_no_bounds_hit(self):
        t = np.array([5, 10, 20, 30, 45, 60, 90, 120.0])
        qt = pso_model(t, 90.0, 0.002)
        result = fit_model_with_ci(
            pso_model,
            t,
            qt,
            p0=[80.0, 0.002],
            bounds=([0, 0], [500.0, 10]),
            param_names=["qe", "k2"],
        )
        assert result["converged"]
        assert result["bounds_hit"] == []


# =============================================================================
# MISSING VALUES MUST NOT BECOME AFFIRMATIVE LABELS
# =============================================================================
class TestSignLabel:
    @pytest.mark.parametrize("missing", [None, float("nan"), np.nan, float("inf"), "abc"])
    def test_missing_values_are_not_labelled_positive(self, missing):
        assert sign_label(missing, "Negative", "Positive") == "—"

    def test_real_signs_are_labelled(self):
        assert sign_label(-1.0, "Exothermic", "Endothermic") == "Exothermic"
        assert sign_label(1.0, "Exothermic", "Endothermic") == "Endothermic"
        assert sign_label(0.0, "Negative", "Positive") == "Zero"

    def test_custom_unavailable_label(self):
        assert sign_label(None, "a", "b", unavailable="n/a") == "n/a"

    def test_numpy_scalars_are_handled(self):
        assert sign_label(np.float64(-3.2), "Negative", "Positive") == "Negative"
        assert sign_label(np.float64("nan"), "Negative", "Positive") == "—"


class TestComparisonTabDeltaG:
    def test_missing_thermo_inputs_give_nan_not_zero(self):
        from adsorblab_pro.tabs.comparison_tab import _apparent_delta_g_298

        assert np.isnan(_apparent_delta_g_298({}))
        assert np.isnan(_apparent_delta_g_298({"delta_H": -20.0}))
        assert np.isnan(_apparent_delta_g_298({"delta_S": 50.0}))
        assert np.isnan(_apparent_delta_g_298({"delta_H": np.nan, "delta_S": 50.0}))

    def test_complete_inputs_compute_apparent_delta_g(self):
        from adsorblab_pro.tabs.comparison_tab import _apparent_delta_g_298

        value = _apparent_delta_g_298({"delta_H": -20.0, "delta_S": 50.0})
        assert value == pytest.approx(-20.0 - 298.15 * 50.0 / 1000)


# =============================================================================
# CONSISTENCY CHECKER: "NOTHING CHECKED" IS NOT "ALL CLEAR"
# =============================================================================
class TestNoChecksState:
    def test_no_applicable_checks_is_distinct_from_consistent(self):
        result = check_mechanism_consistency({})
        assert result["status"] == "no_checks"
        assert result["n_checks"] == 0
        assert result["color"] != "green"

    def test_a_real_passing_check_is_still_consistent(self):
        state = {
            "isotherm_models_fitted": {},
            "kinetic_models_fitted": {},
            "thermo_params": {"delta_H": 20.0},
            "temperature_effect": "increases",
        }
        result = check_mechanism_consistency(state)
        assert result["status"] == "consistent"
        assert result["n_checks"] >= 1
        assert result["color"] == "green"


# =============================================================================
# Kd VALIDATION REPORTS THE OFFENDING ROWS
# =============================================================================
class TestKdValidationDiagnostics:
    def test_error_names_the_offending_row_numbers(self):
        from adsorblab_pro.tabs.thermodynamics_tab import _calculate_kd

        Ce = np.array([10.0, 0.0, 20.0, -1.0])
        qe = np.array([5.0, 5.0, 5.0, 5.0])
        with pytest.raises(ValueError, match=r"row\(s\) 2, 4"):
            _calculate_kd("dimensionless", 100.0, Ce, qe, 0.1, 0.05)

    def test_volume_corrected_enforces_ce_below_c0_like_dimensionless(self):
        from adsorblab_pro.tabs.thermodynamics_tab import _calculate_kd

        Ce = np.array([10.0, 150.0])
        qe = np.array([5.0, 5.0])
        with pytest.raises(ValueError, match="0 < Ce < C0"):
            _calculate_kd("volume_corrected", 100.0, Ce, qe, 0.1, 0.05)

    def test_valid_inputs_still_compute(self):
        from adsorblab_pro.tabs.thermodynamics_tab import _calculate_kd

        Ce = np.array([10.0, 20.0])
        qe = np.array([45.0, 40.0])
        Kd = _calculate_kd("volume_corrected", 100.0, Ce, qe, 0.1, 0.05)
        assert np.all(np.isfinite(Kd)) and np.all(Kd > 0)


# =============================================================================
# REMOVED SURFACES AND REMOVED API
# =============================================================================
class TestRemovedSurfaces:
    def test_synthetic_3d_surfaces_are_gone(self):
        from adsorblab_pro.tabs import threed_explorer_tab

        for name in (
            "_render_isotherm_surface",
            "_render_parameter_space",
            "_render_ph_temp_response",
            "_render_model_comparison",
        ):
            assert not hasattr(threed_explorer_tab, name), f"{name} should have been removed"

    def test_data_derived_surfaces_remain(self):
        from adsorblab_pro.tabs import threed_explorer_tab

        assert hasattr(threed_explorer_tab, "_render_residuals_surface")
        assert hasattr(threed_explorer_tab, "_render_experimental_3d")

    def test_mechanism_classifier_is_removed_from_the_public_api(self):
        import adsorblab_pro.utils as utils

        assert "determine_adsorption_mechanism" not in utils.__all__
        assert not hasattr(utils, "determine_adsorption_mechanism")


class TestVersionResolution:
    def test_version_matches_pyproject(self):
        import re
        from pathlib import Path

        import adsorblab_pro

        pyproject = Path(adsorblab_pro.__file__).resolve().parent.parent / "pyproject.toml"
        declared = re.search(
            r'^\s*version\s*=\s*["\']([^"\']+)["\']', pyproject.read_text(encoding="utf-8"), re.M
        ).group(1)
        assert adsorblab_pro.__version__ == declared
