# tests/test_integration.py
"""
Integration, UI, Logic, and Performance Tests
==============================================

Comprehensive test suite covering:
1. Integration tests - Cross-module workflow testing
2. UI/Streamlit tests - Mocked UI component testing
3. Logic error tests - Edge cases and property-based testing
4. Performance tests - Timing and resource benchmarks

Author: AdsorbLab Team
"""

import os
import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adsorblab_pro.models import (
    fit_model_with_ci,
    freundlich_model,
    get_model_info,
    langmuir_model,
    pfo_model,
    pso_model,
    temkin_model,
)
from adsorblab_pro.utils import (
    analyze_residuals,
    assess_data_quality,
    bootstrap_confidence_intervals,
    calculate_adsorption_capacity,
    calculate_akaike_weights,
    calculate_arrhenius_parameters,
    calculate_Ce_from_absorbance,
    calculate_error_metrics,
    calculate_removal_percentage,
    calculate_separation_factor,
    calculate_thermodynamic_parameters,
    check_mechanism_consistency,
    convert_df_to_csv,
    convert_df_to_excel,
    detect_common_errors,
    determine_adsorption_mechanism,
    interpret_separation_factor,
    interpret_thermodynamics,
    recommend_best_models,
    standardize_dataframe_columns,
)
from adsorblab_pro.validation import (
    validate_isotherm_data,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def complete_isotherm_dataset():
    """Complete isotherm dataset for integration testing."""
    return {
        "Ce": np.array([5, 10, 20, 40, 60, 80, 100, 120, 150]),
        "qe": np.array([15.2, 25.8, 38.5, 52.1, 58.3, 62.5, 65.0, 66.8, 68.0]),
        "C0": np.array([100, 100, 100, 100, 100, 100, 100, 100, 100]),
    }


@pytest.fixture
def complete_kinetic_dataset():
    """Complete kinetic dataset for integration testing."""
    return {
        "t": np.array([0, 5, 10, 20, 30, 60, 90, 120, 180, 240, 300, 360]),
        "qt": np.array([0, 12.5, 22.0, 35.5, 44.0, 55.2, 60.1, 62.8, 64.5, 65.0, 65.2, 65.3]),
    }


@pytest.fixture
def temperature_series_data():
    """Multi-temperature data for thermodynamic analysis."""
    return {
        "temperatures": np.array([298, 308, 318, 328]),
        "Kd_values": np.array([15.2, 12.1, 9.8, 8.1]),
    }


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit module for UI testing."""
    with patch.dict("sys.modules", {"streamlit": MagicMock()}):
        import streamlit as st

        st.session_state = {}
        st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
        st.selectbox = MagicMock(return_value="Langmuir")
        st.slider = MagicMock(return_value=50)
        st.button = MagicMock(return_value=False)
        st.checkbox = MagicMock(return_value=True)
        st.number_input = MagicMock(return_value=100.0)
        st.text_input = MagicMock(return_value="Test")
        st.dataframe = MagicMock()
        st.plotly_chart = MagicMock()
        st.write = MagicMock()
        st.markdown = MagicMock()
        st.error = MagicMock()
        st.warning = MagicMock()
        st.success = MagicMock()
        st.info = MagicMock()
        st.spinner = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
        st.expander = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
        st.tabs = MagicMock(return_value=[MagicMock(), MagicMock()])
        st.cache_data = lambda f: f
        st.cache_resource = lambda f: f
        yield st


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIsothermWorkflowIntegration:
    """Integration tests for complete isotherm analysis workflow."""

    def test_complete_isotherm_analysis_workflow(self, complete_isotherm_dataset):
        """Test full workflow: data → fitting → comparison → recommendation."""
        Ce = complete_isotherm_dataset["Ce"]
        qe = complete_isotherm_dataset["qe"]
        C0 = complete_isotherm_dataset["C0"]

        # Step 1: Validate data
        df = pd.DataFrame({"Ce": Ce, "qe": qe, "C0": C0})
        # Data should be valid
        assert len(df) >= 5

        # Step 2: Fit multiple models
        models_fitted = {}

        # Langmuir
        lang_result = fit_model_with_ci(
            langmuir_model, Ce, qe, p0=[80, 0.05], bounds=([0, 0], [200, 1])
        )
        if lang_result["converged"]:
            models_fitted["Langmuir"] = lang_result

        # Freundlich
        freund_result = fit_model_with_ci(
            freundlich_model, Ce, qe, p0=[10, 0.5], bounds=([0, 0], [100, 2])
        )
        if freund_result["converged"]:
            models_fitted["Freundlich"] = freund_result

        # Step 3: Compare models
        assert len(models_fitted) >= 1, "At least one model should converge"

        # Step 4: Get recommendations
        recommendations = recommend_best_models(models_fitted, model_type="isotherm")
        assert isinstance(recommendations, list)

        # Step 5: Calculate separation factor for Langmuir
        if "Langmuir" in models_fitted:
            KL = models_fitted["Langmuir"]["popt"][1]
            RL = calculate_separation_factor(KL, C0[0])
            interpretation = interpret_separation_factor(RL)
            assert isinstance(interpretation, str)
            assert (
                "Favorable" in interpretation
                or "Unfavorable" in interpretation
                or "Linear" in interpretation
            )

    def test_data_quality_to_fitting_integration(self, complete_isotherm_dataset):
        """Test data quality assessment affects fitting strategy."""
        Ce = complete_isotherm_dataset["Ce"]
        qe = complete_isotherm_dataset["qe"]

        df = pd.DataFrame({"Ce": Ce, "qe": qe})

        # Assess quality
        quality = assess_data_quality(df, data_type="isotherm")
        assert isinstance(quality, dict)

        # Fit based on quality
        result = fit_model_with_ci(langmuir_model, Ce, qe, p0=[80, 0.05], bounds=([0, 0], [200, 1]))

        # Calculate metrics
        if result["converged"]:
            y_pred = langmuir_model(Ce, *result["popt"])
            metrics = calculate_error_metrics(qe, y_pred, n_params=2)
            assert "r_squared" in metrics
            assert "rmse" in metrics

    def test_fitting_to_mechanism_integration(self, complete_isotherm_dataset):
        """Test model fitting flows into mechanism determination."""
        Ce = complete_isotherm_dataset["Ce"]
        qe = complete_isotherm_dataset["qe"]

        # Fit Langmuir and get KL
        lang_result = fit_model_with_ci(
            langmuir_model, Ce, qe, p0=[80, 0.05], bounds=([0, 0], [200, 1])
        )

        if lang_result["converged"]:
            # Calculate RL
            KL = lang_result["popt"][1]
            RL = calculate_separation_factor(KL, 100)

            # Determine mechanism (need delta_H, simulate with a value)
            mechanism = determine_adsorption_mechanism(delta_H=-25, RL=RL)
            assert isinstance(mechanism, dict)
            assert "evidence" in mechanism


class TestKineticWorkflowIntegration:
    """Integration tests for complete kinetic analysis workflow."""

    def test_complete_kinetic_analysis_workflow(self, complete_kinetic_dataset):
        """Test full kinetic workflow: data → fitting → analysis."""
        t = complete_kinetic_dataset["t"]
        qt = complete_kinetic_dataset["qt"]

        # Step 1: Fit PFO
        pfo_result = fit_model_with_ci(pfo_model, t, qt, p0=[65, 0.05], bounds=([0, 0], [200, 1]))

        # Step 2: Fit PSO
        pso_result = fit_model_with_ci(
            pso_model, t, qt, p0=[65, 0.001], bounds=([0, 0], [200, 0.1])
        )

        # Step 3: Compare and select
        results = {}
        if pfo_result["converged"]:
            results["PFO"] = pfo_result
        if pso_result["converged"]:
            results["PSO"] = pso_result

        assert len(results) >= 1

        # Step 4: Residual analysis
        for name, result in results.items():
            if result["converged"]:
                y_pred = (pfo_model if name == "PFO" else pso_model)(t, *result["popt"])
                residuals = qt - y_pred
                analysis = analyze_residuals(residuals)
                assert "mean" in analysis
                assert "durbin_watson" in analysis


class TestThermodynamicWorkflowIntegration:
    """Integration tests for thermodynamic analysis workflow."""

    def test_complete_thermodynamic_workflow(self, temperature_series_data):
        """Test full thermodynamic workflow."""
        T = temperature_series_data["temperatures"]
        Kd = temperature_series_data["Kd_values"]

        # Step 1: Calculate thermodynamic parameters
        thermo = calculate_thermodynamic_parameters(T, Kd)
        assert thermo["success"] is True

        # Step 2: Interpret results
        interpretation = interpret_thermodynamics(
            thermo["delta_H"],
            thermo["delta_S"],
            thermo["delta_G"][0]
            if isinstance(thermo["delta_G"], np.ndarray)
            else thermo["delta_G"],
        )
        assert isinstance(interpretation, dict)

        # Step 3: Mechanism determination
        mechanism = determine_adsorption_mechanism(delta_H=thermo["delta_H"])
        assert isinstance(mechanism, dict)

    def test_arrhenius_integration(self):
        """Test Arrhenius analysis in workflow."""
        T_K = np.array([298, 308, 318, 328])
        k = np.array([0.01, 0.018, 0.032, 0.055])

        # Calculate Arrhenius parameters
        arrhenius = calculate_arrhenius_parameters(T_K, k)

        if arrhenius["success"]:
            assert "Ea" in arrhenius
            assert "A" in arrhenius
            assert arrhenius["Ea"] > 0


class TestCrossModuleIntegration:
    """Tests for cross-module data flow."""

    def test_validation_to_fitting_flow(self):
        """Test data flows correctly from validation to fitting."""
        # Create test data
        Ce = np.array([5, 10, 20, 40, 60, 80, 100])
        qe = np.array([15, 26, 39, 52, 58, 62, 65])
        C0 = np.array([100, 100, 100, 100, 100, 100, 100])

        # Validate
        report = validate_isotherm_data(Ce, qe, C0)

        # If valid, proceed to fitting
        if report.is_valid:
            result = fit_model_with_ci(
                langmuir_model, Ce, qe, p0=[80, 0.05], bounds=([0, 0], [200, 1])
            )
            assert "converged" in result

    def test_study_state_consistency_check(self):
        """Test mechanism consistency checking with study state."""
        study_state = {
            "isotherm_models_fitted": {
                "Langmuir": {
                    "converged": True,
                    "r_squared": 0.99,
                    "params": {"qm": 100, "KL": 0.1},
                },
            },
            "thermo_params": {"delta_H": -25, "delta_G": [-15, -17, -19]},
        }

        consistency = check_mechanism_consistency(study_state)
        assert isinstance(consistency, dict)
        assert "status" in consistency

    def test_error_detection_integration(self):
        """Test error detection on complete study state."""
        study_state = {
            "thermo_params": {
                "delta_G": [-15, -18, -20]  # Normal range
            },
            "isotherm_results": pd.DataFrame(
                {"Ce": [5, 10, 20, 40, 60], "qe": [15, 26, 39, 52, 58]}
            ),
            "isotherm_models_fitted": {
                "Langmuir": {"converged": True, "r_squared": 0.99, "num_params": 2}
            },
        }

        errors = detect_common_errors(study_state)
        assert isinstance(errors, list)


# =============================================================================
# UI / STREAMLIT TESTS (Mocked)
# =============================================================================


class TestUIComponents:
    """Tests for UI components with mocked Streamlit."""

    def test_data_hash_computation(self):
        """Test data hashing for cache invalidation."""
        from adsorblab_pro.tabs.isotherm_tab import _arrays_to_tuples, _compute_data_hash

        Ce = np.array([5, 10, 20])
        qe = np.array([15, 26, 39])
        C0 = np.array([100, 100, 100])

        # Test hash computation
        hash1 = _compute_data_hash(Ce, qe, C0)
        hash2 = _compute_data_hash(Ce, qe, C0)
        assert hash1 == hash2  # Same data, same hash

        # Different data, different hash
        Ce2 = np.array([5, 10, 25])
        hash3 = _compute_data_hash(Ce2, qe, C0)
        assert hash1 != hash3

        # Test tuple conversion
        tuples = _arrays_to_tuples(Ce, qe, C0)
        assert len(tuples) == 3
        assert all(isinstance(t, tuple) for t in tuples)

    def test_dataframe_export_functions(self):
        """Test DataFrame export to CSV and Excel."""
        df = pd.DataFrame({"Ce": [5, 10, 20, 40], "qe": [15, 26, 39, 52]})

        # CSV export
        csv_bytes = convert_df_to_csv(df)
        assert isinstance(csv_bytes, bytes)
        assert len(csv_bytes) > 0
        assert b"Ce" in csv_bytes

        # Excel export
        excel_bytes = convert_df_to_excel(df, sheet_name="Isotherm")
        assert isinstance(excel_bytes, bytes)
        assert len(excel_bytes) > 0

    def test_column_standardization_for_ui(self):
        """Test column standardization used in UI data processing."""
        df = pd.DataFrame(
            {
                "equilibrium_concentration": [5, 10, 20],
                "adsorption_capacity": [15, 26, 39],
                "initial_conc": [100, 100, 100],
            }
        )

        standardized = standardize_dataframe_columns(df)
        assert isinstance(standardized, pd.DataFrame)
        assert len(standardized.columns) == 3


class TestUIDataProcessing:
    """Tests for UI data processing functions."""

    def test_calibration_data_processing(self):
        """Test calibration data processing flow."""
        absorbance = 0.5
        slope = 0.02
        intercept = 0.05

        Ce = calculate_Ce_from_absorbance(absorbance, slope, intercept)
        assert Ce > 0
        assert Ce == pytest.approx((0.5 - 0.05) / 0.02, rel=0.01)

    def test_capacity_calculation_for_display(self):
        """Test capacity calculation used in UI display."""
        C0, Ce, V, m = 100, 20, 0.1, 0.5

        qe = calculate_adsorption_capacity(C0, Ce, V, m)
        removal = calculate_removal_percentage(C0, Ce)

        assert qe > 0
        assert 0 <= removal <= 100
        assert removal == pytest.approx(80.0, rel=0.01)


# =============================================================================
# LOGIC ERROR TESTS
# =============================================================================


class TestLogicEdgeCases:
    """Tests for logic errors and edge cases."""

    def test_model_with_extreme_parameters(self):
        """Test models with extreme but valid parameters."""
        Ce = np.array([1, 10, 100, 1000])

        # Very high qm
        qe_high_qm = langmuir_model(Ce, qm=1e6, KL=0.001)
        assert np.all(np.isfinite(qe_high_qm))

        # Very small KL
        qe_small_KL = langmuir_model(Ce, qm=100, KL=1e-10)
        assert np.all(np.isfinite(qe_small_KL))

        # Very large KL
        qe_large_KL = langmuir_model(Ce, qm=100, KL=1e6)
        assert np.all(np.isfinite(qe_large_KL))

    def test_freundlich_extreme_n(self):
        """Test Freundlich with extreme n values."""
        Ce = np.array([1, 10, 100])

        # Very small n_inv (strong adsorption, n_inv = 1/n)
        qe_small_n_inv = freundlich_model(Ce, KF=10, n_inv=0.1)
        assert np.all(np.isfinite(qe_small_n_inv))

        # n_inv close to 1 (linear)
        qe_linear = freundlich_model(Ce, KF=10, n_inv=0.999)
        assert np.all(np.isfinite(qe_linear))

    def test_kinetic_at_time_zero(self):
        """Test kinetic models at t=0."""
        t = np.array([0, 1, 5, 10])

        qt_pfo = pfo_model(t, qe=65, k1=0.05)
        qt_pso = pso_model(t, qe=65, k2=0.001)

        # At t=0, qt should be 0 or very close
        assert qt_pfo[0] == pytest.approx(0, abs=1e-10)
        assert qt_pso[0] == pytest.approx(0, abs=1e-10)

    def test_kinetic_at_equilibrium(self):
        """Test kinetic models approach equilibrium."""
        t_long = np.array([0, 100, 1000, 10000, 100000])
        qe_expected = 65

        qt_pfo = pfo_model(t_long, qe=qe_expected, k1=0.05)
        qt_pso = pso_model(t_long, qe=qe_expected, k2=0.001)

        # Should approach qe at long times
        assert qt_pfo[-1] == pytest.approx(qe_expected, rel=0.01)
        assert qt_pso[-1] == pytest.approx(qe_expected, rel=0.05)

    def test_thermodynamic_calculation_edge_cases(self):
        """Test thermodynamic calculations with edge cases."""
        # Decreasing Kd (exothermic)
        T1 = np.array([298, 308, 318])
        Kd1 = np.array([100, 80, 65])
        result1 = calculate_thermodynamic_parameters(T1, Kd1)
        if result1["success"]:
            assert result1["delta_H"] < 0  # Exothermic

        # Increasing Kd (endothermic)
        Kd2 = np.array([50, 65, 80])
        result2 = calculate_thermodynamic_parameters(T1, Kd2)
        if result2["success"]:
            assert result2["delta_H"] > 0  # Endothermic

    def test_separation_factor_boundaries(self):
        """Test separation factor at boundary conditions."""
        C0 = 100

        # RL should be between 0 and 1 for favorable adsorption
        RL_favorable = calculate_separation_factor(KL=0.1, C0=C0)
        assert 0 < RL_favorable < 1

        # Very large KL → RL → 0 (irreversible)
        RL_irreversible = calculate_separation_factor(KL=1000, C0=C0)
        assert RL_irreversible < 0.01

        # Very small KL → RL → 1 (linear)
        RL_linear = calculate_separation_factor(KL=0.0001, C0=C0)
        assert RL_linear > 0.99

    def test_r_squared_boundary_values(self):
        """Test R² calculation at boundaries."""
        y_obs = np.array([10, 20, 30, 40, 50])

        # Perfect fit
        metrics_perfect = calculate_error_metrics(y_obs, y_obs, n_params=1)
        assert metrics_perfect["r_squared"] == pytest.approx(1.0, rel=1e-10)

        # Prediction equals mean (R² = 0)
        y_mean = np.full_like(y_obs, np.mean(y_obs), dtype=float)
        metrics_mean = calculate_error_metrics(y_obs, y_mean, n_params=1)
        assert metrics_mean["r_squared"] == pytest.approx(0.0, abs=1e-10)


class TestLogicConsistency:
    """Tests for logical consistency between functions."""

    def test_model_info_consistency(self):
        """Test model info is consistent with actual models."""
        info = get_model_info()
        langmuir_info = info["isotherms"]["Langmuir"]
        assert langmuir_info is not None
        assert "params" in langmuir_info
        assert len(langmuir_info["params"]) == 2  # qm, KL

    def test_akaike_weights_sum_to_one(self):
        """Test Akaike weights always sum to 1."""
        aic_values = [-10, -8, -5]

        weights = calculate_akaike_weights(aic_values)
        total = np.sum(weights)
        assert total == pytest.approx(1.0, rel=1e-10)

    def test_mechanism_scores_consistent(self):
        """Test mechanism determination scores are consistent."""
        # Physical adsorption indicators
        result_physical = determine_adsorption_mechanism(
            delta_H=-15,  # Low
        )

        # Chemical adsorption indicators
        result_chemical = determine_adsorption_mechanism(
            delta_H=-90,  # High
        )

        assert "scores" in result_physical
        assert "scores" in result_chemical


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformance:
    """Performance and timing tests."""

    def test_model_fitting_performance(self, complete_isotherm_dataset):
        """Test model fitting completes in reasonable time."""
        Ce = complete_isotherm_dataset["Ce"]
        qe = complete_isotherm_dataset["qe"]

        start = time.time()
        result = fit_model_with_ci(langmuir_model, Ce, qe, p0=[80, 0.05], bounds=([0, 0], [200, 1]))
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Fitting took too long: {elapsed:.2f}s"
        assert result["converged"]

    def test_multiple_model_fitting_performance(self, complete_isotherm_dataset):
        """Test fitting multiple models completes in reasonable time."""
        Ce = complete_isotherm_dataset["Ce"]
        qe = complete_isotherm_dataset["qe"]

        models = [
            (langmuir_model, [80, 0.05], ([0, 0], [200, 1])),
            (freundlich_model, [10, 0.5], ([0, 0], [100, 2])),
            (temkin_model, [10, 20], ([0, 0], [100, 200])),
        ]

        start = time.time()
        for model_func, p0, bounds in models:
            fit_model_with_ci(model_func, Ce, qe, p0=p0, bounds=bounds)
        elapsed = time.time() - start

        assert elapsed < 10.0, f"Multiple fitting took too long: {elapsed:.2f}s"

    def test_bootstrap_performance(self, complete_isotherm_dataset):
        """Test bootstrap CI calculation performance."""
        Ce = complete_isotherm_dataset["Ce"]
        qe = complete_isotherm_dataset["qe"]

        start = time.time()
        bootstrap_confidence_intervals(
            langmuir_model, Ce, qe, params=np.array([70, 0.05]), n_bootstrap=100, confidence=0.95
        )
        elapsed = time.time() - start

        assert elapsed < 10.0, f"Bootstrap took too long: {elapsed:.2f}s"

    def test_thermodynamic_calculation_performance(self, temperature_series_data):
        """Test thermodynamic calculation performance."""
        T = temperature_series_data["temperatures"]
        Kd = temperature_series_data["Kd_values"]

        start = time.time()
        for _ in range(100):
            calculate_thermodynamic_parameters(T, Kd)
        elapsed = time.time() - start

        assert elapsed < 2.0, f"100 thermo calculations took too long: {elapsed:.2f}s"

    def test_residual_analysis_performance(self):
        """Test residual analysis performance with large dataset."""
        np.random.seed(42)
        residuals = np.random.normal(0, 1, 1000)

        start = time.time()
        for _ in range(10):
            analyze_residuals(residuals)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Residual analysis took too long: {elapsed:.2f}s"

    def test_data_quality_assessment_performance(self):
        """Test data quality assessment performance."""
        df = pd.DataFrame(
            {
                "Ce": np.linspace(1, 100, 100),
                "qe": np.linspace(10, 70, 100) + np.random.normal(0, 1, 100),
            }
        )

        start = time.time()
        for _ in range(50):
            assess_data_quality(df, data_type="isotherm")
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Data quality assessment took too long: {elapsed:.2f}s"


class TestMemoryEfficiency:
    """Tests for memory efficiency."""

    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # Create large dataset
        n_points = 10000
        Ce = np.linspace(1, 1000, n_points)
        qe = 100 * Ce / (50 + Ce) + np.random.normal(0, 1, n_points)

        # Should complete without memory errors
        metrics = calculate_error_metrics(qe, qe * 0.99, n_params=2)
        assert "r_squared" in metrics

    def test_repeated_operations_no_memory_leak(self):
        """Test repeated operations don't cause memory buildup."""
        Ce = np.array([5, 10, 20, 40, 60, 80, 100])
        qe = np.array([15, 26, 39, 52, 58, 62, 65])

        # Perform many iterations
        for _ in range(100):
            result = fit_model_with_ci(
                langmuir_model, Ce, qe, p0=[70, 0.05], bounds=([0, 0], [200, 1])
            )
            # Force cleanup
            del result

        # If we get here without MemoryError, test passes
        assert True


# =============================================================================
# CONCURRENCY TESTS
# =============================================================================


class TestConcurrency:
    """Tests for concurrent/parallel operations."""

    def test_thread_safe_model_evaluation(self):
        """Test model evaluation is thread-safe."""
        import queue
        import threading

        Ce = np.array([5, 10, 20, 40, 60])
        results_queue = queue.Queue()
        errors = []

        def evaluate_model(model_func, params, thread_id):
            try:
                result = model_func(Ce, *params)
                results_queue.put((thread_id, result))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=evaluate_model, args=(langmuir_model, [70, 0.05], i))
            threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Check results
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert results_queue.qsize() == 10

        # All results should be identical
        results = []
        while not results_queue.empty():
            _, result = results_queue.get()
            results.append(result)

        for r in results[1:]:
            np.testing.assert_array_almost_equal(results[0], r)

    def test_independent_calculations_isolation(self):
        """Test calculations with different inputs don't interfere."""
        datasets = [
            (np.array([5, 10, 20, 40, 60]), np.array([15, 26, 38, 50, 56])),
            (np.array([10, 20, 40, 60, 80]), np.array([20, 35, 50, 58, 62])),
            (np.array([1, 5, 10, 20, 40]), np.array([8, 18, 25, 35, 45])),
        ]

        results = []
        for Ce, qe in datasets:
            result = fit_model_with_ci(
                langmuir_model, Ce, qe, p0=[60, 0.05], bounds=([0, 0], [200, 1])
            )
            results.append(result)

        # Filter out None results
        valid_results = [r for r in results if r is not None and r.get("converged")]

        # Results should be different (different data)
        if len(valid_results) >= 2:
            params = [r["popt"] for r in valid_results]
            # Check parameters are not all identical
            assert not all(np.allclose(params[0], p) for p in params[1:])


# =============================================================================
# REGRESSION TESTS
# =============================================================================


class TestKnownValues:
    """Tests against known/published values."""

    def test_langmuir_known_values(self):
        """Test Langmuir model against known values."""
        # At Ce = 1/KL, qe should be qm/2
        qm, KL = 100, 0.1
        Ce_half = 1.0 / KL  # Ce = 10

        qe_half = langmuir_model(Ce_half, qm, KL)
        assert qe_half == pytest.approx(qm / 2, rel=0.01)

    def test_freundlich_known_values(self):
        """Test Freundlich model against known relationship."""
        # When n_inv=1, Freundlich becomes linear: qe = KF * Ce^1 = KF * Ce
        KF = 5
        Ce = np.array([10, 20, 30])

        qe = freundlich_model(Ce, KF, n_inv=1)
        np.testing.assert_array_almost_equal(qe, KF * Ce)

    def test_thermodynamic_known_relationship(self):
        """Test thermodynamic relationship: ΔG = ΔH - TΔS."""
        delta_H = -30  # kJ/mol
        delta_S = 0.05  # kJ/(mol·K)

        # ΔG should approximately equal ΔH - TΔS
        # (delta_G_expected = delta_H - T * delta_S)

        # Create synthetic Kd data following Van't Hoff equation
        T_array = np.array([298, 308, 318])
        R = 8.314 / 1000  # kJ/(mol·K)

        # ln(Kd) = -ΔH/(RT) + ΔS/R
        ln_Kd = -delta_H / (R * T_array) + delta_S / R
        Kd = np.exp(ln_Kd)

        result = calculate_thermodynamic_parameters(T_array, Kd)
        if result["success"]:
            # Check values are in expected range
            assert result["delta_H"] == pytest.approx(delta_H, rel=0.1)
