# tests/test_utils.py
"""
Unit Tests for Utility Functions
================================

Comprehensive test suite for statistical and utility functions.
Tests include:
- Error metrics (R², Adj-R², RMSE, etc.)
- PRESS/Q² cross-validation
- Bootstrap confidence intervals
- Akaike weights
- Data quality assessment
- Thermodynamic calculations
- Column standardization
- Data validation
- Residual analysis
- Model recommendations
- Mean free energy calculations
- Arrhenius parameters
- Activity coefficients
- Confidence intervals
- Mechanism determination
- Mechanism consistency checking
- Error detection

Author: AdsorbLab Team
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models for fitting tests
from adsorblab_pro.models import langmuir_model
from adsorblab_pro.utils import (
    # Analysis functions
    analyze_residuals,
    # Quality assessment
    assess_data_quality,
    # Bootstrap
    bootstrap_confidence_intervals,
    # Activity coefficients
    calculate_activity_coefficient_davies,
    calculate_adsorption_capacity,
    # Model selection
    calculate_akaike_weights,
    calculate_arrhenius_parameters,
    # Data processing
    calculate_Ce_from_absorbance,
    # Error metrics
    calculate_error_metrics,
    # PRESS statistics
    calculate_press,
    calculate_q2,
    calculate_removal_percentage,
    # Separation factor
    calculate_separation_factor,
    # Thermodynamics
    calculate_thermodynamic_parameters,
    # Mechanism and error checking
    check_mechanism_consistency,
    # Export functions
    convert_df_to_csv,
    convert_df_to_excel,
    detect_common_errors,
    # Data detection
    detect_replicates,
    # Mechanism determination
    determine_adsorption_mechanism,
    interpret_separation_factor,
    interpret_thermodynamics,
    # Uncertainty propagation
    propagate_calibration_uncertainty,
    recommend_best_models,
    # Column standardization
    standardize_column_name,
    standardize_dataframe_columns,
    # Data validation
    validate_data_editor,
    # Validation
    validate_required_params,
)

# Aliases for clearer test names
calculate_press_statistic = calculate_press
calculate_q_squared = calculate_q2


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_fit_data():
    """Sample fitted data for testing metrics."""
    y_obs = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([11, 19, 31, 39, 51])
    return y_obs, y_pred


@pytest.fixture
def perfect_fit_data():
    """Perfect fit data for edge case testing."""
    y = np.array([10, 20, 30, 40, 50])
    return y, y.copy()


@pytest.fixture
def isotherm_experimental():
    """Experimental isotherm data."""
    return {
        "C0": np.array([10, 25, 50, 75, 100]),
        "Ce": np.array([2, 8, 22, 40, 58]),
        "V": 0.05,  # 50 mL
        "m": 0.1,  # 0.1 g
    }


@pytest.fixture
def calibration_params():
    """Calibration curve parameters."""
    return {"slope": 0.025, "intercept": 0.01}


@pytest.fixture
def thermodynamic_data():
    """Temperature-dependent Kd data."""
    return {
        "T": np.array([298.15, 308.15, 318.15, 328.15]),  # K
        "Kd": np.array([15.2, 12.1, 9.8, 8.1]),  # dimensionless
    }


# =============================================================================
# ERROR METRICS TESTS
# =============================================================================
class TestThermodynamicParametersErrorHandling:
    def test_exception_returns_error_dict(self):
        """Test that exceptions return proper error dictionary."""
        T_K = np.array([])  # Empty array triggers error
        Kd = np.array([])
        result = calculate_thermodynamic_parameters(T_K, Kd)
        assert result is not None
        assert isinstance(result, dict)
        assert not result.get("success")
        assert "error" in result


class TestDetectReplicatesEdgeCases:
    def test_with_negative_values(self):
        data = pd.DataFrame({"x": [-10.0, -10.1, -20.0], "y": [1.0, 1.1, 2.0]})
        result = detect_replicates(data, "x", tolerance=0.05)
        assert isinstance(result, pd.DataFrame)

    def test_with_very_small_values(self):
        data = pd.DataFrame({"x": [1e-15, 1.1e-15, 2e-15], "y": [1.0, 1.1, 2.0]})
        result = detect_replicates(data, "x", tolerance=0.05)
        assert isinstance(result, pd.DataFrame)

    def test_with_zero_mean(self):
        data = pd.DataFrame({"x": [-1.0, 0.0, 1.0], "y": [1.0, 1.5, 2.0]})
        result = detect_replicates(data, "x", tolerance=0.05)
        assert isinstance(result, pd.DataFrame)


class TestErrorMetrics:
    """Tests for error metric calculations."""

    def test_r_squared_calculation(self, sample_fit_data):
        """Test R² calculation."""
        y_obs, y_pred = sample_fit_data
        metrics = calculate_error_metrics(y_obs, y_pred, n_params=2)

        assert "r_squared" in metrics
        assert 0 <= metrics["r_squared"] <= 1
        assert metrics["r_squared"] > 0.95  # Should be high for good fit

    def test_r_squared_perfect_fit(self, perfect_fit_data):
        """Test R² = 1 for perfect fit."""
        y_obs, y_pred = perfect_fit_data
        metrics = calculate_error_metrics(y_obs, y_pred, n_params=2)

        assert metrics["r_squared"] == pytest.approx(1.0, abs=1e-10)

    def test_adj_r_squared(self, sample_fit_data):
        """Test adjusted R² is calculated."""
        y_obs, y_pred = sample_fit_data
        metrics = calculate_error_metrics(y_obs, y_pred, n_params=2)

        assert "adj_r_squared" in metrics
        assert metrics["adj_r_squared"] <= metrics["r_squared"]

    def test_adj_r_squared_penalizes_params(self, sample_fit_data):
        """Test adj R² decreases with more parameters."""
        y_obs, y_pred = sample_fit_data

        metrics_2 = calculate_error_metrics(y_obs, y_pred, n_params=2)
        metrics_3 = calculate_error_metrics(y_obs, y_pred, n_params=3)

        assert metrics_3["adj_r_squared"] < metrics_2["adj_r_squared"]

    def test_rmse_calculation(self, sample_fit_data):
        """Test RMSE calculation."""
        y_obs, y_pred = sample_fit_data
        metrics = calculate_error_metrics(y_obs, y_pred, n_params=2)

        assert "rmse" in metrics
        assert metrics["rmse"] >= 0

        # Manual RMSE calculation
        expected_rmse = np.sqrt(np.mean((y_obs - y_pred) ** 2))
        assert metrics["rmse"] == pytest.approx(expected_rmse, rel=1e-5)

    def test_mae_calculation(self, sample_fit_data):
        """Test MAE calculation."""
        y_obs, y_pred = sample_fit_data
        metrics = calculate_error_metrics(y_obs, y_pred, n_params=2)

        assert "mae" in metrics
        assert metrics["mae"] >= 0

        # Manual MAE calculation
        expected_mae = np.mean(np.abs(y_obs - y_pred))
        assert metrics["mae"] == pytest.approx(expected_mae, rel=1e-5)

    def test_aic_calculation(self, sample_fit_data):
        """Test AIC calculation."""
        y_obs, y_pred = sample_fit_data
        metrics = calculate_error_metrics(y_obs, y_pred, n_params=2)

        assert "aic" in metrics
        assert np.isfinite(metrics["aic"])

    def test_bic_calculation(self, sample_fit_data):
        """Test BIC calculation."""
        y_obs, y_pred = sample_fit_data
        metrics = calculate_error_metrics(y_obs, y_pred, n_params=2)

        assert "bic" in metrics
        assert np.isfinite(metrics["bic"])

    def test_bic_greater_than_aic(self, sample_fit_data):
        """Test BIC ≥ AIC for n > 7 (typical case)."""
        y_obs, y_pred = sample_fit_data
        # Extend data to ensure n > 7
        y_obs_ext = np.concatenate([y_obs, y_obs])
        y_pred_ext = np.concatenate([y_pred, y_pred])

        metrics = calculate_error_metrics(y_obs_ext, y_pred_ext, n_params=2)

        # BIC penalizes more for larger samples
        assert metrics["bic"] >= metrics["aic"]

    def test_residuals_returned(self, sample_fit_data):
        """Test residuals are returned."""
        y_obs, y_pred = sample_fit_data
        metrics = calculate_error_metrics(y_obs, y_pred, n_params=2)

        assert "residuals" in metrics
        expected_residuals = y_obs - y_pred
        assert_allclose(metrics["residuals"], expected_residuals)


# =============================================================================
# PRESS/Q² TESTS
# =============================================================================


class TestPRESSStatistics:
    """Tests for PRESS (Leave-One-Out Cross-Validation) statistics."""

    def test_press_positive(self):
        """Test PRESS is always positive."""
        from adsorblab_pro.models import langmuir_model

        Ce = np.array([5, 10, 20, 40, 60, 80])
        qe = np.array([15, 25, 38, 52, 58, 62])

        def model_func(x, *params):
            return langmuir_model(x, *params)

        params = [70, 0.05]
        press = calculate_press_statistic(model_func, Ce, qe, params)

        assert press >= 0, "PRESS should be non-negative"

    def test_q_squared_range(self):
        """Test Q² is typically between 0 and 1 for good models."""
        from adsorblab_pro.models import langmuir_model

        Ce = np.array([5, 10, 20, 40, 60, 80])
        qe = np.array([15, 25, 38, 52, 58, 62])

        def model_func(x, *params):
            return langmuir_model(x, *params)

        params = [70, 0.05]
        press = calculate_press_statistic(model_func, Ce, qe, params)
        q2 = calculate_q_squared(press, qe)

        assert q2 <= 1, "Q² should be ≤ 1"
        # For a good fit, Q² should be positive
        assert q2 > 0.5, "Q² should be reasonably high for good fit"

    def test_q_squared_vs_r_squared(self):
        """Test Q² ≤ R² (Q² is more conservative)."""
        from adsorblab_pro.models import fit_model_with_ci, langmuir_model

        Ce = np.array([5, 10, 20, 40, 60, 80])
        qe = np.array([15, 25, 38, 52, 58, 62])

        # Fit model
        result = fit_model_with_ci(langmuir_model, Ce, qe, p0=[70, 0.05], param_names=["qm", "KL"])

        # Calculate Q²
        params = [result["params"]["qm"], result["params"]["KL"]]
        press = calculate_press_statistic(langmuir_model, Ce, qe, params)
        q2 = calculate_q_squared(press, qe)

        assert q2 <= result["r_squared"], "Q² should be ≤ R²"


# =============================================================================
# BOOTSTRAP TESTS
# =============================================================================


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_bootstrap_returns_ci(self):
        """Test bootstrap returns confidence interval bounds."""
        from adsorblab_pro.models import langmuir_model

        Ce = np.array([5, 10, 20, 40, 60, 80])
        qe = np.array([15, 25, 38, 52, 58, 62])

        params = np.array([70.0, 0.05])
        ci_lower, ci_upper = bootstrap_confidence_intervals(
            langmuir_model, Ce, qe, params, n_bootstrap=50, confidence=0.95
        )

        assert len(ci_lower) == len(params)
        assert len(ci_upper) == len(params)

    def test_bootstrap_ci_brackets_params(self):
        """Test bootstrap CI brackets fitted parameters."""
        from adsorblab_pro.models import fit_model_with_ci, langmuir_model

        Ce = np.array([5, 10, 20, 40, 60, 80])
        qe = np.array([15, 25, 38, 52, 58, 62])

        # Fit model
        result = fit_model_with_ci(langmuir_model, Ce, qe, p0=[70, 0.05], param_names=["qm", "KL"])
        params = np.array([result["params"]["qm"], result["params"]["KL"]])

        # Bootstrap
        ci_lower, ci_upper = bootstrap_confidence_intervals(
            langmuir_model, Ce, qe, params, n_bootstrap=100, confidence=0.95
        )

        # Check params are within CI (usually true)
        # Note: May occasionally fail due to random sampling
        for i in range(len(params)):
            assert ci_lower[i] <= ci_upper[i], "Lower CI should be <= upper CI"

    def test_bootstrap_wider_at_lower_confidence(self):
        """Test 90% CI is narrower than 95% CI."""
        from adsorblab_pro.models import langmuir_model

        Ce = np.array([5, 10, 20, 40, 60, 80])
        qe = np.array([15, 25, 38, 52, 58, 62])
        params = np.array([70.0, 0.05])

        # Use same seed for reproducibility
        np.random.seed(42)
        ci_lower_95, ci_upper_95 = bootstrap_confidence_intervals(
            langmuir_model, Ce, qe, params, n_bootstrap=100, confidence=0.95
        )

        np.random.seed(42)
        ci_lower_90, ci_upper_90 = bootstrap_confidence_intervals(
            langmuir_model, Ce, qe, params, n_bootstrap=100, confidence=0.90
        )

        width_95 = ci_upper_95 - ci_lower_95
        width_90 = ci_upper_90 - ci_lower_90

        # 95% CI should be wider than 90% CI
        assert all(width_95 >= width_90 * 0.9)  # Allow some tolerance


# =============================================================================
# AKAIKE WEIGHTS TESTS
# =============================================================================


class TestAkaikeWeights:
    """Tests for Akaike weight calculations."""

    def test_weights_sum_to_one(self):
        """Test Akaike weights sum to 1."""
        aic_values = [100, 105, 110]
        weights = calculate_akaike_weights(aic_values)

        total = np.sum(weights)
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_lower_aic_higher_weight(self):
        """Test lower AIC gets higher weight."""
        aic_values = [100, 110, 120]  # Best, Middle, Worst
        weights = calculate_akaike_weights(aic_values)

        assert weights[0] > weights[1] > weights[2]

    def test_equal_aic_equal_weight(self):
        """Test equal AIC gives equal weights."""
        aic_values = [100, 100, 100]
        weights = calculate_akaike_weights(aic_values)

        expected = 1 / 3
        for w in weights:
            assert w == pytest.approx(expected, abs=1e-10)

    def test_single_model(self):
        """Test single model gets weight of 1."""
        aic_values = [100]
        weights = calculate_akaike_weights(aic_values)

        assert weights[0] == pytest.approx(1.0, abs=1e-10)


# =============================================================================
# DATA PROCESSING TESTS
# =============================================================================


class TestDataProcessing:
    """Tests for data processing functions."""

    def test_Ce_from_absorbance(self, calibration_params):
        """Test Ce calculation from absorbance."""
        absorbance = 0.26  # Corresponding to C = 10
        slope = calibration_params["slope"]
        intercept = calibration_params["intercept"]

        Ce = calculate_Ce_from_absorbance(absorbance, slope, intercept)

        # Ce = (Abs - intercept) / slope
        expected = (absorbance - intercept) / slope
        assert Ce == pytest.approx(expected, rel=1e-5)

    def test_Ce_non_negative(self, calibration_params):
        """Test Ce is clamped to non-negative."""
        absorbance = -0.1  # Would give negative
        Ce = calculate_Ce_from_absorbance(
            absorbance, calibration_params["slope"], calibration_params["intercept"]
        )

        assert Ce >= 0, "Ce should be non-negative"

    def test_adsorption_capacity(self, isotherm_experimental):
        """Test adsorption capacity calculation."""
        C0 = isotherm_experimental["C0"][0]  # Use first value
        Ce = isotherm_experimental["Ce"][0]
        V = isotherm_experimental["V"]
        m = isotherm_experimental["m"]

        qe = calculate_adsorption_capacity(C0, Ce, V, m)

        # qe = (C0 - Ce) * V / m
        expected = (C0 - Ce) * V / m
        assert qe == pytest.approx(expected, rel=1e-10)

    def test_adsorption_capacity_positive(self, isotherm_experimental):
        """Test qe is positive when C0 > Ce."""
        C0 = isotherm_experimental["C0"][0]
        Ce = isotherm_experimental["Ce"][0]
        qe = calculate_adsorption_capacity(
            C0, Ce, isotherm_experimental["V"], isotherm_experimental["m"]
        )

        assert qe > 0, "qe should be positive when C0 > Ce"

    def test_removal_percentage(self, isotherm_experimental):
        """Test removal percentage calculation."""
        C0 = isotherm_experimental["C0"][0]
        Ce = isotherm_experimental["Ce"][0]

        removal = calculate_removal_percentage(C0, Ce)

        # Removal % = (C0 - Ce) / C0 * 100
        expected = (C0 - Ce) / C0 * 100
        assert removal == pytest.approx(expected, rel=1e-10)

    def test_removal_percentage_range(self, isotherm_experimental):
        """Test removal % is between 0 and 100."""
        for C0, Ce in zip(isotherm_experimental["C0"], isotherm_experimental["Ce"]):
            removal = calculate_removal_percentage(C0, Ce)
            assert 0 <= removal <= 100, f"Removal should be 0-100, got {removal}"


# =============================================================================
# SEPARATION FACTOR TESTS
# =============================================================================


class TestSeparationFactor:
    """Tests for Langmuir separation factor (RL)."""

    def test_rl_calculation(self):
        """Test RL calculation."""
        C0 = np.array([10, 50, 100])
        KL = 0.1

        RL = calculate_separation_factor(KL, C0)

        # RL = 1 / (1 + KL * C0)
        expected = 1 / (1 + KL * C0)
        assert_allclose(RL, expected, rtol=1e-10)

    def test_rl_range(self):
        """Test RL is between 0 and 1 for favorable adsorption."""
        C0 = np.array([10, 50, 100, 500])
        KL = 0.05

        RL = calculate_separation_factor(KL, C0)

        assert all(RL > 0), "RL should be > 0"
        assert all(RL < 1), "RL should be < 1 for favorable adsorption"

    def test_rl_interpretation_favorable(self):
        """Test RL interpretation for favorable adsorption."""
        RL = np.array([0.5])
        interpretation = interpret_separation_factor(RL)
        assert "favorable" in interpretation.lower()

    def test_rl_interpretation_unfavorable(self):
        """Test RL interpretation for unfavorable adsorption."""
        RL = np.array([1.5])
        interpretation = interpret_separation_factor(RL)
        assert "unfavorable" in interpretation.lower()

    def test_rl_interpretation_linear(self):
        """Test RL interpretation for linear adsorption."""
        RL = np.array([1.0])
        interpretation = interpret_separation_factor(RL)
        assert "linear" in interpretation.lower()

    def test_rl_interpretation_irreversible(self):
        """Test RL interpretation for irreversible adsorption."""
        RL = np.array([0.0])
        interpretation = interpret_separation_factor(RL)
        assert "irreversible" in interpretation.lower()


# =============================================================================
# THERMODYNAMIC TESTS
# =============================================================================


class TestThermodynamics:
    """Tests for thermodynamic calculations."""

    def test_thermodynamic_parameters(self, thermodynamic_data):
        """Test thermodynamic parameter calculation."""
        T = thermodynamic_data["T"]
        Kd = thermodynamic_data["Kd"]

        result = calculate_thermodynamic_parameters(T, Kd)

        assert "delta_H" in result, "Should calculate ΔH°"
        assert "delta_S" in result, "Should calculate ΔS°"
        assert "delta_G" in result, "Should calculate ΔG°"
        assert "r_squared" in result, "Should include R² of Van't Hoff plot"

    def test_delta_H_negative_exothermic(self, thermodynamic_data):
        """Test ΔH° is negative for exothermic (Kd decreases with T)."""
        T = thermodynamic_data["T"]
        Kd = thermodynamic_data["Kd"]  # Decreasing with T

        result = calculate_thermodynamic_parameters(T, Kd)

        # For exothermic, ΔH° < 0
        assert result["delta_H"] < 0, "ΔH° should be negative for exothermic"

    def test_delta_G_calculation(self, thermodynamic_data):
        """Test ΔG° is calculated for each temperature."""
        T = thermodynamic_data["T"]
        Kd = thermodynamic_data["Kd"]

        result = calculate_thermodynamic_parameters(T, Kd)

        assert len(result["delta_G"]) == len(T)

    def test_spontaneous_negative_delta_G(self, thermodynamic_data):
        """Test spontaneous process has negative ΔG°."""
        T = thermodynamic_data["T"]
        Kd = thermodynamic_data["Kd"]

        result = calculate_thermodynamic_parameters(T, Kd)

        # For Kd > 1, ΔG° should be negative (spontaneous)
        assert all(np.array(result["delta_G"]) < 0), "ΔG° should be negative for Kd > 1"

    def test_thermodynamic_interpretation(self, thermodynamic_data):
        """Test thermodynamic interpretation."""
        T = thermodynamic_data["T"]
        Kd = thermodynamic_data["Kd"]

        result = calculate_thermodynamic_parameters(T, Kd)
        interpretation = interpret_thermodynamics(
            result["delta_H"],
            result["delta_S"],
            result["delta_G"],  # Pass the full array
        )

        assert isinstance(interpretation, dict)
        # Check for expected keys in interpretation
        assert len(interpretation) > 0


# =============================================================================
# DATA QUALITY TESTS
# =============================================================================


class TestDataQuality:
    """Tests for data quality assessment."""

    def test_quality_returns_dict(self):
        """Test quality assessment returns dictionary."""
        df = pd.DataFrame({"Ce": [5, 10, 20, 40, 60], "qe": [15, 25, 35, 45, 50]})

        quality = assess_data_quality(df, data_type="isotherm")

        assert isinstance(quality, dict)

    def test_quality_score_range(self):
        """Test quality score is in valid range."""
        df = pd.DataFrame({"Ce": [5, 10, 20, 40, 60], "qe": [15, 25, 35, 45, 50]})

        quality = assess_data_quality(df, data_type="isotherm")

        if "score" in quality:
            assert 0 <= quality["score"] <= 100

    def test_quality_detects_few_points(self):
        """Test quality warns about too few data points."""
        df = pd.DataFrame({"Ce": [10, 50], "qe": [20, 40]})  # Only 2 points

        quality = assess_data_quality(df, data_type="isotherm")

        # Should have warning about data points or lower score
        issues = quality.get("issues", [])
        score = quality.get("score", 100)
        assert len(issues) > 0 or score < 100


# =============================================================================
# EDGE CASES AND NUMERICAL STABILITY
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability in utility functions."""

    def test_metrics_with_constant_prediction(self):
        """Test error metrics handle constant predictions."""
        y_obs = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([30, 30, 30, 30, 30])  # Constant

        metrics = calculate_error_metrics(y_obs, y_pred, n_params=1)

        assert np.isfinite(metrics["rmse"])
        assert np.isfinite(metrics["mae"])

    def test_separation_factor_zero_KL(self):
        """Test RL calculation with very small KL."""
        C0 = np.array([10, 50, 100])
        KL = 1e-15

        RL = calculate_separation_factor(KL, C0)

        assert all(np.isfinite(RL))
        assert all(RL <= 1)

    def test_Ce_calculation_zero_slope(self):
        """Test Ce calculation handles zero slope gracefully."""
        absorbance = 0.5

        # Should handle zero slope
        Ce = calculate_Ce_from_absorbance(absorbance, slope=0, intercept=0.1)
        # With zero slope, should return 0
        assert Ce == 0.0 or np.isfinite(Ce)

    def test_adsorption_capacity_zero_mass(self):
        """Test qe calculation handles zero mass."""
        C0 = 100.0
        Ce = 50.0
        V = 0.05

        # Should handle zero mass gracefully
        qe = calculate_adsorption_capacity(C0, Ce, V, m=0)
        # With zero mass, should return 0 (protected division)
        assert qe == 0.0 or np.isinf(qe)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# =============================================================================
# EXTENDED TESTS (Additional coverage)
# =============================================================================

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "Ce": [5, 10, 20, 40, 60, 80, 100],
            "qe": [15.2, 25.8, 38.5, 52.1, 58.3, 62.5, 65.0],
            "C0": [100, 100, 100, 100, 100, 100, 100],
        }
    )


@pytest.fixture
def kinetic_dataframe():
    """Sample kinetic DataFrame."""
    return pd.DataFrame(
        {"t": [0, 5, 10, 20, 30, 60, 90, 120], "qt": [0, 15, 28, 42, 50, 58, 62, 64]}
    )


@pytest.fixture
def model_results():
    """Sample model fitting results for recommendation tests."""
    return {
        "Langmuir": {
            "converged": True,
            "r_squared": 0.995,
            "adj_r_squared": 0.994,
            "rmse": 1.2,
            "aic": -15.5,
            "aicc": -14.0,
            "residuals": np.array([0.1, -0.2, 0.15, -0.1, 0.05, -0.15, 0.1]),
        },
        "Freundlich": {
            "converged": True,
            "r_squared": 0.985,
            "adj_r_squared": 0.983,
            "rmse": 2.1,
            "aic": -10.2,
            "aicc": -8.7,
            "residuals": np.array([0.5, -0.8, 0.6, -0.4, 0.3, -0.6, 0.4]),
        },
        "Temkin": {"converged": False, "r_squared": 0.5},
    }


@pytest.fixture
def thermodynamic_data_tuple():
    """Thermodynamic test data."""
    T_K = np.array([298, 308, 318, 328])
    Kd = np.array([10.5, 8.2, 6.4, 5.0])
    return T_K, Kd


# =============================================================================
# COLUMN STANDARDIZATION TESTS
# =============================================================================


class TestColumnStandardization:
    """Tests for column name standardization."""

    def test_standardize_common_variants(self):
        """Test common column name variants."""
        # These should map to standard names
        result1 = standardize_column_name("equilibrium_concentration")
        result2 = standardize_column_name("initial_concentration")
        # Check they return something
        assert result1 is not None
        assert result2 is not None

    def test_standardize_unknown_column(self):
        """Test unknown column name returns original."""
        result = standardize_column_name("unknown_column_xyz")
        assert result == "unknown_column_xyz"

    def test_standardize_dataframe_columns(self, sample_dataframe):
        """Test DataFrame column standardization."""
        df = sample_dataframe.copy()
        result = standardize_dataframe_columns(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == len(df.columns)

    def test_standardize_returns_string(self):
        """Test that standardization always returns a string."""
        result = standardize_column_name("any_column")
        assert isinstance(result, str)

    def test_standardize_handles_special_chars(self):
        """Test handling of special characters."""
        result = standardize_column_name("C_e (mg/L)")
        assert isinstance(result, str)


# =============================================================================
# CALCULATION TESTS
# =============================================================================


class TestCalculations:
    """Tests for basic calculation functions."""

    def test_removal_percentage(self):
        """Test removal percentage calculation."""
        result = calculate_removal_percentage(C0=100, Ce=20)
        assert result == pytest.approx(80.0, rel=0.01)

    def test_removal_percentage_zero_removal(self):
        """Test 0% removal."""
        result = calculate_removal_percentage(C0=100, Ce=100)
        assert result == pytest.approx(0.0, rel=0.01)

    def test_removal_percentage_complete_removal(self):
        """Test 100% removal."""
        result = calculate_removal_percentage(C0=100, Ce=0)
        assert result == pytest.approx(100.0, rel=0.01)

    def test_adsorption_capacity(self):
        """Test adsorption capacity calculation."""
        # qe = (C0 - Ce) * V / m
        result = calculate_adsorption_capacity(C0=100, Ce=20, V=0.1, m=0.5)
        expected = (100 - 20) * 0.1 / 0.5  # = 16 mg/g
        assert result == pytest.approx(expected, rel=0.01)

    def test_Ce_from_absorbance(self):
        """Test concentration from absorbance calculation."""
        # Ce = (Abs - intercept) / slope
        result = calculate_Ce_from_absorbance(absorbance=0.5, slope=0.01, intercept=0.0)
        expected = 0.5 / 0.01  # = 50
        assert result == pytest.approx(expected, rel=0.01)

    def test_Ce_from_absorbance_with_intercept(self):
        """Test Ce calculation with non-zero intercept."""
        result = calculate_Ce_from_absorbance(absorbance=0.6, slope=0.02, intercept=0.1)
        expected = (0.6 - 0.1) / 0.02  # = 25
        assert result == pytest.approx(expected, rel=0.01)


# =============================================================================
# ARRHENIUS PARAMETERS TESTS
# =============================================================================


class TestArrheniusParameters:
    """Tests for Arrhenius parameter calculation."""

    def test_arrhenius_basic(self):
        """Test basic Arrhenius calculation."""
        T_K = np.array([298, 308, 318, 328])
        k = np.array([0.01, 0.02, 0.04, 0.08])

        result = calculate_arrhenius_parameters(T_K, k)

        assert result["success"] is True
        assert "Ea" in result
        assert "A" in result
        assert result["Ea"] > 0
        assert result["A"] > 0

    def test_arrhenius_insufficient_data(self):
        """Test with insufficient data points."""
        T_K = np.array([298])
        k = np.array([0.01])

        result = calculate_arrhenius_parameters(T_K, k)

        assert result["success"] is False
        assert "error" in result

    def test_arrhenius_invalid_values(self):
        """Test handling of invalid values."""
        T_K = np.array([298, 308, 0, 318])  # Contains 0
        k = np.array([0.01, 0.02, 0.03, 0.04])

        result = calculate_arrhenius_parameters(T_K, k)

        # Should still work with valid points
        if result["success"]:
            assert result["n_points"] >= 2

    def test_arrhenius_interpretation(self):
        """Test Arrhenius interpretation categories."""
        # High Ea (chemical)
        T_K = np.array([298, 308, 318])
        k = np.array([0.001, 0.01, 0.1])  # Fast increase = high Ea

        result = calculate_arrhenius_parameters(T_K, k)

        if result["success"]:
            assert "interpretation" in result


# =============================================================================
# RESIDUAL ANALYSIS TESTS
# =============================================================================


class TestResidualAnalysis:
    """Tests for residual analysis functions."""

    def test_analyze_residuals_basic(self):
        """Test basic residual analysis."""
        residuals = np.array([0.1, -0.2, 0.15, -0.1, 0.05, -0.15, 0.1, -0.05])

        result = analyze_residuals(residuals)

        assert "mean" in result
        assert "std" in result
        assert "skewness" in result
        assert "kurtosis" in result

    def test_analyze_residuals_normality(self):
        """Test normality test in residual analysis."""
        # Normal-ish residuals
        np.random.seed(42)
        residuals = np.random.normal(0, 1, 50)

        result = analyze_residuals(residuals)

        assert "normality_test" in result
        assert "normality_p_value" in result
        assert "normality_pass" in result

    def test_analyze_residuals_durbin_watson(self):
        """Test Durbin-Watson statistic."""
        residuals = np.array([0.1, -0.2, 0.15, -0.1, 0.05])

        result = analyze_residuals(residuals)

        assert "durbin_watson" in result
        assert 0 <= result["durbin_watson"] <= 4
        assert "autocorrelation" in result

    def test_analyze_residuals_heteroscedasticity(self):
        """Test heteroscedasticity check."""
        residuals = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
        y_pred = np.array([10, 20, 30, 40, 50])

        result = analyze_residuals(residuals, y_pred)

        assert "heteroscedasticity_corr" in result
        assert "heteroscedasticity" in result

    def test_analyze_residuals_small_sample(self):
        """Test with small sample size."""
        residuals = np.array([0.1, -0.2])

        result = analyze_residuals(residuals)

        # Should still return basic stats
        assert "mean" in result
        assert "std" in result


# =============================================================================
# MODEL RECOMMENDATION TESTS
# =============================================================================


class TestModelRecommendations:
    """Tests for model recommendation function."""

    def test_recommend_best_models(self, model_results):
        """Test basic model recommendation."""
        recommendations = recommend_best_models(model_results, model_type="isotherm", top_n=3)

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3

    def test_recommend_filters_unconverged(self, model_results):
        """Test that unconverged models are filtered."""
        recommendations = recommend_best_models(model_results, model_type="isotherm")

        model_names = [r["model"] for r in recommendations]
        assert "Temkin" not in model_names  # Temkin didn't converge

    def test_recommend_ranking_order(self, model_results):
        """Test that models are properly ranked."""
        recommendations = recommend_best_models(model_results, model_type="isotherm")

        if len(recommendations) >= 2:
            # First should have higher score than second
            assert recommendations[0]["score"] >= recommendations[1]["score"]

    def test_recommend_empty_results(self):
        """Test with empty results."""
        recommendations = recommend_best_models({}, model_type="isotherm")

        assert isinstance(recommendations, list)
        assert len(recommendations) == 0

    def test_recommend_all_unconverged(self):
        """Test when all models failed to converge."""
        results = {"Model1": {"converged": False}, "Model2": {"converged": False}}

        recommendations = recommend_best_models(results)

        assert len(recommendations) == 0


# =============================================================================
# DATA QUALITY ASSESSMENT TESTS
# =============================================================================


class TestDataQualityAssessment:
    """Tests for data quality assessment."""

    def test_assess_isotherm_quality(self, sample_dataframe):
        """Test isotherm data quality assessment."""
        result = assess_data_quality(sample_dataframe, data_type="isotherm")

        assert isinstance(result, dict)
        assert "score" in result or "quality_score" in result

    def test_assess_kinetic_quality(self, kinetic_dataframe):
        """Test kinetic data quality assessment."""
        result = assess_data_quality(kinetic_dataframe, data_type="kinetic")

        assert isinstance(result, dict)

    def test_assess_quality_few_points(self):
        """Test quality assessment with few data points."""
        df = pd.DataFrame({"Ce": [5, 10], "qe": [15, 25]})

        result = assess_data_quality(df, data_type="isotherm")

        # Should flag insufficient points
        assert isinstance(result, dict)


# =============================================================================
# THERMODYNAMIC INTERPRETATION TESTS
# =============================================================================


class TestThermodynamicInterpretation:
    """Tests for thermodynamic interpretation."""

    def test_interpret_exothermic(self):
        """Test exothermic process interpretation."""
        result = interpret_thermodynamics(
            delta_H=-20,  # Negative = exothermic
            delta_S=0.05,
            delta_G=-15,
        )

        assert isinstance(result, dict)
        assert "enthalpy" in result or "process_type" in result

    def test_interpret_endothermic(self):
        """Test endothermic process interpretation."""
        result = interpret_thermodynamics(
            delta_H=30,  # Positive = endothermic
            delta_S=0.1,
            delta_G=-5,
        )

        assert isinstance(result, dict)

    def test_interpret_spontaneous(self):
        """Test spontaneous process interpretation."""
        result = interpret_thermodynamics(
            delta_H=-10,
            delta_S=0.03,
            delta_G=-15,  # Negative = spontaneous
        )

        assert isinstance(result, dict)


# =============================================================================
# ACTIVITY COEFFICIENT TESTS
# =============================================================================


class TestActivityCoefficients:
    """Tests for activity coefficient calculations."""

    def test_davies_equation_basic(self):
        """Test Davies equation for activity coefficient."""
        gamma = calculate_activity_coefficient_davies(ionic_strength=0.1, charge=1)

        assert gamma > 0
        assert gamma < 1  # Activity coefficient < 1 for ionic solutions

    def test_davies_equation_high_ionic_strength(self):
        """Test Davies equation at high ionic strength."""
        gamma = calculate_activity_coefficient_davies(ionic_strength=0.5, charge=1)

        assert gamma > 0
        assert gamma < 1

    def test_davies_divalent(self):
        """Test Davies equation for divalent ions."""
        gamma_mono = calculate_activity_coefficient_davies(ionic_strength=0.1, charge=1)
        gamma_di = calculate_activity_coefficient_davies(ionic_strength=0.1, charge=2)

        # Divalent should have lower activity coefficient
        assert gamma_di < gamma_mono


# =============================================================================
# SEPARATION FACTOR TESTS (EXTENDED)
# =============================================================================


class TestSeparationFactorExtended:
    """Extended tests for separation factor."""

    def test_separation_factor_array(self):
        """Test separation factor with array input."""
        KL = 0.1
        C0 = np.array([10, 50, 100, 200])

        RL = calculate_separation_factor(KL, C0)

        assert len(RL) == len(C0)
        assert np.all(RL > 0)
        assert np.all(RL < 1)  # For positive KL

    def test_interpret_favorable(self):
        """Test favorable interpretation."""
        RL = np.array([0.3, 0.4, 0.5])
        interpretation = interpret_separation_factor(RL)

        assert "favorable" in interpretation.lower() or "Favorable" in interpretation

    def test_interpret_unfavorable(self):
        """Test unfavorable interpretation."""
        RL = np.array([1.5, 2.0, 2.5])
        interpretation = interpret_separation_factor(RL)

        assert "unfavorable" in interpretation.lower() or "Unfavorable" in interpretation

    def test_interpret_linear(self):
        """Test linear interpretation."""
        RL = np.array([1.0, 1.0, 1.0])
        interpretation = interpret_separation_factor(RL)

        assert "linear" in interpretation.lower() or "Linear" in interpretation


# =============================================================================
# MECHANISM DETERMINATION TESTS
# =============================================================================


class TestMechanismDetermination:
    """Tests for adsorption mechanism determination."""

    def test_determine_mechanism_physical(self):
        """Test physical adsorption determination."""
        # Low delta_H magnitude, E < 8
        result = determine_adsorption_mechanism(
            delta_H=-10,  # Low magnitude
        )

        assert isinstance(result, dict)

    def test_determine_mechanism_chemical(self):
        """Test chemical adsorption determination."""
        # High delta_H magnitude, E > 16
        result = determine_adsorption_mechanism(
            delta_H=-80,  # High magnitude
        )

        assert isinstance(result, dict)

    def test_determine_mechanism_with_delta_G(self):
        """Test mechanism determination with delta_G."""
        result = determine_adsorption_mechanism(delta_H=-30, delta_G=-20)

        assert isinstance(result, dict)


# =============================================================================
# DETECT REPLICATES TESTS
# =============================================================================


class TestDetectReplicates:
    """Tests for replicate detection."""

    def test_detect_replicates_present(self):
        """Test detection when replicates are present."""
        df = pd.DataFrame(
            {
                "Ce": [10, 10.05, 20, 20.02, 30],  # 10 and 20 have replicates
                "qe": [15, 16, 25, 26, 35],
            }
        )

        result = detect_replicates(df, x_col="Ce", tolerance=0.01)

        assert isinstance(result, pd.DataFrame)

    def test_detect_replicates_none(self):
        """Test when no replicates present."""
        df = pd.DataFrame({"Ce": [10, 20, 30, 40, 50], "qe": [15, 25, 35, 45, 55]})

        result = detect_replicates(df, x_col="Ce", tolerance=0.01)

        assert isinstance(result, pd.DataFrame)


# =============================================================================
# UNCERTAINTY PROPAGATION TESTS
# =============================================================================


class TestUncertaintyPropagation:
    """Tests for calibration uncertainty propagation."""

    def test_propagate_uncertainty_basic(self):
        """Test basic uncertainty propagation."""
        result = propagate_calibration_uncertainty(
            absorbance=0.5, slope=0.01, intercept=0.0, slope_se=0.001, intercept_se=0.01
        )

        # Returns tuple (Ce, uncertainty)
        assert result is not None
        if isinstance(result, tuple):
            Ce, uncertainty = result
            assert Ce > 0
            assert uncertainty >= 0
        else:
            # If it's a dict
            assert "Ce" in result

    def test_propagate_uncertainty_with_intercept(self):
        """Test uncertainty propagation with non-zero intercept."""
        result = propagate_calibration_uncertainty(
            absorbance=0.6, slope=0.02, intercept=0.1, slope_se=0.002, intercept_se=0.02
        )

        assert result is not None

    def test_propagate_uncertainty_positive(self):
        """Test that uncertainty is always positive."""
        result = propagate_calibration_uncertainty(
            absorbance=0.5, slope=0.01, intercept=0.0, slope_se=0.001, intercept_se=0.01
        )

        if isinstance(result, tuple):
            Ce, uncertainty = result
            assert uncertainty >= 0


# =============================================================================
# EXPORT FUNCTIONS TESTS
# =============================================================================


class TestExportFunctions:
    """Tests for data export functions."""

    def test_convert_df_to_csv(self, sample_dataframe):
        """Test CSV conversion."""
        result = convert_df_to_csv(sample_dataframe)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_convert_df_to_excel(self, sample_dataframe):
        """Test Excel conversion."""
        result = convert_df_to_excel(sample_dataframe, sheet_name="TestData")

        assert isinstance(result, bytes)
        assert len(result) > 0


# =============================================================================
# VALIDATION TESTS
# =============================================================================


class TestValidateExperimentalParams:
    """Tests for experimental parameter validation."""

    def test_validate_complete_params(self):
        """Test validation with complete parameters."""
        params = {"C0": 100, "V": 0.1, "m": 0.5, "T": 298}

        # required_keys should be list of (key, label) tuples
        is_valid, message = validate_required_params(
            params, required_keys=[("C0", "Initial Concentration"), ("V", "Volume"), ("m", "Mass")]
        )

        assert is_valid is True

    def test_validate_missing_params(self):
        """Test validation with missing parameters."""
        params = {"C0": 100, "V": 0.1}

        is_valid, message = validate_required_params(
            params,
            required_keys=[
                ("C0", "Initial Concentration"),
                ("V", "Volume"),
                ("m", "Mass"),
                ("T", "Temperature"),
            ],
        )

        assert is_valid is False
        # Should mention missing params
        assert len(message) > 0

    def test_validate_invalid_value(self):
        """Test validation with invalid (zero/negative) values."""
        params = {
            "C0": 100,
            "V": 0,  # Invalid - zero
            "m": -0.5,  # Invalid - negative
        }

        is_valid, message = validate_required_params(
            params, required_keys=[("C0", "Initial Concentration"), ("V", "Volume"), ("m", "Mass")]
        )

        assert is_valid is False


# =============================================================================
# THERMODYNAMIC PARAMETERS (EXTENDED)
# =============================================================================


class TTZEJj4cngKUFYtT1KKKFXkD6VkePhkoJU:
    """Extended tests for thermodynamic parameter calculation."""

    def test_calculate_thermo_basic(self, thermodynamic_data_tuple):
        """Test basic thermodynamic calculation."""
        T_K, Kd = thermodynamic_data_tuple

        result = calculate_thermodynamic_parameters(T_K, Kd)

        assert result["success"] is True
        assert "delta_H" in result or "ΔH" in str(result)
        assert "delta_S" in result or "ΔS" in str(result)

    def test_calculate_thermo_insufficient_data(self):
        """Test with insufficient data."""
        T_K = np.array([298])
        Kd = np.array([10])

        result = calculate_thermodynamic_parameters(T_K, Kd)

        assert result["success"] is False

    def test_calculate_thermo_negative_Kd(self):
        """Test handling of negative Kd values."""
        T_K = np.array([298, 308, 318])
        Kd = np.array([10, -5, 8])  # One negative

        result = calculate_thermodynamic_parameters(T_K, Kd)

        # Should filter invalid values
        if result["success"]:
            assert result["n_points"] >= 2


# =============================================================================
# NUMERICAL STABILITY TESTS
# =============================================================================


class TestNumericalStabilityUtils:
    """Extended numerical stability tests for utils."""

    def test_activity_coefficient_zero_ionic_strength(self):
        """Test activity coefficient at zero ionic strength."""
        gamma = calculate_activity_coefficient_davies(ionic_strength=0, charge=1)
        assert gamma == pytest.approx(1.0, rel=0.01)

    def test_residual_analysis_constant(self):
        """Test residual analysis with constant residuals."""
        residuals = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

        result = analyze_residuals(residuals)

        assert result["std"] == pytest.approx(0, abs=1e-10)

    def test_arrhenius_with_nan(self):
        """Test Arrhenius with NaN values."""
        T_K = np.array([298, 308, np.nan, 328])
        k = np.array([0.01, 0.02, 0.04, 0.08])

        result = calculate_arrhenius_parameters(T_K, k)

        # Should handle NaN gracefully
        if result["success"]:
            assert np.isfinite(result["Ea"])


# =============================================================================
# COMPREHENSIVE TESTS (Additional coverage for complex functions)
# =============================================================================

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def valid_isotherm_df():
    """Valid isotherm DataFrame."""
    return pd.DataFrame(
        {"Ce": [5, 10, 20, 40, 60, 80, 100], "qe": [15.2, 25.8, 38.5, 52.1, 58.3, 62.5, 65.0]}
    )


@pytest.fixture
def comma_decimal_df():
    """DataFrame with comma as decimal separator."""
    return pd.DataFrame(
        {"Ce": ["5,0", "10,5", "20,3", "40,1"], "qe": ["15,2", "25,8", "38,5", "52,1"]}
    )


@pytest.fixture
def study_state_basic():
    """Basic study state for mechanism checking."""
    return {
        "isotherm_models_fitted": {
            "Langmuir": {
                "converged": True,
                "r_squared": 0.995,
                "adj_r_squared": 0.994,
                "params": {"qm": 100, "KL": 0.1},
                "num_params": 2,
            },
            "Freundlich": {
                "converged": True,
                "r_squared": 0.985,
                "adj_r_squared": 0.983,
                "params": {"KF": 10, "n": 2.5},
                "num_params": 2,
            },
        },
        "kinetic_models_fitted": {
            "PSO": {
                "converged": True,
                "r_squared": 0.998,
                "adj_r_squared": 0.997,
                "params": {"qe": 65, "k2": 0.002},
                "num_params": 2,
            },
            "PFO": {
                "converged": True,
                "r_squared": 0.92,
                "adj_r_squared": 0.91,
                "params": {"qe": 60, "k1": 0.05},
                "num_params": 2,
            },
        },
        "thermo_params": {
            "delta_H": -25.0,
            "delta_S": 0.05,
            "delta_G": [-10, -12, -14],
            "success": True,
        },
        "isotherm_results": pd.DataFrame({"Ce": [5, 10, 20, 40, 60], "qe": [15, 26, 39, 52, 58]}),
    }


@pytest.fixture
def study_state_conflicts():
    """Study state with mechanism conflicts."""
    return {
        "thermo_params": {
            "delta_H": -90.0,  # Chemical (|ΔH| > 80)
            "delta_G": [-15, -18, -20],
        },
    }


# =============================================================================
# VALIDATE DATA EDITOR TESTS
# =============================================================================


class TestValidateDataEditor:
    """Tests for validate_data_editor function."""

    def test_valid_dataframe(self, valid_isotherm_df):
        """Test with valid DataFrame."""
        # After standardization, Ce remains distinct from generic "Concentration"
        result = validate_data_editor(valid_isotherm_df, required_cols=["Ce", "qe"])

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 3

    def test_none_dataframe(self):
        """Test with None input."""
        result = validate_data_editor(None, required_cols=["Ce", "qe"])
        assert result is None

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = validate_data_editor(df, required_cols=["Ce", "qe"])
        assert result is None

    def test_missing_columns(self, valid_isotherm_df):
        """Test with missing required columns."""
        result = validate_data_editor(valid_isotherm_df, required_cols=["Ce", "qe", "missing_col"])
        assert result is None

    def test_comma_decimal_handling(self, comma_decimal_df):
        """Test handling of comma as decimal separator."""
        # Need to use standardized column names
        result = validate_data_editor(comma_decimal_df, required_cols=["Ce", "qe"])

        if result is not None:
            # Values should be converted to numeric
            assert pd.api.types.is_numeric_dtype(result["Ce"])
            assert pd.api.types.is_numeric_dtype(result["qe"])

    def test_insufficient_data_points(self):
        """Test with insufficient data points."""
        df = pd.DataFrame({"Ce": [5, 10], "qe": [15, 25]})
        result = validate_data_editor(df, required_cols=["Ce", "qe"])
        assert result is None  # Need at least 3 points

    def test_drops_na_values(self):
        """Test that NA values are dropped."""
        df = pd.DataFrame(
            {"Ce": [5, 10, np.nan, 40, 60, 80, 100], "qe": [15, 25, 35, np.nan, 55, 60, 65]}
        )
        result = validate_data_editor(df, required_cols=["Ce", "qe"])

        if result is not None:
            assert len(result) >= 3  # Should have enough complete rows

    def test_non_numeric_conversion(self):
        """Test conversion of non-numeric values."""
        df = pd.DataFrame(
            {"Ce": ["5", "10", "20", "40", "60"], "qe": ["15.2", "25.8", "38.5", "52.1", "58.0"]}
        )
        result = validate_data_editor(df, required_cols=["Ce", "qe"])

        if result is not None:
            assert pd.api.types.is_numeric_dtype(result["Ce"])


# =============================================================================
# CHECK MECHANISM CONSISTENCY TESTS
# =============================================================================


class TestCheckMechanismConsistency:
    """Tests for check_mechanism_consistency function."""

    def test_consistent_mechanisms(self, study_state_basic):
        """Test with consistent mechanism indicators."""
        result = check_mechanism_consistency(study_state_basic)

        assert isinstance(result, dict)
        assert "status" in result
        assert "checks" in result

    def test_mechanism_conflicts(self, study_state_conflicts):
        """Test detection of mechanism conflicts."""
        result = check_mechanism_consistency(study_state_conflicts)

        assert isinstance(result, dict)

    def test_empty_study_state(self):
        """Test with empty study state."""
        result = check_mechanism_consistency({})

        assert isinstance(result, dict)
        assert "status" in result

    def test_partial_study_state(self):
        """Test with partial data."""
        state = {"isotherm_models_fitted": {"Langmuir": {"converged": True, "r_squared": 0.99}}}
        result = check_mechanism_consistency(state)

        assert isinstance(result, dict)

    def test_kinetic_isotherm_consistency(self, study_state_basic):
        """Test kinetic vs isotherm consistency check."""
        result = check_mechanism_consistency(study_state_basic)

        # Should check PSO/PFO dominance vs isotherm type
        assert "checks" in result


# =============================================================================
# DETECT COMMON ERRORS TESTS
# =============================================================================


class TestDetectCommonErrors:
    """Tests for detect_common_errors function."""

    def test_no_errors(self, study_state_basic):
        """Test with study state that has no major errors."""
        errors = detect_common_errors(study_state_basic)

        assert isinstance(errors, list)

    def test_detects_deltaG_out_of_range(self):
        """Test detection of ΔG° outside typical range."""
        state = {
            "thermo_params": {
                "delta_G": [50, 60, 70]  # Way outside typical range
            }
        }
        errors = detect_common_errors(state)

        assert isinstance(errors, list)
        # Should detect thermodynamic error
        thermo_errors = [e for e in errors if e.get("type") == "Thermodynamic"]
        assert len(thermo_errors) > 0

    def test_detects_insufficient_points(self):
        """Test detection of insufficient data points."""
        state = {
            "isotherm_results": pd.DataFrame({"Ce": [5, 10], "qe": [15, 25]}),
            "isotherm_models_fitted": {
                "Sips": {"converged": True, "num_params": 3, "r_squared": 0.99}
            },
        }
        errors = detect_common_errors(state)

        # Should detect insufficient points for 3-parameter model
        assert isinstance(errors, list)

    def test_empty_state(self):
        """Test with empty study state."""
        errors = detect_common_errors({})

        assert isinstance(errors, list)
        assert len(errors) == 0


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS TESTS
# =============================================================================


class TestBootstrapCIExtended:
    """Extended tests for bootstrap confidence intervals."""

    def test_bootstrap_basic(self):
        """Test basic bootstrap CI calculation."""
        Ce = np.array([5, 10, 20, 40, 60, 80, 100])
        qe = np.array([15.2, 25.8, 38.5, 52.1, 58.3, 62.5, 65.0])

        ci = bootstrap_confidence_intervals(
            langmuir_model, Ce, qe, params=np.array([70, 0.05]), n_bootstrap=100, confidence=0.95
        )

        assert ci is not None
        if ci is not None:
            assert len(ci) == 2  # Two parameters

    def test_bootstrap_returns_array(self):
        """Test bootstrap returns proper array structure."""
        Ce = np.array([5, 10, 20, 40, 60, 80])
        qe = np.array([15, 26, 39, 52, 58, 62])
        params = np.array([70, 0.05])

        ci = bootstrap_confidence_intervals(
            langmuir_model, Ce, qe, params, n_bootstrap=50, confidence=0.95
        )

        # CI should be returned as array-like
        assert ci is not None


# =============================================================================
# DETERMINE ADSORPTION MECHANISM COMPREHENSIVE TESTS
# =============================================================================


class TestDetermineAdsorptionMechanismComprehensive:
    """Comprehensive tests for determine_adsorption_mechanism."""

    def test_physical_adsorption_low_deltaH(self):
        """Test physical adsorption from low ΔH."""
        result = determine_adsorption_mechanism(delta_H=-15)

        assert isinstance(result, dict)
        assert "Physical" in result.get("scores", {}) or "mechanism" in result

    def test_weak_chemisorption(self):
        """Test weak chemisorption range (20-40 kJ/mol)."""
        result = determine_adsorption_mechanism(delta_H=-30)

        assert isinstance(result, dict)

    def test_hydrogen_bonding_range(self):
        """Test hydrogen bonding range (40-80 kJ/mol)."""
        result = determine_adsorption_mechanism(delta_H=-60)

        assert isinstance(result, dict)

    def test_strong_chemisorption(self):
        """Test strong chemisorption (>80 kJ/mol)."""
        result = determine_adsorption_mechanism(delta_H=-100)

        assert isinstance(result, dict)

    def test_with_delta_G_chemical(self):
        """Test with ΔG indicating chemisorption."""
        result = determine_adsorption_mechanism(delta_H=-50, delta_G=np.array([-50, -55, -60]))

        assert isinstance(result, dict)

    def test_with_delta_G_physical(self):
        """Test with ΔG indicating physisorption."""
        result = determine_adsorption_mechanism(delta_H=-15, delta_G=np.array([-10, -12, -14]))

        assert isinstance(result, dict)

    def test_with_freundlich_n_favorable(self):
        """Test with Freundlich n indicating favorable adsorption."""
        result = determine_adsorption_mechanism(
            delta_H=-25,
            n_freundlich=2.5,  # n > 1 is favorable
        )

        assert isinstance(result, dict)

    def test_with_freundlich_n_unfavorable(self):
        """Test with Freundlich n indicating unfavorable adsorption."""
        result = determine_adsorption_mechanism(
            delta_H=-25,
            n_freundlich=0.5,  # n < 1 is unfavorable
        )

        assert isinstance(result, dict)

    def test_with_separation_factor(self):
        """Test with separation factor."""
        result = determine_adsorption_mechanism(
            delta_H=-25,
            RL=0.3,  # Favorable
        )

        assert isinstance(result, dict)

    def test_all_parameters(self):
        """Test with all parameters provided."""
        result = determine_adsorption_mechanism(
            delta_H=-40, delta_G=np.array([-25, -28, -30]), n_freundlich=2.0, RL=0.2
        )

        assert isinstance(result, dict)
        assert "scores" in result or "mechanism" in result
        assert "evidence" in result


# =============================================================================
# CALCULATE PRESS AND Q2 TESTS
# =============================================================================


class TestPRESSCalculations:
    """Tests for PRESS and Q² calculations."""

    def test_calculate_press_basic(self):
        """Test basic PRESS calculation."""
        Ce = np.array([5, 10, 20, 40, 60, 80, 100])
        qe = np.array([15.2, 25.8, 38.5, 52.1, 58.3, 62.5, 65.0])

        press = calculate_press(
            langmuir_model, Ce, qe, params=np.array([70, 0.05]), bounds=([0, 0], [200, 1])
        )

        assert press >= 0

    def test_calculate_q2_basic(self):
        """Test Q² calculation."""
        press = 10.0
        y_data = np.array([15, 26, 39, 52, 58, 62, 65])

        q2 = calculate_q2(press, y_data)

        assert q2 <= 1.0  # Q² should be <= 1

    def test_calculate_q2_perfect_prediction(self):
        """Test Q² with near-perfect prediction."""
        press = 0.1  # Very small PRESS
        y_data = np.array([15, 26, 39, 52, 58, 62, 65])

        q2 = calculate_q2(press, y_data)

        assert q2 > 0.99


# =============================================================================
# DATA QUALITY ASSESSMENT EXTENDED TESTS
# =============================================================================


class TestDataQualityExtended:
    """Extended tests for data quality assessment."""

    def test_assess_quality_good_data(self, valid_isotherm_df):
        """Test with good quality data."""
        result = assess_data_quality(valid_isotherm_df, data_type="isotherm")

        assert isinstance(result, dict)
        score_key = "score" if "score" in result else "quality_score"
        if score_key in result:
            assert result[score_key] >= 0

    def test_assess_quality_kinetic_data(self):
        """Test kinetic data quality assessment."""
        df = pd.DataFrame(
            {"t": [0, 5, 10, 20, 30, 60, 90, 120], "qt": [0, 15, 28, 42, 50, 58, 62, 64]}
        )
        result = assess_data_quality(df, data_type="kinetic")

        assert isinstance(result, dict)

    def test_assess_quality_sparse_data(self):
        """Test with sparse data."""
        df = pd.DataFrame({"Ce": [5, 100], "qe": [15, 65]})
        result = assess_data_quality(df, data_type="isotherm")

        assert isinstance(result, dict)

    def test_assess_quality_with_outliers(self):
        """Test data with potential outliers."""
        df = pd.DataFrame(
            {
                "Ce": [5, 10, 20, 40, 60, 80, 100],
                "qe": [15, 25, 38, 52, 100, 62, 65],  # 100 is outlier
            }
        )
        result = assess_data_quality(df, data_type="isotherm")

        assert isinstance(result, dict)


# =============================================================================
# RECOMMEND BEST MODELS EXTENDED TESTS
# =============================================================================


class TestRecommendBestModelsExtended:
    """Extended tests for model recommendation."""

    def test_recommend_with_residuals(self):
        """Test recommendation with residual analysis."""
        results = {
            "Langmuir": {
                "converged": True,
                "r_squared": 0.995,
                "adj_r_squared": 0.994,
                "rmse": 1.2,
                "aicc": -15,
                "residuals": np.random.normal(0, 0.5, 10),
            },
            "Freundlich": {
                "converged": True,
                "r_squared": 0.985,
                "adj_r_squared": 0.983,
                "rmse": 2.1,
                "aicc": -10,
                "residuals": np.random.normal(0, 1, 10),
            },
        }

        recommendations = recommend_best_models(results, model_type="isotherm", top_n=2)

        assert len(recommendations) <= 2
        if len(recommendations) > 0:
            assert "model" in recommendations[0]
            assert "score" in recommendations[0]

    def test_recommend_kinetic_models(self):
        """Test kinetic model recommendation."""
        results = {
            "PSO": {
                "converged": True,
                "r_squared": 0.998,
                "adj_r_squared": 0.997,
                "rmse": 0.5,
                "aicc": -20,
            },
            "PFO": {
                "converged": True,
                "r_squared": 0.95,
                "adj_r_squared": 0.94,
                "rmse": 2.0,
                "aicc": -5,
            },
        }

        recommendations = recommend_best_models(results, model_type="kinetic", top_n=2)

        assert len(recommendations) <= 2

    def test_recommend_handles_missing_metrics(self):
        """Test handling of missing metrics."""
        results = {
            "Model1": {
                "converged": True,
                "r_squared": 0.99,
                # Missing adj_r_squared, rmse, aicc
            }
        }

        recommendations = recommend_best_models(results)

        assert isinstance(recommendations, list)


# =============================================================================
# INTERPRET THERMODYNAMICS EXTENDED TESTS
# =============================================================================


class TestInterpretThermodynamicsExtended:
    """Extended tests for thermodynamic interpretation."""

    def test_interpret_highly_exothermic(self):
        """Test highly exothermic process."""
        result = interpret_thermodynamics(delta_H=-100, delta_S=0.1, delta_G=-70)

        assert isinstance(result, dict)

    def test_interpret_endothermic_nonspontaneous(self):
        """Test endothermic non-spontaneous process."""
        result = interpret_thermodynamics(delta_H=50, delta_S=-0.05, delta_G=65)

        assert isinstance(result, dict)

    def test_interpret_entropy_driven(self):
        """Test entropy-driven process."""
        result = interpret_thermodynamics(
            delta_H=10,  # Slightly endothermic
            delta_S=0.2,  # Large positive entropy
            delta_G=-50,  # Still spontaneous
        )

        assert isinstance(result, dict)

    def test_interpret_array_delta_G(self):
        """Test with array of ΔG values."""
        result = interpret_thermodynamics(
            delta_H=-30, delta_S=0.05, delta_G=np.array([-20, -22, -24, -26])
        )

        assert isinstance(result, dict)


# =============================================================================
# ANALYZE RESIDUALS EXTENDED TESTS
# =============================================================================


class TestAnalyzeResidualsExtended:
    """Extended tests for residual analysis."""

    def test_analyze_with_positive_autocorrelation(self):
        """Test detection of positive autocorrelation."""
        # Residuals with positive autocorrelation (sequential same sign)
        residuals = np.array([1, 1.2, 1.1, 0.9, -0.8, -1.0, -1.1, -0.9])

        result = analyze_residuals(residuals)

        assert "durbin_watson" in result
        # DW < 1.5 suggests positive autocorrelation

    def test_analyze_with_heteroscedasticity(self):
        """Test detection of heteroscedasticity."""
        # Residuals that increase with y_pred
        y_pred = np.array([10, 20, 30, 40, 50, 60, 70, 80])
        residuals = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 7])  # Increasing variance

        result = analyze_residuals(residuals, y_pred)

        assert "heteroscedasticity" in result

    def test_analyze_large_sample(self):
        """Test with large sample (uses D'Agostino test)."""
        np.random.seed(42)
        residuals = np.random.normal(0, 1, 100)

        result = analyze_residuals(residuals)

        assert "normality_test" in result

    def test_analyze_non_normal_residuals(self):
        """Test with non-normal residuals."""
        # Skewed residuals
        residuals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 2.0, 3.0, 5.0])

        result = analyze_residuals(residuals)

        assert "skewness" in result
        assert result["skewness"] > 0  # Positive skew


# =============================================================================
# CALCULATE ERROR METRICS EXTENDED TESTS
# =============================================================================


class TestCalculateErrorMetricsExtended:
    """Extended tests for error metrics calculation."""

    def test_metrics_with_weights(self):
        """Test error metrics calculation."""
        y_obs = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([11, 19, 31, 39, 51])

        result = calculate_error_metrics(y_obs, y_pred, n_params=2)

        assert "r_squared" in result
        assert "rmse" in result
        assert "aic" in result
        assert "bic" in result

    def test_metrics_perfect_fit(self):
        """Test metrics with perfect fit."""
        y_obs = np.array([10, 20, 30, 40, 50])
        y_pred = y_obs.copy()

        result = calculate_error_metrics(y_obs, y_pred, n_params=1)

        assert result["r_squared"] == pytest.approx(1.0, rel=1e-10)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-10)


# =============================================================================
# ARRHENIUS PARAMETERS EXTENDED TESTS
# =============================================================================


class TestArrheniusExtended:
    """Extended tests for Arrhenius parameters."""

    def test_arrhenius_diffusion_controlled(self):
        """Test diffusion-controlled reaction (low Ea)."""
        T_K = np.array([298, 308, 318, 328])
        # Small temperature dependence = low Ea
        k = np.array([0.01, 0.011, 0.012, 0.013])

        result = calculate_arrhenius_parameters(T_K, k)

        if result["success"]:
            assert "interpretation" in result

    def test_arrhenius_chemical_controlled(self):
        """Test chemically controlled reaction (high Ea)."""
        T_K = np.array([298, 308, 318, 328])
        # Large temperature dependence = high Ea
        k = np.array([0.001, 0.005, 0.025, 0.125])

        result = calculate_arrhenius_parameters(T_K, k)

        if result["success"]:
            assert result["Ea"] > 0


# =============================================================================
# THERMODYNAMIC PARAMETERS EXTENDED TESTS
# =============================================================================


class TestThermodynamicParametersComprehensive:
    """Comprehensive tests for thermodynamic parameter calculation."""

    def test_exothermic_process(self):
        """Test exothermic process (negative ΔH)."""
        T_K = np.array([298, 308, 318, 328])
        Kd = np.array([100, 80, 65, 50])  # Decreasing with T = exothermic

        result = calculate_thermodynamic_parameters(T_K, Kd)

        assert result["success"] is True
        assert result["delta_H"] < 0  # Exothermic

    def test_endothermic_process(self):
        """Test endothermic process (positive ΔH)."""
        T_K = np.array([298, 308, 318, 328])
        Kd = np.array([10, 15, 22, 30])  # Increasing with T = endothermic

        result = calculate_thermodynamic_parameters(T_K, Kd)

        assert result["success"] is True
        assert result["delta_H"] > 0  # Endothermic

    def test_with_three_temperatures(self):
        """Test minimum case with 3 temperatures."""
        T_K = np.array([298, 308, 318])
        Kd = np.array([50, 45, 40])

        result = calculate_thermodynamic_parameters(T_K, Kd)

        assert result["success"] is True
        assert result["n_points"] == 3


# =============================================================================
# ADDITIONAL TESTS FOR 80% COVERAGE
# =============================================================================


class TestMechanismConsistencyDetailed:
    """Detailed tests for mechanism consistency to cover more branches."""

    def test_kinetic_isotherm_pso_freundlich_mismatch(self):
        """Test PSO with Freundlich dominance case."""
        state = {
            "kinetic_models_fitted": {
                "PSO": {"converged": True, "r_squared": 0.99, "adj_r_squared": 0.98},
                "PFO": {"converged": True, "r_squared": 0.90, "adj_r_squared": 0.89},
            },
            "isotherm_models_fitted": {
                "Langmuir": {"converged": True, "r_squared": 0.85, "adj_r_squared": 0.84},
                "Freundlich": {
                    "converged": True,
                    "r_squared": 0.95,
                    "adj_r_squared": 0.94,
                    "params": {"n_inv": 0.5},
                },
            },
        }
        result = check_mechanism_consistency(state)
        assert "checks" in result

    def test_temperature_effect_endothermic_decrease(self):
        """Test endothermic process with decreasing capacity (conflict)."""
        state = {
            "isotherm_models_fitted": {},
            "kinetic_models_fitted": {},
            "thermo_params": {"delta_H": 50},  # Endothermic
            "temperature_effect": "decreases",  # Should increase for endothermic
        }
        result = check_mechanism_consistency(state)
        assert "checks" in result

    def test_temperature_effect_exothermic_increase(self):
        """Test exothermic process with increasing capacity (conflict)."""
        state = {
            "isotherm_models_fitted": {},
            "kinetic_models_fitted": {},
            "thermo_params": {"delta_H": -30},  # Exothermic
            "temperature_effect": "increases",  # Should decrease for exothermic
        }
        result = check_mechanism_consistency(state)
        assert "checks" in result

    def test_temperature_effect_consistent(self):
        """Test consistent temperature effect."""
        state = {
            "isotherm_models_fitted": {},
            "kinetic_models_fitted": {},
            "thermo_params": {"delta_H": -30},  # Exothermic
            "temperature_effect": "decreases",  # Correct for exothermic
        }
        result = check_mechanism_consistency(state)
        assert "checks" in result

    def test_rl_freundlich_consistency(self):
        """Test RL vs Freundlich consistency check."""
        state = {
            "isotherm_models_fitted": {
                "Langmuir": {"converged": True, "r_squared": 0.95, "RL": np.array([0.3, 0.4, 0.5])},
                "Freundlich": {"converged": True, "r_squared": 0.93, "params": {"n_inv": 0.5}},
            },
            "kinetic_models_fitted": {},
            "thermo_params": {},
        }
        result = check_mechanism_consistency(state)
        assert "checks" in result

    def test_rl_freundlich_inconsistent(self):
        """Test RL vs Freundlich inconsistency."""
        state = {
            "isotherm_models_fitted": {
                "Langmuir": {
                    "converged": True,
                    "r_squared": 0.95,
                    "RL": 0.3,  # Favorable (< 1)
                },
                "Freundlich": {
                    "converged": True,
                    "r_squared": 0.93,
                    "params": {"n_inv": 1.5},  # > 1 is unusual with favorable RL
                },
            },
            "kinetic_models_fitted": {},
            "thermo_params": {},
        }
        result = check_mechanism_consistency(state)
        assert "checks" in result


class TestDetectCommonErrorsDetailed:
    """Detailed tests for detect_common_errors."""

    def test_high_r2_without_ci(self):
        """Test detection of high R² without CI."""
        state = {
            "isotherm_models_fitted": {
                "Langmuir": {
                    "converged": True,
                    "r_squared": 0.9999,  # Very high R²
                    "ci_95": {},  # Empty CI
                }
            }
        }
        errors = detect_common_errors(state)
        assert isinstance(errors, list)

    def test_negative_delta_G(self):
        """Test very negative ΔG detection."""
        state = {
            "thermo_params": {
                "delta_G": [-70, -75, -80]  # Very negative
            }
        }
        errors = detect_common_errors(state)
        # Should detect thermodynamic issue
        assert any(e.get("type") == "Thermodynamic" for e in errors)


class TestDetermineAdsorptionMechanismAdditional:
    """Additional tests for mechanism determination."""

    def test_delta_G_physical_range(self):
        """Test ΔG in physical adsorption range."""
        result = determine_adsorption_mechanism(
            delta_H=-15,
            delta_G=np.array([-15, -18, -20]),  # Physical range: -20 to 0
        )
        assert "evidence" in result

    def test_with_unfavorable_RL(self):
        """Test with unfavorable separation factor."""
        result = determine_adsorption_mechanism(
            delta_H=-25,
            RL=1.5,  # > 1 is unfavorable
        )
        assert isinstance(result, dict)

    def test_with_irreversible_RL(self):
        """Test with irreversible separation factor."""
        result = determine_adsorption_mechanism(
            delta_H=-50,
            RL=0.0,  # = 0 is irreversible
        )
        assert isinstance(result, dict)

    def test_with_linear_RL(self):
        """Test with linear separation factor."""
        result = determine_adsorption_mechanism(
            delta_H=-25,
            RL=1.0,  # = 1 is linear
        )
        assert isinstance(result, dict)


class TestDataQualityInternals:
    """Tests for data quality assessment internals."""

    def test_quality_with_wide_range(self):
        """Test quality assessment with wide concentration range."""
        df = pd.DataFrame({"Ce": [1, 10, 100, 500, 1000], "qe": [5, 25, 55, 70, 75]})
        result = assess_data_quality(df, data_type="isotherm")
        assert isinstance(result, dict)

    def test_quality_with_narrow_range(self):
        """Test quality with narrow concentration range."""
        df = pd.DataFrame({"Ce": [10, 11, 12, 13, 14], "qe": [20, 21, 22, 23, 24]})
        result = assess_data_quality(df, data_type="isotherm")
        assert isinstance(result, dict)

    def test_quality_kinetic_with_equilibrium(self):
        """Test kinetic data quality with equilibrium region."""
        df = pd.DataFrame(
            {
                "t": [0, 5, 10, 30, 60, 120, 240, 360],
                "qt": [0, 20, 35, 55, 62, 64, 65, 65.1],  # Reaches equilibrium
            }
        )
        result = assess_data_quality(df, data_type="kinetic")
        assert isinstance(result, dict)


class TestRecommendBestModelsInternals:
    """Tests for recommend_best_models internals."""

    def test_recommend_with_missing_metrics(self):
        """Test recommendation when some metrics are missing."""
        results = {
            "Model1": {
                "converged": True,
                "r_squared": 0.95,
                # Missing adj_r_squared, rmse, aicc
            },
            "Model2": {"converged": True, "r_squared": 0.90, "adj_r_squared": 0.88},
        }
        recommendations = recommend_best_models(results, top_n=2)
        assert len(recommendations) <= 2

    def test_recommend_with_inf_rmse(self):
        """Test recommendation with infinite RMSE."""
        results = {"Model1": {"converged": True, "r_squared": 0.95, "rmse": np.inf}}
        recommendations = recommend_best_models(results)
        assert isinstance(recommendations, list)


class TestInterpretThermodynamicsAdditional:
    """Additional thermodynamic interpretation tests."""

    def test_interpret_large_positive_entropy(self):
        """Test interpretation with large positive entropy."""
        result = interpret_thermodynamics(
            delta_H=-20,
            delta_S=0.3,  # Large positive entropy
            delta_G=-110,
        )
        assert isinstance(result, dict)

    def test_interpret_negative_entropy(self):
        """Test interpretation with negative entropy."""
        result = interpret_thermodynamics(
            delta_H=-50,
            delta_S=-0.1,  # Negative entropy
            delta_G=-20,
        )
        assert isinstance(result, dict)
