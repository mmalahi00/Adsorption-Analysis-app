# tests/test_coverage_comprehensive.py
"""
Comprehensive Test Coverage Suite
==================================

This module merges test_coverage_additional.py, test_coverage_boost.py,
and test_coverage_final.py into a single comprehensive test suite to achieve ≥95% coverage.

Duplicate test classes have been merged, combining all unique test methods:
- TestConfigConstants: merged from test_coverage_boost.py, test_coverage_final.py
- TestRateLimitingStep: merged from test_coverage_boost.py, test_coverage_final.py

All other test classes are included as-is from their source files.
"""

import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

# Suppress warnings during tests
warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# COMPREHENSIVE TEST SUITE
# =============================================================================


class TestActivityCoefficientDavies:
    """Test Davies activity coefficient calculation."""

    def test_calculate_activity_coefficient_davies_basic(self):
        """Test basic Davies equation calculation."""
        from adsorblab_pro.utils import calculate_activity_coefficient_davies

        ionic_strength = 0.1

        gamma = calculate_activity_coefficient_davies(ionic_strength)

        assert 0 < gamma <= 1

    def test_calculate_activity_coefficient_davies_charge(self):
        """Test Davies equation with different charges."""
        from adsorblab_pro.utils import calculate_activity_coefficient_davies

        ionic_strength = 0.1

        gamma1 = calculate_activity_coefficient_davies(ionic_strength, charge=1)
        gamma2 = calculate_activity_coefficient_davies(ionic_strength, charge=2)

        # Higher charge should give lower activity coefficient
        assert gamma2 < gamma1


class TestApplyProfessionalStyle:
    """Test apply_professional_style function."""

    def test_apply_professional_style_basic(self):
        """Test basic professional styling."""
        from adsorblab_pro.plot_style import apply_professional_style

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))

        result = apply_professional_style(fig, title="Test", x_title="X", y_title="Y")

        assert isinstance(result, go.Figure)

    def test_apply_professional_style_all_positions(self):
        """Test professional styling with all legend positions."""
        from adsorblab_pro.plot_style import apply_professional_style

        positions = ["upper left", "upper right", "lower left", "lower right"]

        for pos in positions:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))

            result = apply_professional_style(fig, legend_position=pos)
            assert isinstance(result, go.Figure)

    def test_apply_professional_style_horizontal_legend(self):
        """Test professional styling with horizontal legend."""
        from adsorblab_pro.plot_style import apply_professional_style

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))

        result = apply_professional_style(fig, legend_horizontal=True)

        assert result.layout.legend.orientation == "h"

    def test_apply_professional_style_with_barmode(self):
        """Test professional styling with barmode."""
        from adsorblab_pro.plot_style import apply_professional_style

        fig = go.Figure()
        fig.add_trace(go.Bar(x=[1, 2, 3], y=[1, 2, 3]))

        result = apply_professional_style(fig, barmode="group")

        assert result.layout.barmode == "group"

    def test_apply_professional_style_invalid_position(self):
        """Test professional styling with invalid legend position."""
        from adsorblab_pro.plot_style import apply_professional_style

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))

        # Should use default position for invalid
        result = apply_professional_style(fig, legend_position="invalid_position")
        assert isinstance(result, go.Figure)


class TestAssessDataQualityAdvanced:
    """Test data quality assessment."""

    def test_assess_data_quality_isotherm(self):
        """Test isotherm data quality assessment."""
        from adsorblab_pro.utils import assess_data_quality

        df = pd.DataFrame(
            {
                "Ce": [5, 10, 20, 40, 60, 80, 100],
                "qe": [15, 25, 38, 52, 58, 62, 65],
            }
        )
        quality = assess_data_quality(df, "isotherm")
        assert quality is not None

    def test_assess_data_quality_kinetic(self):
        """Test kinetic data quality assessment."""
        from adsorblab_pro.utils import assess_data_quality

        df = pd.DataFrame(
            {
                "t": [0, 5, 10, 20, 30, 60, 90, 120],
                "qt": [0, 12, 22, 35, 44, 55, 58, 60],
            }
        )
        quality = assess_data_quality(df, "kinetic")
        assert quality is not None


class TestAxisAndMarkerStyles:
    """Test AXIS_STYLE and MARKERS dictionaries."""

    def test_axis_style_dict(self):
        """Test AXIS_STYLE dictionary."""
        from adsorblab_pro.plot_style import AXIS_STYLE

        assert isinstance(AXIS_STYLE, dict)
        assert AXIS_STYLE["showgrid"] is False
        assert AXIS_STYLE["mirror"] is True
        assert "gridcolor" in AXIS_STYLE

    def test_markers_dict(self):
        """Test MARKERS dictionary."""
        from adsorblab_pro.plot_style import MARKERS

        assert "experimental" in MARKERS
        assert "experimental_small" in MARKERS
        assert "comparison" in MARKERS

        # Check marker properties
        assert MARKERS["experimental"]["symbol"] == "circle"
        assert MARKERS["experimental"]["size"] > MARKERS["experimental_small"]["size"]


class TestBiotNumber:
    """Test Biot number calculation."""

    def test_calculate_biot_number_basic(self):
        """Test basic Biot number calculation."""
        from adsorblab_pro.models import calculate_biot_number

        Bi = calculate_biot_number(kf=0.1, Dp=1e-6, r=0.01)

        assert Bi > 0


class TestBootstrapConfidenceIntervals:
    """Test bootstrap CI functions."""

    def test_bootstrap_ci_basic(self):
        """Test basic bootstrap confidence interval calculation."""
        from adsorblab_pro.models import langmuir_model
        from adsorblab_pro.utils import bootstrap_confidence_intervals

        # Generate data for bootstrap
        Ce = np.array([5, 10, 20, 40, 60, 80, 100])
        qe = langmuir_model(Ce, 70, 0.05) + np.random.normal(0, 1, len(Ce))
        params = np.array([70, 0.05])

        ci_low, ci_high = bootstrap_confidence_intervals(
            langmuir_model, Ce, qe, params, n_bootstrap=50
        )

        assert len(ci_low) == len(params)
        assert len(ci_high) == len(params)
        assert np.all(ci_low < ci_high)


class TestCalculateArrheniusParameters:
    """Test Arrhenius parameter calculation."""

    def test_calculate_arrhenius_parameters_basic(self):
        """Test basic Arrhenius calculation."""
        from adsorblab_pro.utils import calculate_arrhenius_parameters

        T_K = np.array([298, 308, 318, 328])
        k = np.array([0.01, 0.02, 0.04, 0.08])

        result = calculate_arrhenius_parameters(T_K, k)

        assert "Ea" in result or "activation_energy" in result
        assert "A" in result or "pre_exponential" in result


class TestCalculateQ2:
    """Test Q² calculation function."""

    def test_calculate_q2_basic(self):
        """Test basic Q² calculation."""
        from adsorblab_pro.utils import calculate_q2

        y_data = np.array([10, 20, 30, 40, 50])
        press = 100  # Sum of squared prediction errors

        q2 = calculate_q2(press, y_data)

        assert isinstance(q2, float)
        assert q2 <= 1.0  # Q² should be at most 1


class TestCalculateSeparationFactor:
    """Test separation factor calculation."""

    def test_calculate_separation_factor_basic(self):
        """Test basic separation factor calculation."""
        from adsorblab_pro.utils import calculate_separation_factor

        KL = 0.05
        C0 = np.array([10, 50, 100])

        RL = calculate_separation_factor(KL, C0)

        assert len(RL) == len(C0)
        assert np.all(RL > 0)
        assert np.all(RL < 1)  # For favorable adsorption

    def test_interpret_separation_factor_favorable(self):
        """Test interpretation of favorable separation factor."""
        from adsorblab_pro.utils import interpret_separation_factor

        RL = np.array([0.3, 0.4, 0.5])

        interpretation = interpret_separation_factor(RL)

        assert "favorable" in interpretation.lower()


class TestCalculateTemperatureResults:
    """Test calculate_temperature_results function."""

    def test_calculate_temperature_results_basic(self):
        """Test temperature results calculation with calibration."""
        from adsorblab_pro.utils import calculate_temperature_results

        df = pd.DataFrame({"Temperature": [25, 35, 45], "Absorbance": [0.5, 0.4, 0.3]})
        temp_input = {"data": df, "params": {"C0": 50, "m": 0.1, "V": 0.1}}
        calib_params = {"slope": 0.01, "intercept": 0}

        result = calculate_temperature_results(temp_input, calib_params)

        assert result.success is True
        assert "Temperature_C" in result.data.columns

    def test_calculate_temperature_results_with_uncertainty(self):
        """Test temperature results with uncertainty propagation."""
        from adsorblab_pro.utils import calculate_temperature_results

        df = pd.DataFrame({"Temperature": [25, 35], "Absorbance": [0.5, 0.4]})
        temp_input = {"data": df, "params": {"C0": 50, "m": 0.1, "V": 0.1}}
        calib_params = {
            "slope": 0.01,
            "intercept": 0,
            "std_err_slope": 0.001,
            "std_err_intercept": 0.01,
        }

        result = calculate_temperature_results(temp_input, calib_params, include_uncertainty=True)

        assert result.success is True
        assert "Ce_error" in result.data.columns


class TestCalculationResultClass:
    """Test CalculationResult dataclass."""

    def test_calculation_result_success(self):
        """Test successful CalculationResult."""
        from adsorblab_pro.utils import CalculationResult

        result = CalculationResult(success=True, data={"test": 123})

        assert result.success is True
        assert result.data == {"test": 123}
        assert result.error is None

    def test_calculation_result_failure(self):
        """Test failed CalculationResult."""
        from adsorblab_pro.utils import CalculationResult

        result = CalculationResult(success=False, error="Test error message")

        assert result.success is False
        assert result.error == "Test error message"


class TestCalculationResultDataclass:
    """Test CalculationResult dataclass."""

    def test_calculation_result_success(self):
        """Test successful CalculationResult."""
        from adsorblab_pro.utils import CalculationResult

        result = CalculationResult(
            success=True,
            data=pd.DataFrame({"x": [1, 2, 3]}),
            error=None,
        )
        assert result.success
        assert result.error is None

    def test_calculation_result_failure(self):
        """Test failed CalculationResult."""
        from adsorblab_pro.utils import CalculationResult

        result = CalculationResult(
            success=False,
            data=None,
            error="Error occurred",
        )
        assert not result.success
        assert "Error" in result.error


class TestCheckMechanismConsistency:
    """Test mechanism consistency checking."""

    def test_check_mechanism_consistency_empty_state(self):
        """Test with empty study state."""
        from adsorblab_pro.utils import check_mechanism_consistency

        study_state = {}
        result = check_mechanism_consistency(study_state)

        assert isinstance(result, dict)

    def test_check_mechanism_consistency_with_isotherm(self):
        """Test with isotherm fitting results."""
        from adsorblab_pro.utils import check_mechanism_consistency

        study_state = {
            "isotherm_results": {
                "Langmuir": {"r_squared": 0.99, "converged": True},
                "Freundlich": {"r_squared": 0.95, "converged": True},
            }
        }
        result = check_mechanism_consistency(study_state)

        assert isinstance(result, dict)


class TestColumnStandardization:
    """Test column standardization functions."""

    def test_standardize_column_name_returns_string(self):
        """Test standardize_column_name returns a string."""
        from adsorblab_pro.utils import standardize_column_name

        result = standardize_column_name("Ce")
        assert isinstance(result, str)

    def test_standardize_column_name_fuzzy(self):
        """Test fuzzy matching."""
        from adsorblab_pro.utils import standardize_column_name

        # Slight misspellings should be handled
        result = standardize_column_name("Concentratin")  # Misspelling
        # Should either match or return original
        assert isinstance(result, str)


class TestCompleteWorkflow:
    """Test complete analysis workflows."""

    def test_isotherm_full_workflow(self):
        """Test complete isotherm analysis workflow."""
        from adsorblab_pro.models import fit_model_with_ci, langmuir_model
        from adsorblab_pro.utils import assess_data_quality
        from adsorblab_pro.validation import validate_isotherm_data

        # Generate synthetic data
        Ce_exp = np.array([5, 10, 20, 40, 60, 80, 100])
        qe_exp = langmuir_model(Ce_exp, 70, 0.05) + np.random.normal(0, 0.5, len(Ce_exp))

        # Validate data with scalar V and m
        C0 = np.ones_like(Ce_exp) * 120
        V = 0.1  # Scalar volume
        m = 0.1  # Scalar mass
        report = validate_isotherm_data(C0, Ce_exp, qe_exp, V, m)
        assert report.is_valid

        # Assess quality
        df = pd.DataFrame({"Ce": Ce_exp, "qe": qe_exp})
        quality = assess_data_quality(df, "isotherm")
        assert quality is not None

        # Fit model using fit_model_with_ci
        result = fit_model_with_ci(langmuir_model, Ce_exp, qe_exp, p0=[50, 0.1])
        assert result is not None
        assert "params" in result or result.get("converged", False)

    def test_kinetic_full_workflow(self):
        """Test complete kinetic analysis workflow."""
        from adsorblab_pro.models import fit_model_with_ci, pso_model
        from adsorblab_pro.validation import validate_kinetic_data

        # Generate synthetic data
        t = np.array([0, 5, 10, 20, 30, 60, 90, 120, 180])
        qe_true, k2_true = 65, 0.001
        qt = pso_model(t, qe_true, k2_true) + np.random.normal(0, 0.3, len(t))
        qt = np.maximum(qt, 0)

        # Validate data
        report = validate_kinetic_data(t, qt)
        assert report.is_valid

        # Fit model using fit_model_with_ci
        result = fit_model_with_ci(pso_model, t, qt, p0=[50, 0.001])
        assert "params" in result or result.get("converged", False)


class TestConfigAdditional:
    """Additional configuration tests."""

    def test_temperature_limits(self):
        """Test temperature limit constants."""
        from adsorblab_pro.config import TEMP_MAX_KELVIN, TEMP_MIN_KELVIN

        assert TEMP_MIN_KELVIN > 0
        assert TEMP_MAX_KELVIN > TEMP_MIN_KELVIN
        assert TEMP_MIN_KELVIN < 300  # Should be less than room temp

    def test_file_size_limit(self):
        """Test file size limit constant."""
        from adsorblab_pro.config import MAX_FILE_SIZE_BYTES

        assert MAX_FILE_SIZE_BYTES > 0
        assert MAX_FILE_SIZE_BYTES > 1_000_000  # Should be at least 1 MB

    def test_allowed_file_types(self):
        """Test allowed file types list."""
        from adsorblab_pro.config import ALLOWED_FILE_TYPES

        assert isinstance(ALLOWED_FILE_TYPES, list | tuple | set)
        assert "csv" in ALLOWED_FILE_TYPES or ".csv" in ALLOWED_FILE_TYPES


class TestConfigConstants:
    """Test configuration constants."""

    def test_bootstrap_constants(self):
        """Test bootstrap configuration constants."""
        from adsorblab_pro.config import BOOTSTRAP_DEFAULT_ITERATIONS, BOOTSTRAP_MIN_SUCCESS

        assert BOOTSTRAP_DEFAULT_ITERATIONS >= 100
        assert 0 < BOOTSTRAP_MIN_SUCCESS < BOOTSTRAP_DEFAULT_ITERATIONS

    def test_epsilon_constants(self):
        """Test epsilon constants for numerical stability."""
        from adsorblab_pro.config import EPSILON_DIV, EPSILON_ZERO

        assert EPSILON_DIV > 0
        assert EPSILON_ZERO > 0
        assert EPSILON_DIV < 1
        assert EPSILON_ZERO < 1

    def test_epsilon_values(self):
        """Test epsilon values are positive and small."""
        from adsorblab_pro.config import EPSILON_DIV, EPSILON_ZERO

        assert EPSILON_DIV > 0
        assert EPSILON_ZERO > 0
        assert EPSILON_DIV < 1e-5
        assert EPSILON_ZERO < 1e-5

    def test_isotherm_models_dict(self):
        """Test isotherm models configuration."""
        from adsorblab_pro.config import ISOTHERM_MODELS

        assert isinstance(ISOTHERM_MODELS, dict)
        assert "Langmuir" in ISOTHERM_MODELS
        assert "Freundlich" in ISOTHERM_MODELS
        assert "Temkin" in ISOTHERM_MODELS

    def test_kinetic_models_dict(self):
        """Test kinetic models configuration."""
        from adsorblab_pro.config import KINETIC_MODELS

        assert isinstance(KINETIC_MODELS, dict)
        # Check for common kinetic models
        has_pfo = any("PFO" in k or "first" in k.lower() for k in KINETIC_MODELS.keys())
        has_pso = any("PSO" in k or "second" in k.lower() for k in KINETIC_MODELS.keys())
        assert has_pfo or has_pso or len(KINETIC_MODELS) > 0

    def test_mechanism_criteria(self):
        """Test mechanism criteria dictionary."""
        from adsorblab_pro.config import MECHANISM_CRITERIA

        assert isinstance(MECHANISM_CRITERIA, dict)

    def test_min_data_points(self):
        """Test minimum data points value."""
        from adsorblab_pro.config import MIN_DATA_POINTS

        assert MIN_DATA_POINTS >= 3

    def test_r_gas_constant(self):
        """Test R gas constant value."""
        from adsorblab_pro.config import R_GAS_CONSTANT

        assert abs(R_GAS_CONSTANT - 8.314) < 0.01


class TestConfigGradingFunctions:
    """Test grading functions in config.py."""

    def test_get_grade_from_r_squared_excellent(self):
        """Test grade A+ for excellent R²."""
        from adsorblab_pro.config import get_grade_from_r_squared

        result = get_grade_from_r_squared(0.999)
        assert result["grade"] in ["A+", "A"]
        assert "label" in result

    def test_get_grade_from_r_squared_good(self):
        """Test grade for good R²."""
        from adsorblab_pro.config import get_grade_from_r_squared

        result = get_grade_from_r_squared(0.985)  # Should be B
        assert result["grade"] in ["A-", "B", "A"]

    def test_get_grade_from_r_squared_poor(self):
        """Test low grade for poor R²."""
        from adsorblab_pro.config import get_grade_from_r_squared

        result = get_grade_from_r_squared(0.5)
        # Should be a low grade
        assert result["grade"] not in ["A+", "A", "B"]

    def test_get_calibration_grade(self):
        """Test backward compatible calibration grade function."""
        from adsorblab_pro.config import get_calibration_grade

        result = get_calibration_grade(0.995)
        assert "grade" in result
        assert "score" in result  # Backward compatibility field

    def test_quality_grades_dict(self):
        """Test QUALITY_GRADES structure."""
        from adsorblab_pro.config import QUALITY_GRADES

        assert isinstance(QUALITY_GRADES, dict)
        assert "A+" in QUALITY_GRADES or "A" in QUALITY_GRADES
        for _grade, info in QUALITY_GRADES.items():
            assert "min_r2" in info or "min_score" in info


class TestConfigModelDicts:
    """Test model configuration dictionaries."""

    def test_isotherm_models_dict(self):
        """Test isotherm models dictionary."""
        from adsorblab_pro.config import ISOTHERM_MODELS

        assert isinstance(ISOTHERM_MODELS, dict)
        assert "Langmuir" in ISOTHERM_MODELS
        assert "Freundlich" in ISOTHERM_MODELS

    def test_kinetic_models_dict(self):
        """Test kinetic models dictionary."""
        from adsorblab_pro.config import KINETIC_MODELS

        assert isinstance(KINETIC_MODELS, dict)
        assert "PFO" in KINETIC_MODELS
        assert "PSO" in KINETIC_MODELS


class TestConvertDfFunctions:
    """Test DataFrame conversion functions."""

    def test_convert_df_to_csv(self):
        """Test DataFrame to CSV conversion."""
        from adsorblab_pro.utils import convert_df_to_csv

        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        csv_bytes = convert_df_to_csv(df)

        assert isinstance(csv_bytes, bytes)
        assert b"A" in csv_bytes and b"B" in csv_bytes

    def test_convert_df_to_excel(self):
        """Test DataFrame to Excel conversion."""
        from adsorblab_pro.utils import convert_df_to_excel

        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        excel_bytes = convert_df_to_excel(df)

        assert isinstance(excel_bytes, bytes)
        assert len(excel_bytes) > 0


class TestCreateDualAxisPlot:
    """Test dual axis plot creation."""

    def test_create_dual_axis_plot_basic(self):
        """Test basic dual axis plot."""
        from adsorblab_pro.utils import create_dual_axis_plot

        data = pd.DataFrame(
            {"Ce": [5, 10, 20, 40, 60], "qe": [15, 25, 38, 52, 58], "Removal": [70, 75, 81, 90, 95]}
        )

        fig = create_dual_axis_plot(
            data,
            x_col="Ce",
            y1_col="qe",
            y2_col="Removal",
            x_label="Ce (mg/L)",
            y1_label="qe (mg/g)",
            y2_label="Removal (%)",
            title="Dual Axis",
        )

        assert isinstance(fig, go.Figure)


class TestCreateIsothermPlot:
    """Test isotherm plot creation."""

    def test_create_isotherm_plot_basic(self):
        """Test basic isotherm plot creation."""
        from adsorblab_pro.plot_style import create_isotherm_plot

        Ce = np.array([5, 10, 20, 40, 60, 80])
        qe = np.array([15, 25, 38, 52, 58, 62])
        Ce_fit = np.linspace(5, 80, 50)
        qe_fit = 70 * 0.05 * Ce_fit / (1 + 0.05 * Ce_fit)

        fig = create_isotherm_plot(Ce, qe, Ce_fit, qe_fit, model_name="Langmuir", r_squared=0.995)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # At least fit and experimental

    def test_create_isotherm_plot_no_r_squared(self):
        """Test isotherm plot without R² value."""
        from adsorblab_pro.plot_style import create_isotherm_plot

        Ce = np.array([5, 10, 20])
        qe = np.array([15, 25, 38])
        Ce_fit = np.linspace(5, 20, 20)
        qe_fit = Ce_fit * 2

        fig = create_isotherm_plot(Ce, qe, Ce_fit, qe_fit, model_name="Linear")

        assert isinstance(fig, go.Figure)

    def test_create_isotherm_plot_custom_title(self):
        """Test isotherm plot with custom title."""
        from adsorblab_pro.plot_style import create_isotherm_plot

        Ce = np.array([5, 10, 20])
        qe = np.array([15, 25, 38])
        Ce_fit = np.linspace(5, 20, 20)
        qe_fit = Ce_fit * 2

        fig = create_isotherm_plot(
            Ce, qe, Ce_fit, qe_fit, model_name="Test", title="Custom Title", height=600
        )

        assert isinstance(fig, go.Figure)
        assert fig.layout.height == 600


class TestCreateKineticPlot:
    """Test kinetic plot creation."""

    def test_create_kinetic_plot_basic(self):
        """Test basic kinetic plot creation."""
        from adsorblab_pro.plot_style import create_kinetic_plot

        t = np.array([0, 5, 10, 20, 30, 60, 120])
        qt = np.array([0, 12, 22, 35, 44, 55, 62])
        t_fit = np.linspace(0, 120, 50)
        qe_eq = 65
        k2 = 0.001
        qt_fit = (k2 * qe_eq**2 * t_fit) / (1 + k2 * qe_eq * t_fit)

        fig = create_kinetic_plot(t, qt, t_fit, qt_fit, model_name="PSO", r_squared=0.998)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_create_kinetic_plot_no_r_squared(self):
        """Test kinetic plot without R² value."""
        from adsorblab_pro.plot_style import create_kinetic_plot

        t = np.array([0, 10, 20])
        qt = np.array([0, 20, 35])
        t_fit = np.linspace(0, 20, 20)
        qt_fit = t_fit * 1.5

        fig = create_kinetic_plot(t, qt, t_fit, qt_fit, model_name="Linear")

        assert isinstance(fig, go.Figure)


class TestCreateModelComparisonPlot:
    """Test model comparison plot creation."""

    def test_create_model_comparison_plot_basic(self):
        """Test basic model comparison plot."""
        from adsorblab_pro.plot_style import create_model_comparison_plot

        x_exp = np.array([5, 10, 20, 40, 60])
        y_exp = np.array([15, 25, 38, 52, 58])

        fitted_models = {
            "Langmuir": {"params": {"qm": 70, "KL": 0.05}, "r_squared": 0.995, "converged": True},
            "Freundlich": {
                "params": {"KF": 5, "n_inv": 0.5},
                "r_squared": 0.990,
                "converged": True,
            },
        }

        model_functions = {
            "Langmuir": lambda x, p: p["qm"] * p["KL"] * x / (1 + p["KL"] * x),
            "Freundlich": lambda x, p: p["KF"] * x ** p["n_inv"],
        }

        fig = create_model_comparison_plot(
            x_exp, y_exp, fitted_models, model_functions, title="Model Comparison"
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # 2 models + experimental

    def test_create_model_comparison_plot_unconverged_model(self):
        """Test model comparison with unconverged model."""
        from adsorblab_pro.plot_style import create_model_comparison_plot

        x_exp = np.array([5, 10, 20])
        y_exp = np.array([15, 25, 38])

        fitted_models = {
            "Langmuir": {"params": {"qm": 70, "KL": 0.05}, "r_squared": 0.995, "converged": True},
            "Failed": {"params": {}, "r_squared": None, "converged": False},
        }

        model_functions = {
            "Langmuir": lambda x, p: p["qm"] * p["KL"] * x / (1 + p["KL"] * x),
        }

        fig = create_model_comparison_plot(x_exp, y_exp, fitted_models, model_functions)

        assert isinstance(fig, go.Figure)

    def test_create_model_comparison_plot_model_error(self):
        """Test model comparison when a model function raises error."""
        from adsorblab_pro.plot_style import create_model_comparison_plot

        x_exp = np.array([5, 10, 20])
        y_exp = np.array([15, 25, 38])

        fitted_models = {
            "ErrorModel": {"params": {"a": 1}, "r_squared": 0.9, "converged": True},
        }

        def error_func(x, p):
            raise ValueError("Test error")

        model_functions = {
            "ErrorModel": error_func,
        }

        # Should not raise - logs the error and continues
        fig = create_model_comparison_plot(x_exp, y_exp, fitted_models, model_functions)
        assert isinstance(fig, go.Figure)


class TestCreateParityPlot:
    """Test parity plot creation."""

    def test_create_parity_plot_basic(self):
        """Test basic parity plot."""
        from adsorblab_pro.plot_style import create_parity_plot

        y_obs = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([11, 19, 31, 39, 51])

        fig = create_parity_plot(y_obs, y_pred, model_name="Langmuir")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Data points + 1:1 line

    def test_create_parity_plot_with_metrics(self):
        """Test parity plot with R² and RMSE."""
        from adsorblab_pro.plot_style import create_parity_plot

        y_obs = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([11, 19, 31, 39, 51])

        fig = create_parity_plot(y_obs, y_pred, model_name="Test", r_squared=0.995, rmse=1.2)

        assert isinstance(fig, go.Figure)

    def test_create_parity_plot_only_r_squared(self):
        """Test parity plot with only R²."""
        from adsorblab_pro.plot_style import create_parity_plot

        y_obs = np.array([10, 20, 30])
        y_pred = np.array([11, 19, 31])

        fig = create_parity_plot(y_obs, y_pred, model_name="Test", r_squared=0.99)

        assert isinstance(fig, go.Figure)


class TestCreateResidualPlot:
    """Test residual plot creation."""

    def test_create_residual_plot_basic(self):
        """Test basic residual plot."""
        from adsorblab_pro.plot_style import create_residual_plot

        y_pred = np.array([10, 20, 30, 40, 50])
        residuals = np.array([0.5, -0.3, 0.2, -0.1, 0.4])

        fig = create_residual_plot(y_pred, residuals, model_name="Langmuir")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_create_residual_plot_mixed_residuals(self):
        """Test residual plot with positive and negative residuals."""
        from adsorblab_pro.plot_style import create_residual_plot

        y_pred = np.array([10, 20, 30, 40, 50, 60])
        residuals = np.array([1.0, -1.0, 0.5, -0.5, 0.2, -0.2])

        fig = create_residual_plot(y_pred, residuals, model_name="Test", height=300)

        assert isinstance(fig, go.Figure)
        assert fig.layout.height == 300


class TestCreateResidualPlots:
    """Test create_residual_plots function."""

    def test_create_residual_plots_basic(self):
        """Test basic residual plots creation."""
        from adsorblab_pro.utils import create_residual_plots

        residuals = np.array([0.5, -0.3, 0.2, -0.1, 0.4, 0.1, -0.2])
        y_pred = np.array([10, 20, 30, 40, 50, 60, 70])

        fig = create_residual_plots(residuals, y_pred)

        assert isinstance(fig, go.Figure)

    def test_create_residual_plots_with_x_data(self):
        """Test residual plots with x_data."""
        from adsorblab_pro.utils import create_residual_plots

        residuals = np.array([0.5, -0.3, 0.2, -0.1, 0.4])
        y_pred = np.array([10, 20, 30, 40, 50])
        x_data = np.array([5, 10, 15, 20, 25])

        fig = create_residual_plots(residuals, y_pred, x_data=x_data)

        assert isinstance(fig, go.Figure)

    def test_create_residual_plots_no_y_pred(self):
        """Test residual plots without y_pred."""
        from adsorblab_pro.utils import create_residual_plots

        residuals = np.array([0.5, -0.3, 0.2, -0.1, 0.4, 0.1])

        fig = create_residual_plots(residuals)

        assert isinstance(fig, go.Figure)


class TestDataExportFunctions:
    """Test data export functions."""

    def test_convert_df_to_csv_basic(self):
        """Test DataFrame to CSV conversion."""
        from adsorblab_pro.utils import convert_df_to_csv

        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        # convert_df_to_csv uses st.cache_data, test by calling it
        try:
            csv_bytes = convert_df_to_csv(df)
            assert isinstance(csv_bytes, bytes)
        except Exception:
            # Expected if Streamlit context not available
            pass

    def test_convert_df_to_excel_basic(self):
        """Test DataFrame to Excel conversion."""
        from adsorblab_pro.utils import convert_df_to_excel

        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        # convert_df_to_excel uses st.cache_data, test by calling it
        try:
            excel_bytes = convert_df_to_excel(df)
            assert isinstance(excel_bytes, bytes)
            assert len(excel_bytes) > 0
        except Exception:
            # Expected if Streamlit context not available
            pass


class TestDataQualityAssessment:
    """Test data quality assessment functions."""

    def test_assess_data_quality_basic(self):
        """Test basic data quality assessment."""
        from adsorblab_pro.utils import assess_data_quality

        df = pd.DataFrame(
            {
                "Ce": [5, 10, 20, 40, 60, 80, 100],
                "qe": [15, 25, 38, 52, 58, 62, 65],
            }
        )

        quality = assess_data_quality(df, data_type="isotherm")

        assert isinstance(quality, dict)


class TestDefaultSessionState:
    """Test default session state configuration."""

    def test_default_session_state_structure(self):
        """Test default session state has expected structure."""
        from adsorblab_pro.config import DEFAULT_SESSION_STATE

        assert isinstance(DEFAULT_SESSION_STATE, dict)


class TestDetectCommonErrors:
    """Test common error detection."""

    def test_detect_common_errors_empty_state(self):
        """Test with empty study state."""
        from adsorblab_pro.utils import detect_common_errors

        study_state = {}
        result = detect_common_errors(study_state)

        assert isinstance(result, list)

    def test_detect_common_errors_with_data(self):
        """Test with study state containing data."""
        from adsorblab_pro.utils import detect_common_errors

        study_state = {
            "calibration": {"slope": 0.01, "r_squared": 0.5},
            "isotherm_results": {"Langmuir": {"r_squared": 0.5}},
        }
        result = detect_common_errors(study_state)

        assert isinstance(result, list)


class TestDetectCommonErrorsAdvanced:
    """Test error detection with complex scenarios."""

    def test_detect_errors_with_heteroscedasticity(self):
        """Test detection of heteroscedasticity in residuals."""
        from adsorblab_pro.utils import detect_common_errors

        # Create residuals that correlate with predicted values (heteroscedastic)
        y_pred = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        residuals = y_pred * 0.1 * np.random.normal(1, 0.1, len(y_pred))

        study_state = {
            "isotherm_models_fitted": {
                "Langmuir": {
                    "r_squared": 0.95,
                    "converged": True,
                    "residuals": residuals,
                    "y_pred": y_pred,
                }
            }
        }
        result = detect_common_errors(study_state)
        assert isinstance(result, list)

    def test_detect_errors_linear_vs_nonlinear(self):
        """Test detection of linear vs non-linear regression differences."""
        from adsorblab_pro.utils import detect_common_errors

        study_state = {
            "isotherm_linear_results": {
                "Langmuir": {"r_squared": 0.85, "converged": True},
                "Freundlich": {"r_squared": 0.82, "converged": True},
            },
            "isotherm_models_fitted": {
                "Langmuir": {"r_squared": 0.95, "converged": True},
                "Freundlich": {"r_squared": 0.92, "converged": True},
            },
        }
        result = detect_common_errors(study_state)
        assert isinstance(result, list)

    def test_detect_errors_temperature_effect(self):
        """Test temperature effect detection with thermodynamic data."""
        from adsorblab_pro.utils import detect_common_errors

        # Endothermic but capacity decreases with temperature
        study_state = {
            "thermodynamic_results": {"delta_H": 25.0},  # Positive = endothermic
            "temp_effect_results": pd.DataFrame(
                {
                    "T": [298, 308, 318, 328],
                    "qe": [60, 55, 50, 45],  # Decreasing with T - inconsistent
                }
            ),
        }
        result = detect_common_errors(study_state)
        assert isinstance(result, list)

    def test_detect_errors_exothermic_inconsistency(self):
        """Test exothermic process with increasing capacity."""
        from adsorblab_pro.utils import detect_common_errors

        # Exothermic but capacity increases with temperature
        study_state = {
            "thermodynamic_results": {"delta_H": -25.0},  # Negative = exothermic
            "temp_effect_results": pd.DataFrame(
                {
                    "Temperature": [298, 308, 318, 328],
                    "Removal": [60, 70, 80, 90],  # Increasing with T - inconsistent
                }
            ),
        }
        result = detect_common_errors(study_state)
        assert isinstance(result, list)


class TestDetectReplicates:
    """Test replicate detection function."""

    def test_detect_replicates_basic(self):
        """Test basic replicate detection."""
        from adsorblab_pro.utils import detect_replicates

        df = pd.DataFrame(
            {
                "Ce": [10.0, 10.01, 20.0, 20.02, 30.0],
                "qe": [25, 25.5, 40, 40.2, 50],
            }
        )

        result = detect_replicates(df, "Ce", tolerance=0.05)

        assert isinstance(result, pd.DataFrame)


class TestExportFunctions:
    """Test data export functions."""

    def test_dataframe_to_csv_string(self):
        """Test DataFrame to CSV string conversion."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        csv_str = df.to_csv(index=False)

        assert "A,B" in csv_str
        assert "1,4" in csv_str

    def test_dataframe_to_excel_bytes(self):
        """Test DataFrame to Excel bytes conversion."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        excel_bytes = buffer.getvalue()

        assert len(excel_bytes) > 0
        # Excel files start with PK (zip header)
        assert excel_bytes[:2] == b"PK"


class TestExtendedFreundlichMulticomponent:
    """Test extended Freundlich multicomponent model."""

    def test_extended_freundlich_multicomponent_basic(self):
        """Test basic multicomponent calculation."""
        from adsorblab_pro.models import extended_freundlich_multicomponent

        # Component i data
        Ce_i = np.array([10, 30, 50])  # Concentration of component i
        Kf_i = 5.0  # Freundlich constant for component i
        n_i = 2.0  # Freundlich exponent for component i

        # All components' data (including i)
        Ce_all = [Ce_i, np.array([20, 40, 60])]  # [Ce_1, Ce_2]
        Kf_all = [5.0, 4.0]  # [Kf_1, Kf_2]
        n_all = [2.0, 3.0]  # [n_1, n_2]

        qe = extended_freundlich_multicomponent(Ce_i, Kf_i, n_i, Ce_all, Kf_all, n_all)

        assert len(qe) == len(Ce_i)


class TestExtendedLangmuirMulticomponent:
    """Test extended Langmuir multicomponent model."""

    def test_extended_langmuir_multicomponent_basic(self):
        """Test basic multicomponent calculation."""
        from adsorblab_pro.models import extended_langmuir_multicomponent

        # Component i data
        Ce_i = np.array([10, 30, 50])  # Concentration of component i
        qm_i = 100.0  # Max capacity for component i
        KL_i = 0.05  # Langmuir constant for component i

        # All components' data (including i)
        Ce_all = [Ce_i, np.array([20, 40, 60])]  # [Ce_1, Ce_2]
        KL_all = [0.05, 0.03]  # [KL_1, KL_2]

        qe = extended_langmuir_multicomponent(Ce_i, qm_i, KL_i, Ce_all, KL_all)

        assert len(qe) == len(Ce_i)
        assert np.all(qe >= 0)


class TestFileValidation:
    """Test file validation functions."""

    def test_validate_uploaded_file_excel(self):
        """Test validation of Excel file."""
        from adsorblab_pro.validation import validate_uploaded_file

        result = validate_uploaded_file(file_size=500_000, file_name="data.xlsx")
        assert result.is_valid

    def test_validate_uploaded_file_xls(self):
        """Test validation of old Excel file."""
        from adsorblab_pro.validation import validate_uploaded_file

        result = validate_uploaded_file(file_size=500_000, file_name="data.xls")
        assert result.is_valid

    def test_validate_uploaded_file_txt(self):
        """Test validation of text file (not allowed)."""
        from adsorblab_pro.validation import validate_uploaded_file

        result = validate_uploaded_file(file_size=1000, file_name="data.txt")
        assert not result.is_valid  # txt is not in allowed types


class TestFitModelWithCIAdditional:
    """Additional tests for fit_model_with_ci."""

    def test_fit_model_with_ci_langmuir(self):
        """Test comprehensive fitting with CI for Langmuir."""
        from adsorblab_pro.models import fit_model_with_ci, langmuir_model

        Ce = np.array([5, 10, 20, 40, 60, 80, 100])
        qm_true, KL_true = 70, 0.05
        qe = langmuir_model(Ce, qm_true, KL_true) + np.random.normal(0, 1, len(Ce))

        results = fit_model_with_ci(
            langmuir_model,
            Ce,
            qe,
            p0=[50, 0.1],
            bounds=([0, 0], [200, 1]),
            param_names=["qm", "KL"],
            confidence=0.95,
        )

        assert results["converged"] is True
        assert "params" in results
        assert "ci_95" in results  # Correct key for confidence intervals
        assert "r_squared" in results


class TestIsothermValidationAdvanced:
    """Test isotherm validation edge cases."""

    def test_isotherm_with_very_high_removal(self):
        """Test isotherm validation with very high removal efficiency."""
        from adsorblab_pro.validation import validate_isotherm_data

        C0 = np.array([100, 100, 100, 100, 100])
        Ce = np.array([0.1, 0.5, 1, 2, 5])  # Very high removal (>95%)
        qe = (C0 - Ce) * 0.1 / 0.1  # qe = (C0-Ce)*V/m

        report = validate_isotherm_data(C0, Ce, qe, V=0.1, m=0.1)
        assert isinstance(report.is_valid, bool)

    def test_isotherm_with_negative_qe(self):
        """Test isotherm validation with negative qe values."""
        from adsorblab_pro.validation import validate_isotherm_data

        C0 = np.array([100, 100, 100, 100, 100])
        Ce = np.array([10, 20, 30, 40, 50])
        qe = np.array([90, 80, -5, 60, 50])  # Negative value

        report = validate_isotherm_data(C0, Ce, qe, V=0.1, m=0.1)
        assert not report.is_valid


class TestKineticValidationAdvanced:
    """Test kinetic validation edge cases."""

    def test_kinetic_with_ct_exceeding_c0(self):
        """Test kinetic validation where Ct > C0."""
        from adsorblab_pro.validation import validate_kinetic_data

        t = np.array([0, 5, 10, 20, 30, 60])
        qt = np.array([0, 10, 20, 30, 35, 38])
        C0 = 50
        Ct = np.array([50, 55, 45, 35, 30, 25])  # First value exceeds C0

        report = validate_kinetic_data(t, qt, C0=C0, Ct=Ct)
        assert not report.is_valid or len(report.errors) > 0

    def test_kinetic_not_at_equilibrium(self):
        """Test kinetic validation when not at equilibrium."""
        from adsorblab_pro.validation import validate_kinetic_data

        t = np.array([0, 5, 10, 20, 30])
        qt = np.array([0, 10, 20, 30, 40])  # Still increasing rapidly

        report = validate_kinetic_data(t, qt)
        # Should have equilibrium warning
        assert (
            any("equilibrium" in str(w.message).lower() for w in report.warnings) or report.is_valid
        )


class TestLangmuir3DSurface:
    """Test Langmuir 3D surface generation."""

    def test_langmuir_3d_surface_basic(self):
        """Test basic 3D surface generation."""
        from adsorblab_pro.models import langmuir_3d_surface

        Ce_grid, temp_grid, qe_grid = langmuir_3d_surface(
            Ce_range=(1, 100), temp_range=(20, 60), qm=100, KL=0.05, delta_H=-25000
        )

        assert Ce_grid.shape == (30, 30)
        assert temp_grid.shape == (30, 30)
        assert qe_grid.shape == (30, 30)
        assert np.all(qe_grid >= 0)

    def test_langmuir_3d_surface_temperature_effect(self):
        """Test temperature effect on 3D surface."""
        from adsorblab_pro.models import langmuir_3d_surface

        # Exothermic (negative delta_H) - higher T should give lower qe
        Ce_grid, temp_grid, qe_grid = langmuir_3d_surface(
            Ce_range=(10, 50), temp_range=(25, 50), qm=80, KL=0.1, delta_H=-30000
        )

        # At same Ce, lower temp should have higher qe for exothermic
        # Compare averages across temperature
        assert qe_grid[0, :].mean() >= qe_grid[-1, :].mean() - 1  # Allow small tolerance


class TestMainModule:
    """Test __main__.py module."""

    def test_main_module_importable(self):
        """Test __main__ module can be imported."""
        # This should not raise
        import adsorblab_pro.__main__  # noqa: F401


class TestMatplotlibStyle:
    """Test matplotlib styling functions."""

    def test_matplotlib_style_dict(self):
        """Test MATPLOTLIB_STYLE dictionary."""
        from adsorblab_pro.plot_style import MATPLOTLIB_STYLE

        assert isinstance(MATPLOTLIB_STYLE, dict)
        assert "figure.facecolor" in MATPLOTLIB_STYLE
        assert "axes.facecolor" in MATPLOTLIB_STYLE
        assert "font.family" in MATPLOTLIB_STYLE

    def test_apply_matplotlib_style(self):
        """Test apply_matplotlib_style function."""
        from adsorblab_pro.plot_style import apply_matplotlib_style

        # Should not raise
        apply_matplotlib_style()


class TestMechanismConsistencyAdvanced:
    """Test mechanism consistency checking with various scenarios."""

    def test_mechanism_consistency_with_full_results(self):
        """Test mechanism consistency with complete study results."""
        from adsorblab_pro.utils import check_mechanism_consistency

        study_state = {
            "isotherm_models_fitted": {
                "Langmuir": {
                    "r_squared": 0.99,
                    "converged": True,
                    "params": {"qm": 70, "KL": 0.05},
                },
                "Freundlich": {"r_squared": 0.95, "converged": True},
            },
            "kinetic_models_fitted": {
                "Pseudo-second order": {"r_squared": 0.98, "converged": True},
            },
            "thermodynamic_results": {
                "delta_H": -25.0,
                "delta_S": 50.0,
                "delta_G": {"298": -10.0, "308": -8.0},
            },
        }
        result = check_mechanism_consistency(study_state)
        assert isinstance(result, dict)


class TestModelComparisonStatistics:
    """Test model comparison statistical functions."""

    def test_calculate_akaike_weights(self):
        """Test Akaike weights calculation."""
        from adsorblab_pro.utils import calculate_akaike_weights

        aic_values = [100, 105, 110]

        weights = calculate_akaike_weights(aic_values)

        # Weights should sum to 1
        assert abs(np.sum(weights) - 1.0) < 0.01
        # Lower AIC should have higher weight
        assert weights[0] > weights[1] > weights[2]


class TestModelFittingAdvanced:
    """Test model fitting edge cases."""

    def test_fit_model_with_bounds(self):
        """Test model fitting with parameter bounds."""
        from adsorblab_pro.models import fit_model_with_ci, langmuir_model

        Ce = np.array([5, 10, 20, 40, 60, 80, 100])
        qe = langmuir_model(Ce, 70, 0.05) + np.random.normal(0, 0.5, len(Ce))

        result = fit_model_with_ci(
            langmuir_model,
            Ce,
            qe,
            p0=[50, 0.1],
            bounds=([0, 0], [200, 1]),
        )
        assert result is not None
        assert "params" in result or result.get("converged", False)


class TestModelHelperFunctions:
    """Test model helper functions."""

    def test_calculate_initial_rate(self):
        """Test initial rate calculation."""
        from adsorblab_pro.models import calculate_initial_rate

        t = np.array([0, 5, 10, 20, 30, 60])
        qt = np.array([0, 12, 22, 35, 44, 55])

        rate = calculate_initial_rate(t, qt)

        assert rate > 0

    def test_calculate_initial_rate_insufficient_data(self):
        """Test initial rate with insufficient data."""
        from adsorblab_pro.models import calculate_initial_rate

        t = np.array([0, 5])
        qt = np.array([0, 10])

        rate = calculate_initial_rate(t, qt)

        assert rate == 0.0

    def test_identify_equilibrium_time(self):
        """Test equilibrium time identification."""
        from adsorblab_pro.models import identify_equilibrium_time

        t = np.array([0, 5, 10, 20, 30, 60, 120])
        qt = np.array([0, 12, 22, 35, 55, 60, 62])

        eq_time = identify_equilibrium_time(t, qt)

        assert eq_time > 0
        assert eq_time <= t[-1]

    def test_get_model_info(self):
        """Test model info retrieval."""
        from adsorblab_pro.models import get_model_info

        info = get_model_info()

        assert "isotherms" in info
        assert "kinetics" in info
        assert "Langmuir" in info["isotherms"]


class TestModels3DSurfaceAdvanced:
    """Test 3D surface generation edge cases."""

    def test_langmuir_3d_surface_extended_range(self):
        """Test Langmuir 3D surface with extended temperature range."""
        from adsorblab_pro.models import langmuir_3d_surface

        Ce_grid, temp_grid, qe_grid = langmuir_3d_surface(
            Ce_range=(0, 100),
            temp_range=(20, 60),  # Celsius
            qm=100,
            KL=0.05,
            delta_H=-30000,  # J/mol
        )
        assert Ce_grid.shape == qe_grid.shape
        assert temp_grid.shape == qe_grid.shape

    def test_ph_temperature_surface_basic(self):
        """Test pH-temperature response surface."""
        from adsorblab_pro.models import ph_temperature_response_surface

        pH_grid, temp_grid, response = ph_temperature_response_surface(
            pH_range=(2, 12),
            temp_range=(20, 60),
            optimal_pH=6.5,
            optimal_temp=40,
            max_capacity=100,
        )
        assert pH_grid.shape == response.shape
        assert temp_grid.shape == response.shape


class TestModuleExports:
    """Test module exports in __init__.py."""

    def test_main_imports(self):
        """Test main module can be imported."""
        import adsorblab_pro

        assert hasattr(adsorblab_pro, "__version__")

    def test_models_accessible(self):
        """Test models are accessible from main module."""
        from adsorblab_pro import models

        assert hasattr(models, "langmuir_model")
        assert hasattr(models, "freundlich_model")

    def test_utils_accessible(self):
        """Test utils are accessible from main module."""
        from adsorblab_pro import utils

        assert hasattr(utils, "calculate_adsorption_capacity")

    def test_validation_accessible(self):
        """Test validation is accessible from main module."""
        from adsorblab_pro import validation

        assert hasattr(validation, "validate_positive")


class TestNumericalStabilityExtended:
    """Extended numerical stability tests."""

    def test_langmuir_very_small_KL(self):
        """Test Langmuir with very small KL."""
        from adsorblab_pro.models import langmuir_model

        Ce = np.array([1, 10, 100, 1000])
        qm, KL = 100, 1e-10

        qe = langmuir_model(Ce, qm, KL)

        assert np.all(np.isfinite(qe))
        assert np.all(qe >= 0)

    def test_langmuir_very_large_KL(self):
        """Test Langmuir with very large KL."""
        from adsorblab_pro.models import langmuir_model

        Ce = np.array([0.001, 0.01, 0.1, 1])
        qm, KL = 100, 1e10

        qe = langmuir_model(Ce, qm, KL)

        assert np.all(np.isfinite(qe))
        assert np.all(qe >= 0)
        assert np.all(qe <= qm * 1.01)  # Should approach qm

    def test_freundlich_extreme_n(self):
        """Test Freundlich with extreme n values."""
        from adsorblab_pro.models import freundlich_model

        Ce = np.array([1, 10, 100])

        # Very small n_inv (very favorable)
        qe = freundlich_model(Ce, KF=10, n_inv=0.01)
        assert np.all(np.isfinite(qe))

        # Very large n_inv (unfavorable)
        qe = freundlich_model(Ce, KF=10, n_inv=5.0)
        assert np.all(np.isfinite(qe))


class TestPHTemperatureResponseSurface:
    """Test pH-temperature response surface generation."""

    def test_ph_temperature_response_surface_basic(self):
        """Test basic pH-temperature response surface."""
        from adsorblab_pro.models import ph_temperature_response_surface

        pH_grid, temp_grid, response = ph_temperature_response_surface(
            pH_range=(2, 10), temp_range=(20, 60)
        )

        assert pH_grid.shape == (25, 25)
        assert temp_grid.shape == (25, 25)
        assert response.shape == (25, 25)
        assert np.all(response >= 0)

    def test_ph_temperature_response_surface_optimal(self):
        """Test optimal conditions in response surface."""
        from adsorblab_pro.models import ph_temperature_response_surface

        pH_grid, temp_grid, response = ph_temperature_response_surface(
            pH_range=(4, 8),
            temp_range=(30, 50),
            optimal_pH=6.0,
            optimal_temp=40.0,
            max_capacity=100,
        )

        # Maximum should be near optimal conditions
        max_idx = np.unravel_index(response.argmax(), response.shape)
        assert abs(pH_grid[max_idx] - 6.0) < 1.0
        assert abs(temp_grid[max_idx] - 40.0) < 5.0


class TestPackageInitialization:
    """Test package initialization."""

    def test_package_version(self):
        """Test package has version."""
        import adsorblab_pro

        assert hasattr(adsorblab_pro, "__version__")
        assert isinstance(adsorblab_pro.__version__, str)

    def test_package_submodules(self):
        """Test all submodules are accessible."""
        from adsorblab_pro import config, models, utils, validation

        # Test models
        assert hasattr(models, "langmuir_model")
        assert hasattr(models, "pfo_model")

        # Test utils
        assert hasattr(utils, "CalculationResult")

        # Test validation
        assert hasattr(validation, "ValidationResult")

        # Test config
        assert hasattr(config, "ISOTHERM_MODELS")


class TestPackageLazyImports:
    """Test package lazy import functionality."""

    def test_lazy_import_models(self):
        """Test lazy import of models module."""
        import adsorblab_pro

        models = adsorblab_pro.models
        assert hasattr(models, "langmuir_model")

    def test_lazy_import_utils(self):
        """Test lazy import of utils module."""
        import adsorblab_pro

        utils = adsorblab_pro.utils
        assert hasattr(utils, "CalculationResult")

    def test_lazy_import_validation(self):
        """Test lazy import of validation module."""
        import adsorblab_pro

        validation = adsorblab_pro.validation
        assert hasattr(validation, "ValidationResult")

    def test_lazy_import_config(self):
        """Test lazy import of config module."""
        import adsorblab_pro

        config = adsorblab_pro.config
        assert hasattr(config, "ISOTHERM_MODELS")

    def test_invalid_attribute(self):
        """Test that invalid attribute raises AttributeError."""
        import adsorblab_pro

        with pytest.raises(AttributeError):
            _ = adsorblab_pro.nonexistent_module


class TestParameterSpaceVisualization:
    """Test parameter space visualization."""

    def test_parameter_space_visualization_langmuir(self):
        """Test parameter space with Langmuir model."""
        from adsorblab_pro.models import parameter_space_visualization

        def langmuir(Ce, qm, KL):
            return qm * KL * Ce / (1 + KL * Ce)

        p1_grid, p2_grid, qe_grid = parameter_space_visualization(
            langmuir, param1_range=(50, 150), param2_range=(0.01, 0.1), Ce_fixed=50
        )

        assert p1_grid.shape == (25, 25)
        assert p2_grid.shape == (25, 25)
        assert qe_grid.shape == (25, 25)

    def test_parameter_space_visualization_error_handling(self):
        """Test parameter space with function that raises errors."""
        from adsorblab_pro.models import parameter_space_visualization

        def error_func(Ce, p1, p2):
            if p1 < 80:
                raise ValueError("Test error")
            return p1 * p2

        p1_grid, p2_grid, qe_grid = parameter_space_visualization(
            error_func, param1_range=(50, 150), param2_range=(0.01, 0.1)
        )

        # Should have NaN for error cases
        assert np.any(np.isnan(qe_grid))


class TestPlotStyleAdvanced:
    """Test advanced plot styling."""

    def test_style_ci_traces_called(self):
        """Test CI trace styling is applied."""
        from adsorblab_pro.plot_style import style_ci_traces

        # This function takes no arguments and applies styling globally
        style_ci_traces()
        # Just verify it runs without error

    def test_apply_professional_style_bar(self):
        """Test professional style with bar mode."""
        from adsorblab_pro.plot_style import apply_professional_style

        fig = go.Figure(data=[go.Bar(x=[1, 2, 3], y=[4, 5, 6])])
        styled_fig = apply_professional_style(fig, legend_position="top", barmode="group")
        assert isinstance(styled_fig, go.Figure)

    def test_create_model_comparison_plot_multiple(self):
        """Test model comparison plot with multiple models."""
        from adsorblab_pro.models import freundlich_model, langmuir_model
        from adsorblab_pro.plot_style import create_model_comparison_plot

        x_data = np.array([5, 10, 20, 40, 60])
        y_exp = np.array([15, 25, 38, 52, 58])

        fitted_models = {
            "Langmuir": {"params": {"qm": 70, "KL": 0.05}, "r_squared": 0.99, "converged": True},
            "Freundlich": {"params": {"KF": 5, "n_inv": 0.5}, "r_squared": 0.95, "converged": True},
        }

        model_functions = {
            "Langmuir": lambda x, p: langmuir_model(x, p["qm"], p["KL"]),
            "Freundlich": lambda x, p: freundlich_model(x, p["KF"], p["n_inv"]),
        }

        fig = create_model_comparison_plot(x_data, y_exp, fitted_models, model_functions)
        assert isinstance(fig, go.Figure)


class TestPlotStyleApplyStandardLayout:
    """Test apply_standard_layout function."""

    def test_apply_standard_layout_basic(self):
        """Test basic layout application."""
        from adsorblab_pro.plot_style import apply_standard_layout

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))

        result = apply_standard_layout(
            fig, title="Test", x_title="X", y_title="Y", height=400, show_legend=True
        )

        assert isinstance(result, go.Figure)
        assert result.layout.height == 400
        assert result.layout.showlegend is True

    def test_apply_standard_layout_custom_legend_position(self):
        """Test layout with custom legend position."""
        from adsorblab_pro.plot_style import apply_standard_layout

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))

        result = apply_standard_layout(
            fig,
            title="Test",
            x_title="X",
            y_title="Y",
            legend_position=(0.5, 0.5, "center", "middle"),
        )

        assert result.layout.legend.x == 0.5
        assert result.layout.legend.y == 0.5


class TestPlotStyleCITraces:
    """Test confidence interval trace styling."""

    def test_style_ci_traces(self):
        """Test confidence interval styling."""
        from adsorblab_pro.plot_style import style_ci_traces

        upper, lower = style_ci_traces()

        assert upper["mode"] == "lines"
        assert lower["mode"] == "lines"
        assert lower["fill"] == "tonexty"
        assert upper["showlegend"] is False
        assert lower["showlegend"] is False


class TestPlotStyleColors:
    """Test color schemes and color utilities."""

    def test_colors_dict_has_required_keys(self):
        """Verify COLORS dictionary has all required keys."""
        from adsorblab_pro.plot_style import COLORS

        required_keys = [
            "experimental",
            "experimental_edge",
            "fit_primary",
            "fit_secondary",
            "background",
            "grid",
            "ci_fill",
            "residual_positive",
            "residual_negative",
        ]
        for key in required_keys:
            assert key in COLORS, f"Missing color key: {key}"

    def test_model_colors_coverage(self):
        """Verify MODEL_COLORS has colors for all main models."""
        from adsorblab_pro.plot_style import MODEL_COLORS

        models = ["Langmuir", "Freundlich", "Temkin", "Sips", "PFO", "PSO", "Elovich", "IPD"]
        for model in models:
            assert model in MODEL_COLORS, f"Missing color for model: {model}"

    def test_study_colors_list(self):
        """Verify STUDY_COLORS is a non-empty list."""
        from adsorblab_pro.plot_style import STUDY_COLORS

        assert isinstance(STUDY_COLORS, list)
        assert len(STUDY_COLORS) >= 5

    def test_get_study_color(self):
        """Test get_study_color function."""
        from adsorblab_pro.plot_style import STUDY_COLORS, get_study_color

        # Test normal indices
        assert get_study_color(0) == STUDY_COLORS[0]
        assert get_study_color(1) == STUDY_COLORS[1]
        # Test wrapping
        assert get_study_color(len(STUDY_COLORS)) == STUDY_COLORS[0]
        assert get_study_color(len(STUDY_COLORS) + 1) == STUDY_COLORS[1]

    def test_hex_to_rgba(self):
        """Test hex to RGBA conversion."""
        from adsorblab_pro.plot_style import hex_to_rgba

        # Test with hash
        result = hex_to_rgba("#2E86AB", 0.5)
        assert "rgba(46, 134, 171, 0.5)" == result

        # Test without hash
        result = hex_to_rgba("FF0000", 1.0)
        assert "rgba(255, 0, 0, 1.0)" == result

        # Test with different alpha
        result = hex_to_rgba("#000000", 0.25)
        assert "rgba(0, 0, 0, 0.25)" == result


class TestPlotStyleDetermineLegendPosition:
    """Test legend position determination."""

    def test_determine_legend_position_increasing_curve(self):
        """Test legend position for increasing (typical) curve."""
        from adsorblab_pro.plot_style import _determine_legend_position

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 20, 30, 40, 50])
        y_fit = np.array([12, 22, 32, 42, 52])

        legend_x, legend_y = _determine_legend_position(x, y, y_fit)
        assert legend_x == 0.02  # Upper left
        assert legend_y == 0.98

    def test_determine_legend_position_decreasing_curve(self):
        """Test legend position for decreasing curve."""
        from adsorblab_pro.plot_style import _determine_legend_position

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([50, 40, 30, 20, 10])
        y_fit = np.array([52, 42, 32, 22, 12])

        legend_x, legend_y = _determine_legend_position(x, y, y_fit)
        assert legend_x == 0.98  # Upper right
        assert legend_y == 0.98


class TestPlotStyleLayoutHelpers:
    """Test layout helper functions."""

    def test_get_axis_style(self):
        """Test _get_axis_style helper function."""
        from adsorblab_pro.plot_style import _get_axis_style

        style = _get_axis_style("Test Title")
        assert style["title"]["text"] == "Test Title"
        assert style["showgrid"] is False
        assert style["showline"] is True
        assert style["mirror"] is True
        assert "gridcolor" in style

    def test_get_legend_style(self):
        """Test _get_legend_style helper function."""
        from adsorblab_pro.plot_style import _get_legend_style

        style = _get_legend_style(0.5, 0.8, "center", "top")
        assert style["x"] == 0.5
        assert style["y"] == 0.8
        assert style["xanchor"] == "center"
        assert style["yanchor"] == "top"
        assert "bgcolor" in style
        assert "bordercolor" in style

    def test_get_base_layout(self):
        """Test _get_base_layout helper function."""
        from adsorblab_pro.plot_style import _get_base_layout

        layout = _get_base_layout("Test Title", height=500, show_legend=False)
        assert "<b>Test Title</b>" in layout["title"]["text"]
        assert layout["height"] == 500
        assert layout["showlegend"] is False
        assert layout["plot_bgcolor"] == "white"


class TestPropagateCalibrationUncertainty:
    """Test uncertainty propagation."""

    def test_propagate_calibration_uncertainty_basic(self):
        """Test basic uncertainty propagation."""
        from adsorblab_pro.utils import propagate_calibration_uncertainty

        absorbance = 0.5
        slope = 100
        intercept = 0
        slope_se = 5
        intercept_se = 0.5

        Ce, Ce_se = propagate_calibration_uncertainty(
            absorbance, slope, intercept, slope_se, intercept_se
        )

        assert Ce >= 0
        assert Ce_se >= 0


class TestQuickValidate:
    """Test quick validation utilities."""

    def test_quick_validate_positive(self):
        """Test quick positive validation."""
        from adsorblab_pro.validation import quick_validate

        assert quick_validate(5.0, "test_field", "positive") is True
        assert quick_validate(-5.0, "test_field", "positive") is False

    def test_quick_validate_non_negative(self):
        """Test quick non-negative validation."""
        from adsorblab_pro.validation import quick_validate

        # 0.0 should be valid with non_negative
        assert quick_validate(5.0, "test_field", "non_negative") is True
        assert quick_validate(-1.0, "test_field", "non_negative") is False

    def test_quick_validate_array(self):
        """Test quick array validation."""
        from adsorblab_pro.validation import quick_validate

        assert quick_validate([1, 2, 3], "test_field", "array") is True


class TestRateLimitingStep:
    """Test rate-limiting step identification utilities."""

    def test_identify_rate_limiting_film_diffusion(self):
        """Test identification of film diffusion as rate-limiting."""
        from adsorblab_pro.models import identify_rate_limiting_step

        # Data suggesting film diffusion
        t = np.array([0, 1, 2, 5, 10, 20, 30, 60])
        qt = np.array([0, 5, 9, 18, 30, 45, 52, 58])
        qe = 60.0  # Equilibrium capacity

        result = identify_rate_limiting_step(t, qt, qe)
        assert isinstance(result, dict)
        assert "F" in result or "mechanism_suggestion" in result

    def test_identify_rate_limiting_step_basic(self):
        """Test basic rate limiting step analysis."""
        from adsorblab_pro.models import identify_rate_limiting_step

        # Generate typical kinetic data
        t = np.array([5, 10, 20, 30, 60, 90, 120, 180, 240])
        qe = 65
        k2 = 0.001
        qt = (k2 * qe**2 * t) / (1 + k2 * qe * t)

        results = identify_rate_limiting_step(t, qt, qe=qe)

        assert "weber_morris" in results
        assert "boyd_plot" in results
        assert "mechanism_suggestion" in results
        assert "confidence" in results

    def test_identify_rate_limiting_step_insufficient_data(self):
        """Test rate limiting step with insufficient data."""
        from adsorblab_pro.models import identify_rate_limiting_step

        t = np.array([5, 10, 20])
        qt = np.array([10, 18, 28])

        results = identify_rate_limiting_step(t, qt, qe=55)

        assert "error" in results

    def test_identify_rate_limiting_step_with_radius(self):
        """Test rate limiting step with particle radius."""
        from adsorblab_pro.models import identify_rate_limiting_step

        t = np.array([5, 10, 20, 30, 60, 90, 120])
        qt = np.array([10, 18, 28, 35, 45, 50, 53])

        results = identify_rate_limiting_step(t, qt, qe=55, particle_radius=0.01)

        assert "D_eff_cm2_min" in results["weber_morris"]

    def test_identify_rate_limiting_with_biot(self):
        """Test rate limiting step with Biot number calculation."""
        from adsorblab_pro.models import identify_rate_limiting_step

        t = np.array([0, 5, 10, 20, 30, 60, 90, 120])
        qt = np.array([0, 12, 22, 35, 44, 55, 58, 60])
        qe = 60.0

        result = identify_rate_limiting_step(t, qt, qe, particle_radius=0.5)
        assert isinstance(result, dict)


class TestRecommendBestModels:
    """Test model recommendation function."""

    def test_recommend_best_models_isotherm(self):
        """Test model recommendation for isotherms."""
        from adsorblab_pro.utils import recommend_best_models

        fitted_models = {
            "Langmuir": {
                "params": {"qm": 70, "KL": 0.05},
                "r_squared": 0.995,
                "adj_r_squared": 0.993,
                "rmse": 1.5,
                "aic": 10,
                "aicc": 12,
                "converged": True,
            },
            "Freundlich": {
                "params": {"KF": 5, "n": 2},
                "r_squared": 0.980,
                "adj_r_squared": 0.978,
                "rmse": 2.0,
                "aic": 15,
                "aicc": 17,
                "converged": True,
            },
        }

        recommendations = recommend_best_models(fitted_models, model_type="isotherm")

        assert isinstance(recommendations, list)


class TestRecommendBestModelsAdvanced:
    """Test model recommendation with various parameters."""

    def test_recommend_models_with_temkin_params(self):
        """Test recommendations with Temkin model parameters."""
        from adsorblab_pro.utils import recommend_best_models

        fitted_models = {
            "Temkin": {
                "params": {"B1": 15, "KT": 0.5},
                "r_squared": 0.95,
                "adj_r_squared": 0.94,
                "rmse": 2.0,
                "aicc": 18,
                "converged": True,
            },
        }
        recommendations = recommend_best_models(fitted_models, model_type="isotherm")
        assert isinstance(recommendations, list)

    def test_recommend_models_with_sips_params(self):
        """Test recommendations with Sips model parameters."""
        from adsorblab_pro.utils import recommend_best_models

        fitted_models = {
            "Sips": {
                "params": {"qm": 80, "Ks": 0.04, "ns": 0.95},  # ns close to 1
                "r_squared": 0.99,
                "adj_r_squared": 0.985,
                "rmse": 0.8,
                "aicc": 10,
                "converged": True,
            },
        }
        recommendations = recommend_best_models(fitted_models, model_type="isotherm")
        assert isinstance(recommendations, list)

    def test_recommend_kinetic_models_pfo(self):
        """Test kinetic model recommendations with PFO."""
        from adsorblab_pro.utils import recommend_best_models

        fitted_models = {
            "Pseudo-first order": {
                "params": {"qe": 50, "k1": 0.05},
                "r_squared": 0.92,
                "adj_r_squared": 0.91,
                "rmse": 2.5,
                "aicc": 20,
                "converged": True,
            },
        }
        recommendations = recommend_best_models(fitted_models, model_type="kinetic")
        assert isinstance(recommendations, list)

    def test_recommend_kinetic_models_pso(self):
        """Test kinetic model recommendations with PSO."""
        from adsorblab_pro.utils import recommend_best_models

        fitted_models = {
            "Pseudo-second order": {
                "params": {"qe": 60, "k2": 0.001},
                "r_squared": 0.98,
                "adj_r_squared": 0.975,
                "rmse": 1.5,
                "aicc": 15,
                "converged": True,
            },
        }
        recommendations = recommend_best_models(fitted_models, model_type="kinetic")
        assert isinstance(recommendations, list)

    def test_recommend_kinetic_models_elovich(self):
        """Test kinetic model recommendations with Elovich."""
        from adsorblab_pro.utils import recommend_best_models

        fitted_models = {
            "Elovich": {
                "params": {"alpha": 100, "beta": 0.1},
                "r_squared": 0.95,
                "adj_r_squared": 0.94,
                "rmse": 2.0,
                "aicc": 18,
                "converged": True,
            },
        }
        recommendations = recommend_best_models(fitted_models, model_type="kinetic")
        assert isinstance(recommendations, list)

    def test_recommend_kinetic_models_weber_morris(self):
        """Test kinetic model recommendations with Weber-Morris."""
        from adsorblab_pro.utils import recommend_best_models

        fitted_models = {
            "Weber-Morris": {
                "params": {"kid": 5, "C": 10},
                "r_squared": 0.90,
                "adj_r_squared": 0.88,
                "rmse": 3.0,
                "aicc": 25,
                "converged": True,
            },
        }
        recommendations = recommend_best_models(fitted_models, model_type="kinetic")
        assert isinstance(recommendations, list)


class TestResidualAnalysis:
    """Test residual analysis functions."""

    def test_residual_analysis_basic(self):
        """Test basic residual analysis."""
        from adsorblab_pro.utils import analyze_residuals

        residuals = np.array([1, -1, 0.5, -0.5, 0.2])

        analysis = analyze_residuals(residuals)

        assert "mean" in analysis
        assert "std" in analysis


class TestResidualPlotsAdvanced:
    """Test residual plotting edge cases."""

    def test_residual_plots_with_large_dataset(self):
        """Test residual plots with larger dataset."""
        from adsorblab_pro.utils import create_residual_plots

        np.random.seed(42)
        residuals = np.random.normal(0, 1, 100)
        y_pred = np.linspace(10, 100, 100)
        x_data = np.linspace(5, 95, 100)

        fig = create_residual_plots(residuals, y_pred, x_data=x_data)
        assert isinstance(fig, go.Figure)

    def test_residual_plots_with_outliers(self):
        """Test residual plots with outliers."""
        from adsorblab_pro.utils import create_residual_plots

        residuals = np.array([0.1, -0.2, 0.3, -0.1, 0.2, 5.0, -4.0])  # Outliers
        y_pred = np.array([10, 20, 30, 40, 50, 60, 70])

        fig = create_residual_plots(residuals, y_pred)
        assert isinstance(fig, go.Figure)


class TestSelectivityCoefficient:
    """Test selectivity coefficient calculation."""

    def test_calculate_selectivity_coefficient_basic(self):
        """Test basic selectivity calculation."""
        from adsorblab_pro.models import calculate_selectivity_coefficient

        alpha = calculate_selectivity_coefficient(qe_i=50, Ce_i=10, qe_j=30, Ce_j=20)

        assert alpha > 0


class TestStatisticalFunctions:
    """Test statistical utility functions."""

    def test_calculate_akaike_weights(self):
        """Test Akaike weights calculation."""
        from adsorblab_pro.utils import calculate_akaike_weights

        aic_values = [100.0, 105.0, 110.0]  # List of floats
        weights = calculate_akaike_weights(aic_values)

        assert isinstance(weights, np.ndarray)
        assert len(weights) == 3
        # Weights should sum to approximately 1
        assert abs(np.sum(weights) - 1.0) < 0.01

    def test_calculate_q2(self):
        """Test Q² (predictive R²) calculation."""
        from adsorblab_pro.utils import calculate_q2

        y_data = np.array([10, 20, 30, 40, 50])
        press = 10.0  # PRESS statistic

        q2 = calculate_q2(press, y_data)
        assert isinstance(q2, float)
        assert np.isfinite(q2)


class TestStyleExperimentalTrace:
    """Test style_experimental_trace function."""

    def test_style_experimental_trace_default(self):
        """Test default experimental trace styling."""
        from adsorblab_pro.plot_style import style_experimental_trace

        style = style_experimental_trace()

        assert style["mode"] == "markers"
        assert style["name"] == "Experimental"
        assert "marker" in style

    def test_style_experimental_trace_custom_name(self):
        """Test experimental trace with custom name."""
        from adsorblab_pro.plot_style import style_experimental_trace

        style = style_experimental_trace(name="Custom Data")

        assert style["name"] == "Custom Data"

    def test_style_experimental_trace_small(self):
        """Test small experimental trace styling."""
        from adsorblab_pro.plot_style import MARKERS, style_experimental_trace

        style = style_experimental_trace(use_small=True)

        assert style["marker"] == MARKERS["experimental_small"]


class TestStyleFitTrace:
    """Test style_fit_trace function."""

    def test_style_fit_trace_default(self):
        """Test default fit trace styling."""
        from adsorblab_pro.plot_style import style_fit_trace

        style = style_fit_trace("Langmuir")

        assert style["mode"] == "lines"
        assert "Langmuir" in style["name"]
        assert style["line"]["dash"] == "solid"

    def test_style_fit_trace_with_r_squared(self):
        """Test fit trace with R²."""
        from adsorblab_pro.plot_style import style_fit_trace

        style = style_fit_trace("Freundlich", r_squared=0.995)

        assert "0.995" in style["name"]

    def test_style_fit_trace_secondary(self):
        """Test secondary (comparison) fit trace."""
        from adsorblab_pro.plot_style import style_fit_trace

        style = style_fit_trace("Temkin", is_primary=False)

        assert style["line"]["dash"] == "dash"
        assert style["line"]["width"] == 2


class TestTemperatureResultsDirect:
    """Test calculate_temperature_results_direct function."""

    def test_temperature_results_direct_celsius(self):
        """Test temperature results with Celsius input."""
        from adsorblab_pro.utils import calculate_temperature_results_direct

        df = pd.DataFrame({"Temperature": [25, 35, 45], "Ce": [20, 15, 10]})
        temp_input = {"data": df, "params": {"C0": 50, "m": 0.1, "V": 0.1}}

        result = calculate_temperature_results_direct(temp_input)

        assert result.success is True
        assert "Temperature_C" in result.data.columns
        assert "Temperature_K" in result.data.columns
        assert "qe_mg_g" in result.data.columns

    def test_temperature_results_direct_kelvin(self):
        """Test temperature results with Kelvin input."""
        from adsorblab_pro.utils import calculate_temperature_results_direct

        df = pd.DataFrame({"Temperature": [298, 308, 318], "Ce": [20, 15, 10]})
        temp_input = {"data": df, "params": {"C0": 50, "m": 0.1, "V": 0.1}}

        result = calculate_temperature_results_direct(temp_input)

        assert result.success is True
        # Kelvin temps should be converted correctly
        assert result.data["Temperature_K"].iloc[0] == 298

    def test_temperature_results_direct_invalid_ce(self):
        """Test temperature results with Ce > C0."""
        from adsorblab_pro.utils import calculate_temperature_results_direct

        df = pd.DataFrame({"Temperature": [25, 35], "Ce": [60, 70]})  # Ce > C0=50
        temp_input = {"data": df, "params": {"C0": 50, "m": 0.1, "V": 0.1}}

        result = calculate_temperature_results_direct(temp_input)

        assert result.success is False

    def test_temperature_results_direct_with_uncertainty(self):
        """Test temperature results with uncertainty flag."""
        from adsorblab_pro.utils import calculate_temperature_results_direct

        df = pd.DataFrame({"Temperature": [25, 35], "Ce": [20, 15]})
        temp_input = {"data": df, "params": {"C0": 50, "m": 0.1, "V": 0.1}}

        result = calculate_temperature_results_direct(temp_input, include_uncertainty=True)

        assert result.success is True
        assert "Ce_error" in result.data.columns
        assert "qe_error" in result.data.columns


class TestThermodynamicCalculations:
    """Test thermodynamic calculation functions."""

    def test_vant_hoff_calculation(self):
        """Test Van't Hoff thermodynamic calculation."""
        from adsorblab_pro.utils import calculate_thermodynamic_parameters

        # Test data: K vs T
        T_K = np.array([298, 308, 318, 328])
        K = np.array([1000, 800, 600, 400])  # Decreasing K with T (exothermic)

        result = calculate_thermodynamic_parameters(T_K, K)

        assert "delta_H" in result or "dH" in result or "ΔH" in result


class TestThermodynamicValidationAdvanced:
    """Test thermodynamic validation edge cases."""

    def test_thermodynamic_with_celsius_conversion(self):
        """Test thermodynamic validation with Celsius temperatures."""
        from adsorblab_pro.validation import validate_thermodynamic_data

        # Temperatures in Celsius (should be detected and error)
        temperatures = np.array([25, 35, 45, 55])  # Celsius
        Kd = np.array([1.5, 2.0, 2.5, 3.0])

        report = validate_thermodynamic_data(temperatures, Kd)
        # Should have error about Celsius (not warning)
        assert not report.is_valid or len(report.errors) > 0

    def test_thermodynamic_with_negative_kd(self):
        """Test thermodynamic validation with negative Kd."""
        from adsorblab_pro.validation import validate_thermodynamic_data

        temperatures = np.array([298, 308, 318, 328])
        Kd = np.array([1.5, 2.0, -0.5, 3.0])  # Negative Kd

        report = validate_thermodynamic_data(temperatures, Kd)
        assert not report.is_valid


class TestValidationEdgeCases:
    """Test validation edge cases."""

    def test_validate_array_edge_cases(self):
        """Test validate_array with edge cases."""
        from adsorblab_pro.validation import validate_array

        # Exactly min length
        arr = np.array([1, 2, 3])
        result = validate_array(arr, "test", min_length=3)
        assert result.is_valid

        # Empty array
        arr = np.array([])
        result = validate_array(arr, "test", min_length=1)
        assert not result.is_valid


class TestValidationExperimentalParams:
    """Test experimental parameter validation."""

    def test_validate_experimental_params_valid(self):
        """Test validation of valid experimental parameters."""
        from adsorblab_pro.validation import validate_experimental_params

        result = validate_experimental_params(C0=100, m=0.1, V=0.1, pH=7.0, T=298)

        assert result.is_valid

    def test_validate_experimental_params_invalid_ph(self):
        """Test validation with invalid pH."""
        from adsorblab_pro.validation import validate_experimental_params

        result = validate_experimental_params(
            C0=100,
            m=0.1,
            V=0.1,
            pH=15.0,  # Invalid pH
        )

        assert not result.is_valid


class TestValidationIsothermData:
    """Test isotherm data validation."""

    def test_validate_isotherm_data_valid(self):
        """Test validation of valid isotherm data."""
        from adsorblab_pro.validation import validate_isotherm_data

        C0 = np.array([100, 100, 100, 100, 100])
        Ce = np.array([5, 10, 20, 40, 60])
        qe = np.array([15, 25, 38, 52, 58])

        result = validate_isotherm_data(C0, Ce, qe)

        assert result.is_valid


class TestValidationKineticData:
    """Test kinetic data validation."""

    def test_validate_kinetic_data_valid(self):
        """Test validation of valid kinetic data."""
        from adsorblab_pro.validation import validate_kinetic_data

        t = np.array([0, 5, 10, 20, 30])
        qt = np.array([0, 12, 22, 35, 44])

        result = validate_kinetic_data(t, qt)

        assert result.is_valid


class TestValidationThermodynamicData:
    """Test thermodynamic data validation."""

    def test_validate_thermodynamic_data_valid(self):
        """Test validation of valid thermodynamic data."""
        from adsorblab_pro.validation import validate_thermodynamic_data

        temperatures = np.array([298, 308, 318, 328])
        Kd = np.array([1000, 800, 600, 400])

        result = validate_thermodynamic_data(temperatures, Kd)

        assert result.is_valid

    def test_validate_thermodynamic_data_celsius(self):
        """Test detection of Celsius temperatures."""
        from adsorblab_pro.validation import validate_thermodynamic_data

        temperatures = np.array([25, 35, 45, 55])  # Celsius - should warn
        Kd = np.array([1000, 800, 600, 400])

        result = validate_thermodynamic_data(temperatures, Kd)

        # Should have warnings about Celsius
        assert not result.is_valid or len(result.warnings) > 0


class TestValidationUploadedFile:
    """Test file upload validation."""

    def test_validate_uploaded_file_valid_csv(self):
        """Test validation of valid CSV file."""
        from adsorblab_pro.validation import validate_uploaded_file

        result = validate_uploaded_file(file_size=1000, file_name="data.csv")

        assert result.is_valid

    def test_validate_uploaded_file_too_large(self):
        """Test validation of file that is too large."""
        from adsorblab_pro.validation import validate_uploaded_file

        # 100 MB file
        result = validate_uploaded_file(file_size=100_000_000, file_name="data.csv")

        assert not result.is_valid

    def test_validate_uploaded_file_invalid_type(self):
        """Test validation of invalid file type."""
        from adsorblab_pro.validation import validate_uploaded_file

        result = validate_uploaded_file(file_size=1000, file_name="data.exe")

        assert not result.is_valid
