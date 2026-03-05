"""
Extended tests for adsorblab_pro/utils.py - targeting 75%+ coverage.

Covers:
- CalculationResult class
- standardize_dataframe_columns
- validate_data_editor (comma decimals, missing cols, too few rows)
- calculate_press (LOO cross-validation)
- calculate_temperature_results / calculate_temperature_results_direct
- check_mechanism_consistency (full paths)
- detect_common_errors (comprehensive scenarios)
- analyze_residuals (constant, autocorrelation, heteroscedasticity)
- assess_data_quality (edge cases)
- _score_isotherm_params / _score_kinetic_params
- recommend_best_models (full flow)
- determine_adsorption_mechanism (comprehensive)
- calculate_arrhenius_parameters (edge cases)
- propagate_calibration_uncertainty (edge cases)
- detect_replicates
- validate_study_name
- calculate_calibration_stats
- convert_df_to_csv / convert_df_to_excel
- calculate_activity_coefficient_davies
"""

import numpy as np
import pandas as pd
import pytest

from adsorblab_pro.utils import (
    CalculationResult,
    analyze_residuals,
    assess_data_quality,
    bootstrap_confidence_intervals,
    calculate_activity_coefficient_davies,
    calculate_arrhenius_parameters,
    calculate_calibration_stats,
    calculate_error_metrics,
    calculate_press,
    calculate_q2,
    calculate_separation_factor,
    calculate_temperature_results_direct,
    calculate_thermodynamic_parameters,
    check_mechanism_consistency,
    convert_df_to_csv,
    convert_df_to_excel,
    detect_common_errors,
    detect_replicates,
    determine_adsorption_mechanism,
    interpret_separation_factor,
    interpret_thermodynamics,
    propagate_calibration_uncertainty,
    recommend_best_models,
    standardize_column_name,
    standardize_dataframe_columns,
    validate_data_editor,
    validate_study_name,
)
from adsorblab_pro.models import langmuir_model, pso_model


# =============================================================================
# CALCULATION RESULT CLASS
# =============================================================================
class TestCalculationResultExtended:
    def test_success_with_dict(self):
        r = CalculationResult(success=True, data={"a": 1})
        assert r.success is True
        assert r.data == {"a": 1}

    def test_failure(self):
        r = CalculationResult(success=False, data=None, error="error")
        assert r.success is False
        assert r.data is None

    def test_success_with_dataframe(self):
        df = pd.DataFrame({"x": [1, 2]})
        r = CalculationResult(success=True, data=df)
        assert r.success is True
        assert isinstance(r.data, pd.DataFrame)


# =============================================================================
# COLUMN STANDARDIZATION
# =============================================================================
class TestStandardizeDataframeColumns:
    def test_basic_standardization(self):
        df = pd.DataFrame({"C_initial": [10], "C_equilibrium": [5]})
        result = standardize_dataframe_columns(df)
        # Should standardize to recognized names
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 2

    def test_fuzzy_matching(self):
        name = standardize_column_name("concentraton")  # typo
        assert isinstance(name, str)

    def test_unknown_column_unchanged(self):
        name = standardize_column_name("xyzzy_unknown_column_123")
        assert name == "xyzzy_unknown_column_123"


# =============================================================================
# VALIDATE DATA EDITOR
# =============================================================================
class TestValidateDataEditorExtended:
    def test_none_input(self):
        assert validate_data_editor(None, ["Ce", "qe"]) is None

    def test_empty_dataframe(self):
        assert validate_data_editor(pd.DataFrame(), ["Ce", "qe"]) is None

    def test_missing_columns(self):
        df = pd.DataFrame({"Ce": [1, 2, 3]})
        assert validate_data_editor(df, ["Ce", "qe"]) is None

    def test_comma_decimal_separator(self):
        df = pd.DataFrame({"Ce": ["1,5", "2,5", "3,5"], "qe": ["10,0", "20,0", "30,0"]})
        result = validate_data_editor(df, ["Ce", "qe"])
        assert result is not None
        assert len(result) == 3
        assert result["Ce"].iloc[0] == pytest.approx(1.5)

    def test_too_few_rows_after_cleanup(self):
        df = pd.DataFrame({"Ce": [1.0, np.nan], "qe": [10.0, np.nan]})
        assert validate_data_editor(df, ["Ce", "qe"]) is None

    def test_valid_numeric_data(self):
        df = pd.DataFrame({"Ce": [1.0, 5.0, 10.0, 20.0], "qe": [5.0, 15.0, 22.0, 30.0]})
        result = validate_data_editor(df, ["Ce", "qe"])
        assert result is not None
        assert len(result) == 4


# =============================================================================
# CALCULATE PRESS
# =============================================================================
class TestCalculatePressExtended:
    def test_langmuir_press(self):
        Ce = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0])
        qe = langmuir_model(Ce, 100.0, 0.05)
        params = np.array([100.0, 0.05])
        press = calculate_press(langmuir_model, Ce, qe, params)
        assert press >= 0
        # Perfect data → very low PRESS
        assert press < 1.0

    def test_with_bounds(self):
        Ce = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0])
        qe = langmuir_model(Ce, 100.0, 0.05) + np.random.RandomState(42).normal(0, 1, 7)
        params = np.array([100.0, 0.05])
        press = calculate_press(
            langmuir_model,
            Ce,
            qe,
            params,
            bounds=((0, 0), (500, 1)),
        )
        assert press > 0


# =============================================================================
# TEMPERATURE RESULTS DIRECT
# =============================================================================
class TestTemperatureResultsDirectExtended:
    def _make_input(self, temperature, temp_unit, C0, Ce, V, m):
        """Helper to build temp_input dict."""
        import pandas as pd

        T_K = temperature + 273.15 if temp_unit == "Celsius" else temperature
        df = pd.DataFrame({"Temperature": [T_K], "Ce": [Ce]})
        return {"data": df, "params": {"C0": C0, "m": m, "V": V}}

    def test_celsius_to_kelvin_conversion(self):
        temp_input = self._make_input(25.0, "Celsius", 100.0, 20.0, 0.05, 0.1)
        result = calculate_temperature_results_direct(temp_input)
        assert result is not None
        assert result.success is True
        assert result.data["Temperature_K"].iloc[0] == pytest.approx(298.15)

    def test_kelvin_input(self):
        temp_input = self._make_input(298.15, "Kelvin", 100.0, 20.0, 0.05, 0.1)
        result = calculate_temperature_results_direct(temp_input)
        assert result.success is True

    def test_invalid_ce_greater_than_c0(self):
        temp_input = self._make_input(298.15, "Kelvin", 100.0, 150.0, 0.05, 0.1)
        result = calculate_temperature_results_direct(temp_input)
        # Ce > C0 should handle gracefully (may succeed with negative qe)
        assert result is not None

    def test_with_uncertainty(self):
        temp_input = self._make_input(298.15, "Kelvin", 100.0, 20.0, 0.05, 0.1)
        result = calculate_temperature_results_direct(temp_input, include_uncertainty=True)
        assert result.success is True
        if "Ce_error" in result.data.columns:
            assert (result.data["Ce_error"] >= 0).all()


# =============================================================================
# CHECK MECHANISM CONSISTENCY
# =============================================================================
class TestCheckMechanismConsistencyExtended:
    def test_empty_state(self):
        result = check_mechanism_consistency({})
        assert result["status"] == "consistent"
        assert result["color"] == "green"

    def test_pso_freundlich_mismatch(self):
        state = {
            "isotherm_models_fitted": {
                "Langmuir": {"converged": True, "r_squared": 0.90, "adj_r_squared": 0.89},
                "Freundlich": {"converged": True, "r_squared": 0.98, "adj_r_squared": 0.97},
            },
            "kinetic_models_fitted": {
                "PSO": {"converged": True, "r_squared": 0.99, "adj_r_squared": 0.98},
                "PFO": {"converged": True, "r_squared": 0.85, "adj_r_squared": 0.84},
            },
        }
        result = check_mechanism_consistency(state)
        assert result["status"] in ("minor_issues", "consistent")

    def test_temperature_dh_conflict_endothermic_decreasing(self):
        state = {
            "isotherm_models_fitted": {},
            "kinetic_models_fitted": {},
            "thermo_params": {"delta_H": 30.0},
            "temperature_effect": "decreases",
        }
        result = check_mechanism_consistency(state)
        assert result["status"] == "conflicts"
        assert result["conflicts"] >= 1

    def test_temperature_dh_conflict_exothermic_increasing(self):
        state = {
            "isotherm_models_fitted": {},
            "kinetic_models_fitted": {},
            "thermo_params": {"delta_H": -30.0},
            "temperature_effect": "increases",
        }
        result = check_mechanism_consistency(state)
        assert result["status"] == "conflicts"

    def test_consistent_temperature_effect(self):
        state = {
            "isotherm_models_fitted": {},
            "kinetic_models_fitted": {},
            "thermo_params": {"delta_H": -30.0},
            "temperature_effect": "decreases",
        }
        result = check_mechanism_consistency(state)
        assert result["status"] == "consistent"

    def test_rl_freundlich_inconsistency(self):
        state = {
            "isotherm_models_fitted": {
                "Langmuir": {"converged": True, "r_squared": 0.95, "RL": 0.3},
                "Freundlich": {
                    "converged": True,
                    "r_squared": 0.96,
                    "params": {"n_inv": 1.5},
                },
            },
            "kinetic_models_fitted": {},
        }
        result = check_mechanism_consistency(state)
        assert any(c["name"] == "RL vs Freundlich Consistency" for c in result["checks"])

    def test_high_r2_without_ci(self):
        state = {
            "isotherm_models_fitted": {
                "Langmuir": {"converged": True, "r_squared": 0.995, "ci_95": {}},
            },
            "kinetic_models_fitted": {},
        }
        result = check_mechanism_consistency(state)
        assert result["minor_issues"] >= 1


# =============================================================================
# DETECT COMMON ERRORS
# =============================================================================
class TestDetectCommonErrorsExtended:
    def test_empty_state(self):
        errors = detect_common_errors({})
        assert isinstance(errors, list)

    def test_delta_g_out_of_range(self):
        state = {
            "thermo_params": {"delta_G": [-70.0, -65.0]},
        }
        errors = detect_common_errors(state)
        assert any(e["type"] == "Thermodynamic" for e in errors)

    def test_insufficient_data_points(self):
        state = {
            "isotherm_results": [1, 2, 3, 4],  # 4 points
            "isotherm_models_fitted": {
                "Sips": {"converged": True, "num_params": 3},
            },
        }
        errors = detect_common_errors(state)
        assert any(
            e["type"] == "Statistical" and "insufficient" in e["message"].lower() for e in errors
        )

    def test_high_r2_no_ci(self):
        state = {
            "isotherm_models_fitted": {
                "Langmuir": {"converged": True, "r_squared": 0.999, "ci_95": {}},
            },
        }
        errors = detect_common_errors(state)
        assert any(e["type"] == "Reporting" for e in errors)

    def test_heteroscedasticity_detection(self):
        # Create data with heteroscedastic residuals
        y_pred = np.array([1.0, 5.0, 10.0, 50.0, 100.0, 200.0])
        residuals = y_pred * 0.1  # Proportional residuals → heteroscedastic
        state = {
            "isotherm_models_fitted": {
                "Langmuir": {
                    "converged": True,
                    "residuals": residuals,
                    "y_pred": y_pred,
                },
            },
        }
        errors = detect_common_errors(state)
        heteroscedastic = [e for e in errors if "heteroscedasticity" in e["message"].lower()]
        assert len(heteroscedastic) >= 1

    def test_linear_vs_nonlinear_comparison(self):
        state = {
            "isotherm_linear_results": {
                "Langmuir": {"r_squared": 0.85},
            },
            "isotherm_models_fitted": {
                "Langmuir": {"r_squared": 0.95},
            },
        }
        errors = detect_common_errors(state)
        assert any(e["type"] == "Methodology" for e in errors)


# =============================================================================
# ANALYZE RESIDUALS
# =============================================================================
class TestAnalyzeResidualsExtended:
    def test_basic_residuals(self):
        residuals = np.random.RandomState(42).normal(0, 1, 50)
        result = analyze_residuals(residuals)
        assert "n" in result
        assert result["n"] == 50
        assert "skewness" in result
        assert "kurtosis" in result
        assert "normality_test" in result

    def test_constant_residuals(self):
        residuals = np.ones(10) * 0.5
        result = analyze_residuals(residuals)
        assert result["is_constant"] is True
        assert result["skewness"] == 0.0

    def test_few_residuals(self):
        residuals = np.array([0.1, -0.2])
        result = analyze_residuals(residuals)
        assert result["n"] == 2
        assert np.isnan(result["skewness"])

    def test_with_y_pred_heteroscedasticity(self):
        y_pred = np.linspace(1, 100, 50)
        residuals = y_pred * np.random.RandomState(42).normal(0, 0.1, 50)
        result = analyze_residuals(residuals, y_pred=y_pred)
        assert "heteroscedasticity_corr" in result

    def test_durbin_watson(self):
        residuals = np.random.RandomState(42).normal(0, 1, 20)
        result = analyze_residuals(residuals)
        assert "durbin_watson" in result
        assert 0 <= result["durbin_watson"] <= 4

    def test_empty_residuals(self):
        result = analyze_residuals(np.array([]))
        assert result["n"] == 0

    def test_large_sample_normaltest(self):
        residuals = np.random.RandomState(42).normal(0, 1, 6000)
        result = analyze_residuals(residuals)
        assert result["normality_test"] == "D'Agostino-Pearson"

    def test_with_nan_values(self):
        residuals = np.array([1.0, np.nan, -0.5, 2.0, np.nan, -1.0, 0.5, 1.5])
        result = analyze_residuals(residuals)
        assert result["n"] == 6  # NaN filtered out


# =============================================================================
# DETERMINE ADSORPTION MECHANISM
# =============================================================================
class TestDetermineAdsorptionMechanismExtended:
    def test_physical_adsorption(self):
        result = determine_adsorption_mechanism(delta_H=-10.0)
        assert "Physical" in result["mechanism"]

    def test_chemical_adsorption(self):
        result = determine_adsorption_mechanism(delta_H=-100.0)
        assert "Chemical" in result["mechanism"]

    def test_weak_chemisorption(self):
        result = determine_adsorption_mechanism(delta_H=-30.0)
        # 20-40 range → weak chemical or mixed
        assert result["mechanism"] in (
            "Mixed mechanism",
            "Physical adsorption",
            "Chemical adsorption",
        )

    def test_hydrogen_bonding(self):
        result = determine_adsorption_mechanism(delta_H=-60.0)
        assert "Chemical" in result["mechanism"] or "Mixed" in result["mechanism"]

    def test_with_delta_g_physical(self):
        result = determine_adsorption_mechanism(
            delta_H=-10.0,
            delta_G=[-5.0, -8.0, -10.0],
        )
        assert "Physical" in result["mechanism"]
        assert "ΔG° (kJ/mol)" in result["indicators"]

    def test_with_delta_g_chemical(self):
        result = determine_adsorption_mechanism(
            delta_H=-100.0,
            delta_G=[-50.0, -55.0],
        )
        assert "Chemical" in result["mechanism"]

    def test_with_delta_g_strong_physical(self):
        result = determine_adsorption_mechanism(
            delta_H=-10.0,
            delta_G=[-25.0, -30.0],
        )
        assert "indicators" in result
        g_indicator = result["indicators"].get("ΔG° (kJ/mol)", {})
        assert g_indicator.get("classification") == "Strong Physical"

    def test_with_delta_g_non_spontaneous(self):
        result = determine_adsorption_mechanism(delta_H=-10.0, delta_G=[5.0])
        g_ind = result["indicators"].get("ΔG° (kJ/mol)", {})
        assert g_ind.get("classification") == "Non-spontaneous"

    def test_with_freundlich_n_favorable(self):
        result = determine_adsorption_mechanism(
            delta_H=-10.0,
            n_freundlich=0.7,
        )
        assert "1/n (Freundlich)" in result["indicators"]

    def test_with_freundlich_n_highly_favorable(self):
        result = determine_adsorption_mechanism(
            delta_H=-100.0,
            n_freundlich=0.3,
        )
        n_ind = result["indicators"]["1/n (Freundlich)"]
        assert n_ind["classification"] == "Highly favorable"

    def test_with_freundlich_n_unfavorable(self):
        result = determine_adsorption_mechanism(
            delta_H=-10.0,
            n_freundlich=1.5,
        )
        n_ind = result["indicators"]["1/n (Freundlich)"]
        assert n_ind["classification"] == "Unfavorable"

    def test_with_rl_favorable(self):
        result = determine_adsorption_mechanism(delta_H=-10.0, RL=0.5)
        assert "RL (Langmuir)" in result["indicators"]

    def test_with_rl_highly_favorable(self):
        result = determine_adsorption_mechanism(delta_H=-100.0, RL=0.05)
        rl_ind = result["indicators"]["RL (Langmuir)"]
        assert rl_ind["classification"] == "Highly favorable"

    def test_all_indicators(self):
        result = determine_adsorption_mechanism(
            delta_H=-15.0,
            delta_G=[-10.0, -12.0],
            n_freundlich=0.6,
            RL=0.3,
        )
        assert result["confidence"] > 0
        assert len(result["evidence"]) >= 4
        assert len(result["indicators"]) == 4


# =============================================================================
# ARRHENIUS PARAMETERS
# =============================================================================
class TestArrheniusParametersExtended:
    def test_basic_calculation(self):
        T_K = np.array([293.15, 303.15, 313.15, 323.15])
        k = np.array([0.01, 0.02, 0.04, 0.08])
        result = calculate_arrhenius_parameters(T_K, k)
        assert result["success"] is True
        assert result["Ea"] > 0

    def test_insufficient_data(self):
        result = calculate_arrhenius_parameters(np.array([300.0]), np.array([0.01]))
        assert result["success"] is False

    def test_negative_k_filtered(self):
        T_K = np.array([293.15, 303.15, 313.15, 323.15])
        k = np.array([-0.01, 0.02, 0.04, 0.08])
        result = calculate_arrhenius_parameters(T_K, k)
        assert result["success"] is True
        assert result["n_points"] == 3

    def test_diffusion_controlled(self):
        T_K = np.array([293.15, 303.15, 313.15])
        # Very small Ea → diffusion controlled
        k = np.array([0.01, 0.0105, 0.011])
        result = calculate_arrhenius_parameters(T_K, k)
        if result["success"]:
            assert "interpretation" in result

    def test_all_invalid_k(self):
        T_K = np.array([293.15, 303.15])
        k = np.array([-1.0, -2.0])
        result = calculate_arrhenius_parameters(T_K, k)
        assert result["success"] is False


# =============================================================================
# PROPAGATE CALIBRATION UNCERTAINTY
# =============================================================================
class TestPropagateCalibrationUncertaintyExtended:
    def test_basic_propagation(self):
        Ce, Ce_se = propagate_calibration_uncertainty(
            absorbance=0.5,
            slope=0.01,
            intercept=0.02,
            slope_se=0.0005,
            intercept_se=0.003,
        )
        assert Ce > 0
        assert Ce_se > 0
        assert np.isfinite(Ce_se)

    def test_zero_slope(self):
        Ce, Ce_se = propagate_calibration_uncertainty(
            absorbance=0.5,
            slope=0.0,
            intercept=0.02,
            slope_se=0.0005,
            intercept_se=0.003,
        )
        assert Ce == 0.0
        assert Ce_se == np.inf

    def test_with_covariance(self):
        Ce, Ce_se = propagate_calibration_uncertainty(
            absorbance=0.5,
            slope=0.01,
            intercept=0.02,
            slope_se=0.0005,
            intercept_se=0.003,
            cov_slope_intercept=0.0001,
        )
        assert np.isfinite(Ce_se)

    def test_negative_absorbance(self):
        Ce, Ce_se = propagate_calibration_uncertainty(
            absorbance=-0.01,
            slope=0.01,
            intercept=0.02,
            slope_se=0.0005,
            intercept_se=0.003,
        )
        assert Ce == 0.0  # max(0, Ce) when Ce is negative


# =============================================================================
# DETECT REPLICATES
# =============================================================================
class TestDetectReplicatesExtended:
    def test_with_replicates(self):
        df = pd.DataFrame(
            {
                "Ce": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
                "qe": [25.0, 26.0, 24.5, 35.0, 36.0, 34.5],
            }
        )
        result = detect_replicates(df, x_col="Ce")
        assert isinstance(result, pd.DataFrame)

    def test_no_replicates(self):
        df = pd.DataFrame(
            {
                "Ce": [10.0, 20.0, 30.0, 40.0],
                "qe": [25.0, 35.0, 42.0, 48.0],
            }
        )
        result = detect_replicates(df, x_col="Ce")
        assert isinstance(result, pd.DataFrame)

    def test_custom_tolerance(self):
        df = pd.DataFrame(
            {
                "Ce": [10.0, 10.05, 10.1, 20.0],
                "qe": [25.0, 25.5, 24.5, 35.0],
            }
        )
        result = detect_replicates(df, x_col="Ce", tolerance=0.02)
        assert isinstance(result, pd.DataFrame)


# =============================================================================
# VALIDATE STUDY NAME
# =============================================================================
class TestValidateStudyNameExtended:
    def test_empty_name(self):
        valid, msg = validate_study_name("")
        assert valid is False
        assert "enter" in msg.lower()

    def test_whitespace_only(self):
        valid, msg = validate_study_name("   ")
        assert valid is False

    def test_valid_name_no_existing(self):
        valid, msg = validate_study_name("My Study")
        assert valid is True
        assert msg == ""

    def test_duplicate_name(self):
        existing = {"Study A": {}, "Study B": {}}
        valid, msg = validate_study_name("Study A", existing_studies=existing)
        assert valid is False
        assert "exists" in msg.lower()

    def test_unique_name_with_existing(self):
        existing = {"Study A": {}}
        valid, msg = validate_study_name("Study B", existing_studies=existing)
        assert valid is True


# =============================================================================
# CALCULATE CALIBRATION STATS
# =============================================================================
class TestCalculateCalibrationStatsExtended:
    def test_basic_linear(self):
        conc = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        absorb = 0.02 * conc + 0.01
        result = calculate_calibration_stats(conc, absorb)
        assert result["r_squared"] > 0.999
        assert result["slope"] == pytest.approx(0.02)
        assert result["intercept"] == pytest.approx(0.01)
        assert "ci_slope" in result
        assert "ci_intercept" in result

    def test_with_noise(self):
        conc = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        absorb = 0.02 * conc + 0.01 + np.random.RandomState(42).normal(0, 0.005, 6)
        result = calculate_calibration_stats(conc, absorb)
        assert result["r_squared"] > 0.95
        assert result["se_slope"] > 0
        assert result["se_intercept"] > 0

    def test_two_points(self):
        conc = np.array([0.0, 50.0])
        absorb = np.array([0.01, 1.01])
        result = calculate_calibration_stats(conc, absorb)
        assert result["n"] == 2

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            calculate_calibration_stats(np.array([1.0]), np.array([0.5]))

    def test_custom_confidence(self):
        conc = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        absorb = 0.02 * conc + 0.01
        r99 = calculate_calibration_stats(conc, absorb, confidence_level=0.99)
        r95 = calculate_calibration_stats(conc, absorb, confidence_level=0.95)
        # 99% CI should be wider
        ci99_width = r99["ci_slope"][1] - r99["ci_slope"][0]
        ci95_width = r95["ci_slope"][1] - r95["ci_slope"][0]
        assert ci99_width >= ci95_width


# =============================================================================
# CONVERT DF FUNCTIONS
# =============================================================================
class TestConvertDfFunctionsExtended:
    def test_csv_output(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        csv_bytes = convert_df_to_csv(df)
        assert isinstance(csv_bytes, bytes)
        assert b"x" in csv_bytes

    def test_excel_output(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        xlsx_bytes = convert_df_to_excel(df)
        assert isinstance(xlsx_bytes, bytes)
        assert len(xlsx_bytes) > 0

    def test_excel_custom_sheet_name(self):
        df = pd.DataFrame({"a": [1]})
        xlsx_bytes = convert_df_to_excel(df, sheet_name="Results")
        assert isinstance(xlsx_bytes, bytes)


# =============================================================================
# ACTIVITY COEFFICIENT
# =============================================================================
class TestActivityCoefficientExtended:
    def test_zero_ionic_strength(self):
        gamma = calculate_activity_coefficient_davies(0.0)
        assert gamma == pytest.approx(1.0, abs=0.01)

    def test_positive_ionic_strength(self):
        gamma = calculate_activity_coefficient_davies(0.1)
        assert 0 < gamma < 1

    def test_high_charge(self):
        gamma1 = calculate_activity_coefficient_davies(0.1, charge=1)
        gamma2 = calculate_activity_coefficient_davies(0.1, charge=2)
        assert gamma2 < gamma1  # Higher charge → lower activity coefficient


# =============================================================================
# ASSESS DATA QUALITY
# =============================================================================
class TestAssessDataQualityExtended:
    def test_isotherm_quality_good(self):
        df = pd.DataFrame(
            {
                "Ce": np.linspace(1, 100, 15),
                "qe": np.linspace(5, 45, 15),
            }
        )
        result = assess_data_quality(df, data_type="isotherm")
        assert "quality_score" in result
        assert 0 <= result["quality_score"] <= 100

    def test_kinetic_quality(self):
        df = pd.DataFrame(
            {
                "t": np.array([0, 5, 10, 20, 30, 60, 90, 120, 180, 240]),
                "qt": pso_model(np.array([0, 5, 10, 20, 30, 60, 90, 120, 180, 240]), 50, 0.01),
            }
        )
        result = assess_data_quality(df, data_type="kinetic")
        assert "quality_score" in result

    def test_few_data_points(self):
        df = pd.DataFrame({"Ce": [1, 5, 10], "qe": [5, 15, 22]})
        result = assess_data_quality(df, data_type="isotherm")
        assert result["quality_score"] <= 80


# =============================================================================
# RECOMMEND BEST MODELS
# =============================================================================
class TestRecommendBestModelsExtended:
    def test_isotherm_models(self):
        results = {
            "Langmuir": {
                "converged": True,
                "r_squared": 0.98,
                "adj_r_squared": 0.97,
                "rmse": 1.5,
                "aic": 20.0,
                "aicc": 22.0,
                "bic": 21.0,
                "num_params": 2,
                "params": {"qm": 100.0, "KL": 0.05},
            },
            "Freundlich": {
                "converged": True,
                "r_squared": 0.95,
                "adj_r_squared": 0.94,
                "rmse": 2.5,
                "aic": 25.0,
                "aicc": 27.0,
                "bic": 26.0,
                "num_params": 2,
                "params": {"KF": 5.0, "n_inv": 0.5},
            },
        }
        rec = recommend_best_models(results, model_type="isotherm")
        assert isinstance(rec, list)
        assert len(rec) == 2
        assert rec[0]["model"] == "Langmuir"

    def test_kinetic_models(self):
        results = {
            "PFO": {
                "converged": True,
                "r_squared": 0.92,
                "adj_r_squared": 0.91,
                "rmse": 3.0,
                "aic": 30.0,
                "aicc": 32.0,
                "bic": 31.0,
                "num_params": 2,
                "params": {"qe": 45.0, "k1": 0.05},
            },
            "PSO": {
                "converged": True,
                "r_squared": 0.98,
                "adj_r_squared": 0.97,
                "rmse": 1.2,
                "aic": 18.0,
                "aicc": 20.0,
                "bic": 19.0,
                "num_params": 2,
                "params": {"qe": 50.0, "k2": 0.01},
            },
        }
        rec = recommend_best_models(results, model_type="kinetic")
        assert rec[0]["model"] == "PSO"

    def test_with_unconverged_models(self):
        results = {
            "Langmuir": {
                "converged": True,
                "r_squared": 0.98,
                "adj_r_squared": 0.97,
                "rmse": 1.5,
                "aic": 20.0,
                "aicc": 22.0,
                "bic": 21.0,
                "num_params": 2,
                "params": {"qm": 100.0, "KL": 0.05},
            },
            "Sips": {"converged": False},
        }
        rec = recommend_best_models(results, model_type="isotherm")
        assert len(rec) == 1


# =============================================================================
# THERMODYNAMIC PARAMETERS
# =============================================================================
class TestThermodynamicParametersExtended:
    def test_exothermic(self):
        T_K = np.array([298.15, 308.15, 318.15])
        Kd = np.array([5.0, 3.0, 2.0])  # Decreasing Kd → exothermic
        result = calculate_thermodynamic_parameters(T_K, Kd)
        assert result["success"] is True
        assert result["delta_H"] < 0

    def test_endothermic(self):
        T_K = np.array([298.15, 308.15, 318.15])
        Kd = np.array([2.0, 3.0, 5.0])  # Increasing Kd → endothermic
        result = calculate_thermodynamic_parameters(T_K, Kd)
        assert result["success"] is True
        assert result["delta_H"] > 0

    def test_with_two_temperatures(self):
        T_K = np.array([298.15, 318.15])
        Kd = np.array([5.0, 3.0])
        result = calculate_thermodynamic_parameters(T_K, Kd)
        assert result["success"] is True


# =============================================================================
# INTERPRET THERMODYNAMICS
# =============================================================================
class TestInterpretThermodynamicsExtended:
    def test_exothermic_spontaneous(self):
        result = interpret_thermodynamics(
            delta_H=-30.0,
            delta_S=10.0,
            delta_G=[-5.0, -6.0],
        )
        assert "exothermic" in result.get("enthalpy", "").lower()
        assert "spontaneous" in result.get("spontaneity", "").lower()

    def test_endothermic(self):
        result = interpret_thermodynamics(
            delta_H=30.0,
            delta_S=50.0,
            delta_G=[-1.0, -2.0],
        )
        assert "endothermic" in result.get("enthalpy", "").lower()

    def test_positive_delta_g(self):
        result = interpret_thermodynamics(
            delta_H=30.0,
            delta_S=-10.0,
            delta_G=[5.0, 6.0],
        )
        assert "non-spontaneous" in result.get("spontaneity", "").lower()


# =============================================================================
# CALCULATE ERROR METRICS
# =============================================================================
class TestCalculateErrorMetricsExtended:
    def test_perfect_fit(self):
        y = np.array([1, 2, 3, 4, 5], dtype=float)
        metrics = calculate_error_metrics(y, y, n_params=1)
        assert metrics["r_squared"] == pytest.approx(1.0)
        assert metrics["rmse"] == pytest.approx(0.0)

    def test_bad_fit(self):
        y_obs = np.array([1, 2, 3, 4, 5], dtype=float)
        y_pred = np.array([5, 4, 3, 2, 1], dtype=float)
        metrics = calculate_error_metrics(y_obs, y_pred, n_params=1)
        assert metrics["r_squared"] < 0.1

    def test_all_metrics_present(self):
        y_obs = np.array([1, 2, 3, 4, 5], dtype=float)
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        metrics = calculate_error_metrics(y_obs, y_pred, n_params=2)
        for key in ["r_squared", "adj_r_squared", "rmse", "mae", "aic", "bic"]:
            assert key in metrics


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================
class TestBootstrapCIExtended:
    def test_basic_bootstrap(self):
        Ce = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0])
        qe = langmuir_model(Ce, 100.0, 0.05) + np.random.RandomState(42).normal(0, 1, 7)

        ci_lower, ci_upper = bootstrap_confidence_intervals(
            langmuir_model,
            Ce,
            qe,
            params=np.array([80.0, 0.03]),
            n_bootstrap=50,
        )
        assert ci_lower is not None
        assert ci_upper is not None
        assert len(ci_lower) == 2
        assert len(ci_upper) == 2

    def test_bootstrap_with_bounds(self):
        Ce = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0])
        qe = langmuir_model(Ce, 100.0, 0.05) + np.random.RandomState(42).normal(0, 1, 7)

        ci_lower, ci_upper = bootstrap_confidence_intervals(
            langmuir_model,
            Ce,
            qe,
            params=np.array([80.0, 0.03]),
            n_bootstrap=30,
        )
        assert ci_lower is not None
        assert ci_upper is not None


# =============================================================================
# SEPARATION FACTOR
# =============================================================================
class TestSeparationFactorExtended:
    def test_favorable(self):
        RL = calculate_separation_factor(0.05, np.array([10.0, 50.0, 100.0]))
        assert np.all(RL > 0) and np.all(RL < 1)

    def test_interpretation(self):
        RL = np.array([0.3, 0.5, 0.7])
        interp = interpret_separation_factor(RL)
        assert "favorable" in interp.lower()


# =============================================================================
# Q2 STATISTIC
# =============================================================================
class TestQ2Extended:
    def test_good_q2(self):
        y_data = np.array([1, 2, 3, 4, 5], dtype=float)
        press = 0.5
        q2 = calculate_q2(press, y_data)
        assert q2 > 0.9

    def test_bad_q2(self):
        y_data = np.array([1, 2, 3, 4, 5], dtype=float)
        press = 20.0
        q2 = calculate_q2(press, y_data)
        assert q2 < 0.5
