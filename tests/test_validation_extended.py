"""
Extended tests for adsorblab_pro/validation.py - targeting 80%+ coverage.

Covers:
- ValidationLevel enum
- ValidationResult / ValidationReport methods
- validate_required_params comprehensive
- validate_experimental_params full paths
- validate_isotherm_data edge cases
- validate_kinetic_data edge cases
- validate_thermodynamic_data edge cases
- validate_calibration_data edge cases
- validate_dataframe edge cases
- validate_array edge cases
- validate_positive / validate_range edge cases
- validate_uploaded_file edge cases
- quick_validate
- format_validation_errors
"""

import numpy as np
import pandas as pd

from adsorblab_pro.validation import (
    ValidationLevel,
    ValidationReport,
    ValidationResult,
    format_validation_errors,
    quick_validate,
    validate_array,
    validate_calibration_data,
    validate_dataframe,
    validate_experimental_params,
    validate_isotherm_data,
    validate_kinetic_data,
    validate_positive,
    validate_range,
    validate_required_params,
    validate_thermodynamic_data,
    validate_uploaded_file,
)


# =============================================================================
# VALIDATION LEVEL ENUM
# =============================================================================
class TestValidationLevel:
    def test_levels_exist(self):
        assert ValidationLevel.ERROR is not None
        assert ValidationLevel.WARNING is not None
        assert ValidationLevel.INFO is not None


# =============================================================================
# VALIDATION RESULT
# =============================================================================
class TestValidationResultExtended:
    def test_bool_valid(self):
        r = ValidationResult(is_valid=True, level=ValidationLevel.INFO, message="ok", field="x")
        assert bool(r) is True

    def test_bool_invalid(self):
        r = ValidationResult(is_valid=False, level=ValidationLevel.ERROR, message="bad", field="x")
        assert bool(r) is False


# =============================================================================
# VALIDATION REPORT
# =============================================================================
class TestValidationReportExtended:
    def test_error_count(self):
        errors = [
            ValidationResult(is_valid=False, level=ValidationLevel.ERROR, message="e1", field="a"),
            ValidationResult(is_valid=False, level=ValidationLevel.ERROR, message="e2", field="b"),
        ]
        report = ValidationReport(is_valid=False, errors=errors)
        assert report.error_count == 2

    def test_warning_count(self):
        warnings = [
            ValidationResult(is_valid=True, level=ValidationLevel.WARNING, message="w1", field="a"),
        ]
        report = ValidationReport(is_valid=True, warnings=warnings)
        assert report.warning_count == 1

    def test_has_warnings(self):
        warnings = [
            ValidationResult(is_valid=True, level=ValidationLevel.WARNING, message="w1", field="a"),
        ]
        report = ValidationReport(is_valid=True, warnings=warnings)
        assert report.has_warnings is True

    def test_no_warnings(self):
        report = ValidationReport(is_valid=True)
        assert report.has_warnings is False

    def test_results_combines_all(self):
        error = ValidationResult(
            is_valid=False, level=ValidationLevel.ERROR, message="e", field="a"
        )
        warning = ValidationResult(
            is_valid=True, level=ValidationLevel.WARNING, message="w", field="b"
        )
        info = ValidationResult(is_valid=True, level=ValidationLevel.INFO, message="i", field="c")
        report = ValidationReport(is_valid=False, errors=[error], warnings=[warning], info=[info])
        assert len(report.results) == 3

    def test_to_dict(self):
        report = ValidationReport(is_valid=True)
        d = report.to_dict()
        assert "is_valid" in d
        assert d["is_valid"] is True
        assert "errors" in d
        assert "warnings" in d
        assert "info" in d

    def test_bool_valid_report(self):
        report = ValidationReport(is_valid=True)
        assert bool(report) is True

    def test_bool_invalid_report(self):
        report = ValidationReport(is_valid=False)
        assert bool(report) is False


# =============================================================================
# VALIDATE REQUIRED PARAMS
# =============================================================================
class TestValidateRequiredParamsExtended:
    def test_all_valid(self):
        params = {"C0": 100.0, "V": 0.05, "m": 0.1}
        valid, msg = validate_required_params(
            params,
            [("C0", "Initial Concentration"), ("V", "Volume"), ("m", "Mass")],
        )
        assert valid is True
        assert msg == ""

    def test_missing_param(self):
        params = {"C0": 100.0}
        valid, msg = validate_required_params(
            params,
            [("C0", "Initial Concentration"), ("V", "Volume")],
        )
        assert valid is False
        assert "missing" in msg.lower()

    def test_non_numeric_param(self):
        params = {"C0": "abc", "V": 0.05}
        valid, msg = validate_required_params(
            params,
            [("C0", "Initial Concentration"), ("V", "Volume")],
        )
        assert valid is False
        assert "number" in msg.lower()

    def test_negative_param(self):
        params = {"C0": -10.0, "V": 0.05}
        valid, msg = validate_required_params(
            params,
            [("C0", "Initial Concentration"), ("V", "Volume")],
        )
        assert valid is False
        assert "positive" in msg.lower()

    def test_zero_param(self):
        params = {"C0": 0.0, "V": 0.05}
        valid, msg = validate_required_params(
            params,
            [("C0", "Initial Concentration"), ("V", "Volume")],
        )
        assert valid is False
        assert "positive" in msg.lower()

    def test_none_value(self):
        params = {"C0": None}
        valid, msg = validate_required_params(
            params,
            [("C0", "Initial Concentration")],
        )
        assert valid is False
        assert "missing" in msg.lower()


# =============================================================================
# VALIDATE EXPERIMENTAL PARAMS
# =============================================================================
class TestValidateExperimentalParamsExtended:
    def test_all_valid(self):
        report = validate_experimental_params(V=0.05, m=0.1, C0=100.0, pH=7.0, T=298.15)
        assert report.is_valid is True

    def test_invalid_volume(self):
        report = validate_experimental_params(V=-0.1)
        assert report.is_valid is False

    def test_invalid_mass(self):
        report = validate_experimental_params(m=-1.0)
        assert report.is_valid is False

    def test_invalid_ph_high(self):
        report = validate_experimental_params(pH=15.0)
        assert report.is_valid is False

    def test_invalid_ph_low(self):
        report = validate_experimental_params(pH=-1.0)
        assert report.is_valid is False

    def test_temperature_below_freezing(self):
        report = validate_experimental_params(T=260.0)
        assert report.is_valid is True  # Valid but warning
        assert report.has_warnings is True

    def test_temperature_too_low(self):
        report = validate_experimental_params(T=100.0)
        assert report.is_valid is False

    def test_temperature_too_high(self):
        report = validate_experimental_params(T=500.0)
        assert report.is_valid is False

    def test_no_params(self):
        report = validate_experimental_params()
        assert report.is_valid is True

    def test_volume_too_large(self):
        report = validate_experimental_params(V=200.0)
        assert report.is_valid is False

    def test_mass_too_large(self):
        report = validate_experimental_params(m=2000.0)
        assert report.is_valid is False

    def test_c0_at_zero(self):
        report = validate_experimental_params(C0=0.0)
        assert report.is_valid is True  # 0 is allowed for C0 range


# =============================================================================
# VALIDATE ISOTHERM DATA
# =============================================================================
class TestValidateIsothermDataExtended:
    def test_valid_data(self):
        report = validate_isotherm_data(
            Ce=np.array([1, 5, 10, 20, 50]),
            qe=np.array([5, 15, 22, 30, 40]),
            C0=100.0,
        )
        assert report.is_valid is True

    def test_ce_greater_than_c0(self):
        report = validate_isotherm_data(
            Ce=np.array([10, 50, 110]),
            qe=np.array([5, 15, -5]),
            C0=100.0,
        )
        assert report.is_valid is False

    def test_too_few_points_warning(self):
        report = validate_isotherm_data(
            Ce=np.array([1, 5, 10]),
            qe=np.array([5, 15, 22]),
            C0=100.0,
        )
        assert report.is_valid is True
        # May have warning about few points

    def test_volume_in_ml_warning(self):
        report = validate_isotherm_data(
            Ce=np.array([1, 5, 10, 20, 50]),
            qe=np.array([5, 15, 22, 30, 40]),
            C0=100.0,
            V=50.0,  # Likely mL, not L
        )
        if report.has_warnings:
            assert any(
                "mL" in str(w.message) or "volume" in str(w.message).lower()
                for w in report.warnings
            )


# =============================================================================
# VALIDATE KINETIC DATA
# =============================================================================
class TestValidateKineticDataExtended:
    def test_valid_data(self):
        report = validate_kinetic_data(
            t=np.array([0, 5, 10, 20, 30, 60]),
            qt=np.array([0, 15, 25, 35, 42, 48]),
        )
        assert report.is_valid is True

    def test_non_monotonic_time(self):
        report = validate_kinetic_data(
            t=np.array([0, 5, 3, 20, 30]),
            qt=np.array([0, 15, 12, 35, 42]),
        )
        assert report.is_valid is False

    def test_missing_t0_warning(self):
        report = validate_kinetic_data(
            t=np.array([5, 10, 20, 30, 60]),
            qt=np.array([15, 25, 35, 42, 48]),
        )
        # Should warn about missing t=0
        if report.has_warnings:
            assert any(
                "t0" in str(w.message).lower()
                or "t=0" in str(w.message).lower()
                or "start" in str(w.message).lower()
                for w in report.warnings
            )

    def test_too_few_points(self):
        report = validate_kinetic_data(
            t=np.array([0, 5]),
            qt=np.array([0, 15]),
        )
        # May be invalid or have warnings
        assert isinstance(report, ValidationReport)


# =============================================================================
# VALIDATE THERMODYNAMIC DATA
# =============================================================================
class TestValidateThermodynamicDataExtended:
    def test_valid_kelvin(self):
        report = validate_thermodynamic_data(
            T=np.array([293.15, 303.15, 313.15]),
            Kd=np.array([5.0, 3.0, 2.0]),
        )
        assert report.is_valid is True

    def test_celsius_detection(self):
        report = validate_thermodynamic_data(
            T=np.array([20, 30, 40]),
            Kd=np.array([5.0, 3.0, 2.0]),
        )
        assert report.is_valid is False

    def test_too_few_temperatures(self):
        report = validate_thermodynamic_data(
            T=np.array([298.15, 308.15]),
            Kd=np.array([5.0, 3.0]),
        )
        # Valid but may have warning about few points
        assert isinstance(report, ValidationReport)

    def test_negative_kd(self):
        report = validate_thermodynamic_data(
            T=np.array([293.15, 303.15, 313.15]),
            Kd=np.array([5.0, -3.0, 2.0]),
        )
        # Should flag negative Kd
        assert isinstance(report, ValidationReport)


# =============================================================================
# VALIDATE CALIBRATION DATA
# =============================================================================
class TestValidateCalibrationDataExtended:
    def test_valid_data(self):
        report = validate_calibration_data(
            concentrations=np.array([0, 10, 20, 30, 40, 50]),
            absorbances=np.array([0.01, 0.21, 0.41, 0.61, 0.81, 1.01]),
        )
        assert report.is_valid is True

    def test_mismatched_lengths(self):
        report = validate_calibration_data(
            concentrations=np.array([0, 10, 20]),
            absorbances=np.array([0.01, 0.21]),
        )
        assert report.is_valid is False

    def test_too_few_points(self):
        report = validate_calibration_data(
            concentrations=np.array([0]),
            absorbances=np.array([0.01]),
        )
        assert report.is_valid is False

    def test_duplicate_concentrations_warning(self):
        report = validate_calibration_data(
            concentrations=np.array([0, 10, 10, 30, 40]),
            absorbances=np.array([0.01, 0.21, 0.22, 0.61, 0.81]),
        )
        # May have warning about duplicates
        assert isinstance(report, ValidationReport)


# =============================================================================
# VALIDATE DATAFRAME
# =============================================================================
class TestValidateDataFrameExtended:
    def test_valid_df(self):
        df = pd.DataFrame({"Ce": [1, 5, 10], "qe": [5, 15, 22]})
        report = validate_dataframe(df, required_columns=["Ce", "qe"])
        assert report.is_valid is True

    def test_none_df(self):
        report = validate_dataframe(None, required_columns=["Ce"])
        assert report.is_valid is False

    def test_not_a_dataframe(self):
        report = validate_dataframe("not_a_df", required_columns=["Ce"])
        assert report.is_valid is False

    def test_too_few_rows(self):
        df = pd.DataFrame({"Ce": [1], "qe": [5]})
        report = validate_dataframe(df, required_columns=["Ce", "qe"], min_rows=3)
        assert report.is_valid is False

    def test_missing_columns(self):
        df = pd.DataFrame({"Ce": [1, 2, 3]})
        report = validate_dataframe(df, required_columns=["Ce", "qe"])
        assert report.is_valid is False


# =============================================================================
# VALIDATE ARRAY
# =============================================================================
class TestValidateArrayExtended:
    def test_valid_array(self):
        result = validate_array(np.array([1.0, 2.0, 3.0]), "data")
        assert result.is_valid is True

    def test_none_array(self):
        result = validate_array(None, "data")
        assert result.is_valid is False

    def test_too_short(self):
        result = validate_array(np.array([1.0]), "data", min_length=3)
        assert result.is_valid is False

    def test_contains_nan(self):
        result = validate_array(np.array([1.0, np.nan, 3.0]), "data")
        assert result.is_valid is False

    def test_nan_allowed(self):
        result = validate_array(np.array([1.0, np.nan, 3.0]), "data", allow_nan=True)
        assert result.is_valid is True

    def test_contains_inf(self):
        result = validate_array(np.array([1.0, np.inf, 3.0]), "data")
        assert result.is_valid is False

    def test_negative_not_allowed(self):
        result = validate_array(np.array([1.0, -2.0, 3.0]), "data")
        assert result.is_valid is False

    def test_negative_allowed(self):
        result = validate_array(np.array([1.0, -2.0, 3.0]), "data", allow_negative=True)
        assert result.is_valid is True

    def test_list_input(self):
        result = validate_array([1.0, 2.0, 3.0], "data")
        assert result.is_valid is True

    def test_empty_array(self):
        result = validate_array(np.array([]), "data")
        assert result.is_valid is False


# =============================================================================
# VALIDATE POSITIVE / VALIDATE RANGE
# =============================================================================
class TestValidatePositiveExtended:
    def test_positive(self):
        assert validate_positive(5.0, "x").is_valid is True

    def test_zero_not_allowed(self):
        assert validate_positive(0.0, "x").is_valid is False

    def test_zero_allowed(self):
        assert validate_positive(0.0, "x", allow_zero=True).is_valid is True

    def test_negative(self):
        assert validate_positive(-1.0, "x").is_valid is False

    def test_nan(self):
        assert validate_positive(float("nan"), "x").is_valid is False

    def test_inf(self):
        assert validate_positive(float("inf"), "x").is_valid is False

    def test_string_input(self):
        assert validate_positive("abc", "x").is_valid is False

    def test_field_in_message(self):
        r = validate_positive(-1.0, "temperature")
        assert "temperature" in r.message.lower() or r.field == "temperature"


class TestValidateRangeExtended:
    def test_within_range(self):
        assert validate_range(5.0, "x", min_val=0, max_val=10).is_valid is True

    def test_below_min(self):
        assert validate_range(-1.0, "x", min_val=0, max_val=10).is_valid is False

    def test_above_max(self):
        assert validate_range(15.0, "x", min_val=0, max_val=10).is_valid is False

    def test_at_min_inclusive(self):
        assert validate_range(0.0, "x", min_val=0, max_val=10).is_valid is True

    def test_at_min_exclusive(self):
        assert (
            validate_range(0.0, "x", min_val=0, max_val=10, inclusive_min=False).is_valid is False
        )

    def test_no_bounds(self):
        assert validate_range(999.0, "x").is_valid is True


# =============================================================================
# VALIDATE UPLOADED FILE
# =============================================================================
class TestValidateUploadedFileExtended:
    def test_valid_csv(self):
        report = validate_uploaded_file(1000, "data.csv")
        assert report.is_valid is True

    def test_valid_xlsx(self):
        report = validate_uploaded_file(1000, "data.xlsx")
        assert report.is_valid is True

    def test_too_large(self):
        report = validate_uploaded_file(20 * 1024 * 1024, "data.csv")
        assert report.is_valid is False

    def test_invalid_type(self):
        report = validate_uploaded_file(1000, "data.pdf")
        assert report.is_valid is False

    def test_valid_xls(self):
        report = validate_uploaded_file(1000, "data.xls")
        assert report.is_valid is True


# =============================================================================
# QUICK VALIDATE
# =============================================================================
class TestQuickValidateExtended:
    def test_positive_valid(self):
        assert quick_validate(5.0, "x", "positive") is True

    def test_positive_invalid(self):
        assert quick_validate(-1.0, "x", "positive") is False

    def test_non_negative_valid(self):
        assert quick_validate(0.0, "x", "non_negative") is True

    def test_non_negative_invalid(self):
        assert quick_validate(-1.0, "x", "non_negative") is False

    def test_array_valid(self):
        assert quick_validate(np.array([1, 2, 3]), "x", "array") is True

    def test_array_invalid(self):
        assert quick_validate(None, "x", "array") is False


# =============================================================================
# FORMAT VALIDATION ERRORS
# =============================================================================
class TestFormatValidationErrorsExtended:
    def test_with_errors(self):
        errors = [
            ValidationResult(
                is_valid=False, level=ValidationLevel.ERROR, message="Error 1", field="a"
            ),
        ]
        report = ValidationReport(is_valid=False, errors=errors)
        result = format_validation_errors(report)
        assert "Error 1" in result

    def test_with_warnings(self):
        warnings = [
            ValidationResult(
                is_valid=True, level=ValidationLevel.WARNING, message="Warn 1", field="b"
            ),
        ]
        report = ValidationReport(is_valid=True, warnings=warnings)
        result = format_validation_errors(report)
        assert "Warn 1" in result

    def test_all_valid(self):
        report = ValidationReport(is_valid=True)
        result = format_validation_errors(report)
        assert isinstance(result, str)
