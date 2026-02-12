# tests/test_validation.py
"""
Unit Tests for Validation Module
================================

Tests for input validation functions.

Author: AdsorbLab Team
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adsorblab_pro.validation import (
    ValidationLevel,
    ValidationReport,
    # Classes
    ValidationResult,
    format_validation_errors,
    # Utilities
    quick_validate,
    validate_array,
    # Scientific validators
    validate_calibration_data,
    validate_dataframe,
    validate_isotherm_data,
    validate_kinetic_data,
    # Basic validators
    validate_positive,
    validate_range,
    validate_thermodynamic_data,
    # File validators
    validate_uploaded_file,
)

# =============================================================================
# BASIC VALIDATOR TESTS
# =============================================================================


class TestFormatValidationErrors:
    """Tests for error formatting."""

    def test_format_with_errors(self):
        """Test formatting with errors."""
        report = ValidationReport(
            is_valid=False,
            errors=[ValidationResult(False, ValidationLevel.ERROR, "Test error", "field")],
            warnings=[],
            info=[],
        )
        formatted = format_validation_errors(report)
        assert "error" in formatted.lower()

    def test_format_with_warnings(self):
        """Test formatting with warnings."""
        report = ValidationReport(
            is_valid=True,
            errors=[],
            warnings=[ValidationResult(True, ValidationLevel.WARNING, "Test warning", "field")],
            info=[],
        )
        formatted = format_validation_errors(report)
        assert "warning" in formatted.lower()

    def test_format_all_valid(self):
        """Test formatting when all valid."""
        report = ValidationReport(is_valid=True, errors=[], warnings=[], info=[])
        formatted = format_validation_errors(report)
        assert "passed" in formatted.lower() or "✅" in formatted


class TestQuickValidate:
    """Test quick validation utilities."""

    def test_array_validation(self):
        """Test array validation."""
        assert quick_validate([1, 2, 3], "test", "array")

    def test_non_negative_validation(self):
        """Test non-negative validation."""
        assert quick_validate(0, "test", "non_negative")
        assert not quick_validate(-1, "test", "non_negative")

    def test_positive_validation(self):
        """Test positive validation."""
        assert quick_validate(10, "test", "positive")
        assert not quick_validate(-5, "test", "positive")

    def test_quick_validate_array(self):
        """Test quick array validation."""
        from adsorblab_pro.validation import quick_validate

        assert quick_validate([1, 2, 3], "test_field", "array") is True

    def test_quick_validate_non_negative(self):
        """Test quick non-negative validation."""
        from adsorblab_pro.validation import quick_validate

        # 0.0 should be valid with non_negative
        assert quick_validate(5.0, "test_field", "non_negative") is True
        assert quick_validate(-1.0, "test_field", "non_negative") is False

    def test_quick_validate_positive(self):
        """Test quick positive validation."""
        from adsorblab_pro.validation import quick_validate

        assert quick_validate(5.0, "test_field", "positive") is True
        assert quick_validate(-5.0, "test_field", "positive") is False


class TestValidateArray:
    """Tests for validate_array function."""

    def test_valid_array(self):
        """Test valid array passes."""
        arr = np.array([1, 2, 3, 4, 5])
        result = validate_array(arr, "test")
        assert result.is_valid

    def test_none_invalid(self):
        """Test None array fails."""
        result = validate_array(None, "test")
        assert not result.is_valid

    def test_too_short_invalid(self):
        """Test array shorter than min_length fails."""
        arr = np.array([1, 2])
        result = validate_array(arr, "test", min_length=5)
        assert not result.is_valid

    def test_contains_nan_invalid_by_default(self):
        """Test array with NaN fails by default."""
        arr = np.array([1, 2, np.nan, 4])
        result = validate_array(arr, "test")
        assert not result.is_valid

    def test_contains_nan_valid_when_allowed(self):
        """Test array with NaN passes when allowed."""
        arr = np.array([1, 2, np.nan, 4])
        result = validate_array(arr, "test", allow_nan=True)
        assert result.is_valid

    def test_contains_inf_invalid(self):
        """Test array with infinity fails."""
        arr = np.array([1, 2, np.inf, 4])
        result = validate_array(arr, "test")
        assert not result.is_valid

    def test_negative_invalid_by_default(self):
        """Test array with negative values fails by default."""
        arr = np.array([1, -2, 3, 4])
        result = validate_array(arr, "test")
        assert not result.is_valid

    def test_negative_valid_when_allowed(self):
        """Test array with negative values passes when allowed."""
        arr = np.array([1, -2, 3, 4])
        result = validate_array(arr, "test", allow_negative=True)
        assert result.is_valid

    def test_list_converted_to_array(self):
        """Test list input is converted and validated."""
        result = validate_array([1, 2, 3, 4, 5], "test")
        assert result.is_valid


class TestValidateCalibrationData:
    """Tests for calibration data validation."""

    def test_valid_calibration_data(self):
        """Test valid calibration data passes."""
        conc = np.array([0, 5, 10, 20, 50])
        abs_val = np.array([0.01, 0.13, 0.26, 0.51, 1.26])
        report = validate_calibration_data(conc, abs_val)
        assert report.is_valid

    def test_mismatched_lengths_invalid(self):
        """Test mismatched array lengths fail."""
        conc = np.array([0, 5, 10])
        abs_val = np.array([0.01, 0.13])
        report = validate_calibration_data(conc, abs_val)
        assert not report.is_valid

    def test_too_few_points_invalid(self):
        """Test too few points fails."""
        conc = np.array([5, 10])
        abs_val = np.array([0.13, 0.26])
        report = validate_calibration_data(conc, abs_val)
        assert not report.is_valid

    def test_duplicate_concentrations_warning(self):
        """Test duplicate concentrations generates warning."""
        conc = np.array([5, 5, 10, 20])
        abs_val = np.array([0.12, 0.13, 0.26, 0.51])
        report = validate_calibration_data(conc, abs_val)
        assert report.has_warnings


class TestValidateDataFrame:
    """Tests for validate_dataframe function."""

    def test_valid_dataframe(self):
        """Test valid DataFrame passes."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = validate_dataframe(df, "test")
        assert result.is_valid

    def test_none_invalid(self):
        """Test None fails."""
        result = validate_dataframe(None, "test")
        assert not result.is_valid

    def test_not_dataframe_invalid(self):
        """Test non-DataFrame fails."""
        result = validate_dataframe([1, 2, 3], "test")
        assert not result.is_valid

    def test_too_few_rows_invalid(self):
        """Test DataFrame with too few rows fails."""
        df = pd.DataFrame({"A": [1]})
        result = validate_dataframe(df, "test", min_rows=5)
        assert not result.is_valid

    def test_missing_required_columns_invalid(self):
        """Test DataFrame missing required columns fails."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        result = validate_dataframe(df, "test", required_columns=["A", "B", "C"])
        assert not result.is_valid


class TestValidateIsothermData:
    """Tests for isotherm data validation."""

    def test_valid_isotherm_data(self):
        """Test valid isotherm data passes."""
        C0 = np.array([10, 25, 50, 75, 100])
        Ce = np.array([2, 8, 22, 40, 58])
        report = validate_isotherm_data(C0, Ce)
        assert report.is_valid

    def test_Ce_greater_than_C0_invalid(self):
        """Test Ce > C0 fails validation."""
        C0 = np.array([10, 25, 50, 75])
        Ce = np.array([12, 8, 22, 40])  # First point invalid
        report = validate_isotherm_data(C0, Ce)
        assert not report.is_valid

    def test_too_few_points_warning(self):
        """Test few data points generates warning."""
        C0 = np.array([10, 25, 50, 75])
        Ce = np.array([2, 8, 22, 40])
        report = validate_isotherm_data(C0, Ce)
        assert report.has_warnings

    def test_volume_in_mL_warning(self):
        """Test large volume value generates warning."""
        C0 = np.array([10, 25, 50, 75, 100])
        Ce = np.array([2, 8, 22, 40, 58])
        report = validate_isotherm_data(C0, Ce, V=50)  # 50 looks like mL
        assert report.has_warnings


class TestValidateKineticData:
    """Tests for kinetic data validation."""

    def test_valid_kinetic_data(self):
        """Test valid kinetic data passes."""
        t = np.array([0, 5, 10, 20, 30, 60])
        qt = np.array([0, 12, 22, 35, 44, 55])
        report = validate_kinetic_data(t, qt)
        assert report.is_valid

    def test_non_monotonic_time_invalid(self):
        """Test non-monotonic time fails."""
        t = np.array([0, 10, 5, 20, 30])  # Not increasing
        qt = np.array([0, 22, 12, 35, 44])
        report = validate_kinetic_data(t, qt)
        assert not report.is_valid

    def test_missing_t0_warning(self):
        """Test missing t=0 generates warning."""
        t = np.array([5, 10, 20, 30, 60])
        qt = np.array([12, 22, 35, 44, 55])
        report = validate_kinetic_data(t, qt)
        assert report.has_warnings


class TestValidatePositive:
    """Tests for validate_positive function."""

    def test_positive_value_valid(self):
        """Test positive value passes validation."""
        result = validate_positive(10.5, "test_field")
        assert result.is_valid

    def test_zero_invalid_by_default(self):
        """Test zero fails validation by default."""
        result = validate_positive(0, "test_field")
        assert not result.is_valid

    def test_zero_valid_when_allowed(self):
        """Test zero passes when allow_zero=True."""
        result = validate_positive(0, "test_field", allow_zero=True)
        assert result.is_valid

    def test_negative_invalid(self):
        """Test negative value fails validation."""
        result = validate_positive(-5, "test_field")
        assert not result.is_valid

    def test_nan_invalid(self):
        """Test NaN fails validation."""
        result = validate_positive(float("nan"), "test_field")
        assert not result.is_valid

    def test_inf_invalid(self):
        """Test infinity fails validation."""
        result = validate_positive(float("inf"), "test_field")
        assert not result.is_valid

    def test_non_numeric_invalid(self):
        """Test non-numeric value fails validation."""
        result = validate_positive("abc", "test_field")
        assert not result.is_valid

    def test_result_contains_field_name(self):
        """Test result contains field name."""
        result = validate_positive(10, "my_field")
        assert result.field == "my_field"


class TestValidateRange:
    """Tests for validate_range function."""

    def test_within_range_valid(self):
        """Test value within range passes."""
        result = validate_range(50, "test", min_val=0, max_val=100)
        assert result.is_valid

    def test_below_min_invalid(self):
        """Test value below minimum fails."""
        result = validate_range(-5, "test", min_val=0)
        assert not result.is_valid

    def test_above_max_invalid(self):
        """Test value above maximum fails."""
        result = validate_range(150, "test", max_val=100)
        assert not result.is_valid

    def test_at_min_valid_inclusive(self):
        """Test value at minimum passes with inclusive bounds."""
        result = validate_range(0, "test", min_val=0, inclusive=True)
        assert result.is_valid

    def test_at_min_invalid_exclusive(self):
        """Test value at minimum fails with exclusive bounds."""
        result = validate_range(0, "test", min_val=0, inclusive=False)
        assert not result.is_valid

    def test_no_bounds_always_valid(self):
        """Test any value passes with no bounds specified."""
        result = validate_range(-1000, "test")
        assert result.is_valid


class TestValidateThermodynamicData:
    """Tests for thermodynamic data validation."""

    def test_valid_thermodynamic_data(self):
        """Test valid thermodynamic data passes."""
        T = np.array([298.15, 308.15, 318.15, 328.15])
        Kd = np.array([15.2, 12.1, 9.8, 8.1])
        report = validate_thermodynamic_data(T, Kd)
        assert report.is_valid

    def test_celsius_temperatures_invalid(self):
        """Test Celsius temperatures fail (should be Kelvin)."""
        T = np.array([25, 35, 45, 55])  # Celsius
        Kd = np.array([15.2, 12.1, 9.8, 8.1])
        report = validate_thermodynamic_data(T, Kd)
        assert not report.is_valid

    def test_too_few_temperatures_warning(self):
        """Test few temperature points generates warning."""
        T = np.array([298.15, 308.15, 318.15])
        Kd = np.array([15.2, 12.1, 9.8])
        report = validate_thermodynamic_data(T, Kd)
        assert report.has_warnings


class TestValidateUploadedFile:
    """Tests for file upload validation."""

    def test_valid_file(self):
        """Test valid file passes."""
        report = validate_uploaded_file(1000000, "data.xlsx")  # 1 MB
        assert report.is_valid

    def test_file_too_large_invalid(self):
        """Test oversized file fails."""
        report = validate_uploaded_file(100000000, "data.xlsx")  # 100 MB
        assert not report.is_valid

    def test_invalid_file_type(self):
        """Test invalid file type fails."""
        report = validate_uploaded_file(1000, "data.exe")
        assert not report.is_valid

    def test_valid_csv_file(self):
        """Test CSV file passes."""
        report = validate_uploaded_file(1000, "data.csv")
        assert report.is_valid


class TestValidationClasses:
    """Tests for validation result classes."""

    def test_validation_result_bool(self):
        """Test ValidationResult boolean conversion."""
        valid = ValidationResult(True, ValidationLevel.INFO, "OK", "field")
        invalid = ValidationResult(False, ValidationLevel.ERROR, "Fail", "field")

        assert bool(valid) is True
        assert bool(invalid) is False

    def test_validation_report_bool(self):
        """Test ValidationReport boolean conversion."""
        valid_report = ValidationReport(True, [], [], [])
        invalid_report = ValidationReport(
            False, [ValidationResult(False, ValidationLevel.ERROR, "Fail", "f")], [], []
        )

        assert bool(valid_report) is True
        assert bool(invalid_report) is False

    def test_validation_report_to_dict(self):
        """Test ValidationReport serialization."""
        report = ValidationReport(
            is_valid=False,
            errors=[ValidationResult(False, ValidationLevel.ERROR, "Error msg", "field1")],
            warnings=[ValidationResult(True, ValidationLevel.WARNING, "Warning msg", "field2")],
            info=[],
        )
        d = report.to_dict()

        assert "is_valid" in d
        assert "errors" in d
        assert "warnings" in d
        assert len(d["errors"]) == 1
