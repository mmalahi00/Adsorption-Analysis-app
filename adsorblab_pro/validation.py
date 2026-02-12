# validation.py
"""
AdsorbLab Pro - Input Validation Module
=======================================

Comprehensive input validation for adsorption data analysis.
Provides validators for:
- Numeric ranges and types
- Data structure integrity
- Scientific constraints
- Experimental parameter bounds

Author: AdsorbLab Team
Version: 2.0.0
License: MIT
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Import constants from config
from .config import (
    ALLOWED_FILE_TYPES,
    EPSILON_DIV,
    EPSILON_ZERO,
    KD_WARNING_THRESHOLD,
    MAX_FILE_SIZE_BYTES,
    TEMP_MAX_KELVIN,
    TEMP_MIN_KELVIN,
)

__all__ = [
    # Classes
    "ValidationLevel",
    "ValidationResult",
    "ValidationReport",
    # Basic validators
    "validate_positive",
    "validate_range",
    "validate_array",
    "validate_dataframe",
    # Scientific validators
    "validate_calibration_data",
    "validate_isotherm_data",
    "validate_kinetic_data",
    "validate_thermodynamic_data",
    "validate_experimental_params",
    # File validators
    "validate_uploaded_file",
    # Utilities
    "quick_validate",
    "format_validation_errors",
    "validate_required_params",
]
# =============================================================================
# VALIDATION RESULT CLASSES
# =============================================================================


class ValidationLevel(Enum):
    """Validation severity levels."""

    ERROR = "error"  # Critical - blocks analysis
    WARNING = "warning"  # Potential issue - allows continuation
    INFO = "info"  # Informational only


@dataclass
class ValidationResult:
    """
    Result of a validation check.

    Attributes
    ----------
    is_valid : bool
        Whether the validation passed
    level : ValidationLevel
        Severity level of any issues
    message : str
        Human-readable description
    field : str
        Name of the field/parameter being validated
    value : Any
        The actual value that was validated
    suggestion : str, optional
        Suggested fix for the issue
    """

    is_valid: bool
    level: ValidationLevel
    message: str
    field: str
    value: Any = None
    suggestion: str | None = None

    def __bool__(self) -> bool:
        return self.is_valid


@dataclass
class ValidationReport:
    """
    Aggregated validation results.

    Attributes
    ----------
    is_valid : bool
        True if no errors (warnings allowed)
    errors : List[ValidationResult]
        Critical validation failures
    warnings : List[ValidationResult]
        Non-critical issues
    info : List[ValidationResult]
        Informational messages
    """

    is_valid: bool
    errors: list[ValidationResult]
    warnings: list[ValidationResult]
    info: list[ValidationResult]

    def __bool__(self) -> bool:
        return self.is_valid

    @property
    def error_count(self) -> int:
        """Number of critical validation errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Number of non-critical warnings."""
        return len(self.warnings)

    @property
    def has_warnings(self) -> bool:
        """Check if report contains any warnings."""
        return self.warning_count > 0

    @property
    def results(self) -> list[ValidationResult]:
        """All validation results (errors + warnings + info) combined."""
        return self.errors + self.warnings + self.info

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "errors": [{"message": e.message, "field": e.field} for e in self.errors],
            "warnings": [{"message": w.message, "field": w.field} for w in self.warnings],
            "info": [{"message": i.message, "field": i.field} for i in self.info],
        }


# =============================================================================
# BASIC VALIDATORS
# =============================================================================


def validate_positive(value: float | int, field: str, allow_zero: bool = False) -> ValidationResult:
    """
    Validate that a value is positive.

    Parameters
    ----------
    value : float or int
        Value to validate
    field : str
        Name of the field for error messages
    allow_zero : bool
        If True, zero is acceptable

    Returns
    -------
    ValidationResult
    """
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"{field} must be a number",
            field=field,
            value=value,
            suggestion=f"Enter a valid positive number for {field}",
        )

    if np.isnan(val) or np.isinf(val):
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"{field} cannot be NaN or infinite",
            field=field,
            value=value,
        )

    threshold = 0 if allow_zero else EPSILON_ZERO
    if val < threshold:
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"{field} must be {'non-negative' if allow_zero else 'positive'} (got {val})",
            field=field,
            value=val,
            suggestion=f"Use a {'non-negative' if allow_zero else 'positive'} value",
        )

    return ValidationResult(
        is_valid=True,
        level=ValidationLevel.INFO,
        message=f"{field} is valid",
        field=field,
        value=val,
    )


def validate_range(
    value: float | int,
    field: str,
    min_val: float | None = None,
    max_val: float | None = None,
    inclusive: bool = True,
) -> ValidationResult:
    """
    Validate that a value is within a specified range.

    Parameters
    ----------
    value : float or int
        Value to validate
    field : str
        Name of the field
    min_val : float, optional
        Minimum allowed value
    max_val : float, optional
        Maximum allowed value
    inclusive : bool
        If True, bounds are inclusive

    Returns
    -------
    ValidationResult
    """
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"{field} must be a number",
            field=field,
            value=value,
        )

    if min_val is not None:
        if inclusive and val < min_val:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"{field} must be ≥ {min_val} (got {val})",
                field=field,
                value=val,
            )
        elif not inclusive and val <= min_val:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"{field} must be > {min_val} (got {val})",
                field=field,
                value=val,
            )

    if max_val is not None:
        if inclusive and val > max_val:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"{field} must be ≤ {max_val} (got {val})",
                field=field,
                value=val,
            )
        elif not inclusive and val >= max_val:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"{field} must be < {max_val} (got {val})",
                field=field,
                value=val,
            )

    return ValidationResult(
        is_valid=True,
        level=ValidationLevel.INFO,
        message=f"{field} is within valid range",
        field=field,
        value=val,
    )


def validate_array(
    arr: NDArray[np.floating[Any]] | None,
    field: str,
    min_length: int = 1,
    allow_nan: bool = False,
    allow_negative: bool = False,
) -> ValidationResult:
    """
    Validate a numpy array.

    Parameters
    ----------
    arr : np.ndarray
        Array to validate
    field : str
        Name of the field
    min_length : int
        Minimum required length
    allow_nan : bool
        If True, NaN values are acceptable
    allow_negative : bool
        If True, negative values are acceptable

    Returns
    -------
    ValidationResult
    """
    if arr is None:
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"{field} is required",
            field=field,
            value=None,
        )

    try:
        arr = np.asarray(arr, dtype=float)
    except (TypeError, ValueError) as e:
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"{field} must contain numeric values: {e}",
            field=field,
            value=arr,
        )

    if len(arr) < min_length:
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"{field} requires at least {min_length} values (got {len(arr)})",
            field=field,
            value=arr,
            suggestion=f"Add more data points to {field}",
        )

    if not allow_nan and np.any(np.isnan(arr)):
        nan_count = np.sum(np.isnan(arr))
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"{field} contains {nan_count} NaN value(s)",
            field=field,
            value=arr,
            suggestion="Remove or replace NaN values",
        )

    if np.any(np.isinf(arr)):
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"{field} contains infinite values",
            field=field,
            value=arr,
        )

    if not allow_negative and np.any(arr < 0):
        neg_count = np.sum(arr < 0)
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"{field} contains {neg_count} negative value(s)",
            field=field,
            value=arr,
            suggestion="Check data for negative values",
        )

    return ValidationResult(
        is_valid=True,
        level=ValidationLevel.INFO,
        message=f"{field} array is valid ({len(arr)} values)",
        field=field,
        value=arr,
    )


# =============================================================================
# DATAFRAME VALIDATORS
# =============================================================================


def validate_dataframe(
    df: pd.DataFrame | None,
    field: str,
    required_columns: list[str] | None = None,
    min_rows: int = 1,
    max_rows: int | None = None,
) -> ValidationResult:
    """
    Validate a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    field : str
        Name of the field
    required_columns : List[str], optional
        Columns that must be present
    min_rows : int
        Minimum number of rows required
    max_rows : int, optional
        Maximum number of rows allowed

    Returns
    -------
    ValidationResult
    """
    if df is None:
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"{field} is required",
            field=field,
            value=None,
        )

    if not isinstance(df, pd.DataFrame):
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"{field} must be a DataFrame",
            field=field,
            value=type(df).__name__,
        )

    if len(df) < min_rows:
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"{field} requires at least {min_rows} rows (got {len(df)})",
            field=field,
            value=len(df),
        )

    if max_rows and len(df) > max_rows:
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"{field} exceeds maximum of {max_rows} rows (got {len(df)})",
            field=field,
            value=len(df),
        )

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"{field} missing required columns: {missing}",
                field=field,
                value=list(df.columns),
                suggestion=f"Ensure columns {required_columns} are present",
            )

    return ValidationResult(
        is_valid=True,
        level=ValidationLevel.INFO,
        message=f"{field} is valid ({len(df)} rows, {len(df.columns)} columns)",
        field=field,
        value=df.shape,
    )


# =============================================================================
# SCIENTIFIC VALIDATORS
# =============================================================================


def validate_calibration_data(
    concentrations: NDArray[np.floating[Any]], absorbances: NDArray[np.floating[Any]]
) -> ValidationReport:
    """
    Validate calibration curve data.

    Parameters
    ----------
    concentrations : np.ndarray
        Standard concentrations
    absorbances : np.ndarray
        Measured absorbances

    Returns
    -------
    ValidationReport
    """
    errors: list[ValidationResult] = []
    warnings: list[ValidationResult] = []
    info: list[ValidationResult] = []

    # Check concentrations
    conc_result = validate_array(
        concentrations, "Concentrations", min_length=3, allow_negative=False
    )
    if not conc_result.is_valid:
        errors.append(conc_result)
    else:
        info.append(conc_result)

    # Check absorbances
    abs_result = validate_array(absorbances, "Absorbances", min_length=3, allow_negative=True)
    if not abs_result.is_valid:
        errors.append(abs_result)
    else:
        info.append(abs_result)

    # Check equal lengths
    if len(concentrations) != len(absorbances):
        errors.append(
            ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Data length mismatch: {len(concentrations)} concentrations vs {len(absorbances)} absorbances",
                field="Data",
                suggestion="Ensure equal number of concentration and absorbance values",
            )
        )

    # Check for duplicates
    if len(concentrations) > 0:
        unique_conc = len(np.unique(concentrations))
        if unique_conc < len(concentrations):
            warnings.append(
                ValidationResult(
                    is_valid=True,
                    level=ValidationLevel.WARNING,
                    message=f"Duplicate concentrations detected ({len(concentrations) - unique_conc} duplicates)",
                    field="Concentrations",
                    suggestion="Consider averaging replicates",
                )
            )

    # Check concentration range
    if len(concentrations) > 0 and np.max(concentrations) > 0:
        positive_concs = concentrations[concentrations > 0]
        if len(positive_concs) > 0:
            conc_range = np.max(positive_concs) / np.min(positive_concs)
            if conc_range < 5:
                warnings.append(
                    ValidationResult(
                        is_valid=True,
                        level=ValidationLevel.WARNING,
                        message="Narrow concentration range may affect linearity assessment",
                        field="Concentrations",
                        suggestion="Use a wider range of standards (at least 1 order of magnitude)",
                    )
                )

    # Check for zero concentration
    if len(concentrations) > 0 and 0 not in concentrations:
        info.append(
            ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message="No blank (zero concentration) included",
                field="Concentrations",
                suggestion="Consider including a blank for better intercept estimation",
            )
        )

    return ValidationReport(is_valid=len(errors) == 0, errors=errors, warnings=warnings, info=info)


def validate_isotherm_data(
    C0: NDArray[np.floating[Any]],
    Ce: NDArray[np.floating[Any]],
    qe: NDArray[np.floating[Any]] | None = None,
    V: float | None = None,
    m: float | None = None,
) -> ValidationReport:
    """
    Validate isotherm experimental data.

    Parameters
    ----------
    C0 : np.ndarray
        Initial concentrations (mg/L)
    Ce : np.ndarray
        Equilibrium concentrations (mg/L)
    qe : np.ndarray, optional
        Adsorption capacities (mg/g), calculated if not provided
    V : float, optional
        Solution volume (L)
    m : float, optional
        Adsorbent mass (g)

    Returns
    -------
    ValidationReport
    """
    errors: list[ValidationResult] = []
    warnings: list[ValidationResult] = []
    info: list[ValidationResult] = []

    # Validate C0
    c0_result = validate_array(C0, "Initial Concentration (C₀)", min_length=4, allow_negative=False)
    if not c0_result.is_valid:
        errors.append(c0_result)
    else:
        info.append(c0_result)

    # Validate Ce
    ce_result = validate_array(
        Ce, "Equilibrium Concentration (Cₑ)", min_length=4, allow_negative=False
    )
    if not ce_result.is_valid:
        errors.append(ce_result)
    else:
        info.append(ce_result)

    # Check Ce < C0
    if len(C0) == len(Ce) and len(C0) > 0:
        invalid_ce = Ce > C0
        if np.any(invalid_ce):
            errors.append(
                ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Equilibrium concentration exceeds initial at {np.sum(invalid_ce)} point(s)",
                    field="Ce vs C0",
                    suggestion="Check data entry - Cₑ should be ≤ C₀",
                )
            )

        # Check for negligible adsorption
        removal = (C0 - Ce) / np.maximum(C0, EPSILON_DIV) * 100
        if np.all(removal < 5):
            warnings.append(
                ValidationResult(
                    is_valid=True,
                    level=ValidationLevel.WARNING,
                    message="Very low adsorption (<5% removal) at all concentrations",
                    field="Adsorption",
                    suggestion="Check experimental conditions or adsorbent activity",
                )
            )

    # Validate experimental parameters
    if V is not None:
        v_result = validate_positive(V, "Volume (V)")
        if not v_result.is_valid:
            errors.append(v_result)
        elif V > 10:  # Likely in mL, not L
            warnings.append(
                ValidationResult(
                    is_valid=True,
                    level=ValidationLevel.WARNING,
                    message=f"Volume = {V} - ensure units are in Liters",
                    field="Volume",
                    suggestion="Convert from mL to L if needed (divide by 1000)",
                )
            )

    if m is not None:
        m_result = validate_positive(m, "Mass (m)")
        if not m_result.is_valid:
            errors.append(m_result)
        elif m > 100:  # Likely in mg, not g
            warnings.append(
                ValidationResult(
                    is_valid=True,
                    level=ValidationLevel.WARNING,
                    message=f"Mass = {m} - ensure units are in grams",
                    field="Mass",
                    suggestion="Convert from mg to g if needed (divide by 1000)",
                )
            )

    # Validate or calculate qe
    if qe is not None:
        # Validate provided qe
        qe_result = validate_array(
            qe, "Adsorption Capacity (qₑ)", min_length=4, allow_negative=False
        )
        if not qe_result.is_valid:
            errors.append(qe_result)
        else:
            info.append(qe_result)

        # Check qe consistency with mass balance if V and m provided
        if V is not None and m is not None and len(qe) == len(C0):
            qe_calculated = (C0 - Ce) * V / m
            relative_error = np.abs(qe - qe_calculated) / np.maximum(qe_calculated, EPSILON_DIV)
            if np.any(relative_error > 0.1):  # More than 10% deviation
                warnings.append(
                    ValidationResult(
                        is_valid=True,
                        level=ValidationLevel.WARNING,
                        message=f"qₑ values deviate >10% from mass balance at {np.sum(relative_error > 0.1)} point(s)",
                        field="qₑ consistency",
                        suggestion="Verify qₑ = (C₀ - Cₑ) × V / m calculation",
                    )
                )

        # Check for physically reasonable qe values
        if np.any(qe > 1000):  # Very high capacity
            warnings.append(
                ValidationResult(
                    is_valid=True,
                    level=ValidationLevel.WARNING,
                    message=f"Very high qₑ values detected (max: {np.max(qe):.1f} mg/g)",
                    field="qₑ range",
                    suggestion="Typical qₑ range is 1-500 mg/g; verify units and calculations",
                )
            )

    elif V is not None and m is not None:
        # Calculate qe from mass balance for info purposes
        qe_calculated = (C0 - Ce) * V / m
        info.append(
            ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message=f"qₑ calculated from mass balance: {qe_calculated.min():.2f} - {qe_calculated.max():.2f} mg/g",
                field="qₑ calculation",
            )
        )

    # Check data points for isotherm fitting
    if len(C0) >= 4 and len(C0) < 6:
        warnings.append(
            ValidationResult(
                is_valid=True,
                level=ValidationLevel.WARNING,
                message=f"Only {len(C0)} data points - consider adding more for robust fitting",
                field="Data points",
                suggestion="6-10 points recommended for isotherm fitting",
            )
        )

    return ValidationReport(is_valid=len(errors) == 0, errors=errors, warnings=warnings, info=info)


def validate_kinetic_data(
    time: NDArray[np.floating[Any]],
    qt: NDArray[np.floating[Any]] | None = None,
    Ct: NDArray[np.floating[Any]] | None = None,
    C0: float | None = None,
) -> ValidationReport:
    """
    Validate kinetic experimental data.

    Parameters
    ----------
    time : np.ndarray
        Time points (min)
    qt : np.ndarray, optional
        Adsorption capacity at time t (mg/g)
    Ct : np.ndarray, optional
        Concentration at time t (mg/L)
    C0 : float, optional
        Initial concentration

    Returns
    -------
    ValidationReport
    """
    errors: list[ValidationResult] = []
    warnings: list[ValidationResult] = []
    info: list[ValidationResult] = []

    # Validate time
    time_result = validate_array(time, "Time", min_length=5, allow_negative=False)
    if not time_result.is_valid:
        errors.append(time_result)
    else:
        info.append(time_result)

        # Check time starts at zero
        if time[0] != 0:
            warnings.append(
                ValidationResult(
                    is_valid=True,
                    level=ValidationLevel.WARNING,
                    message="Time series does not start at t=0",
                    field="Time",
                    suggestion="Include t=0 point for initial rate calculation",
                )
            )

        # Check monotonically increasing
        if not np.all(np.diff(time) > 0):
            errors.append(
                ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="Time values must be monotonically increasing",
                    field="Time",
                )
            )

    # Validate qt or Ct
    if qt is not None:
        qt_result = validate_array(
            qt, "Adsorption capacity (qt)", min_length=5, allow_negative=False
        )
        if not qt_result.is_valid:
            errors.append(qt_result)
        else:
            info.append(qt_result)

            # Check qt is monotonically increasing (generally)
            if len(qt) > 2:
                decreasing = np.sum(np.diff(qt) < -0.01 * np.max(qt))
                if decreasing > len(qt) * 0.3:
                    warnings.append(
                        ValidationResult(
                            is_valid=True,
                            level=ValidationLevel.WARNING,
                            message=f"qt decreases at {decreasing} points - check for desorption",
                            field="qt",
                        )
                    )

    if Ct is not None:
        ct_result = validate_array(Ct, "Concentration (Ct)", min_length=5, allow_negative=False)
        if not ct_result.is_valid:
            errors.append(ct_result)

        # Check Ct <= C0
        if C0 is not None and np.any(Ct > C0):
            errors.append(
                ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="Concentration exceeds initial at some time points",
                    field="Ct vs C0",
                )
            )

    # Check equilibrium
    if qt is not None and len(qt) >= 5:
        last_values = qt[-3:]
        if np.std(last_values) / np.mean(last_values) > 0.1:
            warnings.append(
                ValidationResult(
                    is_valid=True,
                    level=ValidationLevel.WARNING,
                    message="System may not have reached equilibrium",
                    field="Equilibrium",
                    suggestion="Consider extending experiment time",
                )
            )

    return ValidationReport(is_valid=len(errors) == 0, errors=errors, warnings=warnings, info=info)


def validate_thermodynamic_data(
    temperatures: NDArray[np.floating[Any]], Kd: NDArray[np.floating[Any]]
) -> ValidationReport:
    """
    Validate thermodynamic (Van't Hoff) data.

    Parameters
    ----------
    temperatures : np.ndarray
        Temperatures in Kelvin
    Kd : np.ndarray
        Distribution coefficients (dimensionless)

    Returns
    -------
    ValidationReport
    """
    errors: list[ValidationResult] = []
    warnings: list[ValidationResult] = []
    info: list[ValidationResult] = []

    # Validate temperatures
    temp_result = validate_array(temperatures, "Temperature", min_length=3, allow_negative=False)
    if not temp_result.is_valid:
        errors.append(temp_result)
    else:
        # Check temperature range
        if np.min(temperatures) < TEMP_MIN_KELVIN:
            errors.append(
                ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="Temperature appears to be in Celsius - convert to Kelvin",
                    field="Temperature",
                    suggestion="Add 273.15 to convert °C to K",
                )
            )
        elif np.max(temperatures) > TEMP_MAX_KELVIN:
            warnings.append(
                ValidationResult(
                    is_valid=True,
                    level=ValidationLevel.WARNING,
                    message="High temperature detected - verify units are in Kelvin",
                    field="Temperature",
                )
            )

    # Validate Kd
    kd_result = validate_array(
        Kd, "Distribution coefficient (Kd)", min_length=3, allow_negative=False
    )
    if not kd_result.is_valid:
        errors.append(kd_result)
    else:
        # Check for very small Kd
        if np.any(Kd < KD_WARNING_THRESHOLD):
            warnings.append(
                ValidationResult(
                    is_valid=True,
                    level=ValidationLevel.WARNING,
                    message="Very small Kd values may indicate poor adsorption",
                    field="Kd",
                )
            )

    # Check data points
    if len(temperatures) < 4:
        warnings.append(
            ValidationResult(
                is_valid=True,
                level=ValidationLevel.WARNING,
                message="Only 3 temperature points - consider adding more for robust Van't Hoff analysis",
                field="Data points",
                suggestion="4-5 temperatures recommended",
            )
        )

    return ValidationReport(is_valid=len(errors) == 0, errors=errors, warnings=warnings, info=info)


# =============================================================================
# EXPERIMENTAL PARAMETER VALIDATORS
# =============================================================================


def validate_experimental_params(
    V: float | None = None,
    m: float | None = None,
    C0: float | None = None,
    pH: float | None = None,
    T: float | None = None,
) -> ValidationReport:
    """
    Validate experimental parameters.

    Parameters
    ----------
    V : float, optional
        Solution volume (L)
    m : float, optional
        Adsorbent mass (g)
    C0 : float, optional
        Initial concentration (mg/L)
    pH : float, optional
        Solution pH
    T : float, optional
        Temperature (K)

    Returns
    -------
    ValidationReport
    """
    errors: list[ValidationResult] = []
    warnings: list[ValidationResult] = []
    info: list[ValidationResult] = []

    if V is not None:
        v_result = validate_range(V, "Volume", min_val=0.0001, max_val=100)
        if not v_result.is_valid:
            errors.append(v_result)

    if m is not None:
        m_result = validate_range(m, "Mass", min_val=0.0001, max_val=1000)
        if not m_result.is_valid:
            errors.append(m_result)

    if C0 is not None:
        c0_result = validate_range(C0, "Initial Concentration", min_val=0, max_val=100000)
        if not c0_result.is_valid:
            errors.append(c0_result)

    if pH is not None:
        ph_result = validate_range(pH, "pH", min_val=0, max_val=14)
        if not ph_result.is_valid:
            errors.append(ph_result)

    if T is not None:
        t_result = validate_range(
            T, "Temperature", min_val=TEMP_MIN_KELVIN, max_val=TEMP_MAX_KELVIN
        )
        if not t_result.is_valid:
            errors.append(t_result)
        elif T < 273.15:
            warnings.append(
                ValidationResult(
                    is_valid=True,
                    level=ValidationLevel.WARNING,
                    message="Temperature below freezing point of water",
                    field="Temperature",
                )
            )

    return ValidationReport(is_valid=len(errors) == 0, errors=errors, warnings=warnings, info=info)


def validate_required_params(
    params: dict[str, Any], required_keys: list[tuple[str, str]]
) -> tuple[bool, str]:
    """
    Validate that required parameters are present and positive.

    A lightweight validator for quick checks in UI code.

    Parameters
    ----------
    params : dict
        Dictionary of parameter values
    required_keys : list of tuples
        List of (key, label) tuples to validate.
        - key: The dictionary key to check
        - label: Human-readable name for error messages

    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
        - is_valid: True if all parameters are valid
        - error_message: Empty string if valid, otherwise describes the problem

    Examples
    --------
    >>> params = {'C0': 100, 'V': 0.05, 'm': 0.1}
    >>> is_valid, msg = validate_required_params(
    ...     params,
    ...     required_keys=[('C0', 'Initial Concentration'), ('V', 'Volume')]
    ... )
    >>> print(is_valid)
    True

    >>> params = {'C0': -10, 'V': 0.05}
    >>> is_valid, msg = validate_required_params(
    ...     params,
    ...     required_keys=[('C0', 'Initial Concentration')]
    ... )
    >>> print(is_valid)
    False
    """
    missing_or_invalid = []

    for key, label in required_keys:
        value = params.get(key)

        if value is None:
            missing_or_invalid.append(f"{label} ('{key}') is missing")
        elif not isinstance(value, int | float):
            missing_or_invalid.append(f"{label} ('{key}') must be a number")
        elif value <= 0:
            missing_or_invalid.append(f"{label} ('{key}') must be positive")

    if missing_or_invalid:
        error_msg = "Invalid or missing parameters:\n• " + "\n• ".join(missing_or_invalid)
        return False, error_msg

    return True, ""


# =============================================================================
# FILE VALIDATORS
# =============================================================================


def validate_uploaded_file(file_size: int, file_name: str) -> ValidationReport:
    """
    Validate uploaded file properties.

    Parameters
    ----------
    file_size : int
        File size in bytes
    file_name : str
        Name of the file

    Returns
    -------
    ValidationReport
    """
    errors: list[ValidationResult] = []
    warnings: list[ValidationResult] = []
    info: list[ValidationResult] = []

    # Check file size
    if file_size > MAX_FILE_SIZE_BYTES:
        errors.append(
            ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"File size ({file_size / 1e6:.1f} MB) exceeds limit ({MAX_FILE_SIZE_BYTES / 1e6:.0f} MB)",
                field="File size",
            )
        )

    # Check file extension
    if file_name:
        ext = file_name.split(".")[-1].lower() if "." in file_name else ""
        if ext not in ALLOWED_FILE_TYPES:
            errors.append(
                ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"File type '{ext}' not allowed. Use: {ALLOWED_FILE_TYPES}",
                    field="File type",
                )
            )

    return ValidationReport(is_valid=len(errors) == 0, errors=errors, warnings=warnings, info=info)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def quick_validate(value: Any, field: str, validator_type: str = "positive") -> bool:
    """
    Quick validation returning only True/False.

    Parameters
    ----------
    value : Any
        Value to validate
    field : str
        Field name
    validator_type : str
        Type of validation: 'positive', 'non_negative', 'array'

    Returns
    -------
    bool
        True if valid
    """
    if validator_type == "positive":
        return validate_positive(value, field).is_valid
    elif validator_type == "non_negative":
        return validate_positive(value, field, allow_zero=True).is_valid
    elif validator_type == "array":
        return validate_array(np.asarray(value), field).is_valid
    else:
        return True


def format_validation_errors(report: ValidationReport) -> str:
    """
    Format validation errors for display.

    Parameters
    ----------
    report : ValidationReport
        Validation results

    Returns
    -------
    str
        Formatted error message
    """
    lines = []

    if report.errors:
        lines.append("**Errors:**")
        for e in report.errors:
            lines.append(f"- ❌ {e.message}")
            if e.suggestion:
                lines.append(f"  💡 {e.suggestion}")

    if report.warnings:
        lines.append("\n**Warnings:**")
        for w in report.warnings:
            lines.append(f"- ⚠️ {w.message}")
            if w.suggestion:
                lines.append(f"  💡 {w.suggestion}")

    return "\n".join(lines) if lines else "✅ All validations passed"
