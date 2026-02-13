# utils.py
"""
AdsorbLab Pro - Advanced Utility Functions
==========================================

Advanced utility module providing:
- Advanced statistical analysis with confidence intervals
- Bootstrap resampling for robust error estimation
- PRESS/Q² cross-validation statistics
- Mechanism consistency checking
- Methodological error detection
- Parameter uncertainty propagation
- Comprehensive model selection criteria (R², Adj-R², AIC, BIC, F-test)
- Residual analysis and diagnostics
- Dual-unit calculations (mg/g and % Removal)
- Intelligent rule-based model recommendations
- Data quality assessment
- Advanced figure generation
"""

import io
import logging
import re
import warnings
from collections.abc import Callable

# PIL import deferred to _convert_png_to_tiff_bytes() for faster startup
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import normaltest, shapiro
from scipy.stats import t as t_dist

logger = logging.getLogger(__name__)

# Type variable for generic function typing
_F = TypeVar("_F", bound=Callable[..., Any])

# =============================================================================
# OPTIONAL STREAMLIT IMPORT (enables headless testing/scripting)
# =============================================================================
from adsorblab_pro.streamlit_compat import st, STREAMLIT_AVAILABLE as _STREAMLIT_AVAILABLE


def _optional_cache(func: _F) -> _F:
    """
    Decorator that applies @st.cache_data only when Streamlit is available.

    This allows utils.py to be imported in headless environments (pytest, scripts)
    without requiring Streamlit to be installed.
    """
    if _STREAMLIT_AVAILABLE:
        return st.cache_data(func)
    return func


# =============================================================================
# CONSTANTS - Import from central config for consistency
# =============================================================================
from .config import (
    BOOTSTRAP_DEFAULT_ITERATIONS,
    BOOTSTRAP_MIN_SUCCESS,
    EPSILON_DIV,
    EPSILON_ZERO,
    FONT_FAMILY,
    FUZZY_MATCH_CUTOFF,
    MIN_DATA_POINTS,
    PLOT_TEMPLATE,
    R_GAS_CONSTANT,
    SESSION_INPUT_KEYS_TO_CLEAR,
    SESSION_WIDGET_PREFIXES_TO_CLEAR,
    STUDY_METRIC_DATA_KEYS,
)
from .validation import validate_required_params

__all__ = [
    # Constants
    "CONFIDENCE_LEVEL",
    "COLUMN_SYNONYMS",
    # Data classes
    "CalculationResult",
    # Column standardization
    "standardize_column_name",
    "standardize_dataframe_columns",
    # Data validation
    "validate_data_editor",
    # Basic calculations
    "calculate_removal_percentage",
    "calculate_adsorption_capacity",
    "calculate_Ce_from_absorbance",
    "calculate_temperature_results",
    "calculate_temperature_results_direct",
    # Statistical functions
    "calculate_press",
    "calculate_q2",
    "calculate_error_metrics",
    "calculate_akaike_weights",
    "bootstrap_confidence_intervals",
    "analyze_residuals",
    # Mechanism analysis
    "determine_adsorption_mechanism",
    "check_mechanism_consistency",
    "detect_common_errors",
    # Thermodynamic calculations
    "calculate_thermodynamic_parameters",
    "interpret_thermodynamics",
    "calculate_arrhenius_parameters",
    "calculate_activity_coefficient_davies",
    # Separation factor
    "calculate_separation_factor",
    "interpret_separation_factor",
    # Data quality
    "assess_data_quality",
    "recommend_best_models",
    "detect_replicates",
    # Plotting helpers
    "create_residual_plots",
    "create_dual_axis_plot",
    # Uncertainty propagation
    "propagate_calibration_uncertainty",
    # Export utilities
    "convert_df_to_csv",
    "convert_df_to_excel",
    # Session state helpers
    "get_current_study_state",
    # UI helpers
    "EPSILON_DIV",
    "validate_required_params",
    "display_results_table",
]

CONFIDENCE_LEVEL = 0.95  # Default 95% confidence intervals


# =============================================================================
# COLUMN NAME STANDARDIZATION
# =============================================================================
COLUMN_SYNONYMS = {
    # --- Direct concentration columns (must stay distinct) ---
    "C0": [
        "c0",
        "c_initial",
        "initial_concentration",
        "concentration_initiale",
        "concentrationinitiale",
        "concentrationinitialec0",
        "concentration_initiale_c0",
        "concentration_initiale",
        "c0_mg_l",
        "c0_mg_liter",
    ],
    "Ce": [
        "ce",
        "c_e",
        "equilibrium_concentration",
        "concentration_equilibre",
        "concentrationequilibre",
        "concentration_equilibrium",
        "c_eq",
        "ceq",
        "ce_mg_l",
        "ce_mg_liter",
    ],
    "Ct": [
        "ct",
        "c_t",
        "concentration_t",
        "concentrationtime",
        "concentration_temps",
        "c_temps",
        "ct_mg_l",
        "ct_mg_liter",
    ],
    # --- Absorbance-mode generic concentration column ---
    "Concentration": [
        "conc",
        "concentration",
        "c",
    ],
    "Absorbance": [
        "abs",
        "absorb",
        "absorbance",
        "a",
        "abseq",
        "absorbt",
        "absorbancet",
        "absorbanceequilibre",
        "absorbance_equilibre",
        "abs_eq",
        "abst",
        "abs_t",
        "optical_density",
        "od",
    ],
    "Time": [
        "time",
        "temps",
        "tempsmin",
        "t_min",
        "time_min",
        "t",
        "contact_time",
        "reaction_time",
    ],
    "Mass": [
        "m",
        "mass",
        "masse",
        "masseadsorbant",
        "masseadsorbantg",
        "masse_adsorbant_g",
        "m_g",
        "adsorbent_mass",
        "weight",
        "w",
    ],
    "Volume": ["v", "volume", "vol", "solvolume", "solvolumel", "volume_l", "solution_volume"],
    "pH": ["ph", "ph_value", "acidity"],
    "Temperature": [
        "temp",
        "temperature",
        "temperaturec",
        "temperature_c",
        "t_celsius",
        "temp_c",
        "t_k",
        "temperature_k",
    ],
    "qe": ["qe", "qe_mg_g", "adsorption_capacity", "capacity", "q_eq"],
    "qt": ["qt", "qt_mg_g", "q_t", "capacity_time"],
}


@dataclass
class CalculationResult:
    """Standardized return type for data processing functions."""

    success: bool
    data: pd.DataFrame | None = None
    error: str | None = None


def standardize_column_name(col_name: str) -> str:
    """Standardize column names using known synonyms."""
    col_name_clean = re.sub(r"[^a-z0-9]+", "", col_name.strip().lower())

    for standard, variants in COLUMN_SYNONYMS.items():
        clean_variants = [
            v.lower().replace(" ", "").replace("_", "").replace("-", "") for v in variants
        ]
        if col_name_clean in clean_variants:
            return standard

    # Fuzzy matching
    all_variants = []
    variant_to_standard = {}
    for std, variants in COLUMN_SYNONYMS.items():
        for v in variants:
            clean_v = v.lower().replace(" ", "").replace("_", "").replace("-", "")
            all_variants.append(clean_v)
            variant_to_standard[clean_v] = std

    match = get_close_matches(col_name_clean, all_variants, n=1, cutoff=FUZZY_MATCH_CUTOFF)
    if match:
        return variant_to_standard[match[0]]

    return col_name


def standardize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standardize_column_name to all columns in a DataFrame."""
    df.columns = [standardize_column_name(col) for col in df.columns]
    return df


def validate_data_editor(df: pd.DataFrame | None, required_cols: list[str]) -> pd.DataFrame | None:
    """
    Validate data from Streamlit data editor for analysis.

    Checks that:
    - DataFrame is not None or empty
    - Required columns exist
    - Data contains valid numeric values
    - At least 3 data points are present

    Handles both period (.) and comma (,) as decimal separators.

    Parameters
    ----------
    df : pd.DataFrame or None
        Data from the data editor
    required_cols : list
        List of required column names

    Returns
    -------
    pd.DataFrame or None
        Validated DataFrame ready for analysis, or None if invalid
    """
    if df is None or df.empty:
        return None

    # Standardize column names
    df = standardize_dataframe_columns(df.copy())

    # Check required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return None

    # Extract only required columns
    df_subset = df[required_cols].copy()

    # Convert to numeric, handling comma decimals
    for col in required_cols:
        # If column is not numeric, check for comma decimals
        if not pd.api.types.is_numeric_dtype(df_subset[col]):
            # Convert to string and replace commas with periods
            str_col = df_subset[col].astype(str)
            # Only replace if the value looks like a decimal number with comma
            # (contains comma but not multiple commas which would indicate thousands separator)
            df_subset[col] = str_col.str.replace(",", ".", regex=False)

        # Convert to numeric
        df_subset[col] = pd.to_numeric(df_subset[col], errors="coerce")

    df_clean = df_subset.dropna()

    # Need at least 3 data points for meaningful analysis
    if len(df_clean) < MIN_DATA_POINTS:
        return None
    return df_clean


# =============================================================================
# DUAL UNIT CALCULATIONS
# =============================================================================
def calculate_removal_percentage(C0: float, Ce: float) -> float:
    """Calculate removal percentage: % Removal = [(C0 - Ce) / C0] × 100"""
    if C0 <= EPSILON_ZERO:
        return 0.0
    removal = ((C0 - Ce) / C0) * 100.0
    return max(0.0, min(100.0, removal))


def calculate_adsorption_capacity(C0: float, Ce: float, V: float, m: float) -> float:
    """Calculate adsorption capacity: q = (C0 - Ce) × V / m"""
    if m <= EPSILON_ZERO:
        return 0.0
    return max(0.0, (C0 - Ce) * V / m)


def calculate_Ce_from_absorbance(absorbance: float, slope: float, intercept: float) -> float:
    """Calculate equilibrium concentration from absorbance using calibration."""
    if abs(slope) < EPSILON_DIV:
        return 0.0
    Ce = (absorbance - intercept) / slope
    return max(0.0, Ce)


# =============================================================================
# PRESS STATISTIC (LEAVE-ONE-OUT CROSS-VALIDATION)
# =============================================================================
def calculate_press(
    model_func: Callable[..., Any],
    x_data: NDArray[np.floating[Any]],
    y_data: NDArray[np.floating[Any]],
    params: NDArray[np.floating[Any]],
    bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
) -> float:
    """
    Calculate PRESS (Predicted Residual Error Sum of Squares).

    Better than R² for model comparison in small datasets.
    Uses leave-one-out cross-validation.

    Parameters
    ----------
    model_func : callable
        Model function f(x, *params)
    x_data : np.ndarray
        Independent variable data
    y_data : np.ndarray
        Dependent variable data
    params : np.ndarray
        Fitted parameter values (used as initial guess)
    bounds : tuple, optional
        Parameter bounds for curve_fit

    Returns
    -------
    float
        PRESS statistic (lower is better)
    """
    n = len(x_data)
    press = 0.0

    for i in range(n):
        # Leave out point i
        x_train = np.delete(x_data, i)
        y_train = np.delete(y_data, i)

        try:
            # Refit model without point i
            if bounds:
                popt, _ = curve_fit(
                    model_func, x_train, y_train, p0=params, bounds=bounds, maxfev=5000
                )
            else:
                popt, _ = curve_fit(model_func, x_train, y_train, p0=params, maxfev=5000)

            # Predict left-out point
            y_pred = model_func(x_data[i], *popt)
            press += (y_data[i] - y_pred) ** 2

        except (RuntimeError, ValueError):
            # If fitting fails, use original params
            y_pred = model_func(x_data[i], *params)
            press += (y_data[i] - y_pred) ** 2

    return press


def calculate_temperature_results(
    temp_input: dict[str, Any], calib_params: dict[str, Any], include_uncertainty: bool = False
) -> "CalculationResult":
    """
    Calculate adsorption results at different temperatures.

    Unified function replacing duplicates in temperature_tab.py and thermodynamics_tab.py.

    Parameters
    ----------
    temp_input : dict
        Dictionary containing:
        - 'data': DataFrame with 'Temperature' and 'Absorbance' columns
        - 'params': dict with 'C0', 'm', 'V' values
    calib_params : dict
        Calibration parameters containing:
        - 'slope': calibration slope
        - 'intercept': calibration intercept
        - 'std_err_slope': (optional) standard error of slope
        - 'std_err_intercept': (optional) standard error of intercept
    include_uncertainty : bool, default=False
        If True, include uncertainty columns (Ce_error, qe_error).

    Returns
    -------
    CalculationResult
        Result with DataFrame containing temperature-dependent results.
    """
    df = temp_input["data"].copy()
    params = temp_input["params"]

    slope = calib_params["slope"]
    intercept = calib_params["intercept"]
    C0 = params["C0"]
    m = params["m"]
    V = params["V"]

    # Get calibration uncertainties only if needed
    slope_se = calib_params.get("std_err_slope", 0) if include_uncertainty else 0
    intercept_se = calib_params.get("std_err_intercept", 0) if include_uncertainty else 0

    results = []
    for _, row in df.iterrows():
        T_C = row["Temperature"]
        abs_val = row["Absorbance"]

        Ce = calculate_Ce_from_absorbance(abs_val, slope, intercept)
        qe = calculate_adsorption_capacity(C0, Ce, V, m)
        removal = calculate_removal_percentage(C0, Ce)

        result_row = {
            "Temperature_C": T_C,
            "Temperature_K": T_C + 273.15,
            "Ce_mgL": Ce,
            "qe_mg_g": qe,
            "removal_%": removal,
        }

        if include_uncertainty:
            _, Ce_se = propagate_calibration_uncertainty(
                abs_val, slope, intercept, slope_se, intercept_se
            )
            qe_error = (V / m) * Ce_se if m > 0 else 0
            result_row["Ce_error"] = Ce_se
            result_row["qe_error"] = qe_error

        results.append(result_row)

    if not results:
        return CalculationResult(success=False, error="No valid data points to calculate results.")

    results_df = pd.DataFrame(results).sort_values("Temperature_C")
    return CalculationResult(success=True, data=results_df)


def calculate_temperature_results_direct(
    temp_input: dict[str, Any], include_uncertainty: bool = False
) -> "CalculationResult":
    """
    Calculate adsorption results at different temperatures from direct Ce input.

    This function bypasses calibration and uses Ce values directly from published data.

    Parameters
    ----------
    temp_input : dict
        Dictionary containing:
        - 'data': DataFrame with 'Temperature' and 'Ce' columns
        - 'params': dict with 'C0', 'm', 'V' values
    include_uncertainty : bool, default=False
        If True, include uncertainty columns (set to 0 for direct input).

    Returns
    -------
    CalculationResult
        Result with DataFrame containing temperature-dependent results.
    """
    df = temp_input["data"].copy()
    params = temp_input["params"]

    C0 = params["C0"]
    m = params["m"]
    V = params["V"]

    results = []
    for _, row in df.iterrows():
        T = row["Temperature"]
        Ce = row["Ce"]

        if Ce > C0:
            continue  # Skip invalid data points

        qe = calculate_adsorption_capacity(C0, Ce, V, m)
        removal = calculate_removal_percentage(C0, Ce)

        # Determine if temperature is in Celsius or Kelvin
        # Assume Kelvin if T > 200 (no realistic Celsius would be that high for adsorption)
        if T > 200:
            T_C = T - 273.15
            T_K = T
        else:
            T_C = T
            T_K = T + 273.15

        result_row = {
            "Temperature_C": T_C,
            "Temperature_K": T_K,
            "Ce_mgL": Ce,
            "qe_mg_g": qe,
            "removal_%": removal,
        }

        if include_uncertainty:
            result_row["Ce_error"] = 0.0
            result_row["qe_error"] = 0.0

        results.append(result_row)

    if not results:
        return CalculationResult(success=False, error="No valid data points. Ensure Ce ≤ C0.")

    results_df = pd.DataFrame(results).sort_values("Temperature_C")
    return CalculationResult(success=True, data=results_df)


def calculate_q2(press: float, y_data: np.ndarray) -> float:
    """
    Calculate Q² (predictive R²) from PRESS.

    Q² = 1 - PRESS / SS_tot

    Interpretation:
    - Q² > 0.9: Excellent predictive ability
    - Q² > 0.7: Good predictive ability
    - Q² > 0.5: Acceptable predictive ability
    - Q² < 0.5: Poor predictive ability

    Parameters
    ----------
    press : float
        PRESS statistic
    y_data : np.ndarray
        Original dependent variable data

    Returns
    -------
    float
        Q² value
    """
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    if ss_tot < EPSILON_DIV:
        return 0.0
    return 1 - press / ss_tot


# =============================================================================
# MECHANISM CONSISTENCY CHECKER
# =============================================================================
def check_mechanism_consistency(study_state: dict[str, Any]) -> dict[str, Any]:
    """
    Check consistency between different analysis components.

    Detects conflicts such as:
    1. Kinetic model vs isotherm model inconsistency
    2. Temperature effect vs ΔH° sign mismatch
    3. IPD stages vs expected diffusion behavior
    4. Separation factor vs isotherm shape

    Parameters
    ----------
    study_state : dict
        Current study state containing all analysis results

    Returns
    -------
    dict with:
        - 'status': 'consistent', 'minor_issues', or 'conflicts'
        - 'color': 'green', 'yellow', or 'red'
        - 'checks': list of individual check results
        - 'interpretation': overall interpretation
        - 'suggestions': list of suggestions
    """
    checks = []
    conflicts = 0
    minor_issues = 0

    iso_models = study_state.get("isotherm_models_fitted", {})
    kin_models = study_state.get("kinetic_models_fitted", {})
    thermo_params = study_state.get("thermo_params", {})

    # Check 2: PSO/PFO dominance vs isotherm type
    pso_result = kin_models.get("PSO", {})
    pfo_result = kin_models.get("PFO", {})
    langmuir_result = iso_models.get("Langmuir", {})
    freundlich_result = iso_models.get("Freundlich", {})

    if pso_result.get("converged") and pfo_result.get("converged"):
        pso_r2 = pso_result.get("adj_r_squared", pso_result.get("r_squared", 0))
        pfo_r2 = pfo_result.get("adj_r_squared", pfo_result.get("r_squared", 0))
        kinetic_dominant = "PSO" if pso_r2 > pfo_r2 else "PFO"

        if langmuir_result.get("converged") and freundlich_result.get("converged"):
            lang_r2 = langmuir_result.get("adj_r_squared", langmuir_result.get("r_squared", 0))
            freund_r2 = freundlich_result.get(
                "adj_r_squared", freundlich_result.get("r_squared", 0)
            )
            iso_dominant = "Langmuir" if lang_r2 > freund_r2 else "Freundlich"

            # Note: PSO-Langmuir association is commonly reported but NOT mechanistically valid
            # This check flags potential inconsistency for user awareness, not mechanism confirmation
            if (
                kinetic_dominant == "PSO"
                and iso_dominant == "Freundlich"
                and (freund_r2 - lang_r2) > 0.05
            ):
                checks.append(
                    {
                        "name": "Kinetic-Isotherm Consistency",
                        "status": "minor",
                        "message": "PSO dominance with Freundlich preference is unusual (not a mechanistic concern)",
                        "severity": "medium",
                    }
                )
                minor_issues += 1
            else:
                checks.append(
                    {
                        "name": "Kinetic-Isotherm Consistency",
                        "status": "consistent",
                        "message": f"Kinetic ({kinetic_dominant}) and isotherm ({iso_dominant}) models are consistent",
                        "severity": "none",
                    }
                )

    # Check 3: Temperature effect vs ΔH° sign
    if thermo_params:
        delta_H = thermo_params.get("delta_H", 0)
        temp_effect = study_state.get(
            "temperature_effect"
        )  # Could be 'increases', 'decreases', or None

        if temp_effect:
            if delta_H > 0 and temp_effect == "decreases":
                checks.append(
                    {
                        "name": "Temperature-ΔH° Consistency",
                        "status": "conflict",
                        "message": "Endothermic (ΔH° > 0) should increase with temperature, but capacity decreases",
                        "severity": "high",
                    }
                )
                conflicts += 1
            elif delta_H < 0 and temp_effect == "increases":
                checks.append(
                    {
                        "name": "Temperature-ΔH° Consistency",
                        "status": "conflict",
                        "message": "Exothermic (ΔH° < 0) should decrease with temperature, but capacity increases",
                        "severity": "high",
                    }
                )
                conflicts += 1
            else:
                checks.append(
                    {
                        "name": "Temperature-ΔH° Consistency",
                        "status": "consistent",
                        "message": "Temperature effect matches thermodynamic prediction",
                        "severity": "none",
                    }
                )

    # Check 4: Separation factor (RL) vs isotherm shape
    if langmuir_result.get("converged"):
        RL = langmuir_result.get("RL")
        if RL is not None:
            RL_mean = np.mean(RL) if hasattr(RL, "__len__") else RL

            # Check if Freundlich 1/n is consistent
            if freundlich_result.get("converged"):
                n_inv = freundlich_result.get("params", {}).get("n_inv", 0.5)

                # RL < 1 (favorable) should correspond to 0 < 1/n < 1
                if RL_mean < 1 and n_inv > 1:
                    checks.append(
                        {
                            "name": "RL vs Freundlich Consistency",
                            "status": "minor",
                            "message": f"RL={RL_mean:.3f} suggests favorable adsorption, but 1/n={n_inv:.2f} > 1 is unusual",
                            "severity": "medium",
                        }
                    )
                    minor_issues += 1
                else:
                    checks.append(
                        {
                            "name": "RL vs Freundlich Consistency",
                            "status": "consistent",
                            "message": f"Separation factor (RL={RL_mean:.3f}) consistent with Freundlich behavior",
                            "severity": "none",
                        }
                    )

    # Check 5: High R² without confidence intervals
    for model_name, result in {**iso_models, **kin_models}.items():
        if result and result.get("converged"):
            r2 = result.get("r_squared", 0)
            ci_95 = result.get("ci_95", {})

            if r2 > 0.99 and not ci_95:
                checks.append(
                    {
                        "name": f"{model_name} R² Reporting",
                        "status": "minor",
                        "message": f"High R² ({r2:.4f}) reported without confidence intervals",
                        "severity": "low",
                    }
                )
                minor_issues += 1

    # Determine overall status
    if conflicts > 0:
        status = "conflicts"
        color = "red"
        interpretation = f"{conflicts} conflict(s) detected - results may be unreliable"
    elif minor_issues > 0:
        status = "minor_issues"
        color = "yellow"
        interpretation = f"{minor_issues} minor issue(s) found - review recommended"
    else:
        status = "consistent"
        color = "green"
        interpretation = "All mechanism indicators are internally consistent"

    # Generate suggestions
    suggestions = []
    if conflicts > 0:
        suggestions.append("Review experimental conditions and data quality")
        suggestions.append("Consider alternative models that may better explain the data")
    if minor_issues > 0:
        suggestions.append("Document any inconsistencies in your report")
        suggestions.append("Consider additional experiments to resolve ambiguities")
    if status == "consistent":
        suggestions.append("Results are ready for reporting")

    return {
        "status": status,
        "color": color,
        "checks": checks,
        "conflicts": conflicts,
        "minor_issues": minor_issues,
        "interpretation": interpretation,
        "suggestions": suggestions,
    }


def display_results_table(
    data: dict | pd.DataFrame,
    title: str | None = None,
    use_container_width: bool = True,
    hide_index: bool = True,
    column_config: dict | None = None,
    height: int | str | None = None,
) -> None:
    """
    Display a formatted results table with consistent styling.

    This helper eliminates the 105+ occurrences of duplicate DataFrame
    display code across tabs.

    Parameters
    ----------
    data : dict or pd.DataFrame
        Data to display. If dict, will be converted to DataFrame.
        Dict format: {"Column1": [values], "Column2": [values]}
    title : str, optional
        Section title to display above table
    use_container_width : bool, default=True
        Use full container width
    hide_index : bool, default=True
        Hide DataFrame index
    column_config : dict, optional
        Streamlit column configuration for custom formatting
    height : int, optional
        Fixed table height in pixels

    Example
    -------
    >>> display_results_table(
    ...     {"Parameter": ["qm", "KL"], "Value": [50.2, 0.15]},
    ...     title="Model Parameters"
    ... )
    """

    # Convert dict to DataFrame if needed
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data

    # Display title if provided
    if title:
        st.subheader(title)

    # Display table with consistent styling
    kwargs = dict(
        use_container_width=use_container_width,
        hide_index=hide_index,
        column_config=column_config or {},
    )
    if height is not None:
        kwargs["height"] = height

    st.dataframe(df, **kwargs)


# =============================================================================
# METHODOLOGICAL ERROR DETECTION
# =============================================================================
def detect_common_errors(study_state: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Detect common methodological errors in adsorption studies.

    Checks:
    1. Using linearized models when non-linear is statistically better
    2. Reporting R² > 0.99 without confidence intervals
    3. ΔG° outside reasonable range (-40 to +10 kJ/mol)
    4. ΔH° sign inconsistent with temperature effect
    5. Insufficient data points for model complexity
    6. Heteroscedastic residuals without weighted regression
    7. High multicollinearity in multi-parameter models

    Parameters
    ----------
    study_state : dict
        Current study state

    Returns
    -------
    list of dict
        Each dict contains: severity, type, message, recommendation
    """
    errors = []

    # Check 1: ΔG° range
    thermo = study_state.get("thermo_params", {})
    if thermo:
        delta_G = thermo.get("delta_G", [])
        if isinstance(delta_G, list | np.ndarray):
            for dG in delta_G:
                if dG > 10 or dG < -60:
                    errors.append(
                        {
                            "severity": "HIGH",
                            "type": "Thermodynamic",
                            "message": f"ΔG° = {dG:.2f} kJ/mol is outside typical range (-40 to +10).",
                            "recommendation": "Verify Kd calculation method and units.",
                        }
                    )

    # Check 2: Insufficient data points
    iso_results = study_state.get("isotherm_results")
    if iso_results is not None:
        n_points = len(iso_results) if hasattr(iso_results, "__len__") else 0
        iso_models = study_state.get("isotherm_models_fitted", {})

        for model_name, result in iso_models.items():
            if result and result.get("converged"):
                n_params = result.get("num_params", 2)
                if n_points < n_params * 3:
                    errors.append(
                        {
                            "severity": "MEDIUM",
                            "type": "Statistical",
                            "message": f"{model_name}: {n_points} points for {n_params} parameters may be insufficient.",
                            "recommendation": f"Recommend ≥{n_params * 3} data points for reliable fitting.",
                        }
                    )

    # Check 3: High R² without CI
    for models_key in ["isotherm_models_fitted", "kinetic_models_fitted"]:
        models = study_state.get(models_key, {})
        for model_name, result in models.items():
            if result and result.get("converged"):
                r2 = result.get("r_squared", 0)
                ci = result.get("ci_95", {})

                if r2 > 0.99 and not ci:
                    errors.append(
                        {
                            "severity": "MEDIUM",
                            "type": "Reporting",
                            "message": f"{model_name}: R² = {r2:.4f} reported without confidence intervals.",
                            "recommendation": "Include 95% CI for all parameters to support R² claims.",
                        }
                    )

    # Check 4: Heteroscedasticity detection
    for models_key in ["isotherm_models_fitted", "kinetic_models_fitted"]:
        models = study_state.get(models_key, {})
        for model_name, result in models.items():
            if result and result.get("converged"):
                residuals = result.get("residuals")
                y_pred = result.get("y_pred")

                if residuals is not None and y_pred is not None and len(residuals) > 5:
                    try:
                        # Correlation between |residuals| and predicted values
                        # High correlation suggests variance changes with magnitude (heteroscedasticity)
                        corr = np.corrcoef(np.abs(residuals), y_pred)[0, 1]

                        if not np.isnan(corr) and abs(corr) > 0.5:
                            errors.append(
                                {
                                    "severity": "MEDIUM",
                                    "type": "Statistical",
                                    "message": f"{model_name}: Possible heteroscedasticity detected (|r| = {abs(corr):.2f}).",
                                    "recommendation": "Consider weighted least squares or data transformation.",
                                }
                            )
                    except Exception as e:
                        logger.debug(f"Correlation calculation failed: {e}")

    # Check 5: Linear vs Non-linear comparison
    iso_linear = study_state.get("isotherm_linear_results", {})
    iso_nonlinear = study_state.get("isotherm_models_fitted", {})

    for model_name in ["Langmuir", "Freundlich"]:
        if model_name in iso_linear and model_name in iso_nonlinear:
            linear_r2 = iso_linear[model_name].get("r_squared", 0)
            nonlinear_r2 = iso_nonlinear[model_name].get("r_squared", 0)

            if nonlinear_r2 - linear_r2 > 0.05:
                errors.append(
                    {
                        "severity": "MEDIUM",
                        "type": "Methodology",
                        "message": f"{model_name}: Non-linear (R²={nonlinear_r2:.4f}) significantly better than linear (R²={linear_r2:.4f}).",
                        "recommendation": "Use non-linear regression for parameter estimation.",
                    }
                )

    # Check 6: ΔH° sign vs temperature effect
    temp_results = study_state.get("temp_effect_results")
    if thermo and temp_results is not None:
        delta_H = thermo.get("delta_H", 0)

        # Try to determine temperature effect from results
        if isinstance(temp_results, pd.DataFrame) and len(temp_results) >= 2:
            # Check if qe/removal increases or decreases with temperature
            if "T" in temp_results.columns or "Temperature" in temp_results.columns:
                temp_col = "T" if "T" in temp_results.columns else "Temperature"
                qe_col = None
                for col in ["qe", "Removal", "removal", "Removal_%"]:
                    if col in temp_results.columns:
                        qe_col = col
                        break

                if qe_col:
                    # Simple trend: compare first and last values
                    sorted_df = temp_results.sort_values(temp_col)
                    first_qe = sorted_df[qe_col].iloc[0]
                    last_qe = sorted_df[qe_col].iloc[-1]

                    if last_qe > first_qe * 1.1:  # Increases with T (>10% increase)
                        temp_trend = "increases"
                    elif last_qe < first_qe * 0.9:  # Decreases with T (>10% decrease)
                        temp_trend = "decreases"
                    else:
                        temp_trend = None  # No clear trend

                    # Check consistency
                    if temp_trend:
                        if delta_H > 10 and temp_trend == "decreases":
                            errors.append(
                                {
                                    "severity": "HIGH",
                                    "type": "Thermodynamic",
                                    "message": f"Endothermic reaction (ΔH° = {delta_H:.1f} kJ/mol) should increase with temperature, but capacity decreases.",
                                    "recommendation": "Review thermodynamic calculations or check for experimental errors.",
                                }
                            )
                        elif delta_H < -10 and temp_trend == "increases":
                            errors.append(
                                {
                                    "severity": "HIGH",
                                    "type": "Thermodynamic",
                                    "message": f"Exothermic reaction (ΔH° = {delta_H:.1f} kJ/mol) should decrease with temperature, but capacity increases.",
                                    "recommendation": "Review thermodynamic calculations or check for experimental errors.",
                                }
                            )

    # Check 7: Parameter uncertainty ratio (proxy for multicollinearity)
    for models_key in ["isotherm_models_fitted", "kinetic_models_fitted"]:
        models = study_state.get(models_key, {})
        for model_name, result in models.items():
            if result and result.get("converged"):
                n_params = result.get("num_params", 2)
                ci_95 = result.get("ci_95", {})
                params = result.get("params", {})

                # Only check 3+ parameter models
                if n_params >= 3 and ci_95 and params:
                    for param_name, param_value in params.items():
                        if param_name in ci_95 and param_value != 0:
                            ci_range = ci_95[param_name]
                            if isinstance(ci_range, list | tuple) and len(ci_range) == 2:
                                ci_width = abs(ci_range[1] - ci_range[0])
                                relative_width = (
                                    ci_width / abs(param_value)
                                    if param_value != 0
                                    else float("inf")
                                )

                                # If CI width > 100% of parameter value, flag it
                                if relative_width > 1.0:
                                    errors.append(
                                        {
                                            "severity": "LOW",
                                            "type": "Statistical",
                                            "message": f"{model_name}: Parameter {param_name} has very wide CI (±{relative_width * 50:.0f}%).",
                                            "recommendation": "Consider simpler model or more data points.",
                                        }
                                    )
                                    break  # Only report once per model
    return errors


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS (WITH CACHING)
# =============================================================================


# Import model registry for cached bootstrap (avoid circular import at function level)
def _get_model_registry() -> tuple[
    dict[str, Callable[..., Any]], Callable[[str], Callable[..., Any] | None]
]:
    """Lazy import of model registry to avoid circular imports."""
    from .models import _MODEL_REGISTRY, get_model_by_name

    return _MODEL_REGISTRY, get_model_by_name


def _bootstrap_core(
    model_func: Callable[..., Any],
    x_data: NDArray[np.floating[Any]],
    y_data: NDArray[np.floating[Any]],
    params: NDArray[np.floating[Any]],
    n_bootstrap: int,
    confidence: float,
    early_stopping: bool,
    random_seed: int | None = None,
    progress_callback: Callable[..., Any] | None = None,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """
    Core bootstrap CI calculation.

    This is the single implementation used by cached, non-cached, and
    progress-reporting paths.

    Parameters
    ----------
    model_func : callable
        Model function f(x, *params)
    x_data, y_data : np.ndarray
        Observed data
    params : np.ndarray
        Fitted parameter values (used as initial guesses for resampled fits)
    n_bootstrap : int
        Number of bootstrap iterations
    confidence : float
        Confidence level (e.g. 0.95)
    early_stopping : bool
        Stop early when parameter means converge
    random_seed : int, optional
        If provided, seed the RNG for reproducibility
    progress_callback : callable, optional
        Called as ``callback(current, total, message)`` every 50 iterations
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_data = len(x_data)
    n_params = len(params)
    bootstrap_params = np.zeros((n_bootstrap, n_params))

    # Calculate residuals from original fit
    y_pred = model_func(x_data, *params)
    residuals = y_data - y_pred

    successful_iterations = 0
    convergence_check_interval = min(100, n_bootstrap // 5)
    prev_means = None

    for i in range(n_bootstrap):
        if progress_callback and i % 50 == 0:
            progress_callback(i, n_bootstrap, f"Bootstrap iteration {i}/{n_bootstrap}")

        # Residual resampling
        bootstrap_residuals = np.random.choice(residuals, size=n_data, replace=True)
        y_bootstrap = y_pred + bootstrap_residuals

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                popt, _ = curve_fit(model_func, x_data, y_bootstrap, p0=params, maxfev=2000)
            bootstrap_params[successful_iterations] = popt
            successful_iterations += 1

            # Early stopping check
            if (
                early_stopping
                and successful_iterations > 0
                and successful_iterations % convergence_check_interval == 0
            ):
                current_means = np.mean(bootstrap_params[:successful_iterations], axis=0)
                if prev_means is not None:
                    relative_change = np.abs(current_means - prev_means) / (
                        np.abs(prev_means) + EPSILON_DIV
                    )
                    if np.all(relative_change < 0.01):
                        break
                prev_means = current_means

        except (RuntimeError, ValueError) as e:
            logger.debug(f"Curve fit failed in bootstrap iteration: {e}")
            continue

    if successful_iterations < BOOTSTRAP_MIN_SUCCESS:
        return np.full(n_params, np.nan), np.full(n_params, np.nan)

    # Calculate percentile CI
    alpha = (1 - confidence) / 2
    valid_params = bootstrap_params[:successful_iterations]
    ci_lower = np.percentile(valid_params, alpha * 100, axis=0)
    ci_upper = np.percentile(valid_params, (1 - alpha) * 100, axis=0)

    return ci_lower, ci_upper


def _bootstrap_cached_impl(
    model_name: str,
    x_data_tuple: tuple,
    y_data_tuple: tuple,
    params_tuple: tuple,
    n_bootstrap: int,
    confidence: float,
    early_stopping: bool,
    data_hash: int,
) -> tuple[tuple, tuple]:
    """
    Cached bootstrap implementation with hashable arguments.

    Returns tuples instead of arrays to ensure proper caching behavior.
    """
    _, get_model_by_name = _get_model_registry()
    model_func = get_model_by_name(model_name)

    if model_func is None:
        n_params = len(params_tuple)
        return tuple([float("nan")] * n_params), tuple([float("nan")] * n_params)

    x_data = np.array(x_data_tuple)
    y_data = np.array(y_data_tuple)
    params = np.array(params_tuple)

    # Use data_hash as seed for reproducibility
    ci_lower, ci_upper = _bootstrap_core(
        model_func,
        x_data,
        y_data,
        params,
        n_bootstrap,
        confidence,
        early_stopping,
        random_seed=abs(data_hash) % (2**31),
    )

    return tuple(ci_lower.tolist()), tuple(ci_upper.tolist())


# Apply caching if Streamlit is available
if _STREAMLIT_AVAILABLE:
    _bootstrap_cached = st.cache_data(show_spinner="Running bootstrap analysis...")(
        _bootstrap_cached_impl
    )
else:
    _bootstrap_cached = _bootstrap_cached_impl


def bootstrap_confidence_intervals(
    model_func: Callable[..., Any],
    x_data: NDArray[np.floating[Any]],
    y_data: NDArray[np.floating[Any]],
    params: NDArray[np.floating[Any]],
    n_bootstrap: int = BOOTSTRAP_DEFAULT_ITERATIONS,
    confidence: float = 0.95,
    progress_callback: Callable[..., Any] | None = None,
    early_stopping: bool = True,
    use_cache: bool = True,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """
    Calculate bootstrap confidence intervals for model parameters.

    Uses residual resampling for more robust CI estimation. Results are
    cached when possible to avoid recomputation on Streamlit reruns.

    Parameters
    ----------
    model_func : callable
        Model function f(x, *params)
    x_data : np.ndarray
        Independent variable data
    y_data : np.ndarray
        Dependent variable data
    params : np.ndarray
        Fitted parameter values from initial fit
    n_bootstrap : int
        Number of bootstrap iterations (default 500)
    confidence : float
        Confidence level (default 0.95 for 95% CI)
    progress_callback : callable, optional
        Function to report progress: callback(current, total, message)
        Note: When using cache, progress is not shown on cache hits.
    early_stopping : bool
        Whether to stop early if parameters converge (default True)
    use_cache : bool
        Whether to use caching (default True)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (ci_lower, ci_upper) - Lower and upper confidence interval bounds
    """
    # Try to find model name for caching
    model_name = None
    if use_cache and _STREAMLIT_AVAILABLE:
        model_registry, _ = _get_model_registry()
        for name, func in model_registry.items():
            if func is model_func:
                model_name = name
                break

    # Use cached version if model is registered and no progress callback needed
    # (cached version doesn't support progress callbacks, but shows a spinner)
    if model_name is not None and progress_callback is None:
        x_tuple = tuple(np.asarray(x_data).round(10).tolist())
        y_tuple = tuple(np.asarray(y_data).round(10).tolist())
        params_tuple = tuple(np.asarray(params).round(10).tolist())

        # Create a deterministic hash for reproducibility
        data_hash = hash((x_tuple, y_tuple, params_tuple))

        ci_lower_tuple, ci_upper_tuple = _bootstrap_cached(
            model_name,
            x_tuple,
            y_tuple,
            params_tuple,
            n_bootstrap,
            confidence,
            early_stopping,
            data_hash,
        )

        return np.array(ci_lower_tuple), np.array(ci_upper_tuple)

    # Use non-cached version (with optional progress callback)
    return _bootstrap_core(
        model_func,
        np.asarray(x_data),
        np.asarray(y_data),
        np.asarray(params),
        n_bootstrap,
        confidence,
        early_stopping,
        progress_callback=progress_callback,
    )


# =============================================================================
# ERROR METRICS AND MODEL COMPARISON
# =============================================================================
def calculate_error_metrics(
    y_obs: np.ndarray, y_pred: np.ndarray, n_params: int = 2
) -> dict[str, float]:
    """
    Calculate comprehensive error metrics for model evaluation.

    Parameters
    ----------
    y_obs : np.ndarray
        Observed values
    y_pred : np.ndarray
        Predicted values
    n_params : int
        Number of model parameters (default 2)

    Returns
    -------
    Dict[str, float]
        Dictionary containing: r_squared, adj_r_squared, rmse, mae,
        chi_squared, chi_squared_reduced, aic, aicc, bic, sse, sst, residuals
    """
    n = len(y_obs)
    residuals = y_obs - y_pred

    # Sum of squares
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)

    # R² and Adjusted R²
    r_squared = 1 - ss_res / ss_tot if ss_tot > EPSILON_DIV else 0
    adj_r_squared = (
        1 - (1 - r_squared) * (n - 1) / (n - n_params - 1) if n > n_params + 1 else r_squared
    )

    # RMSE, MAE
    rmse = np.sqrt(ss_res / n)
    mae = np.mean(np.abs(residuals))

    # Chi-squared (reduced)
    y_pred_safe = np.where(np.abs(y_pred) < EPSILON_DIV, EPSILON_DIV, y_pred)
    chi_sq = np.sum(residuals**2 / np.abs(y_pred_safe))
    chi_sq_reduced = chi_sq / (n - n_params) if n > n_params else chi_sq

    # AIC, AICc, BIC
    if ss_res > 0 and n > n_params:
        log_lik = -n / 2 * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)
        aic = -2 * log_lik + 2 * n_params
        aicc = (
            aic + (2 * n_params * (n_params + 1)) / (n - n_params - 1) if n > n_params + 1 else aic
        )
        bic = -2 * log_lik + n_params * np.log(n)
    else:
        aic = aicc = bic = np.inf

    return {
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "rmse": rmse,
        "mae": mae,
        "chi_squared": chi_sq,
        "chi_squared_reduced": chi_sq_reduced,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "sse": ss_res,
        "sst": ss_tot,
        "residuals": residuals,
    }


def calculate_akaike_weights(aic_values: list[float]) -> np.ndarray:
    """Calculate Akaike weights for model comparison."""
    aic_array = np.array(aic_values)
    valid_mask = np.isfinite(aic_array)

    if not np.any(valid_mask):
        return np.zeros(len(aic_values))

    aic_min = np.min(aic_array[valid_mask])
    delta_aic = aic_array - aic_min

    # Avoid overflow
    delta_aic = np.clip(delta_aic, 0, 700)

    weights = np.exp(-0.5 * delta_aic)
    weights[~valid_mask] = 0

    total = np.sum(weights)
    return weights / total if total > 0 else weights


def analyze_residuals(
    residuals: NDArray[np.floating[Any]], y_pred: NDArray[np.floating[Any]] | None = None
) -> dict[str, Any]:
    """Comprehensive residual analysis for model diagnostics (robust to constant data)."""

    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)

    # Base results (safe even for empty/constant arrays)
    results: dict[str, Any] = {
        "n": n,
        "mean": float(np.mean(r)) if n else 0.0,
        "std": float(np.std(r)) if n else 0.0,
    }

    if n < 3:
        # Not enough data for meaningful higher-order stats or normality tests
        results.update(
            {
                "skewness": np.nan,
                "kurtosis": np.nan,
                "normality_test": "Not applicable (n < 3)",
                "normality_stat": np.nan,
                "normality_p_value": np.nan,
                "normality_pass": False,
            }
        )
        return results

    # Detect constant (or nearly constant) residuals: avoids catastrophic cancellation + Shapiro warnings
    res_range = float(np.ptp(r))  # max - min
    is_constant = res_range < EPSILON_ZERO
    results["range"] = res_range
    results["is_constant"] = is_constant

    if is_constant:
        # Moments/normality tests are not meaningful when residuals have (near) zero range
        results.update(
            {
                "skewness": 0.0,
                "kurtosis": 0.0,
                "normality_test": "Not applicable (constant residuals)",
                "normality_stat": np.nan,
                "normality_p_value": np.nan,
                "normality_pass": False,
            }
        )
    else:
        # Skewness/kurtosis can emit RuntimeWarnings for nearly-identical data; guard it.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            results["skewness"] = float(stats.skew(r, bias=False))
            results["kurtosis"] = float(stats.kurtosis(r, bias=False))

        # Normality tests
        try:
            if n < 5000:
                stat, p_value = shapiro(r)
                results["normality_test"] = "Shapiro-Wilk"
            else:
                stat, p_value = normaltest(r)
                results["normality_test"] = "D'Agostino-Pearson"

            results["normality_stat"] = float(stat)
            results["normality_p_value"] = float(p_value)
            results["normality_pass"] = bool(p_value > 0.05)
        except Exception as e:
            logger.debug(f"Normality test failed: {e}")
            results.update(
                {
                    "normality_test": "Failed",
                    "normality_stat": np.nan,
                    "normality_p_value": np.nan,
                    "normality_pass": False,
                }
            )

    # Durbin-Watson test for autocorrelation
    if n > 2:
        diff_residuals = np.diff(r)
        ss_residuals = float(np.sum(r**2))
        if ss_residuals > 0:
            dw = float(np.sum(diff_residuals**2) / ss_residuals)
            results["durbin_watson"] = dw
            results["autocorrelation"] = (
                "positive" if dw < 1.5 else ("negative" if dw > 2.5 else "none")
            )

    # Heteroscedasticity check
    if y_pred is not None and len(y_pred) == n and not is_constant:
        try:
            corr = float(np.corrcoef(np.abs(r), np.asarray(y_pred, dtype=float))[0, 1])
            results["heteroscedasticity_corr"] = corr
            results["heteroscedasticity"] = bool(np.isfinite(corr) and abs(corr) > 0.3)
        except Exception as e:
            logger.debug(f"Heteroscedasticity check failed: {e}")

    return results


def create_residual_plots(
    residuals: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]] | None = None,
    x_data: NDArray[np.floating[Any]] | None = None,
) -> go.Figure:
    """
    Create comprehensive residual diagnostic plots.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals (observed - predicted)
    y_pred : np.ndarray, optional
        Predicted values for residual vs fitted plot
    x_data : np.ndarray, optional
        Independent variable for residual vs x plot

    Returns
    -------
    go.Figure
        Plotly figure with residual diagnostic plots
    """

    # Create 2x2 subplot
    fourth_plot_title = "Residuals vs X" if x_data is not None else "Residuals vs Order"
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Residuals vs Fitted",
            "Q-Q Plot",
            "Histogram of Residuals",
            fourth_plot_title,
        ),
    )

    n = len(residuals)

    # 1. Residuals vs Fitted (or vs index if y_pred not provided)
    x_axis = y_pred if y_pred is not None else np.arange(n)
    x_label = "Fitted Values" if y_pred is not None else "Index"

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=residuals,
            mode="markers",
            marker={"size": 8, "color": "#2E86AB"},
            name="Residuals",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # 2. Q-Q Plot
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))

    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_residuals,
            mode="markers",
            marker={"size": 8, "color": "#2E86AB"},
            name="Q-Q",
        ),
        row=1,
        col=2,
    )

    # Add reference line for Q-Q
    qq_min, qq_max = theoretical_quantiles.min(), theoretical_quantiles.max()
    res_mean, res_std = np.mean(residuals), np.std(residuals)
    fig.add_trace(
        go.Scatter(
            x=[qq_min, qq_max],
            y=[res_mean + res_std * qq_min, res_mean + res_std * qq_max],
            mode="lines",
            line={"color": "red", "dash": "dash"},
            name="Reference",
        ),
        row=1,
        col=2,
    )

    # 3. Histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=min(20, n // 2 + 1),
            marker_color="#2E86AB",
            opacity=0.7,
            name="Distribution",
        ),
        row=2,
        col=1,
    )

    # 4. Residuals vs X (if provided) or vs Order (to detect autocorrelation)
    if x_data is not None:
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=residuals,
                mode="markers",
                marker={"size": 8, "color": "#2E86AB"},
                name="Residuals vs X",
            ),
            row=2,
            col=2,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, n + 1),
                y=residuals,
                mode="lines+markers",
                marker={"size": 6, "color": "#2E86AB"},
                line={"width": 1},
                name="Order",
            ),
            row=2,
            col=2,
        )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)

    # Update layout
    fig.update_layout(
        height=600,
        width=800,
        showlegend=False,
        template=PLOT_TEMPLATE,
        title_text="Residual Diagnostics",
    )

    # Match subplot title annotations to house style (bold + house font)
    for ann in fig.layout.annotations:
        ann.update(
            text=f"<b>{ann.text}</b>",
            font={"size": 14, "family": FONT_FAMILY},
        )

    fig.update_xaxes(title_text=x_label, row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
    fig.update_xaxes(title_text="Residual Value", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Observation Order", row=2, col=2)
    fig.update_yaxes(title_text="Residuals", row=2, col=2)

    return fig


# =============================================================================
# DATA QUALITY ASSESSMENT
# =============================================================================
def assess_data_quality(data: pd.DataFrame, data_type: str = "isotherm") -> dict[str, Any]:
    """
    Assess data quality for adsorption analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    data_type : str
        'isotherm', 'kinetic', or 'calibration'

    Returns
    -------
    dict with quality metrics and recommendations
    """
    quality_score = 100
    issues = []

    n_points = len(data)

    # Minimum data points based on data type
    if data_type == "isotherm":
        min_required = 5
    elif data_type == "kinetic":
        min_required = 8
    elif data_type == "calibration":
        min_required = 5  # At least 5 points for calibration
    else:
        min_required = 5

    if n_points < min_required:
        penalty = (min_required - n_points) * 10
        quality_score -= penalty
        issues.append(f"Insufficient data points: {n_points} < {min_required} recommended")
    elif n_points < min_required + 2 and data_type in ["isotherm", "kinetic"]:
        # Only flag "lower end" for isotherm/kinetic where more points really help
        # Don't flag for calibration - 5-7 points is standard practice
        quality_score -= 5
        issues.append(f"Consider adding more data points: {n_points} (ideal >= {min_required + 3})")

    # Check for duplicates
    if data.duplicated().any():
        n_dups = data.duplicated().sum()
        quality_score -= n_dups * 5
        issues.append(f"{n_dups} duplicate rows detected")

    # Check numeric columns for outliers
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        values = data[col].dropna()
        if len(values) > 3:
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = ((values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)).sum()
            if outliers > 0:
                quality_score -= outliers * 3
                issues.append(f"{outliers} potential outliers in {col}")

    # Check for negative values where not expected
    if data_type == "isotherm":
        for col in ["Ce_mgL", "qe_mg_g", "C0_mgL"]:
            if col in data.columns and (data[col] < 0).any():
                quality_score -= 20
                issues.append(f"Negative values in {col}")

    quality_score = max(0, min(100, quality_score))

    # Use centralized grading system
    from .config import get_grade_from_score

    grade_info = get_grade_from_score(quality_score)

    return {
        "quality_score": quality_score,
        "grade": grade_info["grade"],
        "status": grade_info["status_type"],  # 'success', 'warning', 'error'
        "status_display": grade_info["status"],  # '✅ Excellent', etc.
        "label": grade_info["label"],
        "issues": issues,
        "n_points": n_points,
        "recommendation": "Good" if quality_score >= 80 else "Review data quality",
    }


def _score_isotherm_params(name: str, params: dict[str, Any]) -> tuple[float, list[str]]:
    """Score isotherm model parameters for physical reasonableness."""
    score = 0
    reasons = []

    if name == "Langmuir":
        qm = params.get("qm", 0)
        KL = params.get("KL", 0)
        if qm > 0 and KL > 0:
            score += 5
        if 0 < KL < 10:  # Typical range for favorable adsorption
            score += 5
            reasons.append("KL indicates favorable adsorption")

    elif name == "Freundlich":
        n = params.get("n", 0)
        KF = params.get("KF", 0)
        if KF > 0:
            score += 5
        if n > 1:  # n > 1 indicates favorable adsorption
            score += 5
            reasons.append("n > 1 indicates favorable adsorption")
        elif 0 < n < 1:
            reasons.append("n < 1 suggests cooperative adsorption")

    elif name == "Temkin":
        B1 = params.get("B1", 0)
        if B1 > 0:
            score += 5
        if B1 < 20:  # Low heat of adsorption - physical
            reasons.append("B1 suggests physical adsorption")
        else:
            reasons.append("B1 suggests chemical interaction")

    elif name == "Sips":
        qm = params.get("qm", 0)
        ns = params.get("ns", 0)
        if qm > 0:
            score += 5
        if 0 < ns <= 1:
            score += 5
            if abs(ns - 1) < 0.1:
                reasons.append("ns ≈ 1: Approaches Langmuir behavior")
            else:
                reasons.append("ns < 1: Heterogeneous surface")

    return score, reasons


def _score_kinetic_params(name: str, params: dict[str, Any]) -> tuple[float, list[str]]:
    """Score kinetic model parameters for physical reasonableness."""
    score = 0
    reasons = []

    if name == "Pseudo-first order" or name == "PFO":
        qe = params.get("qe", 0)
        k1 = params.get("k1", 0)
        if qe > 0 and k1 > 0:
            score += 5
        if 0.001 < k1 < 1:  # Typical range (1/min)
            score += 5
            reasons.append("Suggests physisorption mechanism")

    elif name == "Pseudo-second order" or name == "PSO":
        qe = params.get("qe", 0)
        k2 = params.get("k2", 0)
        if qe > 0 and k2 > 0:
            score += 5
        if k2 > 0:
            score += 5
            reasons.append("Suggests chemisorption mechanism")

    elif name == "Elovich":
        alpha = params.get("alpha", 0)
        beta = params.get("beta", 0)
        if alpha > 0 and beta > 0:
            score += 10
            reasons.append("Supports chemisorption on heterogeneous surface")

    elif name == "Weber-Morris" or name == "Intraparticle diffusion":
        kid = params.get("kid", 0)
        C = params.get("C", 0)
        if kid > 0:
            score += 5
        if C > 0:
            score += 5
            reasons.append("C > 0: Boundary layer effect present")
        else:
            reasons.append("C ≈ 0: Intraparticle diffusion is rate-limiting")

    return score, reasons


# =============================================================================
# MODEL RECOMMENDATION
# =============================================================================
def recommend_best_models(
    model_results: dict[str, dict[str, Any]], model_type: str = "isotherm", top_n: int = 3
) -> list[dict[str, Any]]:
    """
    Generate ranked model recommendations based on multiple criteria.

    Parameters
    ----------
    model_results : dict
        Dictionary of model names to result dictionaries
    model_type : str
        Type of model ('isotherm' or 'kinetic') - for context
    top_n : int
        Number of top models to return

    Uses a weighted scoring system considering:
    - Adjusted R² (30%)
    - AICc (25%)
    - RMSE (20%)
    - Residual diagnostics (15%)
    - Parameter reasonableness (10%)
    """
    rankings = []

    for name, result in model_results.items():
        if not result or not result.get("converged"):
            continue

        score = 0
        param_reasons: list[str] = []

        # Adjusted R² (higher is better)
        adj_r2 = result.get("adj_r_squared", result.get("r_squared", 0))
        score += adj_r2 * 30

        # RMSE (lower is better, normalized)
        rmse = result.get("rmse", np.inf)
        if np.isfinite(rmse) and rmse > 0:
            rmse_score = max(0, 1 - rmse / 100)  # Assumes typical RMSE < 100
            score += rmse_score * 20

        # AICc will be compared across models later
        aicc = result.get("aicc", result.get("aic", np.inf))

        # Residual check
        residuals = result.get("residuals")
        if residuals is not None:
            residual_analysis = analyze_residuals(np.array(residuals))
            if residual_analysis.get("normality_pass", False):
                score += 7.5
            if residual_analysis.get("autocorrelation") == "none":
                score += 7.5

        # Parameter reasonableness (10%) - MODEL TYPE SPECIFIC
        params = result.get("params", {})
        if model_type == "isotherm":
            param_score, param_reasons = _score_isotherm_params(name, params)
            score += param_score
        elif model_type == "kinetic":
            param_score, param_reasons = _score_kinetic_params(name, params)
            score += param_score

        # Get r_squared for display
        r_squared = result.get("r_squared", adj_r2)

        rankings.append(
            {
                "model": name,
                "score": score,
                "adj_r2": adj_r2,
                "adj_r_squared": adj_r2,  # Alias
                "r_squared": r_squared,
                "aicc": aicc,
                "rmse": rmse,
                "param_reasons": param_reasons,  # Store for rationale generation
                "result": result,
            }
        )

    # Normalize AICc scores
    if rankings:
        aicc_values = [r["aicc"] for r in rankings]
        aicc_weights = calculate_akaike_weights(aicc_values)
        for i, rank in enumerate(rankings):
            rank["score"] += aicc_weights[i] * 25
            rank["aicc_weight"] = aicc_weights[i]
            rank["aic_weight"] = aicc_weights[i]  # Alias
            # Calculate confidence as normalized score (0-100%)
            rank["confidence"] = min(100, rank["score"])
            # Generate rationale
            reasons = []
            if rank["adj_r2"] >= 0.99:
                reasons.append("Excellent fit (Adj-R² ≥ 0.99)")
            elif rank["adj_r2"] >= 0.95:
                reasons.append("Good fit (Adj-R² ≥ 0.95)")
            if aicc_weights[i] >= 0.5:
                reasons.append("Strong AIC support")
            elif aicc_weights[i] >= 0.2:
                reasons.append("Moderate AIC support")
            if rank["rmse"] < 5:
                reasons.append("Low RMSE")

            # Add model-type-specific parameter insights
            param_reasons = rank.get("param_reasons", [])
            reasons.extend(param_reasons)

            rank["rationale"] = "; ".join(reasons) if reasons else "Best available fit"

    # Sort by score
    rankings.sort(key=lambda x: x["score"], reverse=True)

    return rankings[:top_n]


# =============================================================================
# SEPARATION FACTOR (LANGMUIR)
# =============================================================================
def calculate_separation_factor(KL: float, C0: np.ndarray) -> np.ndarray:
    """Calculate Langmuir separation factor: RL = 1 / (1 + KL × C0)"""
    C0 = np.asarray(C0)
    return 1 / (1 + KL * C0)


def interpret_separation_factor(RL: np.ndarray) -> str:
    """Interpret separation factor values."""
    RL_min, RL_max = np.min(RL), np.max(RL)
    RL_mean = np.mean(RL)

    if RL_mean > 1:
        return f"Unfavorable (RL = {RL_mean:.4f} > 1)"
    elif np.isclose(RL_mean, 1, atol=0.01):
        return "Linear (RL ≈ 1)"
    elif RL_mean > 0:
        if RL_mean < 0.1:
            return f"Highly Favorable (RL = {RL_mean:.4f}, close to irreversible)"
        else:
            return f"Favorable (0 < RL = {RL_min:.4f}-{RL_max:.4f} < 1)"
    else:
        return "Irreversible (RL ≈ 0)"


# =============================================================================
# THERMODYNAMIC PARAMETERS
# =============================================================================
def calculate_thermodynamic_parameters(
    T_K: np.ndarray, Kd: np.ndarray, confidence_level: float = 0.95
) -> dict[str, Any]:
    """
    Calculate thermodynamic parameters from Van't Hoff analysis.

    Van't Hoff equation: ln(Kd) = -ΔH°/RT + ΔS°/R

    Parameters
    ----------
    T_K : np.ndarray
        Temperature in Kelvin
    Kd : np.ndarray
        Distribution coefficients
    confidence_level : float
        Confidence level for intervals (default 0.95)

    Returns
    -------
    dict
        Thermodynamic parameters including ΔH°, ΔS°, ΔG°, R², statistics
    """
    if len(T_K) < 2 or len(Kd) < 2:
        return {"success": False, "error": "Need at least 2 temperature points"}

    # Filter valid Kd values
    valid_mask = (Kd > 0) & np.isfinite(Kd) & np.isfinite(T_K) & (T_K > 0)
    T_K = T_K[valid_mask]
    Kd = Kd[valid_mask]

    if len(T_K) < 2:
        return {"success": False, "error": "Insufficient valid data points"}

    try:
        x = 1 / T_K  # 1/T
        y = np.log(Kd)  # ln(Kd)

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Calculate thermodynamic parameters
        # slope = -ΔH°/R, intercept = ΔS°/R
        delta_H = -slope * R_GAS_CONSTANT / 1000  # kJ/mol
        delta_S = intercept * R_GAS_CONSTANT  # J/(mol·K)

        # Calculate ΔG° at each temperature
        delta_G = delta_H - T_K * delta_S / 1000  # kJ/mol (as array)

        # Standard errors
        n = len(T_K)
        y_pred = slope * x + intercept
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n - 2) if n > 2 else 0

        x_mean = np.mean(x)
        ss_x = np.sum((x - x_mean) ** 2)

        se_slope = np.sqrt(mse / ss_x) if ss_x > 0 else 0
        se_intercept = np.sqrt(mse * (1 / n + x_mean**2 / ss_x)) if ss_x > 0 else 0

        # Propagate to thermodynamic parameters
        delta_H_se = se_slope * R_GAS_CONSTANT / 1000
        delta_S_se = se_intercept * R_GAS_CONSTANT

        # Calculate confidence intervals
        if n > 2:
            t_crit = t_dist.ppf((1 + confidence_level) / 2, n - 2)
            delta_H_ci = t_crit * delta_H_se
            delta_S_ci = t_crit * delta_S_se
        else:
            delta_H_ci = delta_H_se * 2  # Rough estimate
            delta_S_ci = delta_S_se * 2

        return {
            "success": True,
            "delta_H": delta_H,
            "delta_H_se": delta_H_se,
            "delta_H_ci": delta_H_ci,
            "delta_S": delta_S,
            "delta_S_se": delta_S_se,
            "delta_S_ci": delta_S_ci,
            "delta_G": delta_G,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_err": std_err,
            "temperatures": T_K,
            "Kd_values": Kd,
            "n_points": n,
            "confidence_level": confidence_level,
        }

    except Exception as e:
        logger.error(f"Failed to calculate thermodynamic parameters: {e}")
        return {"success": False, "error": str(e)}


def interpret_thermodynamics(delta_H: float, delta_S: float, delta_G: Any) -> dict[str, str]:
    """
    Interpret thermodynamic parameters.

    Parameters
    ----------
    delta_H : float
        Enthalpy change (kJ/mol)
    delta_S : float
        Entropy change (J/(mol·K))
    delta_G : array-like or dict
        Gibbs free energy at different temperatures

    Returns
    -------
    dict
        Interpretations for each parameter
    """
    interpretations: dict[str, str] = {}

    # Enthalpy interpretation
    if delta_H < 0:
        interpretations["enthalpy"] = f"Exothermic (ΔH° = {delta_H:.2f} kJ/mol < 0)"
        if abs(delta_H) < 40:
            interpretations["mechanism_H"] = "Physical adsorption (|ΔH°| < 40 kJ/mol)"
        else:
            interpretations["mechanism_H"] = "Chemical adsorption (|ΔH°| > 40 kJ/mol)"
    else:
        interpretations["enthalpy"] = f"Endothermic (ΔH° = {delta_H:.2f} kJ/mol > 0)"
        if abs(delta_H) < 40:
            interpretations["mechanism_H"] = "Physical adsorption (|ΔH°| < 40 kJ/mol)"
        else:
            interpretations["mechanism_H"] = "Chemical adsorption (|ΔH°| > 40 kJ/mol)"

    # Entropy interpretation
    if delta_S > 0:
        interpretations["entropy"] = (
            f"Increased disorder at interface (ΔS° = {delta_S:.2f} J/(mol·K) > 0)"
        )
    else:
        interpretations["entropy"] = (
            f"Decreased disorder at interface (ΔS° = {delta_S:.2f} J/(mol·K) < 0)"
        )

    # Gibbs free energy interpretation
    # Handle both dict and array-like inputs
    if delta_G is not None:
        G_values: list[float]
        if isinstance(delta_G, dict):
            G_values = list(delta_G.values())
        else:
            # numpy array or list
            G_values = list(np.asarray(delta_G).flatten())

        if len(G_values) > 0:
            if all(g < 0 for g in G_values):
                interpretations["spontaneity"] = "Spontaneous at all temperatures (ΔG° < 0)"
            elif all(g > 0 for g in G_values):
                interpretations["spontaneity"] = "Non-spontaneous at all temperatures (ΔG° > 0)"
            else:
                interpretations["spontaneity"] = "Temperature-dependent spontaneity"

            # Feasibility
            min_G = min(G_values)
            if min_G < -20:
                interpretations["feasibility"] = "Highly favorable adsorption"
            elif min_G < 0:
                interpretations["feasibility"] = "Favorable adsorption"
            else:
                interpretations["feasibility"] = "Unfavorable adsorption"

    return interpretations


def determine_adsorption_mechanism(
    delta_H: float,
    delta_G: Any = None,
    n_freundlich: float | None = None,
    RL: float | None = None,
) -> dict[str, Any]:
    """
    Determine adsorption mechanism from thermodynamic and isotherm data.

    Parameters
    ----------
    delta_H : float
        Enthalpy change (kJ/mol)
    delta_G : array-like, optional
        Gibbs free energy values (kJ/mol)
    n_freundlich : float, optional
        Freundlich exponent (1/n)
    RL : float, optional
        Langmuir separation factor

    Returns
    -------
    dict
        Mechanism determination with confidence, scores, evidence, and indicators
    """
    # Initialize scores for different mechanisms
    scores = {"Physical": 0.0, "Ion Exchange": 0.0, "Chemical": 0.0}

    evidence = []
    indicators = {}
    total_weight = 0

    # 1. Enthalpy-based determination (weight: 30%)
    abs_H = abs(delta_H)
    h_weight = 30
    total_weight += h_weight

    if abs_H < 20:
        scores["Physical"] += h_weight
        evidence.append(
            f"ΔH° indicates physical adsorption: |ΔH°| = {abs_H:.1f} < 20 kJ/mol (van der Waals forces)"
        )
        h_class = "Physical"
    elif abs_H < 40:
        scores["Physical"] += h_weight * 0.5
        scores["Chemical"] += h_weight * 0.5
        evidence.append(f"ΔH° indicates weak chemisorption: 20 < |ΔH°| = {abs_H:.1f} < 40 kJ/mol")
        h_class = "Weak Chemical"
    elif abs_H < 80:
        scores["Chemical"] += h_weight * 0.7
        scores["Physical"] += h_weight * 0.3
        evidence.append(f"ΔH° indicates hydrogen bonding: 40 < |ΔH°| = {abs_H:.1f} < 80 kJ/mol")
        h_class = "H-bonding"
    else:
        scores["Chemical"] += h_weight
        evidence.append(f"ΔH° indicates strong chemisorption: |ΔH°| = {abs_H:.1f} > 80 kJ/mol")
        h_class = "Chemical"

    indicators["ΔH° (kJ/mol)"] = {
        "value": delta_H,
        "classification": h_class,
        "criterion": "< 20: Physical, 20-40: Weak chem., 40-80: H-bond, > 80: Chemical",
        "confidence": "High" if abs_H < 20 or abs_H > 80 else "Medium",
    }

    # 3. Gibbs free energy (weight: 20%)
    if delta_G is not None:
        g_weight = 20
        total_weight += g_weight

        # Handle array or single value
        g_values = np.asarray(delta_G).flatten()
        avg_G = np.mean(g_values)

        if avg_G < -40:
            scores["Chemical"] += g_weight
            evidence.append(f"ΔG° indicates chemisorption: avg ΔG° = {avg_G:.1f} < -40 kJ/mol")
            g_class = "Chemical"
        elif avg_G < -20:
            scores["Physical"] += g_weight * 0.6
            scores["Chemical"] += g_weight * 0.4
            evidence.append(
                f"ΔG° indicates strong physisorption: -40 < avg ΔG° = {avg_G:.1f} < -20 kJ/mol"
            )
            g_class = "Strong Physical"
        elif avg_G < 0:
            scores["Physical"] += g_weight
            evidence.append(
                f"ΔG° indicates spontaneous physical adsorption: avg ΔG° = {avg_G:.1f} kJ/mol"
            )
            g_class = "Physical"
        else:
            evidence.append(
                f"ΔG° indicates non-spontaneous process: avg ΔG° = {avg_G:.1f} > 0 kJ/mol"
            )
            g_class = "Non-spontaneous"

        indicators["ΔG° (kJ/mol)"] = {
            "value": avg_G,
            "classification": g_class,
            "criterion": "0 to -20: Physical, -20 to -40: Strong phys., < -40: Chemical",
            "confidence": "Medium",
        }

    # 4. Freundlich n (weight: 15%)
    if n_freundlich is not None and n_freundlich > 0:
        n_weight = 15
        total_weight += n_weight

        # n > 1 indicates favorable adsorption (n_inv = 1/n, so 1/n < 1 means n > 1)
        if n_freundlich < 0.5:
            scores["Chemical"] += n_weight * 0.7
            scores["Physical"] += n_weight * 0.3
            evidence.append(
                f"Freundlich 1/n = {n_freundlich:.2f} < 0.5: Highly favorable, may indicate chemisorption"
            )
            n_class = "Highly favorable"
        elif n_freundlich < 1:
            scores["Physical"] += n_weight
            evidence.append(f"Freundlich 1/n = {n_freundlich:.2f} < 1: Favorable adsorption")
            n_class = "Favorable"
        else:
            scores["Physical"] += n_weight * 0.5
            evidence.append(f"Freundlich 1/n = {n_freundlich:.2f} ≥ 1: Linear or unfavorable")
            n_class = "Unfavorable"

        indicators["1/n (Freundlich)"] = {
            "value": n_freundlich,
            "classification": n_class,
            "criterion": "< 0.5: Highly favorable, 0.5-1: Favorable, > 1: Unfavorable",
            "confidence": "Medium",
        }

    # 5. Langmuir RL (weight: 10%)
    if RL is not None and 0 < RL < 1:
        rl_weight = 10
        total_weight += rl_weight

        if RL < 0.1:
            scores["Chemical"] += rl_weight * 0.6
            scores["Physical"] += rl_weight * 0.4
            evidence.append(f"RL = {RL:.3f} < 0.1: Highly favorable (near irreversible)")
            rl_class = "Highly favorable"
        else:
            scores["Physical"] += rl_weight
            evidence.append(f"RL = {RL:.3f}: Favorable adsorption")
            rl_class = "Favorable"

        indicators["RL (Langmuir)"] = {
            "value": RL,
            "classification": rl_class,
            "criterion": "0 < RL < 1: Favorable, RL < 0.1: Near irreversible",
            "confidence": "Medium",
        }

    # Normalize scores to percentages
    if total_weight > 0:
        for key in scores:
            scores[key] = (scores[key] / total_weight) * 100

    # Determine final mechanism
    max_score = max(scores.values())
    mechanism = max(scores, key=lambda k: scores.get(k, 0.0))

    # Format mechanism name
    if mechanism == "Physical":
        mechanism_name = "Physical adsorption"
    elif mechanism == "Chemical":
        mechanism_name = "Chemical adsorption"
    else:
        mechanism_name = "Ion exchange"

    # If scores are close, indicate mixed mechanism
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) >= 2 and sorted_scores[0] - sorted_scores[1] < 15:
        mechanism_name = "Mixed mechanism"

    # Calculate confidence based on score margin and number of indicators
    confidence = min(95, max_score + len(indicators) * 5)

    return {
        "mechanism": mechanism_name,
        "confidence": confidence,
        "scores": scores,
        "evidence": evidence,
        "indicators": indicators,
    }


def calculate_arrhenius_parameters(T_K: np.ndarray, k: np.ndarray) -> dict[str, Any]:
    """
    Calculate Arrhenius parameters from rate constants at different temperatures.

    Arrhenius equation: k = A × exp(-Ea/RT)
    ln(k) = ln(A) - Ea/RT

    Parameters
    ----------
    T_K : np.ndarray
        Temperature in Kelvin
    k : np.ndarray
        Rate constants

    Returns
    -------
    dict
        Arrhenius parameters Ea (activation energy) and A (pre-exponential factor)
    """
    if len(T_K) < 2 or len(k) < 2:
        return {"success": False, "error": "Need at least 2 temperature points"}

    # Filter valid values
    valid_mask = (k > 0) & np.isfinite(k) & np.isfinite(T_K) & (T_K > 0)
    T_K = T_K[valid_mask]
    k = k[valid_mask]

    if len(T_K) < 2:
        return {"success": False, "error": "Insufficient valid data points"}

    try:
        x = 1 / T_K  # 1/T
        y = np.log(k)  # ln(k)

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Arrhenius parameters
        # slope = -Ea/R, intercept = ln(A)
        Ea = -slope * R_GAS_CONSTANT / 1000  # kJ/mol
        A = np.exp(intercept)  # Pre-exponential factor

        # Interpretation
        if Ea < 5:
            interpretation = f"Diffusion-controlled (Ea = {Ea:.2f} kJ/mol < 5 kJ/mol)"
        elif Ea < 40:
            interpretation = f"Physical adsorption (5 < Ea = {Ea:.2f} < 40 kJ/mol)"
        else:
            interpretation = f"Chemical adsorption (Ea = {Ea:.2f} kJ/mol > 40 kJ/mol)"

        return {
            "success": True,
            "Ea": Ea,
            "A": A,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "interpretation": interpretation,
            "n_points": len(T_K),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# PROPAGATE CALIBRATION UNCERTAINTY
# =============================================================================
def propagate_calibration_uncertainty(
    absorbance: float,
    slope: float,
    intercept: float,
    slope_se: float,
    intercept_se: float,
    cov_slope_intercept: float = 0,
) -> tuple[float, float]:
    """
    Propagate uncertainty from calibration to concentration.

    Ce = (Abs - intercept) / slope
    """
    if abs(slope) < EPSILON_DIV:
        return 0.0, np.inf

    Ce = (absorbance - intercept) / slope

    # Partial derivatives
    dCe_dAbs = 1 / slope
    dCe_dSlope = -(absorbance - intercept) / slope**2
    dCe_dIntercept = -1 / slope

    # Assume absorbance uncertainty is negligible compared to calibration
    abs_se = 0.001  # Typical spectrophotometer precision

    variance = (
        (dCe_dAbs * abs_se) ** 2
        + (dCe_dSlope * slope_se) ** 2
        + (dCe_dIntercept * intercept_se) ** 2
        + 2 * dCe_dSlope * dCe_dIntercept * cov_slope_intercept
    )

    Ce_se = np.sqrt(max(0, variance))

    return max(0, Ce), Ce_se


# =============================================================================
# DUAL AXIS PLOT
# =============================================================================
def create_dual_axis_plot(
    data: pd.DataFrame,
    x_col: str,
    y1_col: str,
    y2_col: str,
    x_label: str,
    y1_label: str,
    y2_label: str,
    title: str = "",
) -> go.Figure:
    """Create professional-quality plot with two y-axes."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Primary y-axis trace (blue)
    fig.add_trace(
        go.Scatter(
            x=data[x_col],
            y=data[y1_col],
            mode="markers+lines",
            name=y1_label,
            marker={"size": 10, "color": "#1E88E5", "line": {"width": 1.5, "color": "#0D47A1"}},
            line={"width": 2, "color": "#1E88E5"},
        ),
        secondary_y=False,
    )

    # Secondary y-axis trace (red)
    fig.add_trace(
        go.Scatter(
            x=data[x_col],
            y=data[y2_col],
            mode="markers+lines",
            name=y2_label,
            marker={
                "size": 10,
                "color": "#E53935",
                "symbol": "square",
                "line": {"width": 1.5, "color": "#B71C1C"},
            },
            line={"width": 2, "color": "#E53935"},
        ),
        secondary_y=True,
    )

    # Professional layout
    fig.update_layout(
        title={"text": f"<b>{title}</b>", "font": {"size": 16, "family": FONT_FAMILY}},
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=450,
        margin={"l": 70, "r": 70, "t": 60, "b": 60},
        font={"family": FONT_FAMILY, "size": 12},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "black",
            "borderwidth": 1,
            "font": {"size": 11, "family": FONT_FAMILY},
        },
    )

    # Style x-axis
    fig.update_xaxes(
        title_text=x_label,
        title_font={"size": 14, "family": FONT_FAMILY},
        showgrid=True,
        gridwidth=1,
        gridcolor="#E0E0E0",
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        ticks="outside",
        tickfont={"size": 11, "family": FONT_FAMILY, "color": "#424242"},
        zeroline=False,
    )

    # Style primary y-axis
    fig.update_yaxes(
        title_text=y1_label,
        title_font={"size": 14, "family": FONT_FAMILY, "color": "#1E88E5"},
        showgrid=True,
        gridwidth=1,
        gridcolor="#E0E0E0",
        showline=True,
        linewidth=2,
        linecolor="black",
        ticks="outside",
        tickfont={"size": 11, "color": "#1E88E5"},
        zeroline=False,
        secondary_y=False,
    )

    # Style secondary y-axis
    fig.update_yaxes(
        title_text=y2_label,
        title_font={"size": 14, "family": FONT_FAMILY, "color": "#E53935"},
        showline=True,
        linewidth=2,
        linecolor="black",
        ticks="outside",
        tickfont={"size": 11, "color": "#E53935"},
        zeroline=False,
        secondary_y=True,
    )

    return fig


# =============================================================================
# reporting EXPORT HELPERS
# =============================================================================


@_optional_cache
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv(index=False, sep=";").encode("utf-8")


@_optional_cache
def convert_df_to_excel(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    """Convert DataFrame to Excel bytes for download."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.getvalue()


# =============================================================================
# SESSION STATE HELPERS
# =============================================================================
def get_current_study_state() -> dict[str, Any] | None:
    """Safely retrieves the state dictionary for the currently active study."""
    if not _STREAMLIT_AVAILABLE:
        return None
    study_name = st.session_state.get("current_study")
    if study_name and study_name in st.session_state.get("studies", {}):
        return st.session_state.studies[study_name]
    return None


# =============================================================================
# ACTIVITY COEFFICIENT (DAVIES EQUATION)
# =============================================================================
def calculate_activity_coefficient_davies(ionic_strength: float, charge: int = 1) -> float:
    """
    Calculate activity coefficient using Davies equation.

    log(γ) = -A × z² × (√I / (1 + √I) - 0.3I)

    Where A ≈ 0.509 at 25°C

    Parameters
    ----------
    ionic_strength : float
        Ionic strength of solution (mol/L)
    charge : int
        Ion charge (default 1)

    Returns
    -------
    float
        Activity coefficient γ
    """
    A = 0.509  # at 25°C
    sqrt_I = np.sqrt(ionic_strength)
    log_gamma = -A * charge**2 * (sqrt_I / (1 + sqrt_I) - 0.3 * ionic_strength)
    return 10**log_gamma


# =============================================================================
# REPLICATE DETECTION AND ERROR BARS
# =============================================================================
def detect_replicates(data: pd.DataFrame, x_col: str, tolerance: float = 0.01) -> pd.DataFrame:
    """
    Detect replicate measurements and calculate statistics.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    x_col : str
        Column name for x-variable (e.g., concentration, time)
    tolerance : float
        Relative tolerance for grouping replicates

    Returns
    -------
    pd.DataFrame with mean, std, n for each unique x value
    """
    data = data.copy()
    if len(data) == 0 or x_col not in data.columns:
        return data.copy()

    x_values = data[x_col].to_numpy(dtype=float, copy=True)
    if len(x_values) == 0:
        return data.copy()

    mean_x = np.mean(np.abs(x_values))
    if mean_x < EPSILON_DIV:
        mean_x = 1.0  # Fallback for very small values
    decimals = max(0, int(-np.log10(tolerance * mean_x + EPSILON_DIV)))
    x_rounded = np.round(x_values, decimals=decimals)
    data["_x_group"] = x_rounded

    # Get numeric columns for aggregation
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "_x_group"]

    # Group and aggregate
    agg_dict = {col: ["mean", "std", "count"] for col in numeric_cols}
    grouped = data.groupby("_x_group").agg(agg_dict)

    # Flatten column names
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    grouped = grouped.rename(columns={"_x_group": x_col})

    return grouped


# =============================================================================
# SESSION STATE MANAGEMENT UTILITIES (Moved from app_main.py)
# =============================================================================

# Session-state cleanup + metrics keys are defined in config.py (single source of truth)
# - SESSION_INPUT_KEYS_TO_CLEAR
# - SESSION_WIDGET_PREFIXES_TO_CLEAR
# - STUDY_METRIC_DATA_KEYS


def get_study_metrics() -> dict:
    """
    Compute metrics for the current active study.

    Returns
    -------
    dict
        Dictionary with study_count, active_data_count, calib_quality, has_active_study
    """
    if not _STREAMLIT_AVAILABLE:
        return {
            "study_count": 0,
            "active_data_count": 0,
            "calib_quality": 0,
            "has_active_study": False,
        }
    data_keys = STUDY_METRIC_DATA_KEYS
    active_data_total = len(data_keys)

    active_study_name = st.session_state.get("current_study")
    study_count = len(st.session_state.get("studies", {}))

    if not active_study_name or active_study_name not in st.session_state.get("studies", {}):
        return {
            "study_count": study_count,
            "active_data_count": 0,
            "active_data_total": len(STUDY_METRIC_DATA_KEYS),
            "calib_quality": 0,
            "has_active_study": False,
        }

    current_study_state = st.session_state.studies[active_study_name]
    active_data_count = sum(1 for key in data_keys if current_study_state.get(key) is not None)
    calib_params = current_study_state.get("calibration_params")
    calib_quality = calib_params.get("quality_score", 0) if calib_params else 0

    return {
        "study_count": study_count,
        "active_data_count": active_data_count,
        "active_data_total": active_data_total,
        "calib_quality": calib_quality,
        "has_active_study": True,
    }


def cleanup_session_state_keys(
    input_keys: tuple[str, ...] = SESSION_INPUT_KEYS_TO_CLEAR,
    widget_prefixes: tuple[str, ...] = SESSION_WIDGET_PREFIXES_TO_CLEAR,
) -> None:
    """Clean up session state keys when switching or adding studies."""
    if not _STREAMLIT_AVAILABLE:
        return

    if "session_state" not in dir(st):
        return

    for key in input_keys:
        st.session_state.pop(key, None)

    keys_to_remove = []
    for key in [k for k in st.session_state.keys() if isinstance(k, str)]:
        if any(key.startswith(prefix) for prefix in widget_prefixes):
            keys_to_remove.append(key)

    for key in keys_to_remove:
        st.session_state.pop(key, None)


def validate_study_name(name: str, existing_studies: dict | None = None) -> tuple[bool, str]:
    """Validate a study name for creation."""
    name = (name or "").strip()

    if not name:
        return False, "Please enter a study name."

    if existing_studies is None:
        if _STREAMLIT_AVAILABLE:
            existing_studies = st.session_state.get("studies", {})
        else:
            existing_studies = {}

    if name in existing_studies:
        return False, "A study with this name already exists."

    return True, ""


def calculate_calibration_stats(
    concentration: np.ndarray,
    absorbance: np.ndarray,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Calculate linear regression statistics for calibration curves."""
    from scipy.stats import linregress, t as t_dist

    n = len(concentration)
    if n < 2:
        raise ValueError("Need at least 2 points for linear regression")

    slope, intercept, r_value, p_value, std_err = linregress(concentration, absorbance)
    r_squared = r_value**2

    y_pred = slope * concentration + intercept
    residuals = absorbance - y_pred
    ss_res = np.sum(residuals**2)
    se_estimate = np.sqrt(ss_res / (n - 2)) if n > 2 else 0

    se_slope = se_estimate / np.sqrt(np.sum((concentration - np.mean(concentration)) ** 2))
    se_intercept = se_estimate * np.sqrt(
        1 / n + np.mean(concentration) ** 2 / np.sum((concentration - np.mean(concentration)) ** 2)
    )

    alpha = 1 - confidence_level
    t_val = t_dist.ppf(1 - alpha / 2, n - 2) if n > 2 else 2.0

    ci_slope = (slope - t_val * se_slope, slope + t_val * se_slope)
    ci_intercept = (intercept - t_val * se_intercept, intercept + t_val * se_intercept)

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "se_slope": se_slope,
        "se_intercept": se_intercept,
        "ci_slope": ci_slope,
        "ci_intercept": ci_intercept,
        "p_value": p_value,
        "n": n,
    }
