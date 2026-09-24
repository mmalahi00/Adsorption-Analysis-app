# utils.py
"""
AdsorbLab Pro - Advanced Utility Functions
==========================================

Advanced utility module providing:
- Advanced statistical analysis with confidence intervals
- Bootstrap resampling for robust error estimation
- PRESS/Q² cross-validation statistics
- Internal analysis consistency checking
- Methodological error detection
- Parameter uncertainty propagation
- Model selection criteria (R², Adj-R², AIC, AICc, BIC)
- Residual analysis and diagnostics
- Dual-unit calculations (mg/g and % Removal)
- Intelligent rule-based model recommendations
- Data quality assessment
- Advanced figure generation
"""

import hashlib
import io
import logging
import numbers
import re
import warnings
from collections.abc import Callable

# PIL import deferred to _convert_png_to_tiff_bytes() for faster startup
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, cast, TypeVar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import OptimizeWarning, curve_fit
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
    BOOTSTRAP_DEFAULT_SEED,
    BOOTSTRAP_EVALUATIONS_PER_DRAW,
    BOOTSTRAP_MIN_SUCCESS,
    BOOTSTRAP_MIN_SUCCESS_FRACTION,
    EPSILON_DIV,
    EPSILON_ZERO,
    FONT_FAMILY,
    FUZZY_MATCH_CUTOFF,
    MAX_FIT_ITERATIONS,
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
    "load_uploaded_table",
    # Basic calculations
    "calculate_removal_percentage",
    "calculate_adsorption_capacity",
    "calculate_Ce_from_absorbance",
    "calculate_temperature_results",
    "calculate_temperature_results_direct",
    # Statistical functions
    "calculate_press",
    "calculate_press_details",
    "calculate_q2",
    "calculate_error_metrics",
    "calculate_akaike_weights",
    "bootstrap_confidence_intervals",
    "bootstrap_parameter_intervals",
    "analyze_residuals",
    # Mechanism analysis
    "sign_label",
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
    "propagate_kd_uncertainty",
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


# -----------------------------------------------------------------------------
# Units
# -----------------------------------------------------------------------------
# Quantity recognition and unit recognition are separate steps.  A header such as
# "Ce (ug/L)" is the quantity Ce expressed in µg/L; the values must be converted
# before the column may carry the canonical name "Ce", because every downstream
# calculation assumes the canonical unit listed here.  A header without a unit
# follows the stated template convention (the canonical unit).
CANONICAL_UNITS: dict[str, str] = {
    "C0": "mg/L",
    "Ce": "mg/L",
    "Ct": "mg/L",
    "Concentration": "mg/L",
    "Absorbance": "AU",
    "Time": "min",
    "Mass": "g",
    "Volume": "L",
    "pH": "",
    "Temperature": "°C",
    "qe": "mg/g",
    "qt": "mg/g",
}

_QUANTITY_DIMENSION: dict[str, str] = {
    "C0": "concentration",
    "Ce": "concentration",
    "Ct": "concentration",
    "Concentration": "concentration",
    "Absorbance": "absorbance",
    "Time": "time",
    "Mass": "mass",
    "Volume": "volume",
    "pH": "dimensionless",
    "Temperature": "temperature",
    "qe": "capacity",
    "qt": "capacity",
}

# Normalized unit text -> (display unit, factor, offset); canonical = value*factor + offset.
_UNIT_TABLES: dict[str, dict[str, tuple[str, float, float]]] = {
    "concentration": {
        "mg/l": ("mg/L", 1.0, 0.0),
        "mgl": ("mg/L", 1.0, 0.0),
        "mg/dm3": ("mg/L", 1.0, 0.0),
        "ug/ml": ("µg/mL", 1.0, 0.0),
        "g/m3": ("g/m³", 1.0, 0.0),
        "ug/l": ("µg/L", 1e-3, 0.0),
        "ugl": ("µg/L", 1e-3, 0.0),
        "ng/ml": ("ng/mL", 1e-3, 0.0),
        "ng/l": ("ng/L", 1e-6, 0.0),
        "ngl": ("ng/L", 1e-6, 0.0),
        "g/l": ("g/L", 1e3, 0.0),
        "g/dm3": ("g/dm³", 1e3, 0.0),
        "mg/ml": ("mg/mL", 1e3, 0.0),
    },
    "time": {
        "min": ("min", 1.0, 0.0),
        "mins": ("min", 1.0, 0.0),
        "minute": ("min", 1.0, 0.0),
        "minutes": ("min", 1.0, 0.0),
        "s": ("s", 1.0 / 60.0, 0.0),
        "sec": ("s", 1.0 / 60.0, 0.0),
        "secs": ("s", 1.0 / 60.0, 0.0),
        "second": ("s", 1.0 / 60.0, 0.0),
        "seconds": ("s", 1.0 / 60.0, 0.0),
        "h": ("h", 60.0, 0.0),
        "hr": ("h", 60.0, 0.0),
        "hrs": ("h", 60.0, 0.0),
        "hour": ("h", 60.0, 0.0),
        "hours": ("h", 60.0, 0.0),
        "d": ("d", 1440.0, 0.0),
        "day": ("d", 1440.0, 0.0),
        "days": ("d", 1440.0, 0.0),
    },
    "mass": {
        "g": ("g", 1.0, 0.0),
        "mg": ("mg", 1e-3, 0.0),
        "ug": ("µg", 1e-6, 0.0),
        "kg": ("kg", 1e3, 0.0),
    },
    "volume": {
        "l": ("L", 1.0, 0.0),
        "dm3": ("dm³", 1.0, 0.0),
        "ml": ("mL", 1e-3, 0.0),
        "cm3": ("cm³", 1e-3, 0.0),
        "ul": ("µL", 1e-6, 0.0),
    },
    "temperature": {
        "c": ("°C", 1.0, 0.0),
        "degc": ("°C", 1.0, 0.0),
        "celsius": ("°C", 1.0, 0.0),
        "k": ("K", 1.0, -273.15),
        "kelvin": ("K", 1.0, -273.15),
    },
    "absorbance": {
        "au": ("AU", 1.0, 0.0),
        "a.u": ("AU", 1.0, 0.0),
        "abs": ("AU", 1.0, 0.0),
        "absorbanceunits": ("AU", 1.0, 0.0),
        "-": ("AU", 1.0, 0.0),
    },
    "dimensionless": {
        "-": ("", 1.0, 0.0),
        "unitless": ("", 1.0, 0.0),
        "dimensionless": ("", 1.0, 0.0),
        "phunits": ("", 1.0, 0.0),
        "phunit": ("", 1.0, 0.0),
    },
    "capacity": {
        "mg/g": ("mg/g", 1.0, 0.0),
    },
}

# Units that name a quantity but cannot be converted without information the
# application does not have.  They must stop the import rather than be guessed.
_AMBIGUOUS_UNITS: dict[str, str] = {
    "ppm": "'ppm' does not state whether it is mass/mass or mass/volume",
    "ppb": "'ppb' does not state whether it is mass/mass or mass/volume",
    "ppt": "'ppt' does not state whether it is mass/mass or mass/volume",
    "%": "a percentage cannot be converted to a concentration or amount",
    "m": "molar units require the adsorbate molar mass",
    "mm": "molar units require the adsorbate molar mass",
    "um": "molar units require the adsorbate molar mass",
    "nm": "molar units require the adsorbate molar mass",
    "mol/l": "molar units require the adsorbate molar mass",
    "mmol/l": "molar units require the adsorbate molar mass",
    "umol/l": "molar units require the adsorbate molar mass",
    "nmol/l": "molar units require the adsorbate molar mass",
    "mmol/g": "molar capacities require the adsorbate molar mass",
    "f": "Fahrenheit is not supported",
    "degf": "Fahrenheit is not supported",
    "fahrenheit": "Fahrenheit is not supported",
}

_WAVELENGTH_ANNOTATION = re.compile(r"^(?:lambda|λ)?\s*=?\s*\d+(?:\.\d+)?\s*nm$")
_BRACKETED_UNIT = re.compile(r"^(?P<quantity>.*?)\s*[\(\[\{](?P<unit>[^\)\]\}]*)[\)\]\}]\s*$")


@dataclass
class ColumnInterpretation:
    """How one uploaded column header is interpreted.

    ``canonical`` is the recognized quantity (or None).  Values in the column are
    converted to the canonical unit as ``value * factor + offset``.  ``error``
    holds an actionable message when the stated unit cannot be used; such a column
    must not be used for calculations.
    """

    source: str
    canonical: str | None = None
    unit_text: str = ""
    unit: str = ""
    canonical_unit: str = ""
    factor: float = 1.0
    offset: float = 0.0
    explicit_unit: bool = False
    error: str | None = None

    @property
    def is_canonical_unit(self) -> bool:
        """True when no conversion is needed (factor 1, offset 0)."""
        return self.factor == 1.0 and self.offset == 0.0


def _clean_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def _variant_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for standard, variants in COLUMN_SYNONYMS.items():
        for variant in variants:
            lookup[_clean_token(variant)] = standard
    return lookup


_VARIANT_TO_STANDARD = _variant_lookup()


def _normalize_unit_text(unit: str) -> str:
    """Normalize a unit string to a lookup key (e.g. 'µg·L⁻¹' -> 'ug/l')."""
    u = str(unit).strip().lower()
    for micro in ("µ", "μ"):
        u = u.replace(micro, "u")
    u = u.replace("mcg", "ug").replace("°", "").replace("º", "").replace("deg ", "deg")
    u = u.replace("−", "-").replace("–", "-").replace("⁻¹", "-1").replace("^-1", "-1")
    u = u.replace("³", "3").replace("·", " ").replace("*", " ")
    for word, short in (("litres", "l"), ("liters", "l"), ("litre", "l"), ("liter", "l")):
        u = u.replace(word, short)
    u = re.sub(r"\s+per\s+", "/", u)
    u = re.sub(r"\s*/\s*", "/", u)
    u = re.sub(r"\s+", " ", u).strip().rstrip(".")
    # "mg l-1" / "mg/l-1" / "mgl-1" -> "mg/l"
    match = re.fullmatch(r"([a-z]+)\s*/?\s*([a-z0-9]+)-1", u)
    if match:
        u = f"{match.group(1)}/{match.group(2)}"
    # "mg l" (space separated, e.g. from 'Ce_mg_L') -> "mg/l"
    match = re.fullmatch(r"([a-z]+) ([a-z0-9]+)", u)
    if match and match.group(2) in {"l", "ml", "g", "dm3", "m3"}:
        u = f"{match.group(1)}/{match.group(2)}"
    return u.replace(" ", "")


def _unit_dimension(unit_key: str) -> str | None:
    for dimension, table in _UNIT_TABLES.items():
        if unit_key in table:
            return dimension
    return None


def _match_quantity(text: str, allow_fuzzy: bool = True) -> str | None:
    """Recognize a quantity name (without unit) using the synonym table."""
    clean = _clean_token(text)
    if not clean:
        return None
    if clean in _VARIANT_TO_STANDARD:
        return _VARIANT_TO_STANDARD[clean]
    if not allow_fuzzy:
        return None
    # A single trailing unit letter glued to a known name ("Timeh", "Tempk")
    # is ambiguous: fuzzy matching would silently drop the unit.
    if len(clean) > 1 and clean[:-1] in _VARIANT_TO_STANDARD and clean[-1] in "hsdkcgl":
        return None
    match = get_close_matches(clean, list(_VARIANT_TO_STANDARD), n=1, cutoff=FUZZY_MATCH_CUTOFF)
    return _VARIANT_TO_STANDARD[match[0]] if match else None


def _resolve_unit(interp: ColumnInterpretation, unit_text: str) -> ColumnInterpretation:
    """Attach unit information (or an actionable error) to a recognized quantity."""
    canonical = interp.canonical
    if canonical is None:
        return interp
    interp.unit_text = unit_text.strip()
    interp.canonical_unit = CANONICAL_UNITS.get(canonical, "")
    key = _normalize_unit_text(unit_text)
    if key == "":
        interp.unit = interp.canonical_unit
        return interp

    interp.explicit_unit = True
    dimension = _QUANTITY_DIMENSION.get(canonical, "")
    # 't' / 'T' is used for both time and temperature; a temperature unit decides.
    if canonical == "Time" and key in _UNIT_TABLES["temperature"]:
        canonical = interp.canonical = "Temperature"
        dimension = "temperature"
        interp.canonical_unit = CANONICAL_UNITS["Temperature"]

    table = _UNIT_TABLES.get(dimension, {})
    if key in table:
        interp.unit, interp.factor, interp.offset = table[key]
        return interp
    if dimension == "absorbance" and _WAVELENGTH_ANNOTATION.match(key.replace("nm", " nm")):
        interp.unit = "AU"
        return interp

    header = interp.source
    if key in _AMBIGUOUS_UNITS:
        interp.error = (
            f"Column '{header}': unit '{interp.unit_text}' cannot be used for {canonical} — "
            f"{_AMBIGUOUS_UNITS[key]}. Convert the values to {interp.canonical_unit or 'a supported unit'} "
            "and state the unit in the header."
        )
        return interp
    other = _unit_dimension(key)
    supported = ", ".join(sorted({v[0] for v in table.values() if v[0]})) or "no unit"
    if other is not None:
        interp.error = (
            f"Column '{header}': '{interp.unit_text}' is a {other} unit, but {canonical} is a "
            f"{dimension} quantity. Supported units: {supported}."
        )
    else:
        interp.error = (
            f"Column '{header}': unrecognized unit '{interp.unit_text}' for {canonical}. "
            f"Supported units: {supported}."
        )
    return interp


def interpret_column_header(header: Any) -> ColumnInterpretation:
    """
    Interpret an uploaded column header as (quantity, unit).

    Recognized forms include ``"Ce (mg/L)"``, ``"Ce [µg/L]"``, ``"Ce_mg_L"``,
    ``"Time h"`` and the legacy synonyms in :data:`COLUMN_SYNONYMS`.  A header
    without a unit uses the canonical unit of the quantity (template convention).
    Unsupported, ambiguous or contradictory units produce ``error`` instead of a
    guess.
    """
    source = str(header)
    text = source.strip()
    interp = ColumnInterpretation(source=source)

    bracketed = _BRACKETED_UNIT.match(text)
    if bracketed:
        interp.canonical = _match_quantity(bracketed.group("quantity"))
        return _resolve_unit(interp, bracketed.group("unit"))

    # Exact synonym without unit text first (e.g. 'contact_time', 'abs_eq').
    exact = _match_quantity(text, allow_fuzzy=False)

    # Separator-delimited unit suffix: 'Ce_mg_L', 'Time h', 'temperature_k'.
    tokens = [tok for tok in re.split(r"[\s_,;]+", text) if tok]
    for split_at in range(len(tokens) - 1, 0, -1):
        unit_part = " ".join(tokens[split_at:])
        key = _normalize_unit_text(unit_part)
        if _unit_dimension(key) is None:
            continue
        quantity = _match_quantity(" ".join(tokens[:split_at]), allow_fuzzy=False)
        if quantity is not None:
            interp.canonical = quantity
            return _resolve_unit(interp, unit_part)

    if exact is not None:
        interp.canonical = exact
        return _resolve_unit(interp, "")

    # Unit glued to the name without a separator ('Ceugl', 'Timemin').
    clean = _clean_token(text)
    for variant in sorted(_VARIANT_TO_STANDARD, key=len, reverse=True):
        if len(variant) >= 2 and clean.startswith(variant) and len(clean) > len(variant):
            suffix = clean[len(variant) :]
            if len(suffix) >= 2 and _unit_dimension(suffix) is not None:
                interp.canonical = _VARIANT_TO_STANDARD[variant]
                return _resolve_unit(interp, suffix)

    interp.canonical = _match_quantity(text)
    return _resolve_unit(interp, "")


def standardize_column_name(col_name: str) -> str:
    """
    Standardize a column name using known synonyms.

    Only headers whose values are already in the canonical unit are renamed.  A
    header that states another unit (e.g. ``"Ce (ug/L)"``) keeps its original
    name, because renaming it without converting the values would silently change
    their meaning; use :func:`parse_uploaded_table` to convert such columns.
    """
    interp = interpret_column_header(col_name)
    if interp.canonical is not None and interp.error is None and interp.is_canonical_unit:
        return interp.canonical
    return col_name


def standardize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standardize_column_name to all columns in a DataFrame."""
    df.columns = [standardize_column_name(col) for col in df.columns]
    return df


# Unitless temperatures follow the template convention (°C).  Values above this
# threshold are not plausible solution-phase temperatures in °C and are typical
# of kelvin data, so a unitless column containing them is refused rather than
# reinterpreted: the header must state the unit.
UNITLESS_TEMPERATURE_LIMIT_C = 200.0


def _is_blank(value: Any) -> bool:
    """True for an empty cell: None, NaN/NaT/NA or whitespace-only text."""
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


_NUMBER_TEXT = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf|infinity|nan)", re.IGNORECASE
)


def _coerce_numeric_column(values: pd.Series) -> tuple[pd.Series, bool]:
    """
    Convert a column to float, accepting comma decimal separators.

    Numbers are taken exactly as they are; text is read as a decimal number
    (correctly rounded; a comma may be the decimal separator); anything else is
    NaN.  The flag tells whether a comma was read as a decimal point.
    """
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").astype(float), False
    numeric = np.full(len(values), np.nan)
    has_commas = False
    for i, value in enumerate(values.to_numpy()):
        if isinstance(value, numbers.Real) and not isinstance(value, bool | np.bool_):
            numeric[i] = float(value)
        elif isinstance(value, str):
            text = value.strip().replace(",", ".")
            if _NUMBER_TEXT.fullmatch(text):
                numeric[i] = float(text)
                has_commas = has_commas or "," in value
    return pd.Series(numeric, index=values.index), has_commas


def convert_columns_to_canonical(
    df: pd.DataFrame, required_cols: list[str]
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """
    Map, validate and convert the required columns of a raw table.

    Returns the required columns (canonical names, canonical units) followed by
    every other column of the table unchanged (identifiers, uncertainties,
    context such as ``SampleID``, ``Ce_SD`` or ``Matrix``; wholly empty unnamed
    columns are dropped), and a report with ``errors``, ``warnings``, ``info``,
    ``column_map`` and ``metadata_columns``.  Conflicting mappings, missing
    columns and unusable units are errors; nothing is guessed.

    A required-column cell that is not a number (e.g. ``'n.d.'``) keeps its
    original content instead of becoming NaN, so that
    :func:`prepare_analysis_data` reports the row with that content rather than
    treating it as missing or empty; blank cells are NaN.
    """
    report: dict[str, Any] = {"errors": [], "warnings": [], "info": [], "column_map": {}}
    interpretations = [interpret_column_header(col) for col in df.columns]

    by_canonical: dict[str, list[tuple[int, ColumnInterpretation]]] = {}
    for position, interp in enumerate(interpretations):
        if interp.canonical is not None:
            by_canonical.setdefault(interp.canonical, []).append((position, interp))

    converted = pd.DataFrame(index=df.index)
    missing = []
    used_positions: set[int] = set()
    for canonical in required_cols:
        candidates = by_canonical.get(canonical, [])
        if not candidates:
            missing.append(canonical)
            continue
        if len(candidates) > 1:
            names = ", ".join(f"'{interp.source}'" for _, interp in candidates)
            report["errors"].append(
                f"Columns {names} all describe {canonical}. Keep exactly one {canonical} "
                "column (or rename the others) so the analysis cannot pick one silently."
            )
            continue
        position, interp = candidates[0]
        used_positions.add(position)
        if interp.error:
            report["errors"].append(interp.error)
            continue

        cells = df.iloc[:, position]
        values, had_commas = _coerce_numeric_column(cells)
        if had_commas:
            report["info"].append(
                f"Column '{interp.source}': comma decimal separators were read as decimal points."
            )
        canonical_values = values * interp.factor + interp.offset
        unreadable = values.isna().to_numpy() & ~cells.map(_is_blank).to_numpy(dtype=bool)
        if unreadable.any():
            canonical_values = canonical_values.astype(object)
            canonical_values[unreadable] = cells.to_numpy()[unreadable]
        converted[canonical] = canonical_values
        report["column_map"][canonical] = {
            "source": interp.source,
            "unit": interp.unit,
            "canonical_unit": interp.canonical_unit,
            "factor": interp.factor,
            "offset": interp.offset,
            "explicit_unit": interp.explicit_unit,
        }
        if not interp.is_canonical_unit:
            report["info"].append(
                f"Column '{interp.source}' read as {canonical} in {interp.unit} and converted to "
                f"{interp.canonical_unit}."
            )

        if canonical == "Temperature":
            finite = values[np.isfinite(values)]
            if (
                not interp.explicit_unit
                and len(finite)
                and finite.max() > UNITLESS_TEMPERATURE_LIMIT_C
            ):
                report["errors"].append(
                    f"Column '{interp.source}' has no unit and contains values up to "
                    f"{finite.max():g}. Unitless temperatures are read as °C; values this high "
                    "are typical of kelvin. Label the column 'Temperature (K)' or "
                    "'Temperature (°C)'."
                )
            elif interp.unit == "K" and len(finite) and finite.min() < UNITLESS_TEMPERATURE_LIMIT_C:
                report["warnings"].append(
                    f"Column '{interp.source}' is labelled kelvin but contains values as low as "
                    f"{finite.min():g} K. Check the unit."
                )

    if missing:
        report["errors"].append(f"Missing columns: {', '.join(missing)}")
        report["info"].append(f"Found columns: {', '.join(str(c) for c in df.columns)}")

    if report["errors"]:
        return None, report

    # Every other column is kept as metadata, unconverted and under its own header.
    metadata = []
    for position, column in enumerate(df.columns):
        if position in used_positions:
            continue
        name = str(column)
        values = df.iloc[:, position]
        if name.startswith("Unnamed:") and values.isna().all():
            continue
        while name in converted.columns:
            name = f"{name} (input)"
        converted[name] = values.to_numpy()
        metadata.append(name)
    report["metadata_columns"] = metadata
    return converted[list(required_cols) + metadata], report


# Only empty cells are missing when a file is read; every other cell keeps its
# text.  pandas' default missing-value tokens ('NA', 'N/A', 'null', 'NaN', 'None',
# ...) are not applied: they would erase recorded markers and literal identifiers
# (e.g. SampleID 'NA') before the import can report them.
_LITERAL_CELLS: dict[str, Any] = {"keep_default_na": False, "na_values": [""]}


def read_tabular_file(file_content: bytes, file_name: str) -> pd.DataFrame:
    """Read CSV (semicolon/comma separated) or Excel bytes into a DataFrame, cell text kept."""
    if file_name.lower().endswith(".csv"):
        # Blank lines are kept (as empty rows) so row numbers in messages match
        # the data rows of the file; empty rows are reported, not analysed.
        try:
            df = pd.read_csv(
                io.BytesIO(file_content),
                sep=";",
                decimal=",",
                skip_blank_lines=False,
                **_LITERAL_CELLS,
            )
            if len(df.columns) == 1:
                df = pd.read_csv(
                    io.BytesIO(file_content),
                    sep=",",
                    decimal=".",
                    skip_blank_lines=False,
                    **_LITERAL_CELLS,
                )
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
            df = pd.read_csv(
                io.BytesIO(file_content),
                sep=",",
                decimal=".",
                skip_blank_lines=False,
                **_LITERAL_CELLS,
            )
        return df
    return pd.read_excel(io.BytesIO(file_content), **_LITERAL_CELLS)


# Row identifier of every table derived from an upload: the 1-based data row of the
# uploaded file.  The name is reserved; a column of that name in a file is kept,
# with its values unchanged, under a collision-safe name (see _reserve_row_column).
ROW_ID_COLUMN = "source_row"


def _kept_name(name: str, taken: set[str]) -> str:
    target = f"{name} (input)"
    while target in taken:
        target = f"{target} (input)"
    return target


def _reserve_row_column(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Rename file columns called ``source_row`` so they cannot pose as row numbers."""
    taken = {str(column) for column in df.columns}
    if ROW_ID_COLUMN not in taken:
        return df, []
    columns, kept = [], []
    for column in df.columns:
        if str(column) == ROW_ID_COLUMN:
            column = _kept_name(ROW_ID_COLUMN, taken)
            taken.add(column)
            kept.append(column)
        columns.append(column)
    renamed = df.copy()
    renamed.columns = columns
    return renamed, kept


def parse_uploaded_table(
    file_content: bytes, file_name: str, required_cols: list[str], study_type: str
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """
    Parse an uploaded file into the required canonical columns (canonical units).

    Pure function (no Streamlit calls).  Returns ``(DataFrame or None, status)``
    where ``status`` has ``messages`` [(level, text)], ``status`` ('success' or
    'error'), ``quality_report``, ``column_map``, ``metadata_columns`` and
    ``raw_table`` (the file as read, with its ``source_row``).  The DataFrame is
    ``source_row`` (1-based data row of the file, blank rows included) followed
    by :func:`convert_columns_to_canonical`'s columns: every row of the file, in
    file order, with non-numeric cells kept as read.  ``source_row`` is reserved:
    a column of that name in the file is kept unchanged as ``source_row (input)``.
    """
    status: dict[str, Any] = {
        "messages": [],
        "status": "success",
        "quality_report": None,
        "column_map": {},
        "metadata_columns": [],
        "raw_table": None,
    }
    try:
        df = read_tabular_file(file_content, file_name)
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
        ValueError,
        KeyError,
        OSError,
    ) as e:
        status["messages"].append(("error", f"Error reading file: {e}"))
        status["status"] = "error"
        return None, status

    df, kept = _reserve_row_column(df.reset_index(drop=True))
    rows = np.arange(1, len(df) + 1)
    raw = df.copy()
    raw.insert(0, ROW_ID_COLUMN, rows)
    status["raw_table"] = raw
    converted, report = convert_columns_to_canonical(df, list(required_cols))
    status["column_map"] = report["column_map"]
    status["metadata_columns"] = report.get("metadata_columns", [])
    for name in kept:
        report["info"].append(
            f"The file's own '{ROW_ID_COLUMN}' column is kept, unchanged, as '{name}'; "
            f"'{ROW_ID_COLUMN}' numbers the data rows of this file."
        )
    for level in ("errors", "warnings", "info"):
        for message in report[level]:
            status["messages"].append((level.rstrip("s") if level != "info" else "info", message))
    if converted is None:
        status["status"] = "error"
        return None, status

    converted = converted.reset_index(drop=True)
    converted.insert(0, ROW_ID_COLUMN, rows)
    numeric = converted.copy()
    for col in required_cols:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    status["quality_report"] = assess_data_quality(numeric, study_type)
    return converted, status


def _row_numbers(values: pd.Series) -> NDArray[np.int64] | None:
    """Supplied row numbers as integers, or None unless they are unique positive integers."""
    numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if (
        not np.all(np.isfinite(numbers))
        or np.any(numbers < 1)
        or np.any(numbers != np.round(numbers))
        or len(np.unique(numbers)) != len(numbers)
    ):
        return None
    return numbers.astype(np.int64)


def _cell_text(value: Any) -> str:
    return repr(value) if isinstance(value, str) else str(value)


# Text conventionally written for a missing value (pandas' default markers).  In a
# required column such a cell is a recorded missing measurement: the row is kept
# and excluded with the marker quoted.  It is never read as zero, as a detection
# limit or as an empty record, and the same text in other columns stays literal.
MISSING_VALUE_MARKERS = frozenset(
    {
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "<NA>",
        "N/A",
        "NA",
        "NULL",
        "NaN",
        "None",
        "n/a",
        "nan",
        "null",
    }
)


def _is_missing_marker(value: Any) -> bool:
    return isinstance(value, str) and value.strip() in MISSING_VALUE_MARKERS


def prepare_analysis_data(
    df: pd.DataFrame | None, required_cols: list[str], min_rows: int = 1
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """
    Build the numeric analysis view of an input table without dropping observations.

    Every row that has content in any cell is kept, with its 1-based
    ``source_row`` and with its other (metadata) columns unchanged, under their
    own headers, after the required numeric columns; only the required columns
    are coerced to numbers (synonyms of their names are standardized).
    Required values that are missing, non-numeric or non-finite become NaN and are
    described in ``report['row_issues']`` (``{source_row: [text, ...]}``, quoting
    the cell as it was read, e.g. ``"Ce: non-numeric value 'n.d.'"``, or
    ``"Ce: recorded as missing ('N/A')"`` for a :data:`MISSING_VALUE_MARKERS`
    cell); the calculators then mark those rows as excluded with that reason.  A
    record that carries only an identifier, a note or missing-value markers is
    such an excluded observation.  Only rows in which every cell is blank are not
    observations; they are listed in ``report['ignored_empty_rows']``.

    An existing ``source_row`` column is used as the row numbers when it holds
    unique positive integers; otherwise it is kept as ``source_row (input)`` (see
    ``report['info']``) and rows are numbered in table order.  Returns
    ``(None, report)`` when required columns are missing or duplicated, or fewer
    than ``min_rows`` rows have complete numeric values.
    """
    report: dict[str, Any] = {
        "errors": [],
        "info": [],
        "row_issues": {},
        "ignored_empty_rows": [],
        "n_complete": 0,
    }
    if df is None or df.empty:
        report["errors"].append("No data rows.")
        return None, report

    work = df.copy().reset_index(drop=True)
    source = np.arange(1, len(work) + 1)
    if ROW_ID_COLUMN in work.columns:
        supplied = work.pop(ROW_ID_COLUMN)
        numbers = _row_numbers(supplied)
        if numbers is not None:
            source = numbers
        else:
            name = _kept_name(ROW_ID_COLUMN, {str(c) for c in work.columns})
            work.insert(0, name, supplied.to_numpy())
            report["info"].append(
                f"Column '{ROW_ID_COLUMN}' does not hold unique positive row numbers; it is "
                f"kept as '{name}' and the rows are numbered in table order."
            )
    # Synonyms of the required quantities are standardized; every other header is
    # the user's own and stays as it is (e.g. an imported 'qe_mg_g' column).
    standardized = [standardize_column_name(column) for column in work.columns]
    work.columns = [
        new if new in required_cols else old for old, new in zip(work.columns, standardized)
    ]
    names = list(work.columns)
    duplicated = sorted({c for c in required_cols if names.count(c) > 1})
    if duplicated:
        report["errors"].append(
            f"More than one column describes {', '.join(duplicated)}; keep exactly one."
        )
    missing = [c for c in required_cols if c not in names]
    if missing:
        report["errors"].append(f"Missing columns: {', '.join(missing)}")
    if report["errors"]:
        return None, report

    # Emptiness is judged on every cell as supplied, never on coerced values.
    blank = np.ones(len(work), dtype=bool)
    for position in range(work.shape[1]):
        blank &= work.iloc[:, position].map(_is_blank).to_numpy(dtype=bool)

    out = pd.DataFrame({ROW_ID_COLUMN: source.astype(int)})
    for col in required_cols:
        original = work[col]
        numeric, _ = _coerce_numeric_column(original)
        values = numeric.to_numpy(dtype=float)
        cell_blank = original.map(_is_blank).to_numpy(dtype=bool)
        for i in np.flatnonzero(~np.isfinite(values)):
            if cell_blank[i]:
                text = "missing value"
            elif _is_missing_marker(original.iloc[i]):
                text = f"recorded as missing ({_cell_text(original.iloc[i])})"
            elif np.isinf(values[i]):
                text = f"non-finite value {_cell_text(original.iloc[i])}"
            else:
                text = f"non-numeric value {_cell_text(original.iloc[i])}"
            report["row_issues"].setdefault(int(source[i]), []).append(f"{col}: {text}")
        out[col] = values
    for position, name in enumerate(work.columns):
        if name in required_cols:
            continue
        target = str(name)
        while target in out.columns:
            target = f"{target} (input)"
        out[target] = work.iloc[:, position].to_numpy()

    for i in np.flatnonzero(blank):
        report["ignored_empty_rows"].append(int(source[i]))
        report["row_issues"].pop(int(source[i]), None)
    out = out[~blank].reset_index(drop=True)
    complete = np.isfinite(out[required_cols].to_numpy(dtype=float)).all(axis=1)
    report["n_complete"] = int(complete.sum())
    if report["n_complete"] < min_rows:
        report["errors"].append(
            f"Only {report['n_complete']} row(s) have complete numeric values for "
            f"{', '.join(required_cols)}; at least {min_rows} required."
        )
        return None, report
    return out, report


def validate_data_editor(
    df: pd.DataFrame | None, required_cols: list[str], min_rows: int = MIN_DATA_POINTS
) -> pd.DataFrame | None:
    """
    Validate an input table for analysis (compatibility wrapper).

    Returns the analysis view from :func:`prepare_analysis_data` (all non-empty
    rows, ``source_row``, NaN for missing/non-numeric required values) or None when
    columns are missing/duplicated or fewer than ``min_rows`` rows are complete.
    Rows are no longer dropped silently; calculators report them as excluded.
    Handles both period (.) and comma (,) as decimal separators.
    """
    prepared, _ = prepare_analysis_data(df, required_cols, min_rows=min_rows)
    return prepared


def load_uploaded_table(
    file_content: bytes,
    file_name: str,
    required_cols: list[str],
    study_type: str,
    *,
    parse: Callable[..., tuple[pd.DataFrame | None, dict[str, Any]]] | None = None,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """
    The complete import path of one uploaded file: parsing, then preparation.

    Pure function (no Streamlit calls); ``parse`` defaults to
    :func:`parse_uploaded_table` (the sidebar passes its cached wrapper).  Returns
    ``(analysis view or None, upload)``.  ``upload`` holds the messages to show,
    ``messages`` (parsing) and ``row_messages`` (preparation) as (level, text),
    the ``quality_report`` and the source record of this file: ``source_file``,
    ``raw_data`` (the table as read, with ``source_row``), ``column_map`` (source
    header, unit and conversion of each required column), ``metadata_columns``,
    ``row_issues`` and ``ignored_empty_rows``.  The analysis view is None when the
    file is rejected; callers then keep their stored data and source unchanged.
    """
    parse = parse or parse_uploaded_table
    df, status = parse(file_content, file_name, list(required_cols), study_type)
    upload: dict[str, Any] = {
        "source_file": file_name,
        "raw_data": status.get("raw_table"),
        "column_map": status.get("column_map") or {},
        "metadata_columns": list(status.get("metadata_columns") or []),
        "row_issues": {},
        "ignored_empty_rows": [],
        "quality_report": status.get("quality_report"),
        "messages": list(status.get("messages", [])),
        "row_messages": [],
    }
    if df is None:
        return None, upload

    prepared, report = prepare_analysis_data(df, list(required_cols), min_rows=1)
    row_messages = upload["row_messages"]
    row_messages += [("info", text) for text in report["info"]]
    row_messages += [("error", text) for text in report["errors"]]
    if report["ignored_empty_rows"]:
        rows = ", ".join(map(str, report["ignored_empty_rows"]))
        row_messages.append(("info", f"Ignored empty row(s) (no value in any column): {rows}."))
    issues = report["row_issues"]
    if prepared is not None and issues:
        listed = "; ".join(
            f"row {row}: {', '.join(texts)}" for row, texts in sorted(issues.items())[:6]
        )
        more = f"; … and {len(issues) - 6} more" if len(issues) > 6 else ""
        row_messages.append(
            (
                "warning",
                f"{len(issues)} row(s) contain missing or non-numeric values. They are kept and "
                f"shown as excluded in the analysis: {listed}{more}.",
            )
        )
    upload["row_issues"] = issues
    upload["ignored_empty_rows"] = report["ignored_empty_rows"]
    if prepared is None or prepared.empty:
        return None, upload
    return prepared, upload


# =============================================================================
# DUAL UNIT CALCULATIONS
# =============================================================================
# These helpers return the calculated value itself: no clipping to [0, 100] % or
# flooring at zero, and NaN when the quantity is undefined.  Whether a value is
# usable is decided per observation by build_uptake_table, which records why.
def calculate_removal_percentage(C0: float, Ce: float) -> float:
    """Calculate removal percentage: % Removal = [(C0 - Ce) / C0] × 100 (NaN if C0 ≤ 0)."""
    if not C0 > EPSILON_ZERO:
        return float("nan")
    return float(((C0 - Ce) / C0) * 100.0)


def calculate_adsorption_capacity(C0: float, Ce: float, V: float, m: float) -> float:
    """Calculate adsorption capacity: q = (C0 - Ce) × V / m (NaN if m ≤ 0)."""
    if not m > EPSILON_ZERO:
        return float("nan")
    return float((C0 - Ce) * V / m)


def calculate_Ce_from_absorbance(absorbance: float, slope: float, intercept: float) -> float:
    """
    Back-calculate concentration from absorbance: Ce = (A - intercept) / slope.

    The value is returned unclipped (a negative result means the signal is below
    the calibration intercept and is *not* a measured zero); NaN if the slope is
    unusable.
    """
    if not (np.isfinite(slope) and abs(slope) >= EPSILON_DIV):
        return float("nan")
    return float((absorbance - intercept) / slope)


OBSERVATION_OK = "ok"
OBSERVATION_EXCLUDED = "excluded"


def _fixed_or_column(data: pd.DataFrame, value: float | str) -> NDArray[np.floating[Any]]:
    if isinstance(value, str):
        return pd.to_numeric(data[value], errors="coerce").to_numpy(dtype=float)
    return np.full(len(data), float(value))


# What the stored Ce/qe errors contain.  They are standard errors (1 SD of the
# estimate), never replicate SDs, and never zero when unknown.
ERROR_BASIS_NONE = "not available: no uncertainty supplied for this input"


def calibration_error_basis(with_covariance: bool, with_reading: bool) -> str:
    """Text stating which components the calibration-propagated SE includes."""
    parts = ["slope and intercept SEs"]
    parts.append("their covariance" if with_covariance else "covariance not available (ignored)")
    if with_reading:
        parts.append("one reading with the calibration residual SD s(y/x)")
    return (
        "SE from the calibration: "
        + ", ".join(parts)
        + "; C0, V, m, dilution and replicate variability not included"
        + ("" if with_reading else "; sample-reading scatter not included")
    )


def build_uptake_table(
    data: pd.DataFrame,
    *,
    mode: str,
    signal_col: str,
    C0: float | str,
    V: float,
    m: float | str,
    calib_params: dict[str, Any] | None = None,
    extra_numeric: dict[str, str] | None = None,
    row_notes: dict[int, list[str]] | None = None,
) -> pd.DataFrame:
    """
    Per-observation concentration, uptake and removal with explicit eligibility.

    Every input row is returned with its ``source_row``, a ``status`` ('ok' or
    'excluded') and a ``note`` explaining any exclusion or flag.  Nothing is
    clipped, floored or silently dropped:

    * missing/non-numeric/non-finite inputs, non-positive C0 or mass, negative
      direct concentrations and Ce > C0 (negative uptake) are excluded with the
      reason;
    * in absorbance mode a signal below the calibration intercept (negative
      back-calculated Ce) or below the calibration LOD is *unresolved*: Ce, q and
      removal are NaN, never an exact zero or 100 % removal; values between LOD and
      LOQ are kept and flagged as semi-quantitative;
    * excluded rows have NaN analysis values; the note carries the calculated
      value where one exists.

    Parameters
    ----------
    data : DataFrame
        Analysis view (canonical columns; optional ``source_row``).
    mode : {'absorbance', 'direct'}
    signal_col : str
        'Absorbance' (absorbance mode) or the concentration column (direct mode).
    C0, m : float or str
        Fixed value, or the name of a per-row column.
    V : float
        Solution volume (L).
    calib_params : dict, optional
        Active calibration (absorbance mode).
    extra_numeric : dict, optional
        ``{column: label}`` of additional per-row inputs (e.g. time) that must be
        finite for the row to be usable.
    row_notes : dict, optional
        Import issues by source row (from :func:`prepare_analysis_data`).

    Returns
    -------
    DataFrame with columns source_row, C0, m, signal, C, C_error, q, q_error,
    error_basis, removal, status, note (generic names; callers rename them).
    ``C_error``/``q_error`` are standard errors from the calibration (absorbance
    mode; see ``error_basis`` for the components included) and NaN when no
    uncertainty is available (direct input) — never zero.
    """
    n = len(data)
    source = (
        pd.to_numeric(data["source_row"], errors="coerce").to_numpy()
        if "source_row" in data.columns
        else np.arange(1, n + 1)
    )
    C0_arr = _fixed_or_column(data, C0)
    m_arr = _fixed_or_column(data, m)
    signal = pd.to_numeric(data[signal_col], errors="coerce").to_numpy(dtype=float)
    notes_in = row_notes or {}

    slope = intercept = np.nan
    slope_se = intercept_se = 0.0
    cov_ab = reading_sd = None
    lod = loq = None
    if mode != "direct" and calib_params:
        slope = float(calib_params.get("slope", np.nan))
        intercept = float(calib_params.get("intercept", np.nan))
        slope_se = float(calib_params.get("std_err_slope", 0) or 0)
        intercept_se = float(calib_params.get("std_err_intercept", 0) or 0)
        cov_ab = calib_params.get("cov_slope_intercept")
        reading_sd = calib_params.get("std_err_estimate")
        lod = calib_params.get("lod_mgL")
        loq = calib_params.get("loq_mgL")
    error_basis = (
        ERROR_BASIS_NONE
        if mode == "direct"
        else calibration_error_basis(cov_ab is not None, reading_sd is not None)
    )

    rows = []
    for i in range(n):
        row_source = int(source[i]) if np.isfinite(source[i]) else i + 1
        note: list[str] = list(notes_in.get(row_source, []))
        status = OBSERVATION_OK
        conc = conc_se = q = q_se = removal = np.nan
        c0, mass = C0_arr[i], m_arr[i]

        bad_inputs = [
            label
            for label, value in (
                [("C0", c0), ("mass", mass), (signal_col, signal[i])]
                + [
                    (label, pd.to_numeric(data[col].iloc[i], errors="coerce"))
                    for col, label in (extra_numeric or {}).items()
                ]
            )
            if not np.isfinite(float(value))
        ]
        if bad_inputs:
            status = OBSERVATION_EXCLUDED
            if not note:
                note.append(f"missing or non-numeric {', '.join(bad_inputs)}")
        elif c0 <= 0:
            status = OBSERVATION_EXCLUDED
            note.append(f"C0 must be positive (C0 = {c0:g} mg/L)")
        elif mass <= 0:
            status = OBSERVATION_EXCLUDED
            note.append(f"adsorbent mass must be positive (m = {mass:g} g)")
        elif mode == "direct":
            if signal[i] < 0:
                status = OBSERVATION_EXCLUDED
                note.append(f"negative concentration ({signal[i]:g} mg/L)")
            else:
                conc = signal[i]
        else:
            back = calculate_Ce_from_absorbance(signal[i], slope, intercept)
            if not np.isfinite(back):
                status = OBSERVATION_EXCLUDED
                note.append("no valid calibration to convert absorbance")
            elif back < 0:
                status = OBSERVATION_EXCLUDED
                note.append(
                    f"absorbance {signal[i]:g} is below the calibration intercept "
                    f"{intercept:.4g}: back-calculated Ce = {back:.4g} mg/L; not quantifiable "
                    "(not treated as zero)"
                )
            elif lod is not None and np.isfinite(lod) and back < lod:
                status = OBSERVATION_EXCLUDED
                bound = 100.0 * (c0 - lod) / c0
                note.append(
                    f"below the calibration LOD ({lod:.3g} mg/L): back-calculated Ce = "
                    f"{back:.3g} mg/L is not a measured value; removal ≥ {bound:.1f} %"
                )
            else:
                conc = back
                _, conc_se = propagate_calibration_uncertainty(
                    signal[i],
                    slope,
                    intercept,
                    slope_se,
                    intercept_se,
                    float(cov_ab) if cov_ab is not None else 0.0,
                    absorbance_se=float(reading_sd) if reading_sd is not None else None,
                )
                if loq is not None and np.isfinite(loq) and back < loq:
                    note.append(f"below the calibration LOQ ({loq:.3g} mg/L): semi-quantitative")

        if status == OBSERVATION_OK:
            if conc > c0:
                status = OBSERVATION_EXCLUDED
                note.append(
                    f"Ce = {conc:.4g} mg/L exceeds C0 = {c0:.4g} mg/L (negative uptake, "
                    f"q = {(c0 - conc) * V / mass:.4g} mg/g): check C0, units or the measurement"
                )
                conc = conc_se = np.nan
            else:
                q = calculate_adsorption_capacity(c0, conc, V, mass)
                removal = calculate_removal_percentage(c0, conc)
                # Direct input carries no uncertainty: unavailable, never zero.
                # q = (C0 - Ce)·V/m with C0, V and m treated as exact.
                q_se = (V / mass) * conc_se

        rows.append(
            {
                "source_row": row_source,
                "C0": c0,
                "m": mass,
                "signal": signal[i],
                "C": conc,
                "C_error": conc_se,
                "q": q,
                "q_error": q_se,
                "error_basis": error_basis if status == OBSERVATION_OK else "",
                "removal": removal,
                "status": status,
                "note": "; ".join(note),
            }
        )
    columns = ["source_row", "C0", "m", "signal", "C", "C_error", "q", "q_error", "error_basis"]
    result = pd.DataFrame(rows, columns=columns + ["removal", "status", "note"])
    # Identifier, uncertainty and context columns travel with their observation.
    used = {signal_col, "source_row"} | set(extra_numeric or {})
    used |= {value for value in (C0, m) if isinstance(value, str)}
    for column in data.columns:
        if column in used:
            continue
        target = str(column)
        while target in result.columns:
            target = f"{target} (input)"
        result[target] = data[column].to_numpy()
    return result


UPTAKE_TABLE_COLUMNS = (
    "source_row",
    "C0",
    "m",
    "signal",
    "C",
    "C_error",
    "q",
    "q_error",
    "error_basis",
    "removal",
    "status",
    "note",
)


def uptake_results_frame(
    table: pd.DataFrame,
    *,
    leading: dict[str, Any] | None = None,
    conc_name: str = "Ce",
    capacity_name: str = "qe",
    include_c0: bool = False,
    include_signal: bool = False,
    include_errors: bool = True,
    sort_by: str | None = None,
) -> pd.DataFrame:
    """Rename a :func:`build_uptake_table` result to a tab's established columns."""
    frame = pd.DataFrame({"source_row": table["source_row"].to_numpy()})
    for name, values in (leading or {}).items():
        frame[name] = np.asarray(values)
    if include_c0:
        frame["C0_mgL"] = table["C0"].to_numpy()
    if include_signal:
        frame["Absorbance"] = table["signal"].to_numpy()
    frame[f"{conc_name}_mgL"] = table["C"].to_numpy()
    if include_errors:
        frame[f"{conc_name}_error"] = table["C_error"].to_numpy()
    frame[f"{capacity_name}_mg_g"] = table["q"].to_numpy()
    if include_errors:
        frame[f"{capacity_name}_error"] = table["q_error"].to_numpy()
        frame["error_basis"] = table["error_basis"].to_numpy()
    frame["removal_%"] = table["removal"].to_numpy()
    frame["status"] = table["status"].to_numpy()
    frame["note"] = table["note"].to_numpy()
    for column in table.columns:
        if column in UPTAKE_TABLE_COLUMNS:
            continue
        target = column
        while target in frame.columns:
            target = f"{target} (input)"
        frame[target] = table[column].to_numpy()
    if sort_by is not None:
        frame = frame.sort_values([sort_by, "source_row"], kind="stable", na_position="last")
    return frame.reset_index(drop=True)


def metadata_columns(frame: pd.DataFrame) -> list[str]:
    """Imported identifier/uncertainty/context columns carried by a results frame."""
    known = {
        "source_row",
        "C0_mgL",
        "Absorbance",
        "removal_%",
        "status",
        "note",
        "error_basis",
        "Time",
        "pH",
        "Mass_g",
        "Temperature_C",
        "Temperature_K",
    }
    for stem in ("Ce", "Ct", "qe", "qt"):
        known |= {f"{stem}_mgL", f"{stem}_mg_g", f"{stem}_error"}
    return [c for c in frame.columns if c not in known]


def uptake_calculation_result(frame: pd.DataFrame) -> "CalculationResult":
    """Wrap a results frame; failure (with the table) when no row is usable."""
    if frame.empty:
        return CalculationResult(
            success=False, data=frame, error="No valid observations: the table has no data rows."
        )
    if (frame["status"] == OBSERVATION_OK).sum() == 0:
        return CalculationResult(
            success=False,
            data=frame,
            error="No usable observations: " + (exclusion_summary(frame) or "all rows excluded"),
        )
    return CalculationResult(success=True, data=frame)


def eligible_observations(results: pd.DataFrame | None) -> pd.DataFrame:
    """Rows usable for calculations (status 'ok'); all rows for legacy tables."""
    if results is None:
        return pd.DataFrame()
    if "status" not in results.columns:
        return results
    return results[results["status"] == OBSERVATION_OK]


def observation_notice(results: pd.DataFrame | None) -> str | None:
    """One-sentence notice of excluded/flagged observations for a results table."""
    if results is None or "status" not in results.columns:
        return None
    n = len(results)
    excluded = int((results["status"] != OBSERVATION_OK).sum())
    flagged = int(((results["status"] == OBSERVATION_OK) & (results["note"] != "")).sum())
    if excluded == 0 and flagged == 0:
        return None
    parts = []
    if excluded:
        parts.append(
            f"{excluded} of {n} observation(s) are excluded from calculations and model fitting"
        )
    if flagged:
        parts.append(f"{flagged} observation(s) are used but flagged")
    return (
        "; ".join(parts)
        + ". They stay listed with their reasons: "
        + (exclusion_summary(results) or "")
    )


def exclusion_summary(results: pd.DataFrame | None, limit: int = 8) -> str | None:
    """Human-readable list of excluded/flagged rows, or None when there are none."""
    if results is None or "status" not in results.columns or "note" not in results.columns:
        return None
    flagged = results[(results["status"] != OBSERVATION_OK) | (results["note"] != "")]
    if flagged.empty:
        return None
    ordered = flagged.sort_values("source_row")
    items = [
        f"row {row} ({status}): {note}"
        for row, status, note in zip(
            ordered["source_row"].astype(int), ordered["status"], ordered["note"]
        )
    ]
    text = "; ".join(items[:limit])
    if len(items) > limit:
        text += f"; … and {len(items) - limit} more"
    return text


# =============================================================================
# PRESS STATISTIC (LEAVE-ONE-OUT CROSS-VALIDATION)
# =============================================================================
def calculate_press_details(
    model_func: Callable[..., Any],
    x_data: NDArray[np.floating[Any]],
    y_data: NDArray[np.floating[Any]],
    params: Any = None,
    bounds: tuple[Any, Any] | None = None,
    *,
    fit_setup: Callable[[NDArray[np.floating[Any]], NDArray[np.floating[Any]]], Any] | None = None,
) -> dict[str, Any]:
    """
    Leave-one-out PRESS and Q² with every fold refitted honestly.

    Each fold is refitted on the other observations with the original estimator's
    configuration: ``fit_setup(x_train, y_train) -> (p0, bounds)`` reproduces the
    starting-value policy and limits used for the full fit (otherwise ``params``
    and ``bounds`` are used as given), with the same iteration limit.  A fold whose
    refit or prediction fails is recorded and is never replaced by a prediction
    from the full-data parameters (that would include the held-out observation).
    PRESS and Q² are complete only when every fold succeeds; otherwise both are
    NaN (unavailable) and ``partial_press`` over the successful folds is reported
    separately — it is not PRESS.

    Returns ``{"status": 'complete'|'unavailable', "press", "q2", "n_folds",
    "n_failed", "failed_folds": [{"index", "x", "reason"}], "partial_press",
    "partial_folds", "message"}``.
    """
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    n = len(x)
    squared = np.full(n, np.nan)
    failed: list[dict[str, Any]] = []
    for i in range(n):
        keep = np.arange(n) != i
        x_train, y_train = x[keep], y[keep]
        try:
            if fit_setup is not None:
                p0, fold_bounds = fit_setup(x_train, y_train)
            else:
                p0, fold_bounds = params, bounds
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizeWarning)
                if fold_bounds is not None:
                    popt, _ = curve_fit(
                        model_func,
                        x_train,
                        y_train,
                        p0=p0,
                        bounds=fold_bounds,
                        maxfev=MAX_FIT_ITERATIONS,
                    )
                else:
                    popt, _ = curve_fit(
                        model_func, x_train, y_train, p0=p0, maxfev=MAX_FIT_ITERATIONS
                    )
        except (RuntimeError, ValueError, TypeError, ArithmeticError) as exc:
            failed.append({"index": i, "x": float(x[i]), "reason": f"refit failed: {exc}"})
            continue
        try:
            prediction = float(np.asarray(model_func(x[i : i + 1], *popt), dtype=float)[0])
        except (RuntimeError, ValueError, TypeError, ArithmeticError) as exc:
            failed.append({"index": i, "x": float(x[i]), "reason": f"prediction failed: {exc}"})
            continue
        if not np.isfinite(prediction):
            failed.append({"index": i, "x": float(x[i]), "reason": "non-finite prediction"})
            continue
        squared[i] = (y[i] - prediction) ** 2

    succeeded = np.isfinite(squared)
    complete = n > 0 and not failed
    press = float(np.sum(squared)) if complete else float("nan")
    if complete:
        message = f"PRESS from {n} leave-one-out refits."
    else:
        message = (
            f"PRESS/Q² unavailable: {len(failed)} of {n} leave-one-out refits failed"
            + (f" (first: x = {failed[0]['x']:.6g}, {failed[0]['reason']})" if failed else "")
            + "."
        )
    return {
        "status": "complete" if complete else "unavailable",
        "press": press,
        "q2": calculate_q2(press, y),
        "n_folds": n,
        "n_failed": len(failed),
        "failed_folds": failed,
        "partial_press": float(np.sum(squared[succeeded])) if succeeded.any() else float("nan"),
        "partial_folds": int(succeeded.sum()),
        "message": message,
    }


def calculate_press(
    model_func: Callable[..., Any],
    x_data: NDArray[np.floating[Any]],
    y_data: NDArray[np.floating[Any]],
    params: NDArray[np.floating[Any]],
    bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    *,
    fit_setup: Callable[[NDArray[np.floating[Any]], NDArray[np.floating[Any]]], Any] | None = None,
) -> float:
    """
    Calculate PRESS (Predicted Residual Error Sum of Squares) by leave-one-out refits.

    Returns NaN (unavailable) when any fold's refit or prediction fails; see
    :func:`calculate_press_details` for the fold record.

    Parameters
    ----------
    model_func : callable
        Model function f(x, *params)
    x_data : np.ndarray
        Independent variable data
    y_data : np.ndarray
        Dependent variable data
    params : np.ndarray
        Starting values for each fold refit (unless ``fit_setup`` is given)
    bounds : tuple, optional
        Parameter bounds for curve_fit (unless ``fit_setup`` is given)
    fit_setup : callable, optional
        ``(x_train, y_train) -> (p0, bounds)``: the original estimator's policy

    Returns
    -------
    float
        PRESS statistic (lower is better), or NaN when unavailable
    """
    details = calculate_press_details(
        model_func, x_data, y_data, params, bounds, fit_setup=fit_setup
    )
    return float(details["press"])


def resolve_temperature_unit(unit: str | None) -> str | None:
    """Return '°C' or 'K' for a declared temperature unit, None when undeclared."""
    if unit is None or str(unit).strip() == "":
        return None
    key = _normalize_unit_text(str(unit))
    table = _UNIT_TABLES["temperature"]
    if key not in table:
        raise ValueError(f"Unsupported temperature unit '{unit}'. Use °C or K.")
    return table[key][0]


def temperature_to_celsius_kelvin(
    values: Any, unit: str | None = None
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """
    Convert temperatures to (°C, K) using an explicit unit.

    With no declared unit the template convention (°C) applies, but values above
    ``UNITLESS_TEMPERATURE_LIMIT_C`` are refused (``ValueError``) instead of being
    guessed to be kelvin.  Both input modes use this helper so they agree.
    """
    temps = np.asarray(values, dtype=float)
    resolved = resolve_temperature_unit(unit)
    if resolved is None:
        finite = temps[np.isfinite(temps)]
        if finite.size and finite.max() > UNITLESS_TEMPERATURE_LIMIT_C:
            raise ValueError(
                f"Temperature values up to {finite.max():g} were given without a unit. "
                "Unitless temperatures are read as °C; declare the unit explicitly "
                "(e.g. a 'Temperature (K)' column) instead of relying on a guess."
            )
        resolved = "°C"
    if resolved == "K":
        return temps - 273.15, temps
    return temps, temps + 273.15


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
        - 'temperature_unit' (optional): '°C' or 'K'.  When absent, the template
          convention (°C) applies and kelvin-like values (> 200) are refused.
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
    return _temperature_results(temp_input, calib_params, "absorbance", include_uncertainty)


def _temperature_results(
    temp_input: dict[str, Any],
    calib_params: dict[str, Any] | None,
    mode: str,
    include_uncertainty: bool,
) -> "CalculationResult":
    """Shared implementation of both temperature calculators (same unit rule)."""
    df = temp_input["data"].copy()
    params = temp_input["params"]
    try:
        temps_C, temps_K = temperature_to_celsius_kelvin(
            pd.to_numeric(df["Temperature"], errors="coerce").to_numpy(dtype=float),
            temp_input.get("temperature_unit"),
        )
    except ValueError as exc:
        return CalculationResult(success=False, error=str(exc))

    table = build_uptake_table(
        df,
        mode=mode,
        signal_col="Absorbance" if mode != "direct" else "Ce",
        C0=params["C0"],
        V=params["V"],
        m=params["m"],
        calib_params=calib_params,
        extra_numeric={"Temperature": "temperature"},
        row_notes=temp_input.get("row_issues"),
    )
    below_zero = np.isfinite(temps_K) & (temps_K <= 0)
    for i in np.flatnonzero(below_zero & (table["status"] == OBSERVATION_OK).to_numpy()):
        table.loc[i, ["C", "C_error", "q", "q_error", "removal"]] = np.nan
        table.loc[i, "status"] = OBSERVATION_EXCLUDED
        table.loc[i, "note"] = f"temperature {temps_K[i]:g} K is not above absolute zero"
    frame = uptake_results_frame(
        table,
        leading={"Temperature_C": temps_C, "Temperature_K": temps_K},
        include_signal=mode != "direct",
        include_errors=include_uncertainty,
        sort_by="Temperature_C",
    )
    return uptake_calculation_result(frame)


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
        - 'temperature_unit' (optional): '°C' or 'K'.  When absent, the template
          convention (°C) applies and kelvin-like values (> 200) are refused
          rather than guessed.  Absorbance mode applies the same rule.
    include_uncertainty : bool, default=False
        If True, include uncertainty columns (set to 0 for direct input).

    Returns
    -------
    CalculationResult
        Result with DataFrame containing temperature-dependent results.
    """
    return _temperature_results(temp_input, None, "direct", include_uncertainty)


def calculate_q2(press: float, y_data: np.ndarray) -> float:
    """
    Calculate Q² (predictive R²) from PRESS.

    Q² = 1 - PRESS / SS_tot

    Interpretation:
    - Q² > 0.9: Excellent predictive ability
    - Q² > 0.7: Good predictive ability
    - Q² > 0.5: Acceptable predictive ability
    - Q² < 0.5: Poor predictive ability

    NaN when PRESS is unavailable or the observations have no variance.

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
    press = float(press)
    if not np.isfinite(press):
        return float("nan")  # PRESS unavailable
    ss_tot = float(np.sum((y_data - np.mean(y_data)) ** 2))
    if ss_tot < EPSILON_DIV:
        return float("nan")  # undefined for observations without variance
    return 1 - press / ss_tot


# =============================================================================
# ANALYSIS CONSISTENCY CHECKER
# =============================================================================
def check_mechanism_consistency(study_state: dict[str, Any]) -> dict[str, Any]:
    """
    Check internal consistency and reporting completeness.

    This function intentionally does not map a best-fitting kinetic or isotherm
    equation to an adsorption mechanism.  Such mappings are not mechanistically
    diagnostic.  The historical function name is retained for API compatibility.

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

    # Directional temperature check. A mismatch is a prompt to verify that the
    # compared experiments used equivalent conditions, not a thermodynamic proof
    # or a mechanism conflict.
    if thermo_params:
        delta_H = thermo_params.get("delta_H", 0)
        temp_effect = study_state.get(
            "temperature_effect"
        )  # Could be 'increases', 'decreases', or None

        if temp_effect:
            if delta_H > 0 and temp_effect == "decreases":
                checks.append(
                    {
                        "name": "Temperature Trend Review",
                        "status": "minor",
                        "message": (
                            "Capacity decreases while apparent ΔH is positive. Verify that all "
                            "temperature experiments used comparable concentrations, equilibrium "
                            "conditions, and the same Kd definition."
                        ),
                        "severity": "medium",
                    }
                )
                minor_issues += 1
            elif delta_H < 0 and temp_effect == "increases":
                checks.append(
                    {
                        "name": "Temperature Trend Review",
                        "status": "minor",
                        "message": (
                            "Capacity increases while apparent ΔH is negative. Verify that all "
                            "temperature experiments used comparable concentrations, equilibrium "
                            "conditions, and the same Kd definition."
                        ),
                        "severity": "medium",
                    }
                )
                minor_issues += 1
            else:
                checks.append(
                    {
                        "name": "Temperature Trend Review",
                        "status": "consistent",
                        "message": "No obvious directional mismatch under the reported conditions",
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

    # Determine overall status.
    #
    # "no_checks" is distinct from "consistent".  When no check applies — a
    # study with, say, only a converged isotherm fit and no thermodynamics —
    # zero checks run, and reporting that as "consistent" states a clean bill
    # of health that nothing was actually examined to support.  Callers must
    # render this state differently from a passing one.
    #
    # Note on the "conflicts" tier: no check currently raises it.  The
    # temperature-direction check emits a review prompt rather than a conflict,
    # because a directional mismatch between capacity and apparent ΔH is a
    # reason to re-examine experimental comparability, not proof of an error.
    # The tier and its counter are retained so a future check with genuine
    # conflict semantics has somewhere to report, and so the returned shape
    # stays stable for existing callers.
    if not checks:
        status = "no_checks"
        color = "gray"
        interpretation = (
            "No consistency checks applied to this study — nothing here supports "
            "or contradicts your results"
        )
    elif conflicts > 0:
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
        interpretation = (
            f"All {len(checks)} applicable check(s) passed; no internal inconsistencies found"
        )

    # Generate suggestions
    suggestions = []
    if status == "no_checks":
        suggestions.append(
            "Complete a thermodynamic analysis, or fit models with confidence intervals, "
            "to enable consistency checking"
        )
    if conflicts > 0:
        suggestions.append("Review experimental conditions and data quality")
        suggestions.append("Consider alternative models that may better explain the data")
    if minor_issues > 0:
        suggestions.append("Document any inconsistencies in your report")
        suggestions.append("Consider additional experiments to resolve ambiguities")
    if status == "consistent":
        suggestions.append(
            "No internal inconsistencies detected; scientific interpretation still requires domain evidence"
        )

    return {
        "status": status,
        "color": color,
        "checks": checks,
        "n_checks": len(checks),
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
        usable = eligible_observations(iso_results) if hasattr(iso_results, "columns") else None
        n_points = len(usable) if usable is not None else 0
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
def _get_model_registry() -> (
    tuple[dict[str, Callable[..., Any]], Callable[[str], Callable[..., Any] | None]]
):
    """Lazy import of model registry to avoid circular imports."""
    from .models import _MODEL_REGISTRY, get_model_by_name

    return _MODEL_REGISTRY, get_model_by_name


BOOTSTRAP_METHOD = (
    "residual bootstrap (residuals of the original fit resampled with replacement), "
    "each draw refitted with the original start/limit policy; percentile interval"
)


def bootstrap_parameter_intervals(
    model_func: Callable[..., Any],
    x_data: NDArray[np.floating[Any]],
    y_data: NDArray[np.floating[Any]],
    params: Any,
    n_bootstrap: int = BOOTSTRAP_DEFAULT_ITERATIONS,
    confidence: float = 0.95,
    *,
    bounds: tuple[Any, Any] | None = None,
    fit_setup: Callable[[NDArray[np.floating[Any]], NDArray[np.floating[Any]]], Any] | None = None,
    seed: int = BOOTSTRAP_DEFAULT_SEED,
    param_names: list[str] | None = None,
    progress_callback: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """
    Residual-bootstrap percentile intervals with a fully recorded execution.

    Each requested draw is attempted at most once (no retries, and no stopping on
    the stability of parameter estimates).  Each draw is refitted with the
    original estimator's configuration: ``fit_setup(x, y_draw) -> (p0, bounds)``
    reapplies the start/limit policy of the original fit, otherwise ``params`` is
    the start and ``bounds`` the limits; the iteration limit is the same.  The
    random generator is ``numpy.random.default_rng(seed)`` (no global state).

    The total work is capped deterministically, which can stop a run early: the
    function evaluations used so far are compared with
    ``BOOTSTRAP_EVALUATIONS_PER_DRAW`` × requested before each draw, so the last
    refit can exceed the limit (it is not an exact ceiling or a time limit).  Once
    it is reached no further draws are attempted (``attempted`` < ``requested``)
    and ``stopped`` states why.

    The interval is available only when at least ``BOOTSTRAP_MIN_SUCCESS`` draws
    and ``BOOTSTRAP_MIN_SUCCESS_FRACTION`` of the requested draws succeeded;
    otherwise ``ci_lower``/``ci_upper`` are NaN and ``reason`` says why.  An
    available interval is computed from the successful draws only, which may be
    fewer than requested (see :func:`bootstrap_summary_text`).  Returns
    ``{"status", "ci_lower", "ci_upper", "requested", "attempted", "successful",
    "failed", "failure_reasons", "seed", "confidence", "method", "param_names",
    "evaluations", "stopped", "reason"}``.
    """
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    theta = np.asarray(params, dtype=float)
    n_params = len(theta)
    names = list(param_names) if param_names else [f"p{i}" for i in range(n_params)]
    requested = int(n_bootstrap)
    rng = np.random.default_rng(seed)

    y_pred = np.asarray(model_func(x, *theta), dtype=float)
    residuals = y - y_pred
    draws: list[NDArray[np.floating[Any]]] = []
    failure_reasons: dict[str, int] = {}
    attempted = 0
    budget = BOOTSTRAP_EVALUATIONS_PER_DRAW * requested
    evaluations = 0
    stopped = ""
    for i in range(requested):
        if evaluations >= budget:
            stopped = (
                f"evaluation limit reached after {attempted} of {requested} draws "
                f"({evaluations} function evaluations used; limit {budget}, "
                "checked between refits)"
            )
            break
        if progress_callback and i % 50 == 0:
            progress_callback(i, requested, f"Bootstrap draw {i}/{requested}")
        y_draw = y_pred + residuals[rng.integers(0, len(x), size=len(x))]
        attempted += 1
        try:
            if fit_setup is not None:
                p0, draw_bounds = fit_setup(x, y_draw)
            else:
                p0, draw_bounds = theta, bounds
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=OptimizeWarning)
                if draw_bounds is not None:
                    fit = curve_fit(
                        model_func,
                        x,
                        y_draw,
                        p0=p0,
                        bounds=draw_bounds,
                        maxfev=MAX_FIT_ITERATIONS,
                        full_output=True,
                    )
                else:
                    fit = curve_fit(
                        model_func, x, y_draw, p0=p0, maxfev=MAX_FIT_ITERATIONS, full_output=True
                    )
            popt = np.asarray(fit[0], dtype=float)
            evaluations += int(fit[2].get("nfev", 0))
            if not np.all(np.isfinite(popt)):
                raise ValueError("non-finite parameter estimate")
        except (RuntimeError, ValueError, TypeError, ArithmeticError) as exc:
            if "maxfev" in str(exc) or "function evaluations" in str(exc):
                evaluations += MAX_FIT_ITERATIONS
            reason = f"{type(exc).__name__}: {str(exc)[:80]}"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            continue
        draws.append(popt)

    successful = len(draws)
    needed = max(BOOTSTRAP_MIN_SUCCESS, int(np.ceil(BOOTSTRAP_MIN_SUCCESS_FRACTION * requested)))
    out: dict[str, Any] = {
        "requested": requested,
        "attempted": attempted,
        "successful": successful,
        "failed": attempted - successful,
        "failure_reasons": failure_reasons,
        "seed": seed,
        "confidence": float(confidence),
        "method": BOOTSTRAP_METHOD,
        "param_names": names,
        "evaluations": evaluations,
        "stopped": stopped,
    }
    if successful < needed:
        out.update(
            status="unavailable",
            ci_lower=np.full(n_params, np.nan),
            ci_upper=np.full(n_params, np.nan),
            reason=(
                f"only {successful} of {requested} bootstrap refits succeeded "
                f"(at least {needed} required)"
            ),
        )
        return out
    alpha = (1 - confidence) / 2
    sample = np.vstack(draws)
    out.update(
        status="available",
        ci_lower=np.percentile(sample, alpha * 100, axis=0),
        ci_upper=np.percentile(sample, (1 - alpha) * 100, axis=0),
        reason="",
    )
    return out


def _bootstrap_cached_impl(
    model_name: str,
    x_data_tuple: tuple,
    y_data_tuple: tuple,
    params_tuple: tuple,
    n_bootstrap: int,
    confidence: float,
    bounds_lower: tuple | None,
    bounds_upper: tuple | None,
    seed: int,
) -> dict[str, Any]:
    """Cached bootstrap: every setting that changes the result is part of the key."""
    _, get_model_by_name = _get_model_registry()
    model_func = get_model_by_name(model_name)
    if model_func is None:
        raise ValueError(f"Unknown model: {model_name}")
    bounds = None if bounds_lower is None else (bounds_lower, bounds_upper)
    return bootstrap_parameter_intervals(
        model_func,
        np.array(x_data_tuple),
        np.array(y_data_tuple),
        np.array(params_tuple),
        n_bootstrap,
        confidence,
        bounds=bounds,
        seed=seed,
    )


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
    early_stopping: bool = False,
    use_cache: bool = True,
    *,
    bounds: tuple[Any, Any] | None = None,
    fit_setup: Callable[[NDArray[np.floating[Any]], NDArray[np.floating[Any]]], Any] | None = None,
    seed: int = BOOTSTRAP_DEFAULT_SEED,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """
    Bootstrap confidence intervals ``(ci_lower, ci_upper)`` for model parameters.

    See :func:`bootstrap_parameter_intervals` (which also returns the draw
    counts); NaN bounds mean the interval is unavailable.  ``early_stopping`` is
    accepted for compatibility and ignored: draws are never stopped on the
    stability of parameter means.  Results are cached for registered models
    when no progress callback or ``fit_setup`` is given; the cache key includes
    the data, starting values, limits, draw count, confidence level and seed.
    """
    del early_stopping  # no justified interval-stability rule exists
    model_name = None
    if use_cache and _STREAMLIT_AVAILABLE and progress_callback is None and fit_setup is None:
        model_registry, _ = _get_model_registry()
        for name, func in model_registry.items():
            if func is model_func:
                model_name = name
                break

    if model_name is not None:
        from .models import round_significant

        details = _bootstrap_cached(
            model_name,
            tuple(round_significant(x_data).tolist()),
            tuple(round_significant(y_data).tolist()),
            tuple(round_significant(params).tolist()),
            int(n_bootstrap),
            float(confidence),
            None if bounds is None else tuple(float(v) for v in bounds[0]),
            None if bounds is None else tuple(float(v) for v in bounds[1]),
            int(seed),
        )
    else:
        details = bootstrap_parameter_intervals(
            model_func,
            x_data,
            y_data,
            params,
            n_bootstrap,
            confidence,
            bounds=bounds,
            fit_setup=fit_setup,
            seed=seed,
            progress_callback=progress_callback,
        )
    return np.asarray(details["ci_lower"]), np.asarray(details["ci_upper"])


# Outcome of one bootstrap run: every requested draw refitted, an interval from the
# successful draws of an incomplete run, or no interval.
BOOTSTRAP_COMPLETE = "complete"
BOOTSTRAP_PARTIAL = "partial"
BOOTSTRAP_UNAVAILABLE = "unavailable"


def bootstrap_outcome(info: dict[str, Any] | None) -> str:
    """Classify a bootstrap run (see the BOOTSTRAP_* outcome constants)."""
    if not info or info.get("status") != "available":
        return BOOTSTRAP_UNAVAILABLE
    if info["successful"] < info["requested"]:
        return BOOTSTRAP_PARTIAL
    return BOOTSTRAP_COMPLETE


def bootstrap_summary_text(results: dict[str, Any]) -> str:
    """
    Draw counts, stop reason and settings of a stored bootstrap run ('' when none).

    The same text is used on the analysis pages and in the exports.  It always
    gives the attempted, failed and unattempted draws, why a run stopped early,
    and when an interval rests on fewer draws than requested.
    """
    info = results.get("bootstrap") if isinstance(results, dict) else None
    if not info:
        return ""
    requested, attempted = info["requested"], info.get("attempted", info["requested"])
    text = (
        f"{info['successful']} of {requested} bootstrap draws refitted "
        f"({attempted} attempted, {info['failed']} failed, {requested - attempted} not "
        f"attempted; seed {info['seed']}; {info['confidence']:.0%} percentile)"
    )
    if info.get("stopped"):
        text += f"; run stopped early: {info['stopped']}"
    outcome = bootstrap_outcome(info)
    if outcome == BOOTSTRAP_PARTIAL:
        text += f"; interval computed from the {info['successful']} successful draws only"
    elif outcome == BOOTSTRAP_UNAVAILABLE:
        text += f"; interval unavailable: {info.get('reason', '')}"
    return text


def display_bootstrap_intervals(results: dict[str, Any]) -> None:
    """Show stored bootstrap intervals with their draw counts, or why they are unavailable."""
    info = results.get("bootstrap") if isinstance(results, dict) else None
    if not info:
        return
    outcome = bootstrap_outcome(info)
    if outcome == BOOTSTRAP_UNAVAILABLE:
        st.warning(f"⚠️ Bootstrap interval unavailable: {bootstrap_summary_text(results)}.")
        return
    level = f"{info['confidence']:.0%}"
    display_results_table(
        {
            "Parameter": info["param_names"],
            f"Bootstrap {level} CI": [
                f"({lo:.6g}, {hi:.6g})" for lo, hi in zip(info["ci_lower"], info["ci_upper"])
            ],
        }
    )
    if outcome == BOOTSTRAP_PARTIAL:
        st.warning(f"⚠️ Incomplete bootstrap run: {bootstrap_summary_text(results)}.")
        st.caption(f"Method: {info['method']}.")
    else:
        st.caption(f"{bootstrap_summary_text(results)}. Method: {info['method']}.")


def store_bootstrap_result(
    result: dict[str, Any], details: dict[str, Any], model_name: str
) -> tuple[str, str]:
    """
    Attach a bootstrap run to a fit result.

    Returns ``(outcome, summary)`` with outcome ``BOOTSTRAP_COMPLETE``,
    ``BOOTSTRAP_PARTIAL`` (interval from the successful draws of an incomplete
    run) or ``BOOTSTRAP_UNAVAILABLE``.
    """
    result["bootstrap"] = details
    outcome = bootstrap_outcome(details)
    if outcome != BOOTSTRAP_UNAVAILABLE:
        result["bootstrap_ci_95"] = {
            name: (float(lo), float(hi))
            for name, lo, hi in zip(
                details["param_names"], details["ci_lower"], details["ci_upper"]
            )
        }
        result["bootstrap_n"] = details["successful"]  # successful draws, not requested
    else:
        result.pop("bootstrap_ci_95", None)
        result.pop("bootstrap_n", None)
    return outcome, f"{model_name}: {bootstrap_summary_text(result)}"


def report_bootstrap_outcome(summaries: list[tuple[str | bool, str]]) -> None:
    """
    Report per-model bootstrap outcomes with their actual draw counts.

    Only runs in which every requested draw was refitted are reported as
    complete; an interval from an incomplete run is reported as such.  Booleans
    are accepted as outcomes (True: complete, False: unavailable).
    """
    groups: dict[str, list[str]] = {
        BOOTSTRAP_UNAVAILABLE: [],
        BOOTSTRAP_PARTIAL: [],
        BOOTSTRAP_COMPLETE: [],
    }
    for outcome, text in summaries:
        if isinstance(outcome, bool):
            outcome = BOOTSTRAP_COMPLETE if outcome else BOOTSTRAP_UNAVAILABLE
        groups.get(outcome, groups[BOOTSTRAP_UNAVAILABLE]).append(text)
    if groups[BOOTSTRAP_UNAVAILABLE]:
        st.warning(
            "⚠️ Bootstrap interval unavailable — " + " | ".join(groups[BOOTSTRAP_UNAVAILABLE])
        )
    if groups[BOOTSTRAP_PARTIAL]:
        st.warning(
            "⚠️ Bootstrap intervals from incomplete runs — " + " | ".join(groups[BOOTSTRAP_PARTIAL])
        )
    if groups[BOOTSTRAP_COMPLETE]:
        st.success("✅ Bootstrap intervals — " + " | ".join(groups[BOOTSTRAP_COMPLETE]))


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
        normalized_sse, normalized_sse_reduced, aic, aicc, bic, sse, sst,
        residuals. ``chi_squared`` keys are retained as backward-compatible
        aliases; no inferential chi-square claim is made without known variances.
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

    # Relative/normalized SSE.  This historical adsorption error function is
    # not an inferential chi-square statistic unless observation variances are
    # independently known.
    from .models import relative_sse

    normalized_sse = relative_sse(residuals, y_pred, EPSILON_DIV)
    normalized_sse_reduced = normalized_sse / (n - n_params) if n > n_params else normalized_sse

    from .statistical_criteria import information_criteria

    aic, aicc, bic = information_criteria(float(ss_res), n, n_params)

    return {
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "rmse": rmse,
        "mae": mae,
        "normalized_sse": normalized_sse,
        "normalized_sse_reduced": normalized_sse_reduced,
        "chi_squared": normalized_sse,  # Backward-compatible alias
        "chi_squared_reduced": normalized_sse_reduced,  # Backward-compatible alias
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "sse": ss_res,
        "sst": ss_tot,
        "residuals": residuals,
    }


def calculate_akaike_weights(aic_values: list[float]) -> np.ndarray:
    """
    Calculate Akaike weights for model comparison.

    Weights are computed among the finite criterion values only.  A model whose
    criterion is unavailable (non-finite, e.g. undefined AICc) gets NaN — it has no
    weight rather than zero support — and all weights are NaN when no value is
    finite.  Callers must only pass fits to the same observations.
    """
    aic_array = np.array(aic_values, dtype=float)
    valid_mask = np.isfinite(aic_array)

    if not np.any(valid_mask):
        return np.full(len(aic_values), np.nan)

    aic_min = np.min(aic_array[valid_mask])
    delta_aic = aic_array - aic_min

    # Avoid overflow
    delta_aic = np.clip(delta_aic, 0, 700)

    weights = np.exp(-0.5 * delta_aic)
    weights[~valid_mask] = np.nan

    total = np.nansum(weights)
    return weights / total if total > 0 else weights


# Every fit in the application minimises the unweighted sum of squared residuals
# in q (mg/g); information criteria assume iid Gaussian residuals with constant
# variance.  Criteria are only comparable between fits to the same observations
# under this same error model.
FIT_ERROR_MODEL = "unweighted least squares in q (iid Gaussian residuals, constant variance)"

INFORMATION_CRITERIA = {"aic": "AIC", "aicc": "AICc", "bic": "BIC"}


def fit_observation_key(result: dict[str, Any]) -> tuple[str, int]:
    """Identity of the observations a fit used (order-independent), and their count."""
    x = result.get("x_data")
    y = result.get("y_data")
    if x is None or y is None:
        return ("unrecorded", int(result.get("n_points", -1) or -1))
    pairs = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    # Rounded as fit_model_with_ci rounds its (cached) inputs, so a fit to the same
    # observations gets the same key whichever path produced it.
    from .models import round_significant

    pairs = round_significant(pairs)
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    digest = hashlib.md5(pairs.tobytes()).hexdigest()[:16]
    return (digest, len(pairs))


def format_criterion(value: Any, digits: int = 2) -> str:
    """Display text for an information criterion ('undefined' when not finite)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "undefined"
    return f"{number:.{digits}f}" if np.isfinite(number) else "undefined"


def compare_information_criteria(
    fitted_models: dict[str, Any], criterion: str = "aicc"
) -> dict[str, Any]:
    """
    Rank converged fits by an information criterion, within comparable sets only.

    Fits are grouped by the observations they used (and their error model); a
    ranking is produced only inside a group and only among fits whose criterion is
    defined.  Undefined values (e.g. AICc when n ≤ k + 1) never produce a winner.

    Returns ``{"criterion", "best", "status", "message", "per_model", "groups"}``;
    ``per_model[name]`` has ``value``, ``delta``, ``weight``, ``set``, ``n``,
    ``note``.  ``status`` is 'ranked', 'unavailable', 'single' or 'none' for the
    primary set (the one with the most models).
    """
    label = INFORMATION_CRITERIA[criterion]
    converged = {
        name: result
        for name, result in (fitted_models or {}).items()
        if isinstance(result, dict) and result.get("converged")
    }
    out: dict[str, Any] = {
        "criterion": label,
        "best": None,
        "status": "none",
        "message": "No converged fits to compare.",
        "per_model": {},
        "groups": [],
    }
    if not converged:
        return out

    grouped: dict[tuple[Any, ...], list[str]] = {}
    for name, result in converged.items():
        key = (fit_observation_key(result), result.get("error_model", FIT_ERROR_MODEL))
        grouped.setdefault(key, []).append(name)
    ordered = sorted(grouped.items(), key=lambda kv: (-len(kv[1]), -kv[0][0][1]))

    for set_index, (key, names) in enumerate(ordered, start=1):
        n_obs = key[0][1]
        values = {}
        for name in names:
            raw = converged[name].get(criterion)
            values[name] = float(raw) if raw is not None else float("nan")
        finite = {name: v for name, v in values.items() if np.isfinite(v)}
        if len(names) == 1:
            status = "single"
        elif len(finite) < 2:
            status = "unavailable"
        else:
            status = "ranked"
        weights: dict[str, float] = {}
        best = None
        if status == "ranked":
            names_f = list(finite)
            w = calculate_akaike_weights([finite[n] for n in names_f])
            weights = dict(zip(names_f, (float(x) for x in w)))
            best = min(finite, key=lambda n: finite[n])
        minimum = min(finite.values()) if finite else float("nan")
        for name in names:
            value = values[name]
            p = int(converged[name].get("num_params", 0) or 0)
            if np.isfinite(value):
                note = (
                    ""
                    if set_index == 1
                    else "fitted to different observations (not ranked with set 1)"
                )
            elif criterion == "aicc":
                note = f"AICc undefined: n = {n_obs} ≤ k + 1 with k = p + 1 = {p + 1}"
            else:
                note = f"{label} unavailable"
            out["per_model"][name] = {
                "value": value if np.isfinite(value) else float("nan"),
                "delta": value - minimum
                if status == "ranked" and np.isfinite(value)
                else float("nan"),
                "weight": weights.get(name, float("nan")),
                "set": set_index,
                "n": n_obs,
                "note": note,
            }
        out["groups"].append(
            {"set": set_index, "models": names, "n": n_obs, "status": status, "best": best}
        )

    primary = out["groups"][0]
    out["status"] = primary["status"]
    out["best"] = primary["best"]
    models = ", ".join(primary["models"])
    if primary["status"] == "ranked":
        weight = out["per_model"][primary["best"]]["weight"]
        message = (
            f"Lowest {label} among fits to the same {primary['n']} observations ({models}): "
            f"{primary['best']} ({label} weight {weight:.0%})."
        )
    elif primary["status"] == "unavailable":
        message = (
            f"No {label}-based ranking: {label} is defined for fewer than two of the fits to "
            f"the same {primary['n']} observations ({models})."
        )
    else:
        message = f"Only {models} was fitted to these observations; no ranking."
    others = [g for g in out["groups"][1:]]
    if others:
        listed = "; ".join(f"{', '.join(g['models'])} (n = {g['n']})" for g in others)
        message += f" Fitted to different observations and not ranked with these: {listed}."
    out["message"] = message
    return out


def parameter_status_column(results: dict[str, Any], names: list[str | None]) -> list[str]:
    """Per-row identifiability status for a parameter table (None = derived row)."""
    status = results.get("param_status") or {}
    return ["derived" if name is None else status.get(name, "—") for name in names]


def fit_diagnostics_text(results: dict[str, Any]) -> str:
    """Parameters at a limit, poorly identified or strongly correlated ('' if none)."""
    if not isinstance(results, dict):
        return ""
    parts = []
    limited = results.get("bounds_hit") or []
    if limited:
        parts.append(
            "at limit: " + "; ".join(limited) + " (value set by the limit; SE/CI not valid)"
        )
    status = results.get("param_status") or {}
    poor = [name for name, value in status.items() if value == "poorly identified"]
    if poor:
        parts.append("poorly identified: " + ", ".join(poor))
    correlated = results.get("correlated_params") or []
    if correlated:
        parts.append(
            "strongly correlated: " + ", ".join(f"{a}–{b} (r = {r:+.3f})" for a, b, r in correlated)
        )
    return "; ".join(parts)


def display_fit_diagnostics(
    results: dict[str, Any], limit_notes: dict[str, str] | None = None
) -> None:
    """
    Show parameters that stopped at a limit or are poorly identified.

    This is reported separately from numerical convergence: a converged fit can
    still be limited by a bound, or match the data closely while leaving its
    parameters undetermined.
    """
    limited = results.get("bounds_hit") or []
    status = results.get("param_status") or {}
    if limited:
        notes = [
            text for name, text in (limit_notes or {}).items() if status.get(name) == "at limit"
        ]
        st.warning(
            "⚠️ **Stopped at a limit:** "
            + "; ".join(limited)
            + ". The value is set by the limit, not determined by the data, and its "
            "standard error and confidence interval are not valid. A poor fit here "
            "reflects the limit or the data range, not necessarily the model."
            + ("" if not notes else " " + " ".join(notes))
        )
    poor = [name for name, value in status.items() if value == "poorly identified"]
    correlated = results.get("correlated_params") or []
    if poor or correlated:
        text = ""
        if poor:
            text += (
                f"ℹ️ **Poorly identified:** {', '.join(poor)} — the 95% confidence interval "
                "is at least as wide as the estimate. "
            )
        if correlated:
            pairs = ", ".join(f"{a} and {b} (r = {r:+.3f})" for a, b, r in correlated)
            text += (
                f"{'' if poor else 'ℹ️ '}**Strongly correlated:** {pairs}; the data determine "
                "their combination better than each parameter. "
            )
        text += (
            "Close agreement between the fitted curve and the data (R²) does not make "
            "these parameters precise."
        )
        st.info(text)
    st.caption(f"{FIT_ESTIMATOR_NOTE}.")


# How every model fit in the application is estimated (screens and exports).
FIT_ESTIMATOR_NOTE = (
    "Estimated by unweighted least squares in q (iid residuals, constant variance); "
    "standard errors and CIs are linearised (Wald) intervals from the residual scatter. "
    "Ce/qe errors in the data table were not used as weights"
)


def _finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def model_comparison_table(
    fitted_models: dict[str, Any], include_press: bool = False
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Model-comparison table shared by the analysis screens and the exports.

    Columns are labelled with the criterion they contain (AIC, AICc, BIC); an
    undefined criterion is NaN (shown as '—'), never a number.  ``ΔAICc`` and
    ``AICc weight`` are computed only among fits to the same observations
    (``Set``) whose AICc is defined; see :func:`compare_information_criteria`.
    """
    comparison = compare_information_criteria(fitted_models, "aicc")
    rows = []
    for name, result in (fitted_models or {}).items():
        if not (isinstance(result, dict) and result.get("converged")):
            continue
        info = comparison["per_model"][name]
        note = "; ".join(part for part in (info["note"], fit_diagnostics_text(result)) if part)
        row: dict[str, Any] = {
            "Model": name,
            "n": info["n"],
            "R²": _finite_or_nan(result.get("r_squared")),
            "Adj-R²": _finite_or_nan(result.get("adj_r_squared")),
            "RMSE": _finite_or_nan(result.get("rmse")),
            "Relative SSE": _finite_or_nan(result.get("normalized_sse", result.get("chi_squared"))),
            "AIC": _finite_or_nan(result.get("aic")),
            "AICc": _finite_or_nan(result.get("aicc")),
            "BIC": _finite_or_nan(result.get("bic")),
            "ΔAICc": info["delta"],
            "AICc weight": info["weight"],
            "Set": info["set"],
            "Note": note,
        }
        if include_press:
            row["PRESS"] = _finite_or_nan(result.get("press"))
            row["Q²"] = _finite_or_nan(result.get("q2"))
            press_details = result.get("press_details") or {}
            if press_details.get("status") == "unavailable":
                row["Note"] = "; ".join(p for p in (row["Note"], press_details["message"]) if p)
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["Set", "AICc"], na_position="last", kind="stable")
        table = table.reset_index(drop=True)
    return table, comparison


CAPACITY_CRITERION = "Langmuir qm (mg/g)"
CAPACITY_CRITERION_NOTE = (
    "Ordered by fitted Langmuir qm only (the quantity shown on the comparison page). "
    "Langmuir R² describes fit quality and is not combined with qm; capacities are "
    "comparable only when sorbate, units, concentration range, pH, temperature, dosage "
    "and contact conditions are matched."
)


def capacity_comparison(studies_data: dict[str, Any]) -> pd.DataFrame:
    """
    The single capacity comparison used by the comparison page and the exports.

    Studies are ordered by fitted Langmuir qm; fit quality (Langmuir R²) and the
    identifiability of qm are reported alongside, never combined into a score.
    A study without a converged Langmuir fit is listed as not available — it is
    not given qm = 0 and is not ranked.
    """
    rows = []
    for name, data in (studies_data or {}).items():
        langmuir = ((data or {}).get("isotherm_models_fitted") or {}).get("Langmuir") or {}
        if langmuir.get("converged"):
            ci = (langmuir.get("ci_95") or {}).get("qm", (np.nan, np.nan))
            rows.append(
                {
                    "Study": name,
                    CAPACITY_CRITERION: _finite_or_nan(langmuir.get("params", {}).get("qm")),
                    "qm 95% CI lower": _finite_or_nan(ci[0]),
                    "qm 95% CI upper": _finite_or_nan(ci[1]),
                    "qm status": (langmuir.get("param_status") or {}).get("qm", "—"),
                    "Langmuir R² (fit quality)": _finite_or_nan(langmuir.get("r_squared")),
                    "Note": "",
                }
            )
        else:
            rows.append(
                {
                    "Study": name,
                    CAPACITY_CRITERION: np.nan,
                    "qm 95% CI lower": np.nan,
                    "qm 95% CI upper": np.nan,
                    "qm status": "—",
                    "Langmuir R² (fit quality)": np.nan,
                    "Note": "not available: no converged Langmuir fit (not ranked)",
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    available = table[CAPACITY_CRITERION].notna()
    ordered = table[available].sort_values(CAPACITY_CRITERION, ascending=False, kind="stable")
    table = pd.concat([ordered, table[~available]], ignore_index=True)
    n_ranked = int(available.sum())
    order = pd.Series(list(range(1, n_ranked + 1)) + [None] * (len(table) - n_ranked))
    table.insert(0, "Order by qm", order.astype("Int64"))
    return table


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

    # Repeated numeric values are reported, not penalised: they may be valid
    # replicates, which only identifiers (or the user) can establish.
    notices = []
    analysis_cols = [
        c for c in data.columns if c in CANONICAL_UNITS or c in ("C0_mgL", "Ce_mgL", "qe_mg_g")
    ]
    if analysis_cols:
        repeated = data.duplicated(subset=analysis_cols, keep=False)
        if repeated.any():
            n_rep = int(data.duplicated(subset=analysis_cols).sum())
            other = [c for c in data.columns if c not in analysis_cols and c != "source_row"]
            groups = data[repeated].groupby(analysis_cols, dropna=False)
            differing = [c for c in other if (groups[c].nunique(dropna=False) > 1).any()]
            if differing and not data.duplicated(subset=analysis_cols + other).any():
                notices.append(
                    f"{n_rep} row(s) repeat the numeric values of another row but differ in "
                    f"{', '.join(map(str, differing))}; they are kept as separate observations "
                    "(replicates)."
                )
            else:
                notices.append(
                    f"{n_rep} row(s) repeat the numeric values of another row. All are kept "
                    "as observations; if they are replicates, a SampleID or Replicate column "
                    "identifies them, and if they are accidental copies, remove them."
                )

    # Check the analysis columns (not identifiers or metadata) for outliers
    numeric_cols = data[analysis_cols].select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        values = data[col].dropna()
        if len(values) > 3:
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = ((values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)).sum()
            if outliers > 0:
                quality_score -= outliers * 3
                issues.append(f"{outliers} potential outliers in {col}")

    # Check for negative values where not expected (numeric values only: an imported
    # column of that name may hold text, e.g. a re-imported export read as text)
    if data_type == "isotherm":
        for col in ["Ce_mgL", "qe_mg_g", "C0_mgL"]:
            if col in data.columns and (pd.to_numeric(data[col], errors="coerce") < 0).any():
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
        "notices": notices,
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
            reasons.append("B1 is positive")

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
            reasons.append("Positive fitted rate parameter in the configured range")

    elif name == "Pseudo-second order" or name == "PSO":
        qe = params.get("qe", 0)
        k2 = params.get("k2", 0)
        if qe > 0 and k2 > 0:
            score += 5
        if k2 > 0:
            score += 5
            reasons.append("Positive fitted rate parameter")

    elif name == "Elovich":
        alpha = params.get("alpha", 0)
        beta = params.get("beta", 0)
        if alpha > 0 and beta > 0:
            score += 10
            reasons.append("Positive fitted Elovich parameters")

    elif name == "Weber-Morris" or name == "Intraparticle diffusion":
        kid = params.get("kid", 0)
        C = params.get("C", 0)
        if kid > 0:
            score += 5
        if C > 0:
            score += 5
            reasons.append("C > 0: Boundary layer effect present")
        else:
            reasons.append(
                "C ≈ 0: fitted line passes near the origin; inspect diffusion diagnostics"
            )

    return score, reasons


# =============================================================================
# MODEL RECOMMENDATION
# =============================================================================
def recommend_best_models(
    model_results: dict[str, dict[str, Any]], model_type: str = "isotherm", top_n: int = 3
) -> list[dict[str, Any]]:
    """
    Heuristic model ordering from several criteria (not a statistical test).

    Each entry has a ``heuristic_score`` (0–100 points from the weights below).
    It is not a probability or a statistical confidence; ``confidence`` is kept
    only as a deprecated alias of ``heuristic_score`` for backward compatibility.

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
    - AICc weight (25%), only among fits to the same observations whose AICc is
      defined (see :func:`compare_information_criteria`); other fits get no AICc
      contribution and a NaN ``aicc_weight``
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

        # AICc is compared across comparable models below
        aicc = _finite_or_nan(result.get("aicc"))

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

    # AICc weights, only within the primary set of comparable fits
    if rankings:
        comparison = compare_information_criteria(
            {rank["model"]: rank["result"] for rank in rankings}, "aicc"
        )
        primary = comparison["groups"][0]
        for rank in rankings:
            weight = float("nan")
            if primary["status"] == "ranked" and rank["model"] in primary["models"]:
                weight = comparison["per_model"][rank["model"]]["weight"]
            if np.isfinite(weight):
                rank["score"] += weight * 25
            rank["aicc_weight"] = weight
            rank["aic_weight"] = weight  # Alias
            # Heuristic points (0-100), not a statistical confidence.
            rank["heuristic_score"] = min(100, rank["score"])
            rank["confidence"] = rank["heuristic_score"]  # deprecated alias
            # Generate rationale
            reasons = []
            if rank["adj_r2"] >= 0.99:
                reasons.append("Excellent fit (Adj-R² ≥ 0.99)")
            elif rank["adj_r2"] >= 0.95:
                reasons.append("Good fit (Adj-R² ≥ 0.95)")
            if not np.isfinite(weight):
                reasons.append("No comparable AICc")
            elif weight >= 0.5:
                reasons.append("Strong AICc support")
            elif weight >= 0.2:
                reasons.append("Moderate AICc support")
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
    T_K: np.ndarray,
    Kd: np.ndarray,
    confidence_level: float = 0.95,
    Kd_se: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Calculate thermodynamic parameters from Van't Hoff analysis.

    Van't Hoff equation: ln(Kd) = -ΔH°/RT + ΔS°/R

    When ``Kd_se`` is supplied the function performs **both** ordinary
    least-squares (OLS) and weighted least-squares (WLS) regression on
    the Van't Hoff plot. The WLS weights are ``1 / σ²(ln Kd)`` where
    ``σ(ln Kd) = Kd_se / Kd`` (delta method), used as relative weights: WLS
    standard errors are scaled by the weighted residual scatter.  In the
    application ``Kd_se`` carries calibration uncertainty only (not mass,
    volume, C0, temperature or replicate variability) and the correlation
    between points sharing one calibration is ignored, so the WLS result is
    not a complete uncertainty budget.  A point without a finite ``Kd_se``
    is omitted from WLS only; OLS always uses every valid point.

    Parameters
    ----------
    T_K : np.ndarray
        Temperature in Kelvin
    Kd : np.ndarray
        Distribution coefficients
    confidence_level : float
        Confidence level for intervals (default 0.95)
    Kd_se : np.ndarray or None
        Standard errors of Kd values (from ``propagate_kd_uncertainty``).
        When provided, WLS results are included alongside OLS results.

    Returns
    -------
    dict
        Thermodynamic parameters including ΔH°, ΔS°, ΔG°, R², statistics.
        When ``Kd_se`` is provided the dict also contains keys prefixed
        with ``wls_`` holding the weighted-least-squares estimates and
        the ``ln_Kd_se`` array used as weights.
    """
    if len(T_K) < 2 or len(Kd) < 2:
        return {"success": False, "error": "Need at least 2 temperature points"}

    # Filter valid Kd values.  A missing Kd uncertainty only removes the point
    # from the weighted regression, never from the ordinary fit.
    valid_mask = (Kd > 0) & np.isfinite(Kd) & np.isfinite(T_K) & (T_K > 0)
    if Kd_se is not None:
        Kd_se = np.asarray(Kd_se, dtype=float)

    T_K = T_K[valid_mask]
    Kd = Kd[valid_mask]
    if Kd_se is not None:
        Kd_se = Kd_se[valid_mask]

    if len(T_K) < 2:
        return {"success": False, "error": "Insufficient valid data points"}

    try:
        x = 1 / T_K  # 1/T
        y = np.log(Kd)  # ln(Kd)

        # -----------------------------------------------------------------
        # OLS regression (always performed)
        # -----------------------------------------------------------------
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

        result = {
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

        # -----------------------------------------------------------------
        # WLS regression (when Kd_se is provided)
        # -----------------------------------------------------------------
        if Kd_se is not None:
            # Delta method: σ(ln Kd) = σ(Kd) / Kd
            ln_Kd_se = Kd_se / np.maximum(Kd, EPSILON_DIV)

            # Only points with a finite, non-zero propagated uncertainty are weighted
            valid_w = np.isfinite(ln_Kd_se) & (ln_Kd_se > EPSILON_DIV)
            if np.sum(valid_w) >= 2:
                xw = x[valid_w]
                yw = y[valid_w]
                wts = 1.0 / ln_Kd_se[valid_w] ** 2

                # Weighted linear regression:  minimise Σ w_i (y_i - a - b x_i)²
                # β = (X'WX)^{-1} X'Wy   where X = [1, x]
                W = np.diag(wts)
                X = np.column_stack([np.ones_like(xw), xw])
                XtW = X.T @ W
                XtWX = XtW @ X
                try:
                    XtWX_inv = np.linalg.inv(XtWX)
                    beta_wls = XtWX_inv @ (XtW @ yw)

                    wls_intercept = beta_wls[0]
                    wls_slope = beta_wls[1]

                    # Residual variance (weighted)
                    nw = len(xw)
                    yw_pred = wls_intercept + wls_slope * xw
                    wls_residuals = yw - yw_pred
                    wls_mse = np.sum(wts * wls_residuals**2) / (nw - 2) if nw > 2 else 0

                    # Standard errors from covariance matrix
                    cov_beta = wls_mse * XtWX_inv
                    wls_se_intercept = np.sqrt(max(cov_beta[0, 0], 0))
                    wls_se_slope = np.sqrt(max(cov_beta[1, 1], 0))

                    # Thermodynamic params from WLS
                    wls_delta_H = -wls_slope * R_GAS_CONSTANT / 1000
                    wls_delta_S = wls_intercept * R_GAS_CONSTANT
                    wls_delta_G = wls_delta_H - T_K * wls_delta_S / 1000

                    wls_delta_H_se = wls_se_slope * R_GAS_CONSTANT / 1000
                    wls_delta_S_se = wls_se_intercept * R_GAS_CONSTANT

                    if nw > 2:
                        wls_t_crit = t_dist.ppf((1 + confidence_level) / 2, nw - 2)
                        wls_delta_H_ci = wls_t_crit * wls_delta_H_se
                        wls_delta_S_ci = wls_t_crit * wls_delta_S_se
                    else:
                        wls_delta_H_ci = wls_delta_H_se * 2
                        wls_delta_S_ci = wls_delta_S_se * 2

                    # Weighted R²
                    ss_tot_w = np.sum(wts * (yw - np.average(yw, weights=wts)) ** 2)
                    ss_res_w = np.sum(wts * wls_residuals**2)
                    wls_r_squared = 1 - ss_res_w / ss_tot_w if ss_tot_w > EPSILON_DIV else 0

                    result.update(
                        {
                            "ln_Kd_se": ln_Kd_se,
                            "Kd_se": Kd_se,
                            "wls_delta_H": wls_delta_H,
                            "wls_delta_H_se": wls_delta_H_se,
                            "wls_delta_H_ci": wls_delta_H_ci,
                            "wls_delta_S": wls_delta_S,
                            "wls_delta_S_se": wls_delta_S_se,
                            "wls_delta_S_ci": wls_delta_S_ci,
                            "wls_delta_G": wls_delta_G,
                            "wls_slope": wls_slope,
                            "wls_intercept": wls_intercept,
                            "wls_r_squared": wls_r_squared,
                            "wls_n_points": nw,
                        }
                    )
                except np.linalg.LinAlgError:
                    logger.warning("WLS matrix inversion failed; OLS results only.")
            else:
                logger.warning("Too few valid uncertainty points for WLS.")

        return result

    except Exception as e:
        logger.error(f"Failed to calculate thermodynamic parameters: {e}")
        return {"success": False, "error": str(e)}


# Operational distribution ratios used by the thermodynamics page (kd_method_id).
KD_DEFINITIONS: dict[str, tuple[str, str]] = {
    "dimensionless": ("Kd = (C0 − Ce)/Ce", "dimensionless"),
    "mass_based": ("Kd = qe/Ce", "L/g"),
    "volume_corrected": ("Kd = qe·m/(Ce·V)", "dimensionless"),
}
APPARENT_THERMO_NOTE = (
    "Apparent values from a van't Hoff fit of ln Kd with an operational distribution "
    "ratio; not standard-state thermodynamic quantities"
)


def kd_definition(thermo: dict[str, Any] | None) -> str:
    """The Kd definition and units behind a thermodynamic result."""
    method = (thermo or {}).get("kd_method_id")
    if method in KD_DEFINITIONS:
        formula, units = KD_DEFINITIONS[method]
        return f"{formula} ({units})"
    return "Kd definition not recorded"


def thermo_value(thermo: dict[str, Any] | None, key: str) -> float:
    """A stored scalar thermodynamic quantity, or NaN when missing or invalid."""
    raw: Any = (thermo or {}).get(key)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def apparent_delta_g(thermo: dict[str, Any] | None, T_K: float = 298.15) -> float:
    """Apparent ΔG (kJ/mol) at ``T_K`` from the fitted ΔH and ΔS; NaN when either is missing."""
    delta_H = thermo_value(thermo, "delta_H")
    delta_S = thermo_value(thermo, "delta_S")
    if not (np.isfinite(delta_H) and np.isfinite(delta_S)):
        return float("nan")
    return delta_H - T_K * delta_S / 1000


def delta_g_series(
    thermo: dict[str, Any] | None,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """
    Temperatures (K) and apparent ΔG (kJ/mol) from a stored result.

    Accepts ΔG stored as an array/list aligned with ``temperatures`` or as a
    {temperature: value} mapping; anything missing, misaligned or non-numeric is
    NaN (never 0).
    """
    thermo = thermo or {}
    temps = np.atleast_1d(np.asarray(thermo.get("temperatures", []), dtype=float))
    stored = thermo.get("delta_G")
    values = np.full(len(temps), np.nan)
    if isinstance(stored, dict):
        for i, T in enumerate(temps):
            for key in (T, float(T), round(float(T), 2), str(T)):
                if key in stored:
                    try:
                        values[i] = float(stored[key])
                    except (TypeError, ValueError):
                        pass
                    break
    elif stored is not None:
        try:
            array = np.atleast_1d(np.asarray(stored, dtype=float))
        except (TypeError, ValueError):
            array = np.array([])
        if len(array) == len(temps):
            values = array.copy()
    values[~np.isfinite(values)] = np.nan
    return temps, values


def sign_label(
    value: Any,
    negative: str,
    positive: str,
    unavailable: str = "—",
    zero: str = "Zero",
) -> str:
    """
    Label the sign of a quantity, or report that it is unavailable.

    A plain ``"neg" if x < 0 else "pos"`` silently converts *missing* into
    *positive*, because ``nan < 0`` and ``None < 0`` are both falsy.  In a
    generated results table that turns "this study has no ΔG" into the
    affirmative claim "this study has a positive ΔG".  Reporting tables must
    distinguish a measured sign from an absent value, so non-finite and missing
    inputs return ``unavailable`` rather than falling through to ``positive``.

    Parameters
    ----------
    value : Any
        The quantity whose sign is being described. May be ``None``, ``nan``,
        ``inf``, or anything not coercible to float.
    negative : str
        Label to use when ``value < 0``.
    positive : str
        Label to use when ``value > 0``. Exact zero is labelled separately.
    unavailable : str
        Label to use when ``value`` is missing or non-finite (default ``"—"``).

    Returns
    -------
    str
        One of ``negative``, ``positive`` or ``unavailable``.

    Examples
    --------
    >>> sign_label(-12.4, "Exothermic", "Endothermic")
    'Exothermic'
    >>> sign_label(float("nan"), "Exothermic", "Endothermic")
    '—'
    >>> sign_label(None, "Negative", "Positive")
    '—'
    """
    if value is None:
        return unavailable
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return unavailable
    if not np.isfinite(numeric):
        return unavailable
    return negative if numeric < 0 else positive if numeric > 0 else zero


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

    # Report thermodynamic trends without assigning an adsorption mechanism.
    if delta_H < 0:
        interpretations["enthalpy"] = f"Exothermic (ΔH° = {delta_H:.2f} kJ/mol < 0)"
    else:
        interpretations["enthalpy"] = f"Endothermic (ΔH° = {delta_H:.2f} kJ/mol > 0)"

    # Entropy interpretation
    if delta_S > 0:
        interpretations["entropy"] = (
            f"Positive apparent ΔS° ({delta_S:.2f} J/(mol·K)) for the selected Kd convention"
        )
    else:
        interpretations["entropy"] = (
            f"Negative apparent ΔS° ({delta_S:.2f} J/(mol·K)) for the selected Kd convention"
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
                interpretations["delta_g_sign"] = (
                    "Negative at all temperatures for the selected Kd convention"
                )
            elif all(g > 0 for g in G_values):
                interpretations["delta_g_sign"] = (
                    "Positive at all temperatures for the selected Kd convention"
                )
            else:
                interpretations["delta_g_sign"] = "Changes sign across the measured temperatures"

    interpretations["caveat"] = (
        "These values are apparent unless Kd is converted to a justified standard-state "
        "thermodynamic equilibrium constant. They do not identify adsorption mechanism."
    )

    return interpretations


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

        interpretation = (
            f"Apparent activation energy Ea = {Ea:.2f} kJ/mol. "
            "Ea alone does not identify the adsorption mechanism."
        )

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
    absorbance_se: float | None = None,
) -> tuple[float, float]:
    """
    Propagate calibration uncertainty to a back-calculated concentration.

    Ce = (Abs - intercept) / slope.  First-order (delta-method) SE from the
    slope and intercept standard errors and their covariance (pass the fitted
    ``cov_slope_intercept``; 0 means it is ignored).  The uncertainty of the
    sample reading itself is included only when ``absorbance_se`` is given —
    no instrument precision is assumed.  With ``absorbance_se`` equal to the
    calibration's residual SD s(y/x) this is the usual inverse-prediction SE
    for one reading, (s/b)·sqrt(1 + 1/n + (y0 - ȳ)²/(b²·Sxx)).

    Ce is returned unclipped (a negative value is below the calibration
    intercept, not a measured zero); (NaN, NaN) when the slope is unusable.
    """
    if not (np.isfinite(slope) and abs(slope) >= EPSILON_DIV):
        return float("nan"), float("nan")

    Ce = (absorbance - intercept) / slope

    # Partial derivatives
    dCe_dAbs = 1 / slope
    dCe_dSlope = -(absorbance - intercept) / slope**2
    dCe_dIntercept = -1 / slope

    signal_se = 0.0 if absorbance_se is None else float(absorbance_se)

    variance = (
        (dCe_dAbs * signal_se) ** 2
        + (dCe_dSlope * slope_se) ** 2
        + (dCe_dIntercept * intercept_se) ** 2
        + 2 * dCe_dSlope * dCe_dIntercept * cov_slope_intercept
    )

    Ce_se = np.sqrt(max(0, variance))

    return float(Ce), float(Ce_se)


def propagate_kd_uncertainty(
    method_id: str,
    C0: float,
    Ce: np.ndarray,
    qe: np.ndarray,
    m: float,
    V: float,
    Ce_se: np.ndarray,
    qe_se: np.ndarray | None = None,
) -> np.ndarray:
    """
    Propagate Ce/qe uncertainties to the distribution coefficient Kd.

    Uses first-order error propagation (delta method) for each Kd formula.

    Parameters
    ----------
    method_id : str
        One of: 'dimensionless', 'mass_based', 'volume_corrected'
    C0 : float
        Initial concentration (mg/L)
    Ce : np.ndarray
        Equilibrium concentrations (mg/L)
    qe : np.ndarray
        Adsorption capacities (mg/g)
    m : float
        Adsorbent mass (g)
    V : float
        Solution volume (L)
    Ce_se : np.ndarray
        Standard error of Ce values
    qe_se : np.ndarray or None
        Standard error of an *independently measured* qe.  When None (the
        application's case), qe is computed from Ce by mass balance,
        qe = (C0 − Ce)·V/m, so qe and Ce are fully (negatively) correlated and
        the total derivative dKd/dCe is used; C0, V and m are treated as exact.

    Returns
    -------
    np.ndarray
        Standard error of Kd at each temperature point.

    Notes
    -----
    Propagation formulae by Kd method:

    **Dimensionless** Kd = (C0 − Ce) / Ce

        ∂Kd/∂Ce = −C0 / Ce²
        σ(Kd) = |∂Kd/∂Ce| × σ(Ce) = (C0 / Ce²) × σ(Ce)

    **Mass-based** Kd = qe / Ce

        qe from mass balance: Kd = (C0 − Ce)·V/(m·Ce), σ(Kd) = (V/m)·(C0/Ce²)·σ(Ce)
        independent qe:       σ(Kd)² = (1/Ce)² σ(qe)² + (qe/Ce²)² σ(Ce)²

    **Volume-corrected** Kd = (qe × m) / (Ce × V)

        qe from mass balance this equals (C0 − Ce)/Ce: σ(Kd) = (C0/Ce²)·σ(Ce);
        independent qe: same structure as mass-based with an extra m/V factor.

    Treating a mass-balance qe as independent of Ce (the previous behaviour)
    understated σ(Kd), because both terms of dKd/dCe have the same sign.
    """
    Ce = np.asarray(Ce, dtype=float)
    Ce_se = np.asarray(Ce_se, dtype=float)
    qe = np.asarray(qe, dtype=float)
    Ce_safe = np.maximum(Ce, EPSILON_DIV)
    V_safe = max(V, EPSILON_DIV)
    m_safe = max(m, EPSILON_DIV)

    if method_id == "dimensionless":
        # Kd = (C0 - Ce) / Ce = C0/Ce - 1;  |dKd/dCe| = C0 / Ce²
        Kd_se = C0 / Ce_safe**2 * Ce_se
    elif qe_se is None:
        # qe = (C0 - Ce)·V/m: total derivative through Ce (fully correlated)
        if method_id == "mass_based":
            Kd_se = (V_safe / m_safe) * C0 / Ce_safe**2 * Ce_se
        elif method_id == "volume_corrected":
            Kd_se = C0 / Ce_safe**2 * Ce_se
        else:
            Kd_se = np.full_like(Ce, np.nan)
    else:
        qe_se = np.asarray(qe_se, dtype=float)
        if method_id == "mass_based":
            # Kd = qe / Ce with independent qe:  ∂Kd/∂qe = 1/Ce, ∂Kd/∂Ce = -qe/Ce²
            Kd_se = np.sqrt((qe_se / Ce_safe) ** 2 + (qe * Ce_se / Ce_safe**2) ** 2)
        elif method_id == "volume_corrected":
            Kd_se = np.sqrt(
                (m * qe_se / (Ce_safe * V_safe)) ** 2
                + (m * qe * Ce_se / (Ce_safe**2 * V_safe)) ** 2
            )
        else:
            Kd_se = np.full_like(Ce, np.nan)

    return np.abs(Kd_se)


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
def detect_replicates(
    data: pd.DataFrame,
    x_col: str,
    tolerance: float | None = None,
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Summarise replicate groups (mean, std, count of the numeric columns).

    Replicates are grouped by an explicit identifier column (``group_col``,
    e.g. a condition or replicate-set ID) when given, otherwise by *exactly*
    equal ``x_col`` values.  Values are never rounded into groups: nearby but
    different conditions are not replicates.  ``tolerance`` is accepted for
    compatibility and ignored.

    Returns
    -------
    pd.DataFrame with mean, std, count for each group
    """
    del tolerance  # grouping by rounding fabricated replicate groups
    data = data.copy()
    key = group_col if group_col is not None else x_col
    if len(data) == 0 or key not in data.columns:
        return data.copy()
    data["_x_group"] = data[key]

    # Get numeric columns for aggregation
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "_x_group"]

    # Group and aggregate
    agg_dict = {col: ["mean", "std", "count"] for col in numeric_cols}
    # pandas-stubs versions that still support Python 3.10 cannot express this
    # valid multi-aggregation mapping precisely, so retain pandas runtime
    # validation and isolate the compatibility cast at the API boundary.
    grouped = data.groupby("_x_group").agg(cast(Any, agg_dict))

    # Flatten column names
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    grouped = grouped.rename(columns={"_x_group": key})

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
    calib_df = current_study_state.get("calib_df_input")
    calib_quality = (
        assess_data_quality(calib_df, "calibration").get("quality_score", 0)
        if calib_df is not None
        else 0
    )

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
    """Validate a study name for creation.

    Constraints:
    - Non-empty after stripping whitespace
    - Maximum 100 characters
    - No HTML-sensitive characters (< > & " ') to prevent injection
    - Must not duplicate an existing study name
    """
    import re

    name = (name or "").strip()

    if not name:
        return False, "Please enter a study name."

    if len(name) > 100:
        return False, "Study name must be 100 characters or fewer."

    # Reject characters that are dangerous in HTML contexts
    if re.search(r'[<>&"\']', name):
        return False, "Study name cannot contain <, >, &, \", or ' characters."

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


# =============================================================================
# CALIBRATION CANDIDATE VALIDATION AND ACTIVATION
# =============================================================================
# A replacement calibration is validated as a *candidate* and only then made
# active.  An invalid candidate deactivates the previous calibration instead of
# leaving it in force for data it was not built from, and every result that was
# derived from absorbance through the previous calibration is invalidated.
# Results derived from direct concentration input do not depend on the
# calibration and are left untouched.

# Results that depend on the calibration when their input is in absorbance mode.
CALIBRATION_DEPENDENT_RESULTS: dict[str, tuple[str, ...]] = {
    "isotherm_input": (
        "isotherm_results",
        "isotherm_models_fitted",
        "langmuir_params_nl",
        "freundlich_params_nl",
        "temkin_params_nl",
    ),
    "kinetic_input": (
        "kinetic_results_df",
        "kinetic_models_fitted",
        "pso_params_nonlinear",
        "pfo_params_nonlinear",
    ),
    "dosage_input": ("dosage_results",),
    "ph_effect_input": ("ph_effect_results",),
    "temp_effect_input": ("temp_effect_results", "thermo_params"),
}

# The fitted absorbance change across the calibrated range must exceed this many
# residual standard deviations; otherwise the curve shows no measurable response
# and cannot be inverted to concentrations.
CALIBRATION_MIN_RESPONSE_SD = 3.0


def calibration_data_id(calib_df: pd.DataFrame | None, confidence_level: float = 0.95) -> str:
    """Stable identity of a calibration candidate (data and confidence level)."""
    if calib_df is None:
        return ""
    hashed = pd.util.hash_pandas_object(calib_df.reset_index(drop=True), index=False)
    payload = hashed.to_numpy(dtype=np.uint64).tobytes()
    digest = hashlib.md5(payload + str(confidence_level).encode()).hexdigest()
    return digest[:16]


def build_calibration(
    calib_df: pd.DataFrame | None, confidence_level: float = 0.95
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Validate a candidate calibration and compute its parameters.

    Returns ``(params, None)`` for a usable calibration or ``(None, reason)``.
    Rejected: missing data, non-finite values, fewer than 3 standards, fewer than
    2 distinct concentrations, validation errors (e.g. negative concentrations),
    and curves without a measurable response (|slope|·Δc ≤ 3·s_y/x).
    """
    if calib_df is None or len(calib_df) == 0:
        return None, "No calibration data."
    missing = [c for c in ("Concentration", "Absorbance") if c not in calib_df.columns]
    if missing:
        return None, f"Calibration data lack column(s): {', '.join(missing)}."

    conc = pd.to_numeric(calib_df["Concentration"], errors="coerce").to_numpy(dtype=float)
    absb = pd.to_numeric(calib_df["Absorbance"], errors="coerce").to_numpy(dtype=float)
    rows = (
        pd.to_numeric(calib_df["source_row"], errors="coerce").to_numpy()
        if "source_row" in calib_df.columns
        else np.arange(1, len(calib_df) + 1)
    )
    # A standard with a missing or non-finite value is excluded *visibly*: its row
    # is recorded with the calibration instead of being dropped silently.
    bad = ~(np.isfinite(conc) & np.isfinite(absb))
    excluded = [int(r) for r in rows[bad]]
    conc, absb = conc[~bad], absb[~bad]
    n = len(conc)
    if n < 3:
        detail = (
            f" (standard row(s) {', '.join(map(str, excluded))} have missing or non-finite values)"
            if excluded
            else ""
        )
        return None, (
            f"Insufficient calibration data: {n} usable standard(s); at least 3 are required"
            f"{detail}."
        )
    if len(np.unique(conc)) < 2:
        return None, "All calibration concentrations are identical; the slope is undefined."

    from .validation import validate_calibration_data

    report = validate_calibration_data(conc, absb)
    if not report.is_valid:
        return None, "; ".join(e.message for e in report.errors)

    slope, intercept, r_value, p_value, std_err = stats.linregress(conc, absb)
    x_mean = float(np.mean(conc))
    sxx = float(np.sum((conc - x_mean) ** 2))
    y_pred = slope * conc + intercept
    residuals = absb - y_pred
    se_estimate = float(np.sqrt(np.sum(residuals**2) / (n - 2)))
    response_span = abs(float(slope)) * float(np.ptp(conc))
    if not np.isfinite(slope) or response_span <= max(
        CALIBRATION_MIN_RESPONSE_SD * se_estimate, EPSILON_DIV
    ):
        return None, (
            "The calibration shows no measurable absorbance response: the fitted change "
            f"across the calibrated range ({response_span:.3g}) does not exceed "
            f"{CALIBRATION_MIN_RESPONSE_SD:g}× the residual standard deviation ({se_estimate:.3g})."
        )

    se_intercept = se_estimate * np.sqrt(1 / n + x_mean**2 / sxx)
    t_val = float(t_dist.ppf(1 - (1 - confidence_level) / 2, n - 2))
    r_squared = float(r_value**2)
    params: dict[str, Any] = {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "adj_r_squared": 1 - (1 - r_squared) * (n - 1) / (n - 2) if n > 2 else r_squared,
        "p_value": float(p_value),
        "std_err_slope": float(std_err),
        "std_err_intercept": float(se_intercept),
        "std_err_estimate": se_estimate,
        # Cov(intercept, slope) of ordinary least squares: -x̄·s²/Sxx.
        "cov_slope_intercept": float(-x_mean * se_estimate**2 / sxx),
        "slope_ci_95": (float(slope - t_val * std_err), float(slope + t_val * std_err)),
        "intercept_ci_95": (
            float(intercept - t_val * se_intercept),
            float(intercept + t_val * se_intercept),
        ),
        "confidence_level": confidence_level,
        "equation": f"Abs = {slope:.4f} × C + {intercept:.4f}",
        "n_points": n,
        "concentration_range": (float(conc.min()), float(conc.max())),
        "residuals": residuals.tolist(),
        "y_pred": y_pred.tolist(),
        "calibration_id": calibration_data_id(calib_df, confidence_level),
        "excluded_standards": excluded,
    }
    if slope > 0 and se_estimate > 0:
        params["lod_mgL"] = 3 * se_estimate / slope
        params["loq_mgL"] = 10 * se_estimate / slope
    return params, None


def _invalidate_calibration_dependents(study_state: dict[str, Any]) -> list[str]:
    """Clear results derived through the calibration; keep direct-input results."""
    cleared = []
    for input_key, result_keys in CALIBRATION_DEPENDENT_RESULTS.items():
        inp = study_state.get(input_key)
        if isinstance(inp, dict) and inp.get("input_mode") == "direct":
            continue
        for key in result_keys:
            value = study_state.get(key)
            if value is None or (hasattr(value, "__len__") and len(value) == 0):
                continue
            study_state[key] = {} if isinstance(value, dict) else None
            cleared.append(key)
    return cleared


def apply_calibration_update(study_state: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate the study's stored calibration data and activate it atomically.

    Returns ``{"status": ..., "error": ..., "invalidated": [...]}`` with status
    'activated', 'invalid', 'removed', 'unchanged' or 'none'.  The attempted data
    stay in ``calib_df_input``; an invalid candidate sets ``calibration_params``
    to None and records ``calibration_error``.
    """
    candidate = study_state.get("calib_df_input")
    confidence = study_state.get("confidence_level", 0.95)
    if candidate is None:
        had_calibration = (
            study_state.get("calibration_params") is not None
            or study_state.get("calibration_error") is not None
        )
        study_state["calibration_params"] = None
        study_state["calibration_error"] = None
        study_state["previous_calib_df"] = None
        study_state["_calibration_candidate_id"] = None
        if not had_calibration:
            return {"status": "none", "error": None, "invalidated": []}
        return {
            "status": "removed",
            "error": None,
            "invalidated": _invalidate_calibration_dependents(study_state),
        }

    candidate_id = calibration_data_id(candidate, confidence)
    if candidate_id == study_state.get("_calibration_candidate_id"):
        return {
            "status": "unchanged",
            "error": study_state.get("calibration_error"),
            "invalidated": [],
        }
    active = study_state.get("calibration_params")
    previous = study_state.get("previous_calib_df")
    built_from_candidate = active is not None and (
        active.get("calibration_id") == candidate_id
        or (isinstance(previous, pd.DataFrame) and previous.equals(candidate))
    )
    if built_from_candidate:
        study_state["_calibration_candidate_id"] = candidate_id
        return {"status": "unchanged", "error": None, "invalidated": []}

    params, error = build_calibration(candidate, confidence)
    study_state["_calibration_candidate_id"] = candidate_id
    invalidated = _invalidate_calibration_dependents(study_state)
    if params is None:
        study_state["calibration_params"] = None
        study_state["calibration_error"] = error
        study_state["previous_calib_df"] = None
        return {"status": "invalid", "error": error, "invalidated": invalidated}

    study_state["calibration_params"] = params
    study_state["calibration_error"] = None
    study_state["previous_calib_df"] = candidate.copy()
    return {"status": "activated", "error": None, "invalidated": invalidated}
