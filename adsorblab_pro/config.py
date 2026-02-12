# config.py
"""
AdsorbLab Pro v2.0.0 - Configuration Module
============================================

Centralized configuration for all settings, constants, and defaults.
Addresses review item: Configuration Management improvements.
"""

from typing import Any, TypedDict

import numpy as np


class QualityGradeInfo(TypedDict):
    """Type definition for quality grade information."""

    min_score: int
    min_r2: float
    status: str
    status_type: str
    label: str


# =============================================================================
# VERSION INFO
# =============================================================================
from . import __version__ as VERSION

__all__ = [
    # Version
    "VERSION",
    # File upload settings
    "MAX_FILE_SIZE_MB",
    "MAX_FILE_SIZE_BYTES",
    "ALLOWED_FILE_TYPES",
    # Numerical constants
    "EPSILON_DIV",
    "EPSILON_ZERO",
    "EPSILON_LOG",
    "R_GAS_CONSTANT",
    "PI_SQUARED",
    "MAX_FIT_ITERATIONS",
    "BOOTSTRAP_DEFAULT_ITERATIONS",
    "BOOTSTRAP_MIN_SUCCESS",
    # Validation constants
    "TEMP_MIN_KELVIN",
    "TEMP_MAX_KELVIN",
    "KD_WARNING_THRESHOLD",
    "FUZZY_MATCH_CUTOFF",
    "MIN_DATA_POINTS",
    # Model configuration
    "ISOTHERM_MODELS",
    "KINETIC_MODELS",
    "MULTICOMPONENT_MODELS",
    "MULTICOMPONENT_GUIDANCE",
    # Quality thresholds
    "QUALITY_THRESHOLDS",
    "QUALITY_GRADES",
    "MECHANISM_CRITERIA",
    # Plot settings
    "PLOT_TEMPLATE",
    "FONT_FAMILY",
    "EXPORT_DPI",
    # Session state
    "DEFAULT_SESSION_STATE",
    "DEFAULT_GLOBAL_SESSION_STATE",
    "SESSION_INPUT_KEYS_TO_CLEAR",
    "SESSION_WIDGET_PREFIXES_TO_CLEAR",
    "STUDY_METRIC_DATA_KEYS",
    # Functions
    "get_calibration_grade",
    "get_grade_from_score",
    "get_grade_from_r_squared",
]

# =============================================================================
# FILE UPLOAD SECURITY (Addresses Security Review 5.1)
# =============================================================================
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024  # 10 MB
ALLOWED_FILE_TYPES = ["xlsx", "xls", "csv"]
PI_SQUARED = np.pi**2
TEMP_MIN_KELVIN = 250
TEMP_MAX_KELVIN = 400
KD_WARNING_THRESHOLD = 0.1
FUZZY_MATCH_CUTOFF = 0.8
MIN_DATA_POINTS = 3
BOOTSTRAP_MIN_SUCCESS = 10
# =============================================================================
# NUMERICAL CONSTANTS
# =============================================================================
# EPSILON_DIV: Primary safeguard for division by zero and numerical stability
# Used throughout the application for:
# - Preventing division by zero in model calculations
# - Clamping minimum parameter values
# - Threshold for "effectively zero" comparisons
# All modules should import from config.py for consistency
# =============================================================================
EPSILON_DIV = 1e-10  # Prevent division by zero (primary constant)
EPSILON_ZERO = 1e-9  # Threshold for "effectively zero" checks
EPSILON_LOG = 1e-12  # Extra safety margin for log/exp operations
R_GAS_CONSTANT = 8.314  # J/(mol·K)
MAX_FIT_ITERATIONS = 10000
BOOTSTRAP_DEFAULT_ITERATIONS = 500

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
ISOTHERM_MODELS = {
    "Langmuir": {
        "params": ["qm", "KL"],
        "n_params": 2,
        "formula": r"q_e = \frac{q_m K_L C_e}{1 + K_L C_e}",
        "description": "Monolayer adsorption on homogeneous surface",
    },
    "Freundlich": {
        "params": ["KF", "n"],
        "n_params": 2,
        "formula": r"q_e = K_F C_e^{1/n}",
        "description": "Multilayer adsorption on heterogeneous surface",
    },
    "Temkin": {
        "params": ["B1", "KT"],
        "n_params": 2,
        "formula": r"q_e = B_1 \ln(K_T C_e)",
        "description": "Heat of adsorption decreases linearly with coverage",
    },
    "Sips": {
        "params": ["qm", "Ks", "ns"],
        "n_params": 3,
        "formula": r"q_e = \frac{q_m (K_s C_e)^{n_s}}{1 + (K_s C_e)^{n_s}}",
        "description": "Langmuir-Freundlich hybrid",
    },
}

# =============================================================================
# MULTI-COMPONENT COMPETITIVE ADSORPTION MODELS
# =============================================================================
# For predicting adsorption in systems with multiple competing adsorbates
# Uses single-component isotherm parameters to predict competitive behavior
# =============================================================================

MULTICOMPONENT_MODELS = {
    "Extended-Langmuir": {
        "params": ["qm_i", "KL_i", "KL_all"],
        "description": "Butler-Ockrent competitive Langmuir",
        "formula": r"q_{e,i} = \frac{q_{m,i} K_{L,i} C_{e,i}}{1 + \sum_j K_{L,j} C_{e,j}}",
        "assumptions": [
            "Monolayer adsorption",
            "Equal maximum coverage for all species",
            "No lateral interactions between adsorbates",
            "Ideal competition (no synergistic/antagonistic effects)",
        ],
        "reference": "Butler & Ockrent (1930). J. Phys. Chem., 34, 2841-2859",
    },
    "Extended-Freundlich": {
        "params": ["Kf_i", "n_i", "Kf_all", "n_all"],
        "description": "Sheindorf-Rebhun-Sheintuch (SRS) model",
        "formula": r"q_{e,i} = K_{F,i} C_{e,i} \left(\sum_j C_{e,j}\right)^{1/n_i - 1}",
        "assumptions": [
            "Heterogeneous surface",
            "Exponential distribution of adsorption energies",
            "Competition coefficients assumed equal (simplified)",
        ],
        "reference": "Sheindorf et al. (1981). J. Colloid Interface Sci., 79, 136-142",
    },
}

# Guidance on multi-component modeling
MULTICOMPONENT_GUIDANCE = {
    "when_to_use": [
        "Real wastewater with multiple contaminants",
        "Binary/ternary mixture studies",
        "Selectivity and separation factor analysis",
        "Process design for mixed waste streams",
    ],
    "limitations": [
        "Requires single-component isotherms for each species first",
        "Assumes ideal competition (may not hold for all systems)",
        "Does not account for synergistic/antagonistic effects",
        "Accuracy decreases with increasing number of components",
    ],
    "alternatives": [
        "IAST (Ideal Adsorbed Solution Theory) - more rigorous",
        "Real Adsorbed Solution Theory (RAST) - non-ideal systems",
        "Experimental binary/ternary isotherms",
    ],
}

KINETIC_MODELS = {
    "PFO": {
        "params": ["qe", "k1"],
        "n_params": 2,
        "formula": r"q_t = q_e(1 - e^{-k_1 t})",
        "description": "Pseudo-first order (Lagergren)",
    },
    "PSO": {
        "params": ["qe", "k2"],
        "n_params": 2,
        "formula": r"q_t = \frac{k_2 q_e^2 t}{1 + k_2 q_e t}",
        "description": "Pseudo-second order (Ho-McKay)",
        "warning": 'PSO fit does NOT imply chemisorption. ~90% of studies show PSO "best fit" regardless of actual mechanism.',
    },
    "rPSO": {
        "params": ["qe", "k2"],
        "conditions": ["C0", "m", "V"],
        "n_params": 2,
        "n_conditions": 3,
        "formula": r"q_t = \frac{k_2 q_e^2 t}{1 + k_2 q_e t \cdot \varphi}",
        "formula_detail": r"\varphi = 1 + \frac{q_e \cdot m}{C_0 \cdot V}",
        "description": "Revised PSO with concentration correction (Bullen et al., 2021)",
        "reference": "Bullen et al. (2021). Langmuir, 37(10), 3189-3201. DOI: 10.1021/acs.langmuir.1c00142",
        "advantage": "Reduces fitting errors by ~66% across varying experimental conditions",
        "requires_conditions": True,  # Indicates C0, m, V must be provided
    },
    "rPSO_simple": {
        "params": ["qe", "k2"],
        "n_params": 2,
        "formula": r"q_t = \frac{k_2 q_e^2 t}{1 + k_2 q_e t \cdot \varphi}",
        "description": "Revised PSO (simplified, fixed experimental conditions)",
        "reference": "Bullen et al. (2021). Langmuir, 37(10), 3189-3201",
        "note": "Use when C0, m, V are known constants entered separately",
    },
    "Elovich": {
        "params": ["alpha", "beta"],
        "n_params": 2,
        "formula": r"q_t = \frac{1}{\beta} \ln(1 + \alpha \beta t)",
        "description": "Chemisorption on heterogeneous surfaces",
    },
    "IPD": {
        "params": ["kid", "C"],
        "n_params": 2,
        "formula": r"q_t = k_{id} t^{0.5} + C",
        "description": "Intraparticle diffusion (Weber-Morris)",
    },
}

# =============================================================================
# DATA QUALITY THRESHOLDS
# =============================================================================
QUALITY_THRESHOLDS = {
    "calibration": {
        "min_points": 5,
        "ideal_points": 8,
        "min_r_squared": 0.995,
        "ideal_r_squared": 0.999,
    },
    "isotherm": {
        "min_points": 6,
        "ideal_points": 10,
        "min_r_squared": 0.95,
        "ideal_r_squared": 0.99,
    },
    "kinetic": {
        "min_points": 8,
        "ideal_points": 12,
        "min_r_squared": 0.95,
        "ideal_r_squared": 0.99,
    },
}
# =============================================================================
# MECHANISM DETERMINATION CRITERIA
# =============================================================================
MECHANISM_CRITERIA = {
    "delta_H": {
        "physical": (0, 40),  # kJ/mol (absolute)
        "mixed": (40, 80),  # kJ/mol (absolute)
        "chemical": (80, float("inf")),  # kJ/mol (absolute)
    },
    "Ea": {
        "physical": (0, 20),  # kJ/mol
        "activated": (20, 40),  # kJ/mol
        "chemical": (40, float("inf")),  # kJ/mol
    },
    "delta_G": {
        "physical": (-20, 0),  # kJ/mol
        "mixed": (-40, -20),  # kJ/mol
        "chemical": (float("-inf"), -40),  # kJ/mol
    },
}

PLOT_TEMPLATE = "simple_white"
FONT_FAMILY = "Times New Roman"
EXPORT_DPI = 300

# =============================================================================
# DEFAULT SESSION STATE
# =============================================================================
DEFAULT_SESSION_STATE: dict[str, Any] = {
    # Global settings
    "unit_system": "mg/g",
    "confidence_level": 0.95,
    "input_mode_global": "absorbance",  # absorbance=calibration, direct=Ce/Ct
    # Calibration
    "calib_df_input": None,
    "calibration_params": None,
    "previous_calib_df": None,
    # Isotherm study
    "isotherm_input": None,
    "isotherm_results": None,
    "isotherm_models_fitted": {},
    # Kinetic study
    "kinetic_input": None,
    "kinetic_results_df": None,
    "kinetic_models_fitted": {},
    # Effect studies
    "dosage_input": None,
    "dosage_results": None,
    "ph_effect_input": None,
    "ph_effect_results": None,
    "temp_effect_input": None,
    "temp_effect_results": None,
    # Thermodynamics
    "thermo_params": None,
    # NEW: Competitive adsorption
    "competitive_input": None,
    "competitive_results": None,

    # Data quality
    "data_quality_reports": {},
    # Validation
    "validation_results": None,
}


# =============================================================================
# DEFAULT GLOBAL SESSION STATE (root-level st.session_state)
# =============================================================================
# These keys live at the ROOT of st.session_state (not inside a study).
# They are initialized once at app start.
DEFAULT_GLOBAL_SESSION_STATE: dict[str, Any] = {
    "first_time": True,
    # Sidebar upload/editor caches
    "uploaded_calib_data": None,
    "uploaded_iso_data": None,
    "uploaded_kin_data": None,
    "uploaded_dos_data": None,
    "uploaded_ph_data": None,
    "uploaded_temp_data": None,
    "uploaded_competitive_data": None,
}

# =============================================================================
# SESSION STATE SOURCE OF TRUTH (cleanup + workflow metrics)
# =============================================================================
# NOTE:
# - Multi-study data lives under st.session_state.studies[study_name].
# - Some widgets and upload/editor caches live at the root of st.session_state.
#   These constants define exactly what should be cleared on study switch.

# Root-level (global) caches used by sidebar uploads / editors (NOT per-study)
SESSION_INPUT_KEYS_TO_CLEAR: tuple[str, ...] = (
    "uploaded_calib_data",
    "uploaded_iso_data",
    "uploaded_kin_data",
    "uploaded_dos_data",
    "uploaded_ph_data",
    "uploaded_temp_data",
    "uploaded_competitive_data",  # reserved for future UI
)

# Root-level widget key prefixes to clear when switching/adding studies
SESSION_WIDGET_PREFIXES_TO_CLEAR: tuple[str, ...] = (
    # Sidebar data-input widgets
    "calib_",
    "iso_",
    "kin_",
    "dos_",
    "ph_",
    "temp_",
    "comp_",
    "thermo_",
    "3d_",

    # Tab widgets / legacy prefixes (safe to clear on study switch)
    "isotherm_",
    "kinetic_",
    "dosage_",
    "temperature_",

    "param_",
    "gen_",
    "save_",
    "calc_",
    "model_",

    "diffusion_",
    "weber_",
    "boyd_",
    "biot_",

    "langmuir_",
    "freundlich_",
    "temkin_",
    "sips_",

    "pfo_",
    "pso_",
    "elovich_",
    "ipd_",
    "rpso_",
    "rPSO_",
    "rPSO_C0=",

    # UI helpers / dynamic selectors
    "display_input_mode_",
    "manual_",
    "Ce_study_",
    "study_select_",
    "remove_",
)


# Per-study data keys used to compute workflow "Data Entered" metrics
# Keep this list in sync with the sidebar inputs that represent distinct datasets.
STUDY_METRIC_DATA_KEYS: tuple[str, ...] = (
    "calib_df_input",
    "isotherm_input",
    "kinetic_input",
    "dosage_input",
    "ph_effect_input",
    "temp_effect_input",
)

# =============================================================================
# UNIFIED QUALITY GRADING SYSTEM (Single Source of Truth)
# =============================================================================

QUALITY_GRADES: dict[str, QualityGradeInfo] = {
    "A+": {
        "min_score": 98,
        "min_r2": 0.999,
        "status": "✅ Excellent",
        "status_type": "success",
        "label": "Outstanding",
    },
    "A": {
        "min_score": 93,
        "min_r2": 0.995,
        "status": "✅ Excellent",
        "status_type": "success",
        "label": "Excellent",
    },
    "A-": {
        "min_score": 90,
        "min_r2": 0.99,
        "status": "✅ Good",
        "status_type": "success",
        "label": "Very Good",
    },
    "B": {
        "min_score": 80,
        "min_r2": 0.98,
        "status": "✅ Good",
        "status_type": "success",
        "label": "Good",
    },
    "C": {
        "min_score": 70,
        "min_r2": 0.95,
        "status": "⚠️ Acceptable",
        "status_type": "warning",
        "label": "Acceptable",
    },
    "D": {
        "min_score": 50,
        "min_r2": 0.90,
        "status": "⚠️ Poor",
        "status_type": "warning",
        "label": "Needs Improvement",
    },
    "F": {
        "min_score": 0,
        "min_r2": 0.0,
        "status": "❌ Failing",
        "status_type": "error",
        "label": "Unacceptable",
    },
}


def get_grade_from_score(score: float) -> dict[str, Any]:
    """
    Get grade info based on a quality score (0-100).

    Parameters
    ----------
    score : float
        Quality score from 0 to 100

    Returns
    -------
    dict
        Grade information including grade letter, status, and label
    """
    score = max(0, min(100, score))  # Clamp to 0-100

    for grade, info in QUALITY_GRADES.items():
        if score >= info["min_score"]:
            return {
                "grade": grade,
                "score": score,
                "status": info["status"],
                "status_type": info["status_type"],
                "label": info["label"],
            }

    # Fallback (should never reach here)
    return {
        "grade": "F",
        "score": score,
        "status": "❌ Failing",
        "status_type": "error",
        "label": "Unacceptable",
    }


def get_grade_from_r_squared(r_squared: float) -> dict[str, Any]:
    """
    Get grade info based on R² value.

    Parameters
    ----------
    r_squared : float
        Coefficient of determination (0 to 1)

    Returns
    -------
    dict
        Grade information including grade letter, status, and label
    """
    for grade, info in QUALITY_GRADES.items():
        if r_squared >= info["min_r2"]:
            return {
                "grade": grade,
                "r_squared": r_squared,
                "min_score": info["min_score"],
                "status": info["status"],
                "status_type": info["status_type"],
                "label": info["label"],
            }

    # Fallback
    return {
        "grade": "F",
        "r_squared": r_squared,
        "min_score": 0,
        "status": "❌ Failing",
        "status_type": "error",
        "label": "Unacceptable",
    }


# Backward compatibility alias
def get_calibration_grade(r_squared: float) -> dict[str, Any]:
    """Backward compatible alias for get_grade_from_r_squared."""
    result = get_grade_from_r_squared(r_squared)
    # Add 'score' field for backward compatibility
    result["score"] = next(
        (info["min_score"] for g, info in QUALITY_GRADES.items() if g == result["grade"]), 0
    )
    return result