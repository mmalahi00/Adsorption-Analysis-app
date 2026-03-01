# app_main.py
"""
AdsorbLab Pro - Advanced Adsorption Data Analysis Platform
==========================================================
v2.0.0

A comprehensive Streamlit application for analyzing adsorption experiments with:
- Advanced statistical analysis with confidence intervals
- Bootstrap parameter estimation
- Comprehensive model comparison (R², Adj-R², AIC, AICc, BIC)
- Residual diagnostics and outlier detection
- Dual-unit reporting (mg/g and % Removal)
- Intelligent rule-based model recommendations
- 3D visualization and parameter space exploration
- Multi-study comparison with mechanism interpretation

NEW IN v2.0.0:
- Revised PSO (rPSO) model with concentration correction (Bullen et al., 2021)
- PSO mechanistic interpretation warnings (not proof of chemisorption)
- Multi-component competitive adsorption models (Extended Langmuir/Freundlich)
- Diffusion analysis: Biot number, Boyd plot, Weber-Morris multilinearity
- Rate-limiting step identification (Biot number, Weber-Morris, Boyd plots)
- Durbin-Watson autocorrelation test for kinetics (removed from isotherms)

Author: AdsorbLab Team
Version: 2.0.0
License: MIT
"""

import copy
import sys
from html import escape as html_escape
from pathlib import Path

import numpy as np
import streamlit as st
from scipy.stats import linregress

# Add the parent directory to path so the package can be found
# when running directly with streamlit
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from adsorblab_pro.config import (
    DEFAULT_SESSION_STATE,
    DEFAULT_GLOBAL_SESSION_STATE,
    VERSION,
    get_grade_from_r_squared,
)
from datetime import datetime

current_year = datetime.now().year

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
APP_VERSION = VERSION
APP_TITLE = "AdsorbLab Pro"

# Pre-escaped for safe injection into unsafe_allow_html contexts.
# APP_VERSION comes from importlib.metadata (pyproject.toml / .dist-info), so
# in a compromised package installation an attacker could craft a version string
# containing HTML/JS.  Escaping once here ensures every raw-HTML usage is safe.
_APP_VERSION_SAFE = html_escape(str(APP_VERSION))

st.set_page_config(
    page_title=f"{APP_TITLE}", page_icon="🔬", layout="wide", initial_sidebar_state="expanded"
)

# =============================================================================
# LOCAL IMPORTS
# =============================================================================
from adsorblab_pro import sidebar_ui
from adsorblab_pro.tabs import home_tab  # Only home_tab needed immediately for landing page
from adsorblab_pro import utils

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Default session-state cleanup/metrics keys live in config.py (single source of truth)
# See: SESSION_INPUT_KEYS_TO_CLEAR, SESSION_WIDGET_PREFIXES_TO_CLEAR, STUDY_METRIC_DATA_KEYS

# =============================================================================
# LAZY TAB LOADER
# =============================================================================
# Cache imported modules to avoid repeated imports within a session
_tab_cache: dict = {}


def _lazy_render(module_name: str) -> None:
    """
    Lazy-load and render a tab module on demand.

    This implements true lazy loading - modules are only imported when their
    tab content is actually rendered, not at application startup. This provides:
    - Graceful degradation if optional dependencies are missing
    - Reduced memory footprint when users don't visit all tabs

    Note: Streamlit's st.tabs() renders all tab content (hidden via CSS), so all
    imports still occur on page load. However, this approach:
    1. Spreads import cost across the render phase rather than blocking startup
    2. Allows individual tabs to fail gracefully without breaking the app
    3. Caches modules to prevent re-importing on Streamlit reruns
    """
    if module_name not in _tab_cache:
        import importlib

        try:
            _tab_cache[module_name] = importlib.import_module(
                f".tabs.{module_name}", package="adsorblab_pro"
            )
        except ImportError as e:
            _tab_cache[module_name] = None
            st.error(f"⚠️ Could not load {module_name}: {e}")
            st.info("Some dependencies may be missing. Check requirements.txt")
            return
        except Exception as e:
            _tab_cache[module_name] = None
            st.error(f"⚠️ Error loading {module_name}: {e}")
            return

    if _tab_cache[module_name] is not None:
        _tab_cache[module_name].render()


# Initialize the multi-study structure
if "studies" not in st.session_state:
    st.session_state.studies = {}
if "current_study" not in st.session_state:
    st.session_state.current_study = None

# Initialize root-level session keys (single source of truth in config.py)
for _k, _v in DEFAULT_GLOBAL_SESSION_STATE.items():
    if _k not in st.session_state:
        st.session_state[_k] = copy.deepcopy(_v)


# =============================================================================
# CALIBRATION UPDATE FUNCTION
# =============================================================================
def update_calibration():
    """Automatically recalculate calibration with full statistics for the active study."""
    active_study_name = st.session_state.get("current_study")
    if not active_study_name:
        return  # Do nothing if no study is active

    current_study_state = st.session_state.studies[active_study_name]

    new_calib_df = current_study_state.get("calib_df_input")
    old_calib_df = current_study_state.get("previous_calib_df")

    if new_calib_df is not None and len(new_calib_df) >= 3:
        data_changed = old_calib_df is None or not new_calib_df.equals(old_calib_df)

        if data_changed:
            # Reset dependent results within the current study
            keys_to_reset = [
                "isotherm_results",
                "isotherm_models_fitted",
                "kinetic_results_df",
                "kinetic_models_fitted",
                "dosage_results",
                "ph_effect_results",
                "temp_effect_results",
                "thermo_params",
            ]
            for key in keys_to_reset:
                if key in current_study_state:
                    if isinstance(current_study_state.get(key), dict):
                        current_study_state[key] = {}
                    else:
                        current_study_state[key] = None

            # Calculate calibration with full statistics
            try:
                conc = new_calib_df["Concentration"].values
                abs_val = new_calib_df["Absorbance"].values

                slope, intercept, r_value, p_value, std_err = linregress(conc, abs_val)

                n = len(conc)
                y_pred = slope * conc + intercept
                residuals = abs_val - y_pred
                ss_res = np.sum(residuals**2)
                se_estimate = np.sqrt(ss_res / (n - 2)) if n > 2 else 0

                # SE of intercept
                se_intercept = se_estimate * np.sqrt(
                    1 / n + np.mean(conc) ** 2 / np.sum((conc - np.mean(conc)) ** 2)
                )

                # t-value using study's confidence level
                from scipy.stats import t as t_dist

                study_confidence = current_study_state.get("confidence_level", 0.95)
                alpha = 1 - study_confidence
                t_val = t_dist.ppf(1 - alpha / 2, n - 2) if n > 2 else 2.0

                # linregress returns NaN slope (and all other stats) when the
                # concentration column has zero variance (all identical values).
                # abs(NaN) > 1e-9 evaluates to False under IEEE 754, so without
                # this explicit guard the block is silently skipped: no params
                # are saved and no feedback is shown to the user.
                #
                # NaN and near-zero slope are distinct failure modes and need
                # separate messages:
                #   NaN   → degenerate data (constant concentrations)
                #   ~0    → calibration curve is flat (poor experimental design)
                if not np.isfinite(slope):
                    st.warning(
                        "⚠️ Calibration failed: all concentration values are identical "
                        "(zero variance). Provide at least 3 distinct concentration points."
                    )
                elif abs(slope) <= 1e-9:
                    st.warning(
                        "⚠️ Calibration slope is effectively zero — the absorbance values "
                        "show no response to concentration. Check your calibration data."
                    )
                else:
                    # Save results to the current study
                    current_study_state["calibration_params"] = {
                        "slope": slope,
                        "intercept": intercept,
                        "r_squared": r_value**2,
                        "adj_r_squared": 1 - (1 - r_value**2) * (n - 1) / (n - 2)
                        if n > 2
                        else r_value**2,
                        "p_value": p_value,
                        "std_err_slope": std_err,
                        "std_err_intercept": se_intercept,
                        "std_err_estimate": se_estimate,
                        "slope_ci_95": (slope - t_val * std_err, slope + t_val * std_err),
                        "intercept_ci_95": (
                            intercept - t_val * se_intercept,
                            intercept + t_val * se_intercept,
                        ),
                        "confidence_level": study_confidence,
                        "equation": f"Abs = {slope:.4f} × C + {intercept:.4f}",
                        "n_points": n,
                        "quality_score": get_grade_from_r_squared(r_value**2)["min_score"],
                        "residuals": residuals.tolist(),
                        "y_pred": y_pred.tolist(),
                    }
                    current_study_state["previous_calib_df"] = new_calib_df.copy()
            except Exception as e:
                current_study_state["calibration_params"] = None
                st.warning(f"Calibration calculation failed: {e}")
    else:
        # Calibration data is missing or insufficient - clear stale parameters
        # to prevent downstream tabs from using invalid calibration
        if current_study_state.get("calibration_params") is not None:
            current_study_state["calibration_params"] = None
            current_study_state["previous_calib_df"] = None


# =============================================================================
# MAIN APPLICATION HEADER
# =============================================================================
# Title with version badge
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.title(f"🔬 {APP_TITLE}")
with col_badge:
    st.markdown(
        f"""
    <div style="background: linear-gradient(90deg, #2E86AB, #A23B72);
                padding: 8px 16px; border-radius: 20px; text-align: center;
                color: white; font-weight: bold; margin-top: 20px;">
        v{_APP_VERSION_SAFE}
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("""
**Advanced Adsorption Data Analysis Platform**
*Statistical analysis with confidence intervals, model comparison, and residual diagnostics*
""")

# Feature highlights
with st.expander("✨ **Key Features**", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        **📊 Advanced Statistics**
        - 95% Confidence Intervals
        - Bootstrap parameter estimation
        - Adjusted R², AIC, AICc, BIC
        """)
    with col2:
        st.markdown("""
        **🔍 Residual Diagnostics**
        - Normality tests (Shapiro-Wilk)
        - Durbin-Watson autocorrelation
        - Outlier detection
        - Q-Q plots
        """)
    with col3:
        st.markdown("""
        **📈 Comprehensive Models**
        - 4 Isotherm models
        - 5 Kinetic models
        - Multi-stage IPD analysis
        - Van't Hoff thermodynamics
        """)
    with col4:
        st.markdown("""
        **📦 Export Options**
        - Dual-unit reporting
        - Parameter uncertainty
        - TIFF/PNG/SVG/PDF
        - Batch export
        """)

st.markdown("---")

# =============================================================================
# GLOBAL SETTINGS SIDEBAR
# =============================================================================
st.sidebar.header("⚙️ Global Settings")

# --- Get the active study to apply settings ---
active_study_name = st.session_state.get("current_study")
# Use .get() to safely handle the case where no study is selected yet
active_study_state = st.session_state.studies.get(active_study_name, {})

st.sidebar.markdown("### 🔬 Study Management")

# Ensure the key exists BEFORE creating the widget
if "new_study_input" not in st.session_state:
    st.session_state["new_study_input"] = ""


def _add_new_study() -> None:
    """Callback for Add New Study button.

    Runs BEFORE widgets are instantiated on rerun, so it's safe to update
    st.session_state['new_study_input'] here.
    """
    name = (st.session_state.get("new_study_input") or "").strip()
    is_valid, error_msg = utils.validate_study_name(name, st.session_state.get("studies", {}))

    if not is_valid:
        st.session_state["_add_study_msg"] = ("warning", error_msg)
        if name:  # Only clear if user tried to submit something
            st.session_state["new_study_input"] = ""
        return

    # Create and select the new study
    st.session_state.studies[name] = copy.deepcopy(DEFAULT_SESSION_STATE)
    st.session_state.current_study = name
    st.session_state._previous_study_selection = name

    # Clean up stale widget/input keys (uploads etc.)
    utils.cleanup_session_state_keys()

    # Clear input safely (callback)
    st.session_state["new_study_input"] = ""

    st.session_state["_add_study_msg"] = ("success", f"Added and selected study: {name}")


# Widgets (define after callback)
st.sidebar.text_input(
    "Study Name (e.g., AC-MB, Zeolite, Pollutant-A):",
    key="new_study_input",
)
st.sidebar.button(
    "Add New Study",
    key="add_study_button",
    on_click=_add_new_study,
)

# Show feedback once, then clear it
_msg = st.session_state.pop("_add_study_msg", None)
if _msg:
    level, message = _msg
    if level == "success":
        st.sidebar.success(message)
    else:
        st.sidebar.warning(message)

# Dropdown to select the active study
study_names = list(st.session_state.studies.keys())
if study_names:
    # Ensure current_study is valid
    if st.session_state.current_study not in study_names:
        st.session_state.current_study = study_names[0]

    # Capture previous selection BEFORE updating, so the comparison below is valid.
    previous_study = st.session_state.get("_previous_study_selection")

    selected_study = st.sidebar.selectbox(
        "Select Active Study",
        options=study_names,
        index=study_names.index(st.session_state.current_study),
        key="study_selector",
    )

    # Update current study
    st.session_state.current_study = selected_study

    # Unconditionally record the current selection as the new "previous" BEFORE
    # the switch-detection block.  This is the single authoritative write point and
    # provides idempotency against rapid double-clicks:
    #
    #   First rerun  (A→B): previous=A, we write B, condition True  → cleanup + rerun
    #   Second rerun (B→B): previous=B, we write B, condition False → no-op
    #
    # Previously this write lived both inside the if-block (before st.rerun(), which
    # raises RerunException and makes anything after it unreachable in that branch)
    # and after the if-block as a "fallback" for the no-switch case.  Having two
    # write sites was fragile; a single unconditional write here covers both cases.
    st.session_state._previous_study_selection = selected_study

    # Trigger cleanup + rerun only on a genuine study switch (skip on initial load
    # when previous_study is None, and skip on no-change reruns).
    if previous_study is not None and selected_study != previous_study:
        utils.cleanup_session_state_keys()
        st.rerun()
else:
    st.sidebar.info("Add a new study to begin analysis.")

st.sidebar.markdown("---")

# =============================================================================
# DISPLAY UNITS (only show when a study exists)
# =============================================================================
active_study_name = st.session_state.get("current_study")

if active_study_name:
    with st.sidebar.expander("📊 Display Units", expanded=False):
        # Re-read active study state (user may have just switched studies)
        active_study_state = st.session_state.studies.get(active_study_name, {})

        # --- Unit System ---
        unit_options = ["mg/g", "% Removal", "Both"]
        saved_unit = active_study_state.get("unit_system", "mg/g")
        if saved_unit not in unit_options:
            saved_unit = "mg/g"

        unit_system = st.radio(
            "Select unit system:",
            unit_options,
            index=unit_options.index(saved_unit),
            help="Choose how to display adsorption results (plots/tables).",
        )
        st.session_state.studies[active_study_name]["unit_system"] = unit_system

        # --- Input Mode (global, per study) ---
        input_mode_options = ["absorbance", "direct"]
        saved_input_mode = active_study_state.get("input_mode_global", "absorbance")
        if saved_input_mode not in input_mode_options:
            saved_input_mode = "absorbance"

        input_mode_key = f"display_input_mode_{active_study_name}"

        input_mode_global = st.radio(
            "Select data input mode:",
            input_mode_options,
            index=input_mode_options.index(saved_input_mode),
            format_func=lambda x: (
                "📊 Absorbance (requires calibration)"
                if x == "absorbance"
                else "📈 Direct Concentration (Ce/Ct values)"
            ),
            help=(
                "Absorbance mode uses the Calibration Curve. "
                "Direct mode lets you upload Ce/Ct values without calibration."
            ),
            horizontal=True,
            key=input_mode_key,
        )
        st.session_state.studies[active_study_name]["input_mode_global"] = input_mode_global

    st.sidebar.markdown("---")


# =============================================================================
# SIDEBAR DATA INPUT
# =============================================================================
st.sidebar.header("📥 Data Input")
sidebar_ui.render_sidebar_content()

# Update calibration (only when input mode is absorbance)
active_study_name = st.session_state.get("current_study")
active_study_state = st.session_state.studies.get(active_study_name, {})
input_mode_global = active_study_state.get("input_mode_global", "absorbance")

if input_mode_global == "absorbance":
    update_calibration()

    if active_study_name and st.session_state.studies[active_study_name].get("calibration_params"):
        calib_params = st.session_state.studies[active_study_name]["calibration_params"]
        r2 = calib_params["r_squared"]
        with st.sidebar:
            if r2 >= 0.999:
                st.success(f"**R² = {r2:.6f}** ✓ Excellent")
            elif r2 >= 0.99:
                st.info(f"**R² = {r2:.5f}** — Good")
            else:
                st.warning(f"**R² = {r2:.4f}** — Needs improvement")

# =============================================================================
# MAIN TABS
# =============================================================================
main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs(
    [
        "🏠 Home",
        "🧪 Analysis Workflow",
        "🔬 Parameter Effects",
        "📈 Visualization & Reports",
    ]
)

# --- Home Tab ---
with main_tab1:
    home_tab.render()

# --- Analysis Workflow Tab (with sub-tabs) ---
with main_tab2:
    st.header("🧪 Analysis Workflow")

    # Hide the Calibration workflow when the user is in Direct input mode
    active_study_name = st.session_state.get("current_study")
    active_study_state = st.session_state.studies.get(active_study_name, {})
    input_mode_global = active_study_state.get("input_mode_global", "absorbance")

    tab_specs = []
    if input_mode_global == "absorbance":
        tab_specs.append(("📊 Calibration", "calibration_tab"))

    tab_specs.extend(
        [
            ("📈 Isotherms", "isotherm_tab"),
            ("⏱️ Kinetics", "kinetic_tab"),
            ("🌡️ Thermodynamics", "thermodynamics_tab"),
            ("🔄 Multi-Component", "competitive_tab"),
            ("🆚 Comparison", "comparison_tab"),
        ]
    )

    workflow_tabs = st.tabs([label for label, _ in tab_specs])
    for tab, (_, module_name) in zip(workflow_tabs, tab_specs):
        with tab:
            _lazy_render(module_name)

# --- Parameter Effects Tab (with sub-tabs) ---
with main_tab3:
    st.header("🔬 Parameter Effects")
    sub_tab_ph, sub_tab_temp, sub_tab_dosage = st.tabs(
        ["🧪 pH Effect", "🔥 Temperature", "⚖️ Dosage"]
    )
    with sub_tab_ph:
        _lazy_render("ph_effect_tab")
    with sub_tab_temp:
        _lazy_render("temperature_tab")
    with sub_tab_dosage:
        _lazy_render("dosage_tab")

# --- Visualization & Reports Tab (with sub-tabs) ---
with main_tab4:
    st.header("📈 Visualization & Reports")
    sub_tab_report, sub_tab_3d, sub_tab_export = st.tabs(
        ["📊 Study Overview", "🔮 3D Explorer", "📦 Export All"]
    )
    with sub_tab_report:
        _lazy_render("statistical_summary_tab")
    with sub_tab_3d:
        _lazy_render("threed_explorer_tab")
    with sub_tab_export:
        _lazy_render("report_tab")


# =============================================================================
# SIDEBAR - QUICK ACTIONS
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.header("⚡ Quick Actions")

# Get study metrics (computed once, used in multiple sections)
_metrics = utils.get_study_metrics()

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    st.metric("Studies", f"{_metrics['study_count']}")
with col2:
    st.metric(
        "Data Entered", f"{_metrics['active_data_count']}/{_metrics.get('active_data_total', 6)}"
    )
with col3:
    st.metric("Calib. Quality", f"{_metrics['calib_quality']}/100")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    f"""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>AdsorbLab Pro v{_APP_VERSION_SAFE}</strong></p>
    <p>Advanced Adsorption Data Analysis Platform</p>
    <p style="font-size: 0.8em; margin-top: 10px;">
        Features: Bootstrap CI • AIC/BIC Selection • Multi-Study Comparison • Mechanism Interpretation
    </p>
    <p style="font-size: 0.85em; margin-top: 8px;">
        <a href="https://doi.org/10.5281/zenodo.18501799" target="_blank" style="color: #2E86AB; text-decoration: none;">
            DOI: 10.5281/zenodo.18501799
        </a>
    </p>
    <p>© {current_year} Mohamed EL MALLAHI</p>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# FIRST-TIME USER WELCOME
# =============================================================================
if st.session_state.get("first_time", True):
    st.session_state["first_time"] = False
    st.toast(
        f"Welcome to AdsorbLab Pro v{APP_VERSION}! Start by adding a study in the sidebar.",
        icon="👋",
    )