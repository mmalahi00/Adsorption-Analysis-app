# tabs/isotherm_tab.py
"""
Isotherm Tab - AdsorbLab Pro
============================

Multi-model isotherm fitting with statistical comparison.

Features:
- 4 Isotherm models (Langmuir, Freundlich, Temkin, Sips)
- Non-linear regression with 95% CI
- Comprehensive model comparison (R², Adj-R², AIC, BIC)
- Akaike weights for model selection
- Residual diagnostics
- Separation factor (RL) analysis
"""

import hashlib
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from adsorblab_pro.streamlit_compat import st

from ..config import BOOTSTRAP_DEFAULT_ITERATIONS
from ..models import (
    fit_model_with_ci,
    freundlich_model,
    langmuir_model,
    sips_model,
    temkin_model,
)

# Professional plot styling
from ..plot_style import (
    apply_professional_style,
    create_isotherm_plot,
    create_model_comparison_plot,
    create_parity_plot,
    style_experimental_trace,
)
from ..utils import (
    EPSILON_DIV,
    CalculationResult,
    assess_data_quality,
    bootstrap_confidence_intervals,
    calculate_adsorption_capacity,
    calculate_akaike_weights,
    calculate_Ce_from_absorbance,
    calculate_removal_percentage,
    calculate_separation_factor,
    display_results_table,
    get_current_study_state,
    interpret_separation_factor,
    propagate_calibration_uncertainty,
    recommend_best_models,
    validate_required_params,
)
from ..validation import format_validation_errors, validate_isotherm_data


def _validate_isotherm_input(C0, Ce, V, m):
    """
    Validate isotherm data before analysis.

    Returns:
        tuple: (is_valid, validation_report)
    """
    validation_report = validate_isotherm_data(C0=np.array(C0), Ce=np.array(Ce), V=V, m=m)
    return validation_report.is_valid, validation_report


def _display_validation_results(validation_report):
    """Display validation results in Streamlit UI."""
    if not validation_report.is_valid:
        st.error("❌ **Data Validation Failed**")
        st.markdown(format_validation_errors(validation_report))
        return False

    if validation_report.has_warnings:
        with st.expander("⚠️ Data Quality Warnings", expanded=True):
            for w in validation_report.warnings:
                st.warning(w.message)
                if w.suggestion:
                    st.caption(f"💡 Suggestion: {w.suggestion}")

    return True


# =============================================================================
# CACHING UTILITIES
# =============================================================================


def _compute_data_hash(Ce: np.ndarray, qe: np.ndarray, C0: np.ndarray) -> str:
    """
    Compute a hash of the input data for cache invalidation.

    This allows us to detect when data has changed without comparing
    entire arrays, which is much faster.
    """
    # Combine arrays into a single bytes object
    combined = np.concatenate([Ce, qe, C0]).tobytes()
    return hashlib.md5(combined).hexdigest()


def _arrays_to_tuples(Ce: np.ndarray, qe: np.ndarray, C0: np.ndarray):
    """Convert numpy arrays to tuples for cache key hashing."""
    return (
        tuple(np.round(Ce, 8).tolist()),
        tuple(np.round(qe, 8).tolist()),
        tuple(np.round(C0, 8).tolist()),
    )


def _run_isotherm_bootstrap(Ce, qe, fitted_models, n_bootstrap, confidence_level, T_K: float = 298.15):
    """
    Run bootstrap CI on already fitted isotherm models with visual progress.

    This function adds bootstrap confidence intervals to models that have
    already been fitted, showing a progress bar for user feedback.
    """

    valid = (Ce > EPSILON_DIV) & (qe > EPSILON_DIV)
    Ce_v, qe_v = Ce[valid], qe[valid]

    models_to_bootstrap = [
        ("Langmuir", langmuir_model, ["qm", "KL"]),
        ("Freundlich", freundlich_model, ["KF", "n_inv"]),
        ("Temkin", temkin_model, ["B1", "KT"]),
        ("Sips", sips_model, ["qm", "Ks", "ns"]),
    ]

    total_models = len(
        [m for m, _, _ in models_to_bootstrap if fitted_models.get(m, {}).get("converged")]
    )

    if total_models == 0:
        st.warning("No converged models to bootstrap")
        return fitted_models

    # Create progress elements
    progress_bar = st.progress(0)
    status_text = st.empty()

    current_model_idx = 0

    for model_name, model_func, param_names in models_to_bootstrap:
        if not fitted_models.get(model_name, {}).get("converged"):
            continue

        current_model_idx += 1
        status_text.text(f"🔄 Bootstrap {model_name} ({current_model_idx}/{total_models})...")

        # Get current parameters
        params = fitted_models[model_name].get("popt")
        if params is None:
            params = [fitted_models[model_name]["params"][p] for p in param_names]
        params = np.array(params)

        # Run bootstrap with progress
        def model_progress(
            current,
            total,
            message,
            *,
            _model_name=model_name,
            _current_model_idx=current_model_idx,
        ):
            model_progress_pct = current / total
            overall_progress = (_current_model_idx - 1 + model_progress_pct) / total_models
            progress_bar.progress(overall_progress)
            status_text.text(f"🔄 {_model_name}: {message}")

        try:
            ci_lower, ci_upper = bootstrap_confidence_intervals(
                model_func,
                Ce_v,
                qe_v,
                params,
                n_bootstrap=n_bootstrap,
                confidence=confidence_level,
                progress_callback=model_progress,
                early_stopping=True,
            )

            # Update fitted models with bootstrap CI
            if not np.any(np.isnan(ci_lower)):
                fitted_models[model_name]["bootstrap_ci_95"] = {
                    param_names[i]: (ci_lower[i], ci_upper[i]) for i in range(len(param_names))
                }
                fitted_models[model_name]["bootstrap_n"] = n_bootstrap
        except Exception as e:
            st.warning(f"Bootstrap failed for {model_name}: {str(e)}")

    # Clean up
    progress_bar.progress(1.0)
    status_text.text("✅ Bootstrap complete!")

    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    st.success(
        f"✅ Bootstrap CI calculated for {current_model_idx} models ({n_bootstrap} iterations)"
    )

    return fitted_models


# =============================================================================
# CACHED MODEL FITTING - THE KEY OPTIMIZATION
# =============================================================================


@st.cache_data(
    show_spinner=False,  # We'll show our own spinner with more detail
    ttl=3600,  # Cache for 1 hour
    max_entries=50,  # Limit cache size
)
def _fit_all_isotherm_models_cached(
    Ce_tuple: tuple,
    qe_tuple: tuple,
    C0_tuple: tuple,
    confidence_level: float = 0.95,
    T_K: float = 298.15,
) -> dict:
    """
    Fit all supported isotherm models (cached).

    This function is wrapped with Streamlit caching. Inputs are passed as tuples to make them
    hashable, and changing any input (including temperature) invalidates the cache.

    Parameters
    ----------
    Ce_tuple : tuple
        Equilibrium concentrations (Ce) in mg/L (hashable form).
    qe_tuple : tuple
        Adsorption capacities (qe) in mg/g (hashable form).
    C0_tuple : tuple
        Initial concentrations (C0) in mg/L (hashable form).
    confidence_level : float, optional
        Confidence level for confidence intervals / uncertainty summaries (default 0.95).
    T_K : float, optional
        Experiment temperature in Kelvin.
        via the Polanyi potential and included in the cache key (default 298.15 K).

    Returns
    -------
    dict
        Dictionary mapping model names to fit result dictionaries (params, metrics, CI, etc.).
    """
    # Convert back to numpy arrays for computation
    Ce = np.array(Ce_tuple)
    qe = np.array(qe_tuple)
    C0 = np.array(C0_tuple)

    fitted: dict[str, Any] = {}

    # Filter valid data
    valid = (Ce > EPSILON_DIV) & (qe > EPSILON_DIV)
    Ce_v = Ce[valid]
    qe_v = qe[valid]
    C0_v = C0[valid]

    if len(Ce_v) < 4:
        return fitted

    qe_max = qe_v.max()
    Ce_mean = Ce_v.mean()

    # =========================================================================
    # FIT EACH MODEL
    # =========================================================================

    # Langmuir
    try:
        result = fit_model_with_ci(
            langmuir_model,
            Ce_v,
            qe_v,
            p0=[qe_max * 1.5, 0.1],
            bounds=([0, 0], [qe_max * 10, 100]),
            param_names=["qm", "KL"],
            confidence=confidence_level,
        )
        if result and result.get("converged"):
            # Add RL calculation (this is derived data, not a side effect)
            RL = calculate_separation_factor(result["params"]["KL"], C0_v)
            result["RL"] = RL
            result["RL_interpretation"] = interpret_separation_factor(RL)
            fitted["Langmuir"] = result
    except Exception as e:
        fitted["Langmuir"] = {"converged": False, "error": str(e)}

    # Freundlich
    try:
        result = fit_model_with_ci(
            freundlich_model,
            Ce_v,
            qe_v,
            p0=[qe_max / Ce_mean**0.5, 0.5],
            bounds=([0, 0.01], [1000, 5]),
            param_names=["KF", "n_inv"],
            confidence=confidence_level,
        )
        if result and result.get("converged"):
            result["params"]["n"] = (
                1 / result["params"]["n_inv"] if result["params"]["n_inv"] > EPSILON_DIV else np.nan
            )
            fitted["Freundlich"] = result
    except Exception as e:
        fitted["Freundlich"] = {"converged": False, "error": str(e)}

    # Temkin
    try:
        result = fit_model_with_ci(
            temkin_model,
            Ce_v,
            qe_v,
            p0=[qe_max / 5, 1.0],
            bounds=([-100, 0.001], [200, 1000]),
            param_names=["B1", "KT"],
            confidence=confidence_level,
        )
        if result and result.get("converged"):
            # Flag if model predicts negative qe (outside valid domain)
            params = result["params"]
            if params["KT"] * Ce_v.min() < 1:
                result["temkin_domain_warning"] = True
            fitted["Temkin"] = result
    except Exception as e:
        fitted["Temkin"] = {"converged": False, "error": str(e)}

    # Sips
    try:
        result = fit_model_with_ci(
            sips_model,
            Ce_v,
            qe_v,
            p0=[qe_max * 1.5, 0.1, 1.0],
            bounds=([0, 0, 0.1], [qe_max * 10, 100, 5]),
            param_names=["qm", "Ks", "ns"],
            confidence=confidence_level,
        )
        if result and result.get("converged"):
            fitted["Sips"] = result
    except Exception as e:
        fitted["Sips"] = {"converged": False, "error": str(e)}

    return fitted


# =============================================================================
# WRAPPER FUNCTION WITH CACHE MANAGEMENT
# =============================================================================


def fit_isotherm_models_with_cache(Ce, qe, C0, confidence_level, current_study_state, T_K: float = 298.15):
    """
    Wrapper that manages caching and session state updates.

    This function:
    1. Checks if cached results exist and are valid
    2. Calls the cached fitting function if needed
    3. Updates session_state with results (OUTSIDE the cached function)

    Parameters
    ----------
    Ce, qe, C0 : np.ndarray
        Input data arrays
    confidence_level : float
        Confidence level for CI
    current_study_state : dict
        Reference to the current study's state dict

    Returns
    -------
    dict
        Fitted model results
    """
    # Compute data hash for cache validation
    data_hash = _compute_data_hash(Ce, qe, C0)

    # Check if we have valid cached results in session state
    cached_temp = current_study_state.get("_isotherm_T_K")
    cached_hash = current_study_state.get("_isotherm_data_hash")
    cached_models = current_study_state.get("isotherm_models_fitted")
    cached_confidence = current_study_state.get("_isotherm_confidence_level")

    if (
        cached_models
        and cached_hash == data_hash
        and cached_confidence == confidence_level
        and cached_temp == T_K
        and len(cached_models) > 0
    ):
        return cached_models


    # Cache miss - need to fit models
    # Convert arrays to tuples for the cached function
    Ce_tuple, qe_tuple, C0_tuple = _arrays_to_tuples(Ce, qe, C0)

    # Call the cached fitting function
    fitted_models = _fit_all_isotherm_models_cached(Ce_tuple, qe_tuple, C0_tuple, confidence_level, T_K)

    # Update session state with results (OUTSIDE cached function)
    current_study_state["_isotherm_T_K"] = T_K
    current_study_state["isotherm_models_fitted"] = fitted_models
    current_study_state["_isotherm_data_hash"] = data_hash
    current_study_state["_isotherm_confidence_level"] = confidence_level

    # Store individual model params for 3D explorer (session state update)
    if fitted_models.get("Langmuir", {}).get("converged"):
        current_study_state["langmuir_params_nl"] = fitted_models["Langmuir"]
    if fitted_models.get("Freundlich", {}).get("converged"):
        current_study_state["freundlich_params_nl"] = fitted_models["Freundlich"]
    if fitted_models.get("Temkin", {}).get("converged"):
        current_study_state["temkin_params_nl"] = fitted_models["Temkin"]

    return fitted_models


def _check_linear_nonlinear_warning(Ce, qe, fitted_models, current_study_state):
    """
    Check if non-linear regression is significantly better than linear.
    Display warning if difference > 5%.
    """
    # Get linear results if available
    linear_results = current_study_state.get("isotherm_linear_results", {})

    for model_name in ["Langmuir", "Freundlich", "Temkin"]:
        nl_result = fitted_models.get(model_name, {})
        l_result = linear_results.get(model_name, {})

        if nl_result.get("converged") and l_result.get("r_squared"):
            nl_r2 = nl_result.get("r_squared", 0)
            l_r2 = l_result.get("r_squared", 0)
            diff = (nl_r2 - l_r2) * 100

            if diff > 5:
                st.warning(
                    f"⚠️ **{model_name}:** Non-linear R² ({nl_r2:.4f}) is {diff:.1f}% better than "
                    f"linear R² ({l_r2:.4f}). **Non-linear parameters are more accurate.**"
                )

def _get_temperature_k(params: dict) -> float:
    """Return temperature in Kelvin from params (prefers T_K, falls back to T_C, else 298.15)."""
    if not params:
        return 298.15
    if params.get("T_K") is not None:
        return float(params["T_K"])
    if params.get("T_C") is not None:
        return float(params["T_C"]) + 273.15
    return 298.15


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================


def render():
    """Render isotherm analysis with publication-ready statistics."""
    st.subheader("📈 Adsorption Isotherm Analysis")
    st.markdown("*Multi-model fitting with confidence intervals and statistical comparison*")

    current_study_state = get_current_study_state()
    if not current_study_state:
        st.info("Please add or select a study from the sidebar to begin analysis.")
        return

    unit_system = current_study_state.get("unit_system", "mg/g")
    confidence_level = current_study_state.get("confidence_level", 0.95)

    iso_input = current_study_state.get("isotherm_input")
    calib_params = current_study_state.get("calibration_params")

    # Check input mode (default to 'absorbance' for backward compatibility)
    input_mode = iso_input.get("input_mode", "absorbance") if iso_input else "absorbance"

    # Determine if we can proceed based on input mode
    can_proceed = False
    if iso_input:
        if input_mode == "direct":
            # Direct mode doesn't require calibration
            can_proceed = True
        elif calib_params:
            # Absorbance mode requires calibration
            can_proceed = True

    if can_proceed:
        is_valid, error_message = validate_required_params(
            params=iso_input["params"], required_keys=[("m", "Mass"), ("V", "Volume")]
        )
        if not is_valid:
            st.warning(error_message, icon="⚠️")
            return

        # Quality assessment
        quality = assess_data_quality(iso_input["data"], "isotherm")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Quality", f"{quality['quality_score']}/100")
        with col2:
            st.metric("Points", len(iso_input["data"]))
        with col3:
            status = "✅ Good" if quality["quality_score"] >= 70 else "⚠️ Review"
            st.metric("Status", status)

        # Calculate isotherm data based on input mode
        if input_mode == "direct":
            iso_results_obj = _calculate_isotherm_results_direct(iso_input)
            st.caption("📈 *Using direct concentration input (Ce values)*")
        else:
            iso_results_obj = _calculate_isotherm_results(iso_input, calib_params)

        if not iso_results_obj.success:
            st.warning(f"Could not process isotherm data: {iso_results_obj.error}")
            return

        iso_results = iso_results_obj.data
        if iso_results is not None and not iso_results.empty:
            current_study_state["isotherm_results"] = iso_results

            # Data Display Section
            st.markdown("---")
            st.markdown("### 📊 Equilibrium Data")

            col1, col2 = st.columns(2)
            with col1:
                st.latex(r"q_e = \frac{(C_0 - C_e) \cdot V}{m}")
            with col2:
                st.latex(r"\% \text{ Removal} = \frac{(C_0 - C_e)}{C_0} \times 100")

            display_cols = ["C0_mgL", "Ce_mgL", "qe_mg_g", "removal_%"]
            display_results_table(iso_results[display_cols].round(4), hide_index=False)

            params = iso_input["params"]
            T_K = round(_get_temperature_k(params), 2)
            st.caption(
                f"**Conditions:** m = {params['m']} g | V = {params['V']} L | "
                f"T = {T_K - 273.15:.1f} °C ({T_K:.2f} K)"
            )


            # Visualization
            st.markdown("---")
            st.markdown("### 📈 Isotherm Curve")

            # Always remind: model fitting uses qe (mg/g)
            st.caption("Note: model fitting below always uses qₑ (mg/g). Unit selection affects only the overview plots/tables.")

            # --- Plot selection ---
            if unit_system == "mg/g":
                # qe plot
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=iso_results["Ce_mgL"],
                        y=iso_results["qe_mg_g"],
                        **style_experimental_trace(name="Experimental"),
                    )
                )

                fig = apply_professional_style(
                    fig,
                    title="Isotherm Curve",
                    x_title="Concentration C<sub>e</sub> (mg/L)",
                    y_title="q<sub>e</sub> (mg/g)",
                    height=450,
                    show_legend=True,
                    legend_position="upper left",
                )
                fig.update_xaxes(rangemode="tozero")
                fig.update_yaxes(rangemode="tozero")

                st.plotly_chart(fig, use_container_width=True, key="iso_overview_chart")

            elif unit_system == "% Removal":
                # removal plot
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=iso_results["Ce_mgL"],
                        y=iso_results["removal_%"],
                        **style_experimental_trace(name="Experimental"),
                    )
                )

                fig = apply_professional_style(
                    fig,
                    title="Removal vs C<sub>e</sub>",
                    x_title="Concentration C<sub>e</sub> (mg/L)",
                    y_title="Removal (%)",
                    height=450,
                    show_legend=True,
                    legend_position="upper left",
                )
                fig.update_xaxes(rangemode="tozero")
                fig.update_yaxes(rangemode="tozero")

                st.plotly_chart(fig, use_container_width=True, key="iso_overview_chart")

            else:  # Both
                # Prefer two separate plots (cleaner and more publication-friendly than dual-axis)
                st.markdown("**q<sub>e</sub> (mg/g)**")
                fig_qe = go.Figure()
                fig_qe.add_trace(
                    go.Scatter(
                        x=iso_results["Ce_mgL"],
                        y=iso_results["qe_mg_g"],
                        **style_experimental_trace(name="Experimental"),
                    )
                )

                fig_qe = apply_professional_style(
                    fig_qe,
                    title="Isotherm Curve",
                    x_title="Concentration C<sub>e</sub> (mg/L)",
                    y_title="q<sub>e</sub> (mg/g)",
                    height=420,
                    show_legend=True,
                    legend_position="upper left",
                )
                fig_qe.update_xaxes(rangemode="tozero")
                fig_qe.update_yaxes(rangemode="tozero")
                st.plotly_chart(fig_qe, use_container_width=True, key="iso_overview_chart_qe")

                st.markdown("**Removal (%)**")
                fig_rem = go.Figure()
                fig_rem.add_trace(
                    go.Scatter(
                        x=iso_results["Ce_mgL"],
                        y=iso_results["removal_%"],
                        **style_experimental_trace(name="Experimental"),
                    )
                )

                fig_rem = apply_professional_style(
                    fig_rem,
                    title="Removal vs C<sub>e</sub>",
                    x_title="Concentration C<sub>e</sub> (mg/L)",
                    y_title="Removal (%)",
                    height=420,
                    show_legend=True,
                    legend_position="upper left",
                )
                fig_rem.update_xaxes(rangemode="tozero")
                fig_rem.update_yaxes(rangemode="tozero")
                st.plotly_chart(fig_rem, use_container_width=True, key="iso_overview_chart_removal")

            # Model Fitting Section
            st.markdown("---")
            st.markdown("### 🔬 Model Fitting")

            Ce = iso_results["Ce_mgL"].values
            qe = iso_results["qe_mg_g"].values
            C0 = iso_results["C0_mgL"].values
            params = iso_input["params"]
            is_valid, validation_report = _validate_isotherm_input(
                C0=C0, Ce=Ce, V=params["V"], m=params["m"]
            )

            if not _display_validation_results(validation_report):
                st.info("Please correct the data issues above before fitting models.")
                st.stop()

            fitted_models = {}  # Initialize empty dict
            show_results = False  # Initialize flag

            # Options row with advanced settings
            col_btn, col_opt1, col_opt2 = st.columns([1, 1, 1])

            with col_btn:
                calculate_btn = st.button(
                    "🧮 Fit Models",
                    type="primary",
                    help="Click to fit all isotherm models to your data",
                    key="isotherm_calculate_btn",
                )
            with col_opt1:
                run_bootstrap = st.checkbox(
                    "🔄 Bootstrap CI",
                    value=False,
                    help="Calculate more robust confidence intervals using bootstrap resampling (takes 20-60 seconds)",
                    key="isotherm_bootstrap_checkbox",
                )
            with col_opt2:
                calculate_press = st.checkbox(
                    "📊 PRESS/Q²",
                    value=False,
                    help="Calculate PRESS statistic and Q² (predictive R²) using leave-one-out cross-validation",
                    key="isotherm_press_checkbox",
                )

            # Bootstrap iterations (show if bootstrap enabled)
            if run_bootstrap:
                n_bootstrap = st.slider(
                    "Bootstrap iterations",
                    min_value=200,
                    max_value=1000,
                    value=500,
                    step=100,
                    help="More iterations = more accurate CI but slower",
                    key="isotherm_bootstrap_slider",
                )
            else:
                n_bootstrap = BOOTSTRAP_DEFAULT_ITERATIONS
            # Check cache
            data_hash = _compute_data_hash(Ce, qe, C0)
            cached_hash = current_study_state.get("_isotherm_data_hash")
            cached_confidence = current_study_state.get("_isotherm_confidence_level")
            cached_temp = current_study_state.get("_isotherm_T_K")

            has_cached_results = (
                cached_hash == data_hash
                and cached_confidence == confidence_level
                and cached_temp == T_K
                and current_study_state.get("isotherm_models_fitted")
            )


            # Show results if cached OR if calculate button pressed
            if has_cached_results and not calculate_btn:
                st.success("✅ Using cached model results (click 'Fit Models' to recalculate)")
                fitted_models = current_study_state["isotherm_models_fitted"]
                show_results = True
            elif calculate_btn:
                # First, fit the models
                with st.spinner("🔬 Fitting isotherm models..."):
                    fitted_models = fit_isotherm_models_with_cache(
                        Ce, qe, C0, confidence_level, current_study_state, T_K=T_K
                    )


                # Count converged models
                converged_count = sum(
                    1 for m in fitted_models.values() if m and m.get("converged", False)
                )

                # Calculate PRESS/Q² if requested
                if calculate_press:
                    with st.spinner("📊 Calculating PRESS statistics (leave-one-out CV)..."):
                        from ..models import (
                            freundlich_model,
                            langmuir_model,
                            sips_model,
                            temkin_model,
                        )
                        from ..utils import calculate_press, calculate_q2

                        model_funcs = {
                            "Langmuir": (langmuir_model, ["qm", "KL"]),
                            "Freundlich": (freundlich_model, ["KF", "n_inv"]),
                            "Temkin": (temkin_model, ["B1", "KT"]),
                            "Sips": (sips_model, ["qm", "Ks", "ns"]),
                        }

                        for model_name, (func, param_names) in model_funcs.items():
                            if fitted_models.get(model_name, {}).get("converged"):
                                try:
                                    params = [
                                        fitted_models[model_name]["params"][p] for p in param_names
                                    ]
                                    press = calculate_press(func, Ce, qe, params)
                                    q2 = calculate_q2(press, qe)
                                    fitted_models[model_name]["press"] = press
                                    fitted_models[model_name]["q2"] = q2
                                except Exception as e:
                                    st.warning(f"PRESS calculation failed for {model_name}: {e}")

                        current_study_state["isotherm_models_fitted"] = fitted_models
                    st.success("✅ PRESS/Q² calculated!")
                else:
                    # Remove PRESS/Q² values when checkbox is unchecked
                    for model_name in fitted_models:
                        if fitted_models.get(model_name) and fitted_models[model_name].get(
                            "converged"
                        ):
                            fitted_models[model_name].pop("press", None)
                            fitted_models[model_name].pop("q2", None)
                    current_study_state["isotherm_models_fitted"] = fitted_models

                # Run bootstrap if requested
                if run_bootstrap:
                    with st.spinner("🔄 Running bootstrap analysis..."):
                        fitted_models = _run_isotherm_bootstrap(
                            Ce, qe, fitted_models, n_bootstrap, confidence_level, T_K=T_K
                        )
                    current_study_state["isotherm_models_fitted"] = fitted_models

                # Check for linear vs non-linear discrepancy
                if fitted_models:
                    _check_linear_nonlinear_warning(Ce, qe, fitted_models, current_study_state)

                # Only show results if models actually converged
                if converged_count > 0:
                    show_results = True
            else:
                st.info("👆 Click **'Fit Models'** to perform isotherm model fitting")
                show_results = False
                fitted_models = {}
                converged_count = 0

            # Display results only if we have them
            if show_results and fitted_models:
                valid_models = {
                    k: v for k, v in fitted_models.items() if v and v.get("converged", False)
                }

                if valid_models:
                    recommendations = recommend_best_models(valid_models, "isotherm")

                    if recommendations:
                        best = recommendations[0]
                        st.success(f"""
                        **🎯 Recommended Model: {best["model"]}**

                        **Confidence:** {best["confidence"]:.1f}% | **Adj-R²:** {best.get("adj_r_squared", best["r_squared"]):.4f} | **AIC Weight:** {best.get("aic_weight", 0) * 100:.1f}%

                        **Rationale:** {best["rationale"]}
                        """)

                # Model tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    ["Langmuir", "Freundlich", "Temkin", "Sips", "Comparison"]
                )

                with tab1:
                    _display_langmuir(Ce, qe, C0, fitted_models.get("Langmuir"))

                with tab2:
                    _display_freundlich(Ce, qe, fitted_models.get("Freundlich"))

                with tab3:
                    _display_temkin(Ce, qe, fitted_models.get("Temkin"))

                with tab4:
                    _display_sips(Ce, qe, fitted_models.get("Sips"))

                with tab5:
                    _display_model_comparison(fitted_models, Ce, qe, T_K=T_K)

            # Export info
            st.markdown("---")
            st.info(
                "💡 **To download figures and data:** Go to the **📦 Export All** tab for comprehensive exports with format options."
            )

    elif iso_input and input_mode == "absorbance" and not calib_params:
        st.warning(
            "⚠️ Complete calibration first, or switch to **Direct Concentration** input mode in the sidebar"
        )
    elif not iso_input:
        st.info("📥 Enter isotherm data in sidebar")
        _display_guidelines()


# =============================================================================
# CACHED DATA CALCULATION
# =============================================================================


@st.cache_data
def _calculate_isotherm_results(iso_input, calib_params):
    """Calculate isotherm equilibrium data with error propagation."""
    df = iso_input["data"].copy()
    params = iso_input["params"]

    slope = calib_params["slope"]
    intercept = calib_params["intercept"]
    m = params["m"]
    V = params["V"]

    # Get calibration uncertainties (with fallback defaults)
    slope_se = calib_params.get("std_err_slope", 0)
    intercept_se = calib_params.get("std_err_intercept", 0)

    results = []
    for _, row in df.iterrows():
        C0 = row["Concentration"]
        abs_eq = row["Absorbance"]

        Ce = calculate_Ce_from_absorbance(abs_eq, slope, intercept)
        qe = calculate_adsorption_capacity(C0, Ce, V, m)
        removal = calculate_removal_percentage(C0, Ce)

        # Calculate propagated uncertainty for Ce (returns tuple: Ce_calc, Ce_se)
        _, Ce_se = propagate_calibration_uncertainty(
            abs_eq, slope, intercept, slope_se, intercept_se
        )

        # Propagate Ce uncertainty to qe
        qe_error = (V / m) * Ce_se if m > 0 else 0

        results.append(
            {
                "C0_mgL": C0,
                "Ce_mgL": Ce,
                "Ce_error": Ce_se,
                "qe_mg_g": qe,
                "qe_error": qe_error,
                "removal_%": removal,
            }
        )

    if not results:
        return CalculationResult(success=False, error="No valid data points to calculate results.")
    results_df = pd.DataFrame(results).sort_values("C0_mgL")
    return CalculationResult(success=True, data=results_df)


@st.cache_data
def _calculate_isotherm_results_direct(iso_input):
    """
    Calculate isotherm equilibrium data from direct C0/Ce input.

    This function bypasses calibration and uses Ce values directly from published data.
    Useful for validating the application with literature datasets.
    """
    df = iso_input["data"].copy()
    params = iso_input["params"]

    m = params["m"]
    V = params["V"]

    results = []
    for _, row in df.iterrows():
        C0 = row["C0"]
        Ce = row["Ce"]

        # Validate Ce <= C0
        if Ce > C0:
            continue  # Skip invalid data points

        qe = calculate_adsorption_capacity(C0, Ce, V, m)
        removal = calculate_removal_percentage(C0, Ce)

        results.append(
            {
                "C0_mgL": C0,
                "Ce_mgL": Ce,
                "Ce_error": 0.0,  # No calibration uncertainty in direct mode
                "qe_mg_g": qe,
                "qe_error": 0.0,  # No propagated error in direct mode
                "removal_%": removal,
            }
        )

    if not results:
        return CalculationResult(
            success=False, error="No valid data points. Ensure Ce ≤ C0 for all rows."
        )
    results_df = pd.DataFrame(results).sort_values("C0_mgL")
    return CalculationResult(success=True, data=results_df)


# =============================================================================
# DISPLAY FUNCTIONS (unchanged from original)
# =============================================================================


def _display_langmuir(Ce, qe, C0, results):
    """Display Langmuir model results."""
    st.markdown("**Langmuir Isotherm (Monolayer Adsorption)**")
    st.latex(r"q_e = \frac{q_m \cdot K_L \cdot C_e}{1 + K_L \cdot C_e}")

    if results and results.get("converged"):
        params = results["params"]
        ci = results.get("ci_95", {})

        # Parameters table
        display_results_table(
            {
                "Parameter": ["qm (mg/g)", "KL (L/mg)"],
                "Value": [f"{params['qm']:.4f}", f"{params['KL']:.6f}"],
                "Std. Error": [f"{params.get('qm_se', 0):.4f}", f"{params.get('KL_se', 0):.6f}"],
                "95% CI": [
                    f"({ci.get('qm', (np.nan, np.nan))[0]:.4f}, {ci.get('qm', (np.nan, np.nan))[1]:.4f})",
                    f"({ci.get('KL', (np.nan, np.nan))[0]:.6f}, {ci.get('KL', (np.nan, np.nan))[1]:.6f})",
                ],
            }
        )

        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R²", f"{results['r_squared']:.4f}")
        with col2:
            st.metric("Adj-R²", f"{results['adj_r_squared']:.4f}")
        with col3:
            st.metric("RMSE", f"{results['rmse']:.4f}")
        with col4:
            st.metric("AIC", f"{results.get('aicc', results['aic']):.2f}")

        # Separation factor
        if "RL" in results:
            RL = results["RL"]
            st.markdown("**Separation Factor (RL):**")
            st.latex(r"R_L = \frac{1}{1 + K_L \cdot C_0}")

            # RL is calculated for each C0 value - display with corresponding C0
            # Handle both scalar and array cases
            RL_arr = np.atleast_1d(RL)
            C0_arr = np.atleast_1d(C0)

            # If lengths don't match, use only valid C0 values (same filtering as model fitting)
            if len(RL_arr) != len(C0_arr):
                # Filter to match - use first n values where n = len(RL)
                valid_mask = (np.atleast_1d(Ce) > EPSILON_DIV) & (np.atleast_1d(qe) > EPSILON_DIV)
                C0_filtered = C0_arr[valid_mask] if len(C0_arr) > 1 else C0_arr
                if len(C0_filtered) == len(RL_arr):
                    C0_arr = C0_filtered
                else:
                    # Fallback: just show RL values with index
                    C0_arr = np.arange(1, len(RL_arr) + 1)

            rl_df = pd.DataFrame({"C0 (mg/L)": C0_arr[: len(RL_arr)], "RL": RL_arr})
            display_results_table(rl_df.round(4))
            st.info(f"**Interpretation:** {results['RL_interpretation']}")

        # Plot
        Ce_line = np.linspace(0.01, Ce.max() * 1.1, 100)
        qe_pred = langmuir_model(Ce_line, params["qm"], params["KL"])

        fig = create_isotherm_plot(
            Ce,
            qe,
            Ce_line,
            qe_pred,
            model_name="Langmuir",
            r_squared=results["r_squared"],
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True, key="langmuir_plot")
        qe_pred_exp = langmuir_model(Ce, params["qm"], params["KL"])
        fig_parity = create_parity_plot(
            qe, qe_pred_exp, model_name="Langmuir", r_squared=results["r_squared"]
        )
        st.plotly_chart(fig_parity, use_container_width=True, key="langmuir_parity")
    else:
        error_msg = results.get("error", "Unknown error") if results else "Fitting failed"
        st.warning(f"Langmuir model did not converge: {error_msg}")


def _display_freundlich(Ce, qe, results):
    """Display Freundlich model results."""
    st.markdown("**Freundlich Isotherm (Heterogeneous Surface)**")
    st.latex(r"q_e = K_F \cdot C_e^{1/n}")

    if results and results.get("converged"):
        params = results["params"]
        ci = results.get("ci_95", {})

        display_results_table(
            {
                "Parameter": ["KF ((mg/g)(L/mg)^1/n)", "1/n", "n"],
                "Value": [
                    f"{params['KF']:.4f}",
                    f"{params['n_inv']:.4f}",
                    f"{params.get('n', np.nan):.4f}",
                ],
                "Std. Error": [
                    f"{params.get('KF_se', 0):.4f}",
                    f"{params.get('n_inv_se', 0):.4f}",
                    "—",
                ],
                "95% CI": [
                    f"({ci.get('KF', (np.nan, np.nan))[0]:.4f}, {ci.get('KF', (np.nan, np.nan))[1]:.4f})",
                    f"({ci.get('n_inv', (np.nan, np.nan))[0]:.4f}, {ci.get('n_inv', (np.nan, np.nan))[1]:.4f})",
                    "—",
                ],
            }
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R²", f"{results['r_squared']:.4f}")
        with col2:
            st.metric("Adj-R²", f"{results['adj_r_squared']:.4f}")
        with col3:
            st.metric("RMSE", f"{results['rmse']:.4f}")
        with col4:
            st.metric("AIC", f"{results.get('aicc', results['aic']):.2f}")

        # Interpretation
        n = params.get("n", 1)
        if n > 1:
            st.success("**n > 1:** Favorable adsorption")
        elif n < 1:
            st.warning("**n < 1:** Unfavorable adsorption")
        else:
            st.info("**n ≈ 1:** Linear adsorption")

        # Plot
        Ce_line = np.linspace(0.01, Ce.max() * 1.1, 100)
        qe_pred = freundlich_model(Ce_line, params["KF"], params["n_inv"])

        fig = create_isotherm_plot(
            Ce,
            qe,
            Ce_line,
            qe_pred,
            model_name="Freundlich",
            r_squared=results["r_squared"],
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True, key="freundlich_plot")
    else:
        error_msg = results.get("error", "Unknown error") if results else "Fitting failed"
        st.warning(f"Freundlich model did not converge: {error_msg}")


def _display_temkin(Ce, qe, results):
    """Display Temkin model results."""
    st.markdown("**Temkin Isotherm (Heat of Adsorption)**")
    st.latex(r"q_e = B_1 \cdot \ln(K_T \cdot C_e)")

    if results and results.get("converged"):
        params = results["params"]
        ci = results.get("ci_95", {})

        display_results_table(
            {
                "Parameter": ["B1 (J/mol)", "KT (L/mg)"],
                "Value": [f"{params['B1']:.4f}", f"{params['KT']:.6f}"],
                "Std. Error": [f"{params.get('B1_se', 0):.4f}", f"{params.get('KT_se', 0):.6f}"],
                "95% CI": [
                    f"({ci.get('B1', (np.nan, np.nan))[0]:.4f}, {ci.get('B1', (np.nan, np.nan))[1]:.4f})",
                    f"({ci.get('KT', (np.nan, np.nan))[0]:.6f}, {ci.get('KT', (np.nan, np.nan))[1]:.6f})",
                ],
            }
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R²", f"{results['r_squared']:.4f}")
        with col2:
            st.metric("Adj-R²", f"{results['adj_r_squared']:.4f}")
        with col3:
            st.metric("RMSE", f"{results['rmse']:.4f}")
        with col4:
            st.metric("AIC", f"{results.get('aicc', results['aic']):.2f}")

        # Plot
        Ce_line = np.linspace(0.01, Ce.max() * 1.1, 100)
        qe_pred = temkin_model(Ce_line, params["B1"], params["KT"])
        if results.get("temkin_domain_warning"):
            st.warning(
                "⚠️ **Temkin model validity concern:** KT × Ce < 1 for some data points, "
                "producing negative predicted qe values (clamped to 0). This indicates the "
                "Temkin model may not be appropriate for the low-concentration range of your data."
            )

        fig = create_isotherm_plot(
            Ce,
            qe,
            Ce_line,
            qe_pred,
            model_name="Temkin",
            r_squared=results["r_squared"],
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True, key="temkin_plot")
    else:
        error_msg = results.get("error", "Unknown error") if results else "Fitting failed"
        st.warning(f"Temkin model did not converge: {error_msg}")


def _display_sips(Ce, qe, results):
    """Display Sips model results."""
    st.markdown("**Sips Isotherm (Langmuir-Freundlich)**")
    st.latex(r"q_e = \frac{q_m \cdot (K_s \cdot C_e)^{n_s}}{1 + (K_s \cdot C_e)^{n_s}}")

    if results and results.get("converged"):
        params = results["params"]
        ci = results.get("ci_95", {})

        display_results_table(
            {
                "Parameter": ["qm (mg/g)", "Ks (L/mg)", "ns"],
                "Value": [f"{params['qm']:.4f}", f"{params['Ks']:.6f}", f"{params['ns']:.4f}"],
                "Std. Error": [
                    f"{params.get('qm_se', 0):.4f}",
                    f"{params.get('Ks_se', 0):.6f}",
                    f"{params.get('ns_se', 0):.4f}",
                ],
                "95% CI": [
                    f"({ci.get('qm', (np.nan, np.nan))[0]:.4f}, {ci.get('qm', (np.nan, np.nan))[1]:.4f})",
                    f"({ci.get('Ks', (np.nan, np.nan))[0]:.6f}, {ci.get('Ks', (np.nan, np.nan))[1]:.6f})",
                    f"({ci.get('ns', (np.nan, np.nan))[0]:.4f}, {ci.get('ns', (np.nan, np.nan))[1]:.4f})",
                ],
            }
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R²", f"{results['r_squared']:.4f}")
        with col2:
            st.metric("Adj-R²", f"{results['adj_r_squared']:.4f}")
        with col3:
            st.metric("RMSE", f"{results['rmse']:.4f}")
        with col4:
            st.metric("AIC", f"{results.get('aicc', results['aic']):.2f}")

        # Interpretation
        ns = params["ns"]
        if abs(ns - 1) < 0.1:
            st.info("**ns ≈ 1:** Reduces to Langmuir model (homogeneous surface)")
        else:
            st.info(f"**ns = {ns:.2f}:** Surface heterogeneity parameter")

        # Plot
        Ce_line = np.linspace(0.01, Ce.max() * 1.1, 100)
        qe_pred = sips_model(Ce_line, params["qm"], params["Ks"], params["ns"])

        fig = create_isotherm_plot(
            Ce, qe, Ce_line, qe_pred, model_name="Sips", r_squared=results["r_squared"], height=450
        )
        st.plotly_chart(fig, use_container_width=True, key="sips_plot")
    else:
        error_msg = results.get("error", "Unknown error") if results else "Fitting failed"
        st.warning(f"Sips model did not converge: {error_msg}")


def _display_model_comparison(fitted_models, Ce, qe, T_K: float = 298.15):
    """Display comprehensive model comparison with publication-standard error functions."""
    st.markdown("**📊 Model Comparison**")

    # Check if PRESS was calculated
    has_press = any(
        results.get("press") is not None for results in fitted_models.values() if results
    )

    # Build comparison table with extended error functions
    comparison_data = []
    for name, results in fitted_models.items():
        if results and results.get("converged"):
            row = {
                "Model": name,
                "R²": results["r_squared"],
                "Adj-R²": results["adj_r_squared"],
                "RMSE": results["rmse"],
                "χ²": results.get("chi_squared", np.nan),
                "AIC": results.get("aicc", results["aic"]),
                "BIC": results.get("bic", np.nan),
            }
            # Add PRESS/Q² if available
            if has_press:
                row["PRESS"] = results.get("press", np.nan)
                row["Q²"] = results.get("q2", np.nan)
            comparison_data.append(row)

    if not comparison_data:
        st.warning("No models converged successfully.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Calculate Akaike weights
    aicc_values = [fitted_models[row["Model"]].get("aicc", row["AIC"]) for row in comparison_data]
    aic_weights = calculate_akaike_weights(aicc_values)
    comparison_df["AIC Weight"] = aic_weights

    # Sort by Adj-R²
    comparison_df = comparison_df.sort_values("Adj-R²", ascending=False)

    # Error function explanation expander
    with st.expander("📖 Error Function Definitions", expanded=False):
        definitions = """
        | Error Function | Formula | Best For |
        |----------------|---------|----------|
        | **R²** | 1 - SSE/SST | Overall fit quality |
        | **Adj-R²** | Penalizes extra parameters | Model comparison |
        | **RMSE** | √(SSE/n) | Absolute error magnitude |
        | **χ²** | Σ(residual²/predicted) | Relative error |
        | **AIC/BIC** | Information criteria | Model selection |
        """
        if has_press:
            definitions += """| **PRESS** | Leave-one-out CV error | Predictive ability |
        | **Q²** | 1 - PRESS/SS_tot | Predictive R² |
        """
        definitions += """
        *References: Kumar et al. (2008) J Hazard Mater 151:794-804; Foo & Hameed (2010) Chem Eng J 156:2-10*
        """
        st.markdown(definitions)

    # Display main comparison table
    highlight_max_cols = ["R²", "Adj-R²", "AIC Weight"]
    highlight_min_cols = ["RMSE", "AIC", "BIC", "χ²"]

    if has_press:
        highlight_max_cols.append("Q²")
        highlight_min_cols.append("PRESS")

    st.dataframe(
        comparison_df.style.format(
            {
                "R²": "{:.4f}",
                "Adj-R²": "{:.4f}",
                "RMSE": "{:.4f}",
                "χ²": "{:.2f}",
                "AIC": "{:.2f}",
                "BIC": "{:.2f}",
                "AIC Weight": "{:.1%}",
            }
        )
        .highlight_max(subset=highlight_max_cols, color="lightgreen")
        .highlight_min(subset=highlight_min_cols, color="lightblue"),
        use_container_width=True,
        hide_index=True,
    )

    # Best model recommendation based on multiple criteria
    # Handle NaN values safely to avoid KeyError
    try:
        if comparison_df["Adj-R²"].notna().any():
            adj_r2_idx = comparison_df["Adj-R²"].idxmax()
            best_adj_r2 = comparison_df.loc[adj_r2_idx, "Model"]
        else:
            best_adj_r2 = None
    except (KeyError, ValueError):
        best_adj_r2 = None

    try:
        aic_idx = comparison_df["AIC"].idxmin()
        best_aic = comparison_df.loc[aic_idx, "Model"] if pd.notna(aic_idx) else None
    except (KeyError, ValueError):
        best_aic = None

    if best_adj_r2 and best_aic:
        if best_adj_r2 == best_aic:
            st.success(f"**🎯 Best Model: {best_adj_r2}** (highest Adj-R² and lowest AIC)")
        else:
            st.info(
                f"**📊 Model Selection:** Adj-R² favors **{best_adj_r2}**, AIC favors **{best_aic}**"
            )
    elif best_adj_r2:
        st.success(f"**🎯 Best Model: {best_adj_r2}** (highest Adj-R²)")
    elif best_aic:
        st.success(f"**🎯 Best Model: {best_aic}** (lowest AIC)")

    # All models plot
    st.markdown("**📈 All Models Overlay**")

    model_functions = {
        "Langmuir": lambda x, p: langmuir_model(x, p["qm"], p["KL"]),
        "Freundlich": lambda x, p: freundlich_model(x, p["KF"], p["n_inv"]),
        "Temkin": lambda x, p: temkin_model(x, p["B1"], p["KT"]),
        "Sips": lambda x, p: sips_model(x, p["qm"], p["Ks"], p["ns"]),
    }

    fig = create_model_comparison_plot(
        Ce,
        qe,
        fitted_models,
        model_functions,
        x_label="C<sub>e</sub> (mg/L)",
        y_label="q<sub>e</sub> (mg/g)",
        title="Isotherm Model Comparison",
    )
    st.plotly_chart(fig, use_container_width=True, key="model_comparison_plot")


def _display_guidelines():
    """Display isotherm guidelines."""
    with st.expander("Isotherm Analysis Guidelines", expanded=True):
        st.markdown("""
**Best Practices:**

1. **Data Points:** Use 6-10 initial concentrations
2. **Concentration Range:** Cover 10-fold range (e.g., 10-100 mg/L)
3. **Equilibrium:** Ensure true equilibrium (check with kinetics)
4. **Replicates:** Triplicates for error estimation
5. **Report:**
   - All parameters with 95% CI
   - R-squared, Adj-R-squared, RMSE, AIC for each model
   - Akaike weights for model selection
   - Separation factor (RL) for Langmuir
        """)
