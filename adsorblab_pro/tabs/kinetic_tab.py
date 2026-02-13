# tabs/kinetic_tab.py
"""
Kinetic Tab - AdsorbLab Pro
===========================
Features:
- PFO, PSO, rPSO, Elovich, IPD models with non-linear regression
- 95% confidence intervals via bootstrap resampling
- Weber-Morris and Boyd plot diffusion analysis
- Biot number calculation for rate-limiting step identification
- Comprehensive model comparison (R², Adj-R², RMSE, AIC, χ²)
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
    calculate_biot_number,
    calculate_initial_rate,
    elovich_model,
    fit_model_with_ci,
    identify_equilibrium_time,
    identify_rate_limiting_step,
    pfo_model,
    pso_model,
    revised_pso_model_fixed_conditions,
)
from ..plot_style import (
    COLORS,
    MARKERS,
    apply_professional_style,
    create_dual_axis_effect_plot,
    create_kinetic_plot,
    create_model_comparison_plot,
    create_parity_plot,
    style_experimental_trace,
    style_fit_trace,
)
from ..utils import (
    CalculationResult,
    assess_data_quality,
    calculate_adsorption_capacity,
    calculate_akaike_weights,
    calculate_Ce_from_absorbance,
    calculate_error_metrics,
    calculate_removal_percentage,
    display_results_table,
    get_current_study_state,
    propagate_calibration_uncertainty,
    recommend_best_models,
    validate_required_params,
)
from ..validation import format_validation_errors, validate_kinetic_data


def _validate_kinetic_input(time, qt, Ct=None, C0=None):
    """
    Validate kinetic data before analysis.

    Returns:
        tuple: (is_valid, validation_report)
    """
    validation_report = validate_kinetic_data(
        time=np.array(time),
        qt=np.array(qt) if qt is not None else None,
        Ct=np.array(Ct) if Ct is not None else None,
        C0=C0,
    )
    return validation_report.is_valid, validation_report


def _display_kinetic_validation(validation_report):
    """Display kinetic validation results."""
    if not validation_report.is_valid:
        st.error("❌ **Kinetic Data Validation Failed**")
        st.markdown(format_validation_errors(validation_report))
        return False

    if validation_report.has_warnings:
        with st.expander("⚠️ Kinetic Data Quality Warnings", expanded=True):
            for w in validation_report.warnings:
                st.warning(w.message)
                if w.suggestion:
                    st.caption(f"💡 {w.suggestion}")

    return True


# =============================================================================
# CACHING UTILITIES
# =============================================================================


def _compute_kinetic_data_hash(t: np.ndarray, qt: np.ndarray) -> str:
    """Compute hash of kinetic data for cache validation."""
    combined = np.concatenate([t, qt]).tobytes()
    return hashlib.md5(combined).hexdigest()


def _run_bootstrap_on_fitted_models(t, qt, fitted_models, n_bootstrap, confidence_level):
    """
    Run bootstrap CI on already fitted models with visual progress.

    This function adds bootstrap confidence intervals to models that have
    already been fitted, showing a progress bar for user feedback.
    """

    valid = (t >= 0) & (qt >= 0)
    t_v, qt_v = t[valid], qt[valid]

    # Build list of models to bootstrap
    models_to_bootstrap = [
        ("PFO", pfo_model, ["qe", "k1"]),
        ("PSO", pso_model, ["qe", "k2"]),
        ("Elovich", elovich_model, ["alpha", "beta"]),
    ]

    # Add rPSO if it was fitted (requires experimental conditions)
    if fitted_models.get("rPSO", {}).get("converged"):
        cond = fitted_models["rPSO"].get("experimental_conditions", {})
        if cond:
            C0, m, V = cond.get("C0", 0), cond.get("m", 0), cond.get("V", 0)
            if all([C0 > 0, m > 0, V > 0]):
                rpso_model = revised_pso_model_fixed_conditions(C0, m, V)
                models_to_bootstrap.append(("rPSO", rpso_model, ["qe", "k2"]))

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

        # Use appropriate data for Elovich (t > 0)
        if model_name == "Elovich":
            t_data = t_v[t_v > 0]
            qt_data = qt_v[t_v > 0]
        else:
            t_data = t_v
            qt_data = qt_v

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
            from ..utils import bootstrap_confidence_intervals

            ci_lower, ci_upper = bootstrap_confidence_intervals(
                model_func,
                t_data,
                qt_data,
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


def _arrays_to_tuples(t: np.ndarray, qt: np.ndarray):
    """Convert numpy arrays to tuples for cache key hashing."""
    return (tuple(np.round(t, 8).tolist()), tuple(np.round(qt, 8).tolist()))


# =============================================================================
# CACHED MODEL FITTING
# =============================================================================


@st.cache_data(show_spinner=False, ttl=3600, max_entries=50)
def _fit_all_kinetic_models_cached(
    t_tuple: tuple,
    qt_tuple: tuple,
    confidence_level: float = 0.95,
    experimental_conditions: tuple | None = None,  # (C0, m, V) for rPSO
) -> dict:
    """
    Fit all kinetic models with confidence intervals.

    IMPORTANT: This is a cached pure function - no side effects!

    Parameters
    ----------
    t_tuple : tuple
        Time values as tuple (hashable)
    qt_tuple : tuple
        qt values as tuple (hashable)
    confidence_level : float
        Confidence level for CI calculation
    experimental_conditions : tuple, optional
        (C0, m, V) for rPSO model. If None, rPSO is skipped.
        C0 = initial concentration (mg/L)
        m = adsorbent mass (g)
        V = solution volume (L)

    Returns
    -------
    dict
        Dictionary of model results
    """
    # Convert back to numpy arrays
    t = np.array(t_tuple)
    qt = np.array(qt_tuple)

    fitted: dict[str, Any] = {}

    valid = (t >= 0) & (qt >= 0)
    t_v, qt_v = t[valid], qt[valid]

    if len(t_v) < 4:
        return fitted

    qe_exp = qt_v.max()

    # PFO
    try:
        result = fit_model_with_ci(
            pfo_model,
            t_v,
            qt_v,
            p0=[qe_exp, 0.05],
            bounds=([0, 0], [qe_exp * 3, 10]),
            param_names=["qe", "k1"],
            confidence=confidence_level,
        )
        if result and result.get("converged"):
            result["params"]["t_half"] = (
                np.log(2) / result["params"]["k1"] if result["params"]["k1"] > 0 else np.nan
            )
            fitted["PFO"] = result
    except Exception as e:
        fitted["PFO"] = {"converged": False, "error": str(e)}

    # PSO (with mechanistic warning)
    try:
        result = fit_model_with_ci(
            pso_model,
            t_v,
            qt_v,
            p0=[qe_exp, 0.01],
            bounds=([0, 0], [qe_exp * 3, 10]),
            param_names=["qe", "k2"],
            confidence=confidence_level,
        )
        if result and result.get("converged"):
            qe, k2 = result["params"]["qe"], result["params"]["k2"]
            result["params"]["h"] = k2 * qe**2
            result["params"]["t_half"] = 1 / (k2 * qe) if k2 * qe > 0 else np.nan
            # Add mechanistic warning
            result["mechanistic_warning"] = (
                "⚠️ PSO fit does NOT imply chemisorption. ~90% of kinetic studies "
                "report PSO as 'best fit' regardless of actual mechanism. This is "
                "a statistical artifact. See Hubbe et al. (2019), BioResources."
            )
            fitted["PSO"] = result
    except Exception as e:
        fitted["PSO"] = {"converged": False, "error": str(e)}

    # rPSO (Revised PSO with concentration correction) - only if conditions provided
    if experimental_conditions is not None:
        try:
            C0, m, V = experimental_conditions
            if C0 > 0 and m > 0 and V > 0:
                # Create model with fixed experimental conditions
                rpso_model = revised_pso_model_fixed_conditions(C0, m, V)

                result = fit_model_with_ci(
                    rpso_model,
                    t_v,
                    qt_v,
                    p0=[qe_exp, 0.01],
                    bounds=([0, 0], [qe_exp * 3, 10]),
                    param_names=["qe", "k2"],
                    confidence=confidence_level,
                )
                if result and result.get("converged"):
                    qe, k2 = result["params"]["qe"], result["params"]["k2"]
                    # Calculate correction factor phi
                    phi = 1 + (qe * m) / (C0 * V)
                    result["params"]["phi"] = phi
                    result["params"]["h"] = k2 * qe**2  # Initial rate
                    result["params"]["t_half"] = (
                        1 / (k2 * qe * phi) if k2 * qe * phi > 0 else np.nan
                    )
                    result["experimental_conditions"] = {"C0": C0, "m": m, "V": V}
                    result["reference"] = "Bullen et al. (2021). Langmuir, 37(10), 3189-3201"
                    fitted["rPSO"] = result
        except Exception as e:
            fitted["rPSO"] = {"converged": False, "error": str(e)}

    # Elovich
    try:
        t_pos, qt_pos = t_v[t_v > 0], qt_v[t_v > 0]
        if len(t_pos) >= 3:
            result = fit_model_with_ci(
                elovich_model,
                t_pos,
                qt_pos,
                p0=[1.0, 0.1],
                bounds=([0, 0], [1000, 10]),
                param_names=["alpha", "beta"],
                confidence=confidence_level,
            )
            if result and result.get("converged"):
                fitted["Elovich"] = result
    except Exception as e:
        fitted["Elovich"] = {"converged": False, "error": str(e)}

    # IPD (linear model)
    try:
        from scipy.stats import linregress

        sqrt_t = np.sqrt(t_v)
        slope, intercept, r_val, p_val, std_err = linregress(sqrt_t, qt_v)

        y_pred = slope * sqrt_t + intercept
        metrics = calculate_error_metrics(qt_v, y_pred, 2)

        fitted["IPD"] = {
            "converged": True,
            "params": {"kid": slope, "C": intercept, "kid_se": std_err},
            "r_squared": r_val**2,
            "adj_r_squared": metrics["adj_r_squared"],
            "rmse": metrics["rmse"],
            "aic": metrics.get("aic", np.inf),
            "bic": metrics.get("bic", np.nan),
            "mechanism": "Intraparticle diffusion controlled"
            if abs(intercept) < 1
            else "Boundary layer effect present",
        }
    except Exception as e:
        fitted["IPD"] = {"converged": False, "error": str(e)}

    return fitted


# =============================================================================
# WRAPPER WITH CACHE MANAGEMENT
# =============================================================================


def fit_kinetic_models_with_cache(
    t, qt, confidence_level, current_study_state, experimental_conditions=None
):
    """
    Wrapper that manages caching and session state updates.

    Parameters
    ----------
    t, qt : np.ndarray
        Input data arrays
    confidence_level : float
        Confidence level for CI
    current_study_state : dict
        Reference to current study's state
    experimental_conditions : tuple, optional
        (C0, m, V) for rPSO model fitting

    Returns
    -------
    dict
        Fitted model results
    """
    # Compute data hash (include conditions for rPSO)
    data_hash = _compute_kinetic_data_hash(t, qt)
    if experimental_conditions:
        cond_str = f"_{experimental_conditions[0]}_{experimental_conditions[1]}_{experimental_conditions[2]}"
        data_hash = hashlib.md5((data_hash + cond_str).encode()).hexdigest()

    # Check cache validity
    cached_hash = current_study_state.get("_kinetic_data_hash")
    cached_models = current_study_state.get("kinetic_models_fitted")
    cached_confidence = current_study_state.get("_kinetic_confidence_level")

    if (
        cached_models
        and cached_hash == data_hash
        and cached_confidence == confidence_level
        and len(cached_models) > 0
    ):
        # Cache hit!
        return cached_models

    # Cache miss - fit models
    t_tuple, qt_tuple = _arrays_to_tuples(t, qt)

    fitted_models = _fit_all_kinetic_models_cached(
        t_tuple, qt_tuple, confidence_level, experimental_conditions=experimental_conditions
    )

    # Update session state (OUTSIDE cached function)
    current_study_state["kinetic_models_fitted"] = fitted_models
    current_study_state["_kinetic_data_hash"] = data_hash
    current_study_state["_kinetic_confidence_level"] = confidence_level

    # Store for 3D explorer
    if fitted_models.get("PSO", {}).get("converged"):
        current_study_state["pso_params_nonlinear"] = fitted_models["PSO"]
    if fitted_models.get("PFO", {}).get("converged"):
        current_study_state["pfo_params_nonlinear"] = fitted_models["PFO"]

    return fitted_models


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================


def render():
    """Render kinetic analysis with professional statistics."""
    st.subheader("⏱️ Adsorption Kinetics Analysis")
    st.markdown("*Multi-model fitting with confidence intervals and mechanism identification*")

    current_study_state = get_current_study_state()
    if not current_study_state:
        st.info("Please add or select a study from the sidebar to begin analysis.")
        return

    unit_system = current_study_state.get("unit_system", "mg/g")
    confidence_level = current_study_state.get("confidence_level", 0.95)

    kin_input = current_study_state.get("kinetic_input")
    calib_params = current_study_state.get("calibration_params")

    # Check input mode (default to 'absorbance' for backward compatibility)
    input_mode = kin_input.get("input_mode", "absorbance") if kin_input else "absorbance"

    # Determine if we can proceed based on input mode
    can_proceed = False
    if kin_input:
        if input_mode == "direct":
            # Direct mode doesn't require calibration
            can_proceed = True
        elif calib_params:
            # Absorbance mode requires calibration
            can_proceed = True

    if can_proceed:
        is_valid, error_message = validate_required_params(
            params=kin_input["params"], required_keys=[("C0", "C₀"), ("m", "Mass"), ("V", "Volume")]
        )
        if not is_valid:
            st.warning(error_message, icon="⚠️")
            return

        quality = assess_data_quality(kin_input["data"], "kinetic")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Quality", f"{quality['quality_score']}/100")
        with col2:
            st.metric("Points", len(kin_input["data"]))
        with col3:
            status = "✅ Good" if quality["quality_score"] >= 70 else "⚠️ Review"
            st.metric("Status", status)

        # Calculate kinetic data based on input mode
        if input_mode == "direct":
            kin_results_obj = _calculate_kinetic_results_direct(kin_input)
            st.caption("📈 *Using direct concentration input (Ct values)*")
        else:
            kin_results_obj = _calculate_kinetic_results(kin_input, calib_params)

        if not kin_results_obj.success:
            st.warning(f"Could not process kinetic data: {kin_results_obj.error}")
            return

        kin_results = kin_results_obj.data
        if kin_results is not None and not kin_results.empty:
            current_study_state["kinetic_results_df"] = kin_results

            # Data Section
            st.markdown("---")
            st.markdown("### 📊 Kinetic Data")

            st.latex(r"q_t = \frac{(C_0 - C_t) \cdot V}{m}")

            display_cols = ["Time", "Ct_mgL", "qt_mg_g", "removal_%"]
            display_results_table(kin_results[display_cols].round(4), hide_index=False)

            params = kin_input["params"]
            st.caption(
                f"**Conditions:** C₀ = {params['C0']} mg/L | m = {params['m']} g | V = {params['V']} L"
            )

            # Key metrics
            t = kin_results["Time"].values
            qt = kin_results["qt_mg_g"].values

            # NEW: Validate kinetic data before analysis
            params = kin_input["params"]
            is_valid, validation_report = _validate_kinetic_input(
                time=t, qt=qt, C0=params.get("C0")
            )

            if not _display_kinetic_validation(validation_report):
                st.info("Please correct the data issues above before fitting models.")
                st.stop()

            qe_exp = qt[-1]
            eq_time = identify_equilibrium_time(t, qt)
            initial_rate = calculate_initial_rate(t, qt)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("qe (experimental)", f"{qe_exp:.2f} mg/g")
            with col2:
                st.metric("Equilibrium Time", f"{eq_time:.0f} min" if eq_time else "N/A")
            with col3:
                st.metric(
                    "Initial Rate", f"{initial_rate:.4f} mg/(g·min)" if initial_rate else "N/A"
                )

            # Kinetic curve
            st.markdown("---")
            st.markdown("### 📈 Kinetic Curve")

            y_col = "qt_mg_g" if unit_system == "mg/g" else "removal_%"
            y_label = "qt (mg/g)" if unit_system == "mg/g" else "Removal (%)"

            if unit_system == "Both":
                fig_kin = create_dual_axis_effect_plot(
                    x=kin_results["Time"],
                    y1=kin_results["qt_mg_g"],
                    y2=kin_results["removal_%"],
                    title="Kinetic Curve",
                    x_title="Time (min)",
                    y1_title="qt (mg/g)",
                    y2_title="Removal (%)",
                    y1_name="qt (mg/g)",
                    y2_name="Removal (%)",
                    height=450,
                    x_tozero=True,
                    y1_tozero=True,
                    y2_tozero=True,
                )
            else:
                fig_kin = go.Figure()

                tr = style_experimental_trace(name="Experimental")
                tr["mode"] = "markers+lines"
                tr["line"] = {"width": 2.5, "color": COLORS["experimental"]}
                tr["hovertemplate"] = "Time: %{x:.1f}<br>Value: %{y:.4f}<extra></extra>"

                fig_kin.add_trace(
                    go.Scatter(
                        x=kin_results["Time"],
                        y=kin_results[y_col],
                        **tr,
                    )
                )

                fig_kin = apply_professional_style(
                    fig_kin,
                    title="Kinetic Curve",
                    x_title="Time (min)",
                    y_title=y_label,
                    height=450,
                    show_legend=True,
                    legend_position="lower right",
                )
                fig_kin.update_xaxes(rangemode="tozero")
                fig_kin.update_yaxes(rangemode="tozero")

            st.plotly_chart(fig_kin, use_container_width=True, key="kinetic_overview_chart")

            # Model Fitting Section
            st.markdown("---")
            st.markdown("### 🔬 Model Fitting")

            # Options row with advanced settings
            col_btn, col_opt1, col_opt2 = st.columns([1, 1, 1])

            with col_btn:
                calculate_btn = st.button(
                    "🧮 Fit Models",
                    type="primary",
                    help="Click to fit all kinetic models to your data",
                    key="kinetic_calculate_btn",
                )

            with col_opt1:
                run_bootstrap = st.checkbox(
                    "🔄 Bootstrap CI",
                    value=False,
                    help="Calculate more robust confidence intervals using bootstrap resampling (takes 10-30 seconds)",
                    key="kinetic_bootstrap_checkbox",
                )
            with col_opt2:
                calculate_press = st.checkbox(
                    "📊 PRESS/Q²",
                    value=False,
                    help="Calculate PRESS statistic and Q² (predictive R²) using leave-one-out cross-validation",
                    key="kinetic_press_checkbox",
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
                    key="kinetic_bootstrap_slider",
                )
            else:
                n_bootstrap = BOOTSTRAP_DEFAULT_ITERATIONS

            # Check cache - include experimental conditions in hash for consistency
            data_hash = _compute_kinetic_data_hash(t, qt)
            params = kin_input.get("params", {})
            C0_cache = params.get("C0", 0)
            m_cache = params.get("m", 0)
            V_cache = params.get("V", 0)
            if all([C0_cache > 0, m_cache > 0, V_cache > 0]):
                cond_str = f"_{C0_cache}_{m_cache}_{V_cache}"
                data_hash = hashlib.md5((data_hash + cond_str).encode()).hexdigest()

            cached_hash = current_study_state.get("_kinetic_data_hash")
            has_cached_results = cached_hash == data_hash and current_study_state.get(
                "kinetic_models_fitted"
            )

            # Show results if cached OR if calculate button pressed
            if has_cached_results:
                st.success("✅ Using cached model results (click 'Fit Models' to recalculate)")
                fitted_models = current_study_state["kinetic_models_fitted"]
                show_results = True
            elif calculate_btn:
                # Clear old diffusion results when re-fitting models
                if "diffusion_results" in st.session_state:
                    del st.session_state["diffusion_results"]

                # Extract experimental conditions for rPSO model
                params = kin_input.get("params", {})
                C0 = params.get("C0", 0)
                m = params.get("m", 0)
                V = params.get("V", 0)
                experimental_conditions = (C0, m, V) if all([C0 > 0, m > 0, V > 0]) else None

                model_count = "5" if experimental_conditions else "4"
                with st.spinner(f"🔄 Fitting {model_count} kinetic models..."):
                    fitted_models = fit_kinetic_models_with_cache(
                        t,
                        qt,
                        confidence_level,
                        current_study_state,
                        experimental_conditions=experimental_conditions,
                    )
                converged_count = len([m for m in fitted_models.values() if m.get("converged")])

                if converged_count == 0:
                    st.warning(
                        "⚠️ No models could be fitted. Please ensure you have at least 4 data points with valid (positive) time and qt values."
                    )
                    show_results = False
                else:
                    st.success(f"✅ Fitted {converged_count} models successfully!")

                # Calculate PRESS/Q² if requested
                if calculate_press:
                    with st.spinner("📊 Calculating PRESS statistics (leave-one-out CV)..."):
                        from ..utils import calculate_press, calculate_q2

                        model_funcs = {
                            "PFO": (pfo_model, ["qe", "k1"]),
                            "PSO": (pso_model, ["qe", "k2"]),
                            "Elovich": (elovich_model, ["alpha", "beta"]),
                        }

                        # Add rPSO if fitted
                        if fitted_models.get("rPSO", {}).get("converged"):
                            cond = fitted_models["rPSO"].get("experimental_conditions", {})
                            if cond:
                                C0_r, m_r, V_r = (
                                    cond.get("C0", 0),
                                    cond.get("m", 0),
                                    cond.get("V", 0),
                                )
                                if all([C0_r > 0, m_r > 0, V_r > 0]):
                                    rpso_model = revised_pso_model_fixed_conditions(C0_r, m_r, V_r)
                                    model_funcs["rPSO"] = (rpso_model, ["qe", "k2"])

                        for model_name, (func, param_names) in model_funcs.items():
                            if fitted_models.get(model_name, {}).get("converged"):
                                try:
                                    params = [
                                        fitted_models[model_name]["params"][p] for p in param_names
                                    ]
                                    t_data = t[t > 0] if model_name == "Elovich" else t
                                    qt_data = qt[t > 0] if model_name == "Elovich" else qt
                                    press = calculate_press(func, t_data, qt_data, params)
                                    q2 = calculate_q2(press, qt_data)
                                    fitted_models[model_name]["press"] = press
                                    fitted_models[model_name]["q2"] = q2
                                except Exception as e:
                                    st.warning(f"PRESS calculation failed for {model_name}: {e}")

                        current_study_state["kinetic_models_fitted"] = fitted_models
                    st.success("✅ PRESS/Q² calculated!")

                # Run bootstrap if requested
                if run_bootstrap:
                    with st.spinner("🔄 Running bootstrap analysis..."):
                        fitted_models = _run_bootstrap_on_fitted_models(
                            t, qt, fitted_models, n_bootstrap, confidence_level
                        )
                    current_study_state["kinetic_models_fitted"] = fitted_models

                # Only show results if models actually converged
                if converged_count > 0:
                    show_results = True
            else:
                st.info("👆 Click **'Fit Models'** to perform kinetic model fitting")
                show_results = False
                fitted_models = {}

            # Display results only if we have them
            if show_results and fitted_models:
                valid_models = {
                    k: v for k, v in fitted_models.items() if v and v.get("converged", False)
                }

                # Get qe_exp and experimental_conditions for diffusion analysis
                qe_exp = qt.max()
                params = kin_input.get("params", {})
                C0 = params.get("C0", 0)
                m = params.get("m", 0)
                V = params.get("V", 0)
                experimental_conditions = (C0, m, V) if all([C0 > 0, m > 0, V > 0]) else None

                if valid_models:
                    recommendations = recommend_best_models(valid_models, "kinetic")

                    if recommendations:
                        best = recommendations[0]
                        st.success(f"""
                        **🎯 Recommended Model: {best["model"]}**

                        **Confidence:** {best["confidence"]:.1f}% | **Adj-R²:** {best.get("adj_r_squared", best["r_squared"]):.4f}

                        **Rationale:** {best["rationale"]}
                        """)

                # Model tabs
                tab1, tab2, tab2b, tab3, tab4, tab5, tab6 = st.tabs(
                    ["PFO", "PSO", "rPSO", "Elovich", "IPD", "Diffusion", "Comparison"]
                )

                with tab1:
                    _display_pfo(t, qt, fitted_models.get("PFO"))

                with tab2:
                    _display_pso(t, qt, fitted_models.get("PSO"))

                with tab2b:
                    _display_rpso(t, qt, fitted_models.get("rPSO"))

                with tab3:
                    _display_elovich(t, qt, fitted_models.get("Elovich"))

                with tab4:
                    _display_ipd(t, qt, fitted_models.get("IPD"))

                with tab5:
                    _display_diffusion_analysis(t, qt, qe_exp, experimental_conditions)

                with tab6:
                    _display_model_comparison(fitted_models, t, qt)

            # Export info
            st.markdown("---")
            st.info(
                "💡 **To download figures and data:** Go to the **📦 Export All** tab for comprehensive exports with format options."
            )

    elif kin_input and input_mode == "absorbance" and not calib_params:
        st.warning(
            "⚠️ Complete calibration first, or switch to **Direct Concentration** input mode in the sidebar"
        )
    elif not kin_input:
        st.info("📥 Enter kinetic data in sidebar")
        _display_guidelines()


# =============================================================================
# CACHED DATA CALCULATION
# =============================================================================


@st.cache_data
def _calculate_kinetic_results(kin_input, calib_params):
    """Calculate kinetic data from absorbance with error propagation."""
    df = kin_input["data"].copy()
    params = kin_input["params"]

    slope = calib_params["slope"]
    intercept = calib_params["intercept"]
    C0 = params["C0"]
    m = params["m"]
    V = params["V"]

    slope_se = calib_params.get("std_err_slope", 0)
    intercept_se = calib_params.get("std_err_intercept", 0)

    results = []
    for _, row in df.iterrows():
        t = row["Time"]
        abs_val = row["Absorbance"]

        Ct = calculate_Ce_from_absorbance(abs_val, slope, intercept)
        qt = calculate_adsorption_capacity(C0, Ct, V, m)
        removal = calculate_removal_percentage(C0, Ct)

        # Returns tuple: (Ct_calc, Ct_se)
        _, Ct_se = propagate_calibration_uncertainty(
            abs_val, slope, intercept, slope_se, intercept_se
        )
        qt_error = (V / m) * Ct_se if m > 0 else 0

        results.append(
            {
                "Time": t,
                "Absorbance": abs_val,
                "Ct_mgL": Ct,
                "Ct_error": Ct_se,
                "qt_mg_g": qt,
                "qt_error": qt_error,
                "removal_%": removal,
            }
        )

    if not results:
        return CalculationResult(success=False, error="No valid data points.")

    results_df = pd.DataFrame(results).sort_values("Time")
    return CalculationResult(success=True, data=results_df)


@st.cache_data
def _calculate_kinetic_results_direct(kin_input):
    """
    Calculate kinetic data from direct Ct input.

    This function bypasses calibration and uses Ct values directly from published data.
    Useful for validating the application with literature datasets.
    """
    df = kin_input["data"].copy()
    params = kin_input["params"]

    C0 = params["C0"]
    m = params["m"]
    V = params["V"]

    results = []
    for _, row in df.iterrows():
        t = row["Time"]
        Ct = row["Ct"]

        # Validate Ct <= C0
        if Ct > C0:
            continue  # Skip invalid data points

        qt = calculate_adsorption_capacity(C0, Ct, V, m)
        removal = calculate_removal_percentage(C0, Ct)

        results.append(
            {
                "Time": t,
                "Absorbance": Ct,  # Store Ct in Absorbance column for compatibility
                "Ct_mgL": Ct,
                "Ct_error": 0.0,  # No calibration uncertainty in direct mode
                "qt_mg_g": qt,
                "qt_error": 0.0,  # No propagated error in direct mode
                "removal_%": removal,
            }
        )

    if not results:
        return CalculationResult(
            success=False, error="No valid data points. Ensure Ct ≤ C0 for all rows."
        )

    results_df = pd.DataFrame(results).sort_values("Time")
    return CalculationResult(success=True, data=results_df)


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================


def _display_pfo(t, qt, results):
    """Display PFO results."""
    st.markdown("**Pseudo-First Order (Lagergren)**")
    st.latex(r"q_t = q_e (1 - e^{-k_1 t})")

    if results and results.get("converged"):
        params = results["params"]
        ci = results.get("ci_95", {})

        display_results_table(
            {
                "Parameter": ["qe (mg/g)", "k₁ (min⁻¹)", "t₁/₂ (min)"],
                "Value": [
                    f"{params['qe']:.4f}",
                    f"{params['k1']:.6f}",
                    f"{params.get('t_half', np.nan):.2f}",
                ],
                "Std. Error": [
                    f"{params.get('qe_se', 0):.4f}",
                    f"{params.get('k1_se', 0):.6f}",
                    "—",
                ],
                "95% CI": [
                    f"({ci.get('qe', (np.nan, np.nan))[0]:.4f}, {ci.get('qe', (np.nan, np.nan))[1]:.4f})",
                    f"({ci.get('k1', (np.nan, np.nan))[0]:.6f}, {ci.get('k1', (np.nan, np.nan))[1]:.6f})",
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

        # Plot
        t_line = np.linspace(0, t.max() * 1.1, 100)
        qt_pred = pfo_model(t_line, params["qe"], params["k1"])

        fig = create_kinetic_plot(
            t, qt, t_line, qt_pred, model_name="PFO", r_squared=results["r_squared"]
        )
        st.plotly_chart(fig, use_container_width=True, key="pfo_plot")
        # Parity plot
        qt_pred_exp = pfo_model(t, params["qe"], params["k1"])
        fig_parity = create_parity_plot(
            qt, qt_pred_exp, model_name="PFO", r_squared=results["r_squared"]
        )
        st.plotly_chart(fig_parity, use_container_width=True, key="pfo_parity")
    else:
        error_msg = results.get("error", "Unknown error") if results else "Fitting failed"
        st.warning(f"PFO model did not converge: {error_msg}")


def _display_pso(t, qt, results):
    """Display PSO results with mechanistic interpretation warning."""
    st.markdown("**Pseudo-Second Order (Ho-McKay)**")
    st.latex(r"q_t = \frac{k_2 q_e^2 t}{1 + k_2 q_e t}")

    if results and results.get("converged"):
        params = results["params"]
        ci = results.get("ci_95", {})

        display_results_table(
            {
                "Parameter": ["qe (mg/g)", "k₂ (g/(mg·min))", "h (mg/(g·min))", "t₁/₂ (min)"],
                "Value": [
                    f"{params['qe']:.4f}",
                    f"{params['k2']:.6f}",
                    f"{params.get('h', np.nan):.4f}",
                    f"{params.get('t_half', np.nan):.2f}",
                ],
                "Std. Error": [
                    f"{params.get('qe_se', 0):.4f}",
                    f"{params.get('k2_se', 0):.6f}",
                    "—",
                    "—",
                ],
                "95% CI": [
                    f"({ci.get('qe', (np.nan, np.nan))[0]:.4f}, {ci.get('qe', (np.nan, np.nan))[1]:.4f})",
                    f"({ci.get('k2', (np.nan, np.nan))[0]:.6f}, {ci.get('k2', (np.nan, np.nan))[1]:.6f})",
                    "—",
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

        # CRITICAL: Mechanistic interpretation warning
        # See: Hubbe et al. (2019). BioResources, 14(3), 7582-7626.
        st.warning("""
⚠️ **Important Note on Mechanistic Interpretation**

A good fit to the pseudo-second-order model does **NOT** necessarily indicate chemisorption.
The PSO equation can be derived from multiple mechanisms (Azizian 2004; Liu & Shen 2008),
and ~90% of kinetic studies report PSO as "best fit" regardless of actual mechanism—a
statistical artifact, not mechanistic evidence.

**To determine the actual rate-limiting step, additional experiments are required:**
- Boyd analysis (film vs. particle diffusion)
- Varying particle size studies
- Temperature dependence (activation energy)
- Mechanistic modeling with physical diffusion parameters

*Reference: Hubbe et al. (2019). BioResources, 14(3), 7582-7626.*
        """)

        # Plot
        t_line = np.linspace(0, t.max() * 1.1, 100)
        qt_pred = pso_model(t_line, params["qe"], params["k2"])

        fig = create_kinetic_plot(
            t, qt, t_line, qt_pred, model_name="PSO", r_squared=results["r_squared"]
        )
        st.plotly_chart(fig, use_container_width=True, key="pso_plot")
    else:
        error_msg = results.get("error", "Unknown error") if results else "Fitting failed"
        st.warning(f"PSO model did not converge: {error_msg}")


def _display_elovich(t, qt, results):
    """Display Elovich results."""
    st.markdown("**Elovich Model (Chemisorption)**")
    st.latex(r"q_t = \frac{1}{\beta} \ln(1 + \alpha \beta t)")

    if results and results.get("converged"):
        params = results["params"]
        ci = results.get("ci_95", {})

        display_results_table(
            {
                "Parameter": ["α (mg/(g·min))", "β (g/mg)"],
                "Value": [f"{params['alpha']:.4f}", f"{params['beta']:.6f}"],
                "Std. Error": [
                    f"{params.get('alpha_se', 0):.4f}",
                    f"{params.get('beta_se', 0):.6f}",
                ],
                "95% CI": [
                    f"({ci.get('alpha', (np.nan, np.nan))[0]:.4f}, {ci.get('alpha', (np.nan, np.nan))[1]:.4f})",
                    f"({ci.get('beta', (np.nan, np.nan))[0]:.6f}, {ci.get('beta', (np.nan, np.nan))[1]:.6f})",
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

        st.info("""
**Elovich model interpretation:** Originally derived for heterogeneous surface chemisorption.
However, like PSO, a good fit alone is insufficient to confirm mechanism. Consider activation
energy values and complementary characterization data for mechanistic conclusions.
        """)

        # Plot
        t_pos = t[t > 0]
        qt_pos = qt[t > 0]
        t_line = np.linspace(0.1, t.max() * 1.1, 100)
        qt_pred = elovich_model(t_line, params["alpha"], params["beta"])

        fig = create_kinetic_plot(
            t_pos, qt_pos, t_line, qt_pred, model_name="Elovich", r_squared=results["r_squared"]
        )
        st.plotly_chart(fig, use_container_width=True, key="elovich_plot")
    else:
        error_msg = results.get("error", "Unknown error") if results else "Fitting failed"
        st.warning(f"Elovich model did not converge: {error_msg}")


def _display_ipd(t, qt, results):
    """Display IPD results."""
    st.markdown("**Intraparticle Diffusion (Weber-Morris)**")
    st.latex(r"q_t = k_{id} \cdot t^{0.5} + C")

    if results and results.get("converged"):
        params = results["params"]

        display_results_table(
            {
                "Parameter": ["kid (mg/(g·min⁰·⁵))", "C (mg/g)"],
                "Value": [f"{params['kid']:.4f}", f"{params['C']:.4f}"],
                "Std. Error": [f"{params.get('kid_se', 0):.4f}", "—"],
                "Interpretation": [
                    "Diffusion rate constant",
                    results.get("mechanism", "Boundary layer thickness"),
                ],
            }
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R²", f"{results['r_squared']:.4f}")
        with col2:
            st.metric("Adj-R²", f"{results['adj_r_squared']:.4f}")
        with col3:
            st.metric("RMSE", f"{results['rmse']:.4f}")

        # Interpretation
        C = params["C"]
        if abs(C) < 1:
            st.success("**C ≈ 0:** Intraparticle diffusion is the sole rate-limiting step")
        else:
            st.info(
                f"**C = {C:.2f}:** Boundary layer effect present. Multiple mechanisms involved."
            )

        # Plot
        sqrt_t = np.sqrt(t)
        sqrt_t_line = np.linspace(0, sqrt_t.max() * 1.1, 100)
        qt_pred = params["kid"] * sqrt_t_line + params["C"]

        fig = go.Figure()

        exp = style_experimental_trace(name="Experimental")
        exp["hovertemplate"] = "t⁰·⁵: %{x:.2f}<br>qt: %{y:.2f}<extra></extra>"
        fig.add_trace(go.Scatter(x=sqrt_t, y=qt, **exp))

        fit = style_fit_trace("IPD", results.get("r_squared"), is_primary=True)
        fit["name"] = "IPD Fit"
        fig.add_trace(go.Scatter(x=sqrt_t_line, y=qt_pred, **fit))

        fig = apply_professional_style(
            fig,
            title=f"IPD Model (R² = {results['r_squared']:.4f})",
            x_title="t⁰·⁵ (min⁰·⁵)",
            y_title="qt (mg/g)",
            height=400,
            show_legend=True,
            legend_position="lower right",
        )
        fig.update_xaxes(rangemode="tozero")
        fig.update_yaxes(rangemode="tozero")

        st.plotly_chart(fig, use_container_width=True, key="ipd_plot")
    else:
        error_msg = results.get("error", "Unknown error") if results else "Fitting failed"
        st.warning(f"IPD model did not converge: {error_msg}")


def _display_rpso(t, qt, results):
    """Display rPSO (Revised PSO) results."""
    st.markdown("**Revised Pseudo-Second Order (rPSO)**")
    st.latex(r"q_t = \frac{q_e^2 k_2 t}{1 + q_e k_2 t} \cdot \phi")

    st.info("""
    **About rPSO:** The revised PSO model (Bullen et al., 2021) includes a concentration
    correction factor (φ) that accounts for the initial adsorbate concentration. This
    reduces the residual sum of squares by ~66% compared to standard PSO in many cases.
    """)

    if results and results.get("converged"):
        params = results["params"]
        ci = results.get("ci_95", {})

        display_results_table(
            {
                "Parameter": [
                    "qe (mg/g)",
                    "k₂ (g/(mg·min))",
                    "φ (correction)",
                    "h (mg/(g·min))",
                    "t₁/₂ (min)",
                ],
                "Value": [
                    f"{params['qe']:.4f}",
                    f"{params['k2']:.6f}",
                    f"{params.get('phi', 1.0):.4f}",
                    f"{params.get('h', np.nan):.4f}",
                    f"{params.get('t_half', np.nan):.2f}",
                ],
                "Std. Error": [
                    f"{params.get('qe_se', 0):.4f}",
                    f"{params.get('k2_se', 0):.6f}",
                    "—",
                    "—",
                    "—",
                ],
                "95% CI": [
                    f"({ci.get('qe', (np.nan, np.nan))[0]:.4f}, {ci.get('qe', (np.nan, np.nan))[1]:.4f})",
                    f"({ci.get('k2', (np.nan, np.nan))[0]:.6f}, {ci.get('k2', (np.nan, np.nan))[1]:.6f})",
                    "—",
                    "—",
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

        # Comparison note
        st.caption("""
        *Reference: Bullen et al. (2021). Langmuir, 37(10), 3189-3201.*
        """)

        # Plot
        t_line = np.linspace(0, t.max() * 1.1, 100)
        qt_pred = pso_model(t_line, params["qe"], params["k2"])

        fig = create_kinetic_plot(
            t, qt, t_line, qt_pred, model_name="rPSO", r_squared=results["r_squared"]
        )
        st.plotly_chart(fig, use_container_width=True, key="rpso_plot")
    else:
        if results:
            error_msg = results.get("error", "Unknown error")
            st.warning(f"rPSO model did not converge: {error_msg}")
        else:
            st.info("""
            **rPSO requires experimental conditions (C₀, m, V).**

            Make sure you've entered:
            - Initial concentration (C₀, mg/L)
            - Adsorbent mass (m, g)
            - Solution volume (V, L)

            These are entered in the sidebar data input section.
            """)


def _display_diffusion_analysis(t, qt, qe_exp, experimental_conditions):
    """Display diffusion-based mechanism analysis."""
    st.markdown("**🔬 Diffusion Mechanism Analysis**")

    st.info("""
    **Diffusion Models:** These models help identify the rate-limiting step in adsorption:
    - **Film diffusion**: External mass transfer through boundary layer
    - **Pore diffusion (HSDM)**: Intraparticle diffusion within pores
    - **Biot number**: Indicates which mechanism dominates (Bi >> 1 = pore, Bi << 1 = film)
    """)

    # User inputs for diffusion analysis
    st.markdown("#### Particle Parameters")
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        particle_radius = st.number_input(
            "Particle radius (cm)",
            min_value=0.0001,
            max_value=1.0,
            value=0.05,
            step=0.01,
            help="Average radius of adsorbent particles (cm)",
            key="diffusion_particle_radius",
        )

    with col2:
        run_diffusion = st.button(
            "🔬 Analyze Diffusion Mechanism",
            help="Run automated rate-limiting step identification",
            key="diffusion_analyze_btn",
        )

    with col3:
        clear_diffusion = st.button(
            "🗑️ Clear", help="Clear diffusion analysis results", key="diffusion_clear_btn"
        )

    # Clear results if requested
    if clear_diffusion:
        if "diffusion_results" in st.session_state:
            del st.session_state["diffusion_results"]
        st.rerun()

    # Run analysis and store in session state
    if run_diffusion:
        with st.spinner("Analyzing diffusion mechanism..."):
            try:
                results = identify_rate_limiting_step(t, qt, qe_exp, particle_radius)
                st.session_state["diffusion_results"] = {
                    "results": results,
                    "particle_radius": particle_radius,
                    "t": t,
                    "qt": qt,
                }
            except Exception as e:
                st.error(f"Error in diffusion analysis: {str(e)}")
                st.info("Make sure you have sufficient data points (≥5) and valid qe value.")
                return

    # Display results from session state
    if "diffusion_results" in st.session_state:
        stored = st.session_state["diffusion_results"]
        results = stored["results"]
        particle_radius = stored["particle_radius"]
        # Use stored data for consistency
        t_stored = stored["t"]
        qt_stored = stored["qt"]

        st.success(f"**Mechanism Identified:** {results['mechanism_suggestion']}")
        st.info(f"**Confidence:** {results['confidence']}")

        # Display Weber-Morris analysis
        st.markdown("#### Weber-Morris (IPD) Analysis")
        wm = results["weber_morris"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("kid (mg/g·min⁰·⁵)", f"{wm['kid']:.4f}")
        with col2:
            st.metric("Intercept (C)", f"{wm['intercept']:.4f}")
        with col3:
            st.metric("R²", f"{wm['r_squared']:.4f}")

        if wm["passes_through_origin"]:
            st.success("✅ Line passes through origin → IPD is sole rate-controlling step")
        else:
            st.warning("⚠️ Line does NOT pass through origin → Boundary layer effect present")

        # Weber-Morris plot (use stored data)
        sqrt_t = np.sqrt(t_stored)
        sqrt_t_line = np.linspace(0, sqrt_t.max() * 1.1, 100)
        qt_pred = wm["kid"] * sqrt_t_line + wm["intercept"]

        fig_wm = go.Figure()
        fig_wm.add_trace(
            go.Scatter(
                x=sqrt_t,
                y=qt_stored,
                mode="markers",
                name="Experimental",
                marker=MARKERS["experimental"],
            )
        )
        fig_wm.add_trace(
            go.Scatter(
                x=sqrt_t_line,
                y=qt_pred,
                mode="lines",
                name="Linear Fit",
                line={"color": COLORS["fit_tertiary"], "width": 2.5},
            )
        )
        # Add origin line for reference
        fig_wm.add_trace(
            go.Scatter(
                x=sqrt_t_line,
                y=wm["kid"] * sqrt_t_line,
                mode="lines",
                name="Through Origin",
                line={"color": COLORS["residual_line"], "width": 1.5, "dash": "dash"},
            )
        )
        fig_wm = apply_professional_style(
            fig_wm,
            title="Weber-Morris Plot (IPD Analysis)",
            x_title="t<sup>0.5</sup> (min<sup>0.5</sup>)",
            y_title="q<sub>t</sub> (mg/g)",
            height=400,
            show_legend=True,
            legend_horizontal=True,
        )
        fig_wm.update_xaxes(rangemode="tozero")
        fig_wm.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_wm, use_container_width=True, key="weber_morris_plot")

        # Boyd plot analysis
        st.markdown("#### Boyd Plot Analysis")
        boyd = results["boyd_plot"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Slope", f"{boyd['slope']:.4f}")
        with col2:
            st.metric("Intercept", f"{boyd['intercept']:.4f}")
        with col3:
            st.metric("R²", f"{boyd['r_squared']:.4f}")

        if boyd["passes_through_origin"]:
            st.success("✅ Linear through origin → Pore diffusion rate-limiting")
        else:
            st.warning("⚠️ Non-zero intercept → Film diffusion contributes")

        # Boyd plot (use stored data)
        Bt = boyd["Bt"]
        valid_Bt = ~np.isnan(Bt) & ~np.isinf(Bt)
        t_valid = t_stored[valid_Bt]
        Bt_valid = Bt[valid_Bt]

        if len(t_valid) > 2:
            t_line = np.linspace(0, t_valid.max() * 1.1, 100)
            Bt_pred = boyd["slope"] * t_line + boyd["intercept"]

            fig_boyd = go.Figure()
            fig_boyd.add_trace(
                go.Scatter(
                    x=t_valid,
                    y=Bt_valid,
                    mode="markers",
                    name="Bt values",
                    marker=MARKERS["experimental"],
                )
            )
            fig_boyd.add_trace(
                go.Scatter(
                    x=t_line,
                    y=Bt_pred,
                    mode="lines",
                    name="Linear Fit",
                    line={"color": COLORS["fit_secondary"], "width": 2.5},
                )
            )
            # Origin reference line
            fig_boyd.add_trace(
                go.Scatter(
                    x=t_line,
                    y=boyd["slope"] * t_line,
                    mode="lines",
                    name="Through Origin",
                    line={"color": COLORS["residual_line"], "width": 1.5, "dash": "dash"},
                )
            )
            fig_boyd = apply_professional_style(
                fig_boyd,
                title="Boyd Plot (Film/Pore Diffusion)",
                x_title="Time (min)",
                y_title="Bt = -ln(1 - F)",
                height=400,
                show_legend=True,
                legend_horizontal=True,
            )
            fig_boyd.update_xaxes(rangemode="tozero")
            fig_boyd.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig_boyd, use_container_width=True, key="boyd_plot")

        # Biot number section
        st.markdown("#### Biot Number Analysis")
        st.markdown("""
        The Biot number (Bi) indicates the relative importance of film vs. pore diffusion:

        | Bi Value | Rate-Limiting Step |
        |----------|-------------------|
        | Bi < 0.1 | Film diffusion dominates (>90% resistance) |
        | 0.1 < Bi < 100 | Mixed control |
        | Bi > 100 | Pore diffusion dominates (>90% resistance) |
        """)

        # Allow user to input estimated kf and Dp for Biot calculation
        st.markdown("##### Optional: Calculate Biot Number")
        col1, col2 = st.columns(2)
        with col1:
            kf_input = st.number_input(
                "Film mass transfer coef. kf (cm/min)",
                min_value=0.0,
                value=0.01,
                step=0.001,
                format="%.4f",
                help="External mass transfer coefficient (estimate from initial rate)",
                key="biot_kf",
            )
        with col2:
            Dp_input = st.number_input(
                "Pore diffusivity Dp (cm²/min)",
                min_value=0.0,
                value=1e-6,
                step=1e-7,
                format="%.2e",
                help="Effective pore diffusion coefficient",
                key="biot_dp",
            )

        if kf_input > 0 and Dp_input > 0:
            Bi = calculate_biot_number(kf_input, Dp_input, particle_radius)
            st.metric("Biot Number", f"{Bi:.2f}")

            if Bi < 0.1:
                st.warning("**Bi < 0.1:** Film diffusion is the rate-limiting step")
            elif Bi > 100:
                st.success("**Bi > 100:** Pore diffusion is the rate-limiting step")
            else:
                st.info("**0.1 < Bi < 100:** Mixed control (both mechanisms contribute)")

    else:
        st.caption("Click 'Analyze Diffusion Mechanism' to run the analysis.")

        # Show guidance
        with st.expander("📖 Diffusion Analysis Guide", expanded=False):
            st.markdown("""
            **Experimental Requirements for Proper Diffusion Analysis:**

            1. **Sufficient early-time data**: Include points at 0, 5, 10, 15, 20, 30 min
            2. **Particle size**: Know/measure your adsorbent particle radius
            3. **For definitive conclusions**, vary:
               - Stirring speed (film diffusion sensitive)
               - Particle size (pore diffusion sensitive)
               - Initial concentration (both sensitive)

            **Interpretation Guide:**

            | Diagnostic | Result | Indicates |
            |------------|--------|-----------|
            | Weber-Morris | C ≈ 0 | IPD sole rate-limiter |
            | Weber-Morris | C > 0 | Boundary layer effect |
            | Boyd plot | Linear through origin | Pore diffusion |
            | Boyd plot | Non-zero intercept | Film diffusion |

            *References: Boyd et al. (1947) J Am Chem Soc 69:2836; Crank (1975) Mathematics of Diffusion*
            """)


def _display_model_comparison(fitted_models, t, qt):
    """Display kinetic model comparison with standard error functions."""
    st.markdown("**📊 Model Comparison**")

    comparison_data = []
    for name, results in fitted_models.items():
        if results and results.get("converged"):
            comparison_data.append(
                {
                    "Model": name,
                    "R²": results["r_squared"],
                    "Adj-R²": results["adj_r_squared"],
                    "RMSE": results["rmse"],
                    "χ²": results.get("chi_squared", np.nan),
                    "AIC": results.get("aicc", results["aic"]),
                }
            )

    if not comparison_data:
        st.warning("No models converged successfully.")
        return

    comparison_df = pd.DataFrame(comparison_data)
    aicc_values = [fitted_models[row["Model"]].get("aicc", row["AIC"]) for row in comparison_data]
    aic_weights = calculate_akaike_weights(aicc_values)
    comparison_df["AIC Weight"] = aic_weights
    comparison_df = comparison_df.sort_values("Adj-R²", ascending=False)

    # Error function explanation
    with st.expander("📖 Error Function Definitions", expanded=False):
        st.markdown("""
        | Error Function | Best For | Lower = Better |
        |----------------|----------|----------------|
        | **R²/Adj-R²** | Overall fit quality | Higher = Better |
        | **RMSE** | Absolute error magnitude | ✓ |
        | **χ²** | Weighted fit quality | ✓ |
        | **AIC** | Model selection (penalizes complexity) | ✓ |

        *Reference: Kumar et al. (2008) J Hazard Mater 151:794-804*
        """)

    st.dataframe(
        comparison_df.style.format(
            {
                "R²": "{:.4f}",
                "Adj-R²": "{:.4f}",
                "RMSE": "{:.4f}",
                "χ²": "{:.2f}",
                "AIC": "{:.2f}",
                "AIC Weight": "{:.1%}",
            }
        )
        .highlight_max(subset=["R²", "Adj-R²", "AIC Weight"], color="lightgreen")
        .highlight_min(subset=["RMSE", "AIC", "χ²"], color="lightblue"),
        use_container_width=True,
        hide_index=True,
    )

    # Best model recommendation - handle NaN values safely
    try:
        adj_r2_idx = comparison_df["Adj-R²"].idxmax()
        best_adj_r2 = comparison_df.loc[adj_r2_idx, "Model"] if pd.notna(adj_r2_idx) else None
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

    # Note on mechanistic interpretation
    st.caption("""
    ⚠️ **Note:** Best statistical fit ≠ mechanistic evidence. Model selection identifies the best
    *mathematical description* of your data, not the underlying physical mechanism. Additional
    experimental evidence (activation energy, particle size effects, etc.) is required for
    mechanistic conclusions. See Hubbe et al. (2019), BioResources 14(3):7582-7626.
    """)

    # All models plot
    st.markdown("**📈 All Models Overlay**")

    model_functions = {
        "PFO": lambda t, p: pfo_model(t, p["qe"], p["k1"]),
        "PSO": lambda t, p: pso_model(t, p["qe"], p["k2"]),
        "Elovich": lambda t, p: elovich_model(t, p["alpha"], p["beta"]),
        # Note: IPD excluded due to √t x-axis transformation
    }

    fig = create_model_comparison_plot(
        t,
        qt,
        fitted_models,
        model_functions,
        x_label="Time (min)",
        y_label="q<sub>t</sub> (mg/g)",
        title="Kinetic Model Comparison",
    )
    st.plotly_chart(fig, use_container_width=True, key="kinetic_comparison_plot")


def _display_guidelines():
    """Display kinetic guidelines."""
    with st.expander("📖 Kinetic Analysis Guidelines", expanded=True):
        st.markdown("""
        **standards:**

        1. **Time Points:** Include early points (0, 5, 10, 15, 20, 30 min)
        2. **Equilibrium:** Continue until plateau (60-240 min typical)
        3. **Initial Point:** Always include t = 0
        4. **Report:**
           - qe, k values with 95% CI
           - R², Adj-R², RMSE, AIC
           - Initial rate (h) for PSO
           - Half-time (t₁/₂)

        **⚠️ Mechanistic Interpretation (Critical):**

        Best model fit does NOT confirm mechanism. ~90% of studies report PSO as
        "best" regardless of actual mechanism (Hubbe et al. 2019).

        **To support mechanistic claims, you need:**
        - Boyd plot analysis (film vs. pore diffusion)
        - Activation energy from temperature studies (Ea < 40 kJ/mol = physisorption)
        - Particle size variation experiments
        - Spectroscopic evidence (FTIR, XPS before/after)

        **Diffusion Analysis (Programmatic API):**

        For mechanistic analysis, use `models.py` functions:
        - `identify_rate_limiting_step()` - Automated mechanism ID

        *References: Hubbe et al. (2019) BioResources 14(3):7582; Azizian (2004) JCIS 276:47*
        """)
