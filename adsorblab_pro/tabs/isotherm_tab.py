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

from ..config import BOOTSTRAP_DEFAULT_ITERATIONS, BOOTSTRAP_DEFAULT_SEED
from ..models import (
    fit_model_with_ci,
    freundlich_model,
    isotherm_fit_setup,
    langmuir_model,
    round_significant,
    sips_model,
    temkin_curve,
    temkin_model,
)

# Professional plot styling
from ..plot_style import (
    apply_professional_style,
    create_dual_axis_effect_plot,
    create_isotherm_plot,
    create_model_comparison_plot,
    create_parity_plot,
    style_experimental_trace,
)
from ..utils import (
    EPSILON_DIV,
    assess_data_quality,
    BOOTSTRAP_UNAVAILABLE,
    bootstrap_parameter_intervals,
    display_bootstrap_intervals,
    report_bootstrap_outcome,
    store_bootstrap_result as _store_bootstrap,
    build_uptake_table,
    calculate_separation_factor,
    FIT_ERROR_MODEL,
    display_fit_diagnostics,
    display_results_table,
    format_criterion,
    model_comparison_table,
    parameter_status_column,
    eligible_observations,
    get_current_study_state,
    metadata_columns,
    interpret_separation_factor,
    observation_notice,
    uptake_calculation_result,
    uptake_results_frame,
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
    # Significant digits, not decimals: rounding to 8 decimals turned
    # concentrations below 1e-8 mg/L into zero before fitting.
    return (
        tuple(round_significant(Ce).tolist()),
        tuple(round_significant(qe).tolist()),
        tuple(round_significant(C0).tolist()),
    )


def _run_isotherm_bootstrap(
    Ce, qe, fitted_models, n_bootstrap, confidence_level, T_K: float = 298.15
):
    """
    Run bootstrap CI on already fitted isotherm models with visual progress.

    This function adds bootstrap confidence intervals to models that have
    already been fitted, showing a progress bar for user feedback.
    """

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
    summaries: list[tuple[str | bool, str]] = []

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

        # Resample exactly the observations the fit used (incl. model domain).
        Ce_v = np.asarray(fitted_models[model_name].get("x_data", Ce), dtype=float)
        qe_v = np.asarray(fitted_models[model_name].get("y_data", qe), dtype=float)

        try:
            # Every requested draw is refitted once with the full fit's start/limit
            # policy; counts, seed and failures are stored with the result.
            details = bootstrap_parameter_intervals(
                model_func,
                Ce_v,
                qe_v,
                params,
                n_bootstrap,
                confidence_level,
                fit_setup=_isotherm_fold_setup(model_name),
                seed=BOOTSTRAP_DEFAULT_SEED,
                param_names=param_names,
                progress_callback=model_progress,
            )
            summaries.append(_store_bootstrap(fitted_models[model_name], details, model_name))
        except Exception as e:
            fitted_models[model_name].pop("bootstrap", None)
            summaries.append((BOOTSTRAP_UNAVAILABLE, f"{model_name}: bootstrap failed ({e})"))

    # Clean up
    progress_bar.progress(1.0)
    status_text.text("Bootstrap runs finished.")

    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    report_bootstrap_outcome(summaries)
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

    # Observations reaching this function are the eligible rows of the results
    # table.  Zero values are valid observations (Ce = 0: complete removal;
    # qe = 0: no uptake) and are kept wherever a model's domain permits them.
    valid = np.isfinite(Ce) & np.isfinite(qe) & np.isfinite(C0) & (Ce >= 0)
    Ce_v = Ce[valid]
    qe_v = qe[valid]
    C0_v = C0[valid]

    if len(Ce_v) < 4:
        return fitted

    # Temkin's ln(KT·Ce) is undefined at Ce = 0; only that model excludes such rows.
    temkin_domain = Ce_v > 0
    # Starting values and limits scale with the data (no fixed ceilings such as
    # KL ≤ 100 L/mg); see isotherm_fit_setup for the physical constraints kept.
    setup = isotherm_fit_setup(Ce_v, qe_v)
    temkin_setup = isotherm_fit_setup(Ce_v[temkin_domain], qe_v[temkin_domain])["Temkin"]

    # =========================================================================
    # FIT EACH MODEL
    # =========================================================================

    # Langmuir
    try:
        result = fit_model_with_ci(
            langmuir_model,
            Ce_v,
            qe_v,
            p0=setup["Langmuir"]["p0"],
            bounds=setup["Langmuir"]["bounds"],
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
            p0=setup["Freundlich"]["p0"],
            bounds=setup["Freundlich"]["bounds"],
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
            Ce_v[temkin_domain],
            qe_v[temkin_domain],
            p0=temkin_setup["p0"],
            bounds=temkin_setup["bounds"],
            param_names=["B1", "KT"],
            confidence=confidence_level,
        )
        if result and result.get("converged"):
            # KT ≥ 1/min(Ce) is a fit limit (qe ≥ 0 at every observation); a fit
            # stopping there is reported through param_status/bounds_hit.
            n_outside = int((~temkin_domain).sum())
            if n_outside:
                result["domain_note"] = (
                    f"{n_outside} observation(s) with Ce = 0 were not used: ln(KT·Ce) is "
                    "undefined at Ce = 0."
                )
            fitted["Temkin"] = result
    except Exception as e:
        fitted["Temkin"] = {"converged": False, "error": str(e)}

    # Sips
    try:
        result = fit_model_with_ci(
            sips_model,
            Ce_v,
            qe_v,
            p0=setup["Sips"]["p0"],
            bounds=setup["Sips"]["bounds"],
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


def fit_isotherm_models_with_cache(
    Ce, qe, C0, confidence_level, current_study_state, T_K: float = 298.15
):
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
    fitted_models = _fit_all_isotherm_models_cached(
        Ce_tuple, qe_tuple, C0_tuple, confidence_level, T_K
    )

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
    """Render isotherm analysis with professional statistics."""
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
            st.metric(
                "Data checks",
                f"{quality['quality_score']}/100",
                help="Heuristic points for the number of rows, outliers and negative values; not a statistical confidence and not a measure of fit quality.",
            )
        with col2:
            st.metric("Points", len(iso_input["data"]))
        with col3:
            status = "✅ No major flags" if quality["quality_score"] >= 70 else "⚠️ Review"
            st.metric("Data flags", status)

        # Calculate isotherm data based on input mode
        if input_mode == "direct":
            iso_results_obj = _calculate_isotherm_results_direct(iso_input)
            st.caption("📈 *Using direct concentration input (Ce values)*")
        else:
            iso_results_obj = _calculate_isotherm_results(iso_input, calib_params)

        if not iso_results_obj.success:
            st.warning(f"Could not process isotherm data: {iso_results_obj.error}")
            if iso_results_obj.data is not None and not iso_results_obj.data.empty:
                display_results_table(iso_results_obj.data.round(4), hide_index=True)
            current_study_state["isotherm_results"] = None
            return

        iso_results = iso_results_obj.data
        if iso_results is not None and not iso_results.empty:
            # The stored/exported table keeps every row with its status and reason.
            current_study_state["isotherm_results"] = iso_results
            usable = eligible_observations(iso_results)

            # Data Display Section
            st.markdown("---")
            st.markdown("### 📊 Equilibrium Data")

            col1, col2 = st.columns(2)
            with col1:
                st.latex(r"q_e = \frac{(C_0 - C_e) \cdot V}{m}")
            with col2:
                st.latex(r"\% \text{ Removal} = \frac{(C_0 - C_e)}{C_0} \times 100")

            notice = observation_notice(iso_results)
            if notice:
                st.warning(f"⚠️ {notice}")
            display_cols = [
                c
                for c in [
                    "source_row",
                    "C0_mgL",
                    "Absorbance",
                    "Ce_mgL",
                    "qe_mg_g",
                    "removal_%",
                    "status",
                    "note",
                ]
                if c in iso_results.columns
            ] + metadata_columns(iso_results)
            display_results_table(iso_results[display_cols].round(4), hide_index=True)

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
            st.caption(
                "Note: model fitting below always uses qₑ (mg/g). Unit selection affects only the overview plots/tables."
            )

            # --- Plot selection ---
            if unit_system == "mg/g":
                # qe plot
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=usable["Ce_mgL"],
                        y=usable["qe_mg_g"],
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
                        x=usable["Ce_mgL"],
                        y=usable["removal_%"],
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
                fig = create_dual_axis_effect_plot(
                    x=usable["Ce_mgL"],
                    y1=usable["qe_mg_g"],
                    y2=usable["removal_%"],
                    title="Isotherm Curve",
                    x_title="Concentration Ce (mg/L)",
                    y1_title="qe (mg/g)",
                    y2_title="Removal (%)",
                    y1_name="qe (mg/g)",
                    y2_name="Removal (%)",
                    height=500,
                    x_tozero=True,
                    y1_tozero=True,
                    y2_tozero=True,
                )
                st.plotly_chart(fig, use_container_width=True, key="iso_overview_chart")

            # Model Fitting Section
            st.markdown("---")
            st.markdown("### 🔬 Model Fitting")

            # Only usable observations enter validation and fitting; excluded rows
            # are reported above and kept in the stored table.
            Ce = usable["Ce_mgL"].to_numpy(dtype=float)
            qe = usable["qe_mg_g"].to_numpy(dtype=float)
            C0 = usable["C0_mgL"].to_numpy(dtype=float)
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
                        from ..utils import calculate_press_details

                        model_funcs = {
                            "Langmuir": langmuir_model,
                            "Freundlich": freundlich_model,
                            "Temkin": temkin_model,
                            "Sips": sips_model,
                        }

                        unavailable = []
                        for model_name, func in model_funcs.items():
                            if fitted_models.get(model_name, {}).get("converged"):
                                # Each fold is refitted with the same start/limit
                                # policy as the full fit, on that fit's observations.
                                details = calculate_press_details(
                                    func,
                                    np.asarray(fitted_models[model_name]["x_data"]),
                                    np.asarray(fitted_models[model_name]["y_data"]),
                                    fit_setup=_isotherm_fold_setup(model_name),
                                )
                                fitted_models[model_name]["press"] = details["press"]
                                fitted_models[model_name]["q2"] = details["q2"]
                                fitted_models[model_name]["press_details"] = details
                                if details["status"] != "complete":
                                    unavailable.append(f"{model_name}: {details['message']}")

                        current_study_state["isotherm_models_fitted"] = fitted_models
                    if unavailable:
                        st.warning("⚠️ " + " ".join(unavailable))
                    else:
                        st.success("✅ PRESS/Q² calculated (every leave-one-out refit succeeded).")
                else:
                    # Remove PRESS/Q² values when checkbox is unchecked
                    for model_name in fitted_models:
                        if fitted_models.get(model_name) and fitted_models[model_name].get(
                            "converged"
                        ):
                            fitted_models[model_name].pop("press", None)
                            fitted_models[model_name].pop("q2", None)
                            fitted_models[model_name].pop("press_details", None)
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


def _isotherm_fold_setup(model_name: str):
    """Start/limit policy of the full isotherm fit, applied to a fold's training data."""

    def setup(x_train, y_train):
        config = isotherm_fit_setup(x_train, y_train)[model_name]
        return config["p0"], config["bounds"]

    return setup


# =============================================================================
# CACHED DATA CALCULATION
# =============================================================================


@st.cache_data
def _calculate_isotherm_results(iso_input, calib_params):
    """Calculate isotherm equilibrium data (absorbance mode) with per-row status.

    Every input row is kept with its source row, status and reason; rows below the
    calibration intercept/LOD are unresolved (never an exact zero), and Ce > C0 is
    excluded with its reason instead of being clipped.
    """
    params = iso_input["params"]
    table = build_uptake_table(
        iso_input["data"],
        mode="absorbance",
        signal_col="Absorbance",
        C0="Concentration",
        V=params["V"],
        m=params["m"],
        calib_params=calib_params,
        row_notes=iso_input.get("row_issues"),
    )
    frame = uptake_results_frame(table, include_c0=True, include_signal=True, sort_by="C0_mgL")
    return uptake_calculation_result(frame)


@st.cache_data
def _calculate_isotherm_results_direct(iso_input):
    """
    Calculate isotherm equilibrium data from direct C0/Ce input with per-row status.

    This function bypasses calibration and uses Ce values directly from published data.
    Rows with Ce > C0, negative or missing values are kept and marked as excluded with
    the reason; they are never skipped silently.
    """
    params = iso_input["params"]
    table = build_uptake_table(
        iso_input["data"],
        mode="direct",
        signal_col="Ce",
        C0="C0",
        V=params["V"],
        m=params["m"],
        row_notes=iso_input.get("row_issues"),
    )
    frame = uptake_results_frame(table, include_c0=True, sort_by="C0_mgL")
    return uptake_calculation_result(frame)


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
                "Status": parameter_status_column(results, ["qm", "KL"]),
            }
        )
        display_fit_diagnostics(results)
        display_bootstrap_intervals(results)

        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R²", f"{results['r_squared']:.4f}")
        with col2:
            st.metric("Adj-R²", f"{results['adj_r_squared']:.4f}")
        with col3:
            st.metric("RMSE", f"{results['rmse']:.4f}")
        with col4:
            st.metric("AICc", format_criterion(results.get("aicc")))

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
                valid_mask = np.isfinite(np.atleast_1d(Ce)) & (np.atleast_1d(Ce) >= 0)
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

        # Diagnostics
        qe_pred_exp = langmuir_model(Ce, params["qm"], params["KL"])
        with st.expander("Langmuir diagnostics", expanded=False):
            fig_parity = create_parity_plot(
                y_obs=np.asarray(qe, dtype=float),
                y_pred=np.asarray(qe_pred_exp, dtype=float),
                model_name="Langmuir",
                r_squared=results.get("r_squared"),
                rmse=results.get("rmse"),
                height=420,
            )
            st.plotly_chart(fig_parity, use_container_width=True, key="langmuir_parity")
            st.caption(
                "Points close to the 1:1 line indicate good agreement between observed and predicted values."
            )
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
                "Status": parameter_status_column(results, ["KF", "n_inv", None]),
            }
        )
        display_fit_diagnostics(results)
        display_bootstrap_intervals(results)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R²", f"{results['r_squared']:.4f}")
        with col2:
            st.metric("Adj-R²", f"{results['adj_r_squared']:.4f}")
        with col3:
            st.metric("RMSE", f"{results['rmse']:.4f}")
        with col4:
            st.metric("AICc", format_criterion(results.get("aicc")))

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

        # Diagnostics
        qe_pred_exp = freundlich_model(Ce, params["KF"], params["n_inv"])
        with st.expander("Freundlich diagnostics", expanded=False):
            fig_parity = create_parity_plot(
                y_obs=np.asarray(qe, dtype=float),
                y_pred=np.asarray(qe_pred_exp, dtype=float),
                model_name="Freundlich",
                r_squared=results.get("r_squared"),
                rmse=results.get("rmse"),
                height=420,
            )
            st.plotly_chart(fig_parity, use_container_width=True, key="freundlich_parity")
            st.caption(
                "Points close to the 1:1 line indicate good agreement between observed and predicted values."
            )
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
                "Status": parameter_status_column(results, ["B1", "KT"]),
            }
        )
        display_fit_diagnostics(
            results,
            {
                "KT": "KT's lower limit is 1/min(Ce), the smallest value for which the Temkin "
                "equation gives qe ≥ 0 at every observation; the low-concentration data are "
                "not described by this model."
            },
        )
        display_bootstrap_intervals(results)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R²", f"{results['r_squared']:.4f}")
        with col2:
            st.metric("Adj-R²", f"{results['adj_r_squared']:.4f}")
        with col3:
            st.metric("RMSE", f"{results['rmse']:.4f}")
        with col4:
            st.metric("AICc", format_criterion(results.get("aicc")))

        # Plot over the range where the equation is defined (KT·Ce ≥ 1, qe ≥ 0)
        Ce_line = np.linspace(1.0 / params["KT"], Ce.max() * 1.1, 100)
        qe_pred = temkin_curve(Ce_line, params["B1"], params["KT"])
        if results.get("domain_note"):
            st.info(f"ℹ️ {results['domain_note']}")

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

        # Diagnostics (on the observations Temkin was fitted to)
        qe_fit = np.asarray(results.get("y_data", qe), dtype=float)
        qe_pred_exp = temkin_curve(
            np.asarray(results.get("x_data", Ce), dtype=float), params["B1"], params["KT"]
        )
        with st.expander("Temkin diagnostics", expanded=False):
            fig_parity = create_parity_plot(
                y_obs=qe_fit,
                y_pred=np.asarray(qe_pred_exp, dtype=float),
                model_name="Temkin",
                r_squared=results.get("r_squared"),
                rmse=results.get("rmse"),
                height=420,
            )
            st.plotly_chart(fig_parity, use_container_width=True, key="temkin_parity")
            st.caption(
                "Points close to the 1:1 line indicate good agreement between observed and predicted values."
            )
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
                "Status": parameter_status_column(results, ["qm", "Ks", "ns"]),
            }
        )
        display_fit_diagnostics(results)
        display_bootstrap_intervals(results)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R²", f"{results['r_squared']:.4f}")
        with col2:
            st.metric("Adj-R²", f"{results['adj_r_squared']:.4f}")
        with col3:
            st.metric("RMSE", f"{results['rmse']:.4f}")
        with col4:
            st.metric("AICc", format_criterion(results.get("aicc")))

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

        # Diagnostics
        qe_pred_exp = sips_model(Ce, params["qm"], params["Ks"], params["ns"])
        with st.expander("Sips diagnostics", expanded=False):
            fig_parity = create_parity_plot(
                y_obs=np.asarray(qe, dtype=float),
                y_pred=np.asarray(qe_pred_exp, dtype=float),
                model_name="Sips",
                r_squared=results.get("r_squared"),
                rmse=results.get("rmse"),
                height=420,
            )
            st.plotly_chart(fig_parity, use_container_width=True, key="sips_parity")
            st.caption(
                "Points close to the 1:1 line indicate good agreement between observed and predicted values."
            )
    else:
        error_msg = results.get("error", "Unknown error") if results else "Fitting failed"
        st.warning(f"Sips model did not converge: {error_msg}")


def _display_model_comparison(fitted_models, Ce, qe, T_K: float = 298.15):
    """Display model comparison; criteria are ranked only among comparable fits."""
    st.markdown("**📊 Model Comparison**")

    # Check if PRESS was calculated
    has_press = any(
        results.get("press") is not None for results in fitted_models.values() if results
    )

    comparison_df, comparison = model_comparison_table(fitted_models, include_press=has_press)
    if comparison_df.empty:
        st.warning("No models converged successfully.")
        return

    # Error function explanation expander
    with st.expander("📖 Error Function Definitions", expanded=False):
        definitions = """
        | Error Function | Formula | Best For |
        |----------------|---------|----------|
        | **R²** | 1 - SSE/SST | Overall fit quality (descriptive) |
        | **Adj-R²** | Penalizes extra parameters | Descriptive comparison |
        | **RMSE** | √(SSE/n) | Absolute error magnitude |
        | **Relative SSE** | Σ(residual²/|predicted|) | Descriptive relative error |
        | **AIC / AICc / BIC** | Gaussian likelihood, k = p + 1 | Model selection within the same observations |
        """
        if has_press:
            definitions += """| **PRESS** | Leave-one-out CV error | Predictive ability |
        | **Q²** | 1 - PRESS/SS_tot | Predictive R² |
        """
        definitions += """
        AICc is undefined when n ≤ k + 1 and is then shown as —. ΔAICc and AICc weights
        are computed only among fits to the same observations (same *Set*).

        *References: Kumar et al. (2008) J Hazard Mater 151:794-804; Foo & Hameed (2010) Chem Eng J 156:2-10*
        """
        st.markdown(definitions)

    formats = {
        "R²": "{:.4f}",
        "Adj-R²": "{:.4f}",
        "RMSE": "{:.4f}",
        "Relative SSE": "{:.2f}",
        "AIC": "{:.2f}",
        "AICc": "{:.2f}",
        "BIC": "{:.2f}",
        "ΔAICc": "{:.2f}",
        "AICc weight": "{:.1%}",
    }
    if has_press:
        formats.update({"PRESS": "{:.4f}", "Q²": "{:.4f}"})
    st.dataframe(
        comparison_df.style.format(formats, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )

    if comparison["status"] == "ranked":
        st.success(f"**🎯 {comparison['message']}**")
    else:
        st.info(f"**📊 {comparison['message']}**")
    st.caption(
        f"Fits: {FIT_ERROR_MODEL}. A lower AICc indicates a better trade-off between fit "
        "and complexity for these observations; it does not identify a mechanism."
    )

    # All models plot
    st.markdown("**📈 All Models Overlay**")

    model_functions = {
        "Langmuir": lambda x, p: langmuir_model(x, p["qm"], p["KL"]),
        "Freundlich": lambda x, p: freundlich_model(x, p["KF"], p["n_inv"]),
        "Temkin": lambda x, p: temkin_curve(x, p["B1"], p["KT"]),
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
   - R-squared, Adj-R-squared, RMSE, AIC/AICc/BIC for each model
   - AICc weights among fits to the same observations
   - Separation factor (RL) for Langmuir
        """)
