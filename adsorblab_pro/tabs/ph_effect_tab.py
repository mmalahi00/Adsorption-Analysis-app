# tabs/ph_effect_tab.py
"""
pH Effect Tab - AdsorbLab Pro
=============================

Analyzes the effect of solution pH on adsorption capacity.

Features:
- pH-dependent adsorption visualization
- Optimal pH determination
- Surface charge interpretation
"""

import pandas as pd
import plotly.graph_objects as go

from adsorblab_pro.streamlit_compat import st

from ..plot_style import create_effect_plot, create_dual_axis_effect_plot
from ..utils import (
    CalculationResult,
    assess_data_quality,
    calculate_adsorption_capacity,
    calculate_Ce_from_absorbance,
    calculate_removal_percentage,
    display_results_table,
    get_current_study_state,
    propagate_calibration_uncertainty,
    validate_required_params,
)


def render():
    st.subheader("🧪 pH Effect Study")

    current_study_state = get_current_study_state()
    if not current_study_state:
        st.info("Please add or select a study from the sidebar to begin analysis.")
        return

    unit_system = current_study_state.get("unit_system", "mg/g")
    ph_input = current_study_state.get("ph_effect_input")
    calib_params = current_study_state.get("calibration_params")

    # Check input mode (default to 'absorbance' for backward compatibility)
    input_mode = ph_input.get("input_mode", "absorbance") if ph_input else "absorbance"

    # Determine if we can proceed based on input mode
    can_proceed = False
    if ph_input:
        if input_mode == "direct":
            can_proceed = True
        elif calib_params:
            can_proceed = True

    if can_proceed:
        is_valid, error_message = validate_required_params(
            params=ph_input["params"], required_keys=[("C0", "C₀"), ("m", "Mass"), ("V", "Volume")]
        )
        if not is_valid:
            st.warning(error_message, icon="⚠️")
            return

        quality = assess_data_quality(ph_input["data"], "ph_effect")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Quality", f"{quality['quality_score']}/100")
        with col2:
            st.metric("Points", len(ph_input["data"]))
        with col3:
            status = "✅ Good" if quality["quality_score"] >= 70 else "⚠️ Review"
            st.metric("Status", status)

        # Calculate results based on input mode
        if input_mode == "direct":
            results_obj = _calculate_ph_results_direct(ph_input)
            st.caption("📈 *Using direct concentration input (Ce values)*")
        else:
            results_obj = _calculate_ph_results(ph_input, calib_params)

        if results_obj.success:
            results = results_obj.data
            current_study_state["ph_effect_results"] = results

            st.markdown("---")
            st.markdown("### 📊 pH Effect Data")
            display_results_table(results.round(4), hide_index=False)

            st.markdown("---")
            st.markdown("### 📈 Visualization")

            if unit_system == "Both":
                fig = create_dual_axis_effect_plot(
                    x=results["pH"],
                    y1=results["qe_mg_g"],
                    y2=results["removal_%"],
                    title="Effect of pH on Adsorption",
                    x_title="pH",
                    y1_title="qe (mg/g)",
                    y2_title="Removal (%)",
                    y1_name="qe (mg/g)",
                    y2_name="Removal (%)",
                    height=500,
                    x_tozero=False,
                    y1_tozero=True,
                    y2_tozero=True,
                )
            else:

                y_col = "qe_mg_g" if unit_system == "mg/g" else "removal_%"
                y_label = "qe (mg/g)" if unit_system == "mg/g" else "Removal (%)"

                fig = create_effect_plot(
                    x=results["pH"],
                    y=results[y_col],
                    title="Effect of pH on Adsorption",
                    x_title="pH",
                    y_title=y_label,
                    height=500,
                    series_name="pH Effect",
                    show_legend=False,
                    x_tozero=False,
                    y_tozero=True,
                    hovertemplate="pH: %{x:.1f}<br>Value: %{y:.2f}<extra></extra>",
                )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🎯 Optimal pH")

            opt_idx = results["qe_mg_g"].idxmax()
            opt_pH = results.loc[opt_idx, "pH"]
            max_qe = results.loc[opt_idx, "qe_mg_g"]
            max_removal = results.loc[opt_idx, "removal_%"]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Optimal pH", f"{opt_pH:.1f}")
            with col2:
                st.metric("Max qe", f"{max_qe:.2f} mg/g")

            st.success(f"**Optimal:** pH {opt_pH:.1f} with {max_removal:.1f}% removal")

            if opt_pH < 5:
                st.info(
                    "**Low optimal pH:** Suggests anionic adsorbate or positively charged surface"
                )
            elif opt_pH > 8:
                st.info(
                    "**High optimal pH:** Suggests cationic adsorbate or negatively charged surface"
                )
            else:
                st.info("**Neutral pH:** Electrostatic interactions are balanced")

            st.info("💡 **To download:** Go to **📦 Export All** tab")
        else:
            st.warning(f"Could not process pH data: {results_obj.error}")
            return

    elif ph_input and input_mode == "absorbance" and not calib_params:
        st.warning(
            "⚠️ Complete calibration first, or switch to **Direct Concentration** input mode in the sidebar"
        )
    elif not ph_input:
        st.info("📥 Enter pH data in sidebar")


@st.cache_data
def _calculate_ph_results(ph_input, calib_params):
    df = ph_input["data"].copy()
    params = ph_input["params"]

    slope = calib_params["slope"]
    intercept = calib_params["intercept"]
    C0 = params["C0"]
    m = params["m"]
    V = params["V"]

    # Get calibration uncertainties
    slope_se = calib_params.get("std_err_slope", 0)
    intercept_se = calib_params.get("std_err_intercept", 0)

    results = []
    for _, row in df.iterrows():
        pH = row["pH"]
        abs_val = row["Absorbance"]

        Ce = calculate_Ce_from_absorbance(abs_val, slope, intercept)
        qe = calculate_adsorption_capacity(C0, Ce, V, m)
        removal = calculate_removal_percentage(C0, Ce)

        # Calculate propagated uncertainty (returns tuple: Ce_calc, Ce_se)
        _, Ce_se = propagate_calibration_uncertainty(
            abs_val, slope, intercept, slope_se, intercept_se
        )
        qe_error = (V / m) * Ce_se if m > 0 else 0

        results.append(
            {
                "pH": pH,
                "Ce_mgL": Ce,
                "Ce_error": Ce_se,
                "qe_mg_g": qe,
                "qe_error": qe_error,
                "removal_%": removal,
            }
        )

    if not results:
        return CalculationResult(success=False, error="No valid data points to calculate results.")
    results_df = pd.DataFrame(results).sort_values("pH")
    return CalculationResult(success=True, data=results_df)


@st.cache_data
def _calculate_ph_results_direct(ph_input):
    """Calculate pH effect results from direct Ce input."""
    df = ph_input["data"].copy()
    params = ph_input["params"]

    C0 = params["C0"]
    m = params["m"]
    V = params["V"]

    results = []
    for _, row in df.iterrows():
        pH = row["pH"]
        Ce = row["Ce"]

        if Ce <= C0:
            qe = calculate_adsorption_capacity(C0, Ce, V, m)
            removal = calculate_removal_percentage(C0, Ce)

            results.append(
                {
                    "pH": pH,
                    "Ce_mgL": Ce,
                    "Ce_error": 0.0,
                    "qe_mg_g": qe,
                    "qe_error": 0.0,
                    "removal_%": removal,
                }
            )

    if not results:
        return CalculationResult(success=False, error="No valid data points. Ensure Ce ≤ C0.")
    results_df = pd.DataFrame(results).sort_values("pH")
    return CalculationResult(success=True, data=results_df)
