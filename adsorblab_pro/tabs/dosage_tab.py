# tabs/dosage_tab.py
"""
Dosage Effect Tab - AdsorbLab Pro
=================================

Analyzes the effect of adsorbent dosage on removal efficiency.

Features:
- Dosage-capacity relationship visualization
- Optimal dosage determination
- Cost-efficiency analysis support
"""

import pandas as pd

from adsorblab_pro.streamlit_compat import st

from ..plot_style import create_effect_plot, create_dual_axis_effect_plot
from ..utils import (
    EPSILON_DIV,
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
    st.subheader("⚖️ Adsorbent Dosage Effect")

    current_study_state = get_current_study_state()
    if not current_study_state:
        st.info("Please add or select a study from the sidebar to begin analysis.")
        return

    unit_system = current_study_state.get("unit_system", "mg/g")
    dos_input = current_study_state.get("dosage_input")
    calib_params = current_study_state.get("calibration_params")

    # Check input mode (default to 'absorbance' for backward compatibility)
    input_mode = dos_input.get("input_mode", "absorbance") if dos_input else "absorbance"

    # Determine if we can proceed based on input mode
    can_proceed = False
    if dos_input:
        if input_mode == "direct":
            can_proceed = True
        elif calib_params:
            can_proceed = True

    if can_proceed:
        is_valid, error_message = validate_required_params(
            params=dos_input["params"], required_keys=[("C0", "C₀"), ("V", "Volume")]
        )
        if not is_valid:
            st.warning(error_message, icon="⚠️")
            return

        quality = assess_data_quality(dos_input["data"], "dosage")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Quality", f"{quality['quality_score']}/100")
        with col2:
            st.metric("Points", len(dos_input["data"]))
        with col3:
            status = "✅ Good" if quality["quality_score"] >= 70 else "⚠️ Review"
            st.metric("Status", status)

        # Calculate results based on input mode
        if input_mode == "direct":
            results_obj = _calculate_dosage_results_direct(dos_input)
            st.caption("📈 *Using direct concentration input (Ce values)*")
        else:
            results_obj = _calculate_dosage_results(dos_input, calib_params)

        if results_obj.success:
            results = results_obj.data
            current_study_state["dosage_results"] = results

            st.markdown("---")
            st.markdown("### 📊 Dosage Effect Data")
            display_results_table(results.round(4), hide_index=False)

            st.markdown("---")
            st.markdown("### 📈 Visualization")

            if unit_system == "Both":
                fig = create_dual_axis_effect_plot(
                    x=results["Mass_g"],
                    y1=results["qe_mg_g"],
                    y2=results["removal_%"],
                    title="Dosage Effect",
                    x_title="Adsorbent Mass (g)",
                    y1_title="qe (mg/g)",
                    y2_title="Removal (%)",
                    y1_name="qe (mg/g)",
                    y2_name="Removal (%)",
                    height=500,
                    x_tozero=True,
                    y1_tozero=True,
                    y2_tozero=True,
                )
            else:
                y_col = "qe_mg_g" if unit_system == "mg/g" else "removal_%"
                y_label = "qe (mg/g)" if unit_system == "mg/g" else "Removal (%)"

                fig = create_effect_plot(
                    x=results["Mass_g"],
                    y=results[y_col],
                    title="Effect of Adsorbent Dosage",
                    x_title="Mass (g)",
                    y_title=y_label,
                    height=500,
                    series_name="Dosage Effect",
                    show_legend=False,
                    x_tozero=True,
                    y_tozero=True,
                    hovertemplate="Mass: %{x:.4f} g<br>Value: %{y:.2f}<extra></extra>",
                )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🎯 Optimal Dosage")

            if results["removal_%"].max() >= 90:
                opt_idx = (results["removal_%"] >= 90).idxmax()
                opt_mass = results.loc[opt_idx, "Mass_g"]
                st.success(f"**Optimal mass for ≥90% removal:** {opt_mass:.4f} g")
            else:
                max_removal = results["removal_%"].max()
                opt_mass = results.loc[results["removal_%"].idxmax(), "Mass_g"]
                st.info(f"**Maximum removal {max_removal:.1f}% at mass:** {opt_mass:.4f} g")

            st.info("💡 **To download:** Go to **📦 Export All** tab")

        else:
            st.warning(f"Could not process dosage data: {results_obj.error}")
            return

    elif dos_input and input_mode == "absorbance" and not calib_params:
        st.warning(
            "⚠️ Complete calibration first, or switch to **Direct Concentration** input mode in the sidebar"
        )
    elif not dos_input:
        st.info("📥 Enter dosage data in sidebar")


@st.cache_data
def _calculate_dosage_results(dos_input, calib_params):
    df = dos_input["data"].copy()
    params = dos_input["params"]

    slope = calib_params["slope"]
    intercept = calib_params["intercept"]
    C0 = params["C0"]
    V = params["V"]

    # Get calibration uncertainties
    slope_se = calib_params.get("std_err_slope", 0)
    intercept_se = calib_params.get("std_err_intercept", 0)

    results = []
    for _, row in df.iterrows():
        m = row["Mass"]
        abs_val = row["Absorbance"]

        if m > EPSILON_DIV:
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
                    "Mass_g": m,
                    "Ce_mgL": Ce,
                    "Ce_error": Ce_se,
                    "qe_mg_g": qe,
                    "qe_error": qe_error,
                    "removal_%": removal,
                }
            )

    if not results:
        return CalculationResult(success=False, error="No valid data points to calculate results.")
    results_df = pd.DataFrame(results).sort_values("Mass_g")
    return CalculationResult(success=True, data=results_df)


@st.cache_data
def _calculate_dosage_results_direct(dos_input):
    """Calculate dosage results from direct Ce input."""
    df = dos_input["data"].copy()
    params = dos_input["params"]

    C0 = params["C0"]
    V = params["V"]

    results = []
    for _, row in df.iterrows():
        m = row["Mass"]
        Ce = row["Ce"]

        if m > EPSILON_DIV and Ce <= C0:
            qe = calculate_adsorption_capacity(C0, Ce, V, m)
            removal = calculate_removal_percentage(C0, Ce)

            results.append(
                {
                    "Mass_g": m,
                    "Ce_mgL": Ce,
                    "Ce_error": 0.0,
                    "qe_mg_g": qe,
                    "qe_error": 0.0,
                    "removal_%": removal,
                }
            )

    if not results:
        return CalculationResult(
            success=False, error="No valid data points. Ensure Ce ≤ C0 and Mass > 0."
        )
    results_df = pd.DataFrame(results).sort_values("Mass_g")
    return CalculationResult(success=True, data=results_df)
