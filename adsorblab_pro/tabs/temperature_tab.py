# tabs/temperature_tab.py
"""
Temperature Effect Tab - AdsorbLab Pro
======================================

Analyzes the effect of temperature on adsorption capacity.

Features:
- Temperature-dependent adsorption visualization
- Endothermic/exothermic process indication
- Data preparation for thermodynamic analysis
"""

import numpy as np

from adsorblab_pro.streamlit_compat import st

from ..plot_style import create_effect_plot, create_dual_axis_effect_plot
from ..utils import (
    assess_data_quality,
    calculate_temperature_results,
    calculate_temperature_results_direct,
    display_results_table,
    eligible_observations,
    get_current_study_state,
    observation_notice,
    validate_required_params,
)


def render():
    st.subheader("🔥 Temperature Effect Study")

    current_study_state = get_current_study_state()
    if not current_study_state:
        st.info("Please add or select a study from the sidebar to begin analysis.")
        return

    unit_system = current_study_state.get("unit_system", "mg/g")
    temp_input = current_study_state.get("temp_effect_input")
    calib_params = current_study_state.get("calibration_params")

    # Check input mode (default to 'absorbance' for backward compatibility)
    input_mode = temp_input.get("input_mode", "absorbance") if temp_input else "absorbance"

    # Determine if we can proceed based on input mode
    can_proceed = False
    if temp_input:
        if input_mode == "direct":
            can_proceed = True
        elif calib_params:
            can_proceed = True

    if can_proceed:
        is_valid, error_message = validate_required_params(
            params=temp_input["params"],
            required_keys=[("C0", "C₀"), ("m", "Mass"), ("V", "Volume")],
        )
        if not is_valid:
            st.warning(error_message, icon="⚠️")
            return

        quality = assess_data_quality(temp_input["data"], "temperature")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Data checks",
                f"{quality['quality_score']}/100",
                help="Heuristic points for the number of rows, outliers and negative values; not a statistical confidence and not a measure of fit quality.",
            )
        with col2:
            st.metric("Points", len(temp_input["data"]))
        with col3:
            status = "✅ No major flags" if quality["quality_score"] >= 70 else "⚠️ Review"
            st.metric("Data flags", status)

        # Calculate results based on input mode
        if input_mode == "direct":
            results_obj = calculate_temperature_results_direct(temp_input)
            st.caption("📈 *Using direct concentration input (Ce values)*")
        else:
            results_obj = calculate_temperature_results(temp_input, calib_params)

        if results_obj.success:
            all_results = results_obj.data
            # The stored/exported table keeps every row with its status and reason.
            current_study_state["temp_effect_results"] = all_results
            results = eligible_observations(all_results)

            min_temp_c = results["Temperature_C"].min()
            if min_temp_c > 100:  # Threshold to detect possible Kelvin input
                st.warning(
                    f"**Temperature Unit Check:** The lowest temperature entered is {min_temp_c:.1f}°C. "
                    "This application expects temperature in **Celsius (°C)**. If you entered values in Kelvin, "
                    "please correct your input. However, if you are conducting **gas-phase adsorption** or "
                    "**high-temperature studies**, temperatures above 100°C are valid and you may proceed.",
                    icon="🌡️",
                )

            st.markdown("---")
            st.markdown("### 📊 Temperature Effect Data")
            notice = observation_notice(all_results)
            if notice:
                st.warning(f"⚠️ {notice}")
            display_results_table(all_results.round(4), hide_index=True)

            st.markdown("---")
            st.markdown("### 📈 Visualization")

            if unit_system == "Both":
                fig = create_dual_axis_effect_plot(
                    x=results["Temperature_C"],
                    y1=results["qe_mg_g"],
                    y2=results["removal_%"],
                    title="Effect of Temperature on Adsorption",
                    x_title="Temperature (°C)",
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
                    x=results["Temperature_C"],
                    y=results[y_col],
                    title="Effect of Temperature on Adsorption",
                    x_title="Temperature (°C)",
                    y_title=y_label,
                    height=500,
                    series_name="Temperature Effect",
                    show_legend=False,
                    x_tozero=False,
                    y_tozero=True,
                    hovertemplate="T: %{x:.1f}°C<br>Value: %{y:.2f}<extra></extra>",
                )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🔍 Temperature Trend")

            if len(results) >= 2 and results["Temperature_C"].nunique() >= 2:
                slope = np.polyfit(results["Temperature_C"], results["qe_mg_g"], 1)[0]
                if slope > 0:
                    st.info("📈 **Endothermic tendency:** qe increases with temperature")
                elif slope < 0:
                    st.info("📉 **Exothermic tendency:** qe decreases with temperature")
                else:
                    st.info("qe shows no linear trend with temperature")
            else:
                st.info("At least two distinct temperatures are needed to describe a trend.")

            st.success("💡 **Tip:** Go to **Thermodynamics** tab for ΔH°, ΔS°, ΔG° analysis")
            st.info("💡 **To download:** Go to **📦 Export All** tab")
        else:
            st.warning(f"Could not process temperature data: {results_obj.error}")
            if results_obj.data is not None and not results_obj.data.empty:
                display_results_table(results_obj.data.round(4), hide_index=True)
            current_study_state["temp_effect_results"] = None

    elif temp_input and input_mode == "absorbance" and not calib_params:
        st.warning(
            "⚠️ Complete calibration first, or switch to **Direct Concentration** input mode in the sidebar"
        )
    elif not temp_input:
        st.info("📥 Enter temperature data in sidebar")
