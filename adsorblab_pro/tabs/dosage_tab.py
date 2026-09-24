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

from adsorblab_pro.streamlit_compat import st

from ..plot_style import create_effect_plot, create_dual_axis_effect_plot
from ..utils import (
    assess_data_quality,
    build_uptake_table,
    display_results_table,
    eligible_observations,
    get_current_study_state,
    observation_notice,
    uptake_calculation_result,
    uptake_results_frame,
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
            st.metric(
                "Data checks",
                f"{quality['quality_score']}/100",
                help="Heuristic points for the number of rows, outliers and negative values; not a statistical confidence and not a measure of fit quality.",
            )
        with col2:
            st.metric("Points", len(dos_input["data"]))
        with col3:
            status = "✅ No major flags" if quality["quality_score"] >= 70 else "⚠️ Review"
            st.metric("Data flags", status)

        # Calculate results based on input mode
        if input_mode == "direct":
            results_obj = _calculate_dosage_results_direct(dos_input)
            st.caption("📈 *Using direct concentration input (Ce values)*")
        else:
            results_obj = _calculate_dosage_results(dos_input, calib_params)

        if results_obj.success:
            all_results = results_obj.data
            # The stored/exported table keeps every row with its status and reason.
            current_study_state["dosage_results"] = all_results
            results = eligible_observations(all_results)

            st.markdown("---")
            st.markdown("### 📊 Dosage Effect Data")
            notice = observation_notice(all_results)
            if notice:
                st.warning(f"⚠️ {notice}")
            display_results_table(all_results.round(4), hide_index=True)

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
            if len(results) < len(all_results):
                st.caption(
                    "This summary uses quantified observations only; excluded or unresolved "
                    "rows (see the notes above, e.g. below the LOD) are not counted."
                )

            st.info("💡 **To download:** Go to **📦 Export All** tab")

        else:
            st.warning(f"Could not process dosage data: {results_obj.error}")
            if results_obj.data is not None and not results_obj.data.empty:
                display_results_table(results_obj.data.round(4), hide_index=True)
            current_study_state["dosage_results"] = None
            return

    elif dos_input and input_mode == "absorbance" and not calib_params:
        st.warning(
            "⚠️ Complete calibration first, or switch to **Direct Concentration** input mode in the sidebar"
        )
    elif not dos_input:
        st.info("📥 Enter dosage data in sidebar")


def _dosage_frame(dos_input, mode, calib_params=None):
    """Shared dosage calculation: every row kept with its source row and status."""
    params = dos_input["params"]
    data = dos_input["data"]
    table = build_uptake_table(
        data,
        mode=mode,
        signal_col="Absorbance" if mode != "direct" else "Ce",
        C0=params["C0"],
        V=params["V"],
        m="Mass",
        calib_params=calib_params,
        row_notes=dos_input.get("row_issues"),
    )
    frame = uptake_results_frame(
        table,
        leading={"Mass_g": table["m"].to_numpy()},
        include_signal=mode != "direct",
        sort_by="Mass_g",
    )
    return uptake_calculation_result(frame)


@st.cache_data
def _calculate_dosage_results(dos_input, calib_params):
    """Dosage results from absorbance with per-row status (nothing skipped or clipped)."""
    return _dosage_frame(dos_input, "absorbance", calib_params)


@st.cache_data
def _calculate_dosage_results_direct(dos_input):
    """Dosage results from direct Ce input; rows with Ce > C0 or mass <= 0 are kept, excluded."""
    return _dosage_frame(dos_input, "direct")
