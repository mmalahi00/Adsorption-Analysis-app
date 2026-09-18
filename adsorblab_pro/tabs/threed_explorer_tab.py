# tabs/threed_explorer_tab.py
"""
3D Explorer Tab - AdsorbLab Pro
===============================

Interactive 3D visualizations built from measured values.

Features:
- Model residuals surface (real residuals of the fitted isotherm models)
- Experimental data in 3D (measured points only)
- Save figures for export

Deliberately absent: synthetic surfaces generated from slider-adjustable
parameters (isotherm surface over concentration x temperature, parameter-space
explorer, pH-temperature response, model comparison 3D). Those extrapolated
fitted or invented parameters across ranges the user never measured, produced
output indistinguishable from data-derived surfaces, and could be exported into
the Word report. Any simulation feature added here in future must be visibly
marked as simulated and excluded from report export.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from adsorblab_pro.streamlit_compat import st

from ..config import R_GAS_CONSTANT
from ..plot_style import apply_professional_3d_style
from ..models import (
    langmuir_model,
)
from ..utils import display_results_table, get_current_study_state

# =============================================================================
# HELPER FUNCTIONS FOR SAVED FIGURES MANAGEMENT
# =============================================================================


def _get_saved_figures() -> dict:
    """Get saved 3D figures from current study state."""
    current_study = get_current_study_state()
    if current_study is None:
        return {}
    return current_study.get("saved_3d_figures", {})


def _save_figure(fig_id: str, fig: go.Figure, title: str, params: dict):
    """Save a 3D figure to the current study's export collection."""
    study_name = st.session_state.get("current_study")
    if not study_name:
        return False

    if "studies" not in st.session_state:
        return False

    if study_name not in st.session_state.studies:
        return False

    # Initialize saved_3d_figures if not exists
    if "saved_3d_figures" not in st.session_state.studies[study_name]:
        st.session_state.studies[study_name]["saved_3d_figures"] = {}

    # Create unique key with timestamp
    timestamp = datetime.now().strftime("%H%M%S%f")
    unique_id = f"{fig_id}_{timestamp}"

    # Save figure data
    st.session_state.studies[study_name]["saved_3d_figures"][unique_id] = {
        "figure": apply_professional_3d_style(
            go.Figure(fig), title=title, height=700
        ).to_dict(),  # Store styled dict
        "title": title,
        "params": params,
        "created_at": datetime.now().isoformat(),
        "fig_type": fig_id,
    }

    return True


def _remove_figure(unique_id: str):
    """Remove a saved figure from the export collection."""
    study_name = st.session_state.get("current_study")
    if not study_name:
        return

    saved = st.session_state.studies.get(study_name, {}).get("saved_3d_figures", {})
    if unique_id in saved:
        del st.session_state.studies[study_name]["saved_3d_figures"][unique_id]


def _clear_all_figures():
    """Clear all saved 3D figures from current study."""
    study_name = st.session_state.get("current_study")
    if study_name and "studies" in st.session_state:
        if study_name in st.session_state.studies:
            st.session_state.studies[study_name]["saved_3d_figures"] = {}


def _display_saved_figures_panel():
    """Display panel showing saved figures with management options."""
    saved_figs = _get_saved_figures()

    st.markdown("### 💾 Saved 3D Figures for Export")

    if not saved_figs:
        st.info(
            "No 3D figures saved yet. Generate a visualization below and click **'Save to Export'** to add it."
        )
        return

    st.success(f"**{len(saved_figs)}** figure(s) saved for export")

    # Clear all button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        if st.button("🗑️ Clear All", type="secondary", key="clear_all_3d"):
            _clear_all_figures()
            st.rerun()

    # List saved figures
    with st.expander(f"📋 View Saved Figures ({len(saved_figs)})", expanded=False):
        for unique_id, fig_data in saved_figs.items():
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown(f"**{fig_data['title']}**")
                # Show key parameters
                params_str = ", ".join(
                    [f"{k}={v}" for k, v in list(fig_data["params"].items())[:3]]
                )
                st.caption(f"Parameters: {params_str}")

            with col2:
                created = fig_data.get("created_at", "")
                if created:
                    time_str = created.split("T")[1][:8] if "T" in created else created
                    st.caption(f"🕐 {time_str}")

            with col3:
                if st.button("❌", key=f"remove_{unique_id}", help="Remove this figure"):
                    _remove_figure(unique_id)
                    st.rerun()

            st.markdown("---")

    st.markdown("---")


# =============================================================================
# DATA RETRIEVAL FUNCTIONS
# =============================================================================


def get_fitted_parameters(current_study_state):
    """Retrieve fitted model parameters from the active study state."""
    params = {}
    if not current_study_state:
        return params

    # --- Isotherm parameters (nonlinear) ---
    isotherm_fits = current_study_state.get("isotherm_models_fitted", {})

    # Langmuir
    iso_lang_nl = isotherm_fits.get("Langmuir")
    if iso_lang_nl and iso_lang_nl.get("converged"):
        params["langmuir"] = iso_lang_nl.get("params", {})

    # Freundlich
    iso_fr_nl = isotherm_fits.get("Freundlich")
    if iso_fr_nl and iso_fr_nl.get("converged"):
        params["freundlich"] = iso_fr_nl.get("params", {})

    # Temkin
    iso_temkin_nl = isotherm_fits.get("Temkin")
    if iso_temkin_nl and iso_temkin_nl.get("converged"):
        params["temkin"] = iso_temkin_nl.get("params", {})

    # --- Kinetic parameters (nonlinear) ---
    kinetic_fits = current_study_state.get("kinetic_models_fitted", {})

    # PSO
    pso_nl = kinetic_fits.get("PSO")
    if pso_nl and pso_nl.get("converged"):
        params["pso"] = pso_nl.get("params", {})

    # rPSO (revised PSO) - use as fallback if PSO not available
    rpso_nl = kinetic_fits.get("rPSO")
    if rpso_nl and rpso_nl.get("converged"):
        params["rpso"] = rpso_nl.get("params", {})
        # If PSO not fitted, use rPSO params for compatibility
        if "pso" not in params:
            params["pso"] = rpso_nl.get("params", {})

    # PFO
    pfo_nl = kinetic_fits.get("PFO")
    if pfo_nl and pfo_nl.get("converged"):
        params["pfo"] = pfo_nl.get("params", {})

    # --- Thermodynamic parameters ---
    thermo = current_study_state.get("thermo_params")
    if thermo and isinstance(thermo, dict):
        params["thermo"] = thermo

    return params


def get_experimental_data(current_study_state):
    """Retrieve experimental data from the active study state."""
    data = {}
    if not current_study_state:
        return data

    # Isotherm data
    iso_results = current_study_state.get("isotherm_results")
    if iso_results is not None and hasattr(iso_results, "shape") and not iso_results.empty:
        data["isotherm"] = {
            "Ce": iso_results["Ce_mgL"].values,
            "qe": iso_results["qe_mg_g"].values,
            "C0": iso_results["C0_mgL"].values if "C0_mgL" in iso_results.columns else None,
            "removal": iso_results["removal_%"].values
            if "removal_%" in iso_results.columns
            else None,
        }

    # Kinetic data
    kin_results = current_study_state.get("kinetic_results_df")
    if kin_results is not None and hasattr(kin_results, "shape") and not kin_results.empty:
        data["kinetic"] = {
            "t": kin_results["Time"].values,
            "qt": kin_results["qt_mg_g"].values,
            "Ct": kin_results["Ct_mgL"].values if "Ct_mgL" in kin_results.columns else None,
        }

    # Temperature data
    temp_results = current_study_state.get("temp_effect_results")
    if temp_results is not None and hasattr(temp_results, "shape") and not temp_results.empty:
        data["temperature"] = {
            "T_C": temp_results["Temperature_C"].values,
            "T_K": temp_results["Temperature_K"].values,
            "qe": temp_results["qe_mg_g"].values,
            "Ce": temp_results["Ce_mgL"].values if "Ce_mgL" in temp_results.columns else None,
        }

    # pH effect data
    ph_results = current_study_state.get("ph_effect_results")
    if ph_results is not None and hasattr(ph_results, "shape") and not ph_results.empty:
        data["ph_effect"] = {
            "pH": ph_results["pH"].values,
            "qe": ph_results["qe_mg_g"].values,
            "Ce": ph_results["Ce_mgL"].values if "Ce_mgL" in ph_results.columns else None,
        }

    # Dosage data
    dosage_results = current_study_state.get("dosage_results")
    if dosage_results is not None and hasattr(dosage_results, "shape") and not dosage_results.empty:
        data["dosage"] = {
            "mass": dosage_results["Mass_g"].values,
            "qe": dosage_results["qe_mg_g"].values,
            "Ce": dosage_results["Ce_mgL"].values if "Ce_mgL" in dosage_results.columns else None,
        }

    return data


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================


def render():
    """Render the 3D Explorer tab."""
    st.subheader("🔮 3D Model Explorer")

    st.markdown("""
    Explore adsorption models in 3D using your experimental data and fitted parameters.

    **Workflow:**
    1. Select a visualization type
    2. Configure parameters
    3. Click **Generate** to preview
    4. Click **Save to Export** to add to your export collection
    """)

    current_study_state = get_current_study_state()
    if not current_study_state:
        st.info("Please add or select a study from the sidebar to begin analysis.")
        return

    # Show saved figures panel
    _display_saved_figures_panel()

    # Data availability dashboard
    fitted_params = get_fitted_parameters(current_study_state)
    exp_data = get_experimental_data(current_study_state)

    # Validate that we have real experimental data
    has_isotherm_data = "isotherm" in exp_data and len(exp_data["isotherm"].get("Ce", [])) >= 3
    has_kinetic_data = "kinetic" in exp_data and len(exp_data["kinetic"].get("t", [])) >= 3
    has_temp_data = "temperature" in exp_data and len(exp_data["temperature"].get("T_C", [])) >= 2
    has_any_fitted = len(fitted_params) > 0

    # Check if enough data is available
    if not has_any_fitted and not (has_isotherm_data or has_kinetic_data):
        st.warning("""
        ⚠️ **Insufficient Data for 3D Visualization**

        The 3D Explorer requires:
        - Completed isotherm or kinetic analysis with fitted models, OR
        - At least 3 experimental data points

        Please complete the following steps first:
        1. Enter calibration data in the **📊 Calibration** tab
        2. Enter experimental data in **📈 Isotherm** or **⏱️ Kinetics** tabs
        3. Run the model fitting analysis
        """)
        return

    # Show data availability
    with st.expander("📊 Data Availability", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Fitted Parameters:**")
            for model_name in ["langmuir", "freundlich", "temkin", "pso", "rpso", "thermo"]:
                if model_name in fitted_params:
                    st.success(
                        f"✅ {model_name.upper() if model_name == 'rpso' else model_name.title()}"
                    )
                else:
                    st.info(
                        f"📝 {model_name.upper() if model_name == 'rpso' else model_name.title()} (not fitted)"
                    )

        with col2:
            st.markdown("**Experimental Data:**")
            for data_type in ["isotherm", "kinetic", "temperature", "ph_effect", "dosage"]:
                if data_type in exp_data:
                    points = len(exp_data[data_type].get("qe", exp_data[data_type].get("qt", [])))
                    st.success(f"✅ {data_type.replace('_', ' ').title()} ({points} points)")
                else:
                    st.info(f"📝 {data_type.replace('_', ' ').title()} (no data)")

    # Build available visualizations based on data.
    #
    # Only views built from measured values are offered.  The Isotherm Surface,
    # Parameter Space Explorer, pH-Temperature Response and Model Comparison 3D
    # views were removed: each generated a synthetic surface from
    # slider-adjustable parameters — including, where no thermodynamic fit
    # existed, a default ΔH of -25 kJ/mol invented by the UI — and extrapolated
    # it across temperature and concentration ranges the user never measured.
    # The output was visually indistinguishable from a data-derived surface and
    # could be exported straight into the Word report.  This is the same
    # objection that retired the multi-concentration kinetic surface.
    available_viz = []
    viz_requirements = {}

    if has_isotherm_data and ("langmuir" in fitted_params or "freundlich" in fitted_params):
        available_viz.append("Model Residuals Surface")
        viz_requirements["Model Residuals Surface"] = "Requires isotherm data with fitted models"

    if has_isotherm_data or has_kinetic_data or has_temp_data:
        available_viz.append("Experimental Data 3D")
        viz_requirements["Experimental Data 3D"] = "Requires any experimental data"

    if not available_viz:
        st.warning("No 3D visualizations available. Please complete more analyses first.")
        return

    # Visualization selector
    st.markdown("### 🎨 Create New Visualization")

    viz_type = st.selectbox(
        "Choose 3D Visualization:",
        available_viz,
        help="Only visualizations with sufficient data are shown",
    )

    # Show requirements
    if viz_type in viz_requirements:
        st.caption(f"ℹ️ {viz_requirements[viz_type]}")

    st.markdown("---")

    # Render the configuration and generation for selected visualization
    if viz_type == "Model Residuals Surface":
        _render_residuals_surface(fitted_params, exp_data)
    elif viz_type == "Experimental Data 3D":
        _render_experimental_3d(exp_data)


# =============================================================================
# VISUALIZATION RENDER FUNCTIONS
# =============================================================================


def _render_experimental_3d(exp_data):
    """3D scatter plot of experimental data."""
    st.markdown("### 🔬 Experimental Data 3D")

    available_data = []
    if "isotherm" in exp_data:
        available_data.append("Isotherm (Ce vs qe)")
    if "kinetic" in exp_data:
        available_data.append("Kinetic (Time vs qt)")
    if "temperature" in exp_data:
        available_data.append("Temperature Effect")

    if not available_data:
        st.warning("No experimental data available.")
        return

    data_type = st.selectbox("Select data to visualize:", available_data, key="exp_data_type")

    # Generate button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_clicked = st.button(
            "🚀 Generate 3D Scatter",
            type="primary",
            use_container_width=True,
            key="gen_experimental",
        )

    if not generate_clicked and "experimental_fig" not in st.session_state:
        st.caption("Click **Generate** to create the 3D visualization.")
        return

    try:
        with st.spinner("Generating 3D scatter plot..."):
            fig = go.Figure()

            if "Isotherm" in data_type and "isotherm" in exp_data:
                iso = exp_data["isotherm"]
                # Use index as third dimension if no other data
                indices = np.arange(len(iso["Ce"]))

                fig.add_trace(
                    go.Scatter3d(
                        x=iso["Ce"],
                        y=indices,
                        z=iso["qe"],
                        mode="markers",
                        marker={
                            "size": 8,
                            "color": iso["qe"],
                            "colorscale": "Viridis",
                            "showscale": True,
                            "colorbar": {"title": "qe (mg/g)"},
                        },
                        name="Isotherm Data",
                    )
                )

                title = "Isotherm Data 3D (Ce vs Index vs qe)"
                x_title, y_title, z_title = "Ce (mg/L)", "Data Point Index", "qe (mg/g)"

            elif "Kinetic" in data_type and "kinetic" in exp_data:
                kin = exp_data["kinetic"]
                indices = np.arange(len(kin["t"]))

                fig.add_trace(
                    go.Scatter3d(
                        x=kin["t"],
                        y=indices,
                        z=kin["qt"],
                        mode="markers+lines",
                        marker={
                            "size": 6,
                            "color": kin["qt"],
                            "colorscale": "Plasma",
                            "showscale": True,
                        },
                        line={"width": 2, "color": "gray"},
                        name="Kinetic Data",
                    )
                )

                title = "Kinetic Data 3D (Time vs Index vs qt)"
                x_title, y_title, z_title = "Time (min)", "Data Point Index", "qt (mg/g)"

            elif "Temperature" in data_type and "temperature" in exp_data:
                temp = exp_data["temperature"]

                fig.add_trace(
                    go.Scatter3d(
                        x=temp["T_C"],
                        y=temp.get("Ce", np.arange(len(temp["T_C"]))),
                        z=temp["qe"],
                        mode="markers",
                        marker={
                            "size": 10,
                            "color": temp["qe"],
                            "colorscale": "RdYlBu_r",
                            "line": {"width": 1.0, "color": "#000000"},
                            "showscale": True,
                            "colorbar": {"title": "qe (mg/g)"},
                        },
                        name="Temperature Effect Data",
                    )
                )

                title = "Temperature Effect 3D (T vs Ce vs qe)"
                x_title = "Temperature (°C)"
                y_title = "Ce (mg/L)" if temp.get("Ce") is not None else "Index"
                z_title = "qe (mg/g)"

            fig.update_layout(
                scene={
                    "xaxis_title": x_title,
                    "yaxis_title": y_title,
                    "zaxis_title": z_title,
                }
            )
            fig = apply_professional_3d_style(
                fig,
                title=title,
                height=700,
                camera_eye={"x": 1.8, "y": 1.8, "z": 1.2},
            )

            st.session_state["experimental_fig"] = fig
            st.session_state["experimental_params"] = {"data_type": data_type}
            st.session_state["experimental_title"] = title

        st.plotly_chart(fig, use_container_width=True)

        # Save button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "💾 Save to Export Collection",
                type="secondary",
                use_container_width=True,
                key="save_experimental",
            ):
                if _save_figure(
                    "experimental_3d", fig, title, st.session_state["experimental_params"]
                ):
                    st.success("✅ Figure saved to export collection!")
                    st.rerun()
                else:
                    st.error("Failed to save figure")

    except Exception as e:
        st.error(f"Error generating 3D scatter: {str(e)}")


def _render_residuals_surface(fitted_params, exp_data):
    """3D residuals surface for model diagnostics."""
    st.markdown("### 📉 Model Residuals Surface")

    # Check requirements
    has_temp = "temperature" in exp_data and exp_data["temperature"].get("Ce") is not None
    has_langmuir = "langmuir" in fitted_params
    has_thermo = "thermo" in fitted_params

    if not (has_temp and has_langmuir and has_thermo):
        missing = []
        if not has_temp:
            missing.append("temperature effect data with Ce values")
        if not has_langmuir:
            missing.append("fitted Langmuir model")
        if not has_thermo:
            missing.append("thermodynamic parameters")

        st.warning(f"Missing: {', '.join(missing)}")
        st.info("This visualization requires temperature-dependent Langmuir model validation.")
        return

    st.info(
        "This plot shows residuals (experimental - predicted) for the temperature-dependent Langmuir model."
    )

    # Generate button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_clicked = st.button(
            "🚀 Generate Residuals Surface",
            type="primary",
            use_container_width=True,
            key="gen_residuals",
        )

    if not generate_clicked and "residuals_fig" not in st.session_state:
        st.caption("Click **Generate** to create the residuals visualization.")
        return

    try:
        with st.spinner("Calculating residuals..."):
            temp_data = exp_data["temperature"]
            langmuir_params = fitted_params["langmuir"]
            thermo_params = fitted_params["thermo"]

            qm = langmuir_params.get("qm")
            KL_ref = langmuir_params.get("KL")
            delta_H = thermo_params.get("delta_H") * 1000  # kJ to J

            T_ref = 298.15

            # Calculate residuals
            residuals = []
            qe_predicted = []

            for i in range(len(temp_data["T_K"])):
                qe_exp = temp_data["qe"][i]
                Ce_exp = temp_data["Ce"][i]
                T_K = temp_data["T_K"][i]

                # Temperature-adjusted KL
                KL_adj = KL_ref * np.exp(-delta_H / R_GAS_CONSTANT * (1 / T_K - 1 / T_ref))
                qe_pred = langmuir_model(Ce_exp, qm, KL_adj)

                qe_predicted.append(qe_pred)
                residuals.append(qe_exp - qe_pred)

            # Create 3D plot
            fig = go.Figure()

            # Zero plane
            Ce_range = [0, np.max(temp_data["Ce"]) * 1.1]
            T_range = [np.min(temp_data["T_C"]) - 5, np.max(temp_data["T_C"]) + 5]
            fig.add_trace(
                go.Surface(
                    x=Ce_range,
                    y=T_range,
                    z=[[0, 0], [0, 0]],
                    opacity=0.3,
                    colorscale=[[0, "gray"], [1, "gray"]],
                    showscale=False,
                    name="Zero Plane",
                )
            )

            # Residual points
            fig.add_trace(
                go.Scatter3d(
                    x=temp_data["Ce"],
                    y=temp_data["T_C"],
                    z=residuals,
                    mode="markers",
                    marker={
                        "size": 8,
                        "color": residuals,
                        "colorscale": "RdBu",
                        "line": {"width": 1.0, "color": "#000000"},
                        "colorbar_title": "Residual",
                        "showscale": True,
                        "cmin": -max(abs(min(residuals)), abs(max(residuals))),
                        "cmax": max(abs(min(residuals)), abs(max(residuals))),
                    },
                    name="Residuals",
                )
            )

            title = "Temperature-Dependent Langmuir Residuals"
            fig.update_layout(
                scene={
                    "xaxis_title": "Ce (mg/L)",
                    "yaxis_title": "Temperature (°C)",
                    "zaxis_title": "Residual (mg/g)",
                }
            )
            fig = apply_professional_3d_style(
                fig,
                title=title,
                height=700,
                camera_eye={"x": 1.8, "y": -1.8, "z": 1.2},
            )

            st.session_state["residuals_fig"] = fig
            st.session_state["residuals_params"] = {
                "qm": qm,
                "KL": KL_ref,
                "ΔH": delta_H / 1000,
                "n_points": len(residuals),
            }
            st.session_state["residuals_title"] = title

        st.plotly_chart(fig, use_container_width=True)

        # Save button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "💾 Save to Export Collection",
                type="secondary",
                use_container_width=True,
                key="save_residuals",
            ):
                if _save_figure(
                    "residuals_surface", fig, title, st.session_state["residuals_params"]
                ):
                    st.success("✅ Figure saved to export collection!")
                    st.rerun()
                else:
                    st.error("Failed to save figure")

        # Residuals table
        with st.expander("📊 View Residuals Data"):
            res_df = pd.DataFrame(
                {
                    "T (°C)": temp_data["T_C"],
                    "Ce (mg/L)": temp_data["Ce"],
                    "qe_exp (mg/g)": temp_data["qe"],
                    "qe_pred (mg/g)": qe_predicted,
                    "Residual (mg/g)": residuals,
                }
            )
            display_results_table(res_df.round(4))

            st.info("""
            - **Positive Residual:** Model under-predicts
            - **Negative Residual:** Model over-predicts
            """)

    except Exception as e:
        st.error(f"Error generating residuals surface: {str(e)}")
