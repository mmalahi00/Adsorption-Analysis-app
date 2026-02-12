# tabs/threed_explorer_tab.py
"""
3D Explorer Tab - AdsorbLab Pro
===============================

Interactive 3D visualizations and parameter space exploration.

Features:
- 3D surface plots for isotherm models
- Parameter space visualization
- pH-Temperature response surfaces
- Save figures for export
"""

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from adsorblab_pro.streamlit_compat import st

from ..plot_style import COLORS, apply_professional_3d_style
from ..models import (
    freundlich_model,
    langmuir_3d_surface,
    langmuir_model,
    parameter_space_visualization,
    
    pso_model,
    temkin_model,
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
        "figure": apply_professional_3d_style(go.Figure(fig), title=title, height=700).to_dict(),  # Store styled dict
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
    has_ph_data = "ph_effect" in exp_data and len(exp_data["ph_effect"].get("pH", [])) >= 2
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

    # Build available visualizations based on data
    available_viz = []
    viz_requirements = {}

    if has_isotherm_data or "langmuir" in fitted_params:
        available_viz.append("Isotherm Surface (Concentration × Temperature)")
        viz_requirements["Isotherm Surface (Concentration × Temperature)"] = (
            "Requires isotherm data or Langmuir fit"
        )

    if has_isotherm_data and ("langmuir" in fitted_params or "freundlich" in fitted_params):
        available_viz.append("Model Residuals Surface")
        viz_requirements["Model Residuals Surface"] = "Requires isotherm data with fitted models"

    if "langmuir" in fitted_params or "freundlich" in fitted_params:
        available_viz.append("Parameter Space Explorer")
        viz_requirements["Parameter Space Explorer"] = "Requires Langmuir or Freundlich fit"

    if has_temp_data and has_ph_data:
        available_viz.append("pH-Temperature Response")
        viz_requirements["pH-Temperature Response"] = "Requires temperature and pH effect data"
    elif has_temp_data or has_ph_data:
        available_viz.append("pH-Temperature Response")
        viz_requirements["pH-Temperature Response"] = "Limited: needs both pH and temperature data"

    if len([k for k in ["langmuir", "freundlich", "temkin"] if k in fitted_params]) >= 2:
        available_viz.append("Model Comparison 3D")
        viz_requirements["Model Comparison 3D"] = "Requires 2+ fitted isotherm models"

    if has_kinetic_data and "pso" in fitted_params:
        available_viz.append("Kinetic Multi-Concentration")
        viz_requirements["Kinetic Multi-Concentration"] = "Requires kinetic data with PSO fit"

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
    if viz_type == "Isotherm Surface (Concentration × Temperature)":
        _render_isotherm_surface(fitted_params, exp_data)
    elif viz_type == "Model Residuals Surface":
        _render_residuals_surface(fitted_params, exp_data)
    elif viz_type == "Parameter Space Explorer":
        _render_parameter_space(fitted_params, exp_data)
    elif viz_type == "pH-Temperature Response":
        _render_ph_temp_response(fitted_params, exp_data)
    elif viz_type == "Model Comparison 3D":
        _render_model_comparison(fitted_params, exp_data)
    elif viz_type == "Kinetic Multi-Concentration":
        _render_kinetic_3d(fitted_params, exp_data)
    elif viz_type == "Experimental Data 3D":
        _render_experimental_3d(exp_data)


# =============================================================================
# VISUALIZATION RENDER FUNCTIONS
# =============================================================================


def _render_isotherm_surface(fitted_params, exp_data):
    """3D isotherm surface with experimental data integration."""
    st.markdown("### 🌡️ Isotherm Surface: Ce × Temperature")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model Parameters**")

        # Use Langmuir fitted values if available
        if "langmuir" in fitted_params:
            qm_fitted = fitted_params["langmuir"].get("qm", 50.0)
            KL_fitted = fitted_params["langmuir"].get("KL", 0.1)
            st.success(f"Using fitted Langmuir: qm={qm_fitted:.2f}, KL={KL_fitted:.4f}")
            default_qm = float(qm_fitted)
            default_KL = float(KL_fitted)
        else:
            default_qm = 50.0
            default_KL = 0.1
            st.info("Using default parameters (Langmuir not fitted)")

        qm = st.slider("qm (mg/g)", 1.0, 500.0, default_qm, 5.0, key="iso_surf_qm")
        KL = st.slider("KL (L/mg)", 0.001, 5.0, default_KL, 0.01, key="iso_surf_KL")

        # Use thermodynamic ΔH if available
        if "thermo" in fitted_params:
            delta_H_fitted = fitted_params["thermo"].get("delta_H", -25.0) * 1000  # kJ→J
            st.info(f"Using ΔH = {delta_H_fitted / 1000:.1f} kJ/mol from thermodynamics")
            default_delta_H = float(delta_H_fitted / 1000)  # Back to kJ for slider
        else:
            default_delta_H = -25.0
            st.info("Using default ΔH (thermodynamics not calculated)")

        delta_H_kJ = st.slider("ΔH (kJ/mol)", -100.0, 50.0, default_delta_H, 5.0, key="iso_surf_dH")
        delta_H = delta_H_kJ * 1000  # Convert to J

    with col2:
        st.markdown("**Range Settings**")

        # Get concentration range from experimental data
        if "isotherm" in exp_data:
            Ce_max_exp = np.max(exp_data["isotherm"]["Ce"])
            Ce_min_exp = np.min(exp_data["isotherm"]["Ce"])
            Ce_max = max(Ce_max_exp * 1.5, 100.0)
            st.info(f"Experimental Ce range: {Ce_min_exp:.1f} to {Ce_max_exp:.1f} mg/L")
        else:
            Ce_max = 100.0

        Ce_max_slider = st.slider(
            "Max Ce (mg/L)", 1.0, 500.0, float(Ce_max), 10.0, key="iso_surf_Ce"
        )

        # Get temperature range from experimental data
        if "temperature" in exp_data:
            T_min_exp = np.min(exp_data["temperature"]["T_C"])
            T_max_exp = np.max(exp_data["temperature"]["T_C"])
            T_min = max(10.0, T_min_exp - 10)
            T_max = max(T_max_exp + 10, 60.0)
            st.info(f"Experimental T range: {T_min_exp:.1f} to {T_max_exp:.1f} °C")
        else:
            T_min, T_max = 20.0, 60.0

        temp_min = st.slider("Min Temp (°C)", 10.0, 90.0, float(T_min), 5.0, key="iso_surf_Tmin")
        temp_max = st.slider("Max Temp (°C)", 20.0, 100.0, float(T_max), 5.0, key="iso_surf_Tmax")

    # Generate button
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_clicked = st.button(
            "🚀 Generate Visualization",
            type="primary",
            use_container_width=True,
            key="gen_iso_surf",
        )

    if not generate_clicked and "iso_surface_fig" not in st.session_state:
        st.caption("Configure parameters above, then click **Generate** to preview the 3D surface.")
        return

    # Generate the figure
    try:
        with st.spinner("Generating 3D surface..."):
            Ce_grid, temp_grid, qe_grid = langmuir_3d_surface(
                (0.1, Ce_max_slider), (temp_min, temp_max), qm, KL, delta_H
            )

            fig = go.Figure(
                data=[
                    go.Surface(
                        z=qe_grid,
                        x=Ce_grid,
                        y=temp_grid,
                        colorscale="Viridis",
                        opacity=0.85,
                        name="Langmuir Model",
                        showscale=True,
                        contours={
                            "z": {
                                "show": True,
                                "usecolormap": True,
                                "highlightcolor": "limegreen",
                                "project": {"z": True},
                            }
                        },
                    )
                ]
            )

            # Add experimental points if available
            if "temperature" in exp_data and exp_data["temperature"]["Ce"] is not None:
                temp_data = exp_data["temperature"]
                fig.add_trace(
                    go.Scatter3d(
                        x=temp_data["Ce"],
                        y=temp_data["T_C"],
                        z=temp_data["qe"],
                        mode="markers",
                        marker={"size": 6, "color": COLORS["experimental"], "symbol": "diamond", "line": {"width": 1.0, "color": "#000000"}},
                        name="Experimental Data",
                        showlegend=True,
                    )
                )

            title = f"Langmuir Surface: qm={qm:.1f} mg/g, KL={KL:.3f} L/mg"
            fig.update_layout(scene={
                "xaxis_title": "Ce (mg/L)",
                "yaxis_title": "Temperature (°C)",
                "zaxis_title": "qe (mg/g)",
            })
            fig = apply_professional_3d_style(
                fig, title=title, height=700,
                camera_eye={"x": 1.8, "y": 1.8, "z": 1.2},
            )

            # Store in session state for save button
            st.session_state["iso_surface_fig"] = fig
            st.session_state["iso_surface_params"] = {
                "qm": qm,
                "KL": KL,
                "ΔH": delta_H_kJ,
                "Ce_max": Ce_max_slider,
                "T_range": f"{temp_min}-{temp_max}°C",
            }
            st.session_state["iso_surface_title"] = title

        # Display figure
        st.plotly_chart(fig, use_container_width=True)

        # Save button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "💾 Save to Export Collection",
                type="secondary",
                use_container_width=True,
                key="save_iso_surf",
            ):
                if _save_figure(
                    "isotherm_surface", fig, title, st.session_state["iso_surface_params"]
                ):
                    st.success("✅ Figure saved to export collection!")
                    st.rerun()
                else:
                    st.error("Failed to save figure")

        # Show experimental data table
        if "isotherm" in exp_data:
            with st.expander("📊 Experimental Isotherm Data"):
                exp_df = pd.DataFrame(
                    {
                        "Ce (mg/L)": exp_data["isotherm"]["Ce"],
                        "qe (mg/g)": exp_data["isotherm"]["qe"],
                    }
                )
                display_results_table(exp_df.round(4))

    except Exception as e:
        st.error(f"Error generating surface: {str(e)}")
        st.info("Try adjusting the parameter ranges or check your data.")


def _render_parameter_space(fitted_params, exp_data):
    """Parameter space explorer using fitted parameters."""
    st.markdown("### 🎯 Parameter Space Explorer")

    col1, col2 = st.columns(2)

    with col1:
        model = st.selectbox(
            "Model:", ["Langmuir", "Freundlich", "Temkin"], key="param_space_model"
        )

        # Get concentration from experimental data
        if "isotherm" in exp_data:
            Ce_median = np.median(exp_data["isotherm"]["Ce"])
            Ce_fixed = st.slider(
                "Fixed Ce (mg/L)", 0.1, 500.0, float(Ce_median), 1.0, key="param_Ce"
            )
        else:
            Ce_fixed = st.slider("Fixed Ce (mg/L)", 1.0, 200.0, 50.0, 5.0, key="param_Ce")

    with col2:
        # Set parameter ranges based on fitted values
        if model == "Langmuir":
            if "langmuir" in fitted_params:
                qm_default = fitted_params["langmuir"].get("qm", 50.0)
                KL_default = fitted_params["langmuir"].get("KL", 0.1)
                st.success(f"Fitted: qm={qm_default:.1f}, KL={KL_default:.3f}")
            else:
                qm_default, KL_default = 50.0, 0.1

            qm_min_range, qm_max_range = max(1.0, qm_default / 5), qm_default * 3
            KL_min_range, KL_max_range = max(0.001, KL_default / 10), KL_default * 10

            qm_range = st.slider(
                "qm range (mg/g)",
                qm_min_range,
                qm_max_range,
                (max(qm_min_range, qm_default * 0.5), min(qm_max_range, qm_default * 1.5)),
                key="param_qm_range",
            )
            KL_range = st.slider(
                "KL range (L/mg)",
                KL_min_range,
                KL_max_range,
                (max(KL_min_range, KL_default * 0.5), min(KL_max_range, KL_default * 1.5)),
                key="param_KL_range",
            )

            param1_range, param2_range = qm_range, KL_range
            p1_name, p2_name = "qm (mg/g)", "KL (L/mg)"
            model_func = langmuir_model

        elif model == "Freundlich":
            if "freundlich" in fitted_params:
                params = fitted_params["freundlich"]
                KF_default = float(params.get("KF", 20.0) or 20.0)

                n_inv_default = params.get("n_inv", None)
                n_default = params.get("n", None)

                # Robust fallback if only one of (n, n_inv) is available
                try:
                    if n_inv_default is None and n_default is not None and float(n_default) > 0:
                        n_inv_default = 1.0 / float(n_default)
                    if n_default is None and n_inv_default is not None and float(n_inv_default) > 0:
                        n_default = 1.0 / float(n_inv_default)
                except Exception:
                    n_inv_default, n_default = 0.5, 2.0

                if n_inv_default is None:
                    n_inv_default = 0.5
                if n_default is None:
                    n_default = 1.0 / float(n_inv_default) if float(n_inv_default) > 0 else 2.0

                st.success(
                    f"Fitted: KF={KF_default:.1f}, n={float(n_default):.2f} (1/n={float(n_inv_default):.3f})"
                )
            else:
                KF_default, n_inv_default = 20.0, 0.5

            KF_range = st.slider(
                "KF range",
                1.0,
                200.0,
                (max(1.0, KF_default * 0.3), KF_default * 1.7),
                key="param_KF_range",
            )
            n_inv_range = st.slider(
                "1/n range",
                0.1,
                1.5,
                (max(0.1, float(n_inv_default) * 0.5), float(n_inv_default) * 1.5),
                key="param_n_range",
            )
            param1_range, param2_range = KF_range, n_inv_range
            p1_name, p2_name = "KF", "1/n"
            model_func = freundlich_model



        else:  # Temkin
            if "temkin" in fitted_params:
                B1_default = fitted_params["temkin"].get("B1", 15.0)
                KT_default = fitted_params["temkin"].get("KT", 1.0)
                st.success(f"Fitted: B1={B1_default:.1f}, KT={KT_default:.2f}")
            else:
                B1_default, KT_default = 15.0, 1.0

            B1_range = st.slider(
                "B1 range (mg/g)",
                1.0,
                100.0,
                (max(1.0, B1_default * 0.3), B1_default * 1.7),
                key="param_B1_range",
            )
            KT_range = st.slider(
                "KT range (L/mg)",
                0.01,
                20.0,
                (max(0.01, KT_default * 0.1), KT_default * 10),
                key="param_KT_range",
            )
            param1_range, param2_range = B1_range, KT_range
            p1_name, p2_name = "B1 (mg/g)", "KT (L/mg)"
            model_func = temkin_model

    # Generate button
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_clicked = st.button(
            "🚀 Generate Visualization",
            type="primary",
            use_container_width=True,
            key="gen_param_space",
        )

    if not generate_clicked and "param_space_fig" not in st.session_state:
        st.caption("Configure parameters above, then click **Generate** to preview.")
        return

    try:
        with st.spinner("Generating parameter space..."):
            p1_grid, p2_grid, qe_grid = parameter_space_visualization(
                model_func, param1_range, param2_range, Ce_fixed
            )

            fig = go.Figure(
                data=[
                    go.Surface(
                        z=qe_grid,
                        x=p1_grid,
                        y=p2_grid,
                        colorscale="Plasma",
                        opacity=0.9,
                        name=f"{model} Response",
                        contours={"z": {"show": True, "usecolormap": True, "project": {"z": True}}},
                    )
                ]
            )

            # Mark fitted parameter point if available
            if model.lower() in fitted_params:
                if model == "Langmuir":
                    params = fitted_params["langmuir"]
                    p1_fit = params.get("qm")
                    p2_fit = params.get("KL")
                    if p1_fit and p2_fit:
                        qe_fit = langmuir_model(Ce_fixed, p1_fit, p2_fit)
                        fig.add_trace(
                            go.Scatter3d(
                                x=[p1_fit],
                                y=[p2_fit],
                                z=[qe_fit],
                                mode="markers",
                                marker={"size": 10, "color": COLORS["fit_secondary"], "symbol": "diamond", "line": {"width": 1.0, "color": "#000000"}},
                                name="Fitted Parameters",
                            )
                        )
                elif model == "Freundlich":
                    params = fitted_params["freundlich"]
                    p1_fit = params.get("KF")
                    n_inv_fit = params.get("n_inv", None)
                    if n_inv_fit is None:
                        n_val = params.get("n", 2.0)
                        n_inv_fit = 1.0 / float(n_val) if n_val else 0.5
                    p2_fit = float(n_inv_fit)
                    if p1_fit:
                        qe_fit = freundlich_model(Ce_fixed, p1_fit, p2_fit)
                        fig.add_trace(
                            go.Scatter3d(
                                x=[p1_fit],
                                y=[p2_fit],
                                z=[qe_fit],
                                mode="markers",
                                marker={"size": 10, "color": COLORS["fit_secondary"], "symbol": "diamond", "line": {"width": 1.0, "color": "#000000"}},
                                name="Fitted Parameters",
                            )
                        )

            title = f"{model} Parameter Space (Ce = {Ce_fixed:.1f} mg/L)"
            fig.update_layout(scene={
                "xaxis_title": p1_name,
                "yaxis_title": p2_name,
                "zaxis_title": "qe (mg/g)",
            })
            fig = apply_professional_3d_style(
                fig, title=title, height=700,
                camera_eye={"x": 1.8, "y": 1.8, "z": 1.2},
            )

            # Store for save
            st.session_state["param_space_fig"] = fig
            st.session_state["param_space_params"] = {
                "model": model,
                "Ce": Ce_fixed,
                p1_name: f"{param1_range[0]:.2f}-{param1_range[1]:.2f}",
                p2_name: f"{param2_range[0]:.4f}-{param2_range[1]:.4f}",
            }
            st.session_state["param_space_title"] = title

        st.plotly_chart(fig, use_container_width=True)

        # Save button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "💾 Save to Export Collection",
                type="secondary",
                use_container_width=True,
                key="save_param_space",
            ):
                if _save_figure(
                    f"{model.lower()}_param_space",
                    fig,
                    title,
                    st.session_state["param_space_params"],
                ):
                    st.success("✅ Figure saved to export collection!")
                    st.rerun()
                else:
                    st.error("Failed to save figure")

    except Exception as e:
        st.error(f"Error generating parameter space: {str(e)}")


def _render_model_comparison(fitted_params, exp_data):
    """3D comparison of multiple isotherm models."""
    st.markdown("### 📊 Model Comparison 3D")

    available_models = [m for m in ["langmuir", "freundlich", "temkin"] if m in fitted_params]

    if len(available_models) < 2:
        st.warning("Need at least 2 fitted isotherm models for comparison.")
        return

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        if "isotherm" in exp_data:
            Ce_max_exp = np.max(exp_data["isotherm"]["Ce"])
            Ce_max = st.slider(
                "Max Ce (mg/L)", 10.0, 500.0, float(Ce_max_exp * 1.2), 10.0, key="comp_Ce"
            )
        else:
            Ce_max = st.slider("Max Ce (mg/L)", 10.0, 500.0, 100.0, 10.0, key="comp_Ce")

    with col2:
        st.markdown("**Available Models:**")
        for m in available_models:
            st.success(f"✅ {m.title()}")

    # Generate button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_clicked = st.button(
            "🚀 Generate Comparison", type="primary", use_container_width=True, key="gen_comparison"
        )

    if not generate_clicked and "comparison_fig" not in st.session_state:
        st.caption("Click **Generate** to create the 3D model comparison.")
        return

    try:
        with st.spinner("Generating model comparison..."):
            Ce_line = np.linspace(0.1, Ce_max, 50)

            fig = go.Figure()
            colors = {"langmuir": "blue", "freundlich": "green", "temkin": "orange"}

            # Add experimental data points
            if "isotherm" in exp_data:
                fig.add_trace(
                    go.Scatter3d(
                        x=exp_data["isotherm"]["Ce"],
                        y=[0] * len(exp_data["isotherm"]["Ce"]),
                        z=exp_data["isotherm"]["qe"],
                        mode="markers",
                        marker={"size": 8, "color": "black", "symbol": "circle"},
                        name="Experimental Data",
                    )
                )

            # Add model surfaces
            for i, model_name in enumerate(available_models):
                params = fitted_params[model_name]

                if model_name == "langmuir":
                    qe_pred = langmuir_model(Ce_line, params["qm"], params["KL"])
                elif model_name == "freundlich":
                    n_inv = params.get("n_inv", 1 / params.get("n", 2))
                    qe_pred = freundlich_model(Ce_line, params["KF"], n_inv)
                elif model_name == "temkin":
                    qe_pred = temkin_model(Ce_line, params["B1"], params["KT"])
                else:
                    continue

                # Create a ribbon for each model (offset in Y)
                y_offset = i * 0.5
                fig.add_trace(
                    go.Scatter3d(
                        x=Ce_line,
                        y=[y_offset] * len(Ce_line),
                        z=qe_pred,
                        mode="lines",
                        line={"width": 6, "color": colors.get(model_name, "gray")},
                        name=model_name.title(),
                    )
                )

            title = f"Model Comparison: {', '.join([m.title() for m in available_models])}"
            fig.update_layout(scene={
                "xaxis_title": "Ce (mg/L)",
                "yaxis_title": "Model",
                "zaxis_title": "qe (mg/g)",
                "yaxis": {
                    "ticktext": [m.title() for m in available_models],
                    "tickvals": [i * 0.5 for i in range(len(available_models))],
                },
            })
            fig = apply_professional_3d_style(
                fig, title=title, height=700,
                camera_eye={"x": 1.5, "y": 1.5, "z": 1.2},
            )

            st.session_state["comparison_fig"] = fig
            st.session_state["comparison_params"] = {"models": available_models, "Ce_max": Ce_max}
            st.session_state["comparison_title"] = title

        st.plotly_chart(fig, use_container_width=True)

        # Save button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "💾 Save to Export Collection",
                type="secondary",
                use_container_width=True,
                key="save_comparison",
            ):
                if _save_figure(
                    "model_comparison", fig, title, st.session_state["comparison_params"]
                ):
                    st.success("✅ Figure saved to export collection!")
                    st.rerun()
                else:
                    st.error("Failed to save figure")

    except Exception as e:
        st.error(f"Error generating comparison: {str(e)}")


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

            fig.update_layout(scene={
                "xaxis_title": x_title,
                "yaxis_title": y_title,
                "zaxis_title": z_title,
            })
            fig = apply_professional_3d_style(
                fig, title=title, height=700,
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


def _render_ph_temp_response(fitted_params, exp_data):
    """pH-Temperature response surface."""
    st.markdown("### 🌡️ pH-Temperature Response Surface")

    st.info(
        "This surface is an **empirical visualization** built from your 1D pH and/or temperature-effect datasets (separable approximation). "
        "It is not a mechanistic prediction model and should be reported as exploratory visualization in a manuscript."
    )


    has_temp = "temperature" in exp_data
    has_ph = "ph_effect" in exp_data

    if not has_temp and not has_ph:
        st.warning("Need temperature or pH effect data for this visualization.")
        return

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        if has_ph:
            pH_range = (exp_data["ph_effect"]["pH"].min(), exp_data["ph_effect"]["pH"].max())
            st.info(f"pH range from data: {pH_range[0]:.1f} - {pH_range[1]:.1f}")
        else:
            pH_range = (2.0, 12.0)
            st.info("Using default pH range (no pH data)")

        pH_min = st.slider("Min pH", 1.0, 7.0, float(pH_range[0]), 0.5, key="ph_temp_pH_min")
        pH_max = st.slider("Max pH", 7.0, 14.0, float(pH_range[1]), 0.5, key="ph_temp_pH_max")

    with col2:
        if has_temp:
            T_range = (exp_data["temperature"]["T_C"].min(), exp_data["temperature"]["T_C"].max())
            st.info(f"Temperature range from data: {T_range[0]:.1f} - {T_range[1]:.1f} °C")
        else:
            T_range = (20.0, 60.0)
            st.info("Using default temperature range (no temp data)")

        T_min = st.slider("Min Temp (°C)", 10.0, 50.0, float(T_range[0]), 5.0, key="ph_temp_T_min")
        T_max = st.slider("Max Temp (°C)", 30.0, 100.0, float(T_range[1]), 5.0, key="ph_temp_T_max")

    # Generate button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_clicked = st.button(
            "🚀 Generate Response Surface",
            type="primary",
            use_container_width=True,
            key="gen_ph_temp",
        )

    if not generate_clicked and "ph_temp_fig" not in st.session_state:
        st.caption("Click **Generate** to create the response surface.")
        return

    try:
        with st.spinner("Generating response surface..."):
            # ---- Empirical separable surface from available 1D datasets ----
            def _agg_xy(x, y):
                x = np.asarray(x, dtype=float)
                y = np.asarray(y, dtype=float)
                m = np.isfinite(x) & np.isfinite(y)
                x, y = x[m], y[m]
                if x.size == 0:
                    return np.array([0.0, 1.0]), np.array([0.0, 0.0])
                ux = np.unique(x)
                y_mean = np.array([np.mean(y[x == v]) for v in ux], dtype=float)
                order = np.argsort(ux)
                return ux[order], y_mean[order]

            n_points = 25
            pH_vals = np.linspace(pH_min, pH_max, n_points)
            T_vals = np.linspace(T_min, T_max, n_points)
            pH_grid, T_grid = np.meshgrid(pH_vals, T_vals)

            # Scale surface by the maximum observed qe in the available datasets
            qe_candidates = []
            if has_ph:
                _, yph = _agg_xy(exp_data["ph_effect"]["pH"], exp_data["ph_effect"]["qe"])
                qe_candidates.append(np.nanmax(yph))
            if has_temp:
                _, yT = _agg_xy(exp_data["temperature"]["T_C"], exp_data["temperature"]["qe"])
                qe_candidates.append(np.nanmax(yT))
            max_qe = float(np.nanmax(qe_candidates)) if qe_candidates else 1.0
            if not np.isfinite(max_qe) or max_qe <= 0:
                max_qe = 1.0

            # Normalized 1D effects (clipped to keep the surface physical: qe ≥ 0)
            if has_ph:
                xph, yph = _agg_xy(exp_data["ph_effect"]["pH"], exp_data["ph_effect"]["qe"])
                f_ph = np.interp(pH_vals, xph, np.clip(yph / max_qe, 0, None))
            else:
                f_ph = np.ones_like(pH_vals)

            if has_temp:
                xT, yT = _agg_xy(exp_data["temperature"]["T_C"], exp_data["temperature"]["qe"])
                f_T = np.interp(T_vals, xT, np.clip(yT / max_qe, 0, None))
            else:
                f_T = np.ones_like(T_vals)

            qe_grid = max_qe * (f_T[:, None] * f_ph[None, :])

            fig = go.Figure(
                data=[
                    go.Surface(
                        z=qe_grid,
                        x=pH_grid,
                        y=T_grid,
                        colorscale="RdYlBu_r",
                        opacity=0.9,
                        name="Response Surface",
                        contours={"z": {"show": True, "usecolormap": True, "project": {"z": True}}},
                    )
                ]
            )

            # Add experimental points if available
            st.caption("To help interpretation, 1D points are shown as projections (pH points at mean temperature; temperature points at mean pH).")
            if has_ph and has_temp:
                # Show both datasets as projections on the surface:
                # - pH-effect data is plotted at the mean temperature
                # - temperature-effect data is plotted at the mean pH
                mean_T = float(np.mean(exp_data["temperature"]["T_C"]))
                mean_pH = float(np.mean(exp_data["ph_effect"]["pH"]))

                fig.add_trace(
                    go.Scatter3d(
                        x=exp_data["ph_effect"]["pH"],
                        y=[mean_T] * len(exp_data["ph_effect"]["pH"]),
                        z=exp_data["ph_effect"]["qe"],
                        mode="markers",
                        marker={"size": 8, "color": "black", "symbol": "diamond"},
                        name=f"pH Effect Data (proj. @ T={mean_T:.1f}°C)",
                    )
                )

                fig.add_trace(
                    go.Scatter3d(
                        x=[mean_pH] * len(exp_data["temperature"]["T_C"]),
                        y=exp_data["temperature"]["T_C"],
                        z=exp_data["temperature"]["qe"],
                        mode="markers",
                        marker={"size": 8, "color": "black", "symbol": "circle"},
                        name=f"Temperature Data (proj. @ pH={mean_pH:.1f})",
                    )
                )
            elif has_ph:
                fig.add_trace(
                    go.Scatter3d(
                        x=exp_data["ph_effect"]["pH"],
                        y=[np.mean([T_min, T_max])] * len(exp_data["ph_effect"]["pH"]),
                        z=exp_data["ph_effect"]["qe"],
                        mode="markers",
                        marker={"size": 8, "color": "black", "symbol": "diamond"},
                        name="pH Effect Data",
                    )
                )
            elif has_temp:
                fig.add_trace(
                    go.Scatter3d(
                        x=[np.mean([pH_min, pH_max])] * len(exp_data["temperature"]["T_C"]),
                        y=exp_data["temperature"]["T_C"],
                        z=exp_data["temperature"]["qe"],
                        mode="markers",
                        marker={"size": 8, "color": "black", "symbol": "diamond"},
                        name="Temperature Data",
                    )
                )

            title = f"pH-Temperature Response (pH {pH_min}-{pH_max}, T {T_min}-{T_max}°C)"
            fig.update_layout(scene={
                "xaxis_title": "pH",
                "yaxis_title": "Temperature (°C)",
                "zaxis_title": "qe (mg/g)",
            })
            fig = apply_professional_3d_style(
                fig, title=title, height=700,
                camera_eye={"x": 1.8, "y": 1.8, "z": 1.2},
            )

            st.session_state["ph_temp_fig"] = fig
            st.session_state["ph_temp_params"] = {
                "pH_range": f"{pH_min}-{pH_max}",
                "T_range": f"{T_min}-{T_max}°C",
            }
            st.session_state["ph_temp_title"] = title

        st.plotly_chart(fig, use_container_width=True)

        # Save button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "💾 Save to Export Collection",
                type="secondary",
                use_container_width=True,
                key="save_ph_temp",
            ):
                if _save_figure("ph_temp_response", fig, title, st.session_state["ph_temp_params"]):
                    st.success("✅ Figure saved to export collection!")
                    st.rerun()
                else:
                    st.error("Failed to save figure")

    except Exception as e:
        st.error(f"Error generating response surface: {str(e)}")


def _render_kinetic_3d(fitted_params, exp_data):
    """3D kinetic visualization with multi-concentration."""
    st.markdown("### ⏱️ Kinetic Multi-Concentration 3D")

    if "kinetic" not in exp_data or "pso" not in fitted_params:
        st.warning("Need kinetic data and PSO fit for this visualization.")
        return

    pso_params = fitted_params["pso"]
    qe_fit = pso_params.get("qe", 50)
    k2_fit = pso_params.get("k2", 0.01)

    st.info(f"Using fitted PSO: qe={qe_fit:.2f} mg/g, k2={k2_fit:.4f} g/(mg·min)")

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        t_max = st.slider(
            "Max Time (min)",
            60,
            1440,
            int(exp_data["kinetic"]["t"].max() * 1.2),
            30,
            key="kin_t_max",
        )

    with col2:
        C0_range = st.slider(
            "Initial Concentration Range (mg/L)",
            10.0,
            500.0,
            (50.0, 200.0),
            10.0,
            key="kin_C0_range",
        )

    # Generate button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_clicked = st.button(
            "🚀 Generate Kinetic 3D", type="primary", use_container_width=True, key="gen_kinetic_3d"
        )

    if not generate_clicked and "kinetic_3d_fig" not in st.session_state:
        st.caption("Click **Generate** to create the 3D kinetic visualization.")
        return

    try:
        with st.spinner("Generating kinetic 3D surface..."):
            t_line = np.linspace(0.1, t_max, 50)
            C0_line = np.linspace(C0_range[0], C0_range[1], 20)
            T_grid, C0_grid = np.meshgrid(t_line, C0_line)

            # Simple PSO model (not concentration-dependent, but scaled)
            # For visualization, scale qe with C0
            qe_scaled = qe_fit * (C0_grid / 100)  # Rough scaling
            qt_grid = np.zeros_like(T_grid)

            for i in range(qt_grid.shape[0]):
                for j in range(qt_grid.shape[1]):
                    qe_local = qe_scaled[i, j]
                    t_val = T_grid[i, j]
                    qt_grid[i, j] = pso_model(t_val, qe_local, k2_fit)

            fig = go.Figure(
                data=[
                    go.Surface(
                        z=qt_grid,
                        x=T_grid,
                        y=C0_grid,
                        colorscale="Viridis",
                        opacity=0.85,
                        name="PSO Model",
                        contours={"z": {"show": True, "usecolormap": True, "project": {"z": True}}},
                    )
                ]
            )

            # Add experimental points
            kin_data = exp_data["kinetic"]
            fig.add_trace(
                go.Scatter3d(
                    x=kin_data["t"],
                    y=[100] * len(kin_data["t"]),  # Assume C0=100 for exp data
                    z=kin_data["qt"],
                    mode="markers",
                    marker={"size": 6, "color": COLORS["experimental"], "symbol": "diamond", "line": {"width": 1.0, "color": "#000000"}},
                    name="Experimental Data",
                )
            )

            title = f"PSO Kinetics: qe={qe_fit:.1f}, k2={k2_fit:.4f}"
            fig.update_layout(scene={
                "xaxis_title": "Time (min)",
                "yaxis_title": "C₀ (mg/L)",
                "zaxis_title": "qt (mg/g)",
            })
            fig = apply_professional_3d_style(
                fig, title=title, height=700,
                camera_eye={"x": 1.8, "y": 1.8, "z": 1.2},
            )

            st.session_state["kinetic_3d_fig"] = fig
            st.session_state["kinetic_3d_params"] = {
                "qe": qe_fit,
                "k2": k2_fit,
                "t_max": t_max,
                "C0_range": f"{C0_range[0]}-{C0_range[1]}",
            }
            st.session_state["kinetic_3d_title"] = title

        st.plotly_chart(fig, use_container_width=True)

        # Save button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "💾 Save to Export Collection",
                type="secondary",
                use_container_width=True,
                key="save_kinetic_3d",
            ):
                if _save_figure(
                    "kinetic_multi_c", fig, title, st.session_state["kinetic_3d_params"]
                ):
                    st.success("✅ Figure saved to export collection!")
                    st.rerun()
                else:
                    st.error("Failed to save figure")

    except Exception as e:
        st.error(f"Error generating kinetic 3D: {str(e)}")


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

            R_GAS_CONSTANT = 8.314
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
            fig.update_layout(scene={
                "xaxis_title": "Ce (mg/L)",
                "yaxis_title": "Temperature (°C)",
                "zaxis_title": "Residual (mg/g)",
            })
            fig = apply_professional_3d_style(
                fig, title=title, height=700,
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