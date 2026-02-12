# tabs/competitive_tab.py
"""
Multi-Component Competitive Adsorption Tab - AdsorbLab Pro
==========================================================

Analyzes competitive adsorption in multi-component systems.

Features:
- Extended Langmuir model for competitive adsorption
- Extended Freundlich model (SRS equation)
- Selectivity coefficient calculation
- Multi-component equilibrium prediction
- Study-linked parameter selection
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adsorblab_pro.streamlit_compat import st

from ..config import FONT_FAMILY, MULTICOMPONENT_GUIDANCE, MULTICOMPONENT_MODELS
from ..models import (
    calculate_selectivity_coefficient,
    extended_freundlich_multicomponent,
    extended_langmuir_multicomponent,
)
from ..plot_style import COLORS, apply_professional_style, get_axis_style
from ..utils import get_current_study_state


def render():
    """Render the Multi-Component Competitive Adsorption tab."""
    st.subheader("🔄 Multi-Component Competitive Adsorption")

    st.markdown("""
    Analyze competitive adsorption in multi-component systems using extended isotherm models.
    These models predict how adsorbates compete for surface sites when multiple species are present.
    """)

    # Get all studies
    all_studies = st.session_state.get("studies", {})

    if not all_studies:
        st.info("Please create studies from the sidebar to begin analysis.")
        _display_theory()
        return

    # Find studies with fitted isotherms
    studies_with_isotherms = _get_studies_with_isotherms(all_studies)

    # Mode selection
    mode = st.radio(
        "Parameter Source",
        ["📊 From Studies (Recommended)", "✏️ Manual Entry"],
        help="Select whether to use parameters from your fitted studies or enter values manually",
    )

    st.markdown("---")

    if mode == "📊 From Studies (Recommended)":
        _render_study_linked_mode(studies_with_isotherms)
    else:
        _render_manual_entry_mode(studies_with_isotherms)

    # Theory section at bottom
    _display_theory()


def _get_studies_with_isotherms(all_studies: dict) -> dict:
    """Get studies that have fitted Langmuir or Freundlich isotherms."""
    studies_with_isotherms = {}

    for name, data in all_studies.items():
        iso_fitted = data.get("isotherm_models_fitted", {})
        langmuir = iso_fitted.get("Langmuir", {})
        freundlich = iso_fitted.get("Freundlich", {})

        has_langmuir = langmuir.get("converged", False)
        has_freundlich = freundlich.get("converged", False)

        if has_langmuir or has_freundlich:
            studies_with_isotherms[name] = {
                "data": data,
                "has_langmuir": has_langmuir,
                "has_freundlich": has_freundlich,
                "langmuir_params": langmuir.get("params", {}) if has_langmuir else {},
                "freundlich_params": freundlich.get("params", {}) if has_freundlich else {},
            }

    return studies_with_isotherms


def _render_study_linked_mode(studies_with_isotherms: dict):
    """Render the study-linked parameter selection mode."""

    if len(studies_with_isotherms) < 2:
        st.warning(f"""
        ⚠️ **Need at least 2 studies with fitted isotherms**

        Currently available: **{len(studies_with_isotherms)}** study/studies with isotherms

        Multi-component analysis requires single-component isotherm parameters
        for each competing adsorbate. Please:

        1. Create a separate study for each adsorbate (e.g., "MB-Zeolite", "CR-Zeolite")
        2. Enter equilibrium data and fit isotherm models for each
        3. Return here to analyze competition between them

        **Tip:** Each study should represent one adsorbate on the same adsorbent material.
        """)

        # Show available studies
        if studies_with_isotherms:
            st.markdown("**Available studies with isotherms:**")
            for name, info in studies_with_isotherms.items():
                models = []
                if info["has_langmuir"]:
                    models.append("Langmuir")
                if info["has_freundlich"]:
                    models.append("Freundlich")
                st.write(f"• {name}: {', '.join(models)}")

        return

    st.markdown("### 📊 Select Competing Adsorbates from Your Studies")
    st.caption(
        "Each study represents a single-component isotherm experiment. Select which adsorbates compete in your system."
    )

    st.info(
        "Important: Ce values must be the equilibrium concentrations in the *mixture* (Ceᵢ), not single-component Ce curves. "
        "This module estimates qeᵢ at the specified mixture Ceᵢ using extended Langmuir / extended Freundlich."
    )

    # ---- Number of competing adsorbates (guard against invalid slider ranges) ----
    total_adsorbates = len(studies_with_isotherms)

    if total_adsorbates < 2:
        st.warning(
            "You need at least 2 adsorbates (with isotherm data) to run competitive analysis."
        )
        return
    # Number of components
    max_components = min(5, len(studies_with_isotherms))

    if max_components == 2:
        st.info("Exactly 2 studies with fitted isotherms found — comparing both components.")
        n_components = 2
    else:
        n_components = st.slider(
            "Number of competing adsorbates",
            min_value=2,
            max_value=max_components,
            value=2,
            help="Select how many adsorbates compete for surface sites",
        )

    study_names = list(studies_with_isotherms.keys())
    components = []
    selected_studies = []

    st.markdown("---")

    for i in range(n_components):
        st.markdown(f"#### Component {i + 1}")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # Default to different study for each component if possible
            default_idx = i if i < len(study_names) else 0

            selected_study = st.selectbox(
                "Select study",
                study_names,
                index=default_idx,
                key=f"study_select_{i}",
                help="Choose the study containing single-component isotherm data for this adsorbate",
            )

            # Check for duplicate selection
            if selected_study in selected_studies:
                st.warning(
                    f"⚠️ '{selected_study}' already selected. Consider using different studies."
                )
            selected_studies.append(selected_study)

            # Get study info
            study_info = studies_with_isotherms[selected_study]

        with col2:
            Ce = st.number_input(
                "Ce (mg/L)",
                min_value=0.0,
                value=50.0 if i == 0 else 30.0,
                step=5.0,
                key=f"Ce_study_{i}",
                help="Equilibrium concentration of this component in the mixture",
            )

        with col3:
            st.markdown("&nbsp;")  # Spacer
            st.markdown("&nbsp;")

        # Display auto-filled parameters
        param_col1, param_col2 = st.columns(2)

        with param_col1:
            if study_info["has_langmuir"]:
                lang_p = study_info["langmuir_params"]
                qm = lang_p.get("qm", 0)
                KL = lang_p.get("KL", 0)
                st.success(f"**Langmuir:** qm = {qm:.2f} mg/g, KL = {KL:.4f} L/mg")
            else:
                st.info("Langmuir not fitted for this study")
                qm, KL = None, None

        with param_col2:
            if study_info["has_freundlich"]:
                freund_p = study_info["freundlich_params"]
                KF = freund_p.get("KF", 0)
                n_val = 0.0
                try:
                    if "n_inv" in freund_p and freund_p.get("n_inv") is not None:
                        n_inv = float(freund_p.get("n_inv"))
                        n_val = (1.0 / n_inv) if n_inv > 0 else 0.0
                    elif "n" in freund_p and freund_p.get("n") is not None:
                        n_val = float(freund_p.get("n"))
                except Exception:
                    n_val = 0.0
                st.success(f"**Freundlich:** KF = {KF:.4f}, n = {n_val:.2f}")
            else:
                st.info("Freundlich not fitted for this study")
                KF, n_val = None, None

        components.append(
            {
                "name": selected_study,
                "Ce": Ce,
                "qm": qm if qm else 0,
                "KL": KL if KL else 0,
                "KF": KF if KF else 0,
                "n": n_val if n_val else 1,
                "has_langmuir": study_info["has_langmuir"],
                "has_freundlich": study_info["has_freundlich"],
            }
        )

        st.markdown("---")

    # Calculate button
    if st.button("🧮 Calculate Competitive Adsorption", type="primary", key="calc_study"):
        # Check if all components have required parameters
        can_calculate_langmuir = all(c["has_langmuir"] for c in components)
        can_calculate_freundlich = all(c["has_freundlich"] for c in components)

        if not can_calculate_langmuir and not can_calculate_freundlich:
            st.error(
                "❌ Cannot calculate: No common isotherm model fitted across all selected studies."
            )
            return

        if any(c["Ce"] <= 0 for c in components):
            st.error(
                "❌ Please enter Ce values > 0 for all components (mixture equilibrium concentrations)."
            )
            return

        _calculate_and_display(components, can_calculate_langmuir, can_calculate_freundlich)


def _render_manual_entry_mode(studies_with_isotherms: dict):
    """Render manual entry mode for literature values."""

    st.markdown("### ✏️ Manual Parameter Entry")
    st.caption(
        "Enter isotherm parameters from literature or other sources. Component 1 can use your fitted values."
    )

    # Check if current study has isotherms for defaults
    current_study_state = get_current_study_state()
    has_langmuir = False
    has_freundlich = False
    lang_params = {}
    freund_params = {}

    if current_study_state:
        iso_fitted = current_study_state.get("isotherm_models_fitted", {})
        langmuir_fitted = iso_fitted.get("Langmuir", {})
        freundlich_fitted = iso_fitted.get("Freundlich", {})
        has_langmuir = langmuir_fitted.get("converged", False)
        has_freundlich = freundlich_fitted.get("converged", False)
        if has_langmuir:
            lang_params = langmuir_fitted.get("params", {})
        if has_freundlich:
            freund_params = freundlich_fitted.get("params", {})

    n_components = st.slider(
        "Number of competing adsorbates", min_value=2, max_value=5, value=2, key="n_comp_manual"
    )

    components = []

    for i in range(n_components):
        with st.expander(f"**Component {i + 1}**", expanded=(i < 2)):
            # Show source indicator
            if i == 0 and (has_langmuir or has_freundlich):
                st.info("💡 Default values from your active study's fitted isotherms")
            else:
                st.warning("⚠️ Enter values from literature or your experiments")

            col1, col2, col3 = st.columns(3)

            # Set defaults
            if i == 0 and has_langmuir:
                default_qm = max(1e-6, lang_params.get("qm", 100))
                default_KL = max(1e-10, lang_params.get("KL", 0.1))
            else:
                default_qm = 0.0  # Force user to enter
                default_KL = 0.0

            if i == 0 and has_freundlich:
                default_KF = max(1e-10, freund_params.get("KF", 10))
                n_inv_val = freund_params.get("n_inv", freund_params.get("n", 0.5))
                default_n = max(0.01, 1 / n_inv_val if n_inv_val and n_inv_val > 0 else 2)
            else:
                default_KF = 0.0
                default_n = 1.0

            with col1:
                name = st.text_input("Name", value=f"Adsorbate {i + 1}", key=f"manual_name_{i}")
                Ce = st.number_input(
                    "Ce (mg/L)",
                    min_value=0.0,
                    value=50.0 if i == 0 else 0.0,
                    key=f"manual_Ce_{i}",
                    help="Equilibrium concentration of this component in the mixture (Ceᵢ,mg/L).",
                )

            with col2:
                qm = st.number_input(
                    "qm (mg/g)",
                    min_value=0.0,
                    value=float(default_qm),
                    format="%.4g",
                    key=f"manual_qm_{i}",
                    help="Maximum adsorption capacity from Langmuir model",
                )
                KL = st.number_input(
                    "KL (L/mg)",
                    min_value=0.0,
                    value=float(default_KL),
                    format="%.4g",
                    key=f"manual_KL_{i}",
                    help="Langmuir affinity constant",
                )

            with col3:
                KF = st.number_input(
                    "KF",
                    min_value=0.0,
                    value=float(default_KF),
                    format="%.6g",
                    key=f"manual_KF_{i}",
                    help="Freundlich capacity constant",
                )
                n = st.number_input(
                    "n",
                    min_value=0.01,
                    value=float(default_n),
                    format="%.4f",
                    key=f"manual_n_{i}",
                    help="Freundlich heterogeneity factor (n > 1 favorable)",
                )

            # Validation
            if i > 0 and qm == 0 and KL == 0 and KF == 0:
                st.error("⚠️ Please enter valid parameters from literature or experiments")

            components.append(
                {
                    "name": name,
                    "Ce": Ce,
                    "qm": qm,
                    "KL": KL,
                    "KF": KF,
                    "n": n,
                    "has_langmuir": qm > 0 and KL > 0,
                    "has_freundlich": KF > 0 and n > 0,
                }
            )

    st.markdown("---")

    # Calculate button
    if st.button("🧮 Calculate Competitive Adsorption", type="primary", key="calc_manual"):
        # Validate
        valid_langmuir = all(c["qm"] > 0 and c["KL"] > 0 for c in components)
        valid_freundlich = all(c["KF"] > 0 and c["n"] > 0 for c in components)
        valid_Ce = all(c["Ce"] > 0 for c in components)

        if not valid_Ce:
            st.error("❌ Please enter Ce values > 0 for all components")
            return

        if not valid_langmuir and not valid_freundlich:
            st.error(
                "❌ Please enter valid Langmuir (qm, KL) or Freundlich (KF, n) parameters for all components"
            )
            return

        _calculate_and_display(components, valid_langmuir, valid_freundlich)


def _calculate_and_display(components, has_langmuir, has_freundlich):
    """Calculate and display multi-component results."""

    st.markdown("## 📈 Results")

    n = len(components)
    names = [c["name"] for c in components]
    Ce_all = np.array([c["Ce"] for c in components])

    results_data = []
    qe_lang = []
    qe_freund = []

    # Extended Langmuir
    if has_langmuir:
        st.markdown(
            f"### Extended Langmuir ({MULTICOMPONENT_MODELS['Extended-Langmuir']['description']})"
        )
        st.latex(r"q_{e,i} = \frac{q_{m,i} K_{L,i} C_{e,i}}{1 + \sum_{j=1}^{n} K_{L,j} C_{e,j}}")

        KL_all = np.array([c["KL"] for c in components])

        for i, comp in enumerate(components):
            qe_i = extended_langmuir_multicomponent(
                Ce_all[i], comp["qm"], comp["KL"], Ce_all, KL_all
            )
            qe_lang.append(float(qe_i))

            # Single-component qe for comparison
            qe_single = comp["qm"] * comp["KL"] * Ce_all[i] / (1 + comp["KL"] * Ce_all[i])
            reduction = (1 - qe_i / qe_single) * 100 if qe_single > 0 else 0

            results_data.append(
                {
                    "Component": comp["name"],
                    "Ce (mg/L)": Ce_all[i],
                    "qe_single (mg/g)": qe_single,
                    "qe_competitive (mg/g)": float(qe_i),
                    "Reduction (%)": float(reduction),
                    "Model": "Extended Langmuir",
                }
            )

        # Display results table
        lang_df = pd.DataFrame([r for r in results_data if r["Model"] == "Extended Langmuir"])
        st.dataframe(
            lang_df.style.format(
                {
                    "Ce (mg/L)": "{:.2f}",
                    "qe_single (mg/g)": "{:.2f}",
                    "qe_competitive (mg/g)": "{:.2f}",
                    "Reduction (%)": "{:.1f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Selectivity coefficients
        if n >= 2:
            st.markdown("#### Selectivity Coefficients")
            st.latex(r"\alpha_{i/j} = \frac{q_{e,i} / C_{e,i}}{q_{e,j} / C_{e,j}}")

            selectivity_data = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        alpha = calculate_selectivity_coefficient(
                            qe_lang[i], Ce_all[i], qe_lang[j], Ce_all[j]
                        )

                        if np.isnan(alpha):
                            interp = "N/A (Ce=0 or qe≤0)"
                            alpha_out = np.nan
                        elif alpha > 1:
                            interp = f"✅ {names[i]} preferred"
                            alpha_out = float(alpha)
                        elif alpha < 1:
                            interp = f"⬇️ {names[j]} preferred"
                            alpha_out = float(alpha)
                        else:
                            interp = "= Equal preference"
                            alpha_out = float(alpha)

                        selectivity_data.append(
                            {
                                "Pair": f"{names[i]} / {names[j]}",
                                "α": alpha_out,
                                "Interpretation": interp,
                            }
                        )

            sel_df = pd.DataFrame(selectivity_data)
            st.dataframe(
                sel_df.style.format({"α": "{:.3f}"}), use_container_width=True, hide_index=True
            )

        st.markdown("---")

    # Extended Freundlich
    if has_freundlich:
        st.markdown(
            f"### Extended Freundlich ({MULTICOMPONENT_MODELS['Extended-Freundlich']['description']})"
        )
        st.latex(
            r"q_{e,i} = K_{F,i} \cdot C_{e,i} \cdot \left(\sum_{j=1}^{n} a_{ij}\, C_{e,j}\right)^{(1/n_i - 1)}"
        )
        st.latex(r"a_{ij} = \left(\frac{K_{F,i}}{K_{F,j}}\right)^{(n_j/n_i)}")
        st.caption(
            "Sheindorf–Rebhun–Sheintuch (SRS) extended Freundlich with competition coefficients aᵢⱼ computed from KF and n."
        )

        KF_all = np.array([c["KF"] for c in components])
        n_all = np.array([c["n"] for c in components])

        for i, comp in enumerate(components):
            qe_i = extended_freundlich_multicomponent(
                Ce_all[i], comp["KF"], comp["n"], Ce_all, KF_all, n_all
            )
            qe_freund.append(float(qe_i))

            # Single-component qe
            qe_single = comp["KF"] * (Ce_all[i] ** (1 / comp["n"]))
            reduction = (1 - qe_i / qe_single) * 100 if qe_single > 0 else 0

            results_data.append(
                {
                    "Component": comp["name"],
                    "Ce (mg/L)": Ce_all[i],
                    "qe_single (mg/g)": qe_single,
                    "qe_competitive (mg/g)": float(qe_i),
                    "Reduction (%)": float(reduction),
                    "Model": "Extended Freundlich",
                }
            )

        freund_df = pd.DataFrame([r for r in results_data if r["Model"] == "Extended Freundlich"])
        st.dataframe(
            freund_df.style.format(
                {
                    "Ce (mg/L)": "{:.2f}",
                    "qe_single (mg/g)": "{:.2f}",
                    "qe_competitive (mg/g)": "{:.2f}",
                    "Reduction (%)": "{:.1f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

    # Visualization
    st.markdown("### 📊 Visualization")

    _create_visualization(results_data, names, has_langmuir, has_freundlich)

    # Summary interpretation
    _display_interpretation(results_data, names, has_langmuir)


def _create_visualization(results_data, names, has_langmuir, has_freundlich):
    """Create visualization charts."""

    colors = COLORS["model_colors"]  # Use consistent colors from plot_style

    if has_langmuir:
        lang_results = [r for r in results_data if r["Model"] == "Extended Langmuir"]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Single vs Competitive Adsorption", "Capacity Reduction (%)"),
        )

        # Bar chart: Single vs Competitive
        for i, r in enumerate(lang_results):
            color = colors[i % len(colors)]

            fig.add_trace(
                go.Bar(
                    name=f"{r['Component']} (single)",
                    x=[r["Component"]],
                    y=[r["qe_single (mg/g)"]],
                    marker_color=color,
                    marker_line_color="black",
                    marker_line_width=1,
                    opacity=0.5,
                    showlegend=True,
                    legendgroup=r["Component"],
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Bar(
                    name=f"{r['Component']} (competitive)",
                    x=[r["Component"]],
                    y=[r["qe_competitive (mg/g)"]],
                    marker_color=color,
                    marker_line_color="black",
                    marker_line_width=1,
                    opacity=1.0,
                    showlegend=True,
                    legendgroup=r["Component"],
                ),
                row=1,
                col=1,
            )

        # Bar chart: Reduction %
        fig.add_trace(
            go.Bar(
                name="Reduction",
                x=[r["Component"] for r in lang_results],
                y=[r["Reduction (%)"] for r in lang_results],
                marker_color=[colors[i % len(colors)] for i in range(len(lang_results))],
                marker_line_color="black",
                marker_line_width=1,
                text=[f"{r['Reduction (%)']:.1f}%" for r in lang_results],
                textposition="outside",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Apply centralized house style (global layout + col-1 axes + bar normalization)
        fig = apply_professional_style(
            fig,
            title="Extended Langmuir: Competition Effect",
            y_title="qe (mg/g)",
            height=450,
            legend_horizontal=True,
            barmode="group",
        )

        # Style col-2 axes via centralized helper
        fig.update_xaxes(**get_axis_style(""), row=1, col=2)
        fig.update_yaxes(**get_axis_style("Reduction (%)"), row=1, col=2)

        # Move horizontal legend below subplots (better for this layout)
        fig.update_layout(
            legend={"y": -0.3, "yanchor": "top"},
            margin={"b": 80},
        )

        # Match subplot title annotations to house style (bold + house font)
        for ann in fig.layout.annotations:
            ann.update(
                text=f"<b>{ann.text}</b>",
                font={"size": 14, "family": FONT_FAMILY},
            )

        st.plotly_chart(fig, use_container_width=True)


def _display_interpretation(results_data, names, has_langmuir):
    """Display interpretation of results."""

    if not has_langmuir:
        return

    lang_results = [r for r in results_data if r["Model"] == "Extended Langmuir"]

    st.markdown("### 🔬 Interpretation")

    # Find most/least affected
    max_reduction = max(lang_results, key=lambda x: x["Reduction (%)"])
    min_reduction = min(lang_results, key=lambda x: x["Reduction (%)"])
    avg_reduction = np.mean([r["Reduction (%)"] for r in lang_results])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Most Affected", max_reduction["Component"], f"-{max_reduction['Reduction (%)']:.1f}%"
        )

    with col2:
        st.metric(
            "Least Affected", min_reduction["Component"], f"-{min_reduction['Reduction (%)']:.1f}%"
        )

    with col3:
        st.metric("Average Reduction", f"{avg_reduction:.1f}%")

    # Recommendations
    st.markdown("#### 💡 Practical Implications")

    if avg_reduction > 30:
        st.warning(f"""
        **Strong competition effect** (avg. {avg_reduction:.0f}% reduction)

        - Significantly more adsorbent needed for mixed waste treatment
        - Consider pre-treatment to remove competing species
        - May need selective adsorbents for target pollutant
        """)
    elif avg_reduction > 10:
        st.info(f"""
        **Moderate competition effect** (avg. {avg_reduction:.0f}% reduction)

        - Account for ~{avg_reduction:.0f}% capacity loss in process design
        - Single-component isotherms provide reasonable estimates with correction factor
        """)
    else:
        st.success(f"""
        **Low competition effect** (avg. {avg_reduction:.0f}% reduction)

        - Single-component isotherm parameters can be used with minor adjustment
        - Competition effects are minimal for this system
        """)


def _display_theory():
    """Display theory section using centralized config."""

    with st.expander("📖 Theory: Multi-Component Competitive Adsorption", expanded=False):
        st.markdown("""
        ## Overview

        In real wastewater systems, multiple pollutants compete for adsorption sites.
        Extended isotherm models predict this competitive behavior using single-component parameters.

        ---
        """)

        # Extended Langmuir from config
        lang_config = MULTICOMPONENT_MODELS["Extended-Langmuir"]
        st.markdown(f"### Extended Langmuir ({lang_config['description']})")

        st.markdown("**Assumptions:**")
        for assumption in lang_config["assumptions"]:
            st.markdown(f"- {assumption}")

        st.markdown("**Equation:**")
        st.latex(lang_config["formula"])

        st.markdown("""
        **When to use:**
        - Homogeneous surfaces (activated carbon, zeolites)
        - Similar-sized adsorbates
        - When single-component Langmuir fits well (R² > 0.95)
        """)

        st.caption(f"*Reference: {lang_config['reference']}*")

        st.markdown("---")

        # Extended Freundlich from config
        freund_config = MULTICOMPONENT_MODELS["Extended-Freundlich"]
        st.markdown(f"### Extended Freundlich ({freund_config['description']})")

        st.markdown("**Assumptions:**")
        for assumption in freund_config["assumptions"]:
            st.markdown(f"- {assumption}")

        st.markdown("**Equation:**")
        st.latex(freund_config["formula"])

        st.markdown("""
        **When to use:**
        - Heterogeneous surfaces
        - When single-component Freundlich fits well
        - Industrial wastewater with multiple pollutants
        """)

        st.caption(f"*Reference: {freund_config['reference']}*")

        st.markdown("---")

        # Selectivity (not in config, keep as-is)
        st.markdown("### Selectivity Coefficient")
        st.markdown(
            "The selectivity coefficient indicates relative preference for one adsorbate over another:"
        )
        st.latex(
            r"\alpha_{i/j} = \frac{q_{e,i} / C_{e,i}}{q_{e,j} / C_{e,j}} = \frac{K_{d,i}}{K_{d,j}}"
        )

        st.markdown("""
        **Interpretation:**
        | Value | Meaning |
        |-------|---------|
        | α >> 1 | Strong preference for component i |
        | α > 1 | Preference for component i |
        | α = 1 | No selectivity (equal preference) |
        | α < 1 | Preference for component j |
        | α << 1 | Strong preference for component j |
        """)

        st.markdown("---")

        # When to use from MULTICOMPONENT_GUIDANCE
        st.markdown("### When to Use Multi-Component Models")
        for item in MULTICOMPONENT_GUIDANCE["when_to_use"]:
            st.markdown(f"- {item}")

        st.markdown("---")

        # Limitations from MULTICOMPONENT_GUIDANCE
        st.markdown("### Limitations")
        for item in MULTICOMPONENT_GUIDANCE["limitations"]:
            st.markdown(f"- {item}")

        st.markdown("---")

        # Alternatives from MULTICOMPONENT_GUIDANCE
        st.markdown("### Alternatives for Rigorous Analysis")
        for item in MULTICOMPONENT_GUIDANCE["alternatives"]:
            st.markdown(f"- {item}")

        st.markdown("---")

        st.markdown("""
        ### Workflow

        1. **Perform single-component experiments** for each adsorbate separately
        2. **Fit isotherm models** (Langmuir, Freundlich) to get qm, KL, KF, n
        3. **Apply extended models** to predict competitive behavior
        4. **Validate** with actual multi-component experiments (optional but recommended)
        """)
