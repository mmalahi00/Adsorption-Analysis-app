# tabs/comparison_tab.py
"""
Multi-Study Comparison Tab - AdsorbLab Pro
==========================================

Comprehensive comparison of multiple adsorption studies.

Features:
- Isotherm parameter comparison (all models)
- Kinetic parameter comparison (all models)
- Thermodynamic parameter comparison
- Effect studies comparison (pH, temperature, dosage)
- Direct descriptor tables and matched-condition overlays
"""

import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from adsorblab_pro.streamlit_compat import st

from ..plot_style import (
    apply_professional_style,
    get_study_color,
    style_study_trace,
)
from ..utils import (
    APPARENT_THERMO_NOTE,
    CAPACITY_CRITERION,
    CAPACITY_CRITERION_NOTE,
    apparent_delta_g,
    capacity_comparison,
    compare_information_criteria,
    display_results_table,
    eligible_observations,
    kd_definition,
    sign_label,
)


def _finite(value):
    """Float value, or NaN when missing/non-finite (shown as '—')."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _usable(results):
    """Quantified rows of a stored results table (excluded rows are not plotted)."""
    if results is None:
        return None
    usable = eligible_observations(results)
    return usable if not usable.empty else None


def style_dataframe(df, format_dict=None, highlight_max_cols=None, highlight_min_cols=None):
    """
    Safely style a dataframe without requiring matplotlib.
    Uses highlight_max/highlight_min instead of background_gradient.

    Args:
        df: DataFrame to style
        format_dict: Dict of column: format_string for number formatting
        highlight_max_cols: List of columns where max values should be highlighted (green)
        highlight_min_cols: List of columns where min values should be highlighted (green)

    Returns:
        Styled DataFrame
    """
    styler = df.style

    if format_dict:
        styler = styler.format(format_dict, na_rep="—")

    if highlight_max_cols:
        for col in highlight_max_cols:
            if col in df.columns:
                styler = styler.highlight_max(subset=[col], color="#90EE90")  # lightgreen

    if highlight_min_cols:
        for col in highlight_min_cols:
            if col in df.columns:
                styler = styler.highlight_min(subset=[col], color="#90EE90")  # lightgreen

    return styler


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================
def render():
    """Render comprehensive multi-study comparison."""
    st.subheader("🆚 Multi-Study Comparison")
    st.markdown("*Comprehensive comparison of multiple studies across all analyses.*")

    # Check if studies exist
    if not st.session_state.get("studies"):
        st.info("No studies available to compare. Please add and analyze studies first.")
        _display_setup_guide()
        return

    studies_data = st.session_state.studies
    study_names = list(studies_data.keys())

    # Minimum 2 studies for comparison
    if len(study_names) < 2:
        st.warning("⚠️ Add at least 2 studies to enable comparison features.")
        st.info(f"Currently you have **{len(study_names)}** study: {', '.join(study_names)}")
        return

    # Summary metrics
    _render_summary_metrics(studies_data, study_names)

    st.warning(
        "Compare capacities and rate constants only when sorbate identity, units, initial "
        "concentration range, pH, temperature, dosage, particle size, and contact conditions "
        "are matched. AdsorbLab does not yet validate this metadata, so it does not produce an "
        "automatic material ranking."
    )

    st.markdown("---")

    # Create tabs for different comparison categories
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Isotherms",
            "⏱️ Kinetics",
            "🌡️ Thermodynamics",
            "🔬 Effect Studies",
        ]
    )

    with tab1:
        _render_isotherm_comparison(studies_data, study_names)

    with tab2:
        _render_kinetic_comparison(studies_data, study_names)

    with tab3:
        _render_thermodynamic_comparison(studies_data, study_names)

    with tab4:
        _render_effect_studies_comparison(studies_data, study_names)


# =============================================================================
# SUMMARY METRICS
# =============================================================================
def _render_summary_metrics(studies_data: dict, study_names: list):
    """Render summary metrics for all studies."""
    st.markdown("### 📋 Studies Overview")

    cols = st.columns(len(study_names))

    for i, name in enumerate(study_names):
        data = studies_data[name]
        with cols[i]:
            safe_name = html.escape(str(name))
            # Count completed analyses
            analyses_done = sum(
                [
                    1 if data.get("calibration_params") else 0,
                    1 if data.get("isotherm_models_fitted") else 0,
                    1 if data.get("kinetic_models_fitted") else 0,
                    1 if data.get("thermo_params") else 0,
                    1 if data.get("ph_effect_results") is not None else 0,
                    1 if data.get("temp_effect_results") is not None else 0,
                    1 if data.get("dosage_results") is not None else 0,
                ]
            )

            # Get best qm if available
            langmuir = data.get("isotherm_models_fitted", {}).get("Langmuir", {})
            qm = langmuir.get("params", {}).get("qm", None) if langmuir.get("converged") else None

            st.markdown(
                f"""
            <div style="background: linear-gradient(135deg, {get_study_color(i)}, {get_study_color(i)}88);
                        padding: 15px; border-radius: 10px; color: white; text-align: center;">
                <h4 style="margin: 0; color: white;">{safe_name}</h4>
                <p style="margin: 5px 0; font-size: 0.9em;">{analyses_done}/7 analyses</p>
                {f'<p style="margin: 0; font-size: 1.2em; font-weight: bold;">qm = {qm:.2f} mg/g</p>' if qm else ""}
            </div>
            """,
                unsafe_allow_html=True,
            )


# =============================================================================
# ISOTHERM COMPARISON
# =============================================================================
def _render_isotherm_comparison(studies_data: dict, study_names: list):
    """Render comprehensive isotherm model comparison."""
    st.markdown("### 📈 Isotherm Model Comparison")

    # Collect all isotherm data
    all_iso_data = []
    has_isotherm_data = False

    for name in study_names:
        data = studies_data[name]
        iso_models = data.get("isotherm_models_fitted", {})

        if iso_models:
            has_isotherm_data = True
            for model_name, results in iso_models.items():
                if results and results.get("converged"):
                    row = {
                        "Study": name,
                        "Model": model_name,
                        "R²": results.get("r_squared", np.nan),
                        "Adj-R²": results.get("adj_r_squared", np.nan),
                        "RMSE": results.get("rmse", np.nan),
                        "AICc": _finite(results.get("aicc")),
                        "BIC": _finite(results.get("bic")),
                        "Relative SSE": results.get(
                            "normalized_sse", results.get("chi_squared", np.nan)
                        ),
                    }
                    # Add model-specific parameters
                    params = results.get("params", {})
                    if model_name == "Langmuir":
                        row["qm (mg/g)"] = params.get("qm", np.nan)
                        row["KL (L/mg)"] = params.get("KL", np.nan)
                    elif model_name == "Freundlich":
                        row["KF"] = params.get("KF", np.nan)
                        row["n"] = params.get("n", np.nan)
                    elif model_name == "Temkin":
                        row["B1"] = params.get("B1", np.nan)
                        row["KT (L/mg)"] = params.get("KT", np.nan)
                    elif model_name == "Sips":
                        row["qm (mg/g)"] = params.get("qm", np.nan)
                        row["ns"] = params.get("ns", np.nan)

                    all_iso_data.append(row)

    if not has_isotherm_data:
        st.info("No isotherm data available. Complete isotherm analysis for at least 2 studies.")
        return

    iso_df = pd.DataFrame(all_iso_data)

    # --- Section 1: Summary Table ---
    st.markdown("#### 1. Parameter Summary Table")

    # Pivot table for key parameters
    pivot_cols = ["Study", "Model", "R²", "Adj-R²", "RMSE", "AICc"]
    display_df = iso_df[pivot_cols].copy()

    # Criteria are not comparable between studies (different observations), so
    # nothing is highlighted across the combined table.
    st.dataframe(
        style_dataframe(
            display_df,
            format_dict={"R²": "{:.4f}", "Adj-R²": "{:.4f}", "RMSE": "{:.4f}", "AICc": "{:.2f}"},
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --- Section 2: Langmuir qm Comparison ---
    st.markdown("#### 2. Maximum Adsorption Capacity (qm) Comparison")

    # The same capacity comparison (order and quantity) as the exports.
    capacity = capacity_comparison({name: studies_data[name] for name in study_names})
    if not capacity.empty and capacity[CAPACITY_CRITERION].notna().any():
        qm_df = capacity[capacity[CAPACITY_CRITERION].notna()].rename(
            columns={CAPACITY_CRITERION: "qm (mg/g)"}
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            # Bar chart with error bars if CI available
            fig_qm = go.Figure()

            colors = [get_study_color(study_names.index(m)) for m in qm_df["Study"]]

            fig_qm.add_trace(
                go.Bar(
                    x=qm_df["Study"],
                    y=qm_df["qm (mg/g)"],
                    marker_color=colors,
                    text=qm_df["qm (mg/g)"].round(2),
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>qm = %{y:.2f} mg/g<extra></extra>",
                )
            )

            fig_qm = apply_professional_style(
                fig_qm,
                title="Langmuir Maximum Adsorption Capacity (qm)",
                x_title="Study",
                y_title="qm (mg/g)",
                show_legend=False,
            )

            st.plotly_chart(fig_qm, use_container_width=True)

        with col2:
            st.markdown("**Order by fitted Langmuir qm:**")
            for _, row in qm_df.iterrows():
                flag = "" if row["qm status"] in ("identified", "—") else f" ({row['qm status']})"
                st.markdown(
                    f"{int(row['Order by qm'])}. **{row['Study']}**: "
                    f"{row['qm (mg/g)']:.2f} mg/g{flag} — R² {row['Langmuir R² (fit quality)']:.4f}"
                )
            for _, row in capacity[capacity[CAPACITY_CRITERION].isna()].iterrows():
                st.markdown(f"– **{row['Study']}**: {row['Note']}")
            st.caption(CAPACITY_CRITERION_NOTE)

    # --- Section 3: Model Comparison Grouped Bar Chart ---
    st.markdown("#### 3. R² Comparison Across Models")

    # Create grouped bar chart

    fig_r2 = go.Figure()

    for i, study_name in enumerate(study_names):
        study_data = iso_df[iso_df["Study"] == study_name]
        if not study_data.empty:
            fig_r2.add_trace(
                go.Bar(
                    name=study_name,
                    x=study_data["Model"],
                    y=study_data["R²"],
                    marker_color=get_study_color(i),
                    text=study_data["R²"].round(4),
                    textposition="outside",
                )
            )

    fig_r2 = apply_professional_style(
        fig_r2,
        title="Model Fit Comparison (R²) by Study",
        x_title="Isotherm Model",
        y_title="R²",
        height=450,
        barmode="group",
    )

    st.plotly_chart(fig_r2, use_container_width=True)

    # --- Section 4: Model Selection within each study ---
    st.markdown("#### 4. Model Selection within Each Study (by AICc)")

    best_models = []
    for name in study_names:
        comparison = compare_information_criteria(
            studies_data[name].get("isotherm_models_fitted", {}), "aicc"
        )
        if comparison["status"] == "none":
            continue
        best = comparison["best"]
        best_models.append(
            {
                "Study": name,
                "Lowest AICc (same observations)": best or "—",
                "AICc weight": comparison["per_model"][best]["weight"] if best else np.nan,
                "Note": "" if best else comparison["message"],
            }
        )

    if best_models:
        best_df = pd.DataFrame(best_models)
        st.dataframe(
            best_df.style.format({"AICc weight": "{:.1%}"}, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "AICc ranks models only within one study's observations; it does not rank "
            "materials or identify mechanisms."
        )

    # --- Section 5: Isotherm Curves Overlay ---
    st.markdown("#### 5. Isotherm Curves Overlay")

    fig_curves = go.Figure()

    for i, name in enumerate(study_names):
        data = studies_data[name]
        iso_results = _usable(data.get("isotherm_results"))

        if iso_results is not None and hasattr(iso_results, "shape") and not iso_results.empty:
            Ce = iso_results["Ce_mgL"].values
            qe = iso_results["qe_mg_g"].values

            # Experimental data points
            fig_curves.add_trace(
                go.Scatter(
                    x=Ce,
                    y=qe,
                    mode="markers",
                    name=f"{name} (exp)",
                    marker={"size": 10, "color": get_study_color(i), "symbol": "circle"},
                    legendgroup=name,
                )
            )

            # Fitted curve (Langmuir if available)
            langmuir = data.get("isotherm_models_fitted", {}).get("Langmuir", {})
            if langmuir.get("converged"):
                qm = langmuir["params"].get("qm", 0)
                KL = langmuir["params"].get("KL", 0)
                Ce_fit = np.linspace(0, max(Ce) * 1.1, 100)
                qe_fit = (qm * KL * Ce_fit) / (1 + KL * Ce_fit)

                fig_curves.add_trace(
                    go.Scatter(
                        x=Ce_fit,
                        y=qe_fit,
                        mode="lines",
                        name=f"{name} (Langmuir)",
                        line={"color": get_study_color(i), "width": 2},
                        legendgroup=name,
                    )
                )

    fig_curves = apply_professional_style(
        fig_curves,
        title="Adsorption Isotherms Comparison",
        x_title="Equilibrium Concentration, Ce (mg/L)",
        y_title="Adsorption Capacity, qe (mg/g)",
        height=500,
    )

    st.plotly_chart(fig_curves, use_container_width=True)


# =============================================================================
# KINETIC COMPARISON
# =============================================================================
def _render_kinetic_comparison(studies_data: dict, study_names: list):
    """Render comprehensive kinetic model comparison."""
    st.markdown("### ⏱️ Kinetic Model Comparison")

    # Collect all kinetic data
    all_kin_data = []
    has_kinetic_data = False

    for name in study_names:
        data = studies_data[name]
        kin_models = data.get("kinetic_models_fitted", {})

        if kin_models:
            has_kinetic_data = True
            for model_name, results in kin_models.items():
                if results and results.get("converged"):
                    row = {
                        "Study": name,
                        "Model": model_name,
                        "R²": results.get("r_squared", np.nan),
                        "Adj-R²": results.get("adj_r_squared", np.nan),
                        "RMSE": results.get("rmse", np.nan),
                        "AICc": _finite(results.get("aicc")),
                    }
                    # Add model-specific parameters
                    params = results.get("params", {})
                    if model_name == "PFO":
                        row["qe (mg/g)"] = params.get("qe", np.nan)
                        row["k1 (1/min)"] = params.get("k1", np.nan)
                    elif model_name == "PSO":
                        row["qe (mg/g)"] = params.get("qe", np.nan)
                        row["k2 (g/mg·min)"] = params.get("k2", np.nan)
                    elif model_name == "rPSO":
                        row["qe (mg/g)"] = params.get("qe", np.nan)
                        row["k2 (g/mg·min)"] = params.get("k2", np.nan)
                        row["φ (correction)"] = params.get("phi", np.nan)
                    elif model_name == "Elovich":
                        row["α (mg/g·min)"] = params.get("alpha", np.nan)
                        row["β (g/mg)"] = params.get("beta", np.nan)
                    elif model_name == "IPD":
                        row["kid (mg/g·min⁰·⁵)"] = params.get("kid", np.nan)
                        row["C"] = params.get("C", np.nan)

                    all_kin_data.append(row)

    if not has_kinetic_data:
        st.info("No kinetic data available. Complete kinetic analysis for at least 2 studies.")
        return

    kin_df = pd.DataFrame(all_kin_data)

    # --- Section 1: Summary Table ---
    st.markdown("#### 1. Kinetic Parameters Summary")

    st.dataframe(
        style_dataframe(
            kin_df,
            format_dict={
                "R²": "{:.4f}",
                "Adj-R²": "{:.4f}",
                "RMSE": "{:.4f}",
                "AICc": "{:.2f}",
                "qe (mg/g)": "{:.2f}",
                "k1 (1/min)": "{:.4f}",
                "k2 (g/mg·min)": "{:.6f}",
            },
            highlight_max_cols=["R²"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --- Section 2: PSO Parameters Comparison ---
    st.markdown("#### 2. Pseudo-Second Order (PSO) Parameters")

    pso_data = []
    for name in study_names:
        data = studies_data[name]
        pso = data.get("kinetic_models_fitted", {}).get("PSO", {})
        if pso.get("converged"):
            pso_data.append(
                {
                    "Study": name,
                    "qe (mg/g)": pso["params"].get("qe", 0),
                    "k2 (g/mg·min)": pso["params"].get("k2", 0),
                    "R²": pso.get("r_squared", 0),
                }
            )

    if pso_data:
        pso_df = pd.DataFrame(pso_data)

        col1, col2 = st.columns(2)

        with col1:
            # qe comparison
            fig_qe = go.Figure()
            colors = [get_study_color(study_names.index(m)) for m in pso_df["Study"]]

            fig_qe.add_trace(
                go.Bar(
                    x=pso_df["Study"],
                    y=pso_df["qe (mg/g)"],
                    marker_color=colors,
                    text=pso_df["qe (mg/g)"].round(2),
                    textposition="outside",
                )
            )

            fig_qe = apply_professional_style(
                fig_qe,
                title="Equilibrium Capacity (qe) - PSO Model",
                x_title="Study",
                y_title="qe (mg/g)",
                height=400,
                legend_horizontal=False,
            )
            st.plotly_chart(fig_qe, use_container_width=True)

        with col2:
            # k2 comparison
            fig_k2 = go.Figure()

            fig_k2.add_trace(
                go.Bar(
                    x=pso_df["Study"],
                    y=pso_df["k2 (g/mg·min)"],
                    marker_color=colors,
                    text=pso_df["k2 (g/mg·min)"].apply(lambda x: f"{x:.4f}"),
                    textposition="outside",
                )
            )

            fig_k2 = apply_professional_style(
                fig_k2,
                title="Rate Constant (k₂) - PSO Model",
                x_title="Study",
                y_title="k₂ (g/mg·min)",
                height=400,
                legend_horizontal=False,
            )

            st.plotly_chart(fig_k2, use_container_width=True)

        # Mechanistic interpretation warning
        st.warning("""
        ⚠️ **Important Note on PSO Fit:**

        A good fit to PSO does **not** confirm chemisorption. ~90% of kinetic studies report PSO as
        "best fit" regardless of actual mechanism (Hubbe et al., 2019). The PSO equation can be derived
        from multiple mechanisms including diffusion control (Azizian, 2004).

        **For mechanistic evidence, use:**
        - Boyd plot analysis (film vs. pore diffusion)
        - Activation energy from temperature studies
        - Particle size variation experiments
        """)

    # --- Section 3: Kinetic Curves Overlay ---
    st.markdown("#### 3. Kinetic Curves Overlay")

    fig_kin_curves = go.Figure()

    for i, name in enumerate(study_names):
        data = studies_data[name]
        kin_results = _usable(data.get("kinetic_results_df"))

        if kin_results is not None and hasattr(kin_results, "shape") and not kin_results.empty:
            t = kin_results["Time"].values
            qt = kin_results["qt_mg_g"].values

            # Experimental data
            fig_kin_curves.add_trace(
                go.Scatter(
                    x=t,
                    y=qt,
                    mode="markers",
                    name=f"{name} (exp)",
                    marker={"size": 8, "color": get_study_color(i)},
                    legendgroup=name,
                )
            )

            # PSO fitted curve if available
            pso = data.get("kinetic_models_fitted", {}).get("PSO", {})
            if pso.get("converged"):
                qe = pso["params"].get("qe", 0)
                k2 = pso["params"].get("k2", 0)
                t_fit = np.linspace(0, max(t) * 1.1, 100)
                qt_fit = (qe**2 * k2 * t_fit) / (1 + qe * k2 * t_fit)

                fig_kin_curves.add_trace(
                    go.Scatter(
                        x=t_fit,
                        y=qt_fit,
                        mode="lines",
                        name=f"{name} (PSO)",
                        line={"color": get_study_color(i), "width": 2},
                        legendgroup=name,
                    )
                )

    fig_kin_curves = apply_professional_style(
        fig_kin_curves,
        title="Adsorption Kinetics Comparison",
        x_title="Time (min)",
        y_title="qt (mg/g)",
        height=500,
    )

    st.plotly_chart(fig_kin_curves, use_container_width=True)


# =============================================================================
# THERMODYNAMIC COMPARISON
# =============================================================================
def _apparent_delta_g_298(thermo: dict) -> float:
    """
    Apparent ΔG at 298.15 K from the fitted ΔH and ΔS, in kJ/mol (shared with the
    exports); ``nan`` when either input is missing or non-finite, never 0.
    """
    return apparent_delta_g(thermo)


def _render_thermodynamic_comparison(studies_data: dict, study_names: list):
    """Render thermodynamic parameters comparison."""
    st.markdown("### 🌡️ Thermodynamic Parameters Comparison")

    thermo_data = []

    for name in study_names:
        data = studies_data[name]
        thermo = data.get("thermo_params")

        if thermo:
            thermo_data.append(
                {
                    "Study": name,
                    "ΔH° (kJ/mol)": thermo.get("delta_H", np.nan),
                    "ΔS° (J/mol·K)": thermo.get("delta_S", np.nan),
                    "Apparent ΔG at 298K (kJ/mol)": _apparent_delta_g_298(thermo),
                    "R² (Van't Hoff)": thermo.get("r_squared", np.nan),
                    "Process": sign_label(thermo.get("delta_H"), "Exothermic", "Endothermic"),
                    "ΔG sign": sign_label(_apparent_delta_g_298(thermo), "Negative", "Positive"),
                    "Kd definition": kd_definition(thermo),
                }
            )

    if not thermo_data:
        st.info(
            "No thermodynamic data available. Complete thermodynamic analysis for at least 2 studies."
        )
        return

    thermo_df = pd.DataFrame(thermo_data)

    # --- Table ---
    st.markdown("#### 1. Thermodynamic Parameters Table")
    st.caption(f"{APPARENT_THERMO_NOTE}. Compare studies only when their Kd definitions match.")
    st.dataframe(
        thermo_df.style.format(
            {
                "ΔH° (kJ/mol)": "{:.2f}",
                "ΔS° (J/mol·K)": "{:.2f}",
                "Apparent ΔG at 298K (kJ/mol)": "{:.2f}",
                "R² (Van't Hoff)": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --- Bar Charts ---
    st.markdown("#### 2. Thermodynamic Parameters Visualization")

    col1, col2, col3 = st.columns(3)

    colors = [get_study_color(study_names.index(m)) for m in thermo_df["Study"]]

    with col1:
        fig_dh = go.Figure(
            go.Bar(
                x=thermo_df["Study"],
                y=thermo_df["ΔH° (kJ/mol)"],
                marker_color=colors,
                text=thermo_df["ΔH° (kJ/mol)"].round(2),
                textposition="outside",
            )
        )
        fig_dh.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_dh = apply_professional_style(
            fig_dh,
            title="ΔH° (kJ/mol)",
            x_title="Study",
            y_title="ΔH° (kJ/mol)",
            height=350,
            legend_horizontal=False,
        )
        st.plotly_chart(fig_dh, use_container_width=True)

    with col2:
        fig_ds = go.Figure(
            go.Bar(
                x=thermo_df["Study"],
                y=thermo_df["ΔS° (J/mol·K)"],
                marker_color=colors,
                text=thermo_df["ΔS° (J/mol·K)"].round(2),
                textposition="outside",
            )
        )
        fig_ds.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_ds = apply_professional_style(
            fig_ds,
            title="ΔS° (J/mol·K)",
            x_title="Study",
            y_title="ΔS° (J/mol·K)",
            height=350,
            legend_horizontal=False,
        )
        st.plotly_chart(fig_ds, use_container_width=True)

    with col3:
        fig_dg = go.Figure(
            go.Bar(
                x=thermo_df["Study"],
                y=thermo_df["Apparent ΔG at 298K (kJ/mol)"],
                marker_color=colors,
                text=thermo_df["Apparent ΔG at 298K (kJ/mol)"].round(2),
                textposition="outside",
            )
        )
        fig_dg.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_dg = apply_professional_style(
            fig_dg,
            title="Apparent ΔG at 298K (kJ/mol)",
            x_title="Study",
            y_title="Apparent ΔG (kJ/mol)",
            height=350,
            legend_horizontal=False,
        )
        st.plotly_chart(fig_dg, use_container_width=True)

    # --- Thermodynamic interpretation ---
    st.markdown("#### 3. Thermodynamic Trend Summary")

    for _, row in thermo_df.iterrows():
        study_label = html.escape(str(row["Study"]))
        process_label = html.escape(str(row["Process"]))
        dg_label = html.escape(str(row["ΔG sign"]))
        st.markdown(f"**{study_label}:** {process_label}; {dg_label.lower()} apparent ΔG")

    st.warning(
        "ΔH magnitude, apparent ΔG, and fitted isotherms do not identify physical versus "
        "chemical adsorption. Mechanism claims require independent experimental evidence."
    )


# =============================================================================
# EFFECT STUDIES COMPARISON
# =============================================================================
def _render_effect_studies_comparison(studies_data: dict, study_names: list):
    """Render pH, temperature, and dosage effect comparisons (house style)."""
    st.markdown("### 🔬 Effect Studies Comparison")

    effect_tab1, effect_tab2, effect_tab3 = st.tabs(
        ["🧪 pH Effect", "🌡️ Temperature Effect", "⚖️ Dosage Effect"]
    )

    # --- pH Effect ---
    with effect_tab1:
        st.markdown("#### pH Effect Comparison")

        fig_ph = go.Figure()
        has_ph_data = False
        skipped_ph = []

        for i, name in enumerate(study_names):
            data = studies_data.get(name, {})
            ph_results = _usable(data.get("ph_effect_results"))
            if ph_results is None or getattr(ph_results, "empty", True):
                skipped_ph.append(f"{name} (no pH data)")
                continue
            if "pH" not in ph_results.columns or "qe_mg_g" not in ph_results.columns:
                skipped_ph.append(f"{name} (missing required columns)")
                continue

            has_ph_data = True
            tr = style_study_trace(i, name, marker_size=10)
            tr["hovertemplate"] = "pH: %{x:.1f}<br>qe: %{y:.2f}<extra></extra>"
            fig_ph.add_trace(go.Scatter(x=ph_results["pH"], y=ph_results["qe_mg_g"], **tr))

        if has_ph_data:
            fig_ph = apply_professional_style(
                fig_ph,
                title="Effect of pH on Adsorption Capacity",
                x_title="pH",
                y_title="qe (mg/g)",
                height=450,
            )
            fig_ph.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig_ph, use_container_width=True)

            st.markdown("**Optimal pH by Study:**")
            opt_ph_data = []
            for name in study_names:
                data = studies_data.get(name, {})
                ph_results = _usable(data.get("ph_effect_results"))
                if ph_results is None or getattr(ph_results, "empty", True):
                    continue
                if "pH" not in ph_results.columns or "qe_mg_g" not in ph_results.columns:
                    continue
                opt_idx = ph_results["qe_mg_g"].idxmax()
                opt_ph_data.append(
                    {
                        "Study": name,
                        "Optimal pH": float(ph_results.loc[opt_idx, "pH"]),
                        "Max qe (mg/g)": float(ph_results.loc[opt_idx, "qe_mg_g"]),
                    }
                )
            if opt_ph_data:
                display_results_table(pd.DataFrame(opt_ph_data))
        else:
            st.info("No pH effect data available.")
        if skipped_ph:
            st.caption(f"Studies skipped: {', '.join(skipped_ph)}")

    # --- Temperature Effect ---
    with effect_tab2:
        st.markdown("#### Temperature Effect Comparison")

        fig_temp = go.Figure()
        has_temp_data = False
        skipped_temp = []

        for i, name in enumerate(study_names):
            data = studies_data.get(name, {})
            temp_results = _usable(data.get("temp_effect_results"))
            if temp_results is None or getattr(temp_results, "empty", True):
                skipped_temp.append(f"{name} (no temperature data)")
                continue

            x_col = (
                "Temperature_C"
                if "Temperature_C" in temp_results.columns
                else ("Temperature" if "Temperature" in temp_results.columns else None)
            )
            if x_col is None or "qe_mg_g" not in temp_results.columns:
                skipped_temp.append(f"{name} (missing required columns)")
                continue

            has_temp_data = True
            tr = style_study_trace(i, name, marker_size=10)
            tr["hovertemplate"] = "T: %{x:.1f}°C<br>qe: %{y:.2f}<extra></extra>"
            fig_temp.add_trace(go.Scatter(x=temp_results[x_col], y=temp_results["qe_mg_g"], **tr))

        if has_temp_data:
            fig_temp = apply_professional_style(
                fig_temp,
                title="Effect of Temperature on Adsorption Capacity",
                x_title="Temperature (°C)",
                y_title="qe (mg/g)",
                height=450,
            )
            fig_temp.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig_temp, use_container_width=True)
        else:
            st.info("No temperature effect data available.")
        if skipped_temp:
            st.caption(f"Studies skipped: {', '.join(skipped_temp)}")

    # --- Dosage Effect ---
    with effect_tab3:
        st.markdown("#### Dosage Effect Comparison")

        fig_dos = go.Figure()
        has_dos_data = False
        skipped_dos = []

        for i, name in enumerate(study_names):
            data = studies_data.get(name, {})
            dos_results = _usable(data.get("dosage_results"))
            if dos_results is None or getattr(dos_results, "empty", True):
                skipped_dos.append(f"{name} (no dosage data)")
                continue

            # x column
            if "Dosage_gL" in dos_results.columns:
                x_col = "Dosage_gL"
                x_label = "Dosage (g/L)"
            elif "Mass_g" in dos_results.columns:
                x_col = "Mass_g"
                x_label = "Mass (g)"
            else:
                skipped_dos.append(f"{name} (missing dosage/mass column)")
                continue

            # y preference: qe if present else removal
            if "qe_mg_g" in dos_results.columns:
                y_col = "qe_mg_g"
                y_label = "qe (mg/g)"
                hover = "Dosage: %{x:.4f}<br>qe: %{y:.2f}<extra></extra>"
            elif "removal_%" in dos_results.columns:
                y_col = "removal_%"
                y_label = "Removal (%)"
                hover = "Dosage: %{x:.4f}<br>Removal: %{y:.2f}%<extra></extra>"
            else:
                skipped_dos.append(f"{name} (missing qe/removal column)")
                continue

            has_dos_data = True
            tr = style_study_trace(i, name, marker_size=10)
            tr["hovertemplate"] = hover
            fig_dos.add_trace(go.Scatter(x=dos_results[x_col], y=dos_results[y_col], **tr))

        if has_dos_data:
            fig_dos = apply_professional_style(
                fig_dos,
                title="Effect of Adsorbent Dosage",
                x_title=x_label,
                y_title=y_label,
                height=450,
            )
            fig_dos.update_xaxes(rangemode="tozero")
            fig_dos.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig_dos, use_container_width=True)
        else:
            st.info("No dosage effect data available.")
        if skipped_dos:
            st.caption(f"Studies skipped: {', '.join(skipped_dos)}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _display_setup_guide():
    """Display guide for setting up multi-study comparison."""
    with st.expander("📖 How to Set Up Multi-Study Comparison", expanded=True):
        st.markdown("""
        **To compare multiple studies:**

        1. **Add Study 1:** Click "Add New Study" in the sidebar, name it (e.g., "Zeolite")
        2. **Analyze Study 1:** Complete calibration, isotherm, kinetics, and other analyses
        3. **Add Study 2:** Click "Add New Study" again, name it (e.g., "Activated Carbon")
        4. **Analyze Study 2:** Complete the same analyses
        5. **Compare:** Return to this tab to see comprehensive comparisons

        **Tips:**
        - Use consistent experimental conditions across studies for fair comparison
        - Complete at least isotherm and kinetic analyses for meaningful comparison
        - The more analyses completed, the richer the comparison
        """)
