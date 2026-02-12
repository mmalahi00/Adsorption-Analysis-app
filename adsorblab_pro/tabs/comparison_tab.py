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
- Radar charts for multi-dimensional comparison
"""

import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from adsorblab_pro.streamlit_compat import st

from ..plot_style import (
    apply_professional_style,
    apply_professional_polar_style,
    get_study_color,
    hex_to_rgba,
    style_study_trace,
)
from ..utils import display_results_table


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

    # Key insights (auto-generated)
    _render_key_insights(studies_data, study_names)

    st.markdown("---")

    # Create tabs for different comparison categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📈 Isotherms",
            "⏱️ Kinetics",
            "🌡️ Thermodynamics",
            "🔬 Effect Studies",
            "📊 Overall Ranking",
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

    with tab5:
        _render_overall_ranking(studies_data, study_names)


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
# KEY INSIGHTS (AUTO-GENERATED)
# =============================================================================
def _render_key_insights(studies_data: dict, study_names: list):
    """Generate automatic key insights from comparison data."""
    st.markdown("### 💡 Key Insights")

    insights = []

    # --- Capacity Comparison ---
    qm_values = {}
    for name in study_names:
        data = studies_data[name]
        langmuir = data.get("isotherm_models_fitted", {}).get("Langmuir", {})
        if langmuir.get("converged"):
            qm_values[name] = langmuir["params"].get("qm", 0)

    if len(qm_values) >= 2:
        best_qm = max(qm_values, key=qm_values.get)
        worst_qm = min(qm_values, key=qm_values.get)
        ratio = qm_values[best_qm] / qm_values[worst_qm] if qm_values[worst_qm] > 0 else 0

        if ratio > 1.5:
            insights.append(
                f"🏆 **{best_qm}** has {ratio:.1f}× higher adsorption capacity than {worst_qm} (qm = {qm_values[best_qm]:.1f} vs {qm_values[worst_qm]:.1f} mg/g)"
            )
        else:
            insights.append(
                f"📊 Adsorption capacities are similar: {best_qm} ({qm_values[best_qm]:.1f} mg/g) vs {worst_qm} ({qm_values[worst_qm]:.1f} mg/g)"
            )

    # --- Kinetic Speed Comparison ---
    k2_values = {}
    qe_values = {}
    for name in study_names:
        data = studies_data[name]
        pso = data.get("kinetic_models_fitted", {}).get("PSO", {})
        if pso.get("converged"):
            k2_values[name] = pso["params"].get("k2", 0)
            qe_values[name] = pso["params"].get("qe", 0)

    if len(k2_values) >= 2:
        fastest = max(k2_values, key=k2_values.get)
        slowest = min(k2_values, key=k2_values.get)

        # Estimate time to 90% equilibrium: t_90 ≈ 9/(k2*qe)
        if k2_values[fastest] > 0 and qe_values.get(fastest, 0) > 0:
            t90_fast = 9 / (k2_values[fastest] * qe_values[fastest])
            t90_slow = (
                9 / (k2_values[slowest] * qe_values[slowest])
                if k2_values[slowest] > 0 and qe_values.get(slowest, 0) > 0
                else float("inf")
            )

            if t90_fast < t90_slow * 0.7:
                insights.append(
                    f"⚡ **{fastest}** reaches equilibrium fastest (~{t90_fast:.0f} min to 90% vs ~{t90_slow:.0f} min for {slowest})"
                )

    # --- Thermodynamic Comparison ---
    spontaneous = []
    non_spontaneous = []
    endothermic = []
    exothermic = []

    for name in study_names:
        data = studies_data[name]
        thermo = data.get("thermo_params")
        if thermo:
            delta_H = thermo.get("delta_H", 0)
            delta_S = thermo.get("delta_S", 0)
            delta_G = delta_H - 298.15 * delta_S / 1000  # At 25°C

            if delta_G < 0:
                spontaneous.append(name)
            else:
                non_spontaneous.append(name)

            if delta_H > 0:
                endothermic.append(name)
            else:
                exothermic.append(name)

    if spontaneous:
        if len(spontaneous) == len(study_names):
            insights.append("✅ All studies show **spontaneous adsorption** (ΔG° < 0)")
        else:
            insights.append(f"✅ Spontaneous adsorption: {', '.join(spontaneous)}")

    if endothermic and exothermic:
        insights.append(
            f"🌡️ Mechanism differs: {', '.join(endothermic)} (endothermic) vs {', '.join(exothermic)} (exothermic)"
        )
    elif endothermic:
        insights.append(
            "🌡️ All studies show **endothermic** adsorption (ΔH° > 0) - higher temperature favors adsorption"
        )
    elif exothermic:
        insights.append(
            "🌡️ All studies show **exothermic** adsorption (ΔH° < 0) - lower temperature favors adsorption"
        )

    # --- Model Fit Quality ---
    r2_values = {}
    for name in study_names:
        data = studies_data[name]
        langmuir = data.get("isotherm_models_fitted", {}).get("Langmuir", {})
        if langmuir.get("converged"):
            r2_values[name] = langmuir.get("r_squared", 0)

    if r2_values:
        avg_r2 = sum(r2_values.values()) / len(r2_values)
        if avg_r2 >= 0.99:
            insights.append(f"📈 Excellent model fits across all studies (avg R² = {avg_r2:.4f})")
        elif avg_r2 >= 0.95:
            insights.append(f"📈 Good model fits (avg R² = {avg_r2:.4f})")

    # --- Display insights ---
    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.info("Complete more analyses to generate comparative insights.")


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
                        "AIC": results.get("aicc", results.get("aic", np.nan)),
                        "BIC": results.get("bic", np.nan),
                        "χ²": results.get("chi_squared", np.nan),
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
    pivot_cols = ["Study", "Model", "R²", "Adj-R²", "RMSE", "AIC"]
    display_df = iso_df[pivot_cols].copy()

    st.dataframe(
        style_dataframe(
            display_df,
            format_dict={"R²": "{:.4f}", "Adj-R²": "{:.4f}", "RMSE": "{:.4f}", "AIC": "{:.2f}"},
            highlight_max_cols=["R²", "Adj-R²"],
            highlight_min_cols=["RMSE", "AIC"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --- Section 2: Langmuir qm Comparison ---
    st.markdown("#### 2. Maximum Adsorption Capacity (qm) Comparison")

    qm_data = []
    for name in study_names:
        data = studies_data[name]
        langmuir = data.get("isotherm_models_fitted", {}).get("Langmuir", {})
        if langmuir.get("converged"):
            qm_data.append(
                {
                    "Study": name,
                    "qm (mg/g)": langmuir["params"].get("qm", 0),
                    "KL (L/mg)": langmuir["params"].get("KL", 0),
                    "R²": langmuir.get("r_squared", 0),
                }
            )

    if qm_data:
        qm_df = pd.DataFrame(qm_data).sort_values("qm (mg/g)", ascending=False)

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
            st.markdown("**Ranking by qm:**")
            for i, row in qm_df.iterrows():
                rank = qm_df.index.get_loc(i) + 1
                medal = (
                    "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
                )
                st.markdown(f"{medal} **{row['Study']}**: {row['qm (mg/g)']:.2f} mg/g")

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

    # --- Section 4: Best Model Selection ---
    st.markdown("#### 4. Best Model Selection (by AIC)")

    best_models = []
    for name in study_names:
        study_data = iso_df[iso_df["Study"] == name]
        if not study_data.empty:
            try:
                best_idx = study_data["AIC"].idxmin()
                if pd.notna(best_idx):
                    best_row = study_data.loc[best_idx]
                    best_models.append(
                        {
                            "Study": name,
                            "Best Model": best_row["Model"],
                            "AIC": best_row["AIC"],
                            "R²": best_row["R²"],
                        }
                    )
            except (KeyError, ValueError):
                pass

    if best_models:
        best_df = pd.DataFrame(best_models)
        display_results_table(best_df)

    # --- Section 5: Isotherm Curves Overlay ---
    st.markdown("#### 5. Isotherm Curves Overlay")

    fig_curves = go.Figure()

    for i, name in enumerate(study_names):
        data = studies_data[name]
        iso_results = data.get("isotherm_results")

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
                        "AIC": results.get("aicc", results.get("aic", np.nan)),
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
                "AIC": "{:.2f}",
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
        kin_results = data.get("kinetic_results_df")

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
                    "ΔG° at 298K (kJ/mol)": thermo.get("delta_H", 0)
                    - 298.15 * thermo.get("delta_S", 0) / 1000,
                    "R² (Van't Hoff)": thermo.get("r_squared", np.nan),
                    "Process": "Exothermic" if thermo.get("delta_H", 0) < 0 else "Endothermic",
                    "Spontaneity": "Spontaneous"
                    if (thermo.get("delta_H", 0) - 298.15 * thermo.get("delta_S", 0) / 1000) < 0
                    else "Non-spontaneous",
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
    st.dataframe(
        thermo_df.style.format(
            {
                "ΔH° (kJ/mol)": "{:.2f}",
                "ΔS° (J/mol·K)": "{:.2f}",
                "ΔG° at 298K (kJ/mol)": "{:.2f}",
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
                y=thermo_df["ΔG° at 298K (kJ/mol)"],
                marker_color=colors,
                text=thermo_df["ΔG° at 298K (kJ/mol)"].round(2),
                textposition="outside",
            )
        )
        fig_dg.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_dg = apply_professional_style(
            fig_dg,
            title="ΔG° at 298K (kJ/mol)",
            x_title="Study",
            y_title="ΔG° (kJ/mol)",
            height=350,
            legend_horizontal=False,
        )
        st.plotly_chart(fig_dg, use_container_width=True)

    # --- Mechanism Interpretation ---
    st.markdown("#### 3. Mechanism Interpretation")

    for _, row in thermo_df.iterrows():
        abs_H = abs(row["ΔH° (kJ/mol)"])
        if abs_H < 40:
            mechanism = "Physical Adsorption"
            color = "blue"
        elif abs_H < 80:
            mechanism = "Mixed Mechanism"
            color = "orange"
        else:
            mechanism = "Chemical Adsorption"
            color = "red"

        study_label = html.escape(str(row["Study"]))
        process_label = html.escape(str(row["Process"]))
        spont_label = html.escape(str(row["Spontaneity"]))
        st.markdown(
            f"**{study_label}:** <span style='color:{color}'>{mechanism}</span> (|ΔH°| = {abs_H:.2f} kJ/mol) - {process_label}, {spont_label}",
            unsafe_allow_html=True,
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

        for i, name in enumerate(study_names):
            data = studies_data.get(name, {})
            ph_results = data.get("ph_effect_results")
            if ph_results is None or getattr(ph_results, "empty", True):
                continue
            if "pH" not in ph_results.columns or "qe_mg_g" not in ph_results.columns:
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
                ph_results = data.get("ph_effect_results")
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

    # --- Temperature Effect ---
    with effect_tab2:
        st.markdown("#### Temperature Effect Comparison")

        fig_temp = go.Figure()
        has_temp_data = False

        for i, name in enumerate(study_names):
            data = studies_data.get(name, {})
            temp_results = data.get("temp_effect_results")
            if temp_results is None or getattr(temp_results, "empty", True):
                continue

            x_col = (
                "Temperature_C"
                if "Temperature_C" in temp_results.columns
                else ("Temperature" if "Temperature" in temp_results.columns else None)
            )
            if x_col is None or "qe_mg_g" not in temp_results.columns:
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

    # --- Dosage Effect ---
    with effect_tab3:
        st.markdown("#### Dosage Effect Comparison")

        fig_dos = go.Figure()
        has_dos_data = False

        for i, name in enumerate(study_names):
            data = studies_data.get(name, {})
            dos_results = data.get("dosage_effect_results")
            if dos_results is None or getattr(dos_results, "empty", True):
                continue

            # x column
            if "Dosage_gL" in dos_results.columns:
                x_col = "Dosage_gL"
                x_label = "Dosage (g/L)"
            elif "Mass_g" in dos_results.columns:
                x_col = "Mass_g"
                x_label = "Mass (g)"
            else:
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


def _render_overall_ranking(studies_data: dict, study_names: list):
    """Render overall study ranking with radar chart."""
    st.markdown("### 📊 Overall Study Ranking")

    # Collect scores for each material
    ranking_data = []

    for name in study_names:
        data = studies_data[name]

        scores = {"Study": name}

        # Langmuir qm (higher is better)
        langmuir = data.get("isotherm_models_fitted", {}).get("Langmuir", {})
        scores["qm (mg/g)"] = langmuir["params"].get("qm", 0) if langmuir.get("converged") else 0

        # PSO qe (higher is better)
        pso = data.get("kinetic_models_fitted", {}).get("PSO", {})
        scores["qe (mg/g)"] = pso["params"].get("qe", 0) if pso.get("converged") else 0

        # PSO k2 (higher = faster kinetics)
        scores["k2 (g/mg·min)"] = pso["params"].get("k2", 0) if pso.get("converged") else 0

        # Model fit quality (R²)
        scores["Isotherm R²"] = langmuir.get("r_squared", 0) if langmuir.get("converged") else 0
        scores["Kinetic R²"] = pso.get("r_squared", 0) if pso.get("converged") else 0

        # Thermodynamic favorability (negative ΔG is better)
        thermo = data.get("thermo_params")
        if thermo:
            delta_G = thermo.get("delta_H", 0) - 298.15 * thermo.get("delta_S", 0) / 1000
            scores["ΔG° (kJ/mol)"] = delta_G
            scores["Spontaneous"] = 1 if delta_G < 0 else 0
        else:
            scores["ΔG° (kJ/mol)"] = 0
            scores["Spontaneous"] = 0

        ranking_data.append(scores)

    ranking_df = pd.DataFrame(ranking_data)

    # --- Summary Table ---
    st.markdown("#### 1. Performance Summary Table")
    st.dataframe(
        style_dataframe(
            ranking_df,
            format_dict={
                "qm (mg/g)": "{:.2f}",
                "qe (mg/g)": "{:.2f}",
                "k2 (g/mg·min)": "{:.6f}",
                "Isotherm R²": "{:.4f}",
                "Kinetic R²": "{:.4f}",
                "ΔG° (kJ/mol)": "{:.2f}",
            },
            highlight_max_cols=["qm (mg/g)", "qe (mg/g)", "Isotherm R²", "Kinetic R²"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --- Radar Chart ---
    st.markdown("#### 2. Multi-Dimensional Performance Comparison (Radar Chart)")

    # Normalize scores for radar chart (0-1 scale)
    radar_metrics = ["qm (mg/g)", "qe (mg/g)", "Isotherm R²", "Kinetic R²"]

    # Check if we have data for radar
    has_radar_data = any(ranking_df[radar_metrics].sum(axis=1) > 0)

    if has_radar_data:
        normalized_df = ranking_df.copy()
        for col in radar_metrics:
            max_val = normalized_df[col].max()
            if max_val > 0:
                normalized_df[col + "_norm"] = normalized_df[col] / max_val
            else:
                normalized_df[col + "_norm"] = 0

        fig_radar = go.Figure()

        for i, name in enumerate(study_names):
            row = normalized_df[normalized_df["Study"] == name].iloc[0]
            values = [row[col + "_norm"] for col in radar_metrics]
            values.append(values[0])  # Close the radar

            fig_radar.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=radar_metrics + [radar_metrics[0]],
                    fill="toself",
                    name=name,
                    line={"color": get_study_color(i), "width": 2},
                    fillcolor=hex_to_rgba(get_study_color(i), 0.25),  # 25% opacity
                )
            )

        fig_radar = apply_professional_polar_style(
            fig_radar,
            title="Normalized Performance Comparison",
            height=500,
            show_legend=True,
            legend_position="upper left",
            radial_range=(0.0, 1.0),
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    # --- Consolidated Summary Table ---
    st.markdown("#### 3. Consolidated Summary Table")
    st.markdown("*All key parameters in one table*")

    pub_table_data = []
    for name in study_names:
        data = studies_data[name]

        row = {"Study/Adsorbent": name}

        # Isotherm parameters
        langmuir = data.get("isotherm_models_fitted", {}).get("Langmuir", {})
        freundlich = data.get("isotherm_models_fitted", {}).get("Freundlich", {})

        if langmuir.get("converged"):
            row["qm (mg/g)"] = langmuir["params"].get("qm", np.nan)
            row["KL (L/mg)"] = langmuir["params"].get("KL", np.nan)
            row["R²_L"] = langmuir.get("r_squared", np.nan)
        else:
            row["qm (mg/g)"] = np.nan
            row["KL (L/mg)"] = np.nan
            row["R²_L"] = np.nan

        if freundlich.get("converged"):
            row["KF"] = freundlich["params"].get("KF", np.nan)
            row["n"] = freundlich["params"].get("n", np.nan)
        else:
            row["KF"] = np.nan
            row["n"] = np.nan

        # Kinetic parameters
        pso = data.get("kinetic_models_fitted", {}).get("PSO", {})
        if pso.get("converged"):
            row["qe (mg/g)"] = pso["params"].get("qe", np.nan)
            row["k2 (g/mg·min)"] = pso["params"].get("k2", np.nan)
            row["R²_PSO"] = pso.get("r_squared", np.nan)
        else:
            row["qe (mg/g)"] = np.nan
            row["k2 (g/mg·min)"] = np.nan
            row["R²_PSO"] = np.nan

        # Thermodynamic parameters
        thermo = data.get("thermo_params")
        if thermo:
            row["ΔH° (kJ/mol)"] = thermo.get("delta_H", np.nan)
            row["ΔS° (J/mol·K)"] = thermo.get("delta_S", np.nan)
            delta_G_vals = thermo.get("delta_G_values", [])
            row["ΔG° (kJ/mol)"] = delta_G_vals[0] if delta_G_vals else np.nan
        else:
            row["ΔH° (kJ/mol)"] = np.nan
            row["ΔS° (J/mol·K)"] = np.nan
            row["ΔG° (kJ/mol)"] = np.nan

        pub_table_data.append(row)

    pub_df = pd.DataFrame(pub_table_data)

    st.dataframe(
        style_dataframe(
            pub_df,
            format_dict={
                "qm (mg/g)": "{:.2f}",
                "KL (L/mg)": "{:.4f}",
                "R²_L": "{:.4f}",
                "KF": "{:.2f}",
                "n": "{:.2f}",
                "qe (mg/g)": "{:.2f}",
                "k2 (g/mg·min)": "{:.6f}",
                "R²_PSO": "{:.4f}",
                "ΔH° (kJ/mol)": "{:.2f}",
                "ΔS° (J/mol·K)": "{:.2f}",
                "ΔG° (kJ/mol)": "{:.2f}",
            },
            highlight_max_cols=["qm (mg/g)", "qe (mg/g)", "R²_L", "R²_PSO"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --- Mechanism Interpretation ---
    st.markdown("#### 4. Mechanism Interpretation")

    mechanism_data = []
    for name in study_names:
        data = studies_data[name]

        row = {"Study": name}
        interpretations = []

        # Freundlich n - favorability
        freundlich = data.get("isotherm_models_fitted", {}).get("Freundlich", {})
        if freundlich.get("converged"):
            n = freundlich["params"].get("n", 0)
            row["n (Freundlich)"] = n
            if n > 1:
                row["Favorability"] = "Favorable"
                interpretations.append("n > 1 → favorable adsorption")
            elif n == 1:
                row["Favorability"] = "Linear"
                interpretations.append("n = 1 → linear isotherm")
            else:
                row["Favorability"] = "Unfavorable"
                interpretations.append("n < 1 → unfavorable adsorption")
        else:
            row["n (Freundlich)"] = np.nan
            row["Favorability"] = "—"

        # Thermodynamics
        thermo = data.get("thermo_params")
        if thermo:
            delta_H = thermo.get("delta_H", 0)
            delta_S = thermo.get("delta_S", 0)

            if delta_H > 0:
                row["Process"] = "Endothermic"
                interpretations.append("ΔH° > 0 → endothermic")
            else:
                row["Process"] = "Exothermic"
                interpretations.append("ΔH° < 0 → exothermic")

            # Enthalpy magnitude for mechanism
            abs_H = abs(delta_H)
            if abs_H < 40:
                row["Bonding"] = "Physical"
                interpretations.append("|ΔH°| < 40 kJ/mol → physical bonding")
            elif abs_H < 80:
                row["Bonding"] = "Mixed"
                interpretations.append("40 < |ΔH°| < 80 kJ/mol → mixed mechanism")
            else:
                row["Bonding"] = "Chemical"
                interpretations.append("|ΔH°| > 80 kJ/mol → chemical bonding")

            if delta_S > 0:
                interpretations.append("ΔS° > 0 → increased randomness at interface")
        else:
            row["Process"] = "—"
            row["Bonding"] = "—"

        row["Interpretation"] = "; ".join(interpretations) if interpretations else "—"
        mechanism_data.append(row)

    mech_df = pd.DataFrame(mechanism_data)

    # Display mechanism table
    display_cols = ["Study", "Adsorption Type", "Favorability", "Process", "Bonding"]
    if all(col in mech_df.columns for col in display_cols):
        display_results_table(mech_df[display_cols])

    # Show detailed interpretations in expander
    with st.expander("📖 Detailed Mechanism Interpretations", expanded=False):
        for row in mechanism_data:
            if row.get("Interpretation") and row["Interpretation"] != "—":
                st.markdown(f"**{row['Study']}:** {row['Interpretation']}")

    # Mechanistic interpretation guidance
    st.caption("""
    **⚠️ Note on Mechanistic Interpretation:**
    The mechanisms above are inferred from thermodynamic parameters (ΔH°) and isotherm parameters (n from Freundlich).
    These provide supporting evidence but are **not definitive proof** of mechanism.

    **Do NOT infer mechanism solely from kinetic model fit** (e.g., claiming chemisorption because PSO fits best).
    For robust mechanistic evidence, use: Boyd/Weber-Morris plots, activation energy studies,
    particle size variation, and spectroscopic analysis (FTIR, XPS).

    *Reference: Hubbe et al. (2019). BioResources, 14(3), 7582-7626.*
    """)

    # --- Final Ranking ---
    st.markdown("#### 5. Final Ranking (by qm)")

    if ranking_df["qm (mg/g)"].sum() > 0:
        final_ranking = ranking_df.sort_values("qm (mg/g)", ascending=False)[
            ["Study", "qm (mg/g)", "qe (mg/g)", "Isotherm R²"]
        ]

        for i, (_, row) in enumerate(final_ranking.iterrows()):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i + 1}."
            st.markdown(f"""
            {medal} **{row["Study"]}** — qm = {row["qm (mg/g)"]:.2f} mg/g | R² = {row["Isotherm R²"]:.4f}
            """)
    else:
        st.info("Complete isotherm analysis to see the final ranking.")

    # --- Export info ---
    st.markdown("---")
    st.info("💡 **To download all comparison data:** Go to **📦 Export All** tab")


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
