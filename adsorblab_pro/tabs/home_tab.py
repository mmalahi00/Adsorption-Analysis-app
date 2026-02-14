# tabs/home_tab.py
"""
Home Tab - AdsorbLab Pro
========================

The welcome page and main landing area.

Features:
- Application overview and capabilities
- Quick start guide with workflow diagram
- Current session status dashboard
- Feature highlights and tips
"""

from adsorblab_pro.streamlit_compat import st


def render():
    """Render the home tab landing page."""

    # ==========================================================================
    # HERO SECTION
    # ==========================================================================
    st.markdown(
        """
    <div style="background: linear-gradient(135deg, #2E86AB 0%, #A23B72 50%, #F18F01 100%);
                padding: 40px; border-radius: 15px; margin-bottom: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
        <h1 style="color: white; margin: 0; font-size: 2.5em;">Welcome to AdsorbLab Pro</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2em; margin-top: 10px;">
            Advanced Adsorption Data Analysis Platform
        </p>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.95em; margin-top: 5px;">
            Statistical analysis • Confidence intervals • Multi-model comparison • Mechanism interpretation
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ==========================================================================
    # SESSION STATUS DASHBOARD
    # ==========================================================================
    _render_session_status()

    st.markdown("---")

    # ==========================================================================
    # MAIN CONTENT
    # ==========================================================================

    _render_quick_start_guide()

    # ==========================================================================
    # WORKFLOW DIAGRAM
    # ==========================================================================
    _render_workflow_diagram()

    st.markdown("---")

    # ==========================================================================
    # CAPABILITIES & MODELS
    # ==========================================================================
    _render_capabilities()

    st.markdown("---")

    # ==========================================================================
    # TIPS & BEST PRACTICES
    # ==========================================================================
    _render_tips_section()


# =============================================================================
# SESSION STATUS
# =============================================================================
def _render_session_status():
    """Render current session status dashboard."""
    st.markdown("### 📊 Current Session Status")

    studies = st.session_state.get("studies", {})
    current_study = st.session_state.get("current_study")

    if not studies:
        # No studies yet
        st.info(
            "👋 **Getting Started:** No studies created yet. Add your first study using the sidebar!"
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Studies", "0")
        with col2:
            st.metric("Analyses", "0")
        with col3:
            st.metric("Active Study", "None")
        with col4:
            st.metric("Status", "Ready")
    else:
        # Show dashboard
        total_analyses = 0
        studies_with_calib = 0
        studies_with_iso = 0
        studies_with_kin = 0

        for _name, data in studies.items():
            if data.get("calibration_params"):
                studies_with_calib += 1
                total_analyses += 1
            if data.get("isotherm_models_fitted"):
                studies_with_iso += 1
                total_analyses += 1
            if data.get("kinetic_models_fitted"):
                studies_with_kin += 1
                total_analyses += 1
            if data.get("thermo_params"):
                total_analyses += 1
            if data.get("ph_effect_results") is not None:
                total_analyses += 1
            if data.get("temp_effect_results") is not None:
                total_analyses += 1
            if data.get("dosage_results") is not None:
                total_analyses += 1

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Studies", len(studies))
        with col2:
            st.metric("Total Analyses", total_analyses)
        with col3:
            st.metric("Active Study", current_study or "None")
        with col4:
            completion = (total_analyses / (len(studies) * 7)) * 100 if studies else 0
            st.metric("Completion", f"{completion:.0f}%")

        st.caption(
            f"📊 Breakdown: {studies_with_calib} calibrations, {studies_with_iso} isotherms, {studies_with_kin} kinetics"
        )

        # Study cards
        if len(studies) > 0:
            st.markdown("#### 📁 Your Studies")

            study_cols = st.columns(min(len(studies), 4))

            for i, (name, data) in enumerate(studies.items()):
                with study_cols[i % 4]:
                    # Count analyses for this study
                    analyses = []
                    if data.get("calibration_params"):
                        analyses.append("✅ Calibration")
                    if data.get("isotherm_models_fitted"):
                        analyses.append("✅ Isotherms")
                    if data.get("kinetic_models_fitted"):
                        analyses.append("✅ Kinetics")
                    if data.get("thermo_params"):
                        analyses.append("✅ Thermo")

                    # Get qm if available
                    langmuir = data.get("isotherm_models_fitted", {}).get("Langmuir", {})
                    qm = langmuir.get("params", {}).get("qm") if langmuir.get("converged") else None

                    is_active = name == current_study
                    border_color = "#2E86AB" if is_active else "#ddd"
                    bg_color = "#f0f8ff" if is_active else "#fafafa"

                    st.markdown(
                        f"""
                    <div style="border: 2px solid {border_color}; border-radius: 10px;
                                padding: 15px; background: {bg_color}; margin-bottom: 10px;">
                        <h4 style="margin: 0; color: #333;">{"🔬 " if is_active else ""}{name}</h4>
                        <p style="font-size: 0.85em; color: #666; margin: 5px 0;">
                            {len(analyses)}/7 analyses complete
                        </p>
                        {f'<p style="font-size: 0.9em; margin: 0;"><strong>qm = {qm:.2f} mg/g</strong></p>' if qm else ""}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )


# =============================================================================
# QUICK START GUIDE
# =============================================================================
def _render_quick_start_guide():
    """Render quick start guide."""
    st.markdown("### 🚀 Quick Start Guide")

    st.markdown(
        """
    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #2E86AB;">

    **Step 1: Create a Study** 📁
    > In the sidebar, enter a name for your study (e.g., material name, pollutant, or condition) and click "Add Study"

    **Step 2: Calibration Curve** 📊
    > Go to **Analysis Workflow → Calibration** and enter your standard solutions data

    **Step 3: Isotherm Analysis** 📈
    > Enter equilibrium data to fit Langmuir, Freundlich, and other models

    **Step 4: Kinetic Analysis** ⏱️
    > Enter time-series data to determine adsorption kinetics (PFO, PSO, etc.)

    **Step 5: Additional Studies** 🔬
    > Analyze pH, temperature, and dosage effects; calculate thermodynamic parameters

    **Step 6: Compare & Export** 📋
    > Compare multiple studies and export figures

    </div>
    """,
        unsafe_allow_html=True,
    )


# =============================================================================
# WORKFLOW DIAGRAM
# =============================================================================
def _render_workflow_diagram():
    """Render the analysis workflow diagram."""
    st.markdown("### 🔄 Analysis Workflow")

    workflow_html = """
<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 30px; border-radius: 15px; margin: 10px 0;">
<div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 20px;">
<div style="background: linear-gradient(135deg, #2E86AB, #1a5276); color: white; padding: 15px 18px; border-radius: 10px; min-width: 100px; text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.15);">
<strong>1. Calibration</strong><br><span style="font-size: 0.75em; opacity: 0.9;">Standard curve</span>
</div>
<span style="font-size: 1.3em; color: #888;">→</span>
<div style="background: linear-gradient(135deg, #A23B72, #7b2d56); color: white; padding: 15px 18px; border-radius: 10px; min-width: 100px; text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.15);">
<strong>2. Isotherms</strong><br><span style="font-size: 0.75em; opacity: 0.9;">Equilibrium</span>
</div>
<span style="font-size: 1.3em; color: #888;">→</span>
<div style="background: linear-gradient(135deg, #F18F01, #c47400); color: white; padding: 15px 18px; border-radius: 10px; min-width: 100px; text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.15);">
<strong>3. Kinetics</strong><br><span style="font-size: 0.75em; opacity: 0.9;">Time series</span>
</div>
<span style="font-size: 1.3em; color: #888;">→</span>
<div style="background: linear-gradient(135deg, #6C5B7B, #4a3f54); color: white; padding: 15px 18px; border-radius: 10px; min-width: 100px; text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.15);">
<strong>4. Thermo</strong><br><span style="font-size: 0.75em; opacity: 0.9;">ΔH°, ΔS°, ΔG°</span>
</div>
<span style="font-size: 1.3em; color: #888;">→</span>
<div style="background: linear-gradient(135deg, #1B998B, #147a6e); color: white; padding: 15px 18px; border-radius: 10px; min-width: 100px; text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.15);">
<strong>5. Compare</strong><br><span style="font-size: 0.75em; opacity: 0.9;">Multi-study</span>
</div>
</div>
<div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 12px;">
<div style="background: linear-gradient(135deg, #C73E1D, #9c3117); color: white; padding: 12px 16px; border-radius: 10px; min-width: 130px; text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.15);">
<strong>🔬 Effects</strong><br><span style="font-size: 0.75em; opacity: 0.9;">pH, T, Dosage</span>
</div>
<div style="background: linear-gradient(135deg, #E85A4F, #c44a40); color: white; padding: 12px 16px; border-radius: 10px; min-width: 130px; text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.15);">
<strong>🔮 3D Explorer</strong><br><span style="font-size: 0.75em; opacity: 0.9;">Visualization</span>
</div>
<div style="background: linear-gradient(135deg, #26A69A, #1e8e83); color: white; padding: 12px 16px; border-radius: 10px; min-width: 130px; text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.15);">
<strong>📦 Export All</strong><br><span style="font-size: 0.75em; opacity: 0.9;">TIFF, PNG, PDF</span>
</div>
</div>
<div style="text-align: center; margin-top: 15px;">
<span style="background: #28a745; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.75em;">v2.0.0</span>
</div>
</div>
"""
    st.markdown(workflow_html, unsafe_allow_html=True)


# =============================================================================
# CAPABILITIES
# =============================================================================
def _render_capabilities():
    """Render detailed capabilities."""
    st.markdown("### 🔬 Available Models & Analyses")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📈 Isotherm Models")
        st.markdown("""
        | Model | Parameters |
        |-------|------------|
        | Langmuir | qm, KL |
        | Freundlich | KF, n |
        | Temkin | B1, KT |
        | Sips | qm, KS, ns |
        """)

    with col2:
        st.markdown("#### ⏱️ Kinetic Models")
        st.markdown("""
        **Pseudo-Models:**
        | Model | Parameters |
        |-------|------------|
        | Pseudo-First Order | qe, k1 |
        | Pseudo-Second Order | qe, k2 |
        | Revised PSO (rPSO) | qe, k2, φ |
        | Elovich | α, β |
        """)

    with col3:
        st.markdown("#### 📊 Statistical Metrics")
        st.markdown("""
        | Metric | Purpose |
        |--------|---------|
        | R² | Goodness of fit |
        | Adjusted R² | Penalized fit |
        | RMSE | Error magnitude |
        | χ² (Chi-squared) | Residual analysis |
        | AIC | Model selection |
        | BIC | Model selection |
        | 95% CI | Uncertainty |
        | Akaike Weights | Model probability |
        """)


# =============================================================================
# TIPS SECTION
# =============================================================================
def _render_tips_section():
    """Render tips and best practices."""
    st.markdown("### 💡 Tips & Best Practices")

    with st.expander("📊 Calibration Best Practices", expanded=False):
        st.markdown("""
        - Use **6-10 concentration points** spanning your expected range
        - Always include a **blank (C = 0)**
        - Target **R² ≥ 0.999** for best results
        - Report confidence intervals for slope and intercept
        - Check residuals for systematic patterns
        """)

    with st.expander("📈 Isotherm Analysis Tips", expanded=False):
        st.markdown("""
        - Use **8-12 initial concentrations** for robust fitting
        - Ensure **true equilibrium** is reached (check with kinetics)
        - Compare multiple models using **AIC/BIC**, not just R²
        - Report **qm with 95% CI** from the best-fit model
        - Calculate and report the **separation factor (RL)** for Langmuir
        - For **multi-component systems** (real wastewaters):
          - Use Extended Langmuir for competitive adsorption
          - Calculate selectivity coefficients (α > 1 = preferred)
          - Single-component params needed first for prediction
        """)

    with st.expander("⏱️ Kinetic Analysis Tips", expanded=False):
        st.markdown("""
        - Sample frequently in **early time points** (0-30 min)
        - Continue until **plateau** is clearly reached
        - ⚠️ **PSO "best fit" does NOT prove chemisorption** — it's a statistical artifact observed in ~90% of studies
        - Use **rPSO** (revised PSO) for concentration-corrected kinetics
        - For mechanism identification, use **diffusion models**:
          - Boyd plot (Bt vs t): Linear through origin = film diffusion
          - Weber-Morris (qt vs √t): Linear through origin = pore diffusion
          - Biot number: Bi >> 1 = pore control, Bi << 1 = film control
        - Report both **qe and k** values with uncertainties
        - Consider **double-exponential model** for two-site kinetics
        """)

    with st.expander("🌡️ Thermodynamic Analysis Tips", expanded=False):
        st.markdown("""
        - Use **at least 4 temperatures** (e.g., 25, 35, 45, 55°C)
        - Ensure equilibrium at **each temperature**
        - Choose the appropriate **Kd calculation method**
        - Report ΔH°, ΔS°, and ΔG° with **confidence intervals**
        - Interpret mechanism based on |ΔH°| values
        """)

    with st.expander("📄 Analysis Checklist", expanded=False):
        st.markdown("""
        **Before submission, verify:**

        ✅ Calibration R² ≥ 0.999
        ✅ Multiple isotherm models compared (≥3)
        ✅ Multiple kinetic models compared (≥3)
        ✅ 95% CI reported for all key parameters
        ✅ AIC/BIC used for model selection
        ✅ Adjusted R² reported (not just R²)
        ✅ Residual plots show no systematic patterns
        ✅ Thermodynamic parameters with interpretation
        ✅ Figures exported at ≥300 DPI
        ✅ All units clearly stated

        **Mechanistic Interpretation:**
        ⚠️ Do NOT claim chemisorption based solely on PSO fit
        ✅ Use Boyd/Weber-Morris plots for diffusion mechanism
        ✅ Report activation energy (Ea) from temperature studies
        ✅ Consider spectroscopic evidence (FTIR, XPS) for mechanism
        """)

    # Footer tip
    st.info("""
    💡 **Pro Tip:** Use the **Multi-Study Comparison** tab to compare different studies
    side-by-side — whether comparing adsorbents, pollutants, or experimental conditions.
    """)
