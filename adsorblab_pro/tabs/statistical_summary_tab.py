# tabs/statistical_summary_tab.py
"""
Study Overview Tab - AdsorbLab Pro
==================================

Provides a comprehensive dashboard for the active study.

Features:
- Status cards for all analyses
- Parameter summaries with 95% CI
- Analysis completeness checklist
"""

import numpy as np
import pandas as pd

from adsorblab_pro.streamlit_compat import st

from ..config import QUALITY_THRESHOLDS
from ..utils import (
    calculate_akaike_weights,
    check_mechanism_consistency,
    detect_common_errors,
    display_results_table,
    get_current_study_state,
)


def render():
    """Render study overview dashboard."""
    st.subheader("📊 Study Overview")
    st.markdown("*Quick summary of all analyses for the active study*")

    current_study_state = get_current_study_state()
    if not current_study_state:
        st.info("Please add or select a study from the sidebar to see a summary.")
        return

    # Collect all results from the active study
    calib_params = current_study_state.get("calibration_params")
    iso_models = current_study_state.get("isotherm_models_fitted", {})
    kin_models = current_study_state.get("kinetic_models_fitted", {})
    thermo_params = current_study_state.get("thermo_params")

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        calib_status = "✅" if calib_params else "❌"
        calib_r2 = f"{calib_params['r_squared']:.4f}" if calib_params else "N/A"
        st.metric("Calibration", calib_status, calib_r2)

    with col2:
        iso_status = "✅" if iso_models else "❌"
        n_iso = len([m for m in iso_models.values() if m and m.get("converged")])
        st.metric("Isotherms", iso_status, f"{n_iso} models")

    with col3:
        kin_status = "✅" if kin_models else "❌"
        n_kin = len([m for m in kin_models.values() if m and m.get("converged")])
        st.metric("Kinetics", kin_status, f"{n_kin} models")

    with col4:
        thermo_status = "✅" if thermo_params else "❌"
        st.metric("Thermodynamics", thermo_status)

    st.markdown("---")

    # Section 1: Calibration Summary
    if calib_params:
        st.markdown("### 1. 📊 Calibration Summary")

        calib_df = pd.DataFrame(
            {
                "Parameter": ["Slope", "Intercept", "R²", "Adjusted R²", "Std. Error (slope)", "n"],
                "Value": [
                    f"{calib_params['slope']:.6f}",
                    f"{calib_params['intercept']:.6f}",
                    f"{calib_params['r_squared']:.6f}",
                    f"{calib_params.get('adj_r_squared', calib_params['r_squared']):.6f}",
                    f"{calib_params.get('std_err_slope', 0):.6f}",
                    str(calib_params.get("n_points", "N/A")),
                ],
                "95% CI": [
                    f"({calib_params.get('slope_ci_95', (np.nan, np.nan))[0]:.6f}, {calib_params.get('slope_ci_95', (np.nan, np.nan))[1]:.6f})",
                    f"({calib_params.get('intercept_ci_95', (np.nan, np.nan))[0]:.6f}, {calib_params.get('intercept_ci_95', (np.nan, np.nan))[1]:.6f})",
                    "—",
                    "—",
                    "—",
                    "—",
                ],
            }
        )
        display_results_table(calib_df)

    # Section 2: Isotherm Models Summary
    if iso_models:
        st.markdown("---")
        st.markdown("### 2. 📈 Isotherm Models Summary")

        iso_data = []
        for name, results in iso_models.items():
            if results and results.get("converged"):
                params = results.get("params", {})

                row = {
                    "Model": name,
                    "R²": results.get("r_squared", 0),
                    "Adj-R²": results.get("adj_r_squared", 0),
                    "RMSE": results.get("rmse", np.inf),
                    "χ²": results.get("chi_squared", np.inf),
                    "AIC": results.get("aicc", results.get("aic", np.inf)),
                    "BIC": results.get("bic", np.inf),
                }

                # Add key parameters
                if name == "Langmuir":
                    row["Key Parameter"] = f"qm = {params.get('qm', 0):.2f} mg/g"
                elif name == "Freundlich":
                    row["Key Parameter"] = (
                        f"KF = {params.get('KF', 0):.2f}, n = {params.get('n', 0):.2f}"
                    )
                elif name == "Temkin":
                    row["Key Parameter"] = f"B1 = {params.get('B1', 0):.2f}"
                elif name == "Sips":
                    row["Key Parameter"] = (
                        f"qm = {params.get('qm', 0):.2f}, ns = {params.get('ns', 0):.2f}"
                    )
                else:
                    row["Key Parameter"] = "—"

                iso_data.append(row)

        if iso_data:
            iso_df = pd.DataFrame(iso_data)

            # Calculate AIC weights
            weights = calculate_akaike_weights(iso_df["AIC"].tolist())
            iso_df["AIC Weight"] = [f"{w:.1%}" for w in weights]

            # Style and display
            st.dataframe(
                iso_df.style.format(
                    {
                        "R²": "{:.4f}",
                        "Adj-R²": "{:.4f}",
                        "RMSE": "{:.4f}",
                        "χ²": "{:.2f}",
                        "AIC": "{:.2f}",
                        "BIC": "{:.2f}",
                    }
                )
                .highlight_max(subset=["R²", "Adj-R²"], color="lightgreen")
                .highlight_min(subset=["RMSE", "AIC", "BIC", "χ²"], color="lightblue"),
                use_container_width=True,
            )

            # Best model
            try:
                best_idx = iso_df["Adj-R²"].idxmax()
                if pd.notna(best_idx):
                    best_model = iso_df.loc[best_idx, "Model"]
                    st.success(
                        f"**Best Isotherm Model:** {best_model} (Adj-R² = {iso_df.loc[best_idx, 'Adj-R²']:.4f})"
                    )
            except (KeyError, ValueError):
                pass

    # Section 3: Kinetic Models Summary
    if kin_models:
        st.markdown("---")
        st.markdown("### 3. ⏱️ Kinetic Models Summary")

        kin_data = []
        for name, results in kin_models.items():
            if results and results.get("converged"):
                params = results.get("params", {})

                row = {
                    "Model": name,
                    "R²": results.get("r_squared", 0),
                    "Adj-R²": results.get("adj_r_squared", 0),
                    "RMSE": results.get("rmse", np.inf),
                    "AIC": results.get("aicc", results.get("aic", np.inf)),
                }

                # Key parameters
                if name == "PFO":
                    row["qe (mg/g)"] = params.get("qe", 0)
                    row["k"] = f"k₁ = {params.get('k1', 0):.4f} min⁻¹"
                elif name == "PSO":
                    row["qe (mg/g)"] = params.get("qe", 0)
                    row["k"] = f"k₂ = {params.get('k2', 0):.6f} g/(mg·min)"
                elif name == "rPSO":
                    row["qe (mg/g)"] = params.get("qe", 0)
                    row["k"] = f"k₂ = {params.get('k2', 0):.6f} g/(mg·min)"
                elif name == "Elovich":
                    row["qe (mg/g)"] = "—"
                    row["k"] = f"α = {params.get('alpha', 0):.2f}"
                elif name == "IPD":
                    row["qe (mg/g)"] = "—"
                    row["k"] = f"kid = {params.get('kid', 0):.4f}"
                else:
                    row["qe (mg/g)"] = "—"
                    row["k"] = "—"

                kin_data.append(row)

        if kin_data:
            kin_df = pd.DataFrame(kin_data)

            st.dataframe(
                kin_df.style.format(
                    {
                        "R²": "{:.4f}",
                        "Adj-R²": "{:.4f}",
                        "RMSE": "{:.4f}",
                        "AIC": "{:.2f}",
                        "qe (mg/g)": "{:.2f}"
                        if pd.api.types.is_numeric_dtype(kin_df["qe (mg/g)"])
                        else "{}",
                    }
                )
                .highlight_max(subset=["R²", "Adj-R²"], color="lightgreen")
                .highlight_min(subset=["RMSE", "AIC"], color="lightblue"),
                use_container_width=True,
            )

            # Best model
            try:
                best_idx = kin_df["Adj-R²"].idxmax()
                if pd.notna(best_idx):
                    best_model = kin_df.loc[best_idx, "Model"]
                    st.success(
                        f"**Best Kinetic Model:** {best_model} (Adj-R² = {kin_df.loc[best_idx, 'Adj-R²']:.4f})"
                    )
                    st.caption(
                        "⚠️ Note: Best statistical fit ≠ mechanistic evidence. Use Boyd/Weber-Morris plots and activation energy for mechanism identification."
                    )
            except (KeyError, ValueError):
                pass

    # Section 4: Thermodynamic Summary
    if thermo_params:
        st.markdown("---")
        st.markdown("### 4. 🌡️ Thermodynamic Parameters Summary")

        delta_H = thermo_params["delta_H"]
        delta_S = thermo_params["delta_S"]
        delta_G = delta_H - 298.15 * delta_S / 1000  # kJ/mol

        thermo_df = pd.DataFrame(
            {
                "Parameter": [
                    "ΔH° (kJ/mol)",
                    "ΔS° (J/(mol·K))",
                    "ΔG° at 298K (kJ/mol)",
                    "R² (Van't Hoff)",
                ],
                "Value": [
                    f"{delta_H:.2f}",
                    f"{delta_S:.2f}",
                    f"{delta_G:.2f}",
                    f"{thermo_params['r_squared']:.4f}",
                ],
                "95% CI": [
                    f"({thermo_params.get('delta_H_ci_95', (np.nan, np.nan))[0]:.2f}, {thermo_params.get('delta_H_ci_95', (np.nan, np.nan))[1]:.2f})"
                    if "delta_H_ci_95" in thermo_params
                    else "—",
                    f"({thermo_params.get('delta_S_ci_95', (np.nan, np.nan))[0]:.2f}, {thermo_params.get('delta_S_ci_95', (np.nan, np.nan))[1]:.2f})"
                    if "delta_S_ci_95" in thermo_params
                    else "—",
                    "—",
                    "—",
                ],
                "Interpretation": [
                    "Exothermic" if delta_H < 0 else "Endothermic",
                    "Increased disorder" if delta_S > 0 else "Decreased disorder",
                    "Spontaneous" if delta_G < 0 else "Non-spontaneous",
                    "Excellent" if thermo_params["r_squared"] > 0.99 else "Good",
                ],
            }
        )
        display_results_table(thermo_df)

        # Mechanism
        abs_H = abs(delta_H)
        if abs_H < 40:
            mechanism = "Physical Adsorption"
        elif abs_H < 80:
            mechanism = "Mixed Mechanism"
        else:
            mechanism = "Chemical Adsorption"

        st.info(f"**Adsorption Mechanism:** {mechanism} (|ΔH°| = {abs_H:.2f} kJ/mol)")

    # ==========================================================================
    # Section 5: MECHANISM CONSISTENCY CHECK (NEW)
    # ==========================================================================
    st.markdown("---")
    _render_consistency_check(current_study_state)

    # ==========================================================================
    # Section 6: COMMON ERRORS DETECTION (NEW)
    # ==========================================================================
    st.markdown("---")
    _render_common_errors(current_study_state)

    # Section 7: Analysis Checklist
    st.markdown("---")
    st.markdown("### 7. 📝 Analysis Checklist")

    # Build checklist with explicit boolean values
    checklist = {
        f"Calibration R² ≥ {QUALITY_THRESHOLDS['calibration']['ideal_r_squared']}": False,
        "Multiple isotherm models compared": False,
        "Multiple kinetic models compared": False,
        "95% CI reported for parameters": False,
        "AIC/BIC used for model selection": False,
        "Thermodynamic analysis complete": False,
        "Adjusted R² reported for models": False,
    }

    if calib_params:
        thresh = QUALITY_THRESHOLDS["calibration"]["ideal_r_squared"]
        checklist[f"Calibration R² ≥ {thresh}"] = calib_params.get("r_squared", 0) >= thresh

    if iso_models:
        n_iso_converged = len([m for m in iso_models.values() if m and m.get("converged")])
        checklist["Multiple isotherm models compared"] = n_iso_converged >= 2

    if kin_models:
        n_kin_converged = len([m for m in kin_models.values() if m and m.get("converged")])
        checklist["Multiple kinetic models compared"] = n_kin_converged >= 2

    if iso_models or kin_models:
        checklist["95% CI reported for parameters"] = True
        checklist["AIC/BIC used for model selection"] = True
        checklist["Adjusted R² reported for models"] = True

    if thermo_params:
        checklist["Thermodynamic analysis complete"] = True

    # --- Render the checklist ---
    col1, col2 = st.columns(2)

    items = list(checklist.items())
    midpoint = (len(items) + 1) // 2

    with col1:
        for item, status in items[:midpoint]:
            if status:
                st.success(f"✅ {item}")
            else:
                st.warning(f"⬜ {item}")

    with col2:
        for item, status in items[midpoint:]:
            if status:
                st.success(f"✅ {item}")
            else:
                st.warning(f"⬜ {item}")

    # --- Render the progress bar ---
    completed = sum(checklist.values())
    total = len(checklist)
    progress = completed / total if total > 0 else 0

    st.progress(progress)
    st.caption(f"**Analysis completeness: {completed}/{total} ({progress:.0%})**")

    if progress >= 0.8:
        st.success("🎉 Comprehensive analysis complete!")
    elif progress >= 0.5:
        st.info("📊 Good progress! Complete remaining analyses.")
    else:
        st.warning("📝 Continue with your analyses.")

    # Export reminder
    st.markdown("---")
    st.info("💡 **To export all figures and tables:** Go to **📦 Export All** tab")


# =============================================================================
# MECHANISM CONSISTENCY CHECK UI
# =============================================================================
def _render_consistency_check(study_state: dict):
    """
    Render the mechanism consistency check panel.

    Shows conflicts between different analysis methods:
    - Kinetic model vs isotherm model
    - Temperature effect vs ΔH° sign
    - Separation factor vs Freundlich 1/n

    Parameters
    ----------
    study_state : dict
        Current study state containing all analysis results
    """
    st.markdown("### 🔍 Mechanism Consistency Check")
    st.markdown("*Cross-validation of your analysis results*")

    # Check if we have enough data to run checks
    iso_models = study_state.get("isotherm_models_fitted", {})
    kin_models = study_state.get("kinetic_models_fitted", {})
    thermo_params = study_state.get("thermo_params")

    has_analyses = bool(iso_models) or bool(kin_models) or bool(thermo_params)

    if not has_analyses:
        st.info("""
        **No checks available yet.**

        Complete isotherm, kinetic, or thermodynamic analyses to enable consistency checking.

        The checker validates:
        - Kinetic model vs. isotherm model agreement
        - Temperature effect vs. ΔH° sign
        - Separation factor vs. Freundlich behavior
        """)
        return

    # Run the consistency check
    result = check_mechanism_consistency(study_state)

    # Count checks performed
    n_checks = len(result["checks"])
    n_passed = sum(1 for c in result["checks"] if c["status"] == "consistent")

    # Define status configurations
    status_configs = {
        "consistent": {
            "icon": "✅",
            "bg": "linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%)",
            "border": "#28a745",
            "text": "#155724",
            "title": "All Clear",
        },
        "minor_issues": {
            "icon": "⚠️",
            "bg": "linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%)",
            "border": "#ffc107",
            "text": "#856404",
            "title": "Review Recommended",
        },
        "conflicts": {
            "icon": "❌",
            "bg": "linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%)",
            "border": "#dc3545",
            "text": "#721c24",
            "title": "Conflicts Detected",
        },
    }

    config = status_configs.get(result["status"], status_configs["consistent"])

    # Overall status banner
    st.markdown(
        f"""
    <div style="background: {config["bg"]}; padding: 20px; border-radius: 10px;
                margin-bottom: 15px; border-left: 5px solid {config["border"]};
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 2em;">{config["icon"]}</span>
            <div>
                <h4 style="color: {config["text"]}; margin: 0; font-size: 1.15em;">
                    {result["interpretation"]}
                </h4>
                <p style="color: {config["text"]}; margin: 5px 0 0 0; font-size: 0.9em; opacity: 0.8;">
                    {n_passed}/{n_checks} checks passed
                </p>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Detailed check results in expander
    if result["checks"]:
        # Auto-expand if there are issues
        expand_default = result["status"] != "consistent"

        with st.expander("📋 Detailed Check Results", expanded=expand_default):
            # Group checks by status for better organization
            conflicts = [c for c in result["checks"] if c["status"] == "conflict"]
            minor = [c for c in result["checks"] if c["status"] == "minor"]
            passed = [c for c in result["checks"] if c["status"] == "consistent"]

            # Show conflicts first (most critical)
            if conflicts:
                st.markdown("#### ❌ Conflicts Detected")
                for check in conflicts:
                    st.error(f"**{check['name']}**\n\n{check['message']}")
                st.markdown("")

            # Show minor issues
            if minor:
                st.markdown("#### ⚠️ Minor Issues")
                for check in minor:
                    st.warning(f"**{check['name']}**\n\n{check['message']}")
                st.markdown("")

            # Show passed checks
            if passed:
                st.markdown("#### ✅ Passed Checks")
                for check in passed:
                    st.success(f"**{check['name']}**\n\n{check['message']}")

    # Suggestions section (only show if there are issues)
    if result["suggestions"] and result["status"] != "consistent":
        st.markdown("#### 💡 Recommendations")
        for i, suggestion in enumerate(result["suggestions"], 1):
            st.markdown(f"{i}. {suggestion}")

    # Educational information
    with st.expander("ℹ️ What does this check?"):
        st.markdown("""
        The **Mechanism Consistency Check** cross-validates your results to ensure
        different analyses tell a consistent story about the adsorption mechanism.

        | Check | What It Validates |
        |-------|------------------|
        | **Kinetic-Isotherm** | Best-fit kinetic model should be consistent with isotherm type |
        | **Temperature-ΔH°** | If ΔH° > 0 (endothermic), capacity should increase with temperature |
        | **RL vs 1/n** | Langmuir separation factor should agree with Freundlich exponent |
        | **R² Reporting** | High R² values should be accompanied by confidence intervals |

        ---

        **Interpretation of ΔH°:**
        | |ΔH°| Range | Mechanism |
        |-------------|-----------|
        | < 40 kJ/mol | Physical adsorption |
        | 40-80 kJ/mol | Mixed mechanism |
        | > 80 kJ/mol | Chemical adsorption |

        ---
        """)


# =============================================================================
# COMMON ERRORS DETECTION UI
# =============================================================================
def _render_common_errors(study_state: dict):
    """
    Render common methodological errors detection panel.

    Helps researchers avoid common mistakes in adsorption studies that
    could lead to unreliable conclusions or paper rejections.

    Parameters
    ----------
    study_state : dict
        Current study state containing analysis results
    """
    st.markdown("### ⚠️ Methodological Quality Check")
    st.markdown("*Detection of common errors in adsorption research*")

    # Run error detection
    errors = detect_common_errors(study_state)

    if not errors:
        st.markdown(
            """
        <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                    padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 2em;">✅</span>
                <div>
                    <h4 style="color: #155724; margin: 0;">No Common Errors Detected</h4>
                    <p style="color: #155724; margin: 5px 0 0 0; opacity: 0.8;">
                        Your methodology appears sound based on available data
                    </p>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return

    # Count by severity
    high_count = sum(1 for e in errors if e["severity"] == "HIGH")
    medium_count = sum(1 for e in errors if e["severity"] == "MEDIUM")
    low_count = sum(1 for e in errors if e["severity"] == "LOW")

    # Determine overall status based on severity
    if high_count > 0:
        bg_color = "linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%)"
        border_color = "#dc3545"
        text_color = "#721c24"
        icon = "❌"
        status_text = f"{high_count} critical issue(s) require attention"
    elif medium_count > 0:
        bg_color = "linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%)"
        border_color = "#ffc107"
        text_color = "#856404"
        icon = "⚠️"
        status_text = f"{medium_count} issue(s) should be reviewed"
    else:
        bg_color = "linear-gradient(135deg, #cce5ff 0%, #b8daff 100%)"
        border_color = "#004085"
        text_color = "#004085"
        icon = "ℹ️"
        status_text = f"{low_count} minor suggestion(s)"

    # Status banner
    st.markdown(
        f"""
    <div style="background: {bg_color}; padding: 20px; border-radius: 10px;
                margin-bottom: 15px; border-left: 5px solid {border_color};
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 2em;">{icon}</span>
            <div>
                <h4 style="color: {text_color}; margin: 0;">{status_text}</h4>
                <p style="color: {text_color}; margin: 5px 0 0 0; opacity: 0.8;">
                    {len(errors)} total issue(s) detected
                </p>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Detailed issues grouped by severity
    with st.expander("📋 Detailed Issues", expanded=(high_count > 0)):
        # HIGH severity first (most critical)
        high_errors = [e for e in errors if e["severity"] == "HIGH"]
        if high_errors:
            st.markdown("#### 🔴 Critical Issues")
            for error in high_errors:
                st.error(f"""
                **{error["type"]}**: {error["message"]}

                💡 *{error["recommendation"]}*
                """)
            st.markdown("")

        # MEDIUM severity
        medium_errors = [e for e in errors if e["severity"] == "MEDIUM"]
        if medium_errors:
            st.markdown("#### 🟡 Moderate Issues")
            for error in medium_errors:
                st.warning(f"""
                **{error["type"]}**: {error["message"]}

                💡 *{error["recommendation"]}*
                """)
            st.markdown("")

        # LOW severity
        low_errors = [e for e in errors if e["severity"] == "LOW"]
        if low_errors:
            st.markdown("#### 🔵 Suggestions")
            for error in low_errors:
                st.info(f"""
                **{error["type"]}**: {error["message"]}

                💡 *{error["recommendation"]}*
                """)

    # Educational section
    with st.expander("ℹ️ Why these checks matter"):
        st.markdown("""
        These checks help identify common mistakes that lead to unreliable conclusions
        or paper rejections in adsorption research.

        | Check | Why It Matters |
        |-------|----------------|
        | **ΔG° Range** | Values outside -60 to +10 kJ/mol often indicate calculation or unit errors |
        | **Data Points** | Too few points for model parameters leads to overfitting and unreliable parameters |
        | **R² without CI** | High R² alone doesn't prove model validity; confidence intervals are essential |
        | **Heteroscedasticity** | Non-constant variance violates regression assumptions, biasing results |
        | **Linear vs Non-linear** | Linearization can introduce systematic bias in parameter estimates |
        | **ΔH° vs Temperature** | Thermodynamic inconsistency suggests calculation or experimental errors |
        | **Wide CI** | Very uncertain parameters suggest model is too complex for available data |

        ---

        **Common sources of these errors:**

        1. **Unit conversion mistakes** — especially J vs kJ for thermodynamic parameters
        2. **Insufficient equilibrium time** — leading to scattered kinetic data
        3. **Using linearized models** — when non-linear fitting is more appropriate
        4. **Over-parameterized models** — using 3-parameter models with only 5-6 data points
        5. **Calculation errors** — especially in Kd for thermodynamic analysis

        Addressing these issues significantly improves the reliability of your conclusions
        and the likelihood of successful peer review.
        """)
