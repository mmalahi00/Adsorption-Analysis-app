# tabs/calibration_tab.py
"""
Calibration Tab - AdsorbLab Pro
===============================

Establishes the absorbance-concentration relationship via linear regression.

Features:
- Full regression statistics with CI
- Residual analysis
- Prediction intervals
- LOD/LOQ estimation
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import t as t_dist

from adsorblab_pro.streamlit_compat import st

from ..config import EPSILON_DIV, FONT_FAMILY
from ..plot_style import (
    COLORS,
    MARKERS,
    apply_professional_style,
    create_residual_plot,
)
from ..utils import assess_data_quality, display_results_table, get_current_study_state
from ..validation import format_validation_errors, validate_calibration_data


def render():
    """Render the calibration tab with professional statistics."""
    st.subheader("📊 Calibration Curve Analysis")
    st.markdown("*Linear regression with confidence intervals and residual diagnostics*")

    current_study_state = get_current_study_state()
    if not current_study_state:
        st.info("Please add or select a study from the sidebar to begin analysis.")
        return

    # Get confidence level from study state
    confidence_level = current_study_state.get("confidence_level", 0.95)
    ci_pct = int(confidence_level * 100)

    calib_df = current_study_state.get("calib_df_input")
    calib_params = current_study_state.get("calibration_params")

    if calib_df is not None and len(calib_df) >= 3:
        # NEW: Validate calibration data before processing
        concentrations = (
            calib_df["Concentration"].values if "Concentration" in calib_df.columns else None
        )
        absorbances = calib_df["Absorbance"].values if "Absorbance" in calib_df.columns else None

        if concentrations is not None and absorbances is not None:
            validation_report = validate_calibration_data(concentrations, absorbances)

            if not validation_report.is_valid:
                st.error("❌ **Calibration Data Validation Failed**")
                st.markdown(format_validation_errors(validation_report))
                st.info("Please correct the data issues and re-upload.")
                return

            if validation_report.has_warnings:
                with st.expander("⚠️ Data Quality Notes", expanded=False):
                    for w in validation_report.warnings:
                        st.warning(f"{w.message}")
                        if w.suggestion:
                            st.caption(f"💡 {w.suggestion}")
        # Data-quality checks and R² are shown separately: R² alone does not
        # validate calibration range, lack of fit, replicate precision, or blanks.
        quality_report = assess_data_quality(calib_df, "calibration")
        r2 = calib_params.get("r_squared", np.nan) if calib_params else np.nan

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² (linearity indicator)", f"{r2:.6f}" if np.isfinite(r2) else "—")
        with col2:
            st.metric("Data Points", len(calib_df))
        with col3:
            st.metric("Data Checks", f"{quality_report['quality_score']}/100")

        with st.expander("📋 Quality Report", expanded=False):
            if calib_params:
                r2 = calib_params.get("r_squared", 0)
                st.info(
                    f"R² = {r2:.6f}. Treat this as a linearity indicator only; also inspect "
                    "residuals, replicate precision, blank response, and back-calculated standards."
                )

            # Data points check
            if len(calib_df) >= 6:
                st.success(f"✓ {len(calib_df)} calibration points available")
            elif len(calib_df) >= 5:
                st.info(f"ℹ {len(calib_df)} calibration points available")
            else:
                st.warning(f"⚠ Limited data: {len(calib_df)} points (recommend ≥5)")

            # Duplicates check
            if not calib_df.duplicated().any():
                st.success("✓ No duplicate values detected")

            # Show any issues from quality report
            if quality_report["issues"]:
                st.markdown("**Additional notes:**")
                for issue in quality_report["issues"]:
                    st.info(f"ℹ {issue}")

        st.markdown("---")

        if calib_params:
            # Section 1: Key Results (always visible)
            st.markdown("### 📈 Calibration Results")

            # Equation
            st.latex(
                rf"Absorbance = {calib_params['slope']:.6f} \times Concentration + {calib_params['intercept']:.6f}"
            )

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Slope", f"{calib_params['slope']:.6f}")
            with col2:
                st.metric("Intercept", f"{calib_params['intercept']:.6f}")
            with col3:
                st.metric("R²", f"{calib_params['r_squared']:.6f}")
            with col4:
                st.metric("Std. Error", f"{calib_params.get('std_err_estimate', np.nan):.6f}")

            # Advanced Statistics (collapsed)
            with st.expander("📊 Advanced Statistics", expanded=False):
                col1, col2 = st.columns([2, 1])

                with col1:
                    stats_df = pd.DataFrame(
                        {
                            "Parameter": [
                                "Slope",
                                "Intercept",
                                "R²",
                                "Adjusted R²",
                                "Std. Error of Estimate",
                            ],
                            "Value": [
                                f"{calib_params['slope']:.6f}",
                                f"{calib_params['intercept']:.6f}",
                                f"{calib_params['r_squared']:.6f}",
                                f"{calib_params.get('adj_r_squared', calib_params['r_squared']):.6f}",
                                f"{calib_params.get('std_err_estimate', 0):.6f}",
                            ],
                            "Std. Error": [
                                f"{calib_params.get('std_err_slope', 0):.6f}",
                                f"{calib_params.get('std_err_intercept', 0):.6f}",
                                "—",
                                "—",
                                "—",
                            ],
                            f"{ci_pct}% CI": [
                                f"({calib_params['slope_ci_95'][0]:.6f}, {calib_params['slope_ci_95'][1]:.6f})"
                                if "slope_ci_95" in calib_params
                                else "—",
                                f"({calib_params['intercept_ci_95'][0]:.6f}, {calib_params['intercept_ci_95'][1]:.6f})"
                                if "intercept_ci_95" in calib_params
                                else "—",
                                "—",
                                "—",
                                "—",
                            ],
                        }
                    )
                    display_results_table(stats_df)

                with col2:
                    # p-value significance
                    p_val = calib_params.get("p_value", 0)
                    if p_val < 0.001:
                        st.success("**p-value:** < 0.001 ***")
                    elif p_val < 0.01:
                        st.success(f"**p-value:** {p_val:.4f} **")
                    elif p_val < 0.05:
                        st.info(f"**p-value:** {p_val:.4f} *")
                    else:
                        st.warning(f"**p-value:** {p_val:.4f} ns")

                    st.markdown("""
                    **Significance:**
                    - *** p < 0.001
                    - ** p < 0.01
                    - * p < 0.05
                    - ns: not significant
                    """)

            st.markdown("---")

            # Section 2: Calibration Plot (always visible - essential)
            st.markdown("### 📊 Calibration Plot")

            conc = calib_df["Concentration"].values
            abs_val = calib_df["Absorbance"].values

            # Predictions
            x_line = np.linspace(0, conc.max() * 1.1, 100)
            y_line = calib_params["slope"] * x_line + calib_params["intercept"]

            # Confidence band calculation
            n = len(conc)
            x_mean = np.mean(conc)
            se_estimate = calib_params.get("std_err_estimate", 0)
            alpha = 1 - confidence_level
            t_val = t_dist.ppf(1 - alpha / 2, n - 2) if n > 2 else 2.0

            # Standard error of prediction for CI band
            ss_xx = np.sum((conc - x_mean) ** 2)
            se_y = se_estimate * np.sqrt(1 / n + (x_line - x_mean) ** 2 / ss_xx) if ss_xx > 0 else 0

            ci_upper = y_line + t_val * se_y
            ci_lower = y_line - t_val * se_y

            # Create professional plot
            fig = go.Figure()

            # CI band (added first so it's behind other traces)
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_line, x_line[::-1]]),
                    y=np.concatenate([ci_upper, ci_lower[::-1]]),
                    fill="toself",
                    fillcolor=COLORS["ci_fill"],
                    line={"color": "rgba(255,255,255,0)"},
                    name=f"{ci_pct}% CI",
                    hoverinfo="skip",
                )
            )

            # Regression line (fit)
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Linear Fit",
                    line={"color": COLORS["fit_primary"], "width": 2.5},
                )
            )

            # Experimental data points
            fig.add_trace(
                go.Scatter(
                    x=conc,
                    y=abs_val,
                    mode="markers",
                    name="Experimental",
                    marker=MARKERS["experimental"],
                    hovertemplate="Conc: %{x:.2f} mg/L<br>Abs: %{y:.4f}<extra></extra>",
                )
            )

            fig = apply_professional_style(
                fig,
                title=f"Calibration Curve with {ci_pct}% Confidence Band",
                x_title="Concentration (mg/L)",
                y_title="Absorbance",
                height=500,
            )

            # Equation annotation (positioned to not overlap legend)
            intercept = calib_params["intercept"]
            intercept_sign = "+" if intercept >= 0 else "−"
            intercept_val = abs(intercept)
            fig.add_annotation(
                x=0.98,
                y=0.05,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="bottom",
                text=f"y = {calib_params['slope']:.4f}x {intercept_sign} {intercept_val:.4f}<br>R² = {calib_params['r_squared']:.6f}",
                showarrow=False,
                font={"size": 12, "family": FONT_FAMILY},
                align="right",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
            )

            st.plotly_chart(fig, use_container_width=True, key="calib_main_chart")

            # Section 3: Residual Diagnostics (collapsed)
            st.markdown("---")

            # Calculate residuals for all sections
            abs_pred = calib_params["slope"] * conc + calib_params["intercept"]
            residuals = abs_val - abs_pred
            std_residuals = (
                residuals / np.std(residuals) if np.std(residuals) > EPSILON_DIV else residuals
            )

            with st.expander("🔍 Residual Diagnostics", expanded=False):
                # Data table
                st.markdown("**Data Table with Residuals:**")
                results_df = pd.DataFrame(
                    {
                        "Concentration (mg/L)": conc,
                        "Absorbance (exp)": abs_val,
                        "Absorbance (pred)": np.round(abs_pred, 6),
                        "Residual": np.round(residuals, 6),
                        "Std. Residual": np.round(std_residuals, 4),
                    }
                )
                display_results_table(results_df, hide_index=False)

                st.markdown("---")
                st.markdown("**Residual Plots:**")

                col1, col2 = st.columns(2)

                with col1:
                    # Residuals vs Fitted - Professional styling
                    fig_res = create_residual_plot(abs_pred, residuals, model_name="Calibration")
                    st.plotly_chart(fig_res, use_container_width=True, key="calib_residual_chart")

                with col2:
                    # Q-Q Plot - Professional styling
                    sorted_residuals = np.sort(std_residuals)
                    theoretical_quantiles = stats.norm.ppf(
                        np.linspace(0.05, 0.95, len(sorted_residuals))
                    )

                    fig_qq = go.Figure()
                    fig_qq.add_trace(
                        go.Scatter(
                            x=theoretical_quantiles,
                            y=sorted_residuals,
                            mode="markers",
                            name="Residuals",
                            marker=MARKERS["experimental"],
                            hovertemplate="Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>",
                        )
                    )
                    qq_line = np.linspace(
                        min(theoretical_quantiles), max(theoretical_quantiles), 100
                    )
                    fig_qq.add_trace(
                        go.Scatter(
                            x=qq_line,
                            y=qq_line,
                            mode="lines",
                            name="Reference",
                            line={"color": COLORS["fit_primary"], "dash": "dash", "width": 2},
                            showlegend=False,
                        )
                    )
                    fig_qq = apply_professional_style(
                        fig_qq,
                        title="Normal Q-Q Plot",
                        x_title="Theoretical Quantiles",
                        y_title="Standardized Residuals",
                        height=350,
                        show_legend=False,
                    )
                    st.plotly_chart(fig_qq, use_container_width=True, key="calib_qq_chart")

                # Residual statistics
                st.markdown("---")
                st.markdown("**Residual Statistics:**")
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                with res_col1:
                    st.metric("Mean Residual", f"{np.mean(residuals):.6f}")
                with res_col2:
                    st.metric("Std. Dev.", f"{np.std(residuals):.6f}")
                with res_col3:
                    st.metric("Max |Residual|", f"{np.max(np.abs(residuals)):.6f}")
                with res_col4:
                    # Durbin-Watson
                    dw = (
                        np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2)
                        if np.sum(residuals**2) > EPSILON_DIV
                        else 2.0
                    )
                    st.metric("Durbin-Watson", f"{dw:.3f}")

                st.markdown("""
                **Interpretation:**
                - **Mean Residual** ≈ 0 indicates unbiased predictions
                - **Durbin-Watson** ≈ 2 indicates no autocorrelation (1.5-2.5 acceptable)
                - Q-Q plot points on diagonal indicate normally distributed residuals
                """)

            # Section 4: LOD/LOQ Estimation (already collapsed)
            with st.expander("📐 Detection Limits (LOD/LOQ)", expanded=False):
                st.markdown("""
                **Limit of Detection (LOD):** 3 × σ / slope
                **Limit of Quantification (LOQ):** 10 × σ / slope

                Where σ is the standard deviation of the regression.
                """)

                sigma = se_estimate
                slope = calib_params["slope"]

                if slope > EPSILON_DIV:
                    lod = 3 * sigma / slope
                    loq = 10 * sigma / slope

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("LOD (mg/L)", f"{lod:.4f}")
                    with col2:
                        st.metric("LOQ (mg/L)", f"{loq:.4f}")

            # Export reminder
            st.markdown("---")
            st.info("💡 **Export:** Go to **📦 Export All** tab for high-resolution figures.")

        else:
            st.warning("⚠️ Calibration calculation failed. Check your data.")

    else:
        st.info("📥 Enter calibration data in the sidebar to begin analysis.")

        with st.expander("📖 Calibration Guidelines", expanded=True):
            st.markdown("""
            **Best Practices:**

            1. **Minimum Points:** Use 6-10 concentration levels
            2. **Include Blank:** Always include C = 0 (blank)
            3. **Range:** Cover expected sample concentration range
            4. **Linearity:** define an acceptance criterion for your method and inspect residuals
            5. **Replicates:** Prepare triplicates for error estimation
            6. **Report:** Include slope, intercept, R², and confidence intervals

            **Required Format:**
            | Concentration | Absorbance |
            |--------------|------------|
            | 0 | 0.002 |
            | 5 | 0.125 |
            | 10 | 0.248 |
            | ... | ... |
            """)
