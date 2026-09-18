# tabs/thermodynamics_tab.py
"""
Thermodynamics Tab - AdsorbLab Pro
==================================

Van't Hoff analysis for thermodynamic parameter determination.

Features:
- Van't Hoff equation analysis
- ΔH°, ΔS°, ΔG° calculation with 95% CI
- Multiple Kd calculation methods
- Thermodynamic trend summaries with explicit limitations
- Professional outputs
"""

import numpy as np
import pandas as pd

from adsorblab_pro.streamlit_compat import st

from ..plot_style import create_vant_hoff_plot
from ..utils import (
    calculate_temperature_results,
    calculate_temperature_results_direct,
    calculate_thermodynamic_parameters,
    display_results_table,
    get_current_study_state,
    interpret_thermodynamics,
    propagate_kd_uncertainty,
    validate_required_params,
)
from ..validation import format_validation_errors, validate_thermodynamic_data

# =============================================================================
# Kd CALCULATION METHODS
# =============================================================================
KD_METHODS = {
    "Dimensionless: (C₀-Cₑ)/Cₑ": {
        "id": "dimensionless",
        "formula": r"K_d = \frac{C_0 - C_e}{C_e}",
        "units": "dimensionless",
        "description": (
            "Operational dimensionless concentration ratio. Report the definition explicitly; "
            "it is not automatically a standard-state thermodynamic equilibrium constant."
        ),
        "reference": "Liu, Y. (2009). J. Chem. Eng. Data, 54, 1981-1985.",
    },
    "Mass-based: qₑ/Cₑ (L/g)": {
        "id": "mass_based",
        "formula": r"K_d = \frac{q_e}{C_e}",
        "units": "L/g",
        "description": "Common in literature but not dimensionless. Results are 'apparent' values.",
        "reference": "Most adsorption papers use this form.",
    },
    "Volume-corrected: (qₑ×m)/(Cₑ×V)": {
        "id": "volume_corrected",
        "formula": r"K_d = \frac{q_e \times m}{C_e \times V}",
        "units": "dimensionless",
        "description": (
            "Operational dimensionless ratio using experimental parameters; not automatically "
            "a standard-state equilibrium constant."
        ),
        "reference": "Milonjić, S.K. (2007). J. Serb. Chem. Soc., 72, 1363-1367.",
    },
}


def _calculate_kd(
    method_id: str, C0: float, Ce: np.ndarray, qe: np.ndarray, m: float, V: float
) -> np.ndarray:
    """
    Calculate distribution coefficient using selected method.

    Parameters
    ----------
    method_id : str
        One of: 'dimensionless', 'mass_based', 'volume_corrected'
    C0 : float
        Initial concentration (mg/L)
    Ce : np.ndarray
        Equilibrium concentration (mg/L)
    qe : np.ndarray
        Adsorption capacity (mg/g)
    m : float
        Adsorbent mass (g)
    V : float
        Solution volume (L)

    Returns
    -------
    np.ndarray
        Distribution coefficient values
    """
    Ce = np.asarray(Ce, dtype=float)
    qe = np.asarray(qe, dtype=float)

    def _rows(mask: np.ndarray) -> str:
        """1-based row numbers of the offending observations, for the error text."""
        return ", ".join(str(int(i) + 1) for i in np.flatnonzero(np.atleast_1d(mask)))

    bad = ~np.isfinite(Ce) | ~np.isfinite(qe)
    if np.any(bad):
        raise ValueError(f"Ce and qe must contain only finite values (row(s) {_rows(bad)}).")
    bad = Ce <= 0
    if np.any(bad):
        raise ValueError(
            f"All Ce values must be greater than zero to calculate ln(Kd) (row(s) {_rows(bad)})."
        )
    # Both dimensionless and volume-corrected are mass-balance ratios of the
    # amount adsorbed to the amount remaining, so both require Ce < C0.  The
    # volume-corrected form expresses that through qe rather than through Ce
    # directly, but the physical requirement is identical, and checking only
    # one of them left the two algebraically-equivalent methods with different
    # admissible input ranges.
    if method_id in {"dimensionless", "volume_corrected"}:
        bad = Ce >= C0
        if np.any(bad):
            raise ValueError(
                f"{method_id} Kd requires 0 < Ce < C0 for every observation "
                f"(row(s) {_rows(bad)} have Ce >= C0)."
            )
    if method_id in {"mass_based", "volume_corrected"}:
        bad = qe <= 0
        if np.any(bad):
            raise ValueError(f"{method_id} Kd requires positive qe values (row(s) {_rows(bad)}).")
    if method_id == "volume_corrected" and (m <= 0 or V <= 0):
        raise ValueError("Mass and volume must be positive for volume-corrected Kd.")

    if method_id == "dimensionless":
        # Kd = (C0 - Ce) / Ce  [dimensionless]
        Kd = (C0 - Ce) / Ce

    elif method_id == "mass_based":
        # Kd = qe / Ce  [L/g]
        Kd = qe / Ce

    elif method_id == "volume_corrected":
        # Kd = (qe × m) / (Ce × V)  [dimensionless]
        # This equals (C0 - Ce) × V / Ce × V = (C0 - Ce) / Ce when qe = (C0-Ce)×V/m
        Kd = (qe * m) / (Ce * V)

    else:
        raise ValueError(f"Unknown Kd method: {method_id}")

    if not np.all(np.isfinite(Kd)) or np.any(Kd <= 0):
        raise ValueError("Kd must be finite and greater than zero for Van't Hoff analysis.")

    return Kd


def render():
    """Render thermodynamic analysis with professional statistics."""
    st.subheader("🌡️ Thermodynamic Analysis")
    st.markdown("*Van't Hoff analysis with confidence intervals and explicit Kd limitations*")

    current_study_state = get_current_study_state()
    if not current_study_state:
        st.info("Please add or select a study from the sidebar to begin analysis.")
        return

    # Get confidence level from study state
    confidence_level = current_study_state.get("confidence_level", 0.95)

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

        # Calculate temperature effect results based on input mode
        if input_mode == "direct":
            temp_results_obj = calculate_temperature_results_direct(
                temp_input, include_uncertainty=True
            )
            st.caption("📈 *Using direct concentration input (Ce values)*")
        else:
            temp_results_obj = calculate_temperature_results(
                temp_input, calib_params, include_uncertainty=True
            )

        if temp_results_obj.success:
            temp_results = temp_results_obj.data
            if len(temp_results) < 3:
                st.warning("⚠️ Need at least 3 temperature points for thermodynamic analysis.")
                return

            min_temp_c = temp_results["Temperature_C"].min()
            if min_temp_c > 100:  # Threshold to detect possible Kelvin input
                st.warning(
                    f"**Temperature Unit Check:** The lowest temperature entered is {min_temp_c:.1f}°C. "
                    "This application expects temperature in **Celsius (°C)**. If you entered values in Kelvin, "
                    "please correct your input. However, if you are conducting **gas-phase adsorption** or "
                    "**high-temperature studies**, temperatures above 100°C are valid and you may proceed.",
                    icon="🌡️",
                )
            st.markdown("---")
            st.markdown("### 1. 📊 Temperature-Dependent Data")

            display_cols = ["Temperature_C", "Temperature_K", "Ce_mgL", "qe_mg_g", "removal_%"]
            display_results_table(temp_results[display_cols].round(4), hide_index=False)

            params = temp_input["params"]
            C0 = params["C0"]
            m = params["m"]
            V = params["V"]
            st.caption(f"**Conditions:** C₀ = {C0} mg/L | m = {m} g | V = {V} L")

            # =================================================================
            # Kd Method Selection
            # =================================================================
            st.markdown("---")
            st.markdown("### 2. 📐 Distribution Coefficient Calculation")

            # Method selection with explanation
            with st.expander("ℹ️ **Important: Choose Kd Calculation Method**", expanded=True):
                st.markdown("""
                The distribution coefficient (Kd) can be calculated using different methods.
                **Your choice affects the calculated ΔS° value** (ΔH° is unaffected as it depends only on the slope).

                | Method | Units | Best For |
                |--------|-------|----------|
                | **Dimensionless ratio** | - | Operational comparison when the definition is reported |
                | **Mass-based (qₑ/Cₑ)** | L/g | Comparing with most published papers |
                | **Volume-corrected ratio** | - | Mass-balance-equivalent operational ratio |

                None of these concentration-ratio definitions is automatically a standard-state
                thermodynamic equilibrium constant. Report the exact definition and treat derived
                values as **apparent thermodynamic parameters** unless a justified activity/standard-state
                conversion is provided.
                """)

            # User selects method
            selected_method = st.radio(
                "Select Kd calculation method:",
                list(KD_METHODS.keys()),
                index=0,  # Default to dimensionless
                horizontal=True,
                key="kd_method_selector",
            )

            method_info = KD_METHODS[selected_method]
            method_id = method_info["id"]

            # Display selected formula
            col1, col2 = st.columns([2, 1])
            with col1:
                st.latex(method_info["formula"])
            with col2:
                st.info(f"**Units:** {method_info['units']}")

            st.caption(f"*{method_info['description']}*")
            st.caption(f"📚 Reference: {method_info['reference']}")

            # Calculate Kd using selected method
            Ce = temp_results["Ce_mgL"].values
            qe = temp_results["qe_mg_g"].values

            try:
                Kd = _calculate_kd(method_id, C0, Ce, qe, m, V)
            except ValueError as exc:
                st.error(f"❌ Cannot calculate Kd: {exc}")
                st.info("Check that every observation satisfies 0 < Ce < C₀ and qe > 0.")
                return

            # ----- Propagate uncertainty to Kd (Phase 1.3) -----
            Kd_se_arr = None
            has_uncertainty = (
                "Ce_error" in temp_results.columns and temp_results["Ce_error"].sum() > 0
            )
            if has_uncertainty:
                Ce_se_vals = temp_results["Ce_error"].values
                qe_se_vals = (
                    temp_results["qe_error"].values if "qe_error" in temp_results.columns else None
                )
                Kd_se_arr = propagate_kd_uncertainty(
                    method_id,
                    C0,
                    Ce,
                    qe,
                    m,
                    V,
                    Ce_se=Ce_se_vals,
                    qe_se=qe_se_vals,
                )

            temp_results["Kd"] = Kd
            temp_results["ln_Kd"] = np.log(Kd)
            temp_results["1/T"] = 1 / temp_results["Temperature_K"]

            # Display Kd table
            kd_cols = ["Temperature_K", "qe_mg_g", "Ce_mgL", "Kd", "ln_Kd", "1/T"]
            display_results_table(temp_results[kd_cols].round(6), hide_index=False)

            # Store method used for reporting
            st.session_state["kd_method_used"] = selected_method

            # Warning for mass-based method
            if method_id == "mass_based":
                st.warning("""
                ⚠️ **Note:** You selected the mass-based method (qₑ/Cₑ) which has units of L/g.

                This is common in the adsorption literature, but Kd has units of L/g. The resulting
                intercept-dependent ΔS° and ΔG° values are therefore apparent and depend on the
                unit convention. The fitted slope may also be affected if the conversion factor is
                temperature-dependent. Report the definition and units explicitly.
                """)

            # Van't Hoff Analysis
            st.markdown("---")
            st.markdown("### 3. 📈 Van't Hoff Analysis")

            st.latex(r"\ln(K_d) = \frac{\Delta S°}{R} - \frac{\Delta H°}{RT}")

            T_K = temp_results["Temperature_K"].values

            # NEW: Validate thermodynamic data before Van't Hoff analysis
            validation_report = validate_thermodynamic_data(temperatures=T_K, Kd=Kd)

            if not validation_report.is_valid:
                st.error("❌ **Thermodynamic Data Validation Failed**")
                st.markdown(format_validation_errors(validation_report))
                st.info(
                    "Please correct the data issues above before calculating thermodynamic parameters."
                )
                return

            if validation_report.has_warnings:
                with st.expander("⚠️ Thermodynamic Data Quality Notes", expanded=False):
                    for w in validation_report.warnings:
                        st.warning(w.message)
                        if w.suggestion:
                            st.caption(f"💡 {w.suggestion}")

            # Calculate button
            calculate_btn = st.button(
                "🧮 Calculate Thermodynamic Parameters",
                type="primary",
                help="Click to calculate ΔH°, ΔS°, ΔG° and perform van't Hoff analysis",
                key="thermo_calculate_btn",
            )

            # Check for cached results
            has_cached_results = current_study_state.get("thermo_params") is not None

            if has_cached_results and not calculate_btn:
                st.success("✅ Using cached thermodynamic results (click button to recalculate)")
                thermo_params = current_study_state["thermo_params"]
                show_results = True
            elif calculate_btn:
                # Pass confidence_level to thermodynamic calculation
                thermo_result = calculate_thermodynamic_parameters(
                    T_K,
                    Kd,
                    confidence_level,
                    Kd_se=Kd_se_arr,
                )

                if thermo_result.get("success"):
                    thermo_params = thermo_result
                    # Store method info in thermo_params for reporting
                    thermo_params["kd_method"] = selected_method
                    thermo_params["kd_method_id"] = method_id
                    thermo_params["kd_units"] = method_info["units"]

                    current_study_state["thermo_params"] = thermo_params
                    show_results = True
                else:
                    st.error("Failed to calculate thermodynamic parameters")
                    show_results = False
                    thermo_params = None
            else:
                st.info(
                    "👆 Click **'Calculate Thermodynamic Parameters'** to perform van't Hoff analysis"
                )
                show_results = False
                thermo_params = None

            if show_results and thermo_params:
                # Van't Hoff Plot
                x = 1 / T_K
                y = np.log(Kd)

                slope = thermo_params["slope"]
                intercept = thermo_params["intercept"]

                fig = create_vant_hoff_plot(
                    invT=x,
                    lnKd=y,
                    slope=slope,
                    intercept=intercept,
                    r_squared=float(thermo_params["r_squared"]),
                    title=f"Van't Hoff Plot (Kd method: {method_info['units']})",
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Thermodynamic Parameters
                st.markdown("---")
                st.markdown("### 4. 🔬 Thermodynamic Parameters")

                # Show confidence level indicator
                ci_pct = int(confidence_level * 100)
                st.caption(f"📊 Confidence Intervals calculated at **{ci_pct}%** level")

                delta_H = thermo_params["delta_H"]
                delta_S = thermo_params["delta_S"]
                delta_G = thermo_params["delta_G"]

                # Parameters with CI
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Enthalpy Change (ΔH°)**")
                    st.latex(
                        rf"\Delta H° = {delta_H:.2f} \pm {thermo_params.get('delta_H_se', 0):.2f} \text{{ kJ/mol}}"
                    )

                    if "delta_H_ci" in thermo_params:
                        ci_width = thermo_params["delta_H_ci"]
                        st.caption(
                            f"{ci_pct}% CI: ({delta_H - ci_width:.2f}, {delta_H + ci_width:.2f}) kJ/mol"
                        )

                with col2:
                    st.markdown("**Entropy Change (ΔS°)**")
                    st.latex(
                        rf"\Delta S^\circ = {delta_S:.2f} \pm {thermo_params.get('delta_S_se', 0):.2f} \; \text{{J/(mol$\cdot$K)}}"
                    )

                    if "delta_S_ci" in thermo_params:
                        ci_width = thermo_params["delta_S_ci"]
                        st.caption(
                            f"{ci_pct}% CI: ({delta_S - ci_width:.2f}, {delta_S + ci_width:.2f}) J/(mol·K)"
                        )

                    # Note about ΔS° dependency on Kd method
                    if method_id == "mass_based":
                        st.caption("⚠️ *Apparent value (Kd has units)*")

                # ΔG° at each temperature
                st.markdown("**Gibbs Free Energy (ΔG°)**")
                st.latex(r"\Delta G° = \Delta H° - T \Delta S°")

                dG_df = pd.DataFrame(
                    {
                        "Temperature (K)": T_K,
                        "Temperature (°C)": T_K - 273.15,
                        "ΔG° (kJ/mol)": delta_G,
                    }
                )
                display_results_table(dG_df.round(2))

                # =============================================================
                # WLS vs OLS comparison (Phase 1.3 — Error Propagation)
                # =============================================================
                if "wls_delta_H" in thermo_params:
                    st.markdown("---")
                    st.markdown("### 4.1 🔗 Propagated Uncertainty (WLS vs OLS)")
                    st.markdown(
                        "*When calibration uncertainty is provided, a weighted "
                        "least-squares (WLS) Van't Hoff regression is performed "
                        "using propagated σ(ln Kd) as weights, completing the "
                        "uncertainty chain from calibration to thermodynamic "
                        "parameters.*"
                    )

                    wls_H = thermo_params["wls_delta_H"]
                    wls_S = thermo_params["wls_delta_S"]
                    wls_G = thermo_params["wls_delta_G"]

                    comp_df = pd.DataFrame(
                        {
                            "Parameter": [
                                "ΔH° (kJ/mol)",
                                "SE(ΔH°)",
                                f"{ci_pct}% CI half-width",
                                "ΔS° (J/(mol·K))",
                                "SE(ΔS°)",
                                f"{ci_pct}% CI half-width",
                                "R²",
                            ],
                            "OLS (regression-only)": [
                                f"{delta_H:.4f}",
                                f"{thermo_params.get('delta_H_se', 0):.4f}",
                                f"{thermo_params.get('delta_H_ci', 0):.4f}",
                                f"{delta_S:.4f}",
                                f"{thermo_params.get('delta_S_se', 0):.4f}",
                                f"{thermo_params.get('delta_S_ci', 0):.4f}",
                                f"{thermo_params.get('r_squared', 0):.6f}",
                            ],
                            "WLS (propagated)": [
                                f"{wls_H:.4f}",
                                f"{thermo_params.get('wls_delta_H_se', 0):.4f}",
                                f"{thermo_params.get('wls_delta_H_ci', 0):.4f}",
                                f"{wls_S:.4f}",
                                f"{thermo_params.get('wls_delta_S_se', 0):.4f}",
                                f"{thermo_params.get('wls_delta_S_ci', 0):.4f}",
                                f"{thermo_params.get('wls_r_squared', 0):.6f}",
                            ],
                        }
                    )
                    display_results_table(comp_df)

                    # ΔG° comparison table
                    dG_cmp = pd.DataFrame(
                        {
                            "Temperature (K)": T_K,
                            "ΔG° OLS (kJ/mol)": delta_G,
                            "ΔG° WLS (kJ/mol)": wls_G,
                        }
                    )
                    display_results_table(dG_cmp.round(4))

                    st.caption(
                        "**OLS** uses only the residual scatter of the Van't Hoff "
                        "fit.  **WLS** additionally accounts for the propagated "
                        "measurement uncertainty in each ln(Kd) point, giving "
                        "more weight to data points with smaller experimental "
                        "error."
                    )

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("ΔH° (kJ/mol)", f"{delta_H:.2f}")
                with col2:
                    st.metric("ΔS° (J/mol·K)", f"{delta_S:.2f}")
                with col3:
                    st.metric("ΔG° range", f"{delta_G.min():.2f} to {delta_G.max():.2f}")
                with col4:
                    st.metric("R²", f"{thermo_params['r_squared']:.4f}")

                # Thermodynamic trend summary (not a mechanism classifier)
                st.markdown("---")
                st.markdown("### 5. 📋 Thermodynamic Trend Summary")

                interpretation = interpret_thermodynamics(delta_H, delta_S, delta_G)

                # Map keys to labels and display appropriately
                key_labels = {
                    "enthalpy": "Enthalpy",
                    "entropy": "Entropy",
                    "delta_g_sign": "Apparent ΔG° sign",
                    "caveat": "Interpretation limit",
                }

                for key, description in interpretation.items():
                    label = key_labels.get(key, key.replace("_", " ").title())
                    st.info(f"**{label}:** {description}")

                st.warning(
                    "Model fit, ΔH° magnitude, ΔG°, Freundlich n, and Rₗ do not by themselves "
                    "identify physisorption, chemisorption, or ion exchange. Support mechanism claims "
                    "with independent evidence such as spectroscopy, desorption/regeneration, ionic-strength "
                    "tests, or appropriately designed kinetic experiments."
                )

                # Results summary
                st.markdown("---")
                st.markdown("### 6. 📋 Results Summary")

                # Add Kd method to summary
                summary_df = pd.DataFrame(
                    {
                        "Parameter": ["Kd Method", "ΔH°", "ΔS°", "ΔG° (298 K)", "R²"],
                        "Value": [
                            f"{selected_method}",
                            f"{delta_H:.2f} kJ/mol",
                            f"{delta_S:.2f} J/(mol·K)",
                            f"{delta_H - 298.15 * delta_S / 1000:.2f} kJ/mol",
                            f"{thermo_params['r_squared']:.4f}",
                        ],
                        "Interpretation": [
                            f"Units: {method_info['units']}",
                            "Exothermic" if delta_H < 0 else "Endothermic",
                            "Positive apparent ΔS°" if delta_S > 0 else "Negative apparent ΔS°",
                            "Negative apparent ΔG°"
                            if (delta_H - 298.15 * delta_S / 1000) < 0
                            else "Positive apparent ΔG°",
                            "Excellent fit" if thermo_params["r_squared"] > 0.99 else "Good fit",
                        ],
                    }
                )
                display_results_table(summary_df)

                # Methods section text for reporting
                with st.expander("📝 **Suggested Methods Text**"):
                    if method_id == "dimensionless":
                        methods_text = f"""
**Thermodynamic Analysis**

Thermodynamic parameters were determined using the Van't Hoff equation.
The distribution coefficient (Kd) was calculated as the dimensionless ratio
of adsorbate removed to adsorbate remaining in solution:

Kd = (C₀ - Cₑ) / Cₑ

where C₀ and Cₑ are the initial and equilibrium concentrations (mg/L), respectively.

The apparent enthalpy change (ΔH° = {delta_H:.2f} kJ/mol) and entropy change
(ΔS° = {delta_S:.2f} J/(mol·K)) were obtained from the slope and intercept of
the Van't Hoff plot (ln Kd vs 1/T, R² = {thermo_params["r_squared"]:.4f}).
The Kd definition is an operational concentration ratio and was not treated as a
standard-state thermodynamic equilibrium constant.
"""
                    else:
                        methods_text = f"""
**Thermodynamic Analysis**

Thermodynamic parameters were determined using the Van't Hoff equation.
The distribution coefficient (Kd) was calculated as:

Kd = qₑ / Cₑ

where qₑ is the equilibrium adsorption capacity (mg/g) and Cₑ is the equilibrium
concentration (mg/L).

Note: This approach yields apparent thermodynamic parameters as Kd has units of L/g.

The enthalpy change (ΔH° = {delta_H:.2f} kJ/mol) and entropy change
(ΔS° = {delta_S:.2f} J/(mol·K)) were obtained from the slope and intercept of
the Van't Hoff plot (ln Kd vs 1/T, R² = {thermo_params["r_squared"]:.4f}).
"""
                    st.code(methods_text, language=None)

                # Export info
                st.info(
                    "💡 **To download figures and data:** Go to the **📦 Export All** tab for comprehensive exports with format options."
                )

            else:
                st.error(
                    "❌ Thermodynamic calculation failed. Ensure valid temperature data with positive Kd values."
                )

        else:
            st.warning("⚠️ Need at least 3 temperature points for thermodynamic analysis")

    elif temp_input and input_mode == "absorbance" and not calib_params:
        st.warning(
            "⚠️ Complete calibration first, or switch to **Direct Concentration** input mode in the sidebar"
        )
    elif not temp_input:
        st.info("📥 Enter temperature data in sidebar to enable thermodynamic analysis")
        _display_guidelines()


def _display_guidelines():
    """Display thermodynamics guidelines."""
    with st.expander("📖 Thermodynamic Analysis Guidelines", expanded=True):
        st.markdown("""
        **Best Practices:**

        1. **Temperature Range:** At least 3 temperatures (preferably 4-5)
        2. **Temperature Spacing:** 10-15°C intervals (e.g., 25, 35, 45, 55°C)
        3. **Equilibrium:** Ensure equilibrium is reached at each temperature
        4. **Report:**
           - Kd calculation method used
           - ΔH° with uncertainty (from Van't Hoff slope)
           - ΔS° with uncertainty (from intercept)
           - ΔG° at each temperature
           - R² of Van't Hoff plot

        **Kd Method Selection:**

        | Method | Formula | When to Use |
        |--------|---------|-------------|
        | Dimensionless ratio | (C₀-Cₑ)/Cₑ | Operational comparison; define explicitly |
        | Mass-based | qₑ/Cₑ | Comparing with older literature |
        | Volume-corrected ratio | (qₑ×m)/(Cₑ×V) | Mass-balance-equivalent operational ratio |

        **Interpretation Guidelines:**

        | Parameter | Range | Interpretation |
        |-----------|-------|----------------|
        | ΔH° | < 0 | Exothermic |
        | ΔH° | > 0 | Endothermic |
        | ΔG° | < 0 | Negative apparent free-energy change for the stated Kd convention |
        | ΔS° | > 0 | Increased randomness |

        Do not infer adsorption mechanism from ΔH° magnitude, kinetic-model fit,
        Freundlich n, or Rₗ alone. Mechanism claims require independent experimental evidence.
        """)
