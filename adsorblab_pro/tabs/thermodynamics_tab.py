# tabs/thermodynamics_tab.py
"""
Thermodynamics Tab - AdsorbLab Pro
==================================

Van't Hoff analysis for thermodynamic parameter determination.

Features:
- Van't Hoff equation analysis
- ΔH°, ΔS°, ΔG° calculation with 95% CI
- Multiple Kd calculation methods
- Mechanistic interpretation
- Publication-ready outputs
"""

import numpy as np
import pandas as pd

from adsorblab_pro.streamlit_compat import st

from ..plot_style import create_vant_hoff_plot
from ..utils import (
    EPSILON_DIV,
    calculate_temperature_results,
    calculate_temperature_results_direct,
    calculate_thermodynamic_parameters,
    determine_adsorption_mechanism,
    display_results_table,
    get_current_study_state,
    interpret_thermodynamics,
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
        "description": "Thermodynamically rigorous. Recommended approach.",
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
        "description": "Dimensionless form using experimental parameters.",
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
    # Prevent division by zero
    Ce_safe = np.maximum(Ce, EPSILON_DIV)

    if method_id == "dimensionless":
        # Kd = (C0 - Ce) / Ce  [dimensionless]
        Kd = (C0 - Ce_safe) / Ce_safe

    elif method_id == "mass_based":
        # Kd = qe / Ce  [L/g]
        Kd = qe / Ce_safe

    elif method_id == "volume_corrected":
        # Kd = (qe × m) / (Ce × V)  [dimensionless]
        # This equals (C0 - Ce) × V / Ce × V = (C0 - Ce) / Ce when qe = (C0-Ce)×V/m
        Kd = (qe * m) / (Ce_safe * V)

    else:
        raise ValueError(f"Unknown Kd method: {method_id}")

    # Ensure positive values for ln(Kd)
    Kd = np.maximum(Kd, EPSILON_DIV)

    return Kd


def render():
    """Render thermodynamic analysis with publication-ready statistics."""
    st.subheader("🌡️ Thermodynamic Analysis")
    st.markdown("*Van't Hoff equation with confidence intervals and mechanistic interpretation*")

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
                | **Dimensionless** | - | Thermodynamic rigor, theoretical work |
                | **Mass-based (qₑ/Cₑ)** | L/g | Comparing with most published papers |
                | **Volume-corrected** | - | Alternative dimensionless form |

                **Recommendation:** Use **Dimensionless** for new publications, or **Mass-based**
                if comparing with older literature that used qₑ/Cₑ.
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

            Kd = _calculate_kd(method_id, C0, Ce, qe, m, V)

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

                This is common in literature but technically incorrect for thermodynamics.
                Your ΔH° will be correct, but ΔS° will be an "apparent" value that includes
                a contribution from the unit conversion factor.

                **For rigorous thermodynamics, consider using the Dimensionless method.**
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
                thermo_result = calculate_thermodynamic_parameters(T_K, Kd, confidence_level)

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

                # Interpretation
                st.markdown("---")
                st.markdown("### 5. 🎯 Mechanistic Interpretation")

                interpretation = interpret_thermodynamics(delta_H, delta_S, delta_G)

                # Map keys to labels and display appropriately
                key_labels = {
                    "enthalpy": "Enthalpy",
                    "mechanism_H": "Mechanism (from ΔH°)",
                    "entropy": "Entropy",
                    "spontaneity": "Spontaneity",
                    "feasibility": "Feasibility",
                }

                for key, description in interpretation.items():
                    label = key_labels.get(key, key.replace("_", " ").title())
                    # Determine status based on content
                    if (
                        "spontaneous" in description.lower()
                        and "non-spontaneous" not in description.lower()
                    ):
                        st.success(f"**{label}:** {description}")
                    elif (
                        "favorable" in description.lower()
                        and "unfavorable" not in description.lower()
                    ):
                        st.success(f"**{label}:** {description}")
                    elif (
                        "unfavorable" in description.lower()
                        or "non-spontaneous" in description.lower()
                    ):
                        st.warning(f"**{label}:** {description}")
                    else:
                        st.info(f"**{label}:** {description}")

                # =============================================================
                # NEW: Comprehensive Mechanism Determination Panel
                # =============================================================
                st.markdown("---")
                st.markdown("### 5.1 🔬 Adsorption Mechanism Analysis")

                # Pull fitted isotherm results (if any) for mechanism cross-referencing
                iso_models = current_study_state.get("isotherm_models_fitted") or {}

                # Get Freundlich n if available
                n_freundlich = None
                if iso_models and "Freundlich" in iso_models:
                    freu_result = iso_models["Freundlich"]
                    if freu_result and freu_result.get("converged"):
                        n_freundlich = freu_result.get("params", {}).get("n")

                # Get Langmuir RL if available
                RL = None
                if iso_models and "Langmuir" in iso_models:
                    lang_result = iso_models["Langmuir"]
                    if lang_result and lang_result.get("converged"):
                        RL = lang_result.get("params", {}).get("RL")

                # Call mechanism determination
                mechanism_result = determine_adsorption_mechanism(
                    delta_H=delta_H, delta_G=delta_G, n_freundlich=n_freundlich, RL=RL
                )

                # Display mechanism result
                col1, col2 = st.columns([2, 1])

                with col1:
                    mechanism = mechanism_result["mechanism"]
                    confidence = mechanism_result["confidence"]

                    if "Physical" in mechanism:
                        st.success(f"**🧲 {mechanism}**")
                    elif "Chemical" in mechanism:
                        st.error(f"**⚗️ {mechanism}**")
                    elif "Ion Exchange" in mechanism:
                        st.warning(f"**🔄 {mechanism}**")
                    else:
                        st.info(f"**🔀 {mechanism}**")

                    st.caption(f"Confidence: {confidence:.1f}%")

                with col2:
                    # Display score breakdown as mini chart
                    scores = mechanism_result.get("scores", {})
                    if scores:
                        score_df = pd.DataFrame(
                            {"Mechanism": list(scores.keys()), "Score (%)": list(scores.values())}
                        )
                        st.dataframe(
                            score_df.style.format({"Score (%)": "{:.1f}"}),
                            hide_index=True,
                            use_container_width=True,
                        )

                # Evidence list
                evidence = mechanism_result.get("evidence", [])
                if evidence:
                    with st.expander("📋 Evidence Details", expanded=False):
                        for e in evidence:
                            st.markdown(f"• {e}")

                # Indicator table
                indicators = mechanism_result.get("indicators", {})
                if indicators:
                    with st.expander("📊 Indicator Analysis", expanded=False):
                        ind_data = []
                        for name, info in indicators.items():
                            ind_data.append(
                                {
                                    "Indicator": name,
                                    "Value": f"{info['value']:.2f}"
                                    if isinstance(info["value"], float)
                                    else str(info["value"]),
                                    "Classification": info["classification"],
                                    "Criterion": info["criterion"],
                                    "Confidence": info["confidence"],
                                }
                            )
                        if ind_data:
                            ind_df = pd.DataFrame(ind_data)
                            display_results_table(ind_df)

                        st.caption("""
                        **Reference Criteria:**
                        - Enthalpy (|ΔH°|): < 40 kJ/mol (Physical), 40-80 (Mixed), > 80 (Chemical)
                        - Gibbs (ΔG°): > -20 kJ/mol (Physical), -20 to -40 (Mixed), < -40 (Chemical)
                        """)

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
                            "Increased disorder" if delta_S > 0 else "Decreased disorder",
                            "Spontaneous"
                            if (delta_H - 298.15 * delta_S / 1000) < 0
                            else "Non-spontaneous",
                            "Excellent fit" if thermo_params["r_squared"] > 0.99 else "Good fit",
                        ],
                    }
                )
                display_results_table(summary_df)

                # Methods section text for publication
                with st.expander("📝 **Suggested Methods Text**"):
                    if method_id == "dimensionless":
                        methods_text = f"""
**Thermodynamic Analysis**

Thermodynamic parameters were determined using the Van't Hoff equation.
The distribution coefficient (Kd) was calculated as the dimensionless ratio
of adsorbate removed to adsorbate remaining in solution:

Kd = (C₀ - Cₑ) / Cₑ

where C₀ and Cₑ are the initial and equilibrium concentrations (mg/L), respectively.

The standard enthalpy change (ΔH° = {delta_H:.2f} kJ/mol) and entropy change
(ΔS° = {delta_S:.2f} J/(mol·K)) were obtained from the slope and intercept of
the Van't Hoff plot (ln Kd vs 1/T, R² = {thermo_params["r_squared"]:.4f}).
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
        | Dimensionless | (C₀-Cₑ)/Cₑ | New publications, thermodynamic rigor |
        | Mass-based | qₑ/Cₑ | Comparing with older literature |
        | Volume-corrected | (qₑ×m)/(Cₑ×V) | Alternative dimensionless form |

        **Interpretation Guidelines:**

        | Parameter | Range | Interpretation |
        |-----------|-------|----------------|
        | ΔH° | < 0 | Exothermic |
        | ΔH° | > 0 | Endothermic |
        | |ΔH°| | < 40 kJ/mol | Physical adsorption |
        | |ΔH°| | 40-80 kJ/mol | Mixed mechanism |
        | |ΔH°| | > 80 kJ/mol | Chemical adsorption |
        | ΔG° | < 0 | Spontaneous |
        | ΔS° | > 0 | Increased randomness |
        """)
