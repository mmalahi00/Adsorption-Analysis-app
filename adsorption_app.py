# adsorption_app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress 
from translations import _t, TRANSLATIONS 
import sidebar_ui
from tabs import calibration_tab, isotherm_tab, kinetic_tab, ph_effect_tab, dosage_tab, temperature_tab, thermodynamics_tab

# --- Initialize session state for language FIRST ---
if 'language' not in st.session_state:
    st.session_state.language = 'en' 

# --- Page Configuration ---
st.set_page_config(
    page_title=_t("app_page_title"),
    page_icon="🔬",
    layout="wide"
)

# --- Main Title and Subtitle ---
st.title(_t("app_title"))
st.markdown(_t("app_subtitle"))
st.markdown("---")

# --- Session State Initialization for Data and Parameters ---
default_keys = {
    'calib_df_input': None, 'calibration_params': None, 'previous_calib_df': None,
    'isotherm_input': None, 'isotherm_results': None,
    'langmuir_params_lin': None, 'freundlich_params_lin': None,
    'ph_effect_input': None, 'ph_effect_results': None,
    'temp_effect_input': None, 'temp_effect_results': None, 'thermo_params': None,
    'kinetic_input': None, 'kinetic_results_df': None,
    'pfo_params_nonlinear': None, 
    'pso_params_nonlinear': None, 
    'ipd_params_list': [],
    'dosage_input': None, 'dosage_results': None,
}
for key, default_value in default_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- SIDEBAR ---
st.sidebar.header(_t("sidebar_header"))

# Language Selector - 
current_lang_index = 0 if st.session_state.language == 'fr' else 1
selected_lang = st.sidebar.selectbox(
    _t("language_select_label"),
    options=['fr', 'en'],
    format_func=lambda x: "Français" if x == 'fr' else "English",
    index=current_lang_index,
    key='lang_selector_main_app' 
)
if selected_lang != st.session_state.language:
    st.session_state.language = selected_lang
    st.rerun() 

# Render the rest of the sidebar content using the dedicated module
sidebar_ui.render_sidebar_content()

# --- AUTOMATIC CALIBRATION LOGIC ---
new_calib_df = st.session_state.get('calib_df_input')
old_calib_df = st.session_state.get('previous_calib_df')

if new_calib_df is not None and len(new_calib_df) >= 2:
    if old_calib_df is None or not new_calib_df.equals(old_calib_df):
        try:
            slope, intercept, r_value, _, _ = linregress(new_calib_df['Concentration'], new_calib_df['Absorbance'])
            if abs(slope) > 1e-9: # Avoid near-zero slope issues
                st.session_state['calibration_params'] = {'slope': slope, 'intercept': intercept, 'r_squared': r_value**2}
            else:
                st.session_state['calibration_params'] = None
                st.sidebar.warning(_t("calib_slope_near_zero_warning"), icon="⚠️")
        except Exception as e:
            st.session_state['calibration_params'] = None
            st.sidebar.error(_t("calib_error_calc_warning", error=e), icon="🔥")
        
        st.session_state['previous_calib_df'] = new_calib_df.copy() # Store current data for future comparison

elif new_calib_df is None or len(new_calib_df) < 2:
    # Reset calibration if input becomes invalid or insufficient
    if st.session_state.get('calibration_params') is not None:
        st.session_state['calibration_params'] = None
    if st.session_state.get('previous_calib_df') is not None: 
        st.session_state['previous_calib_df'] = None


# --- MAIN CONTENT AREA WITH TABS ---
st.header(_t("main_results_header"))

tab_names = [
    _t("tab_calib"), _t("tab_isotherm"), _t("tab_kinetic"), _t("tab_dosage"), _t("tab_ph"),   
     _t("tab_temp"), _t("tab_thermo")
]
tab_calib_ui, tab_iso_ui, tab_kin_ui, tab_dosage_ui, tab_ph_ui, tab_temp_ui, tab_thermo_ui = st.tabs(tab_names)

with tab_calib_ui:
    calibration_tab.render()

with tab_iso_ui:
    isotherm_tab.render()

with tab_kin_ui:
    kinetic_tab.render()

with tab_dosage_ui:
    dosage_tab.render()

with tab_ph_ui:
    ph_effect_tab.render()


with tab_temp_ui:
    temperature_tab.render()

with tab_thermo_ui:
    thermodynamics_tab.render()

st.markdown("---")
st.caption("Analyse Adsorption v2.4") 
