# sidebar_ui.py
import streamlit as st
import pandas as pd
from utils import validate_data_editor

# --- Helper Functions for UI Rendering ---

def _read_uploaded_file(uploaded_file, required_cols):
    """
    Reads a CSV or XLSX file, checks for required columns, and returns a DataFrame.
    Displays errors/warnings in the sidebar.
    """
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep=';')
        else:
            df = pd.read_excel(uploaded_file)
        
        # Check for required columns
        if all(col in df.columns for col in required_cols):
            return df[required_cols]
        else:
            missing = [col for col in required_cols if col not in df.columns]
            st.sidebar.warning(f"Uploaded file is missing columns: {', '.join(missing)}")
            return None
            
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        return None

def _handle_input_change(state_key, new_input_dict, dependent_keys):
    """
    Compares the new input with the one in session_state.
    If changed, updates the state and resets all dependent result keys.
    """
    current_input = st.session_state.get(state_key)
    needs_update = False

    if new_input_dict is None:
        if current_input is not None:
            needs_update = True # Data was cleared
    elif current_input is None:
        needs_update = True # First time data is added
    else:
        # Compare dataframes and parameters for changes
        data_equal = new_input_dict['data'].equals(current_input.get('data'))
        params_equal = new_input_dict['params'] == current_input.get('params', {})
        if not data_equal or not params_equal:
            needs_update = True
    
    if needs_update:
        st.session_state[state_key] = new_input_dict
        # Reset all dependent results
        for key in dependent_keys:
            if key in st.session_state:
                st.session_state[key] = None

# --- UI Rendering Functions ---

def _render_calibration_input():
    """Renders the specific UI for calibration data input."""
    with st.sidebar.expander("1. Calibration", expanded=False):
        st.write("Concentration vs. Absorbance:")
        
        required_cols = ['Concentration', 'Absorbance']
        
        uploaded_file = st.file_uploader(
            "Upload Calibration Data", type=['csv', 'xlsx'], key="calib_uploader",
            help=f"File must contain '{required_cols[0]}' and '{required_cols[1]}' columns."
        )

        initial_data = _read_uploaded_file(uploaded_file, required_cols)
        if initial_data is None:
            initial_data = pd.DataFrame({col: [] for col in required_cols})

        edited_calib = st.data_editor(
            initial_data.astype(float), num_rows="dynamic", key="calib_editor",
            column_config={
                "Concentration": st.column_config.NumberColumn(format="%.4f", required=True, help="Standard concentration (mg/L or equivalent unit)"),
                "Absorbance": st.column_config.NumberColumn(format="%.4f", required=True, help="Corresponding measured absorbance")
            })
        
        # Calibration state is simpler, so it's handled directly
        st.session_state['calib_df_input'] = validate_data_editor(edited_calib, required_cols)

def _render_study_input(config):
    """
    Renders a generic study input section based on a configuration dictionary.
    Handles fixed parameters, file upload, data editor, validation, and state changes.
    """
    with st.sidebar.expander(config['expander_title'], expanded=False):
        st.write(config['intro_text'])

        # 1. Render fixed parameter inputs
        fixed_params = {}
        for param_key, param_config in config['fixed_params'].items():
            fixed_params[param_key] = st.number_input(
                label=param_config['label'],
                min_value=param_config.get('min_value'),
                value=param_config.get('value'),
                format=param_config.get('format', "%.4f"),
                key=config['key_prefix'] + param_key,
                help=param_config['help']
            )

        # 2. File Uploader
        required_cols = config['required_cols']
        uploaded_file = st.file_uploader(
            f"Upload {config['study_name']} Data",
            type=['csv', 'xlsx'],
            key=config['key_prefix'] + "uploader",
            help=f"File must contain '{' and '.join(required_cols)}' columns."
        )
        
        # 3. Read file or create empty DataFrame
        initial_data = _read_uploaded_file(uploaded_file, required_cols)
        if initial_data is None:
            initial_data = pd.DataFrame({col: [] for col in required_cols})

        # 4. Data Editor
        st.write(config['editor_intro'])
        edited_df = st.data_editor(
            initial_data,
            num_rows="dynamic",
            key=config['key_prefix'] + "editor",
            column_config=config['column_config']
        )

        # 5. Validate data and handle state update
        validated_df = validate_data_editor(edited_df, required_cols)
        
        new_input = None
        if validated_df is not None:
            # Handle special case for kinetic data sorting
            if config.get('sort_by'):
                validated_df = validated_df.sort_values(by=config['sort_by']).reset_index(drop=True)
            new_input = {'data': validated_df, 'params': fixed_params}

        _handle_input_change(
            state_key=config['state_key'],
            new_input_dict=new_input,
            dependent_keys=config['dependent_keys']
        )

def render_sidebar_content():
    """
    Defines configurations for each study and renders all input sections
    in the sidebar.
    """
    # --- Render Calibration Input  ---
    _render_calibration_input()

    # --- Configuration Dictionaries for Each Study ---
    isotherm_config = {
        'study_name': "Isotherm",
        'expander_title': "2. Isotherm Study",
        'intro_text': "Fixed conditions for this isotherm:",
        'editor_intro': "Enter C0 (variable) vs. Absorbance Eq.:",
        'key_prefix': "iso_",
        'state_key': 'isotherm_input',
        'required_cols': ['Concentration_Initiale_C0', 'Absorbance_Equilibre'],
        'fixed_params': {
            'm': {'label': 'Ads. Mass (g)', 'value': 0.02, 'help': 'Mass of adsorbent used', 'min_value': 1e-9},
            'V': {'label': 'Sol. Volume (L)', 'value': 0.05, 'help': 'Volume of adsorbate solution', 'min_value': 1e-6}
        },
        'column_config': {
            "Concentration_Initiale_C0": st.column_config.NumberColumn("C0 (mg/L)", format="%.4f", required=True, help="Initial adsorbate concentration"),
            "Absorbance_Equilibre": st.column_config.NumberColumn("Abs Eq.", format="%.4f", required=True, help="Measured absorbance at equilibrium")
        },
        'dependent_keys': ['isotherm_results', 'langmuir_params_lin', 'freundlich_params_lin', 'temkin_params_lin', 'langmuir_params_nl', 'freundlich_params_nl', 'temkin_params_nl']
    }
    
    kinetic_config = {
        'study_name': "Kinetic",
        'expander_title': "5. Kinetic Study",
        'intro_text': "Fixed conditions for ONE kinetic experiment:",
        'editor_intro': "Enter Time (variable) vs. Absorbance(t):",
        'key_prefix': "kin_",
        'state_key': 'kinetic_input',
        'required_cols': ['Temps_min', 'Absorbance_t'],
        'sort_by': 'Temps_min', 
        'fixed_params': {
            'C0': {'label': 'C0 (mg/L)', 'value': 10.0, 'help': 'Constant initial concentration', 'min_value': 0.0},
            'm': {'label': 'Ads. Mass (g)', 'value': 0.02, 'help': 'Constant adsorbent mass', 'min_value': 1e-9},
            'V': {'label': 'Sol. Volume (L)', 'value': 0.05, 'help': 'Constant solution volume', 'min_value': 1e-6}
        },
        'column_config': {
            "Temps_min": st.column_config.NumberColumn("Time (min)", format="%.2f", required=True, min_value=0, help="Time elapsed since start"),
            "Absorbance_t": st.column_config.NumberColumn("Abs(t)", format="%.4f", required=True, help="Measured absorbance at time t")
        },
        'dependent_keys': ['kinetic_results_df', 'pfo_params_nonlinear', 'pso_params_nonlinear', 'ipd_params_list']
    }

    dosage_config = {
        'study_name': "Dosage Effect",
        'expander_title': "6. Dosage Study",
        'intro_text': "Fixed conditions for mass study:",
        'editor_intro': "Enter Ads. Mass vs. Absorbance Eq.:",
        'key_prefix': "dos_",
        'state_key': 'dosage_input',
        'required_cols': ['Masse_Adsorbant_g', 'Absorbance_Equilibre'],
        'fixed_params': {
            'C0': {'label': 'C0 (mg/L)', 'value': 20.0, 'help': 'Constant initial concentration', 'min_value': 0.0},
            'V': {'label': 'Sol. Volume (L)', 'value': 0.05, 'help': 'Constant solution volume', 'min_value': 1e-6}
        },
        'column_config': {
            "Masse_Adsorbant_g": st.column_config.NumberColumn("m (g)", format="%.4f", required=True, min_value=1e-9, help="Variable mass of adsorbent used"),
            "Absorbance_Equilibre": st.column_config.NumberColumn("Abs Eq.", format="%.4f", required=True, help="Measured absorbance at equilibrium")
        },
        'dependent_keys': ['dosage_results']
    }

    ph_config = {
        'study_name': "pH Effect",
        'expander_title': "3. pH Effect Study",
        'intro_text': "Fixed conditions for pH study:",
        'editor_intro': "Enter pH (variable) vs. Absorbance Eq.:",
        'key_prefix': "ph_",
        'state_key': 'ph_effect_input',
        'required_cols': ['pH', 'Absorbance_Equilibre'],
        'fixed_params': {
            'C0': {'label': 'C0 (mg/L)', 'value': 20.0, 'help': 'Constant initial concentration', 'min_value': 0.0},
            'm': {'label': 'Ads. Mass (g)', 'value': 0.02, 'help': 'Constant adsorbent mass', 'min_value': 1e-9},
            'V': {'label': 'Sol. Volume (L)', 'value': 0.05, 'help': 'Constant solution volume', 'min_value': 1e-6}
        },
        'column_config': {
            "pH": st.column_config.NumberColumn("pH", format="%.2f", required=True, help="Variable pH of the experiment"),
            "Absorbance_Equilibre": st.column_config.NumberColumn("Abs Eq.", format="%.4f", required=True, help="Measured absorbance at equilibrium")
        },
        'dependent_keys': ['ph_effect_results']
    }

    temp_config = {
        'study_name': "Temperature Effect",
        'expander_title': "4. Temperature Effect Study",
        'intro_text': "Fixed conditions for T° study:",
        'editor_intro': "Enter T° (variable) vs. Absorbance Eq.:",
        'key_prefix': "temp_",
        'state_key': 'temp_effect_input',
        'required_cols': ['Temperature_C', 'Absorbance_Equilibre'],
        'fixed_params': {
            'C0': {'label': 'C0 (mg/L)', 'value': 50.0, 'help': 'Constant initial concentration', 'min_value': 0.0},
            'm': {'label': 'Ads. Mass (g)', 'value': 0.02, 'help': 'Constant adsorbent mass', 'min_value': 1e-9},
            'V': {'label': 'Sol. Volume (L)', 'value': 0.05, 'help': 'Constant solution volume', 'min_value': 1e-6}
        },
        'column_config': {
            "Temperature_C": st.column_config.NumberColumn("T (°C)", format="%.1f", required=True, help="Variable temperature of the experiment"),
            "Absorbance_Equilibre": st.column_config.NumberColumn("Abs Eq.", format="%.4f", required=True, help="Measured absorbance at equilibrium")
        },
        'dependent_keys': ['temp_effect_results', 'thermo_params']
    }

    # --- Render All Generic Study Inputs ---
    _render_study_input(isotherm_config)
    _render_study_input(kinetic_config)
    _render_study_input(dosage_config)
    _render_study_input(ph_config)
    _render_study_input(temp_config)