# sidebar_ui.py
import streamlit as st
import pandas as pd
from translations import _t
from utils import validate_data_editor

def _render_calibration_input():
    with st.sidebar.expander(_t("sidebar_expander_calib"), expanded=False):
        st.write(_t("sidebar_calib_intro"))
        uploaded_file_calib = st.file_uploader(
            "Upload Calibration Data",
            type=['csv', 'xlsx'], 
            key="calib_uploader",
            help="File must contain 'Concentration' and 'Absorbance' columns."
        )

        initial_calib_data = pd.DataFrame({'Concentration': [], 'Absorbance': []})

        if uploaded_file_calib is not None:
            try:
                if uploaded_file_calib.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file_calib, sep=';')
                else:
                    df = pd.read_excel(uploaded_file_calib)
                
                if 'Concentration' in df.columns and 'Absorbance' in df.columns:
                    initial_calib_data = df[['Concentration', 'Absorbance']]
                else:
                    st.sidebar.warning("Uploaded file must contain 'Concentration' and 'Absorbance' columns.")
            except Exception as e:
                st.sidebar.error(f"Error reading calibration file: {e}")
        
        edited_calib = st.data_editor(
            initial_calib_data.astype(float), num_rows="dynamic", key="calib_editor",
            column_config={
                "Concentration": st.column_config.NumberColumn(format="%.4f", required=True, help=_t("col_concentration_help")),
                "Absorbance": st.column_config.NumberColumn(format="%.4f", required=True, help=_t("col_absorbance_help"))
            })
        st.session_state['calib_df_input'] = validate_data_editor(edited_calib, ['Concentration', 'Absorbance'])

def _render_isotherm_input():
    with st.sidebar.expander(_t("sidebar_expander_isotherm"), expanded=False):
        st.write(_t("sidebar_isotherm_fixed_conditions"))
        m_iso = st.number_input(_t("sidebar_isotherm_mass_label"), min_value=1e-9, value=0.02, format="%.4f", key="m_iso_input_sidebar", help=_t("sidebar_isotherm_mass_help"))
        V_iso = st.number_input(_t("sidebar_isotherm_volume_label"), min_value=1e-6, value=0.05, format="%.4f", key="V_iso_input_sidebar", help=_t("sidebar_isotherm_volume_help"))
        
        # ---  File Uploader ---
        uploaded_file_iso = st.file_uploader(
            "Upload Isotherm Data",
            type=['csv', 'xlsx'], 
            key="iso_uploader",
            help="File must contain 'Concentration_Initiale_C0' and 'Absorbance_Equilibre' columns."
        )

        # --- Logic to handle uploaded file ---
        initial_iso_data = pd.DataFrame({'Concentration_Initiale_C0': [], 'Absorbance_Equilibre': []})
        
        if uploaded_file_iso is not None:
            try:
                if uploaded_file_iso.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file_iso, sep=';')
                else:
                    df = pd.read_excel(uploaded_file_iso)
                
                if 'Concentration_Initiale_C0' in df.columns and 'Absorbance_Equilibre' in df.columns:
                    initial_iso_data = df[['Concentration_Initiale_C0', 'Absorbance_Equilibre']]
                else:
                    st.sidebar.warning("Uploaded file must contain 'Concentration_Initiale_C0' and 'Absorbance_Equilibre' columns.")
            except Exception as e:
                st.sidebar.error(f"Error reading isotherm file: {e}")

        # --- The data editor logic ---
        st.write(_t("sidebar_isotherm_c0_abs_intro"))
        edited_iso = st.data_editor(
            initial_iso_data,
            num_rows="dynamic", key="iso_editor",
            column_config={
                "Concentration_Initiale_C0": st.column_config.NumberColumn("C0 (mg/L)", format="%.4f", required=True, help=_t("col_c0_help")),
                "Absorbance_Equilibre": st.column_config.NumberColumn("Abs Eq.", format="%.4f", required=True, help=_t("col_abs_eq_help"))
            })

        iso_df_validated = validate_data_editor(edited_iso, ['Concentration_Initiale_C0', 'Absorbance_Equilibre'])
        current_iso_input_in_state = st.session_state.get('isotherm_input')

        if iso_df_validated is not None:
            new_iso_input = {'data': iso_df_validated, 'params': {'m': m_iso, 'V': V_iso}}
            needs_update = False
            if current_iso_input_in_state is None or not isinstance(current_iso_input_in_state, dict):
                 needs_update = True
            else:
                data_equal = iso_df_validated.equals(current_iso_input_in_state.get('data'))
                params_equal = (m_iso == current_iso_input_in_state.get('params', {}).get('m') and
                                V_iso == current_iso_input_in_state.get('params', {}).get('V'))
                if not data_equal or not params_equal:
                    needs_update = True
            
            if needs_update:
                st.session_state['isotherm_input'] = new_iso_input
                st.session_state['isotherm_results'] = None
                st.session_state['langmuir_params_nl'] = None 
                st.session_state['freundlich_params_nl'] = None
                st.session_state['temkin_params_nl'] = None
                st.session_state['langmuir_params_lin'] = None 
                st.session_state['freundlich_params_lin'] = None 
                st.session_state['temkin_params_lin'] = None
        elif current_iso_input_in_state is not None:
            st.session_state['isotherm_input'] = None
            st.session_state['isotherm_results'] = None
            st.session_state['langmuir_params_nl'] = None
            st.session_state['freundlich_params_nl'] = None
            st.session_state['temkin_params_nl'] = None
            st.session_state['langmuir_params_lin'] = None
            st.session_state['freundlich_params_lin'] = None
            st.session_state['temkin_params_lin'] = None

def _render_kinetic_input():
    with st.sidebar.expander(_t("sidebar_expander_kinetic"), expanded=False):
        st.write(_t("sidebar_kinetic_fixed_conditions"))
        C0_k = st.number_input("C0 (mg/L)", min_value=0.0, value=10.0, format="%.4f", key="C0_k_cin_sidebar", help=_t("sidebar_kinetic_c0_help_const"))
        m_k = st.number_input(_t("sidebar_isotherm_mass_label"), min_value=1e-9, value=0.02, format="%.4f", key="m_k_cin_sidebar", help=_t("sidebar_kinetic_mass_help_const"))
        V_k = st.number_input(_t("sidebar_isotherm_volume_label"), min_value=1e-6, value=0.05, format="%.4f", key="V_k_cin_sidebar", help=_t("sidebar_kinetic_volume_help_const"))
        
        uploaded_file_kin = st.file_uploader(
            "Upload Kinetic Data", type=['csv', 'xlsx'], key="kin_uploader",
            help="File must contain 'Temps_min' and 'Absorbance_t' columns."
        )
        initial_kinetic_data = pd.DataFrame({'Temps_min': [], 'Absorbance_t': []})
        if uploaded_file_kin is not None:
            try:
                if uploaded_file_kin.name.endswith('.csv'): df = pd.read_csv(uploaded_file_kin, sep=';')
                else: df = pd.read_excel(uploaded_file_kin)
                if 'Temps_min' in df.columns and 'Absorbance_t' in df.columns:
                    initial_kinetic_data = df[['Temps_min', 'Absorbance_t']]
                else:
                    st.sidebar.warning("Uploaded file must contain 'Temps_min' and 'Absorbance_t' columns.")
            except Exception as e:
                st.sidebar.error(f"Error reading kinetic file: {e}")

        st.write(_t("sidebar_kinetic_time_abs_intro"))
        edited_kin = st.data_editor(
            initial_kinetic_data, num_rows="dynamic", key="kin_editor",
            column_config={
                "Temps_min": st.column_config.NumberColumn(_t("col_time_min"), format="%.2f", required=True, min_value=0, help=_t("col_time_min_help")),
                "Absorbance_t": st.column_config.NumberColumn("Abs(t)", format="%.4f", required=True, help=_t("col_abs_t_help"))
            })

        kin_df_validated = validate_data_editor(edited_kin, ['Temps_min', 'Absorbance_t'])
        current_kin_input_in_state = st.session_state.get('kinetic_input')

        if kin_df_validated is not None:
            kin_df_validated.sort_values(by='Temps_min', inplace=True)
            new_kin_input = {'data': kin_df_validated, 'params': {'C0': C0_k, 'm': m_k, 'V': V_k}}
            needs_update = False
            if current_kin_input_in_state is None or not isinstance(current_kin_input_in_state, dict):
                needs_update = True
            else:
                stored_data = current_kin_input_in_state.get('data')
                data_equal = False
                if stored_data is not None and isinstance(stored_data, pd.DataFrame):
                    try:
                       stored_data_sorted = stored_data.sort_values(by='Temps_min').reset_index(drop=True)
                       kin_df_validated_sorted = kin_df_validated.reset_index(drop=True)
                       data_equal = kin_df_validated_sorted.equals(stored_data_sorted)
                    except AttributeError: pass 
                params_equal = (C0_k == current_kin_input_in_state.get('params', {}).get('C0') and
                                m_k == current_kin_input_in_state.get('params', {}).get('m') and
                                V_k == current_kin_input_in_state.get('params', {}).get('V'))
                if not data_equal or not params_equal:
                    needs_update = True
            
            if needs_update:
                st.session_state['kinetic_input'] = new_kin_input
                st.session_state['kinetic_results_df'] = None
                st.session_state['pfo_params_nonlinear'] = None
                st.session_state['pso_params_nonlinear'] = None
                st.session_state['ipd_params_list'] = None
        elif current_kin_input_in_state is not None:
            st.session_state['kinetic_input'] = None
            st.session_state['kinetic_results_df'] = None
            st.session_state['pfo_params_nonlinear'] = None
            st.session_state['pso_params_nonlinear'] = None
            st.session_state['ipd_params_list'] = None

def _render_ph_input():
    with st.sidebar.expander(_t("sidebar_expander_ph"), expanded=False):
        st.write(_t("sidebar_ph_fixed_conditions"))
        C0_ph = st.number_input("C0 (mg/L)", min_value=0.0, value=20.0, format="%.4f", key="C0_ph_input_sidebar", help=_t("sidebar_kinetic_c0_help_const"))
        m_ph = st.number_input(_t("sidebar_isotherm_mass_label"), min_value=1e-9, value=0.02, format="%.4f", key="m_ph_input_sidebar", help=_t("sidebar_kinetic_mass_help_const"))
        V_ph = st.number_input(_t("sidebar_isotherm_volume_label"), min_value=1e-6, value=0.05, format="%.4f", key="V_ph_input_sidebar", help=_t("sidebar_kinetic_volume_help_const"))

        uploaded_file_ph = st.file_uploader(
            "Upload pH Effect Data", type=['csv', 'xlsx'], key="ph_uploader",
            help="File must contain 'pH' and 'Absorbance_Equilibre' columns."
        )
        initial_ph_data = pd.DataFrame({'pH': [], 'Absorbance_Equilibre': []})
        if uploaded_file_ph is not None:
            try:
                if uploaded_file_ph.name.endswith('.csv'): df = pd.read_csv(uploaded_file_ph, sep=';')
                else: df = pd.read_excel(uploaded_file_ph)
                if 'pH' in df.columns and 'Absorbance_Equilibre' in df.columns:
                    initial_ph_data = df[['pH', 'Absorbance_Equilibre']]
                else:
                    st.sidebar.warning("Uploaded file must contain 'pH' and 'Absorbance_Equilibre' columns.")
            except Exception as e:
                st.sidebar.error(f"Error reading pH file: {e}")

        st.write(_t("sidebar_ph_ph_abs_intro"))
        edited_ph = st.data_editor(
            initial_ph_data, num_rows="dynamic", key="ph_editor",
            column_config={
                "pH": st.column_config.NumberColumn("pH", format="%.2f", required=True, help=_t("col_ph_help")),
                "Absorbance_Equilibre": st.column_config.NumberColumn("Abs Eq.", format="%.4f", required=True, help=_t("col_abs_eq_help"))
            })

        ph_df_validated = validate_data_editor(edited_ph, ['pH', 'Absorbance_Equilibre'])
        current_ph_input_in_state = st.session_state.get('ph_effect_input')

        if ph_df_validated is not None:
            new_ph_input = {'data': ph_df_validated, 'params': {'C0': C0_ph, 'm': m_ph, 'V': V_ph}}
            needs_update = False
            if current_ph_input_in_state is None or not isinstance(current_ph_input_in_state, dict):
                 needs_update = True
            else:
                data_equal = ph_df_validated.equals(current_ph_input_in_state.get('data'))
                params_equal = (C0_ph == current_ph_input_in_state.get('params', {}).get('C0') and
                                m_ph == current_ph_input_in_state.get('params', {}).get('m') and
                                V_ph == current_ph_input_in_state.get('params', {}).get('V'))
                if not data_equal or not params_equal:
                    needs_update = True
            
            if needs_update:
                st.session_state['ph_effect_input'] = new_ph_input
                st.session_state['ph_effect_results'] = None
        elif current_ph_input_in_state is not None:
            st.session_state['ph_effect_input'] = None
            st.session_state['ph_effect_results'] = None

def _render_temp_input():
    with st.sidebar.expander(_t("sidebar_expander_temp"), expanded=False):
        st.write(_t("sidebar_temp_fixed_conditions"))
        C0_t = st.number_input("C0 (mg/L)", min_value=0.0, value=50.0, format="%.4f", key="C0_t_input_sidebar", help=_t("sidebar_kinetic_c0_help_const"))
        m_t = st.number_input(_t("sidebar_isotherm_mass_label"), min_value=1e-9, value=0.02, format="%.4f", key="m_t_input_sidebar", help=_t("sidebar_kinetic_mass_help_const"))
        V_t = st.number_input(_t("sidebar_isotherm_volume_label"), min_value=1e-6, value=0.05, format="%.4f", key="V_t_input_sidebar", help=_t("sidebar_kinetic_volume_help_const"))

        uploaded_file_temp = st.file_uploader(
            "Upload Temperature Effect Data", type=['csv', 'xlsx'], key="temp_uploader",
            help="File must contain 'Temperature_C' and 'Absorbance_Equilibre' columns."
        )
        initial_t_data = pd.DataFrame({'Temperature_C': [], 'Absorbance_Equilibre': []})
        if uploaded_file_temp is not None:
            try:
                if uploaded_file_temp.name.endswith('.csv'): df = pd.read_csv(uploaded_file_temp, sep=';')
                else: df = pd.read_excel(uploaded_file_temp)
                if 'Temperature_C' in df.columns and 'Absorbance_Equilibre' in df.columns:
                    initial_t_data = df[['Temperature_C', 'Absorbance_Equilibre']]
                else:
                    st.sidebar.warning("Uploaded file must contain 'Temperature_C' and 'Absorbance_Equilibre' columns.")
            except Exception as e:
                st.sidebar.error(f"Error reading temperature file: {e}")

        st.write(_t("sidebar_temp_temp_abs_intro"))
        edited_t = st.data_editor(
            initial_t_data, num_rows="dynamic", key="temp_editor",
            column_config={
                "Temperature_C": st.column_config.NumberColumn("T (°C)", format="%.1f", required=True, help=_t("col_temp_c_help")),
                "Absorbance_Equilibre": st.column_config.NumberColumn("Abs Eq.", format="%.4f", required=True, help=_t("col_abs_eq_help"))
            })

        temp_df_validated = validate_data_editor(edited_t, ['Temperature_C', 'Absorbance_Equilibre'])
        current_temp_input_in_state = st.session_state.get('temp_effect_input')

        if temp_df_validated is not None:
            new_temp_input = {'data': temp_df_validated, 'params': {'C0': C0_t, 'm': m_t, 'V': V_t}}
            needs_update = False
            if current_temp_input_in_state is None or not isinstance(current_temp_input_in_state, dict):
                needs_update = True
            else:
                data_equal = temp_df_validated.equals(current_temp_input_in_state.get('data'))
                params_equal = (C0_t == current_temp_input_in_state.get('params', {}).get('C0') and
                                m_t == current_temp_input_in_state.get('params', {}).get('m') and
                                V_t == current_temp_input_in_state.get('params', {}).get('V'))
                if not data_equal or not params_equal:
                    needs_update = True
            
            if needs_update:
                st.session_state['temp_effect_input'] = new_temp_input
                st.session_state['temp_effect_results'] = None
                st.session_state['thermo_params'] = None
        elif current_temp_input_in_state is not None:
            st.session_state['temp_effect_input'] = None
            st.session_state['temp_effect_results'] = None
            st.session_state['thermo_params'] = None

def _render_dosage_input():
    with st.sidebar.expander(_t("sidebar_expander_dosage"), expanded=False):
        st.write(_t("sidebar_dosage_fixed_conditions"))
        C0_dos = st.number_input("C0 (mg/L)", min_value=0.0, value=20.0, format="%.4f", key="C0_dos_input_sidebar", help=_t("sidebar_kinetic_c0_help_const"))
        V_dos = st.number_input(_t("sidebar_isotherm_volume_label"), min_value=1e-6, value=0.05, format="%.4f", key="V_dos_input_sidebar", help=_t("sidebar_kinetic_volume_help_const"))
        
        uploaded_file_dos = st.file_uploader(
            "Upload Dosage Effect Data", type=['csv', 'xlsx'], key="dos_uploader",
            help="File must contain 'Masse_Adsorbant_g' and 'Absorbance_Equilibre' columns."
        )
        initial_dos_data = pd.DataFrame({'Masse_Adsorbant_g': [], 'Absorbance_Equilibre': []})
        if uploaded_file_dos is not None:
            try:
                if uploaded_file_dos.name.endswith('.csv'): df = pd.read_csv(uploaded_file_dos, sep=';')
                else: df = pd.read_excel(uploaded_file_dos)
                if 'Masse_Adsorbant_g' in df.columns and 'Absorbance_Equilibre' in df.columns:
                    initial_dos_data = df[['Masse_Adsorbant_g', 'Absorbance_Equilibre']]
                else:
                    st.sidebar.warning("Uploaded file must contain 'Masse_Adsorbant_g' and 'Absorbance_Equilibre' columns.")
            except Exception as e:
                st.sidebar.error(f"Error reading dosage file: {e}")

        st.write(_t("sidebar_dosage_mass_abs_intro"))
        edited_dos = st.data_editor(
            initial_dos_data, num_rows="dynamic", key="dos_editor",
            column_config={
                "Masse_Adsorbant_g": st.column_config.NumberColumn("m (g)", format="%.4f", required=True, min_value=1e-9, help=_t("col_mass_ads_g_help_variable")),
                "Absorbance_Equilibre": st.column_config.NumberColumn("Abs Eq.", format="%.4f", required=True, help=_t("col_abs_eq_help"))
            })

        dos_df_validated = validate_data_editor(edited_dos, ['Masse_Adsorbant_g', 'Absorbance_Equilibre'])
        current_dos_input_in_state = st.session_state.get('dosage_input')

        if dos_df_validated is not None:
            new_dos_input = {'data': dos_df_validated, 'params': {'C0': C0_dos, 'V': V_dos}}
            needs_update = False
            if current_dos_input_in_state is None or not isinstance(current_dos_input_in_state, dict):
                needs_update = True
            else:
                data_equal = dos_df_validated.equals(current_dos_input_in_state.get('data'))
                current_params = current_dos_input_in_state.get('params', {})
                params_equal = (C0_dos == current_params.get('C0') and
                                V_dos == current_params.get('V'))
                if not data_equal or not params_equal:
                    needs_update = True
            
            if needs_update:
                st.session_state['dosage_input'] = new_dos_input
                st.session_state['dosage_results'] = None
        elif current_dos_input_in_state is not None:
            st.session_state['dosage_input'] = None
            st.session_state['dosage_results'] = None

def render_sidebar_content():
    """Renders all input sections in the sidebar."""
    
    _render_calibration_input()
    _render_isotherm_input()
    _render_kinetic_input()
    _render_dosage_input()
    _render_ph_input() 
    _render_temp_input()