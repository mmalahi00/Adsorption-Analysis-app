# tabs/thermodynamics_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress
import io
# from ..translations import _t
# from ..utils import convert_df_to_csv
from translations import _t
from utils import convert_df_to_csv

def render():
    st.header(_t("thermo_tab_header"))
    st.markdown(_t("thermo_tab_intro_markdown"))
    temp_results_for_thermo = st.session_state.get('temp_effect_results')
    thermo_params = st.session_state.get('thermo_params')

    if temp_results_for_thermo is not None and not temp_results_for_thermo.empty and thermo_params is None:
        with st.spinner(_t("thermo_spinner_analysis")):
            R_gas_const = 8.314 # J/mol·K
            df_thermo = temp_results_for_thermo.copy()
            df_thermo = df_thermo[(df_thermo['Ce'] > 1e-9) & (df_thermo['qe'] >= 0)].copy()
            if len(df_thermo['Temperature_C'].unique()) >= 2:
                try:
                    df_thermo['T_K'] = df_thermo['Temperature_C'] + 273.15
                    df_thermo['inv_T_K'] = 1 / df_thermo['T_K']
                    df_thermo['Kd'] = df_thermo['qe'] / df_thermo['Ce'] # L/g
                    df_thermo_valid_kd = df_thermo[df_thermo['Kd'] > 1e-9].copy()

                    if len(df_thermo_valid_kd['T_K'].unique()) >= 2:
                        df_thermo_valid_kd['ln_Kd'] = np.log(df_thermo_valid_kd['Kd'])
                        inv_T_valid = df_thermo_valid_kd['inv_T_K'].values
                        ln_Kd_valid = df_thermo_valid_kd['ln_Kd'].values
                        temps_K_valid = df_thermo_valid_kd['T_K'].values
                        temps_C_valid = df_thermo_valid_kd['Temperature_C'].values
                        kd_values_valid = df_thermo_valid_kd['Kd'].values
                        
                        if df_thermo_valid_kd['inv_T_K'].nunique() < 2 or df_thermo_valid_kd['ln_Kd'].nunique() < 2:
                             st.warning(_t("thermo_warning_insufficient_variation_vant_hoff"))
                             raise ValueError("Insufficient variation for Van't Hoff")
                        
                        slope, intercept, r_val, _, _ = linregress(inv_T_valid, ln_Kd_valid)
                        delta_H_J_mol = -slope * R_gas_const
                        delta_S_J_mol_K = intercept * R_gas_const
                        r_squared_vt = r_val**2
                        delta_G_kJ_mol_dict = {round(T_c, 1): (delta_H_J_mol - T_k * delta_S_J_mol_K) / 1000
                                                for T_k, T_c in zip(temps_K_valid, temps_C_valid)}
                        
                        st.session_state['thermo_params'] = {
                            'Delta_H_kJ_mol': delta_H_J_mol / 1000, 'Delta_S_J_mol_K': delta_S_J_mol_K,
                            'Delta_G_kJ_mol': delta_G_kJ_mol_dict, 'R2_Van_t_Hoff': r_squared_vt,
                            'ln_K': ln_Kd_valid.tolist(), 'inv_T': inv_T_valid.tolist(), 
                            'temps_K_valid': temps_K_valid.tolist(), 
                            'temps_C_valid': temps_C_valid.tolist(), # Added for K_values keys
                            'K_values': dict(zip(temps_C_valid, kd_values_valid)), # Store Kd vs Temp_C
                            'Analysis_Type': 'Kd' 
                        }
                        st.success(_t("thermo_success_analysis_kd"))
                    else:
                        st.warning(_t("thermo_warning_not_enough_distinct_temps_kd"))
                        st.session_state['thermo_params'] = None
                except ValueError as ve: 
                    # Warning for insufficient variation already shown by the code raising it or will be shown if it's a different ValueError
                    if "Insufficient variation" not in str(ve): # Avoid double message if this was the cause
                        st.error(_t("thermo_error_vant_hoff_kd", e_vth=ve))
                    st.session_state['thermo_params'] = None
                except Exception as e_vth:
                    st.error(_t("thermo_error_vant_hoff_kd", e_vth=e_vth))
                    st.session_state['thermo_params'] = None
            else:
                # This condition usually means not enough unique TEMPERATURES with valid Ce>0, qe>=0 points.
                st.warning(_t("thermo_warning_not_enough_distinct_temps_ce")) 
                st.session_state['thermo_params'] = None
            thermo_params = st.session_state.get('thermo_params') # Re-fetch

    if thermo_params and thermo_params.get('Analysis_Type') == 'Kd':
        st.markdown(_t("thermo_calculated_params_header"))
        col_th1, col_th2 = st.columns(2)
        with col_th1:
            st.metric("ΔH° (kJ/mol)", f"{thermo_params['Delta_H_kJ_mol']:.2f}", help=_t("thermo_delta_h_help"))
            st.metric("ΔS° (J/mol·K)", f"{thermo_params['Delta_S_J_mol_K']:.2f}", help=_t("thermo_delta_s_help"))
            st.metric("R² (Van't Hoff)", f"{thermo_params['R2_Van_t_Hoff']:.3f}", help=_t("thermo_r2_vant_hoff_help"))
        with col_th2:
            st.write(_t("thermo_delta_g_header"))
            if thermo_params['Delta_G_kJ_mol']:
                 # Use the translated key for Temperature column name
                 dG_df = pd.DataFrame(list(thermo_params['Delta_G_kJ_mol'].items()), columns=[_t("thermo_kd_table_temp_c"), 'ΔG° (kJ/mol)'])
                 dG_df = dG_df.sort_values(by=_t("thermo_kd_table_temp_c")).reset_index(drop=True)
                 st.dataframe(dG_df.style.format({_t("thermo_kd_table_temp_c"): '{:.1f}','ΔG° (kJ/mol)': '{:.2f}'}), height=min(200, (len(dG_df)+1)*35 + 3))
                 st.caption(_t("thermo_delta_g_spontaneous_caption"))
            else: st.write(_t("thermo_delta_g_not_calculated"))

        st.markdown(_t("thermo_vant_hoff_plot_header"))
        try:
            if thermo_params.get('inv_T') and thermo_params.get('ln_K'):
                df_vt = pd.DataFrame({'1/T (1/K)': thermo_params['inv_T'], 'ln(Kd)': thermo_params['ln_K']})
                fig_vt = px.scatter(df_vt, x='1/T (1/K)', y='ln(Kd)', title=_t("thermo_vant_hoff_plot_title"), labels={'1/T (1/K)': '1 / T (1/K)', 'ln(Kd)': 'ln(Kd)'})
                R_gas = 8.314 # Re-define locally for clarity, though it's a const
                slope_vt_plot = -thermo_params['Delta_H_kJ_mol'] * 1000 / R_gas
                intercept_vt_plot = thermo_params['Delta_S_J_mol_K'] / R_gas
                inv_T_line = np.linspace(min(thermo_params['inv_T']), max(thermo_params['inv_T']), 50)
                ln_K_line = slope_vt_plot * inv_T_line + intercept_vt_plot
                fig_vt.add_trace(go.Scatter(x=inv_T_line, y=ln_K_line, mode='lines', name=_t("thermo_vant_hoff_plot_legend_fit", r2_vt=thermo_params["R2_Van_t_Hoff"])))
                fig_vt.update_layout(template="simple_white")
                st.plotly_chart(fig_vt, use_container_width=True)
                try:
                    fig_vt_styled = go.Figure() 
                    fig_vt_styled.add_trace(go.Scatter(x=thermo_params['inv_T'],y=thermo_params['ln_K'],mode='markers',marker=dict(symbol='square', color='black', size=10),name=_t("isotherm_exp_plot_legend")))
                    fig_vt_styled.add_trace(go.Scatter(x=inv_T_line,y=ln_K_line,mode='lines',line=dict(color='red', width=3),name=_t("calib_tab_legend_reg")))
                    fig_vt_styled.update_layout(width=1000,height=800,plot_bgcolor='white',paper_bgcolor='white',font=dict(family="Times New Roman", size=22, color="black"),margin=dict(l=80, r=40, t=60, b=80),xaxis=dict(title="1 / T (1/K)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),yaxis=dict(title="ln(Kd)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),showlegend=False)
                    fig_vt_styled.add_annotation(xref="paper", yref="paper",x=0.05, y=0.95,text=f"y = {slope_vt_plot:.4f}x + {intercept_vt_plot:.4f}<br>R² = {thermo_params['R2_Van_t_Hoff']:.4f}",showarrow=False,font=dict(size=20, color="black"),align="left")
                    vt_img_buffer = io.BytesIO(); fig_vt_styled.write_image(vt_img_buffer, format="png", width=1000, height=800, scale=2); vt_img_buffer.seek(0)
                    st.download_button(label=_t("download_png_button"),data=vt_img_buffer,file_name=_t("thermo_download_vant_hoff_styled_filename"),mime="image/png",key="dl_vt_stylise_tab_thermo") # Unique key
                except Exception as e: st.warning(_t("thermo_error_export_vant_hoff_styled", e=e))
        except Exception as e_vt_plot: st.warning(_t("thermo_error_plot_vant_hoff", e_vt_plot=e_vt_plot))

        st.markdown(_t("thermo_kd_coeffs_header"))
        if thermo_params.get('K_values'): 
            # K_values uses Temp_C as key, ensures correct T for display
            k_vals_list_display = [{'Température (°C)': T_c, 'Kd (L/g)': Kd_val} 
                                   for T_c, Kd_val in thermo_params['K_values'].items()]
            k_vals_df_display = pd.DataFrame(k_vals_list_display).sort_values(by='Température (°C)').reset_index(drop=True)
            st.dataframe(k_vals_df_display.style.format({'Température (°C)': '{:.1f}','Kd (L/g)': '{:.4g}'}))
        else: st.write(_t("thermo_kd_unavailable"))
        
        col_dlt1, col_dlt2 = st.columns(2)
        with col_dlt1:
            thermo_res_export = {'Delta_H_kJ_mol': thermo_params['Delta_H_kJ_mol'], 'Delta_S_J_mol_K': thermo_params['Delta_S_J_mol_K'], 'R2_Van_t_Hoff': thermo_params['R2_Van_t_Hoff'], **{f'Delta_G_kJ_mol_{T_C}C': G for T_C, G in thermo_params['Delta_G_kJ_mol'].items()}}
            thermo_df_export = pd.DataFrame([thermo_res_export])
            csv_t_params = convert_df_to_csv(thermo_df_export)
            st.download_button(_t("thermo_download_params_kd_button"), csv_t_params, _t("thermo_download_params_kd_filename"), "text/csv", key="dl_t_p_kd_tab_thermo_params") # Unique key
        with col_dlt2:
             if thermo_params.get('inv_T') and thermo_params.get('ln_K'):
                df_vt_export = pd.DataFrame({'1/T (1/K)': thermo_params['inv_T'], 'ln(Kd)': thermo_params['ln_K']})
                csv_vt_data = convert_df_to_csv(df_vt_export)
                st.download_button(_t("thermo_download_data_vant_hoff_kd_button"), csv_vt_data, _t("thermo_download_data_vant_hoff_kd_filename"), "text/csv", key="dl_vt_d_kd_tab_thermo_data") # Unique key

    elif temp_results_for_thermo is None or temp_results_for_thermo.empty:
         st.info(_t("thermo_info_provide_temp_data"))
    elif temp_results_for_thermo is not None and len(temp_results_for_thermo['Temperature_C'].unique()) < 2:
         st.warning(_t("thermo_warning_less_than_2_distinct_temps"))
    elif thermo_params is None and st.session_state.get('temp_effect_results') is not None: 
         st.warning(_t("thermo_warning_analysis_not_done_kd"))
    elif thermo_params and thermo_params.get('Analysis_Type') != 'Kd': 
        st.warning(_t("thermo_warning_params_calculated_differently"))