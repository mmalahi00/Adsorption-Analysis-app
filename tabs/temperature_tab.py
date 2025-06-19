# tabs/temperature_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
# from ..translations import _t
# from ..utils import convert_df_to_csv
from translations import _t
from utils import convert_df_to_csv

def render():
    st.header(_t("temp_effect_tab_header"))
    temp_input = st.session_state.get('temp_effect_input')
    calib_params = st.session_state.get('calibration_params')
    temp_results = st.session_state.get('temp_effect_results')

    if temp_input and calib_params:
        if temp_results is None: # Calculate Ce/qe
            with st.spinner(_t("temp_effect_spinner_ce_qe")):
                results_list_temp = []
                df_temp_data = temp_input['data'].copy()
                params_temp = temp_input['params']
                try:
                    if abs(calib_params['slope']) < 1e-9:
                        st.error(_t("temp_effect_error_slope_zero"))
                        st.session_state['temp_effect_results'] = None; return
                    m_fixed = params_temp['m']
                    if m_fixed <= 0:
                        st.error(_t("temp_effect_error_mass_non_positive", m_fixed=m_fixed))
                        st.session_state['temp_effect_results'] = None; return
                    
                    for _, row in df_temp_data.iterrows():
                        T_val = row['Temperature_C']
                        abs_eq = row['Absorbance_Equilibre']
                        ce = max(0, (abs_eq - calib_params['intercept']) / calib_params['slope'])
                        c0_fixed, v_fixed = params_temp['C0'], params_temp['V']
                        qe = max(0, (c0_fixed - ce) * v_fixed / m_fixed)
                        results_list_temp.append({'Temperature_C': T_val, 'Abs_Eq': abs_eq, 'Ce': ce, 'qe': qe, 
                                                   'C0_fixe': c0_fixed, 'Masse_fixe_g': m_fixed, 'Volume_fixe_L': v_fixed})
                    if results_list_temp:
                        st.session_state['temp_effect_results'] = pd.DataFrame(results_list_temp)
                        st.success(_t("temp_effect_success_ce_qe_calc"))
                        st.session_state['thermo_params'] = None # Reset thermo if temp data changes
                    else:
                        st.warning(_t("temp_effect_warning_no_valid_points"))
                        st.session_state['temp_effect_results'] = pd.DataFrame(columns=['Temperature_C', 'Abs_Eq', 'Ce', 'qe', 'C0_fixe', 'Masse_fixe_g', 'Volume_fixe_L'])
                except (ZeroDivisionError, ValueError): # Specific errors already handled
                    st.session_state['temp_effect_results'] = None
                except Exception as e:
                    st.error(_t("temp_effect_error_ce_qe_calc_general", e=e))
                    st.session_state['temp_effect_results'] = None
                temp_results = st.session_state.get('temp_effect_results') # Re-fetch

        if temp_results is not None and not temp_results.empty:
            st.markdown(_t("temp_effect_calculated_data_header"))
            st.dataframe(temp_results[['Temperature_C', 'Abs_Eq', 'Ce', 'qe']].style.format({'Temperature_C': '{:.1f}', 'Abs_Eq': '{:.4f}', 'Ce': '{:.4f}', 'qe': '{:.4f}'}))
            st.caption(_t("temp_effect_conditions_caption", C0=temp_input['params']['C0'], m=temp_input['params']['m'], V=temp_input['params']['V']))
            csv_t_res = convert_df_to_csv(temp_results)
            st.download_button(_t("temp_effect_download_data_button"), csv_t_res, _t("temp_effect_download_data_filename"), "text/csv", key='dl_t_eff_data_tab_temp') # Unique key
            st.markdown(_t("temp_effect_plot_header"))
            try:
                temp_results_sorted = temp_results.sort_values('Temperature_C')
                fig_t = px.scatter(temp_results_sorted, x='Temperature_C', y='qe', title=_t("temp_effect_plot_title"), labels={'Temperature_C': _t("temp_effect_plot_xaxis"), 'qe': 'qe (mg/g)'}, hover_data=temp_results_sorted.columns)
                fig_t.add_trace(go.Scatter(x=temp_results_sorted['Temperature_C'], y=temp_results_sorted['qe'], mode='lines', name=_t("temp_effect_plot_legend_trend"), showlegend=False))
                fig_t.update_layout(template="simple_white")
                st.plotly_chart(fig_t, use_container_width=True)
                try:
                    df_temp_styled_dl = temp_results_sorted.copy()
                    fig_temp_styled = go.Figure() 
                    fig_temp_styled.add_trace(go.Scatter(x=df_temp_styled_dl['Temperature_C'],y=df_temp_styled_dl['qe'],mode='markers+lines',marker=dict(symbol='square', color='black', size=10),line=dict(color='red', width=3),name=_t("isotherm_exp_plot_legend")))
                    fig_temp_styled.update_layout(width=1000,height=800,plot_bgcolor='white',paper_bgcolor='white',font=dict(family="Times New Roman", size=22, color="black"),margin=dict(l=80, r=40, t=60, b=80),xaxis=dict(title=_t("temp_effect_plot_xaxis"),linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),yaxis=dict(title="qe (mg/g)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),showlegend=False)
                    temp_img_buffer = io.BytesIO(); fig_temp_styled.write_image(temp_img_buffer, format="png", width=1000, height=800, scale=2); temp_img_buffer.seek(0)
                    st.download_button(label=_t("download_png_button"),data=temp_img_buffer,file_name=_t("temp_effect_download_styled_plot_filename"),mime="image/png",key='dl_temp_fig_stylisee_tab_temp') # Unique key
                except Exception as e_export_temp: st.warning(_t("temp_effect_error_export_styled_plot", e_export_temp=e_export_temp))
            except Exception as e_t_plot: st.warning(_t("temp_effect_error_plot_general", e_t_plot=e_t_plot))

        elif temp_results is not None and temp_results.empty:
            st.warning(_t("temp_effect_warning_ce_qe_no_results"))

    elif not calib_params:
        st.warning(_t("isotherm_warning_provide_calib_data"))
    else:
        st.info(_t("temp_effect_info_enter_temp_data"))