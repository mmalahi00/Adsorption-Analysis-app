# tabs/ph_effect_tab.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from translations import _t
from utils import convert_df_to_csv

def render():
    st.header(_t("ph_effect_tab_header"))
    ph_input = st.session_state.get('ph_effect_input')
    calib_params = st.session_state.get('calibration_params')
    ph_results = st.session_state.get('ph_effect_results')

    if ph_input and calib_params:
        if ph_results is None: # Calculate Ce/qe
            with st.spinner(_t("ph_effect_spinner_ce_qe")):
                results_list_ph = []
                df_ph_data = ph_input['data'].copy()
                params_ph = ph_input['params']
                try:
                    if abs(calib_params['slope']) < 1e-9:
                        st.error(_t("ph_effect_error_slope_zero"))
                        st.session_state['ph_effect_results'] = None; return
                    m_fixed = params_ph['m']
                    if m_fixed <= 0:
                        st.error(_t("ph_effect_error_mass_non_positive", m_fixed=m_fixed))
                        st.session_state['ph_effect_results'] = None; return

                    for _, row in df_ph_data.iterrows():
                        ph_val = row['pH']
                        abs_eq = row['Absorbance_Equilibre']
                        ce = max(0, (abs_eq - calib_params['intercept']) / calib_params['slope'])
                        c0_fixed, v_fixed = params_ph['C0'], params_ph['V']
                        qe = max(0, (c0_fixed - ce) * v_fixed / m_fixed)
                        results_list_ph.append({'pH': ph_val, 'Abs_Eq': abs_eq, 'Ce': ce, 'qe': qe, 
                                                'C0_fixe': c0_fixed, 'Masse_fixe_g': m_fixed, 'Volume_fixe_L': v_fixed})
                    if results_list_ph:
                        st.session_state['ph_effect_results'] = pd.DataFrame(results_list_ph)
                        st.success(_t("ph_effect_success_ce_qe_calc"))
                    else:
                        st.warning(_t("ph_effect_warning_no_valid_points"))
                        st.session_state['ph_effect_results'] = pd.DataFrame(columns=['pH', 'Abs_Eq', 'Ce', 'qe', 'C0_fixe', 'Masse_fixe_g', 'Volume_fixe_L'])
                except (ZeroDivisionError, ValueError): # Specific errors already handled with st.error
                    st.session_state['ph_effect_results'] = None 
                except Exception as e:
                    st.error(_t("ph_effect_error_ce_qe_calc_general", e=e))
                    st.session_state['ph_effect_results'] = None
                ph_results = st.session_state.get('ph_effect_results') # Re-fetch


        if ph_results is not None and not ph_results.empty:
            st.markdown(_t("ph_effect_calculated_data_header"))
            st.dataframe(ph_results[['pH', 'Abs_Eq', 'Ce', 'qe']].style.format({'pH': '{:.2f}', 'Abs_Eq': '{:.4f}', 'Ce': '{:.4f}', 'qe': '{:.4f}'}))
            st.caption(_t("ph_effect_conditions_caption", C0=ph_input['params']['C0'], m=ph_input['params']['m'], V=ph_input['params']['V']))
            csv_ph_res = convert_df_to_csv(ph_results)
            st.download_button(_t("ph_effect_download_data_button"), csv_ph_res, _t("ph_effect_download_data_filename"), "text/csv", key='dl_ph_eff_data_tab')
            st.markdown(_t("ph_effect_plot_header"))
            try:
                fig_ph = px.scatter(ph_results, x='pH', y='qe', title=_t("ph_effect_plot_title"), labels={'pH': 'pH', 'qe': 'qe (mg/g)'}, hover_data=ph_results.columns)
                ph_results_sorted = ph_results.sort_values('pH')
                fig_ph.add_trace(go.Scatter(x=ph_results_sorted['pH'], y=ph_results_sorted['qe'], mode='lines', name=_t("ph_effect_plot_legend_trend"), showlegend=False))
                fig_ph.update_layout(template="simple_white", width=500, height=500)
                st.plotly_chart(fig_ph, use_container_width=False)
                try:
                    df_ph_styled_dl = ph_results.sort_values(by='pH').copy()
                    fig_ph_styled = go.Figure() 
                    fig_ph_styled.add_trace(go.Scatter(x=df_ph_styled_dl['pH'],y=df_ph_styled_dl['qe'],mode='markers+lines',marker=dict(symbol='square', color='black', size=10),line=dict(color='red', width=3),name=_t("isotherm_exp_plot_legend")))
                    fig_ph_styled.update_layout(width=1000,height=800,plot_bgcolor='white',paper_bgcolor='white',font=dict(family="Times New Roman", size=22, color="black"),margin=dict(l=80, r=40, t=60, b=80),xaxis=dict(title="pH",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),yaxis=dict(title="qe (mg/g)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),showlegend=False)
                    ph_img_buffer = io.BytesIO(); fig_ph_styled.write_image(ph_img_buffer, format="png", width=1000, height=800, scale=2); ph_img_buffer.seek(0)
                    st.download_button(label=_t("download_png_button"),data=ph_img_buffer,file_name=_t("ph_effect_download_styled_plot_filename"),mime="image/png",key='dl_ph_fig_stylisee_tab_ph') # Unique key
                except Exception as e_export_ph: st.warning(_t("ph_effect_error_export_styled_plot", e_export_ph=e_export_ph))
            except Exception as e_ph_plot: st.warning(_t("ph_effect_error_plot_general", e_ph_plot=e_ph_plot))

        elif ph_results is not None and ph_results.empty:
            st.warning(_t("ph_effect_warning_ce_qe_no_results"))
            
    elif not calib_params:
        st.warning(_t("isotherm_warning_provide_calib_data")) 
    else:
        st.info(_t("ph_effect_info_enter_ph_data"))