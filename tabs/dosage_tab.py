# tabs/dosage_tab.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from translations import _t
from utils import convert_df_to_csv

def render():
    st.header(_t("dosage_tab_header"))
    dosage_input = st.session_state.get('dosage_input')
    calib_params = st.session_state.get('calibration_params')
    dosage_results = st.session_state.get('dosage_results')

    if dosage_input and calib_params:
        if dosage_results is None: 
            with st.spinner(_t("dosage_spinner_ce_qe")):
                results_list_dos = []
                df_dos_data = dosage_input['data'].copy()
                params_dos = dosage_input['params']
                try:
                    if abs(calib_params['slope']) < 1e-9:
                        st.error(_t("dosage_error_slope_zero"))
                        st.session_state['dosage_results'] = None; return
                    v_fixed = params_dos.get('V', 0)
                    c0_fixed = params_dos.get('C0', 0)
                    if v_fixed <= 0:
                        st.error(_t("dosage_error_volume_non_positive", v_fixed=v_fixed))
                        st.session_state['dosage_results'] = None; return

                    for _, row in df_dos_data.iterrows():
                        m_adsorbant = row['Masse_Adsorbant_g']
                        abs_eq = row['Absorbance_Equilibre']
                        ce = max(0, (abs_eq - calib_params['intercept']) / calib_params['slope'])
                        qe = max(0, (c0_fixed - ce) * v_fixed / m_adsorbant)
                        results_list_dos.append({'Masse_Adsorbant_g': m_adsorbant, 'Abs_Eq': abs_eq, 'Ce': ce, 'qe': qe,
                                                 'C0_fixe': c0_fixed, 'Volume_fixe_L': v_fixed})
                    if results_list_dos:
                        st.session_state['dosage_results'] = pd.DataFrame(results_list_dos)
                        st.success(_t("dosage_success_ce_qe_calc"))
                    else:
                        st.warning(_t("dosage_warning_no_valid_points"))
                        st.session_state['dosage_results'] = pd.DataFrame(columns=['Masse_Adsorbant_g', 'Abs_Eq', 'Ce', 'qe', 'C0_fixe', 'Volume_fixe_L'])
                except (ZeroDivisionError, ValueError) as calc_err_dos:
                    if 'dosage_results' not in st.session_state or st.session_state.get('dosage_results') is not None:
                        st.error(_t("dosage_error_ce_qe_calc_general", calc_err_dos=calc_err_dos))
                    st.session_state['dosage_results'] = None
                except Exception as e:
                    st.error(_t("dosage_error_ce_qe_calc_unexpected", e=e))
                    st.session_state['dosage_results'] = None
                dosage_results = st.session_state.get('dosage_results') 

        if dosage_results is not None and not dosage_results.empty:
            st.markdown(_t("dosage_calculated_data_header"))
            display_cols = ['Masse_Adsorbant_g', 'Abs_Eq', 'Ce', 'qe']
            st.dataframe(dosage_results[display_cols].style.format({'Masse_Adsorbant_g': '{:.4f}', 'Abs_Eq': '{:.4f}', 'Ce': '{:.4f}', 'qe': '{:.4f}'}))
            st.caption(_t("dosage_conditions_caption", C0=dosage_input.get('params', {}).get('C0', 'N/A'), V=dosage_input.get('params', {}).get('V', 'N/A')))
            csv_dos_res = convert_df_to_csv(dosage_results)
            st.download_button(_t("dosage_download_data_button"), csv_dos_res, _t("dosage_download_data_filename"), "text/csv", key='dl_dos_eff_data_tab_dos') 
            st.markdown(_t("dosage_plot_header"))
            try:
                dosage_results_sorted = dosage_results.sort_values('Masse_Adsorbant_g')
                fig_dos = px.scatter(dosage_results_sorted, x='Masse_Adsorbant_g', y='qe', title=_t("dosage_plot_title"), labels={'Masse_Adsorbant_g': _t("dosage_plot_xaxis"), 'qe': 'qe (mg/g)'}, hover_data=dosage_results_sorted.columns)
                fig_dos.add_trace(go.Scatter(x=dosage_results_sorted['Masse_Adsorbant_g'], y=dosage_results_sorted['qe'], mode='lines', name=_t("dosage_plot_legend_trend"), showlegend=False))
                fig_dos.update_layout(template="simple_white", width=500, height=500)
                st.plotly_chart(fig_dos, use_container_width=False)
                try:
                    df_dos_styled_dl = dosage_results_sorted.copy()
                    fig_dos_styled = go.Figure() 
                    fig_dos_styled.add_trace(go.Scatter(x=df_dos_styled_dl['Masse_Adsorbant_g'],y=df_dos_styled_dl['qe'],mode='markers+lines',marker=dict(symbol='square', color='black', size=10),line=dict(color='red', width=3),name=_t("isotherm_exp_plot_legend")))
                    fig_dos_styled.update_layout(width=1000,height=800,plot_bgcolor='white',paper_bgcolor='white',font=dict(family="Times New Roman", size=22, color="black"),margin=dict(l=80, r=40, t=60, b=80),xaxis=dict(title=_t("dosage_plot_xaxis"),linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),yaxis=dict(title="qe (mg/g)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),showlegend=False)
                    dos_img_buffer = io.BytesIO(); fig_dos_styled.write_image(dos_img_buffer, format="png", width=1000, height=800, scale=2); dos_img_buffer.seek(0)
                    st.download_button(label=_t("download_png_button"),data=dos_img_buffer,file_name=_t("dosage_download_styled_plot_filename"),mime="image/png",key='dl_dos_fig_stylisee_tab_dos') 
                except Exception as e_export_dos: st.warning(_t("dosage_error_export_styled_plot", e_export_dos=e_export_dos))
            except Exception as e_dos_plot: st.warning(_t("dosage_error_plot_general", e_dos_plot=e_dos_plot))

        elif dosage_results is not None and dosage_results.empty:
            st.warning(_t("dosage_warning_ce_qe_no_results"))

    elif not calib_params:
        st.warning(_t("isotherm_warning_provide_calib_data"))
    else:
        st.info(_t("dosage_info_enter_dosage_data"))