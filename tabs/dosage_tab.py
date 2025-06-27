# tabs/dosage_tab.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from utils import convert_df_to_csv

def render():
    st.header("Effect of Adsorbent Dose (Mass)")
    dosage_input = st.session_state.get('dosage_input')
    calib_params = st.session_state.get('calibration_params')
    dosage_results = st.session_state.get('dosage_results')

    if dosage_input and calib_params:
        if dosage_results is None: 
            with st.spinner("Calculating Ce/qe for dosage effect..."):
                results_list_dos = []
                df_dos_data = dosage_input['data'].copy()
                params_dos = dosage_input['params']
                try:
                    if abs(calib_params['slope']) < 1e-9:
                        st.error("Error calculating Ce/qe: Calibration slope is zero.")
                        st.session_state['dosage_results'] = None; return
                    v_fixed = params_dos.get('V', 0)
                    c0_fixed = params_dos.get('C0', 0)
                    if v_fixed <= 0:
                        st.error(f"Error calculating qe: Fixed volume ({v_fixed}L) is invalid.")
                        st.session_state['dosage_results'] = None; return

                    for _, row in df_dos_data.iterrows():
                        m_adsorbant = row['Mass']
                        abs_eq = row['Absorbance']
                        ce = max(0, (abs_eq - calib_params['intercept']) / calib_params['slope'])
                        qe = max(0, (c0_fixed - ce) * v_fixed / m_adsorbant)
                        results_list_dos.append({'Mass': m_adsorbant, 'Abs_Eq': abs_eq, 'Ce': ce, 'qe': qe,
                                                 'C0_fixe': c0_fixed, 'Volume_fixe_L': v_fixed})
                    if results_list_dos:
                        st.session_state['dosage_results'] = pd.DataFrame(results_list_dos)
                        st.success("Ce/qe calculation for dosage effect complete.")
                    else:
                        st.warning("No valid dosage points after Ce/qe calculation.")
                        st.session_state['dosage_results'] = pd.DataFrame(columns=['Mass', 'Abs_Eq', 'Ce', 'qe', 'C0_fixe', 'Volume_fixe_L'])
                except (ZeroDivisionError, ValueError) as calc_err_dos:
                    st.error(f"Error calculating Ce/qe for dosage effect: {calc_err_dos}")
                    st.session_state['dosage_results'] = None
                except Exception as e:
                    st.error(f"Unexpected error calculating Ce/qe for dosage effect: {e}")
                    st.session_state['dosage_results'] = None
                dosage_results = st.session_state.get('dosage_results') 

        if dosage_results is not None and not dosage_results.empty:
            st.markdown("##### Calculated Data (qe vs Adsorbent Mass)")
            display_cols = ['Mass', 'Abs_Eq', 'Ce', 'qe']
            st.dataframe(dosage_results[display_cols].style.format({'Mass': '{:.4f}', 'Abs_Eq': '{:.4f}', 'Ce': '{:.4f}', 'qe': '{:.4f}'}))
            st.caption(f"Fixed conditions: C0={dosage_input.get('params', {}).get('C0', 'N/A')}mg/L, V={dosage_input.get('params', {}).get('V', 'N/A')}L")
            csv_dos_res = convert_df_to_csv(dosage_results)
            st.download_button("📥 DL Dosage Effect Data", csv_dos_res, "dosage_effect_results.csv", "text/csv", key='dl_dos_eff_data_tab_dos') 
            st.markdown("##### qe vs Adsorbent Mass Plot")
            try:
                dosage_results_sorted = dosage_results.sort_values('Mass')
                fig_dos = px.scatter(dosage_results_sorted, x='Mass', y='qe', title="Effect of Adsorbent Mass on qe", labels={'Mass': "Adsorbent Mass (g)", 'qe': 'qe (mg/g)'}, hover_data=dosage_results_sorted.columns)
                fig_dos.add_trace(go.Scatter(x=dosage_results_sorted['Mass'], y=dosage_results_sorted['qe'], mode='lines', name="Trend", showlegend=False))
                fig_dos.update_layout(template="simple_white", width=500, height=500)
                st.plotly_chart(fig_dos, use_container_width=False)
                try:
                    df_dos_styled_dl = dosage_results_sorted.copy()
                    fig_dos_styled = go.Figure() 
                    fig_dos_styled.add_trace(go.Scatter(x=df_dos_styled_dl['Mass'],y=df_dos_styled_dl['qe'],mode='markers+lines',marker=dict(symbol='square', color='black', size=10),line=dict(color='red', width=3),name="Experimental data"))
                    fig_dos_styled.update_layout(width=1000,height=800,plot_bgcolor='white',paper_bgcolor='white',font=dict(family="Times New Roman", size=22, color="black"),margin=dict(l=80, r=40, t=60, b=80),xaxis=dict(title="Adsorbent Mass (g)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),yaxis=dict(title="qe (mg/g)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),showlegend=False)
                    dos_img_buffer = io.BytesIO(); fig_dos_styled.write_image(dos_img_buffer, format="png", width=1000, height=800, scale=2); dos_img_buffer.seek(0)
                    st.download_button(label="📥 Download Figure (PNG)",data=dos_img_buffer,file_name="dosage_effect_styled.png",mime="image/png",key='dl_dos_fig_stylisee_tab_dos') 
                except Exception as e_export_dos: st.warning(f"Error exporting styled dosage figure: {e_export_dos}")
            except Exception as e_dos_plot: st.warning(f"Error plotting Dosage Effect: {e_dos_plot}")

        elif dosage_results is not None and dosage_results.empty:
            st.warning("Ce/qe calculation produced no valid results for dosage effect.")

    elif not calib_params:
        st.warning("Please provide valid calibration data and calculate parameters first.")
    else:
        st.info("Please enter data for the dosage effect study (adsorbent mass).")