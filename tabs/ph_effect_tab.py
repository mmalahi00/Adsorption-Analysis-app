# tabs/ph_effect_tab.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from utils import convert_df_to_csv

def render():
    st.header("Effect of pH on Adsorption Capacity (qe)")
    ph_input = st.session_state.get('ph_effect_input')
    calib_params = st.session_state.get('calibration_params')
    ph_results = st.session_state.get('ph_effect_results')

    if ph_input and calib_params:
        if ph_results is None:
            with st.spinner("Calculating Ce/qe for pH effect..."):
                results_list_ph = []
                df_ph_data = ph_input['data'].copy()
                params_ph = ph_input['params']
                try:
                    if abs(calib_params['slope']) < 1e-9:
                        st.error("Error calculating Ce/qe: Calibration slope is zero.")
                        st.session_state['ph_effect_results'] = None; return
                    m_fixed = params_ph['m']
                    if m_fixed <= 0:
                        st.error(f"Error calculating qe: Fixed mass is non-positive ({m_fixed}g).")
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
                        st.success("Ce/qe calculation for pH effect complete.")
                    else:
                        st.warning("No valid pH points after Ce/qe calculation.")
                        st.session_state['ph_effect_results'] = pd.DataFrame(columns=['pH', 'Abs_Eq', 'Ce', 'qe', 'C0_fixe', 'Masse_fixe_g', 'Volume_fixe_L'])
                except (ZeroDivisionError, ValueError): 
                    st.session_state['ph_effect_results'] = None 
                except Exception as e:
                    st.error(f"Error calculating Ce/qe for pH effect: {e}")
                    st.session_state['ph_effect_results'] = None
                ph_results = st.session_state.get('ph_effect_results') 


        if ph_results is not None and not ph_results.empty:
            st.markdown("##### Calculated Data (qe vs pH)")
            st.dataframe(ph_results[['pH', 'Abs_Eq', 'Ce', 'qe']].style.format({'pH': '{:.2f}', 'Abs_Eq': '{:.4f}', 'Ce': '{:.4f}', 'qe': '{:.4f}'}))
            st.caption(f"Fixed conditions: C0={ph_input['params']['C0']}mg/L, m={ph_input['params']['m']}g, V={ph_input['params']['V']}L")
            csv_ph_res = convert_df_to_csv(ph_results)
            st.download_button("📥 DL pH Effect Data", csv_ph_res, "ph_effect_results.csv", "text/csv", key='dl_ph_eff_data_tab')
            st.markdown("##### qe vs pH Plot")
            try:
                fig_ph = px.scatter(ph_results, x='pH', y='qe', title="Effect of pH on qe", labels={'pH': 'pH', 'qe': 'qe (mg/g)'}, hover_data=ph_results.columns)
                ph_results_sorted = ph_results.sort_values('pH')
                fig_ph.add_trace(go.Scatter(x=ph_results_sorted['pH'], y=ph_results_sorted['qe'], mode='lines', name="Trend", showlegend=False))
                fig_ph.update_layout(template="simple_white", width=500, height=500)
                st.plotly_chart(fig_ph, use_container_width=False)
                try:
                    df_ph_styled_dl = ph_results.sort_values(by='pH').copy()
                    fig_ph_styled = go.Figure() 
                    fig_ph_styled.add_trace(go.Scatter(x=df_ph_styled_dl['pH'],y=df_ph_styled_dl['qe'],mode='markers+lines',marker=dict(symbol='square', color='black', size=10),line=dict(color='red', width=3),name="Experimental data"))
                    fig_ph_styled.update_layout(width=1000,height=800,plot_bgcolor='white',paper_bgcolor='white',font=dict(family="Times New Roman", size=22, color="black"),margin=dict(l=80, r=40, t=60, b=80),xaxis=dict(title="pH",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),yaxis=dict(title="qe (mg/g)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),showlegend=False)
                    ph_img_buffer = io.BytesIO(); fig_ph_styled.write_image(ph_img_buffer, format="png", width=1000, height=800, scale=2); ph_img_buffer.seek(0)
                    st.download_button(label="📥 Download Figure (PNG)",data=ph_img_buffer,file_name="ph_effect_styled.png",mime="image/png",key='dl_ph_fig_stylisee_tab_ph') 
                except Exception as e_export_ph: st.warning(f"Error exporting styled pH figure: {e_export_ph}")
            except Exception as e_ph_plot: st.warning(f"Error plotting pH Effect: {e_ph_plot}")

        elif ph_results is not None and ph_results.empty:
            st.warning("Ce/qe calculation produced no valid results for pH effect.")
            
    elif not calib_params:
        st.warning("Please provide valid calibration data and calculate parameters first.") 
    else:
        st.info("Please enter data for the pH effect study.")