# tabs/temperature_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from utils import convert_df_to_csv

def render():
    st.header("Effect of Temperature on Adsorption Capacity (qe)")
    temp_input = st.session_state.get('temp_effect_input')
    calib_params = st.session_state.get('calibration_params')
    temp_results = st.session_state.get('temp_effect_results')

    if temp_input and calib_params:
        if temp_results is None: 
            with st.spinner("Calculating Ce/qe for T° effect..."):
                results_list_temp = []
                df_temp_data = temp_input['data'].copy()
                params_temp = temp_input['params']
                try:
                    if abs(calib_params['slope']) < 1e-9:
                        st.error("Error calculating Ce/qe: Calibration slope is zero.")
                        st.session_state['temp_effect_results'] = None; return
                    m_fixed = params_temp['m']
                    if m_fixed <= 0:
                        st.error(f"Error calculating qe: Fixed mass is non-positive ({m_fixed}g).")
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
                        st.success("Ce/qe calculation for T° effect complete.")
                        st.session_state['thermo_params'] = None # Reset thermo if temp data changes
                    else:
                        st.warning("No valid T° points after Ce/qe calculation.")
                        st.session_state['temp_effect_results'] = pd.DataFrame(columns=['Temperature_C', 'Abs_Eq', 'Ce', 'qe', 'C0_fixe', 'Masse_fixe_g', 'Volume_fixe_L'])
                except (ZeroDivisionError, ValueError): 
                    st.session_state['temp_effect_results'] = None
                except Exception as e:
                    st.error(f"Error calculating Ce/qe for T° effect: {e}")
                    st.session_state['temp_effect_results'] = None
                temp_results = st.session_state.get('temp_effect_results') 

        if temp_results is not None and not temp_results.empty:
            st.markdown("##### Calculated Data (qe vs T°)")
            st.dataframe(temp_results[['Temperature_C', 'Abs_Eq', 'Ce', 'qe']].style.format({'Temperature_C': '{:.1f}', 'Abs_Eq': '{:.4f}', 'Ce': '{:.4f}', 'qe': '{:.4f}'}))
            st.caption(f"Fixed conditions: C0={temp_input['params']['C0']}mg/L, m={temp_input['params']['m']}g, V={temp_input['params']['V']}L")
            csv_t_res = convert_df_to_csv(temp_results)
            st.download_button("📥 DL T° Effect Data", csv_t_res, "temp_effect_results.csv", "text/csv", key='dl_t_eff_data_tab_temp') 
            st.markdown("##### qe vs T° Plot")
            try:
                temp_results_sorted = temp_results.sort_values('Temperature_C')
                fig_t = px.scatter(temp_results_sorted, x='Temperature_C', y='qe', title="Effect of T° on qe", labels={'Temperature_C': "Temperature (°C)", 'qe': 'qe (mg/g)'}, hover_data=temp_results_sorted.columns)
                fig_t.add_trace(go.Scatter(x=temp_results_sorted['Temperature_C'], y=temp_results_sorted['qe'], mode='lines', name="Trend", showlegend=False))
                fig_t.update_layout(template="simple_white")
                st.plotly_chart(fig_t, use_container_width=True)
                try:
                    df_temp_styled_dl = temp_results_sorted.copy()
                    fig_temp_styled = go.Figure() 
                    fig_temp_styled.add_trace(go.Scatter(x=df_temp_styled_dl['Temperature_C'],y=df_temp_styled_dl['qe'],mode='markers+lines',marker=dict(symbol='square', color='black', size=10),line=dict(color='red', width=3),name="Experimental data"))
                    fig_temp_styled.update_layout(width=1000,height=800,plot_bgcolor='white',paper_bgcolor='white',font=dict(family="Times New Roman", size=22, color="black"),margin=dict(l=80, r=40, t=60, b=80),xaxis=dict(title="Temperature (°C)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),yaxis=dict(title="qe (mg/g)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),showlegend=False)
                    temp_img_buffer = io.BytesIO(); fig_temp_styled.write_image(temp_img_buffer, format="png", width=1000, height=800, scale=2); temp_img_buffer.seek(0)
                    st.download_button(label="📥 Download Figure (PNG)",data=temp_img_buffer,file_name="temperature_effect_styled.png",mime="image/png",key='dl_temp_fig_stylisee_tab_temp') 
                except Exception as e_export_temp: st.warning(f"Error exporting styled temperature figure: {e_export_temp}")
            except Exception as e_t_plot: st.warning(f"Error plotting T° Effect: {e_t_plot}")

        elif temp_results is not None and temp_results.empty:
            st.warning("Ce/qe calculation produced no valid results for T° effect.")

    elif not calib_params:
        st.warning("Please provide valid calibration data and calculate parameters first.")
    else:
        st.info("Please enter data for the temperature effect study.")