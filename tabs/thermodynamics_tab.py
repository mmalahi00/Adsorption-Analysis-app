# tabs/thermodynamics_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress
import io
from utils import convert_df_to_csv

R_gas_const = 8.314 # J/mol·K  

def render():
    st.header("Thermodynamic Analysis")
    st.markdown("""
        This analysis uses data from the **Temperature Effect** study.
        It calculates Kd = qe / Ce (L/g) then uses **Van't Hoff** (ln(Kd) vs 1/T) to determine ΔH° and ΔS°.
        """)
    temp_results_for_thermo = st.session_state.get('temp_effect_results')
    thermo_params = st.session_state.get('thermo_params')

    if temp_results_for_thermo is not None and not temp_results_for_thermo.empty and thermo_params is None:
        with st.spinner("Thermodynamic analysis based on Kd..."):
            df_thermo = temp_results_for_thermo.copy()
            df_thermo = df_thermo[(df_thermo['Ce'] > 1e-9) & (df_thermo['qe'] >= 0)].copy()
            if len(df_thermo['Temperature'].unique()) >= 2:
                try:
                    df_thermo['T_K'] = df_thermo['Temperature'] + 273.15
                    df_thermo['inv_T_K'] = 1 / df_thermo['T_K']
                    df_thermo['Kd'] = df_thermo['qe'] / df_thermo['Ce'] # L/g
                    df_thermo_valid_kd = df_thermo[df_thermo['Kd'] > 1e-9].copy()

                    if len(df_thermo_valid_kd['T_K'].unique()) >= 2:
                        df_thermo_valid_kd['ln_Kd'] = np.log(df_thermo_valid_kd['Kd'])
                        inv_T_valid = df_thermo_valid_kd['inv_T_K'].values
                        ln_Kd_valid = df_thermo_valid_kd['ln_Kd'].values
                        temps_K_valid = df_thermo_valid_kd['T_K'].values
                        temps_C_valid = df_thermo_valid_kd['Temperature'].values
                        kd_values_valid = df_thermo_valid_kd['Kd'].values
                        
                        if df_thermo_valid_kd['inv_T_K'].nunique() < 2 or df_thermo_valid_kd['ln_Kd'].nunique() < 2:
                             st.warning("Van't Hoff analysis impossible: insufficient variation in 1/T or ln(Kd).")
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
                            'temps_C_valid': temps_C_valid.tolist(), 
                            'K_values': dict(zip(temps_C_valid, kd_values_valid)), 
                            'Analysis_Type': 'Kd' 
                        }
                        st.success("Thermodynamic analysis based on Kd complete.")
                    else:
                        st.warning("Not enough distinct T° with Kd > 0 for Van't Hoff.")
                        st.session_state['thermo_params'] = None
                except ValueError as ve: 
                    if "Insufficient variation" not in str(ve): 
                        st.error(f"Van't Hoff analysis error (Kd): {ve}")
                    st.session_state['thermo_params'] = None
                except Exception as e_vth:
                    st.error(f"Van't Hoff analysis error (Kd): {e_vth}")
                    st.session_state['thermo_params'] = None
            else:
                st.warning("Fewer than 2 distinct T° with Ce > 0 for thermo analysis.") 
                st.session_state['thermo_params'] = None
            thermo_params = st.session_state.get('thermo_params') 

    if thermo_params and thermo_params.get('Analysis_Type') == 'Kd':
        st.markdown("#### Calculated Thermodynamic Parameters")
        col_th1, col_th2 = st.columns(2)
        with col_th1:
            st.metric("ΔH° (kJ/mol)", f"{thermo_params['Delta_H_kJ_mol']:.2f}", help="< 0: Exothermic, > 0: Endothermic.")
            st.metric("ΔS° (J/mol·K)", f"{thermo_params['Delta_S_J_mol_K']:.2f}", help="> 0: Increased disorder.")
            st.metric("R² (Van't Hoff)", f"{thermo_params['R2_Van_t_Hoff']:.3f}", help="Goodness of fit for ln(Kd) vs 1/T.")
        with col_th2:
            st.write("ΔG° (kJ/mol) at different T°:")
            if thermo_params['Delta_G_kJ_mol']:
                 dG_df = pd.DataFrame(list(thermo_params['Delta_G_kJ_mol'].items()), columns=["Temperature (°C)", 'ΔG° (kJ/mol)'])
                 dG_df = dG_df.sort_values(by="Temperature (°C)").reset_index(drop=True)
                 st.dataframe(dG_df.style.format({"Temperature (°C)": '{:.1f}','ΔG° (kJ/mol)': '{:.2f}'}), height=min(200, (len(dG_df)+1)*35 + 3))
                 st.caption("ΔG° < 0 : Spontaneous.")
            else: st.write("Not calculated.")

        st.markdown("#### Van't Hoff Plot (ln(Kd) vs 1/T)")
        try:
            if thermo_params.get('inv_T') and thermo_params.get('ln_K'):
                df_vt = pd.DataFrame({'1/T (1/K)': thermo_params['inv_T'], 'ln(Kd)': thermo_params['ln_K']})
                fig_vt = px.scatter(df_vt, x='1/T (1/K)', y='ln(Kd)', title="Van't Hoff Plot", labels={'1/T (1/K)': '1 / T (1/K)', 'ln(Kd)': 'ln(Kd)'})
                slope_vt_plot = -thermo_params['Delta_H_kJ_mol'] * 1000 / R_gas_const 
                intercept_vt_plot = thermo_params['Delta_S_J_mol_K'] / R_gas_const 
                inv_T_line = np.linspace(min(thermo_params['inv_T']), max(thermo_params['inv_T']), 50)
                ln_K_line = slope_vt_plot * inv_T_line + intercept_vt_plot
                fig_vt.add_trace(go.Scatter(x=inv_T_line, y=ln_K_line, mode='lines', name=f"Linear Fit (R²={thermo_params['R2_Van_t_Hoff']:.3f})"))
                fig_vt.update_layout(template="simple_white")
                st.plotly_chart(fig_vt, use_container_width=True)
                try:
                    fig_vt_styled = go.Figure() 
                    fig_vt_styled.add_trace(go.Scatter(x=thermo_params['inv_T'],y=thermo_params['ln_K'],mode='markers',marker=dict(symbol='square', color='black', size=10),name="Experimental data"))
                    fig_vt_styled.add_trace(go.Scatter(x=inv_T_line,y=ln_K_line,mode='lines',line=dict(color='red', width=3),name="Linear regression"))
                    fig_vt_styled.update_layout(width=1000,height=800,plot_bgcolor='white',paper_bgcolor='white',font=dict(family="Times New Roman", size=22, color="black"),margin=dict(l=80, r=40, t=60, b=80),xaxis=dict(title="1 / T (1/K)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),yaxis=dict(title="ln(Kd)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),showlegend=False)
                    operator_vt = "-" if intercept_vt_plot < 0 else "+"
                    equation_text_vt = f"y = {slope_vt_plot:.4f}x {operator_vt} {abs(intercept_vt_plot):.4f}"
                    fig_vt_styled.add_annotation(xref="paper", yref="paper",x=0.05, y=0.95,text=f"{equation_text_vt}<br>R² = {thermo_params['R2_Van_t_Hoff']:.4f}",showarrow=False,font=dict(size=20, color="black"),align="left")
                    vt_img_buffer = io.BytesIO(); fig_vt_styled.write_image(vt_img_buffer, format="png", width=1000, height=800, scale=2); vt_img_buffer.seek(0)
                    st.download_button(label="📥 Download Figure (PNG)",data=vt_img_buffer,file_name="vant_hoff_styled.png",mime="image/png",key="dl_vt_stylise_tab_thermo") 
                except Exception as e: st.warning(f"Error exporting styled Van’t Hoff: {e}")
        except Exception as e_vt_plot: st.warning(f"Error plotting Van't Hoff: {e_vt_plot}")

        st.markdown("##### Distribution Coefficients (Kd) Used")
        if thermo_params.get('K_values'): 
            k_vals_list_display = [{'Temperature (°C)': T_c, 'Kd (L/g)': Kd_val} 
                                   for T_c, Kd_val in thermo_params['K_values'].items()]
            k_vals_df_display = pd.DataFrame(k_vals_list_display).sort_values(by='Temperature (°C)').reset_index(drop=True)
            st.dataframe(k_vals_df_display.style.format({'Temperature (°C)': '{:.1f}','Kd (L/g)': '{:.4g}'}))
        else: st.write("Not available.")
        
        col_dlt1, col_dlt2 = st.columns(2)
        with col_dlt1:
            thermo_res_export = {'Delta_H_kJ_mol': thermo_params['Delta_H_kJ_mol'], 'Delta_S_J_mol_K': thermo_params['Delta_S_J_mol_K'], 'R2_Van_t_Hoff': thermo_params['R2_Van_t_Hoff'], **{f'Delta_G_kJ_mol_{T_C}C': G for T_C, G in thermo_params['Delta_G_kJ_mol'].items()}}
            thermo_df_export = pd.DataFrame([thermo_res_export])
            csv_t_params = convert_df_to_csv(thermo_df_export)
            st.download_button("📥 DL Thermo Parameters (Kd)", csv_t_params, "thermo_params_kd.csv", "text/csv", key="dl_t_p_kd_tab_thermo_params") 
        with col_dlt2:
             if thermo_params.get('inv_T') and thermo_params.get('ln_K'):
                df_vt_export = pd.DataFrame({'1/T (1/K)': thermo_params['inv_T'], 'ln(Kd)': thermo_params['ln_K']})
                csv_vt_data = convert_df_to_csv(df_vt_export)
                st.download_button("📥 DL Van't Hoff Data (Kd)", csv_vt_data, "vant_hoff_data_kd.csv", "text/csv", key="dl_vt_d_kd_tab_thermo_data")

    elif temp_results_for_thermo is None or temp_results_for_thermo.empty:
         st.info("Please provide valid data for the T° Effect study.")
    elif temp_results_for_thermo is not None and len(temp_results_for_thermo['Temperature'].unique()) < 2:
         st.warning("Fewer than 2 distinct T° for thermo analysis.")
    elif thermo_params is None and st.session_state.get('temp_effect_results') is not None: 
         st.warning("Thermo analysis based on Kd not performed (check messages/data).")
    elif thermo_params and thermo_params.get('Analysis_Type') != 'Kd': 
        st.warning("Existing thermo parameters calculated differently. Reset if needed.")