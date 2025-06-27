# tabs/isotherm_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress
from scipy.optimize import curve_fit
import io
from utils import convert_df_to_csv
from models import langmuir_model, freundlich_model, temkin_model_nonlinear

def render():
    st.subheader("Adsorption Isotherm Analysis")
    iso_input = st.session_state.get('isotherm_input')
    calib_params = st.session_state.get('calibration_params')
    iso_results = st.session_state.get('isotherm_results')

    R_GAS_CONSTANT = 8.314

    if iso_input and calib_params:
        
        if iso_results is None:
            with st.spinner("Calculating Ce/qe for isotherms..."):
                results_list = []
                df_iso = iso_input['data'].copy()
                params_iso = iso_input['params']
                try:
                    if abs(calib_params['slope']) < 1e-9:
                         st.error("Error calculating Ce/qe: Calibration slope is zero or near zero.")
                         st.session_state['isotherm_results'] = None
                         return
                    m_adsorbant = params_iso['m']
                    volume = params_iso['V']
                    if m_adsorbant <= 0:
                        st.error(f"Invalid fixed adsorbent mass ({m_adsorbant}g). Mass must be positive.")
                        st.session_state['isotherm_results'] = None
                        return

                    for _, row in df_iso.iterrows():
                        c0 = row['Concentration']
                        abs_eq = row['Absorbance']
                        ce = (abs_eq - calib_params['intercept']) / calib_params['slope']
                        ce = max(0.0, ce) 

                        qe = (c0 - ce) * volume / m_adsorbant
                        qe = max(0.0, qe) 

                        results_list.append({
                            'C0': c0, 'Abs_Eq': abs_eq, 'Ce': ce, 'qe': qe,
                            'Masse_Adsorbant_g': m_adsorbant, 'Volume_L': volume,
                        })

                    if not results_list:
                         st.warning("No valid isotherm data points after Ce/qe calculation and mass check.")
                         st.session_state['isotherm_results'] = pd.DataFrame(columns=['C0', 'Abs_Eq', 'Ce', 'qe', 'Masse_Adsorbant_g', 'Volume_L'])
                    else:
                        st.session_state['isotherm_results'] = pd.DataFrame(results_list)
                        st.success("Ce/qe calculation for isotherms complete.")

                except ZeroDivisionError:
                     st.error("Error calculating Ce/qe: Division by zero detected (check calibration slope).")
                     st.session_state['isotherm_results'] = None
                except Exception as e:
                    st.error(f"Error calculating Ce/qe for isotherm: {e}")
                    st.session_state['isotherm_results'] = None
                iso_results = st.session_state.get('isotherm_results')


        if iso_results is not None and not iso_results.empty:
            st.markdown("##### Calculated Data (Ce vs qe)")
            st.dataframe(
                iso_results[['C0', 'Abs_Eq', 'Ce', 'qe']]
                .rename(columns={"C0": "C₀ (mg/L)", "Abs_Eq": "Absorbance", "Ce": "Ce (mg/L)", "qe": "qe (mg/g)"})
                .style.format("{:.4f}")
                         )
            csv_iso_res = convert_df_to_csv(iso_results)
            st.download_button("📥 DL Isotherm Data (Ce/qe)", csv_iso_res, "isotherm_results.csv", "text/csv", key="dl_iso_res_iso_tab")
            st.caption(f"Conditions: m={iso_input['params']['m']}g, V={iso_input['params']['V']}L")
            st.markdown("---")

            st.markdown("##### Experimental Adsorption Curve (qe vs Ce)")
            try:
                iso_results_sorted_exp = iso_results.sort_values(by='Ce')
                fig_exp_only = go.Figure()
                fig_exp_only.add_trace(go.Scatter(
                    x=iso_results_sorted_exp['Ce'], y=iso_results_sorted_exp['qe'],
                    mode='lines+markers',
                    name="Experimental data",
                    line=dict(color='blue'), marker=dict(size=8)
                ))
                fig_exp_only.update_layout(xaxis_title="Ce (mg/L)", yaxis_title="qe (mg/g)", template="simple_white", width=500, height=500)
                st.plotly_chart(fig_exp_only, use_container_width=False)

                fig_exp_styled = go.Figure()
                fig_exp_styled.add_trace(go.Scatter(
                    x=iso_results_sorted_exp['Ce'], y=iso_results_sorted_exp['qe'],
                    mode='markers+lines', marker=dict(symbol='square', color='black', size=10),
                    line=dict(color='red', width=3), name="Experimental data"
                ))
                fig_exp_styled.update_layout(
                    width=1000, height=800, plot_bgcolor='white', paper_bgcolor='white',
                    font=dict(family="Times New Roman", size=22, color="black"),
                    margin=dict(l=80, r=40, t=60, b=80),
                    xaxis=dict(title="Ce (mg/L)", linecolor='black', mirror=True, ticks='outside', showline=True, showgrid=False, zeroline=False),
                    yaxis=dict(title="qe (mg/g)", linecolor='black', mirror=True, ticks='outside', showline=True, showgrid=False, zeroline=False),
                    showlegend=False
                )
                exp_img_buffer = io.BytesIO()
                fig_exp_styled.write_image(exp_img_buffer, format="png", width=1000, height=800, scale=2)
                exp_img_buffer.seek(0)
                st.download_button(
                    label="📥 Download Figure (PNG)", data=exp_img_buffer,
                    file_name="experimental_curve_qe_Ce.png", mime="image/png", key="dl_iso_exp_fig"
                )
            except Exception as e_exp_plot:
                 st.warning(f"Error plotting experimental curve: {e_exp_plot}")
            st.markdown("---")

            # --- LINEAR MODEL FITTING ---
            st.markdown("##### Model Linearization")
            st.caption("Parameters (qm, KL, KF, n, KT, B₁) are determined from these linear regressions.")
            iso_filtered_lin_main = iso_results[(iso_results['Ce'] > 1e-9) & (iso_results['qe'] > 1e-9)].copy()

            if len(iso_filtered_lin_main) >= 2:
                # Langmuir Linearized 
                st.markdown("###### Linearized Langmuir (Ce/qe vs Ce)")
                iso_filtered_lang_lin = iso_filtered_lin_main.copy()
                if not iso_filtered_lang_lin.empty and len(iso_filtered_lang_lin) >=2 :
                    try:
                        
                        iso_filtered_lang_lin['Ce_div_qe'] = iso_filtered_lang_lin['Ce'] / iso_filtered_lang_lin['qe']
                        if iso_filtered_lang_lin['Ce'].nunique() < 2 or iso_filtered_lang_lin['Ce_div_qe'].nunique() < 2:
                            st.warning("Insufficient variation in Ce or Ce/qe for Langmuir regression.")
                            raise ValueError("Insufficient variation for Langmuir linregress (Ce/qe vs Ce)")
                        slope_L_lin, intercept_L_lin, r_val_L_lin, _, _ = linregress(iso_filtered_lang_lin['Ce'], iso_filtered_lang_lin['Ce_div_qe'])
                        r2_L_lin = r_val_L_lin**2
                        qm_L_lin = 1 / slope_L_lin if abs(slope_L_lin) > 1e-12 else np.nan
                        KL_L_lin = slope_L_lin / intercept_L_lin if abs(intercept_L_lin) > 1e-12 and abs(slope_L_lin) > 1e-12 else np.nan 
                        st.session_state['langmuir_params_lin'] = {'qm': qm_L_lin, 'KL': KL_L_lin, 'r_squared': r2_L_lin}

                        # Plotting
                        fig_L_lin = px.scatter(iso_filtered_lang_lin, x='Ce', y='Ce_div_qe', title=f"Ce/qe vs Ce (R²={r2_L_lin:.4f})", labels={'Ce': 'Ce (mg/L)', 'Ce_div_qe': 'Ce / qe (g/mg)'})
                        x_min_L_lin_plot, x_max_L_lin_plot = iso_filtered_lang_lin['Ce'].min(), iso_filtered_lang_lin['Ce'].max()
                        x_range_L_lin_plot = x_max_L_lin_plot - x_min_L_lin_plot if x_max_L_lin_plot > x_min_L_lin_plot else 1.0
                        x_line_L_lin = np.linspace(max(0.0, x_min_L_lin_plot - 0.1 * x_range_L_lin_plot), x_max_L_lin_plot + 0.1 * x_range_L_lin_plot, 100) 
                        y_line_L_lin = intercept_L_lin + slope_L_lin * x_line_L_lin
                        fig_L_lin.add_trace(go.Scatter(x=x_line_L_lin, y=y_line_L_lin, mode='lines', name="Linear Fit"))
                        fig_L_lin.update_layout(template="simple_white", width=600, height=500)
                        st.plotly_chart(fig_L_lin, use_container_width=False)
                        st.caption(f"Slope = {slope_L_lin:.4f} (1 / qm), Intercept = {intercept_L_lin:.4f} (1 / (qm·KL))")

                        try:
                            x_vals_lang_dl = np.array([iso_filtered_lang_lin['Ce'].min(), iso_filtered_lang_lin['Ce'].max()])
                            y_vals_lang_dl = intercept_L_lin + slope_L_lin * x_vals_lang_dl
                            fig_lang_Ce_div_qe = go.Figure()
                            fig_lang_Ce_div_qe.add_trace(go.Scatter(x=iso_filtered_lang_lin['Ce'],y=iso_filtered_lang_lin['Ce_div_qe'],mode='markers',marker=dict(symbol='square', color='black', size=10),name="Experimental data"))
                            fig_lang_Ce_div_qe.add_trace(go.Scatter(x=x_vals_lang_dl,y=y_vals_lang_dl,mode='lines',line=dict(color='red', width=3),name="Linear regression"))
                            fig_lang_Ce_div_qe.update_layout(width=1000,height=800,plot_bgcolor='white',paper_bgcolor='white',font=dict(family="Times New Roman", size=22, color="black"),margin=dict(l=80, r=40, t=60, b=80),xaxis=dict(title="Ce (mg/L)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),yaxis=dict(title="Ce / qe (g/mg)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),showlegend=False)
                            operator_L = "-" if intercept_L_lin < 0 else "+"
                            equation_text_L = f"y = {slope_L_lin:.4f}x {operator_L} {abs(intercept_L_lin):.4f}"
                            fig_lang_Ce_div_qe.add_annotation(xref="paper", yref="paper",x=0.05, y=0.95,text=f"{equation_text_L}<br>R² = {r2_L_lin:.4f}",showarrow=False,font=dict(size=20, color="black"),align="left")
                            img_buffer_lang_Ce_div_qe = io.BytesIO()
                            fig_lang_Ce_div_qe.write_image(img_buffer_lang_Ce_div_qe, format="png", width=1000, height=800, scale=2)
                            img_buffer_lang_Ce_div_qe.seek(0)
                            
                            st.download_button(label="📥 Download Figure (PNG)",data=img_buffer_lang_Ce_div_qe,file_name="langmuir_linear_Ce_div_qe.png",mime="image/png", key="dl_lang_lin_Ce_div_qe_iso_tab")
                        except Exception as e_dl_L: st.warning(f"Error exporting linearized Langmuir (Ce/qe vs Ce): {e_dl_L}")

                    except ValueError as ve:
                         if "Insufficient variation" not in str(ve): st.warning(f"Error in linearized Langmuir regression: {ve}")
                         st.session_state['langmuir_params_lin'] = None
                    except Exception as e_lin_L:
                        st.warning(f"Error creating linearized Langmuir plot: {e_lin_L}")
                        st.session_state['langmuir_params_lin'] = None
                else:
                    st.info("No valid data (Ce>0, qe>0) for linearized Langmuir plot.")
                st.markdown("---")

                # Freundlich Linearized 
                st.markdown("###### Linearized Freundlich (ln qe vs ln Ce)")
                iso_filtered_freund_lin = iso_filtered_lin_main.copy() 
                if not iso_filtered_freund_lin.empty and len(iso_filtered_freund_lin) >= 2:
                    try:
                        iso_filtered_freund_lin['ln_Ce'] = np.log(iso_filtered_freund_lin['Ce'])
                        iso_filtered_freund_lin['ln_qe'] = np.log(iso_filtered_freund_lin['qe'])

                        if iso_filtered_freund_lin['ln_Ce'].nunique() < 2 or iso_filtered_freund_lin['ln_qe'].nunique() < 2:
                            st.warning("Insufficient variation in ln(Ce) or ln(qe) for Freundlich regression.")
                            raise ValueError("Insufficient variation for Freundlich linregress")

                        # Linregress on ln(Ce) (x) vs ln(qe) (y)
                        slope_F_lin, intercept_F_lin, r_val_F_lin, _, _ = linregress(iso_filtered_freund_lin['ln_Ce'], iso_filtered_freund_lin['ln_qe'])
                        r2_F_lin = r_val_F_lin**2

                        # Calculate n and KF from slope and intercept
                        n_F_lin = 1 / slope_F_lin if abs(slope_F_lin) > 1e-12 else np.nan
                        KF_F_lin = np.exp(intercept_F_lin)
                        st.session_state['freundlich_params_lin'] = {'KF': KF_F_lin, 'n': n_F_lin, 'r_squared': r2_F_lin}

                        # Plotting
                        fig_F_lin = px.scatter(iso_filtered_freund_lin, x='ln_Ce', y='ln_qe', title=f"ln(qe) vs ln(Ce) (R²={r2_F_lin:.4f})", labels={'ln_Ce': 'ln(Ce)', 'ln_qe': 'ln(qe)'})
                        x_min_F_lin_plot, x_max_F_lin_plot = iso_filtered_freund_lin['ln_Ce'].min(), iso_filtered_freund_lin['ln_Ce'].max()
                        x_range_F_lin_plot = x_max_F_lin_plot - x_min_F_lin_plot if x_max_F_lin_plot > x_min_F_lin_plot else 1.0
                        x_line_F_lin = np.linspace(x_min_F_lin_plot - 0.1 * abs(x_range_F_lin_plot) - 0.01, x_max_F_lin_plot + 0.1 * abs(x_range_F_lin_plot) + 0.01, 100)
                        y_line_F_lin = intercept_F_lin + slope_F_lin * x_line_F_lin
                        fig_F_lin.add_trace(go.Scatter(x=x_line_F_lin, y=y_line_F_lin, mode='lines', name="Linear Fit"))
                        fig_F_lin.update_layout(template="simple_white", width=600, height=500)
                        st.plotly_chart(fig_F_lin, use_container_width=False)
                        st.caption(f"Slope = {slope_F_lin:.4f} (1/n), Intercept = {intercept_F_lin:.4f} (ln KF)\nKF = exp({intercept_F_lin:.4f}) = {KF_F_lin:.4f}")
                        try:
                            x_vals_freund_dl = np.array([iso_filtered_freund_lin['ln_Ce'].min(), iso_filtered_freund_lin['ln_Ce'].max()])
                            y_vals_freund_dl = intercept_F_lin + slope_F_lin * x_vals_freund_dl
                            fig_freund_lin = go.Figure()
                            fig_freund_lin.add_trace(go.Scatter(x=iso_filtered_freund_lin['ln_Ce'],y=iso_filtered_freund_lin['ln_qe'],mode='markers',marker=dict(symbol='square', color='black', size=10),name="Experimental data"))
                            fig_freund_lin.add_trace(go.Scatter(x=x_vals_freund_dl,y=y_vals_freund_dl,mode='lines',line=dict(color='red', width=3),name="Linear regression"))
                            fig_freund_lin.update_layout(width=1000,height=800,plot_bgcolor='white',paper_bgcolor='white',font=dict(family="Times New Roman", size=22, color="black"),margin=dict(l=80, r=40, t=60, b=80),xaxis=dict(title="ln(Ce)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),yaxis=dict(title="ln(qe)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),showlegend=False)
                            operator_F = "-" if intercept_F_lin < 0 else "+"
                            equation_text_F = f"y = {slope_F_lin:.4f}x {operator_F} {abs(intercept_F_lin):.4f}"
                            fig_freund_lin.add_annotation(xref="paper", yref="paper",x=0.05, y=0.95,text=f"{equation_text_F}<br>R² = {r2_F_lin:.4f}",showarrow=False,font=dict(size=20, color="black"),align="left")
                            freund_img_buffer = io.BytesIO()
                            fig_freund_lin.write_image(freund_img_buffer, format="png", width=1000, height=800, scale=2)
                            freund_img_buffer.seek(0)
                            st.download_button(label="📥 Download Figure (PNG)",data=freund_img_buffer,file_name="freundlich_linear.png",mime="image/png", key="dl_freund_lin_iso_tab")
                        except Exception as e_dl_F: st.warning(f"Error exporting linearized Freundlich: {e_dl_F}")
                    except ValueError as ve:
                         if "Insufficient variation" not in str(ve) and "No valid data for log" not in str(ve): st.warning(f"Error in linearized Freundlich regression: {ve}")
                         st.session_state['freundlich_params_lin'] = None
                    except Exception as e_lin_F:
                        st.warning(f"Error creating linearized Freundlich plot: {e_lin_F}")
                        st.session_state['freundlich_params_lin'] = None
                else:
                    st.info("No valid data (Ce>0, qe>0) for linearized Freundlich plot.")
                st.markdown("---")

                # Temkin Linearized 
                st.markdown("###### Linearized Temkin (qe vs ln Ce)")
                iso_filtered_temkin_lin = iso_results[(iso_results['Ce'] > 1e-9) & (iso_results['qe'] >= 0)].copy() 
                if not iso_filtered_temkin_lin.empty and len(iso_filtered_temkin_lin) >= 2:
                    try:
                        # The Temkin linearization 
                        iso_filtered_temkin_lin_plot_df = iso_filtered_temkin_lin.copy()
                        iso_filtered_temkin_lin_plot_df['ln_Ce'] = np.log(iso_filtered_temkin_lin_plot_df['Ce'])
                        if iso_filtered_temkin_lin_plot_df['ln_Ce'].nunique() < 2 or iso_filtered_temkin_lin_plot_df['qe'].nunique() < 2:
                            st.warning("Insufficient variation in ln(Ce) or qe for Temkin regression.")
                            raise ValueError("Insufficient variation for Temkin linregress")

                        slope_T_lin, intercept_T_lin, r_val_T_lin, _, _ = linregress(iso_filtered_temkin_lin_plot_df['ln_Ce'], iso_filtered_temkin_lin_plot_df['qe'])
                        r2_T_lin = r_val_T_lin**2

                        B1_T_lin = slope_T_lin
                        KT_T_lin = np.exp(intercept_T_lin / B1_T_lin) if abs(B1_T_lin) > 1e-9 else np.nan

                        st.session_state['temkin_params_lin'] = {'B1': B1_T_lin, 'KT': KT_T_lin, 'r_squared': r2_T_lin, 'slope': slope_T_lin, 'intercept': intercept_T_lin}

                        # Plotting
                        fig_T_lin = px.scatter(iso_filtered_temkin_lin_plot_df, x='ln_Ce', y='qe',
                                             title=f"qe vs ln(Ce) (R²={r2_T_lin:.4f})",
                                             labels={'ln_Ce': 'ln(Ce)', 'qe': 'qe (mg/g)'})
                        x_min_T_lin_plot, x_max_T_lin_plot = iso_filtered_temkin_lin_plot_df['ln_Ce'].min(), iso_filtered_temkin_lin_plot_df['ln_Ce'].max()
                        x_range_T_lin_plot = x_max_T_lin_plot - x_min_T_lin_plot if x_max_T_lin_plot > x_min_T_lin_plot else 1.0
                        x_line_T_lin = np.linspace(x_min_T_lin_plot - 0.1 * abs(x_range_T_lin_plot) - 0.01, x_max_T_lin_plot + 0.1 * abs(x_range_T_lin_plot) + 0.01, 100)
                        y_line_T_lin = intercept_T_lin + slope_T_lin * x_line_T_lin
                        fig_T_lin.add_trace(go.Scatter(x=x_line_T_lin, y=y_line_T_lin, mode='lines', name="Linear Fit"))
                        fig_T_lin.update_layout(template="simple_white", width=600, height=500)
                        st.plotly_chart(fig_T_lin, use_container_width=False)
                        st.caption(f"Slope = {slope_T_lin:.3f} (B₁), Intercept = {intercept_T_lin:.3f} (B₁ ln Kᴛ)\nKᴛ = {KT_T_lin:.3f} L/mg, B₁ = {B1_T_lin:.3f} mg/g (RT/bᴛ)")
                        if abs(B1_T_lin) < 1e-9:
                            st.warning("B₁ (slope/parameter) is close to zero, Kᴛ or bᴛ cannot be reliably calculated.")
                        try:
                            x_vals_temkin_dl_lin = np.linspace(iso_filtered_temkin_lin_plot_df['ln_Ce'].min(), iso_filtered_temkin_lin_plot_df['ln_Ce'].max(), 100)
                            y_vals_temkin_dl_lin = intercept_T_lin + slope_T_lin * x_vals_temkin_dl_lin
                            fig_temkin_lin_styled = go.Figure()
                            fig_temkin_lin_styled.add_trace(go.Scatter(x=iso_filtered_temkin_lin_plot_df['ln_Ce'],y=iso_filtered_temkin_lin_plot_df['qe'],mode='markers',marker=dict(symbol='square', color='black', size=10),name="Experimental data"))
                            fig_temkin_lin_styled.add_trace(go.Scatter(x=x_vals_temkin_dl_lin,y=y_vals_temkin_dl_lin,mode='lines',line=dict(color='red', width=3),name="Linear regression"))
                            fig_temkin_lin_styled.update_layout(width=1000,height=800,plot_bgcolor='white',paper_bgcolor='white',font=dict(family="Times New Roman", size=22, color="black"),margin=dict(l=80, r=40, t=60, b=80),xaxis=dict(title="ln(Ce)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),yaxis=dict(title="qe (mg/g)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),showlegend=False)
                            operator_T = "-" if intercept_T_lin < 0 else "+"
                            equation_text_T = f"y = {slope_T_lin:.4f}x {operator_T} {abs(intercept_T_lin):.4f}"
                            fig_temkin_lin_styled.add_annotation(xref="paper", yref="paper",x=0.05, y=0.95,text=f"{equation_text_T}<br>R² = {r2_T_lin:.4f}",showarrow=False,font=dict(size=20, color="black"),align="left")
                            temkin_lin_img_buffer = io.BytesIO()
                            fig_temkin_lin_styled.write_image(temkin_lin_img_buffer, format="png", width=1000, height=800, scale=2)
                            temkin_lin_img_buffer.seek(0)
                            st.download_button(label="📥 Download Figure (PNG)",data=temkin_lin_img_buffer,file_name="temkin_linear_styled.png",mime="image/png", key="dl_temkin_lin_iso_tab")
                        except Exception as e_dl_T_lin: st.warning(f"Error exporting Temkin linearized plot: {e_dl_T_lin}")

                    except ValueError as ve_T_lin:
                         if "Insufficient variation" not in str(ve_T_lin): st.warning(f"Error in Temkin linearized regression: {ve_T_lin}")
                         st.session_state['temkin_params_lin'] = None
                    except Exception as e_T_lin:
                        st.warning(f"Error creating Temkin linearized plot: {e_T_lin}")
                        st.session_state['temkin_params_lin'] = None
                else:
                    st.info("No valid data (Ce>0, qe>=0) for the linearized Temkin plot.")
                st.markdown("---")


                st.markdown("##### Parameters Derived from Linearized Models")
                params_lin_data_display = {'Model': [], 'Parameter': [], 'Value': [], 'R² (Linearized)': []}
                params_L_lin_state = st.session_state.get('langmuir_params_lin')
                params_F_lin_state = st.session_state.get('freundlich_params_lin')
                params_T_lin_state = st.session_state.get('temkin_params_lin')

                if params_L_lin_state and isinstance(params_L_lin_state, dict) and not np.isnan(params_L_lin_state.get('qm', np.nan)):
                    params_lin_data_display['Model'].extend(['Langmuir (Lin)', 'Langmuir (Lin)'])
                    params_lin_data_display['Parameter'].extend(['qm (mg/g)', 'KL (L/mg)'])
                    params_lin_data_display['Value'].extend([f"{params_L_lin_state['qm']:.4f}", f"{params_L_lin_state['KL']:.4f}"])
                    params_lin_data_display['R² (Linearized)'].extend([f"{params_L_lin_state['r_squared']:.4f}"] * 2)

                if params_F_lin_state and isinstance(params_F_lin_state, dict) and not np.isnan(params_F_lin_state.get('KF', np.nan)):
                    params_lin_data_display['Model'].extend(['Freundlich (Lin)', 'Freundlich (Lin)'])
                    params_lin_data_display['Parameter'].extend(['KF ((mg/g)(L/mg)¹/ⁿ)', 'n'])
                    params_lin_data_display['Value'].extend([f"{params_F_lin_state['KF']:.4f}", f"{params_F_lin_state['n']:.4f}"])
                    params_lin_data_display['R² (Linearized)'].extend([f"{params_F_lin_state['r_squared']:.4f}"] * 2)

                if params_T_lin_state and isinstance(params_T_lin_state, dict) and not np.isnan(params_T_lin_state.get('B1', np.nan)):
                    params_lin_data_display['Model'].extend(['Temkin (Lin)', 'Temkin (Lin)'])
                    params_lin_data_display['Parameter'].extend(['B₁ (RT/bᴛ) (mg/g)', 'Kᴛ (L/mg)'])
                    params_lin_data_display['Value'].extend([f"{params_T_lin_state['B1']:.3f}", f"{params_T_lin_state.get('KT', np.nan):.3f}"])
                    params_lin_data_display['R² (Linearized)'].extend([f"{params_T_lin_state['r_squared']:.4f}"] * 2)


                if params_lin_data_display['Model']:
                    params_lin_df_display = pd.DataFrame(params_lin_data_display)
                    st.dataframe(params_lin_df_display.set_index('Model'), use_container_width=True)
                    csv_lin_params = convert_df_to_csv(params_lin_df_display)
                    st.download_button(
                        label="📥 DL Linearized Model Parameters", 
                        data=csv_lin_params,
                        file_name="isotherm_params_linearized.csv",
                        mime="text/csv",
                        key="dl_iso_params_lin_table"
                    )
                else:
                    st.info("Parameters could not be calculated from linearized fits (check data, plots, and messages).")

            elif not iso_results.empty:
                 st.warning("Fewer than 2 data points with Ce > 0 and qe > 0. Cannot perform linearized fits.")

            st.markdown("---")

            # --- NON-LINEAR MODEL FITTING SECTION ---
            st.markdown("##### Non-Linear Model Fitting")
            st.caption("Parameters are determined directly from the qe vs Ce fit.")

            iso_filtered_nl = iso_results[(iso_results['Ce'] > 1e-9) & (iso_results['qe'] >= 0)].copy()

            if not iso_filtered_nl.empty and len(iso_filtered_nl) >= 2:
                Ce_data_nl = iso_filtered_nl['Ce'].values
                qe_data_nl = iso_filtered_nl['qe'].values
                if Ce_data_nl.size > 0:
                    min_ce_nl, max_ce_nl = Ce_data_nl.min(), Ce_data_nl.max()
                    if min_ce_nl == max_ce_nl:
                        Ce_line_for_plot_nl = np.linspace(min_ce_nl * 0.8, min_ce_nl * 1.2, 200)
                    else:
                        Ce_line_for_plot_nl = np.linspace(min_ce_nl * 0.9, max_ce_nl * 1.1, 200)
                    Ce_line_for_plot_nl = np.maximum(Ce_line_for_plot_nl, 1e-9) 
                else:
                    Ce_line_for_plot_nl = np.array([]) 

                fig_nl_fits = go.Figure()
                if Ce_data_nl.size > 0:
                    fig_nl_fits.add_trace(go.Scatter(
                        x=Ce_data_nl, y=qe_data_nl, mode='markers',
                        name="Experimental data",
                        marker=dict(color='black', symbol='diamond-open', size=10)
                    ))

                # Langmuir Non-Linear
                st.markdown("###### Non-Linear Langmuir")
                params_L_nl_current = st.session_state.get('langmuir_params_nl')
                if params_L_nl_current is None and Ce_data_nl.size >= 2:
                    with st.spinner("Non-linear fitting for Langmuir..."):
                        try:
                            qm_guess_nl = qe_data_nl.max() if len(qe_data_nl) > 0 else 1.0
                            KL_guess_nl = 0.1
                            popt_L_nl, _ = curve_fit(langmuir_model, Ce_data_nl, qe_data_nl, p0=[qm_guess_nl, KL_guess_nl], bounds=([0,0], [np.inf, np.inf]), maxfev=5000)
                            qm_nl, KL_nl = popt_L_nl
                            qe_pred_L_nl = langmuir_model(Ce_data_nl, qm_nl, KL_nl)
                            ss_res_L_nl = np.sum((qe_data_nl - qe_pred_L_nl)**2)
                            ss_tot_L_nl = np.sum((qe_data_nl - np.mean(qe_data_nl))**2)
                            r2_L_nl = 1 - (ss_res_L_nl / ss_tot_L_nl) if ss_tot_L_nl > 1e-9 else 0.0
                            st.session_state['langmuir_params_nl'] = {
                                'qm': qm_nl, 'KL': KL_nl, 'r_squared': r2_L_nl
                            }
                        except Exception as e_nl_L:
                            st.warning(f"Error during non-linear fitting of Langmuir: {e_nl_L}")
                            st.session_state['langmuir_params_nl'] = {'qm': np.nan, 'KL': np.nan, 'r_squared': np.nan}


                params_L_nl_to_display = st.session_state.get('langmuir_params_nl')
                if params_L_nl_to_display and isinstance(params_L_nl_to_display, dict) and not np.isnan(params_L_nl_to_display.get('qm', np.nan)):
                    st.caption(f"Parameters: qm = {params_L_nl_to_display['qm']:.3f} mg/g, KL = {params_L_nl_to_display['KL']:.3f} L/mg, R² = {params_L_nl_to_display['r_squared']:.4f}")
                    if Ce_line_for_plot_nl.size > 0:
                        qe_langmuir_fit_on_plot = langmuir_model(Ce_line_for_plot_nl, params_L_nl_to_display['qm'], params_L_nl_to_display['KL'])
                        fig_nl_fits.add_trace(go.Scatter(x=Ce_line_for_plot_nl, y=qe_langmuir_fit_on_plot, mode='lines', name="Langmuir Fit"))




                # Freundlich Non-Linear
                st.markdown("###### Non-Linear Freundlich")
                params_F_nl_current = st.session_state.get('freundlich_params_nl')
                if params_F_nl_current is None and Ce_data_nl.size >= 2:
                    with st.spinner("Non-linear fitting for Freundlich..."):
                        try:
                            KF_guess_nl = np.median(qe_data_nl) / (np.median(Ce_data_nl)**0.5) if len(Ce_data_nl) > 0 and np.median(Ce_data_nl) > 0 else 1.0
                            n_inv_guess_nl = 0.5
                            popt_F_nl, _ = curve_fit(freundlich_model, Ce_data_nl, qe_data_nl, p0=[KF_guess_nl, n_inv_guess_nl], bounds=([0,0], [np.inf, np.inf]), maxfev=5000)
                            KF_nl, n_inv_nl = popt_F_nl
                            n_nl = 1 / n_inv_nl if abs(n_inv_nl) > 1e-9 else np.nan
                            qe_pred_F_nl = freundlich_model(Ce_data_nl, KF_nl, n_inv_nl)
                            ss_res_F_nl = np.sum((qe_data_nl - qe_pred_F_nl)**2)
                            ss_tot_F_nl = np.sum((qe_data_nl - np.mean(qe_data_nl))**2)
                            r2_F_nl = 1 - (ss_res_F_nl / ss_tot_F_nl) if ss_tot_F_nl > 1e-9 else 0.0
                            st.session_state['freundlich_params_nl'] = {
                                'KF': KF_nl, 'n': n_nl, 'n_inv': n_inv_nl, 'r_squared': r2_F_nl
                            }
                        except Exception as e_nl_F:
                            st.warning(f"Error during non-linear fitting of Freundlich: {e_nl_F}")
                            st.session_state['freundlich_params_nl'] = {'KF': np.nan, 'n': np.nan, 'n_inv': np.nan, 'r_squared': np.nan}

                params_F_nl_to_display = st.session_state.get('freundlich_params_nl')
                if params_F_nl_to_display and isinstance(params_F_nl_to_display, dict) and not np.isnan(params_F_nl_to_display.get('KF', np.nan)):
                    st.caption(f"Parameters: KF = {params_F_nl_to_display['KF']:.3f} (mg/g)(L/mg)¹/ⁿ, n = {params_F_nl_to_display['n']:.3f}, R² = {params_F_nl_to_display['r_squared']:.4f}")
                    if Ce_line_for_plot_nl.size > 0:
                        qe_freundlich_fit_on_plot = freundlich_model(Ce_line_for_plot_nl, params_F_nl_to_display['KF'], params_F_nl_to_display['n_inv'])
                        fig_nl_fits.add_trace(go.Scatter(x=Ce_line_for_plot_nl, y=qe_freundlich_fit_on_plot, mode='lines', name="Freundlich Fit"))




                # Temkin Non-Linear
                st.markdown("###### Non-Linear Temkin")
                _params_T_nl_for_widget_default = st.session_state.get('temkin_params_nl')
                _default_T_K_for_widget_value = 298.15
                if _params_T_nl_for_widget_default and isinstance(_params_T_nl_for_widget_default, dict):
                    _default_T_K_for_widget_value = _params_T_nl_for_widget_default.get('T_K_used', 298.15)

                temp_K_for_bT_nl = st.number_input(
                    "Temperature for bᴛ calculation (K)",
                    min_value=0.1,
                    value=_default_T_K_for_widget_value,
                    format="%.2f",
                    key="temp_K_bT_nl_temkin_isotherm"
                )

                temkin_params_nl_in_state_before_calc = st.session_state.get('temkin_params_nl')
                recalc_temkin_nl = False
                if temkin_params_nl_in_state_before_calc is None and Ce_data_nl.size >=2: recalc_temkin_nl = True
                elif isinstance(temkin_params_nl_in_state_before_calc, dict) and temkin_params_nl_in_state_before_calc.get('T_K_used') != temp_K_for_bT_nl and Ce_data_nl.size >=2 : recalc_temkin_nl = True

                if recalc_temkin_nl and Ce_data_nl.size >= 2:
                    with st.spinner("Non-linear fitting for Temkin..."):
                        try:
                            B1_guess_nl_temkin = st.session_state.get('temkin_params_lin', {}).get('B1', 10.0)
                            KT_guess_nl_temkin = st.session_state.get('temkin_params_lin', {}).get('KT', 0.1)
                            if not np.isfinite(B1_guess_nl_temkin) or abs(B1_guess_nl_temkin) < 1e-6:
                                 B1_guess_nl_temkin = 1.0
                            if not np.isfinite(KT_guess_nl_temkin) or KT_guess_nl_temkin <= 1e-9:
                                 KT_guess_nl_temkin = 0.1


                            popt_T_nl, _ = curve_fit(temkin_model_nonlinear, Ce_data_nl, qe_data_nl,
                                                     p0=[B1_guess_nl_temkin, KT_guess_nl_temkin],
                                                     bounds=([-np.inf, 1e-9], [np.inf, np.inf]), 
                                                     maxfev=5000)
                            B1_nl, KT_nl = popt_T_nl
                            qe_pred_T_nl = temkin_model_nonlinear(Ce_data_nl, B1_nl, KT_nl)
                            ss_res_T_nl = np.sum((qe_data_nl - qe_pred_T_nl)**2)
                            ss_tot_T_nl = np.sum((qe_data_nl - np.mean(qe_data_nl))**2)
                            r2_T_nl = 1 - (ss_res_T_nl / ss_tot_T_nl) if ss_tot_T_nl > 1e-9 else 0.0
                            bT_nl_calc = (R_GAS_CONSTANT * temp_K_for_bT_nl) / B1_nl if abs(B1_nl) > 1e-9 else np.nan
                            st.session_state['temkin_params_nl'] = {
                                'B1': B1_nl, 'KT': KT_nl, 'bT': bT_nl_calc, 'r_squared': r2_T_nl,
                                'T_K_used': temp_K_for_bT_nl
                                }
                        except Exception as e_nl_T:
                            st.warning(f"Error during non-linear fitting of Temkin: {e_nl_T}")
                            st.session_state['temkin_params_nl'] = {'B1': np.nan, 'KT': np.nan, 'bT': np.nan, 'r_squared': np.nan, 'T_K_used': temp_K_for_bT_nl}


                params_T_nl_for_display = st.session_state.get('temkin_params_nl')
                if params_T_nl_for_display and isinstance(params_T_nl_for_display, dict) and not np.isnan(params_T_nl_for_display.get('B1',np.nan)):
                    st.caption(f"Parameters: B₁ = {params_T_nl_for_display['B1']:.3f} mg/g (RT/bᴛ), Kᴛ = {params_T_nl_for_display['KT']:.3f} L/mg, R² = {params_T_nl_for_display['r_squared']:.4f}")
                    if not np.isnan(params_T_nl_for_display.get('bT', np.nan)):
                        st.caption(f"bᴛ (calculated) = {params_T_nl_for_display['bT']:.2f} J/mol" + f" (at T={params_T_nl_for_display.get('T_K_used', 'N/A'):.2f}K)")
                    if abs(params_T_nl_for_display['B1']) < 1e-9:
                        st.warning("B₁ (slope/parameter) is close to zero, Kᴛ or bᴛ cannot be reliably calculated.")

                    if Ce_line_for_plot_nl.size > 0:
                        qe_temkin_fit_on_plot = temkin_model_nonlinear(Ce_line_for_plot_nl, params_T_nl_for_display['B1'], params_T_nl_for_display['KT'])
                        fig_nl_fits.add_trace(go.Scatter(x=Ce_line_for_plot_nl, y=qe_temkin_fit_on_plot, mode='lines', name="Temkin Fit"))


                if Ce_data_nl.size > 0 or Ce_line_for_plot_nl.size > 0 : 
                    fig_nl_fits.update_layout(
                        title="Non-Linear Fits and Experimental Data",
                        xaxis_title="Ce (mg/L)", yaxis_title="qe (mg/g)",
                        template="simple_white", legend_title_text="Models", width=600, height=500
                    )
                    st.plotly_chart(fig_nl_fits, use_container_width=False)
                    try:
                        nl_fits_img_buffer_all = io.BytesIO()
                        fig_nl_fits.write_image(nl_fits_img_buffer_all, format="png", width=1000, height=800, scale=2)
                        nl_fits_img_buffer_all.seek(0)
                        st.download_button(
                            label="📥 Download Figure (PNG)",
                            data=nl_fits_img_buffer_all,
                            file_name="isotherm_all_nl_fits_plot.png",
                            mime="image/png",
                            key="dl_iso_all_nl_fits_plot"
                        )
                    except Exception as e_nl_dl_all:
                        st.warning(f"Error exporting all non-linear fits plot: {e_nl_dl_all}")

            else:
                 st.info("No valid data (Ce>0, qe>=0) for non-linear fitting.")

            st.markdown("---")
            st.markdown("##### Derived Parameters from Non-Linear Models")
            params_nl_data_disp = {'Model': [], 'Parameter': [], 'Value': [], 'R² (NL)': [], 'Add. Info': []}

            params_L_nl_final = st.session_state.get('langmuir_params_nl', {})
            params_F_nl_final = st.session_state.get('freundlich_params_nl', {})
            params_T_nl_final = st.session_state.get('temkin_params_nl', {})

            if params_L_nl_final and isinstance(params_L_nl_final, dict) and not np.isnan(params_L_nl_final.get('qm', np.nan)):
                params_nl_data_disp['Model'].extend(['Langmuir (NL)', 'Langmuir (NL)'])
                params_nl_data_disp['Parameter'].extend(['qm (mg/g)', 'KL (L/mg)'])
                params_nl_data_disp['Value'].extend([f"{params_L_nl_final['qm']:.4f}", f"{params_L_nl_final['KL']:.4f}"])
                params_nl_data_disp['R² (NL)'].extend([f"{params_L_nl_final['r_squared']:.4f}"] * 2)
                params_nl_data_disp['Add. Info'].extend(["", ""])

            if params_F_nl_final and isinstance(params_F_nl_final, dict) and not np.isnan(params_F_nl_final.get('KF', np.nan)):
                params_nl_data_disp['Model'].extend(['Freundlich (NL)', 'Freundlich (NL)'])
                params_nl_data_disp['Parameter'].extend(['KF ((mg/g)(L/mg)¹/ⁿ)', 'n'])
                params_nl_data_disp['Value'].extend([f"{params_F_nl_final['KF']:.4f}", f"{params_F_nl_final['n']:.4f}"])
                params_nl_data_disp['R² (NL)'].extend([f"{params_F_nl_final['r_squared']:.4f}"] * 2)
                params_nl_data_disp['Add. Info'].extend(["", ""])

            if params_T_nl_final and isinstance(params_T_nl_final, dict) and not np.isnan(params_T_nl_final.get('B1', np.nan)):
                params_nl_data_disp['Model'].extend(['Temkin (NL)', 'Temkin (NL)', 'Temkin (NL)'])
                params_nl_data_disp['Parameter'].extend(['B₁ (RT/bᴛ) (mg/g)', 'Kᴛ (L/mg)', 'bᴛ (J/mol)'])
                bT_val_str_disp = f"{params_T_nl_final['bT']:.2f}" if not np.isnan(params_T_nl_final.get('bT', np.nan)) else "N/A"
                T_K_used_str_disp = f"(T={params_T_nl_final.get('T_K_used', 'N/A'):.2f}K)"
                params_nl_data_disp['Value'].extend([f"{params_T_nl_final['B1']:.3f}", f"{params_T_nl_final['KT']:.3f}", bT_val_str_disp])
                params_nl_data_disp['R² (NL)'].extend([f"{params_T_nl_final['r_squared']:.4f}"] * 3)
                params_nl_data_disp['Add. Info'].extend(["", "", T_K_used_str_disp])


            if params_nl_data_disp['Model']:
                params_nl_df_disp = pd.DataFrame(params_nl_data_disp)
                st.dataframe(params_nl_df_disp.set_index('Model'), use_container_width=True)
                csv_nl_params = convert_df_to_csv(params_nl_df_disp)
                st.download_button(
                    label="📥 Download Data (Non-Linearized Params.)", 
                    data=csv_nl_params,
                    file_name="isotherm_params_nonlinear.csv",
                    mime="text/csv",
                    key="dl_iso_params_nl_table"
                )
            else:
                st.info("Non-linear parameters not calculated or calculation error.")
            st.markdown("---")
            st.markdown("### Model Comparison Summary")
                
            # Fetch all calculated parameters from session state
            params_L_lin = st.session_state.get('langmuir_params_lin') or {}
            params_F_lin = st.session_state.get('freundlich_params_lin') or {}
            params_T_lin = st.session_state.get('temkin_params_lin') or {}
            params_L_nl = st.session_state.get('langmuir_params_nl') or {}
            params_F_nl = st.session_state.get('freundlich_params_nl') or {}
            params_T_nl = st.session_state.get('temkin_params_nl') or {}

            summary_data = {
                    "Model": [
                        "Langmuir (Linear)", "Langmuir (Non-Linear)", 
                        "Freundlich (Linear)", "Freundlich (Non-Linear)",
                        "Temkin (Linear)", "Temkin (Non-Linear)"
                    ],
                    "R²": [
                        params_L_lin.get('r_squared'), params_L_nl.get('r_squared'),
                        params_F_lin.get('r_squared'), params_F_nl.get('r_squared'),
                        params_T_lin.get('r_squared'), params_T_nl.get('r_squared')
                    ],
                    "qm (mg/g)": [
                        params_L_lin.get('qm'), params_L_nl.get('qm'),
                        None, None, None, None # N/A for other models
                    ],
                    "KL (L/mg)": [
                        params_L_lin.get('KL'), params_L_nl.get('KL'),
                        None, None, None, None
                    ],
                    "KF ((mg/g)(L/mg)¹/ⁿ)": [
                        None, None,
                        params_F_lin.get('KF'), params_F_nl.get('KF'),
                        None, None
                    ],
                    "n (Freundlich)": [
                        None, None,
                        params_F_lin.get('n'), params_F_nl.get('n'),
                        None, None
                    ],
                    "B₁ (Temkin)": [
                        None, None, None, None,
                        params_T_lin.get('B1'), params_T_nl.get('B1')
                    ],
                    "Kᴛ (L/mg)": [
                        None, None, None, None,
                        params_T_lin.get('KT'), params_T_nl.get('KT')
                    ]
                }
            summary_df = pd.DataFrame(summary_data)
            summary_df.dropna(subset=['R²'], inplace=True)

            if not summary_df.empty:
                    st.dataframe(
                        summary_df.set_index('Model').style.format("{:.4f}", na_rep="—").highlight_max(subset="R²", color='lightgreen'),
                        use_container_width=True
                    )
            else:
                    st.info("No models were calculated for comparison.")

        elif iso_results is not None and iso_results.empty:
             st.warning("Ce/qe calculation produced no valid results (check calibration and input data).")

    elif not calib_params:
        st.warning("Please provide valid calibration data and calculate parameters first.")
    else:
        st.info("Please enter data for the isotherm study in the sidebar.")