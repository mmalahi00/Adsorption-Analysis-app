# tabs/kinetic_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress
from scipy.optimize import curve_fit
import io
from utils import convert_df_to_csv
from models import pfo_model, pso_model

def render():
    st.header("Adsorption Kinetic Analysis")
    kinetic_input = st.session_state.get('kinetic_input')
    calib_params = st.session_state.get('calibration_params')
    kinetic_results = st.session_state.get('kinetic_results_df')
    
    pfo_params_nl = st.session_state.get('pfo_params_nonlinear')
    pso_params_nl = st.session_state.get('pso_params_nonlinear') 
    ipd_params_list = st.session_state.get('ipd_params_list', [])

    if kinetic_input and calib_params:
        if kinetic_results is None: 
            with st.spinner("Calculating qt for kinetics..."):
                df_kin = kinetic_input['data'].copy()
                params_kin = kinetic_input['params']
                C0_k, V_k, m_k = params_kin['C0'], params_kin['V'], params_kin['m']

                if 'Absorbance' in df_kin.columns and m_k > 0 and V_k > 0:
                    try:
                        if abs(calib_params['slope']) < 1e-9:
                            st.error("Error calculating Ct/qt: Calibration slope is zero or near zero.")
                            st.session_state['kinetic_results_df'] = None
                            return 
                        
                        df_kin['Ct'] = (df_kin['Absorbance'] - calib_params['intercept']) / calib_params['slope']
                        df_kin['Ct'] = df_kin['Ct'].clip(lower=0)
                        df_kin['qt'] = (C0_k - df_kin['Ct']) * V_k / m_k
                        df_kin['qt'] = df_kin['qt'].clip(lower=0)
                        df_kin['sqrt_t'] = np.sqrt(df_kin['Time'])

                        st.session_state['kinetic_results_df'] = df_kin[['Time', 'Absorbance', 'Ct', 'qt', 'sqrt_t']].copy() 
                        st.success("qt calculation for kinetics complete.")
                    except ZeroDivisionError:
                        if 'kinetic_results_df' not in st.session_state or st.session_state.get('kinetic_results_df') is not None:
                            st.error("Error calculating Ct/qt: Division by zero detected (calibration slope zero?).")
                        st.session_state['kinetic_results_df'] = None
                    except Exception as e_qt:
                        st.error(f"Error calculating qt for kinetics: {e_qt}")
                        st.session_state['kinetic_results_df'] = None
                elif not ('Absorbance' in df_kin.columns):
                     st.error("'Absorbance' column missing in kinetic data.")
                     st.session_state['kinetic_results_df'] = None
                else:
                     st.error("Adsorbent mass and volume must be positive for qt calculation.")
                     st.session_state['kinetic_results_df'] = None
                
                kinetic_results = st.session_state.get('kinetic_results_df') 

        if kinetic_results is not None and not kinetic_results.empty:
            st.subheader("Calculated Data (qt vs t)")
            st.dataframe(kinetic_results.style.format(precision=4))
            csv_kin_res = convert_df_to_csv(kinetic_results)
            st.download_button("📥 DL Kinetic Data (qt)", csv_kin_res, "kinetic_results.csv", "text/csv", key='dl_kin_res_kin_tab_nl_top')
            st.caption(f"Conditions: C0={kinetic_input['params']['C0']}mg/L, m={kinetic_input['params']['m']}g, V={kinetic_input['params']['V']}L")
            st.markdown("---")

            st.subheader("1. Non-Linear Models (qt vs t)") 

            fig_qt_vs_t_combined = go.Figure()
            fig_qt_vs_t_combined.add_trace(go.Scatter(
                x=kinetic_results['Time'], 
                y=kinetic_results['qt'], 
                mode='markers', 
                name="Exp. Data",
                marker=dict(color='black', symbol='diamond-open', size=10)
            ))
            
            if len(kinetic_results) >= 3:
                t_data_kin = kinetic_results['Time'].values
                qt_data_kin = kinetic_results['qt'].values
                qe_exp_kin = qt_data_kin[-1] if len(qt_data_kin) > 0 else np.nan

                if pfo_params_nl is None: 
                     with st.spinner("Calculating PFO parameters (non-linear)..."):
                         try:
                             p0_PFO_nl = [qe_exp_kin if not np.isnan(qe_exp_kin) and qe_exp_kin > 1e-6 else 1.0, 0.01]
                             params_PFO_nl_fit, _ = curve_fit(pfo_model, t_data_kin, qt_data_kin, p0=p0_PFO_nl, maxfev=5000, bounds=([0, 0], [np.inf, np.inf]))
                             qe_PFO_nl_val, k1_PFO_nl_val = params_PFO_nl_fit
                             qt_pred_PFO_nl = pfo_model(t_data_kin, qe_PFO_nl_val, k1_PFO_nl_val)
                             ss_res_PFO_nl = np.sum((qt_data_kin - qt_pred_PFO_nl)**2)
                             ss_tot_PFO_nl = np.sum((qt_data_kin - np.mean(qt_data_kin))**2)
                             r2_PFO_nl_val = 1 - (ss_res_PFO_nl / ss_tot_PFO_nl) if ss_tot_PFO_nl > 1e-9 else 0.0
                             st.session_state['pfo_params_nonlinear'] = {'qe_PFO_nl': qe_PFO_nl_val, 'k1_nl': k1_PFO_nl_val, 'R2_PFO_nl': r2_PFO_nl_val}
                             pfo_params_nl = st.session_state['pfo_params_nonlinear'] 
                         except Exception as e_pfo_nl_fit:
                             st.warning(f"PFO parameter calculation (non-linear) failed: {e_pfo_nl_fit}")
                             st.session_state['pfo_params_nonlinear'] = {'qe_PFO_nl': np.nan, 'k1_nl': np.nan, 'R2_PFO_nl': np.nan}
                             pfo_params_nl = st.session_state.get('pfo_params_nonlinear')
                             pfo_params_nl = st.session_state['pfo_params_nonlinear']
                
                if pfo_params_nl and not np.isnan(pfo_params_nl.get('qe_PFO_nl', np.nan)):
                    t_line_kin_nl_plot = np.linspace(0, t_data_kin.max()*1.1 if t_data_kin.size > 0 else 10, 200)
                    qt_pfo_fit_on_plot = pfo_model(t_line_kin_nl_plot, pfo_params_nl['qe_PFO_nl'], pfo_params_nl['k1_nl'])
                    fig_qt_vs_t_combined.add_trace(go.Scatter(x=t_line_kin_nl_plot, y=qt_pfo_fit_on_plot, mode='lines', name="PFO Fit"))

                # PSO Non-Linear Fit 
                pso_params_nl = st.session_state.get('pso_params_nonlinear') 
                if pso_params_nl is None:
                     with st.spinner("Calculating PSO parameters (non-linear)..."):
                         try:
                             qe_guess_pso_nl = qe_exp_kin if not np.isnan(qe_exp_kin) and qe_exp_kin > 1e-6 else 1.0
                             k2_guess_nl = 0.01 / qe_guess_pso_nl if qe_guess_pso_nl > 1e-6 else 0.01
                             p0_PSO_nl = [qe_guess_pso_nl, k2_guess_nl]
                             params_PSO_nl_fit, _ = curve_fit(pso_model, t_data_kin, qt_data_kin, p0=p0_PSO_nl, maxfev=5000, bounds=([0, 0], [np.inf, np.inf]))
                             qe_PSO_nl_val, k2_PSO_nl_val = params_PSO_nl_fit
                             qt_pred_PSO_nl = pso_model(t_data_kin, qe_PSO_nl_val, k2_PSO_nl_val)
                             ss_res_PSO_nl = np.sum((qt_data_kin - qt_pred_PSO_nl)**2)
                             ss_tot_PSO_nl = np.sum((qt_data_kin - np.mean(qt_data_kin))**2)
                             r2_PSO_nl_val = 1 - (ss_res_PSO_nl / ss_tot_PSO_nl) if ss_tot_PSO_nl > 1e-9 else 0.0
                             st.session_state['pso_params_nonlinear'] = {'qe_PSO_nl': qe_PSO_nl_val, 'k2_nl': k2_PSO_nl_val, 'R2_PSO_nl': r2_PSO_nl_val}
                             pso_params_nl = st.session_state['pso_params_nonlinear'] 
                         except Exception as e_pso_nl_fit:
                             st.warning(f"PSO parameter calculation (non-linear) failed: {e_pso_nl_fit}")
                             st.session_state['pso_params_nonlinear'] = {'qe_PSO_nl': np.nan, 'k2_nl': np.nan, 'R2_PSO_nl': np.nan}
                             pso_params_nl = st.session_state['pso_params_nonlinear']
                
                if pso_params_nl and not np.isnan(pso_params_nl.get('qe_PSO_nl', np.nan)):
                    t_line_kin_nl_plot = np.linspace(0, t_data_kin.max()*1.1 if t_data_kin.size > 0 else 10, 200)
                    qt_pso_fit_on_plot = pso_model(t_line_kin_nl_plot, pso_params_nl['qe_PSO_nl'], pso_params_nl['k2_nl'])
                    fig_qt_vs_t_combined.add_trace(go.Scatter(x=t_line_kin_nl_plot, y=qt_pso_fit_on_plot, mode='lines', name="PSO Fit"))

            fig_qt_vs_t_combined.update_layout(
                title="Evolution of Adsorption Over Time", 
                xaxis_title="Time (min)", 
                yaxis_title='qt (mg/g)',
                template="simple_white",
                legend_title_text="Data/Models"
            )
            st.plotly_chart(fig_qt_vs_t_combined, use_container_width=True)
            
            try:
                qt_combined_img_buffer = io.BytesIO()
                fig_qt_vs_t_combined.write_image(qt_combined_img_buffer, format="png", width=1000, height=800, scale=2)
                qt_combined_img_buffer.seek(0)
                st.download_button(
                    label="📥 Download Figure (PNG)",
                    data=qt_combined_img_buffer,
                    file_name="qt_vs_time.png",
                    mime="image/png",
                    key="dl_qt_vs_t_combined_fig_kin_tab"
                )
            except Exception as e_export_combined:
                st.warning(f"Error exporting combined kinetic plot: {e_export_combined}")
            
            # Display Non-Linear Kinetic Parameters Table
            st.markdown("Non-Linear Kinetic Parameters")
            
            kin_nl_params_data_disp = {'Model': [], 'Parameter': [], 'Value': [], 'R² (Non-Linear)': []}
            if pfo_params_nl and not np.isnan(pfo_params_nl.get('qe_PFO_nl', np.nan)):
                kin_nl_params_data_disp['Model'].extend(['PFO (NL)', 'PFO (NL)'])
                kin_nl_params_data_disp['Parameter'].extend(['qe (mg/g)', 'k1 (min⁻¹)'])
                kin_nl_params_data_disp['Value'].extend([f"{pfo_params_nl['qe_PFO_nl']:.3f}", f"{pfo_params_nl['k1_nl']:.4f}"])
                kin_nl_params_data_disp['R² (Non-Linear)'].extend([f"{pfo_params_nl['R2_PFO_nl']:.4f}"] * 2)
            
            if pso_params_nl and not np.isnan(pso_params_nl.get('qe_PSO_nl', np.nan)):
                kin_nl_params_data_disp['Model'].extend(['PSO (NL)', 'PSO (NL)'])
                kin_nl_params_data_disp['Parameter'].extend(['qe (mg/g)', 'k2 (g·mg⁻¹·min⁻¹)'])
                kin_nl_params_data_disp['Value'].extend([f"{pso_params_nl['qe_PSO_nl']:.3f}", f"{pso_params_nl['k2_nl']:.4f}"])
                kin_nl_params_data_disp['R² (Non-Linear)'].extend([f"{pso_params_nl['R2_PSO_nl']:.4f}"] * 2)

            if kin_nl_params_data_disp['Model']:
                kin_nl_params_df_disp = pd.DataFrame(kin_nl_params_data_disp)
                st.dataframe(kin_nl_params_df_disp.set_index('Model'), use_container_width=True)
            else:
                st.info("Non-linear kinetic parameters not calculated or calculation error.")
            st.markdown("---")


            # --- LINEARIZED MODELS ---
            st.subheader("2. Analysis of Linearized Models")
            if len(kinetic_results) >= 3:
                st.markdown("###### Linearized PFO: ln(qe-qt) vs t")
                if pfo_params_nl and not np.isnan(pfo_params_nl.get('qe_PFO_nl', np.nan)):
                    qe_calc_pfo_for_lin = pfo_params_nl['qe_PFO_nl']
                    df_pfo_lin = kinetic_results[(kinetic_results['qt'] < qe_calc_pfo_for_lin - 1e-9) & (kinetic_results['qt'] >= 0) & (kinetic_results['Time'] > 1e-9)].copy()
                    if len(df_pfo_lin) >= 2:
                        try:
                            df_pfo_lin['ln_qe_qt'] = np.log(qe_calc_pfo_for_lin - df_pfo_lin['qt'])
                            t_pfo_lin, y_pfo_lin = df_pfo_lin['Time'], df_pfo_lin['ln_qe_qt']
                            if t_pfo_lin.nunique() < 2 or y_pfo_lin.nunique() < 2 :
                                 st.warning("Insufficient variation for linearized PFO regression.")
                                 raise ValueError("Insufficient variation for PFO linregress")
                            slope_pfo_lin, intercept_pfo_lin, r_val_pfo, _, _ = linregress(t_pfo_lin, y_pfo_lin)
                            r2_pfo_lin = r_val_pfo**2
                            k1_lin_val = -slope_pfo_lin

                            fig_pfo_lin_plot = px.scatter(df_pfo_lin, x='Time', y='ln_qe_qt', title=f"Linearized PFO (R²={r2_pfo_lin:.4f})", labels={'Time': "Time (min)", 'ln_qe_qt': 'ln(qe - qt)'})
                            t_min_plot_pfo_lin, t_max_plot_pfo_lin = t_pfo_lin.min(), t_pfo_lin.max(); t_range_plot_pfo_lin = max(1.0, t_max_plot_pfo_lin - t_min_plot_pfo_lin)
                            t_line_for_pfo_lin = np.linspace(t_min_plot_pfo_lin - 0.05 * t_range_plot_pfo_lin, t_max_plot_pfo_lin + 0.05 * t_range_plot_pfo_lin, 50)
                            t_line_for_pfo_lin = np.maximum(0, t_line_for_pfo_lin) 
                            y_line_for_pfo_lin = intercept_pfo_lin + slope_pfo_lin * t_line_for_pfo_lin
                            fig_pfo_lin_plot.add_trace(go.Scatter(x=t_line_for_pfo_lin, y=y_line_for_pfo_lin, mode='lines', name="Linear Fit"))
                            fig_pfo_lin_plot.update_layout(template="simple_white")
                            st.plotly_chart(fig_pfo_lin_plot, use_container_width=True)
                            st.caption(f"Slope = {slope_pfo_lin:.4f} (-k1), Intercept = {intercept_pfo_lin:.4f} (ln qe)\nR² = {r2_pfo_lin:.4f}, k1_lin = {k1_lin_val:.4f} min⁻¹")
                            try:
                                x_vals_pfo_dl_lin = np.linspace(t_pfo_lin.min(), t_pfo_lin.max(), 100)
                                y_vals_pfo_dl_lin = intercept_pfo_lin + slope_pfo_lin * x_vals_pfo_dl_lin
                                fig_pfo_styled_lin = go.Figure() 
                                fig_pfo_styled_lin.add_trace(go.Scatter(x=t_pfo_lin, y=y_pfo_lin, mode='markers', marker=dict(symbol='square', color='black', size=10), name="Experimental data"))
                                fig_pfo_styled_lin.add_trace(go.Scatter(x=x_vals_pfo_dl_lin, y=y_vals_pfo_dl_lin, mode='lines', line=dict(color='red', width=3), name="Linear regression"))
                                fig_pfo_styled_lin.update_layout(width=1000,height=800,plot_bgcolor='white',paper_bgcolor='white',font=dict(family="Times New Roman", size=22, color="black"),margin=dict(l=80, r=40, t=60, b=80),xaxis=dict(title="Time (min)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),yaxis=dict(title="ln(qe - qt)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),showlegend=False)
                                operator_pfo = "-" if intercept_pfo_lin < 0 else "+"
                                equation_text_pfo = f"y = {slope_pfo_lin:.4f}x {operator_pfo} {abs(intercept_pfo_lin):.4f}"
                                fig_pfo_styled_lin.add_annotation(xref="paper", yref="paper",x=0.05, y=0.95,text=f"{equation_text_pfo}<br>R² = {r2_pfo_lin:.4f}",showarrow=False,font=dict(size=20, color="black"),align="left")
                                pfo_img_buffer_lin = io.BytesIO(); fig_pfo_styled_lin.write_image(pfo_img_buffer_lin, format="png", width=1000, height=800, scale=2); pfo_img_buffer_lin.seek(0)
                                st.download_button(label="📥 Download Figure (PNG)", data=pfo_img_buffer_lin, file_name="pfo_linear.png", mime="image/png", key="dl_pfo_lin_fig_kin_tab_lin")
                            except Exception as e: st.warning(f"Error exporting linearized PFO: {e}")
                        except ValueError as ve_pfo_lin:
                            if "Insufficient variation" not in str(ve_pfo_lin): st.warning(f"Error in linearized PFO plot (regression): {ve_pfo_lin}")
                        except Exception as e_pfo_lin_exc: st.warning(f"Error in linearized PFO plot: {e_pfo_lin_exc}")
                    else: st.warning("Not enough valid points (t>0, qt < qe_calc) for linearized PFO plot.")
                else:
                    st.warning("Non-linear PFO parameter calculation (for qe) required but failed.")
                    st.info("The linearized PFO plot uses the `qe` value determined by non-linear fitting.")
                st.markdown("---")

                # PSO Linearized
                st.markdown("###### Linearized PSO: t/qt vs t")
                df_pso_lin_sec = kinetic_results[(kinetic_results['Time'] > 1e-9) & (kinetic_results['qt'] > 1e-9)].copy()
                if len(df_pso_lin_sec) >= 2:
                    try:
                        df_pso_lin_sec['t_div_qt'] = df_pso_lin_sec['Time'] / df_pso_lin_sec['qt']
                        t_pso_lin_sec, y_pso_lin_sec = df_pso_lin_sec['Time'], df_pso_lin_sec['t_div_qt']
                        if t_pso_lin_sec.nunique() < 2 or y_pso_lin_sec.nunique() < 2 :
                            st.warning("Insufficient variation for linearized PSO regression.")
                            raise ValueError("Insufficient variation for PSO linregress")
                        slope_pso_lin_sec, intercept_pso_lin_sec, r_val_pso_lin_sec, _, _ = linregress(t_pso_lin_sec, y_pso_lin_sec)
                        r2_pso_lin_val_sec = r_val_pso_lin_sec**2
                        qe_lin_val_sec = 1 / slope_pso_lin_sec if abs(slope_pso_lin_sec) > 1e-12 else np.nan
                        k2_lin_val_sec = slope_pso_lin_sec**2 / intercept_pso_lin_sec if abs(intercept_pso_lin_sec) > 1e-12 and not np.isnan(qe_lin_val_sec) else np.nan
                        
                        fig_pso_lin_plot_sec = px.scatter(df_pso_lin_sec, x='Time', y='t_div_qt', title=f"Linearized PSO (R²={r2_pso_lin_val_sec:.4f})", labels={'Time': "Time (min)", 't_div_qt': 't / qt (min·g/mg)'})
                        t_min_plot_pso_lin_sec, t_max_plot_pso_lin_sec = t_pso_lin_sec.min(), t_pso_lin_sec.max(); t_range_plot_pso_lin_sec = max(1.0, t_max_plot_pso_lin_sec - t_min_plot_pso_lin_sec)
                        t_line_for_pso_lin_sec = np.linspace(t_min_plot_pso_lin_sec - 0.05 * t_range_plot_pso_lin_sec, t_max_plot_pso_lin_sec + 0.05 * t_range_plot_pso_lin_sec, 50)
                        t_line_for_pso_lin_sec = np.maximum(0, t_line_for_pso_lin_sec)
                        y_line_for_pso_lin_sec = intercept_pso_lin_sec + slope_pso_lin_sec * t_line_for_pso_lin_sec
                        fig_pso_lin_plot_sec.add_trace(go.Scatter(x=t_line_for_pso_lin_sec, y=y_line_for_pso_lin_sec, mode='lines', name="Linear Fit"))
                        fig_pso_lin_plot_sec.update_layout(template="simple_white")
                        st.plotly_chart(fig_pso_lin_plot_sec, use_container_width=True)
                        st.caption(f"Slope = {slope_pso_lin_sec:.4f} (1/qe), Intercept = {intercept_pso_lin_sec:.4f} (1/(k2·qe²))\nR² = {r2_pso_lin_val_sec:.4f}, qe_lin = {qe_lin_val_sec:.3f} mg/g, k2_lin = {k2_lin_val_sec:.4f} g·mg⁻¹·min⁻¹")
                        try:
                            x_vals_pso_dl_lin_sec = np.linspace(t_pso_lin_sec.min(), t_pso_lin_sec.max(), 100)
                            y_vals_pso_dl_lin_sec = intercept_pso_lin_sec + slope_pso_lin_sec * x_vals_pso_dl_lin_sec
                            fig_pso_styled_lin_sec = go.Figure() 
                            fig_pso_styled_lin_sec.add_trace(go.Scatter(x=t_pso_lin_sec, y=y_pso_lin_sec, mode='markers', marker=dict(symbol='square', color='black', size=10), name="Experimental data"))
                            fig_pso_styled_lin_sec.add_trace(go.Scatter(x=x_vals_pso_dl_lin_sec, y=y_vals_pso_dl_lin_sec, mode='lines', line=dict(color='red', width=3), name="Linear regression"))
                            fig_pso_styled_lin_sec.update_layout(width=1000,height=800,plot_bgcolor='white',paper_bgcolor='white',font=dict(family="Times New Roman", size=22, color="black"),margin=dict(l=80, r=40, t=60, b=80),xaxis=dict(title="Time (min)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),yaxis=dict(title="t / qt (min·g/mg)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),showlegend=False)
                            operator_pso = "-" if intercept_pso_lin_sec < 0 else "+"
                            equation_text_pso = f"y = {slope_pso_lin_sec:.4f}x {operator_pso} {abs(intercept_pso_lin_sec):.4f}"
                            fig_pso_styled_lin_sec.add_annotation(xref="paper", yref="paper",x=0.05, y=0.95,text=f"{equation_text_pso}<br>R² = {r2_pso_lin_val_sec:.4f}",showarrow=False,font=dict(size=20, color="black"),align="left")
                            pso_img_buffer_lin_sec = io.BytesIO(); fig_pso_styled_lin_sec.write_image(pso_img_buffer_lin_sec, format="png", width=1000, height=800, scale=2); pso_img_buffer_lin_sec.seek(0)
                            st.download_button(label="📥 Download Figure (PNG)", data=pso_img_buffer_lin_sec, file_name="pso_linear.png", mime="image/png", key="dl_pso_lin_fig_kin_tab_lin")
                        except Exception as e_pso_export: st.warning(f"Error exporting linearized PSO: {e_pso_export}")
                    except ValueError as ve_pso_lin:
                        if "Insufficient variation" not in str(ve_pso_lin): st.warning(f"Error in linearized PSO plot (regression): {ve_pso_lin}")
                    except Exception as e_pso_lin_exc_sec: st.warning(f"Error in linearized PSO plot: {e_pso_lin_exc_sec}")
                else: st.warning("Not enough valid points (t>0 and qt>0) for linearized PSO plot.")
                st.markdown("---")

                # IPD
                st.subheader("Intraparticle Diffusion (IPD)")
                if not ipd_params_list and not kinetic_results.empty:
                     with st.spinner("Intraparticle Diffusion (IPD) Analysis..."):
                         ipd_df_calc_sec = kinetic_results[kinetic_results['Time'] > 1e-9].copy()
                         if len(ipd_df_calc_sec) >= 2:
                             try:
                                 if ipd_df_calc_sec['sqrt_t'].nunique() < 2 or ipd_df_calc_sec['qt'].nunique() < 2:
                                     st.warning("Insufficient variation in √t or qt for IPD regression.")
                                 else:
                                     slope_ipd_calc_sec, intercept_ipd_calc_sec, r_val_ipd_calc_sec, _, _ = linregress(ipd_df_calc_sec['sqrt_t'], ipd_df_calc_sec['qt'])
                                     if not np.isnan(r_val_ipd_calc_sec):
                                         st.session_state['ipd_params_list'] = [{'k_id': slope_ipd_calc_sec, 'C_ipd': intercept_ipd_calc_sec, 'R2_IPD': r_val_ipd_calc_sec**2, 'stage': 'Global'}]
                                         ipd_params_list = st.session_state['ipd_params_list'] 
                             except ValueError as ve_ipd_sec:
                                st.warning(f"IPD calculation failed (linear regression): {ve_ipd_sec}")
                             except Exception as e_ipd_sec:
                                 st.warning(f"IPD calculation failed: {e_ipd_sec}")
                         else:
                             st.warning("Not enough points (Time > 0) for IPD analysis.")
                
                if not kinetic_results.empty:
                    fig_i_plot_sec = px.scatter(kinetic_results, x='sqrt_t', y='qt', title="qt vs √Time", labels={'sqrt_t': "√Time (min⁰⁵)", 'qt': 'qt (mg/g)'})
                    if ipd_params_list: 
                        ipd_param_disp_sec = ipd_params_list[0]
                        if not np.isnan(ipd_param_disp_sec.get('k_id', np.nan)) and not np.isnan(ipd_param_disp_sec.get('C_ipd', np.nan)):
                            sqrt_t_min_ipd_sec, sqrt_t_max_ipd_sec = kinetic_results['sqrt_t'].min(), kinetic_results['sqrt_t'].max(); sqrt_t_range_ipd_sec = max(1.0, sqrt_t_max_ipd_sec - sqrt_t_min_ipd_sec)
                            sqrt_t_line_ipd_sec = np.linspace(max(0, sqrt_t_min_ipd_sec - 0.05*sqrt_t_range_ipd_sec), sqrt_t_max_ipd_sec + 0.05*sqrt_t_range_ipd_sec, 100)
                            qt_ipd_line_plot_sec = ipd_param_disp_sec['k_id'] * sqrt_t_line_ipd_sec + ipd_param_disp_sec['C_ipd']
                            qt_ipd_line_plot_sec = np.maximum(0, qt_ipd_line_plot_sec)
                            fig_i_plot_sec.add_trace(go.Scatter(x=sqrt_t_line_ipd_sec, y=qt_ipd_line_plot_sec, mode='lines', name=f"IPD Fit Global (R²={ipd_param_disp_sec.get('R2_IPD', np.nan):.3f})"))
                            fig_i_plot_sec.update_layout(template="simple_white")
                            st.plotly_chart(fig_i_plot_sec, use_container_width=True)
                            st.caption(f"IPD Parameters (global): k_id = {ipd_param_disp_sec.get('k_id', np.nan):.4f} mg·g⁻¹·min⁻⁰⁵, C = {ipd_param_disp_sec.get('C_ipd', np.nan):.3f} mg·g⁻¹, R² = {ipd_param_disp_sec.get('R2_IPD', np.nan):.4f}")
                            st.caption("If the line does not pass through the origin (C ≠ 0), intraparticle diffusion is not the sole rate-limiting step.")
                            try:
                                ipd_df_dl_lin_sec = kinetic_results[(kinetic_results['Time'] > 0)].copy()
                                fig_ipd_styled_lin_sec = go.Figure() 
                                fig_ipd_styled_lin_sec.add_trace(go.Scatter(x=ipd_df_dl_lin_sec['sqrt_t'], y=ipd_df_dl_lin_sec['qt'], mode='markers', marker=dict(symbol='square', color='black', size=10), name="Experimental data"))
                                if ipd_params_list and not np.isnan(ipd_params_list[0]['k_id']): 
                                    slope_ipd_dl_lin_sec, intercept_ipd_dl_lin_sec, r2_ipd_dl_lin_sec = ipd_params_list[0]['k_id'], ipd_params_list[0]['C_ipd'], ipd_params_list[0]['R2_IPD']
                                    x_line_ipd_dl_lin_sec = np.linspace(ipd_df_dl_lin_sec['sqrt_t'].min(), ipd_df_dl_lin_sec['sqrt_t'].max(), 100)
                                    y_line_ipd_dl_lin_sec = slope_ipd_dl_lin_sec * x_line_ipd_dl_lin_sec + intercept_ipd_dl_lin_sec
                                    fig_ipd_styled_lin_sec.add_trace(go.Scatter(x=x_line_ipd_dl_lin_sec, y=y_line_ipd_dl_lin_sec, mode='lines', line=dict(color='red', width=3), name="Linear regression"))
                                    operator_ipd = "-" if intercept_ipd_dl_lin_sec < 0 else "+"
                                    equation_text_ipd = f"y = {slope_ipd_dl_lin_sec:.4f}x {operator_ipd} {abs(intercept_ipd_dl_lin_sec):.4f}"
                                    fig_ipd_styled_lin_sec.add_annotation(xref="paper", yref="paper",x=0.05, y=0.95,text=f"{equation_text_ipd}<br>R² = {r2_ipd_dl_lin_sec:.4f}",showarrow=False,font=dict(size=20, color="black"),align="left")
                                fig_ipd_styled_lin_sec.update_layout(width=1000,height=800,plot_bgcolor='white',paper_bgcolor='white',font=dict(family="Times New Roman", size=22, color="black"),margin=dict(l=80, r=40, t=60, b=80),xaxis=dict(title="√t (min¹ᐟ²)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),yaxis=dict(title="qt (mg/g)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),showlegend=False)
                                ipd_img_buffer_lin_sec = io.BytesIO(); fig_ipd_styled_lin_sec.write_image(ipd_img_buffer_lin_sec, format="png", width=1000, height=800, scale=2); ipd_img_buffer_lin_sec.seek(0)
                                st.download_button(label="📥 Download Figure (PNG)", data=ipd_img_buffer_lin_sec, file_name="ipd_linear.png", mime="image/png", key="dl_ipd_lin_fig_kin_tab_lin")
                            except Exception as e_ipd_export: st.warning(f"Error exporting linearized IPD: {e_ipd_export}")
                        else:
                            fig_i_plot_sec.update_layout(template="simple_white")
                            st.plotly_chart(fig_i_plot_sec, use_container_width=True)
                            st.warning("IPD parameter calculation failed.")
                    else:
                        fig_i_plot_sec.update_layout(template="simple_white")
                        st.plotly_chart(fig_i_plot_sec, use_container_width=True)
                        st.write("IPD fit unavailable (not enough points or calculation error).")
                else: st.warning("No kinetic data to plot for IPD.")
            else:
                 st.warning("Fewer than 3 kinetic data points available. Cannot analyze models.")
                 
        elif kinetic_results is not None and kinetic_results.empty:
             st.warning("qt calculation produced no valid results.")

    elif not calib_params:
        st.warning("Please provide valid calibration data first.")
    else:
        st.info("Please enter data for the kinetic study.")