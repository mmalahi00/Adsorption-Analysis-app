# tabs/kinetic_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress
from scipy.optimize import curve_fit
import io
from translations import _t
from utils import convert_df_to_csv
from models import pfo_model, pso_model

def render():
    st.header(_t("kinetic_tab_main_header"))
    kinetic_input = st.session_state.get('kinetic_input')
    calib_params = st.session_state.get('calibration_params')
    kinetic_results = st.session_state.get('kinetic_results_df')
    
    # Initialize local variables from session state at the beginning
    pfo_params_nl = st.session_state.get('pfo_params_nonlinear')
    pso_params_nl = st.session_state.get('pso_params_nonlinear') 
    ipd_params_list = st.session_state.get('ipd_params_list', [])

    if kinetic_input and calib_params:
        if kinetic_results is None: 
            with st.spinner(_t("kinetic_spinner_qt_calc")):
                df_kin = kinetic_input['data'].copy()
                params_kin = kinetic_input['params']
                C0_k, V_k, m_k = params_kin['C0'], params_kin['V'], params_kin['m']

                if 'Absorbance_t' in df_kin.columns and m_k > 0 and V_k > 0:
                    try:
                        if abs(calib_params['slope']) < 1e-9:
                            st.error(_t("kinetic_error_qt_calc_slope_zero"))
                            st.session_state['kinetic_results_df'] = None
                            return 
                        
                        df_kin['Ct'] = (df_kin['Absorbance_t'] - calib_params['intercept']) / calib_params['slope']
                        df_kin['Ct'] = df_kin['Ct'].clip(lower=0)
                        df_kin['qt'] = (C0_k - df_kin['Ct']) * V_k / m_k
                        df_kin['qt'] = df_kin['qt'].clip(lower=0)
                        df_kin['sqrt_t'] = np.sqrt(df_kin['Temps_min'])

                        st.session_state['kinetic_results_df'] = df_kin[['Temps_min', 'Absorbance_t', 'Ct', 'qt', 'sqrt_t']].copy()
                        # Reset all derived kinetic parameters upon recalculation
                        st.session_state['pfo_params_nonlinear'] = None
                        st.session_state['pso_params_nonlinear'] = None
                        st.session_state['ipd_params_list'] = []
                        # Also reset local variables
                        pfo_params_nl = None 
                        pso_params_nl = None 
                        ipd_params_list = [] 
                        st.success(_t("kinetic_success_qt_calc"))
                    except ZeroDivisionError:
                        if 'kinetic_results_df' not in st.session_state or st.session_state.get('kinetic_results_df') is not None:
                            st.error(_t("kinetic_error_qt_calc_div_by_zero"))
                        st.session_state['kinetic_results_df'] = None
                    except Exception as e_qt:
                        st.error(_t("kinetic_error_qt_calc_general", e_qt=e_qt))
                        st.session_state['kinetic_results_df'] = None
                elif not ('Absorbance_t' in df_kin.columns):
                     st.error(_t("kinetic_error_missing_abs_t_col"))
                     st.session_state['kinetic_results_df'] = None
                else:
                     st.error(_t("kinetic_error_mass_volume_non_positive"))
                     st.session_state['kinetic_results_df'] = None
                
                kinetic_results = st.session_state.get('kinetic_results_df') # Re-fetch after calculation

        if kinetic_results is not None and not kinetic_results.empty:
            st.subheader(_t("kinetic_calculated_data_subheader"))
            st.dataframe(kinetic_results.style.format(precision=4))
            csv_kin_res = convert_df_to_csv(kinetic_results)
            st.download_button(_t("kinetic_download_data_button"), csv_kin_res, _t("kinetic_download_data_filename"), "text/csv", key='dl_kin_res_kin_tab_nl_top')
            st.caption(f"Conditions: C0={kinetic_input['params']['C0']}mg/L, m={kinetic_input['params']['m']}g, V={kinetic_input['params']['V']}L")
            st.markdown("---")

            # --- [BUG FIX] NON-LINEAR KINETIC MODEL FITTING & PLOT (Moved to be primary) ---
            # This section is moved before the linearized models because the linearized PFO
            # model depends on the 'qe' value calculated here.
            st.subheader(_t("kinetic_nonlinear_header")) 

            fig_qt_vs_t_combined = go.Figure() # Initialize figure for experimental data + NL fits
            fig_qt_vs_t_combined.add_trace(go.Scatter(
                x=kinetic_results['Temps_min'], 
                y=kinetic_results['qt'], 
                mode='markers', 
                name=_t("kinetic_plot_qt_vs_t_legend"),
                marker=dict(color='black', symbol='diamond-open', size=10)
            ))
            
            if len(kinetic_results) >= 3:
                t_data_kin = kinetic_results['Temps_min'].values
                qt_data_kin = kinetic_results['qt'].values
                qe_exp_kin = qt_data_kin[-1] if len(qt_data_kin) > 0 else np.nan

                # PFO Non-Linear Fit (re-fetch or calculate)
                pfo_params_nl = st.session_state.get('pfo_params_nonlinear') # Re-get from state
                if pfo_params_nl is None: 
                     with st.spinner(_t("kinetic_spinner_pfo_nl_calc")):
                         try:
                             p0_PFO_nl = [qe_exp_kin if not np.isnan(qe_exp_kin) and qe_exp_kin > 1e-6 else 1.0, 0.01]
                             params_PFO_nl_fit, _ = curve_fit(pfo_model, t_data_kin, qt_data_kin, p0=p0_PFO_nl, maxfev=5000, bounds=([0, 0], [np.inf, np.inf]))
                             qe_PFO_nl_val, k1_PFO_nl_val = params_PFO_nl_fit
                             qt_pred_PFO_nl = pfo_model(t_data_kin, qe_PFO_nl_val, k1_PFO_nl_val)
                             ss_res_PFO_nl = np.sum((qt_data_kin - qt_pred_PFO_nl)**2)
                             ss_tot_PFO_nl = np.sum((qt_data_kin - np.mean(qt_data_kin))**2)
                             r2_PFO_nl_val = 1 - (ss_res_PFO_nl / ss_tot_PFO_nl) if ss_tot_PFO_nl > 1e-9 else 0.0
                             st.session_state['pfo_params_nonlinear'] = {'qe_PFO_nl': qe_PFO_nl_val, 'k1_nl': k1_PFO_nl_val, 'R2_PFO_nl': r2_PFO_nl_val}
                             pfo_params_nl = st.session_state['pfo_params_nonlinear'] # Update local var
                         except Exception as e_pfo_nl_fit:
                             st.warning(_t("kinetic_warning_pfo_nl_calc_failed", e=e_pfo_nl_fit))
                             st.session_state['pfo_params_nonlinear'] = {'qe_PFO_nl': np.nan, 'k1_nl': np.nan, 'R2_PFO_nl': np.nan}
                             pfo_params_nl = st.session_state['pfo_params_nonlinear']
                
                if pfo_params_nl and not np.isnan(pfo_params_nl.get('qe_PFO_nl', np.nan)):
                    t_line_kin_nl_plot = np.linspace(0, t_data_kin.max()*1.1 if t_data_kin.size > 0 else 10, 200)
                    qt_pfo_fit_on_plot = pfo_model(t_line_kin_nl_plot, pfo_params_nl['qe_PFO_nl'], pfo_params_nl['k1_nl'])
                    fig_qt_vs_t_combined.add_trace(go.Scatter(x=t_line_kin_nl_plot, y=qt_pfo_fit_on_plot, mode='lines', name=_t("kinetic_nl_legend_fit", model_name="PFO")))

                # PSO Non-Linear Fit (re-fetch or calculate)
                pso_params_nl = st.session_state.get('pso_params_nonlinear') # Re-get from state
                if pso_params_nl is None:
                     with st.spinner(_t("kinetic_spinner_pso_nl_calc")):
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
                             pso_params_nl = st.session_state['pso_params_nonlinear'] # Update local var
                         except Exception as e_pso_nl_fit:
                             st.warning(_t("kinetic_warning_pso_nl_calc_failed", e=e_pso_nl_fit))
                             st.session_state['pso_params_nonlinear'] = {'qe_PSO_nl': np.nan, 'k2_nl': np.nan, 'R2_PSO_nl': np.nan}
                             pso_params_nl = st.session_state['pso_params_nonlinear']
                
                if pso_params_nl and not np.isnan(pso_params_nl.get('qe_PSO_nl', np.nan)):
                    t_line_kin_nl_plot = np.linspace(0, t_data_kin.max()*1.1 if t_data_kin.size > 0 else 10, 200)
                    qt_pso_fit_on_plot = pso_model(t_line_kin_nl_plot, pso_params_nl['qe_PSO_nl'], pso_params_nl['k2_nl'])
                    fig_qt_vs_t_combined.add_trace(go.Scatter(x=t_line_kin_nl_plot, y=qt_pso_fit_on_plot, mode='lines', name=_t("kinetic_nl_legend_fit", model_name="PSO")))

            fig_qt_vs_t_combined.update_layout(
                title=_t("kinetic_plot_qt_vs_t_title"), 
                xaxis_title=_t("kinetic_plot_qt_vs_t_xaxis"), 
                yaxis_title='qt (mg/g)',
                template="simple_white",
                legend_title_text="Données/Modèles"
            )
            st.plotly_chart(fig_qt_vs_t_combined, use_container_width=True)
            
            try:
                qt_combined_img_buffer = io.BytesIO()
                fig_qt_vs_t_combined.write_image(qt_combined_img_buffer, format="png", width=1000, height=800, scale=2)
                qt_combined_img_buffer.seek(0)
                st.download_button(
                    label=_t("download_png_button") + " (NL Fits Plot)",
                    data=qt_combined_img_buffer,
                    file_name=_t("kinetic_download_qt_vs_t_filename"),
                    mime="image/png",
                    key="dl_qt_vs_t_combined_fig_kin_tab"
                )
            except Exception as e_export_combined:
                st.warning(f"Error exporting combined kinetic plot: {e_export_combined}")
            
            # Display Non-Linear Kinetic Parameters Table
            st.markdown(_t("kinetic_nl_data_display_header"))
            
            kin_nl_params_data_disp = {'Modèle': [], 'Paramètre': [], 'Valeur': [], 'R² (Non-Linéaire)': []}
            if pfo_params_nl and not np.isnan(pfo_params_nl.get('qe_PFO_nl', np.nan)):
                kin_nl_params_data_disp['Modèle'].extend(['PFO (NL)', 'PFO (NL)'])
                kin_nl_params_data_disp['Paramètre'].extend(['qe (mg/g)', 'k1 (min⁻¹)'])
                kin_nl_params_data_disp['Valeur'].extend([f"{pfo_params_nl['qe_PFO_nl']:.3f}", f"{pfo_params_nl['k1_nl']:.4f}"])
                kin_nl_params_data_disp['R² (Non-Linéaire)'].extend([f"{pfo_params_nl['R2_PFO_nl']:.4f}"] * 2)
            
            if pso_params_nl and not np.isnan(pso_params_nl.get('qe_PSO_nl', np.nan)):
                kin_nl_params_data_disp['Modèle'].extend(['PSO (NL)', 'PSO (NL)'])
                kin_nl_params_data_disp['Paramètre'].extend(['qe (mg/g)', 'k2 (g·mg⁻¹·min⁻¹)'])
                kin_nl_params_data_disp['Valeur'].extend([f"{pso_params_nl['qe_PSO_nl']:.3f}", f"{pso_params_nl['k2_nl']:.4f}"])
                kin_nl_params_data_disp['R² (Non-Linéaire)'].extend([f"{pso_params_nl['R2_PSO_nl']:.4f}"] * 2)

            if kin_nl_params_data_disp['Modèle']:
                kin_nl_params_df_disp = pd.DataFrame(kin_nl_params_data_disp)
                st.dataframe(kin_nl_params_df_disp.set_index('Modèle'), use_container_width=True)
            else:
                st.info("Paramètres cinétiques non-linéaires non calculés ou erreur de calcul.")
            st.markdown("---")
            # --- END OF MOVED NON-LINEAR SECTION ---


            # --- LINEARIZED MODELS ---
            st.subheader(_t("kinetic_linearized_models_subheader"))
            if len(kinetic_results) >= 3:
                # PFO Linearized
                st.markdown(_t("kinetic_pfo_lin_header"))
                # This now correctly uses the pfo_params_nl calculated in the section above
                if pfo_params_nl and not np.isnan(pfo_params_nl.get('qe_PFO_nl', np.nan)):
                    qe_calc_pfo_for_lin = pfo_params_nl['qe_PFO_nl']
                    df_pfo_lin = kinetic_results[(kinetic_results['qt'] < qe_calc_pfo_for_lin - 1e-9) & (kinetic_results['qt'] >= 0) & (kinetic_results['Temps_min'] > 1e-9)].copy()
                    if len(df_pfo_lin) >= 2:
                        try:
                            df_pfo_lin['ln_qe_qt'] = np.log(qe_calc_pfo_for_lin - df_pfo_lin['qt'])
                            t_pfo_lin, y_pfo_lin = df_pfo_lin['Temps_min'], df_pfo_lin['ln_qe_qt']
                            if t_pfo_lin.nunique() < 2 or y_pfo_lin.nunique() < 2 :
                                 st.warning(_t("kinetic_warning_pfo_lin_insufficient_variation"))
                                 raise ValueError("Insufficient variation for PFO linregress")
                            slope_pfo_lin, intercept_pfo_lin, r_val_pfo, _, _ = linregress(t_pfo_lin, y_pfo_lin)
                            r2_pfo_lin = r_val_pfo**2
                            k1_lin_val = -slope_pfo_lin

                            fig_pfo_lin_plot = px.scatter(df_pfo_lin, x='Temps_min', y='ln_qe_qt', title=_t("kinetic_pfo_lin_plot_title", r2_pfo_lin=r2_pfo_lin), labels={'Temps_min': _t("kinetic_plot_qt_vs_t_xaxis"), 'ln_qe_qt': 'ln(qe - qt)'})
                            t_min_plot_pfo_lin, t_max_plot_pfo_lin = t_pfo_lin.min(), t_pfo_lin.max(); t_range_plot_pfo_lin = max(1.0, t_max_plot_pfo_lin - t_min_plot_pfo_lin)
                            t_line_for_pfo_lin = np.linspace(t_min_plot_pfo_lin - 0.05 * t_range_plot_pfo_lin, t_max_plot_pfo_lin + 0.05 * t_range_plot_pfo_lin, 50)
                            t_line_for_pfo_lin = np.maximum(0, t_line_for_pfo_lin) 
                            y_line_for_pfo_lin = intercept_pfo_lin + slope_pfo_lin * t_line_for_pfo_lin
                            fig_pfo_lin_plot.add_trace(go.Scatter(x=t_line_for_pfo_lin, y=y_line_for_pfo_lin, mode='lines', name=_t("isotherm_lin_plot_legend_fit")))
                            fig_pfo_lin_plot.update_layout(template="simple_white")
                            st.plotly_chart(fig_pfo_lin_plot, use_container_width=True)
                            st.caption(_t("kinetic_pfo_lin_caption", slope_pfo_lin=slope_pfo_lin, intercept_pfo_lin=intercept_pfo_lin, r2_pfo_lin=r2_pfo_lin, k1_lin=k1_lin_val))
                            try:
                                x_vals_pfo_dl_lin = np.linspace(t_pfo_lin.min(), t_pfo_lin.max(), 100)
                                y_vals_pfo_dl_lin = intercept_pfo_lin + slope_pfo_lin * x_vals_pfo_dl_lin
                                fig_pfo_styled_lin = go.Figure() 
                                fig_pfo_styled_lin.add_trace(go.Scatter(x=t_pfo_lin, y=y_pfo_lin, mode='markers', marker=dict(symbol='square', color='black', size=10), name=_t("isotherm_exp_plot_legend")))
                                fig_pfo_styled_lin.add_trace(go.Scatter(x=x_vals_pfo_dl_lin, y=y_vals_pfo_dl_lin, mode='lines', line=dict(color='red', width=3), name=_t("calib_tab_legend_reg")))
                                fig_pfo_styled_lin.update_layout(width=1000,height=800,plot_bgcolor='white',paper_bgcolor='white',font=dict(family="Times New Roman", size=22, color="black"),margin=dict(l=80, r=40, t=60, b=80),xaxis=dict(title=_t("kinetic_plot_qt_vs_t_xaxis"),linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),yaxis=dict(title="ln(qe - qt)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),showlegend=False)
                                fig_pfo_styled_lin.add_annotation(xref="paper", yref="paper",x=0.05, y=0.95,text=f"y = {slope_pfo_lin:.4f}x + {intercept_pfo_lin:.4f}<br>R² = {r2_pfo_lin:.4f}",showarrow=False,font=dict(size=20, color="black"),align="left")
                                pfo_img_buffer_lin = io.BytesIO(); fig_pfo_styled_lin.write_image(pfo_img_buffer_lin, format="png", width=1000, height=800, scale=2); pfo_img_buffer_lin.seek(0)
                                st.download_button(label=_t("download_png_button"), data=pfo_img_buffer_lin, file_name=_t("kinetic_download_pfo_lin_filename"), mime="image/png", key="dl_pfo_lin_fig_kin_tab_lin")
                            except Exception as e: st.warning(_t("kinetic_error_export_pfo_lin", e=e))
                        except ValueError as ve_pfo_lin:
                            if "Insufficient variation" not in str(ve_pfo_lin): st.warning(_t("kinetic_error_pfo_lin_plot_regression", ve=ve_pfo_lin))
                        except Exception as e_pfo_lin_exc: st.warning(_t("kinetic_error_pfo_lin_plot_general", e_pfo_lin=e_pfo_lin_exc))
                    else: st.warning(_t("kinetic_warning_pfo_lin_not_enough_points"))
                else:
                    st.warning(_t("kinetic_warning_pfo_nl_calc_required"))
                    st.info(_t("kinetic_info_pfo_lin_uses_nl_qe"))
                st.markdown("---")

                # PSO Linearized
                st.markdown(_t("kinetic_pso_lin_header"))
                df_pso_lin_sec = kinetic_results[(kinetic_results['Temps_min'] > 1e-9) & (kinetic_results['qt'] > 1e-9)].copy()
                if len(df_pso_lin_sec) >= 2:
                    try:
                        df_pso_lin_sec['t_div_qt'] = df_pso_lin_sec['Temps_min'] / df_pso_lin_sec['qt']
                        t_pso_lin_sec, y_pso_lin_sec = df_pso_lin_sec['Temps_min'], df_pso_lin_sec['t_div_qt']
                        if t_pso_lin_sec.nunique() < 2 or y_pso_lin_sec.nunique() < 2 :
                            st.warning(_t("kinetic_warning_pso_lin_insufficient_variation"))
                            raise ValueError("Insufficient variation for PSO linregress")
                        slope_pso_lin_sec, intercept_pso_lin_sec, r_val_pso_lin_sec, _, _ = linregress(t_pso_lin_sec, y_pso_lin_sec)
                        r2_pso_lin_val_sec = r_val_pso_lin_sec**2
                        qe_lin_val_sec = 1 / slope_pso_lin_sec if abs(slope_pso_lin_sec) > 1e-12 else np.nan
                        k2_lin_val_sec = slope_pso_lin_sec**2 / intercept_pso_lin_sec if abs(intercept_pso_lin_sec) > 1e-12 and not np.isnan(qe_lin_val_sec) else np.nan
                        
                        fig_pso_lin_plot_sec = px.scatter(df_pso_lin_sec, x='Temps_min', y='t_div_qt', title=_t("kinetic_pso_lin_plot_title", r2_pso_lin=r2_pso_lin_val_sec), labels={'Temps_min': _t("kinetic_plot_qt_vs_t_xaxis"), 't_div_qt': 't / qt (min·g/mg)'})
                        t_min_plot_pso_lin_sec, t_max_plot_pso_lin_sec = t_pso_lin_sec.min(), t_pso_lin_sec.max(); t_range_plot_pso_lin_sec = max(1.0, t_max_plot_pso_lin_sec - t_min_plot_pso_lin_sec)
                        t_line_for_pso_lin_sec = np.linspace(t_min_plot_pso_lin_sec - 0.05 * t_range_plot_pso_lin_sec, t_max_plot_pso_lin_sec + 0.05 * t_range_plot_pso_lin_sec, 50)
                        t_line_for_pso_lin_sec = np.maximum(0, t_line_for_pso_lin_sec)
                        y_line_for_pso_lin_sec = intercept_pso_lin_sec + slope_pso_lin_sec * t_line_for_pso_lin_sec
                        fig_pso_lin_plot_sec.add_trace(go.Scatter(x=t_line_for_pso_lin_sec, y=y_line_for_pso_lin_sec, mode='lines', name=_t("isotherm_lin_plot_legend_fit")))
                        fig_pso_lin_plot_sec.update_layout(template="simple_white")
                        st.plotly_chart(fig_pso_lin_plot_sec, use_container_width=True)
                        st.caption(_t("kinetic_pso_lin_caption", slope_pso_lin=slope_pso_lin_sec, intercept_pso_lin=intercept_pso_lin_sec, r2_pso_lin=r2_pso_lin_val_sec, qe_lin=qe_lin_val_sec, k2_lin=k2_lin_val_sec))
                        try:
                            x_vals_pso_dl_lin_sec = np.linspace(t_pso_lin_sec.min(), t_pso_lin_sec.max(), 100)
                            y_vals_pso_dl_lin_sec = intercept_pso_lin_sec + slope_pso_lin_sec * x_vals_pso_dl_lin_sec
                            fig_pso_styled_lin_sec = go.Figure() 
                            fig_pso_styled_lin_sec.add_trace(go.Scatter(x=t_pso_lin_sec, y=y_pso_lin_sec, mode='markers', marker=dict(symbol='square', color='black', size=10), name=_t("isotherm_exp_plot_legend")))
                            fig_pso_styled_lin_sec.add_trace(go.Scatter(x=x_vals_pso_dl_lin_sec, y=y_vals_pso_dl_lin_sec, mode='lines', line=dict(color='red', width=3), name=_t("calib_tab_legend_reg")))
                            fig_pso_styled_lin_sec.update_layout(width=1000,height=800,plot_bgcolor='white',paper_bgcolor='white',font=dict(family="Times New Roman", size=22, color="black"),margin=dict(l=80, r=40, t=60, b=80),xaxis=dict(title=_t("kinetic_plot_qt_vs_t_xaxis"),linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),yaxis=dict(title="t / qt (min·g/mg)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),showlegend=False)
                            fig_pso_styled_lin_sec.add_annotation(xref="paper", yref="paper",x=0.05, y=0.95,text=f"y = {slope_pso_lin_sec:.4f}x + {intercept_pso_lin_sec:.4f}<br>R² = {r2_pso_lin_val_sec:.4f}",showarrow=False,font=dict(size=20, color="black"),align="left")
                            pso_img_buffer_lin_sec = io.BytesIO(); fig_pso_styled_lin_sec.write_image(pso_img_buffer_lin_sec, format="png", width=1000, height=800, scale=2); pso_img_buffer_lin_sec.seek(0)
                            st.download_button(label=_t("download_png_button"), data=pso_img_buffer_lin_sec, file_name=_t("kinetic_download_pso_lin_filename"), mime="image/png", key="dl_pso_lin_fig_kin_tab_lin")
                        except Exception as e_pso_export: st.warning(_t("kinetic_error_export_pso_lin", e=e_pso_export))
                    except ValueError as ve_pso_lin:
                        if "Insufficient variation" not in str(ve_pso_lin): st.warning(_t("kinetic_error_pso_lin_plot_regression", ve=ve_pso_lin))
                    except Exception as e_pso_lin_exc_sec: st.warning(_t("kinetic_error_pso_lin_plot_general", e_pso_lin=e_pso_lin_exc_sec))
                else: st.warning(_t("kinetic_warning_pso_lin_not_enough_points"))
                st.markdown("---")

                # IPD
                st.subheader(_t("kinetic_ipd_subheader"))
                ipd_params_list = st.session_state.get('ipd_params_list', []) # Re-fetch
                if not ipd_params_list and not kinetic_results.empty:
                     with st.spinner(_t("kinetic_spinner_ipd_analysis")):
                         ipd_df_calc_sec = kinetic_results[kinetic_results['Temps_min'] > 1e-9].copy()
                         if len(ipd_df_calc_sec) >= 2:
                             try:
                                 if ipd_df_calc_sec['sqrt_t'].nunique() < 2 or ipd_df_calc_sec['qt'].nunique() < 2:
                                     st.warning(_t("kinetic_warning_ipd_insufficient_variation"))
                                 else:
                                     slope_ipd_calc_sec, intercept_ipd_calc_sec, r_val_ipd_calc_sec, _, _ = linregress(ipd_df_calc_sec['sqrt_t'], ipd_df_calc_sec['qt'])
                                     if not np.isnan(r_val_ipd_calc_sec):
                                         st.session_state['ipd_params_list'] = [{'k_id': slope_ipd_calc_sec, 'C_ipd': intercept_ipd_calc_sec, 'R2_IPD': r_val_ipd_calc_sec**2, 'stage': 'Global'}]
                                         ipd_params_list = st.session_state['ipd_params_list'] 
                             except ValueError as ve_ipd_sec:
                                st.warning(_t("kinetic_warning_ipd_calc_failed_regression", ve=ve_ipd_sec))
                             except Exception as e_ipd_sec:
                                 st.warning(_t("kinetic_warning_ipd_calc_failed_general", e=e_ipd_sec))
                         else:
                             st.warning(_t("kinetic_warning_ipd_not_enough_points"))
                
                if not kinetic_results.empty:
                    fig_i_plot_sec = px.scatter(kinetic_results, x='sqrt_t', y='qt', title=_t("kinetic_ipd_plot_title"), labels={'sqrt_t': _t("kinetic_ipd_plot_xaxis"), 'qt': 'qt (mg/g)'})
                    if ipd_params_list: 
                        ipd_param_disp_sec = ipd_params_list[0]
                        if not np.isnan(ipd_param_disp_sec.get('k_id', np.nan)) and not np.isnan(ipd_param_disp_sec.get('C_ipd', np.nan)):
                            sqrt_t_min_ipd_sec, sqrt_t_max_ipd_sec = kinetic_results['sqrt_t'].min(), kinetic_results['sqrt_t'].max(); sqrt_t_range_ipd_sec = max(1.0, sqrt_t_max_ipd_sec - sqrt_t_min_ipd_sec)
                            sqrt_t_line_ipd_sec = np.linspace(max(0, sqrt_t_min_ipd_sec - 0.05*sqrt_t_range_ipd_sec), sqrt_t_max_ipd_sec + 0.05*sqrt_t_range_ipd_sec, 100)
                            qt_ipd_line_plot_sec = ipd_param_disp_sec['k_id'] * sqrt_t_line_ipd_sec + ipd_param_disp_sec['C_ipd']
                            qt_ipd_line_plot_sec = np.maximum(0, qt_ipd_line_plot_sec)
                            fig_i_plot_sec.add_trace(go.Scatter(x=sqrt_t_line_ipd_sec, y=qt_ipd_line_plot_sec, mode='lines', name=_t("kinetic_ipd_plot_legend_fit", r2_ipd=ipd_param_disp_sec.get('R2_IPD', np.nan))))
                            fig_i_plot_sec.update_layout(template="simple_white")
                            st.plotly_chart(fig_i_plot_sec, use_container_width=True)
                            st.caption(_t("kinetic_ipd_caption_params", k_id=ipd_param_disp_sec.get('k_id', np.nan), C_ipd=ipd_param_disp_sec.get('C_ipd', np.nan), R2_IPD=ipd_param_disp_sec.get('R2_IPD', np.nan)))
                            st.caption(_t("kinetic_ipd_caption_interp"))
                            try:
                                ipd_df_dl_lin_sec = kinetic_results[(kinetic_results['Temps_min'] > 0)].copy()
                                fig_ipd_styled_lin_sec = go.Figure() 
                                fig_ipd_styled_lin_sec.add_trace(go.Scatter(x=ipd_df_dl_lin_sec['sqrt_t'], y=ipd_df_dl_lin_sec['qt'], mode='markers', marker=dict(symbol='square', color='black', size=10), name=_t("isotherm_exp_plot_legend")))
                                if ipd_params_list and not np.isnan(ipd_params_list[0]['k_id']): 
                                    slope_ipd_dl_lin_sec, intercept_ipd_dl_lin_sec, r2_ipd_dl_lin_sec = ipd_params_list[0]['k_id'], ipd_params_list[0]['C_ipd'], ipd_params_list[0]['R2_IPD']
                                    x_line_ipd_dl_lin_sec = np.linspace(ipd_df_dl_lin_sec['sqrt_t'].min(), ipd_df_dl_lin_sec['sqrt_t'].max(), 100)
                                    y_line_ipd_dl_lin_sec = slope_ipd_dl_lin_sec * x_line_ipd_dl_lin_sec + intercept_ipd_dl_lin_sec
                                    fig_ipd_styled_lin_sec.add_trace(go.Scatter(x=x_line_ipd_dl_lin_sec, y=y_line_ipd_dl_lin_sec, mode='lines', line=dict(color='red', width=3), name=_t("calib_tab_legend_reg")))
                                    fig_ipd_styled_lin_sec.add_annotation(xref="paper", yref="paper",x=0.05, y=0.95,text=f"y = {slope_ipd_dl_lin_sec:.4f}x + {intercept_ipd_dl_lin_sec:.4f}<br>R² = {r2_ipd_dl_lin_sec:.4f}",showarrow=False,font=dict(size=20, color="black"),align="left")
                                fig_ipd_styled_lin_sec.update_layout(width=1000,height=800,plot_bgcolor='white',paper_bgcolor='white',font=dict(family="Times New Roman", size=22, color="black"),margin=dict(l=80, r=40, t=60, b=80),xaxis=dict(title="√t (min¹ᐟ²)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),yaxis=dict(title="qt (mg/g)",linecolor='black',mirror=True,ticks='outside',showline=True,showgrid=False,zeroline=False),showlegend=False)
                                ipd_img_buffer_lin_sec = io.BytesIO(); fig_ipd_styled_lin_sec.write_image(ipd_img_buffer_lin_sec, format="png", width=1000, height=800, scale=2); ipd_img_buffer_lin_sec.seek(0)
                                st.download_button(label=_t("download_png_button"), data=ipd_img_buffer_lin_sec, file_name=_t("kinetic_download_ipd_lin_filename"), mime="image/png", key="dl_ipd_lin_fig_kin_tab_lin")
                            except Exception as e_ipd_export: st.warning(_t("kinetic_error_export_ipd_lin", e=e_ipd_export))
                        else:
                            fig_i_plot_sec.update_layout(template="simple_white")
                            st.plotly_chart(fig_i_plot_sec, use_container_width=True)
                            st.warning(_t("kinetic_warning_ipd_params_calc_failed"))
                    else:
                        fig_i_plot_sec.update_layout(template="simple_white")
                        st.plotly_chart(fig_i_plot_sec, use_container_width=True)
                        st.write(_t("kinetic_info_ipd_fit_unavailable"))
                else: st.warning(_t("kinetic_warning_no_data_for_ipd"))
            else:
                 st.warning(_t("kinetic_warning_less_than_3_points"))
                 
        elif kinetic_results is not None and kinetic_results.empty:
             st.warning(_t("kinetic_warning_qt_calc_no_results"))

    elif not calib_params:
        st.warning(_t("kinetic_warning_provide_calib_data"))
    else:
        st.info(_t("kinetic_info_enter_kinetic_data"))