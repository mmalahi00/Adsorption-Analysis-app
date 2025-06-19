# tabs/calibration_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress
import io
from translations import _t 
from utils import convert_df_to_csv

# --- AUTOMATIC CALIBRATION LOGIC ---
new_calib_df = st.session_state.get('calib_df_input')
old_calib_df = st.session_state.get('previous_calib_df')

if new_calib_df is not None and len(new_calib_df) >= 2:
    if old_calib_df is None or not new_calib_df.equals(old_calib_df):
        try:
            slope, intercept, r_value, _, _ = linregress(new_calib_df['Concentration'], new_calib_df['Absorbance'])
            if abs(slope) > 1e-9: # Avoid near-zero slope issues
                st.session_state['calibration_params'] = {'slope': slope, 'intercept': intercept, 'r_squared': r_value**2}
            else:
                st.session_state['calibration_params'] = None
                st.sidebar.warning(_t("calib_slope_near_zero_warning"), icon="⚠️")
        except Exception as e:
            st.session_state['calibration_params'] = None
            st.sidebar.error(_t("calib_error_calc_warning", error=e), icon="🔥")
        
        st.session_state['previous_calib_df'] = new_calib_df.copy() # Store current data for future comparison

elif new_calib_df is None or len(new_calib_df) < 2:
    # Reset calibration if input becomes invalid or insufficient
    if st.session_state.get('calibration_params') is not None:
        st.session_state['calibration_params'] = None
    if st.session_state.get('previous_calib_df') is not None: 
        st.session_state['previous_calib_df'] = None
        
def render():
    st.subheader(_t("calib_tab_subheader"))
    calib_params = st.session_state.get('calibration_params')
    calib_data = st.session_state.get('calib_df_input')

    if calib_data is not None and not calib_data.empty:
        if calib_params and len(calib_data) >= 2:
            col_plot, col_param = st.columns([2, 1])

            with col_plot:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=calib_data['Concentration'],
                    y=calib_data['Absorbance'],
                    mode='markers',
                    marker=dict(color='blue', symbol='circle', size=8),
                    name=_t("calib_tab_legend_exp")
                ))

                try:
                    slope = calib_params['slope']
                    intercept = calib_params['intercept']
                    r_squared = calib_params['r_squared']

                    x_min_data = calib_data['Concentration'].min()
                    x_max_data = calib_data['Concentration'].max()
                    x_range_ext = (x_max_data - x_min_data) * 0.1 if x_max_data > x_min_data else 0.5
                    x_start = max(0, x_min_data - x_range_ext)
                    x_end = x_max_data + x_range_ext
                    x_line = np.array([x_start, x_end])
                    y_line = slope * x_line + intercept

                    fig.add_trace(go.Scatter(
                        x=x_line, y=y_line, mode='lines',
                        line=dict(color='red', width=1.5),
                        name=_t("calib_tab_legend_reg")
                    ))

                    equation_text = f"y = {slope:.4f}x {intercept:+.4f}"
                    fig.add_annotation(
                        x=x_max_data * 0.95, 
                        y=y_line[-1] * 0.1 + intercept * 0.9, # Heuristic positioning
                        text=equation_text, showarrow=False,
                        font=dict(family="Times New Roman, serif", size=12, color="red"),
                        align='right'
                    )
                    
                    y_max_data = calib_data['Absorbance'].max()
                    fig.update_layout(
                        title=_t("calib_tab_plot_title"), 
                        xaxis_title="Concentration (mg/L)", 
                        yaxis_title="Absorbance (A)",      
                        plot_bgcolor='white',             
                        xaxis=dict(showgrid=True, gridcolor='LightGrey', gridwidth=1, zeroline=False, range=[0, x_end * 1.05]),
                        yaxis=dict(showgrid=True, gridcolor='LightGrey', gridwidth=1, zeroline=False, range=[0, y_max_data * 1.1]),
                        legend=dict(x=0.02, y=0.98, traceorder='normal', bgcolor='rgba(255,255,255,0.8)', bordercolor='Black', borderwidth=1),
                        font=dict(family="Times New Roman, serif", size=12),
                        margin=dict(l=40, r=30, t=50, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Figure for download
                    fig_styled = go.Figure()
                    fig_styled.add_trace(go.Scatter(
                        x=calib_data['Concentration'], y=calib_data['Absorbance'],
                        mode='markers', marker=dict(symbol='square', color='black', size=10),
                        name=_t("calib_tab_legend_exp")
                    ))
                    # Ensure x_vals_dl covers the data range, even if only one point for regression line
                    if len(calib_data['Concentration']) > 1 :
                        x_vals_dl_min = calib_data['Concentration'].min()
                        x_vals_dl_max = calib_data['Concentration'].max()
                    else: # Handle single point case for line drawing
                        x_vals_dl_min = calib_data['Concentration'].iloc[0] - 0.5 
                        x_vals_dl_max = calib_data['Concentration'].iloc[0] + 0.5
                    x_vals_dl = np.array([x_vals_dl_min, x_vals_dl_max])

                    y_vals_dl = slope * x_vals_dl + intercept
                    fig_styled.add_trace(go.Scatter(
                        x=x_vals_dl, y=y_vals_dl, mode='lines',
                        line=dict(color='red', width=3), name=_t("calib_tab_legend_reg")
                    ))
                    fig_styled.update_layout(
                        width=1000, height=800, plot_bgcolor='white', paper_bgcolor='white',
                        font=dict(family="Times New Roman", size=22, color="black"),
                        margin=dict(l=80, r=40, t=60, b=80),
                        xaxis=dict(title=_t("calib_tab_styled_xaxis_label"), linecolor='black', mirror=True, ticks='outside', showline=True, showgrid=False, zeroline=False),
                        yaxis=dict(title="Absorbance (A)", linecolor='black', mirror=True, ticks='outside', showline=True, showgrid=False, zeroline=False),
                        showlegend=False
                    )
                    fig_styled.add_annotation(
                        xref="paper", yref="paper", x=0.05, y=0.95,
                        text=f"y = {slope:.4f}x + {intercept:.4f}<br>R² = {r_squared:.4f}",
                        showarrow=False, font=dict(size=20, color="black"), align="left"
                    )
                    img_buffer = io.BytesIO()
                    fig_styled.write_image(img_buffer, format="png", width=1000, height=800, scale=2)
                    img_buffer.seek(0)
                    st.download_button(
                        label=_t("download_png_button"), data=img_buffer,
                        file_name=_t("calib_tab_download_styled_png_filename"), mime="image/png"
                    )
                except Exception as e_plot:
                    st.warning(_t("calib_plot_error_warning", e_plot=e_plot))

            with col_param:
                st.markdown(f"##### {_t('calib_tab_params_header')}")
                st.metric(_t("calib_tab_slope_metric"), f"{calib_params['slope']:.4f}")
                st.metric("Intercept (A)", f"{calib_params['intercept']:.4f}")
                st.metric(_t("calib_tab_r2_metric"), f"{calib_params['r_squared']:.4f}")
        
        elif len(calib_data) < 2:
             st.warning(_t("calib_tab_min_2_points_warning"))
        else: # calib_params is None but data is present
             st.warning(_t("calib_tab_params_not_calc_warning"), icon="⚠️")
             # Optionally display the raw data plot without regression
             fig_raw = px.scatter(calib_data, x='Concentration', y='Absorbance',
                                 title=_t("calib_tab_raw_data_plot_title"), 
                                 labels={'Concentration': 'Concentration', 'Absorbance': 'Absorbance'})
             fig_raw.update_layout(template="simple_white")
             st.plotly_chart(fig_raw, use_container_width=True)
    else:
        st.info(_t("calib_tab_enter_data_info"))