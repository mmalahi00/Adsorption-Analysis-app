# tabs/calibration_tab.py
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

def render():
    st.subheader("Calibration Curve")
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
                    name="Experimental points"
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
                        name="Linear regression"
                    ))

                    operator = "-" if intercept < 0 else "+"
                    equation_text = f"y = {slope:.4f}x {operator} {abs(intercept):.4f}"

                    fig.add_annotation(
                        x=x_max_data * 0.95, 
                        y=y_line[-1] * 0.1 + intercept * 0.9, 
                        text=f"{equation_text}", showarrow=False,
                        font=dict(family="Times New Roman, serif", size=12, color="red"),
                        align='right'
                    )
                    
                    y_max_data = calib_data['Absorbance'].max()
                    fig.update_layout(
                        title="Absorbance vs. Concentration", 
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
                        name="Experimental points"
                    ))

                    fig_styled.add_trace(go.Scatter(
                        x=x_line, y=y_line, mode='lines',
                        line=dict(color='red', width=3), name="Linear regression"
                    ))
                    fig_styled.update_layout(
                        width=1000, height=800, plot_bgcolor='white', paper_bgcolor='white',
                        font=dict(family="Times New Roman", size=22, color="black"),
                        margin=dict(l=80, r=40, t=60, b=80),
                        xaxis=dict(title="Concentration (mg/L)", linecolor='black', mirror=True, ticks='outside', showline=True, showgrid=False, zeroline=False),
                        yaxis=dict(title="Absorbance (A)", linecolor='black', mirror=True, ticks='outside', showline=True, showgrid=False, zeroline=False),
                        showlegend=False
                    )
                    fig_styled.add_annotation(
                        xref="paper", yref="paper", x=0.05, y=0.95,
                        text=f"{equation_text}<br>R² = {r_squared:.4f}",
                        showarrow=False, font=dict(size=20, color="black"), align="left"
                    )
                    img_buffer = io.BytesIO()
                    fig_styled.write_image(img_buffer, format="png", width=1000, height=800, scale=2)
                    img_buffer.seek(0)
                    st.download_button(
                        label="📥 Download Figure (PNG)", data=img_buffer,
                        file_name="calibration_styled.png", mime="image/png"
                    )
                except Exception as e_plot:
                    st.warning(f"Error creating calibration plot: {e_plot}")

            with col_param:
                st.markdown("##### Calibration Parameters")
                st.metric("Slope (A / (mg/L))", f"{calib_params['slope']:.4f}")
                st.metric("Intercept (A)", f"{calib_params['intercept']:.4f}")
                st.metric("Coefficient of Determination (R²)", f"{calib_params['r_squared']:.4f}")
        
        elif len(calib_data) < 2:
             st.warning("At least 2 valid data points are required for calibration.")
        else: 
             st.warning("Calibration parameters could not be calculated. Check data.", icon="⚠️")
             fig_raw = px.scatter(calib_data, x='Concentration', y='Absorbance',
                                 title="Raw Calibration Data", 
                                 labels={'Concentration': 'Concentration', 'Absorbance': 'Absorbance'})
             fig_raw.update_layout(template="simple_white")
             st.plotly_chart(fig_raw, use_container_width=True)
    else:
        st.info("Enter valid calibration data in the sidebar.")