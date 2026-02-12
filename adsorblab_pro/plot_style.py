# plot_style.py
"""
AdsorbLab Pro - Publication-Quality Plot Styling
=================================================

This module provides consistent, professional styling for all plots in
AdsorbLab Pro, matching the verified visualization style.

Features:
- Plotly template for interactive charts
- Matplotlib style for static exports
- Consistent color schemes
- Professional marker and line styles
- Easy integration with existing code

Usage:
    from plot_style import (
        apply_professional_style, create_isotherm_plot, create_kinetic_plot,
        COLORS, MARKERS, get_plotly_template
    )

    # For new plots
    fig = create_isotherm_plot(Ce, qe, Ce_fit, qe_fit, model_name='Langmuir', r_squared=0.998)

    # For existing plots
    fig = apply_professional_style(existing_fig)
"""

import logging

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)
from collections.abc import Callable
from typing import Any

from .config import FONT_FAMILY, PLOT_TEMPLATE

# =============================================================================
# COLOR SCHEMES
# =============================================================================
__all__ = [
# Color schemes
    "COLORS",
    "MODEL_COLORS",
    "STUDY_COLORS",
    # Style dictionaries
    "MARKERS",
    "AXIS_STYLE",
    "MATPLOTLIB_STYLE",
    # Layout helpers
    "get_axis_style",
    "get_legend_style",
    # Color utilities
    "get_study_color",
    "hex_to_rgba",
    # Plot creation functions
    "create_isotherm_plot",
    "create_kinetic_plot",
    "create_model_comparison_plot",
    "create_residual_plot",
    "create_parity_plot",
    # Styling functions
    "apply_professional_style",
    "apply_professional_polar_style",
    "apply_professional_3d_style",
    "apply_matplotlib_style",
    # Trace styling helpers
    "style_experimental_trace",
    "style_fit_trace",
    "style_ci_traces",
    "create_effect_plot",
    "create_vant_hoff_plot",
    "create_dual_axis_effect_plot",
    "style_study_trace",
    "finalize_figure",
    "infer_figure_kind",
    "prepare_figure_for_export",
]
COLORS = {
    # Experimental points (case-study style)
    "experimental": "#000000",  # Black markers
    "experimental_edge": "#000000",

    # Core line palette (Matplotlib tab10 – publication-friendly)
    "fit_primary": "#1f77b4",      # Blue
    "fit_secondary": "#ff7f0e",    # Orange
    "fit_tertiary": "#2ca02c",     # Green
    "fit_quaternary": "#d62728",   # Red
    "fit_quinary": "#9467bd",      # Purple (optional)

    # Extended palette for multi-model comparison
    "model_colors": [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
    ],

    # Background and grid
    "background": "#FFFFFF",
    "grid": "#E0E0E0",
    "grid_zero": "#BDBDBD",
    "tick_text": "#424242",

    # Confidence interval
    "ci_fill": "rgba(31, 119, 180, 0.15)",
    "ci_line": "rgba(31, 119, 180, 0.40)",

    # Residuals
    "residual_positive": "#2ca02c",
    "residual_negative": "#d62728",
    "residual_line": "#757575",
}

# Model name to color mapping
MODEL_COLORS = {
    # Isotherms (match case_studies)
    "Langmuir": "#1f77b4",
    "Freundlich": "#ff7f0e",
    "Temkin": "#2ca02c",
    "Sips": "#d62728",

    # Kinetics (consistent palette)
    "PFO": "#1f77b4",
    "PSO": "#ff7f0e",
    "rPSO": "#9467bd",
    "Elovich": "#2ca02c",
    "IPD": "#d62728",
}


STUDY_COLORS = [
    "#2E86AB",  # Blue
    "#A23B72",  # Magenta
    "#F18F01",  # Orange
    "#C73E1D",  # Red
    "#3B1F2B",  # Dark purple
    "#95C623",  # Lime green
    "#5C4D7D",  # Purple
    "#E84855",  # Coral red
    "#2D3047",  # Dark blue
    "#1B998B",  # Teal
]

# =============================================================================
# LAYOUT HELPER FUNCTIONS (DRY - Don't Repeat Yourself)
# =============================================================================


def get_axis_style(title: str) -> dict:
    """
    Get standard axis configuration.

    Parameters
    ----------
    title : str
        Axis title (supports HTML like '<sub>e</sub>')

    Returns
    -------
    dict
        Axis configuration for Plotly
    """
    return {
        "title": {"text": title, "font": {"size": 14, "family": FONT_FAMILY}},
        "showgrid": False,
        "gridwidth": 1,
        "gridcolor": COLORS["grid"],
        "showline": True,
        "linewidth": 2,
        "linecolor": "black",
        "mirror": True,
        "ticks": "outside",
        "tickfont": {"size": 11, "family": FONT_FAMILY, "color": COLORS["tick_text"]},
        "zeroline": False,
    }


def get_legend_style(x: float, y: float, xanchor: str = "left", yanchor: str = "top") -> dict:
    """
    Get standard legend configuration.

    Parameters
    ----------
    x, y : float
        Legend position (0-1 range)
    xanchor, yanchor : str
        Anchor points for positioning

    Returns
    -------
    dict
        Legend configuration for Plotly
    """
    return {
        "x": x,
        "y": y,
        "xanchor": xanchor,
        "yanchor": yanchor,
        "bgcolor": "rgba(255, 255, 255, 0.9)",
        "bordercolor": "black",
        "borderwidth": 1,
    }


# -----------------------------------------------------------------------------
# Backwards-compatible private helpers
#
# The public API uses get_axis_style/get_legend_style. Some internal callers
# and tests refer to the older private names.
# -----------------------------------------------------------------------------


def _get_axis_style(title: str) -> dict:
    """Alias for :func:`get_axis_style` (kept for backwards compatibility)."""

    return get_axis_style(title)


def _get_legend_style(x: float, y: float, xanchor: str = "left", yanchor: str = "top") -> dict:
    """Alias for :func:`get_legend_style` (kept for backwards compatibility)."""

    return get_legend_style(x, y, xanchor=xanchor, yanchor=yanchor)


def _get_base_layout(title: str, height: int = 450, show_legend: bool = True) -> dict:
    """
    Get base layout configuration for all plots.

    Parameters
    ----------
    title : str
        Plot title
    height : int
        Figure height in pixels
    show_legend : bool
        Whether to show legend

    Returns
    -------
    dict
        Base layout configuration for Plotly
    """
    return {
        "title": {"text": f"<b>{title}</b>", "font": {"size": 16, "family": FONT_FAMILY}},
        "height": height,
        "showlegend": show_legend,
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "font": {"family": FONT_FAMILY, "size": 12},
        "margin": {"l": 70, "r": 40, "t": 60, "b": 60},
    }


def apply_standard_layout(
    fig: go.Figure,
    title: str,
    x_title: str,
    y_title: str,
    height: int = 450,
    legend_position: tuple = (0.02, 0.98, "left", "top"),
    show_legend: bool = True,
) -> go.Figure:
    """
    Apply standard professional layout to a figure.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to style
    title : str
        Plot title
    x_title : str
        X-axis title
    y_title : str
        Y-axis title
    height : int
        Figure height in pixels
    legend_position : tuple
        (x, y, xanchor, yanchor) for legend placement
    show_legend : bool
        Whether to show legend

    Returns
    -------
    go.Figure
        Styled figure
    """
    legend_x, legend_y, xanchor, yanchor = legend_position

    layout = _get_base_layout(title, height, show_legend)
    layout["legend"] = _get_legend_style(legend_x, legend_y, xanchor, yanchor)
    layout["xaxis"] = _get_axis_style(x_title)
    layout["yaxis"] = _get_axis_style(y_title)

    fig.update_layout(**layout)
    return fig


def get_study_color(index: int) -> str:
    """
    Get a consistent color for a study/material based on its index.

    Parameters
    ----------
    index : int
        Index of the study/material

    Returns
    -------
    str
        Hex color string
    """
    return STUDY_COLORS[index % len(STUDY_COLORS)]


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """
    Convert hex color to rgba format for Plotly compatibility.

    Args:
        hex_color: Hex color string (e.g., '#2E86AB')
        alpha: Alpha/opacity value from 0.0 to 1.0

    Returns:
        RGBA string (e.g., 'rgba(46, 134, 171, 0.25)')
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


# =============================================================================
# MARKER STYLES
# =============================================================================

MARKERS = {
    "experimental": {
        "size": 12,
        "color": COLORS["experimental"],
        "line": {"width": 1.5, "color": COLORS["experimental_edge"]},
        "symbol": "circle",
    },
    "experimental_small": {
        "size": 8,
        "color": COLORS["experimental"],
        "line": {"width": 1.0, "color": COLORS["experimental_edge"]},
        "symbol": "circle",
    },
    "comparison": {
        "size": 10,
        "color": COLORS["fit_secondary"],
        "line": {"width": 0},
        "symbol": "diamond",
    },
}


# =============================================================================
# AXIS CONFIGURATION (for borders)
# =============================================================================

AXIS_STYLE = {
    "showgrid": False,
    "gridwidth": 1,
    "gridcolor": COLORS["grid"],
    "showline": True,
    "linewidth": 2,
    "linecolor": "black",
    "zeroline": False,
    "mirror": True,
    "ticks": "outside",
    "tickfont": {"size": 11, "family": FONT_FAMILY, "color": COLORS["tick_text"]},
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _determine_legend_position(
    x_data: np.ndarray, y_data: np.ndarray, y_fit: np.ndarray
) -> tuple[float, float]:
    """
    Determine optimal legend position based on data distribution.

    For typical isotherm/kinetic curves that start low-left and go high-right,
    place legend in upper-left. For other patterns, adjust accordingly.

    Returns (x, y) position in paper coordinates (0-1 range)
    """
    # Check if curve is "typical" (increasing, saturating)
    y_start = np.mean(y_fit[: len(y_fit) // 4]) if len(y_fit) > 4 else y_fit[0]
    y_end = np.mean(y_fit[-len(y_fit) // 4 :]) if len(y_fit) > 4 else y_fit[-1]

    if y_end > y_start:
        # Typical saturating curve - legend in upper left
        return 0.02, 0.98
    else:
        # Decreasing curve - legend in upper right
        return 0.98, 0.98


def style_ci_traces(model_name: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Get trace styling for confidence interval (upper, lower).

    Parameters
    ----------
    model_name : str, optional
        Model name (currently unused, reserved for future customization)

    Returns
    -------
    tuple
        (upper_line_style, lower_line_style)
    """
    upper: dict[str, Any] = {
        "mode": "lines",
        "name": "95% CI Upper",
        "line": {"color": COLORS["ci_line"], "width": 1, "dash": "dot"},
        "showlegend": False,
        "hoverinfo": "skip",
    }

    lower = {
        "mode": "lines",
        "name": "95% CI Lower",
        "line": {"color": COLORS["ci_line"], "width": 1, "dash": "dot"},
        "fill": "tonexty",
        "fillcolor": COLORS["ci_fill"],
        "showlegend": False,
        "hoverinfo": "skip",
    }

    return upper, lower


# =============================================================================
# MAIN PLOT CREATION FUNCTIONS
# =============================================================================


def create_isotherm_plot(
    Ce: np.ndarray,
    qe: np.ndarray,
    Ce_fit: np.ndarray,
    qe_fit: np.ndarray,
    model_name: str = "Model",
    r_squared: float | None = None,
    title: str | None = None,
    ci_upper: np.ndarray | None = None,
    ci_lower: np.ndarray | None = None,
    height: int = 450,
) -> go.Figure:
    """
    Create a publication-quality isotherm plot with legend inside the frame.

    Parameters
    ----------
    Ce : np.ndarray
        Experimental equilibrium concentrations
    qe : np.ndarray
        Experimental adsorption capacities
    Ce_fit : np.ndarray
        Concentration values for fitted curve
    qe_fit : np.ndarray
        Fitted adsorption capacity values
    model_name : str
        Name of the model
    r_squared : float, optional
        R² value
    title : str, optional
        Custom title (default: auto-generated)
    ci_upper, ci_lower : np.ndarray, optional
        Confidence interval bounds
    height : int
        Figure height

    Returns
    -------
    go.Figure
        Styled isotherm plot
    """
    fig = go.Figure()

    # Add confidence interval first (so it's behind other traces)
    if ci_upper is not None and ci_lower is not None:
        upper_style, lower_style = style_ci_traces(model_name)
        fig.add_trace(go.Scatter(x=Ce_fit, y=ci_upper, **upper_style))
        fig.add_trace(go.Scatter(x=Ce_fit, y=ci_lower, **lower_style))

    # Add fitted curve FIRST (so it appears first in legend)
    fit_kwargs = style_fit_trace(model_name, r_squared, is_primary=True)
    fit_kwargs["legendrank"] = 2
    fig.add_trace(go.Scatter(x=Ce_fit, y=qe_fit, **fit_kwargs))

    # Add experimental data SECOND (on top visually, second in legend)
    exp_kwargs = style_experimental_trace(name="Experimental")
    exp_kwargs["marker"] = MARKERS["experimental"]
    exp_kwargs["hovertemplate"] = "Ce: %{x:.2f}<br>qe: %{y:.2f}<extra></extra>"
    exp_kwargs["legendrank"] = 1
    fig.add_trace(go.Scatter(x=Ce, y=qe, **exp_kwargs))

    # Generate title
    if title is None:
        if r_squared is not None:
            title = f"{model_name} Isotherm (R² = {r_squared:.4f})"
        else:
            title = f"{model_name} Isotherm"

    # Determine legend position
    legend_x, legend_y = _determine_legend_position(Ce, qe, qe_fit)
    xanchor = "left" if legend_x < 0.5 else "right"
    yanchor = "top" if legend_y > 0.5 else "bottom"

    # Apply standard professional layout
    apply_standard_layout(
        fig,
        title=title,
        x_title="C<sub>e</sub> (mg/L)",
        y_title="q<sub>e</sub> (mg/g)",
        height=height,
        legend_position=(legend_x, legend_y, xanchor, yanchor),
    )

    # Publication-friendly axes (start at 0 where possible)
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")

    return fig


def create_kinetic_plot(
    t: np.ndarray,
    qt: np.ndarray,
    t_fit: np.ndarray,
    qt_fit: np.ndarray,
    model_name: str = "Model",
    r_squared: float | None = None,
    title: str | None = None,
    ci_upper: np.ndarray | None = None,
    ci_lower: np.ndarray | None = None,
    height: int = 450,
) -> go.Figure:
    """
    Create a publication-quality kinetic plot with legend inside the frame.

    Parameters
    ----------
    t : np.ndarray
        Experimental time values
    qt : np.ndarray
        Experimental adsorption capacities at time t
    t_fit : np.ndarray
        Time values for fitted curve
    qt_fit : np.ndarray
        Fitted adsorption capacity values
    model_name : str
        Name of the kinetic model
    r_squared : float, optional
        R² value
    title : str, optional
        Custom title
    ci_upper, ci_lower : np.ndarray, optional
        Confidence interval bounds
    height : int
        Figure height

    Returns
    -------
    go.Figure
        Styled kinetic plot
    """
    fig = go.Figure()

    # Add confidence interval first
    if ci_upper is not None and ci_lower is not None:
        upper_style, lower_style = style_ci_traces(model_name)
        fig.add_trace(go.Scatter(x=t_fit, y=ci_upper, **upper_style))
        fig.add_trace(go.Scatter(x=t_fit, y=ci_lower, **lower_style))

    # Add fitted curve FIRST (so it appears first in legend)
    fit_kwargs = style_fit_trace(model_name, r_squared, is_primary=True)
    fit_kwargs["legendrank"] = 2
    fig.add_trace(go.Scatter(x=t_fit, y=qt_fit, **fit_kwargs))

    # Add experimental data SECOND
    exp_kwargs = style_experimental_trace(name="Experimental")
    exp_kwargs["marker"] = MARKERS["experimental"]
    exp_kwargs["hovertemplate"] = "t: %{x:.1f} min<br>qt: %{y:.2f}<extra></extra>"
    exp_kwargs["legendrank"] = 1
    fig.add_trace(go.Scatter(x=t, y=qt, **exp_kwargs))

    # Generate title
    if title is None:
        if r_squared is not None:
            title = f"{model_name} Kinetic Model (R² = {r_squared:.4f})"
        else:
            title = f"{model_name} Kinetic Model"

    # For kinetic plots, legend typically goes in lower-right (plateau region)
    legend_x, legend_y = 0.98, 0.02
    xanchor, yanchor = "right", "bottom"

    # Apply standard professional layout
    apply_standard_layout(
        fig,
        title=title,
        x_title="Time (min)",
        y_title="q<sub>t</sub> (mg/g)",
        height=height,
        legend_position=(legend_x, legend_y, xanchor, yanchor),
    )

    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")

    return fig


def create_model_comparison_plot(
    x_exp: np.ndarray,
    y_exp: np.ndarray,
    fitted_models: dict[str, dict[str, Any]],
    model_functions: dict[str, Callable],
    x_label: str = "C<sub>e</sub> (mg/L)",
    y_label: str = "q<sub>e</sub> (mg/g)",
    title: str = "Model Comparison",
    height: int = 500,
) -> go.Figure:
    """
    Create a comparison plot with multiple model fits.

    Works for both isotherm (Ce vs qe) and kinetic (t vs qt) data.

    Parameters
    ----------
    x_exp : np.ndarray
        Experimental x values (Ce for isotherm, time for kinetic)
    y_exp : np.ndarray
        Experimental y values (qe for isotherm, qt for kinetic)
    fitted_models : dict
        Dictionary of fitted model results with structure:
        {'ModelName': {'params': dict, 'r_squared': float, 'converged': bool}}
    model_functions : dict
        Dictionary mapping model names to prediction functions:
        {'ModelName': lambda x, params: model_func(x, params['p1'], params['p2'])}
    x_label : str
        X-axis label (supports HTML subscripts)
    y_label : str
        Y-axis label (supports HTML subscripts)
    title : str
        Plot title
    height : int
        Figure height in pixels

    Returns
    -------
    go.Figure
        Multi-model comparison plot with professional styling

    Examples
    --------
    Isotherm comparison:
    >>> model_functions = {
    ...     'Langmuir': lambda x, p: langmuir_model(x, p['qm'], p['KL']),
    ...     'Freundlich': lambda x, p: freundlich_model(x, p['KF'], p['n_inv']),
    ... }
    >>> fig = create_model_comparison_plot(Ce, qe, fitted_models, model_functions,
    ...                                    x_label='C<sub>e</sub> (mg/L)',
    ...                                    y_label='q<sub>e</sub> (mg/g)')

    Kinetic comparison:
    >>> model_functions = {
    ...     'PFO': lambda t, p: pfo_model(t, p['qe'], p['k1']),
    ...     'PSO': lambda t, p: pso_model(t, p['qe'], p['k2']),
    ... }
    >>> fig = create_model_comparison_plot(t, qt, fitted_models, model_functions,
    ...                                    x_label='Time (min)',
    ...                                    y_label='q<sub>t</sub> (mg/g)')
    """
    fig = go.Figure()

    # Determine "best" model (used for solid line; others dashed)
    best_model = None
    candidates = []
    for name, r in (fitted_models or {}).items():
        if not r or not r.get("converged"):
            continue
        aic = r.get("aicc", r.get("aic"))
        adjr2 = r.get("adj_r_squared")
        r2 = r.get("r_squared")
        candidates.append((name, aic, adjr2, r2))

    # Prefer lowest AICc/AIC if available, else highest Adj-R², else highest R²
    aic_vals = [c for c in candidates if c[1] is not None]
    if aic_vals:
        best_model = min(aic_vals, key=lambda t: t[1])[0]
    else:
        adj_vals = [c for c in candidates if c[2] is not None]
        if adj_vals:
            best_model = max(adj_vals, key=lambda t: t[2])[0]
        else:
            r2_vals = [c for c in candidates if c[3] is not None]
            if r2_vals:
                best_model = max(r2_vals, key=lambda t: t[3])[0]

    # Smooth x for fit curves
    x_min = max(0.0, float(np.min(x_exp)) * 0.9)
    x_max = float(np.max(x_exp)) * 1.05
    x_line = np.linspace(x_min, x_max, 200)

    # Add model fits (lines first, markers last so experimental stays on top)
    for model_name, results in (fitted_models or {}).items():
        if not results or not results.get("converged"):
            continue
        if model_name not in model_functions:
            continue

        params = results.get("params", {})
        try:
            y_fit = model_functions[model_name](x_line, params)
        except Exception as e:
            logger.debug(f"Could not plot {model_name} curve: {e}")
            continue

        color = MODEL_COLORS.get(model_name, COLORS["model_colors"][len(fig.data) % len(COLORS["model_colors"])])
        r2 = results.get("r_squared")

        legend_name = f"{model_name} (R²={r2:.4f})" if isinstance(r2, (int, float)) else model_name

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_fit,
                mode="lines",
                name=legend_name,
                legendrank=2,
                line={
                    "color": color,
                    "width": 2.2,
                    "dash": "solid" if (best_model and model_name == best_model) else "dash",
                },
                hovertemplate=f"{model_name}: %{{y:.2f}}<extra></extra>",
            )
        )

    # Experimental data (on top visually, first in legend)
    exp_kwargs = style_experimental_trace(name="Experimental")
    exp_kwargs["marker"] = MARKERS["experimental"]
    exp_kwargs["legendrank"] = 1
    exp_kwargs["hovertemplate"] = "%{x:.2f}, %{y:.2f}<extra></extra>"
    fig.add_trace(go.Scatter(x=x_exp, y=y_exp, **exp_kwargs))

    # Apply standard professional layout (case-study legend placement)
    apply_standard_layout(
        fig,
        title=title,
        x_title=x_label,
        y_title=y_label,
        height=height,
        legend_position=(0.02, 0.98, "left", "top"),
    )

    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")

    return fig



def create_residual_plot(
    y_pred: np.ndarray, residuals: np.ndarray, model_name: str = "Model", height: int = 350
) -> go.Figure:
    """
    Create a residual analysis plot.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values
    residuals : np.ndarray
        Residuals (observed - predicted)
    model_name : str
        Model name for title
    height : int
        Figure height

    Returns
    -------
    go.Figure
        Residual plot
    """
    fig = go.Figure()

    # Color points by sign
    colors = [
        COLORS["residual_positive"] if r >= 0 else COLORS["residual_negative"] for r in residuals
    ]

    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            name="Residuals",
            marker={
                "size": 10,
                "color": colors,
                "line": {"width": 1, "color": "#424242"},
            },
            hovertemplate="Predicted: %{x:.2f}<br>Residual: %{y:.3f}<extra></extra>",
        )
    )

    # Add zero line
    fig.add_hline(
        y=0,
        line={"color": COLORS["residual_line"], "width": 2, "dash": "dash"},
    )

    apply_standard_layout(
        fig,
        title=f"{model_name} Residuals",
        x_title="Predicted Values",
        y_title="Residuals",
        height=height,
        show_legend=False,
    )

    return fig


def create_parity_plot(
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    r_squared: float | None = None,
    rmse: float | None = None,
    height: int = 400,
) -> go.Figure:
    """
    Create a parity (observed vs predicted) plot.

    Parameters
    ----------
    y_obs : np.ndarray
        Observed values
    y_pred : np.ndarray
        Predicted values
    model_name : str
        Model name for title
    r_squared : float, optional
        R² value to display in title
    rmse : float, optional
        RMSE value to display in title
    height : int
        Figure height

    Returns
    -------
    go.Figure
        Parity plot with 1:1 line
    """
    fig = go.Figure()

    # Add data points
    fig.add_trace(
        go.Scatter(
            x=y_obs,
            y=y_pred,
            mode="markers",
            name="Data",
            marker=MARKERS["experimental"],
            hovertemplate="Observed: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
        )
    )

    # Add 1:1 line
    min_val = min(min(y_obs), min(y_pred))
    max_val = max(max(y_obs), max(y_pred))
    margin = (max_val - min_val) * 0.05

    fig.add_trace(
        go.Scatter(
            x=[min_val - margin, max_val + margin],
            y=[min_val - margin, max_val + margin],
            mode="lines",
            name="1:1 Line",
            line={"color": COLORS["fit_primary"], "width": 2, "dash": "dash"},
        )
    )

    # Build title with optional metrics
    title = f"{model_name} - Parity Plot"
    if r_squared is not None or rmse is not None:
        metrics = []
        if r_squared is not None:
            metrics.append(f"R²={r_squared:.4f}")
        if rmse is not None:
            metrics.append(f"RMSE={rmse:.4f}")
        title = f"{model_name} | {', '.join(metrics)}"

    # Apply standard professional layout
    apply_standard_layout(
        fig, title=title, x_title="Observed", y_title="Predicted", height=height, show_legend=True
    )

    # Add custom axis ranges for parity plot
    fig.update_xaxes(range=[min_val - margin, max_val + margin])
    fig.update_yaxes(range=[min_val - margin, max_val + margin], scaleanchor="x")

    return fig


# =============================================================================
# STYLE APPLICATION FUNCTIONS
# =============================================================================


def style_study_trace(
    study_idx: int,
    name: str,
    marker_size: int = 10,
    mode: str = "markers+lines",
    dash: str = "solid",
) -> dict:
    """Consistent multi-study trace styling (house style)."""
    c = get_study_color(study_idx)
    return {
        "mode": mode,
        "name": name,
        "marker": {
            "size": marker_size,
            "color": c,
            "symbol": "circle",
            "line": {"width": 1.0, "color": "#000000"},
        },
        "line": {"color": c, "width": 2.5, "dash": dash},
    }


def create_effect_plot(
    x,
    y,
    title: str,
    x_title: str,
    y_title: str,
    height: int = 500,
    series_name: str = "Effect",
    show_legend: bool = False,
    x_tozero: bool = False,
    y_tozero: bool = True,
    hovertemplate: str | None = None,
) -> go.Figure:
    """Single-series effect plot (pH / temperature / dosage) with house style."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    msk = np.isfinite(x) & np.isfinite(y)
    x = x[msk]
    y = y[msk]

    fig = go.Figure()

    tr = style_experimental_trace(name=series_name)
    tr["mode"] = "markers+lines"
    tr["marker"] = MARKERS["experimental"]
    tr["line"] = {"width": 2.5, "color": COLORS["experimental"]}
    tr["hovertemplate"] = hovertemplate or "%{x:.3f}<br>%{y:.3f}<extra></extra>"
    tr["legendrank"] = 1
    fig.add_trace(go.Scatter(x=x, y=y, **tr))

    fig = apply_professional_style(
        fig,
        title=title,
        x_title=x_title,
        y_title=y_title,
        height=height,
        show_legend=show_legend,
    )

    if x_tozero:
        fig.update_xaxes(rangemode="tozero")
    if y_tozero:
        fig.update_yaxes(rangemode="tozero")

    return fig


def create_vant_hoff_plot(
    invT: np.ndarray,
    lnKd: np.ndarray,
    slope: float,
    intercept: float,
    r_squared: float,
    title: str,
    height: int = 450,
) -> go.Figure:
    """Van't Hoff plot in house style."""
    invT = np.asarray(invT, dtype=float)
    lnKd = np.asarray(lnKd, dtype=float)
    msk = np.isfinite(invT) & np.isfinite(lnKd)
    invT = invT[msk]
    lnKd = lnKd[msk]

    invT_line = np.linspace(invT.min() * 0.98, invT.max() * 1.02, 200)
    y_line = slope * invT_line + intercept

    fig = go.Figure()

    exp = style_experimental_trace(name="Experimental")
    exp["marker"] = MARKERS["experimental"]
    exp["hovertemplate"] = "1000/T: %{x:.3f}<br>ln(Kd): %{y:.3f}<extra></extra>"
    exp["legendrank"] = 1
    fig.add_trace(go.Scatter(x=invT * 1000.0, y=lnKd, **exp))

    fig.add_trace(
        go.Scatter(
            x=invT_line * 1000.0,
            y=y_line,
            mode="lines",
            name=f"Linear Fit (R² = {float(r_squared):.4f})",
            line={"color": COLORS["fit_primary"], "width": 2.5},
            legendrank=2,
        )
    )

    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"y = {slope:.2f}x + {intercept:.4f}<br>R² = {float(r_squared):.4f}",
        showarrow=False,
        font={"size": 12, "family": FONT_FAMILY},
        bgcolor="rgba(255, 255, 255, 0.90)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
    )

    fig = apply_professional_style(
        fig,
        title=title,
        x_title="1000/T (K⁻¹)",
        y_title="ln(Kd)",
        height=height,
        show_legend=True,
        legend_position="upper left",
    )
    return fig


def create_dual_axis_effect_plot(
    x,
    y1,
    y2,
    title: str,
    x_title: str,
    y1_title: str,
    y2_title: str,
    y1_name: str = "qe (mg/g)",
    y2_name: str = "Removal (%)",
    height: int = 450,
    x_tozero: bool = False,
    y1_tozero: bool = True,
    y2_tozero: bool = True,
) -> go.Figure:
    """Dual-axis effect plot (qe left, removal right) in house style."""
    x = np.asarray(x, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float) if y2 is not None else None

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    tr1 = style_experimental_trace(name=y1_name)
    tr1["mode"] = "markers+lines"
    tr1["marker"] = MARKERS["experimental"]
    tr1["line"] = {"width": 2.5, "color": COLORS["experimental"]}
    tr1["legendrank"] = 1
    fig.add_trace(go.Scatter(x=x, y=y1, **tr1), secondary_y=False)

    if y2 is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y2,
                mode="markers+lines",
                name=y2_name,
                marker={
                    "size": 10,
                    "color": COLORS["fit_secondary"],
                    "symbol": "circle",
                    "line": {"width": 1.0, "color": "#000000"},
                },
                line={"width": 2.5, "color": COLORS["fit_secondary"], "dash": "dash"},
                legendrank=2,
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title={"text": f"<b>{title}</b>", "font": {"size": 16, "family": FONT_FAMILY}},
        height=height,
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font={"family": FONT_FAMILY, "size": 12},
        margin={"l": 70, "r": 70, "t": 60, "b": 60},
        legend=get_legend_style(0.02, 0.98),
    )
    fig.update_xaxes(**get_axis_style(x_title))
    fig.update_yaxes(**get_axis_style(y1_title), secondary_y=False)
    fig.update_yaxes(**{**get_axis_style(y2_title), "showgrid": False}, secondary_y=True)

    if x_tozero:
        fig.update_xaxes(rangemode="tozero")
    if y1_tozero:
        fig.update_yaxes(rangemode="tozero", secondary_y=False)
    if y2_tozero and y2 is not None:
        fig.update_yaxes(rangemode="tozero", secondary_y=True)

    return fig

def apply_professional_style(
    fig: go.Figure,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int = 450,
    show_legend: bool = True,
    legend_position: str = "upper left",
    legend_horizontal: bool = False,
    barmode: str | None = None,
) -> go.Figure:
    """
    Apply professional styling to an existing Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        Existing Plotly figure to style
    title : str, optional
        Override title
    x_title : str, optional
        Override x-axis title
    y_title : str, optional
        Override y-axis title
    height : int
        Figure height in pixels
    show_legend : bool
        Whether to show legend
    legend_position : str
        Legend position: 'upper left', 'upper right', 'lower left', 'lower right'
    legend_horizontal : bool
        If True, places legend horizontally above the plot (overrides legend_position)
    barmode : str, optional
        Bar mode for bar charts ('group', 'stack', 'overlay', 'relative')

    Returns
    -------
    go.Figure
        Styled figure
    """
    # Map position names to coordinates
    position_map = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }

    x, y, xanchor, yanchor = position_map.get(legend_position, (0.02, 0.98, "left", "top"))

    layout_update: dict[str, Any] = {
        "height": height,
        "showlegend": show_legend,
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "font": {"family": FONT_FAMILY, "size": 12},
        "margin": {"l": 70, "r": 40, "t": 60, "b": 60},
        "xaxis": AXIS_STYLE.copy(),
        "yaxis": AXIS_STYLE.copy(),
    }

    # Set legend style based on horizontal flag
    if legend_horizontal:
        layout_update["legend"] = {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "black",
            "borderwidth": 1,
            "font": {"size": 11, "family": FONT_FAMILY},
        }
    else:
        layout_update["legend"] = {
            "x": x,
            "y": y,
            "xanchor": xanchor,
            "yanchor": yanchor,
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "black",
            "borderwidth": 1,
            "font": {"size": 11, "family": FONT_FAMILY},
        }

    # Add barmode if specified
    if barmode:
        layout_update["barmode"] = barmode

    if title:
        layout_update["title"] = {
            "text": f"<b>{title}</b>",
            "font": {"size": 16, "family": FONT_FAMILY},
        }

    if x_title:
        layout_update["xaxis"]["title"] = {
            "text": x_title,
            "font": {"size": 14, "family": FONT_FAMILY},
        }

    if y_title:
        layout_update["yaxis"]["title"] = {
            "text": y_title,
            "font": {"size": 14, "family": FONT_FAMILY},
        }

    fig.update_layout(template=PLOT_TEMPLATE, **layout_update)

    # Normalize bar traces (professional outlines + unclipped labels)
    for tr in fig.data:
        if getattr(tr, "type", "") == "bar":
            try:
                tr.update(marker=dict(line=dict(color="#000000", width=1.0)))
                tr.update(cliponaxis=False)
                if getattr(tr, "text", None) is not None and getattr(tr, "textfont", None) is None:
                    tr.update(textfont={"size": 12, "family": FONT_FAMILY})
            except Exception:
                pass

    return fig


# =============================================================================
# EXPORT SCALING
# =============================================================================


def prepare_figure_for_export(
    fig: go.Figure,
    scale: float,
    target_width_inches: float = 6.5,
) -> go.Figure:
    """
    Scale all visual elements so the figure looks professional when exported
    at high resolution and printed/inserted into a document.

    **Problem**: Plotly's ``to_image(scale=N)`` enlarges the raster canvas by
    *N*× but keeps font sizes, line widths, and marker sizes at their original
    CSS-pixel values.  When the resulting image is inserted into a Word/LaTeX
    document at journal column width (~6.5 in), text shrinks to unreadable
    sizes (e.g. a 14 px axis title → ~5.5 pt printed).

    **Solution**: Before calling ``to_image``, multiply every size property by
    *scale* so that text and graphical elements maintain the same physical
    proportions they had on screen.

    Parameters
    ----------
    fig : go.Figure
        The figure to prepare.  A **deep copy** is returned; the original is
        unchanged so the interactive Streamlit view is not affected.
    scale : float
        The same scale value that will be passed to ``fig.to_image(scale=…)``.
    target_width_inches : float
        Intended print width.  Used only for the informational DPI
        calculation in the returned figure's metadata; does not change
        the image dimensions.

    Returns
    -------
    go.Figure
        A new figure with all font sizes, line widths, marker sizes, and
        annotation sizes multiplied by *scale*.

    Notes
    -----
    Call this **once** right before ``to_image`` and pass the returned figure
    to ``to_image``.  Do not apply it to figures that will be displayed on
    screen — they would look oversized in the Streamlit app.
    """
    import copy

    fig = copy.deepcopy(fig)

    if scale <= 1.0:
        return fig  # nothing to do for screen-resolution exports

    # ── Helper: scale a numeric value if present ──────────────────────
    def _scaled(val: float | int | None) -> float | int | None:
        if val is None:
            return None
        return round(val * scale, 1)

    def _scale_font(font_dict: dict | None) -> dict | None:
        """Scale the 'size' key inside a Plotly font dict."""
        if font_dict is None:
            return None
        d = dict(font_dict)
        if "size" in d and d["size"] is not None:
            d["size"] = _scaled(d["size"])
        return d

    def _scale_line(line_dict: dict | None) -> dict | None:
        """Scale the 'width' key inside a Plotly line dict."""
        if line_dict is None:
            return None
        d = dict(line_dict)
        if "width" in d and d["width"] is not None:
            d["width"] = _scaled(d["width"])
        return d

    # ── 1. Layout-level fonts ─────────────────────────────────────────
    layout = fig.layout

    # Global base font
    if layout.font and layout.font.size:
        fig.update_layout(font={"size": _scaled(layout.font.size)})

    # Title
    if layout.title and layout.title.font and layout.title.font.size:
        fig.update_layout(title_font_size=_scaled(layout.title.font.size))

    # ── 2. Axis fonts and line widths ─────────────────────────────────
    for ax_attr in ("xaxis", "yaxis", "xaxis2", "yaxis2", "xaxis3", "yaxis3"):
        ax = getattr(layout, ax_attr, None)
        if ax is None:
            continue

        updates: dict[str, Any] = {}

        # Axis title font
        if ax.title and ax.title.font and ax.title.font.size:
            updates["title_font_size"] = _scaled(ax.title.font.size)

        # Tick label font
        if ax.tickfont and ax.tickfont.size:
            updates["tickfont_size"] = _scaled(ax.tickfont.size)

        # Axis border line
        if ax.linewidth:
            updates["linewidth"] = _scaled(ax.linewidth)

        # Grid line width
        if ax.gridwidth:
            updates["gridwidth"] = _scaled(ax.gridwidth)

        # Tick length
        if ax.ticklen:
            updates["ticklen"] = _scaled(ax.ticklen)

        if updates:
            fig.update_layout(**{ax_attr: updates})

    # ── 3. Legend font ────────────────────────────────────────────────
    if layout.legend and layout.legend.font and layout.legend.font.size:
        fig.update_layout(legend_font_size=_scaled(layout.legend.font.size))
    if layout.legend and layout.legend.borderwidth:
        fig.update_layout(legend_borderwidth=_scaled(layout.legend.borderwidth))

    # ── 4. Margins (keep proportional to text) ────────────────────────
    if layout.margin:
        m = layout.margin
        fig.update_layout(
            margin={
                "l": _scaled(m.l) if m.l else m.l,
                "r": _scaled(m.r) if m.r else m.r,
                "t": _scaled(m.t) if m.t else m.t,
                "b": _scaled(m.b) if m.b else m.b,
            }
        )

    # ── 5. Annotations ───────────────────────────────────────────────
    for ann in fig.layout.annotations:
        if ann.font and ann.font.size:
            ann.font.size = _scaled(ann.font.size)
        if ann.borderwidth:
            ann.borderwidth = _scaled(ann.borderwidth)
        if ann.borderpad:
            ann.borderpad = _scaled(ann.borderpad)

    # ── 6. Trace-level: markers, lines, text ──────────────────────────
    for trace in fig.data:
        # Line width
        if hasattr(trace, "line") and trace.line and trace.line.width:
            trace.line.width = _scaled(trace.line.width)

        # Marker size and outline
        if hasattr(trace, "marker") and trace.marker:
            if trace.marker.size is not None:
                trace.marker.size = _scaled(trace.marker.size)
            if trace.marker.line and trace.marker.line.width:
                trace.marker.line.width = _scaled(trace.marker.line.width)
            # Colorbar fonts (3D surface/scatter) — Plotly 5.x API
            try:
                cb = trace.marker.colorbar
                if cb and cb.title and cb.title.font and cb.title.font.size:
                    cb.title.font.size = _scaled(cb.title.font.size)
                if cb and cb.tickfont and cb.tickfont.size:
                    cb.tickfont.size = _scaled(cb.tickfont.size)
            except Exception:
                pass

        # Text font on traces (bar labels, etc.)
        if hasattr(trace, "textfont") and trace.textfont and trace.textfont.size:
            trace.textfont.size = _scaled(trace.textfont.size)

        # Error bars
        if hasattr(trace, "error_y") and trace.error_y and trace.error_y.thickness:
            trace.error_y.thickness = _scaled(trace.error_y.thickness)
        if hasattr(trace, "error_x") and trace.error_x and trace.error_x.thickness:
            trace.error_x.thickness = _scaled(trace.error_x.thickness)

        # Bar outline
        if hasattr(trace, "marker") and trace.marker:
            if hasattr(trace.marker, "line") and trace.marker.line:
                if trace.marker.line.width:
                    # Already handled above, but bar-specific traces sometimes
                    # have marker.line set separately from the scatter line.
                    pass

    # ── 7. 3D scene axes (if present) ─────────────────────────────────
    if layout.scene:
        scene_updates: dict[str, Any] = {}
        for ax_name in ("xaxis", "yaxis", "zaxis"):
            ax = getattr(layout.scene, ax_name, None)
            if ax is None:
                continue
            ax_up: dict[str, Any] = {}
            if ax.title and ax.title.font and ax.title.font.size:
                ax_up["title_font_size"] = _scaled(ax.title.font.size)
            if ax.tickfont and ax.tickfont.size:
                ax_up["tickfont_size"] = _scaled(ax.tickfont.size)
            if ax.linewidth:
                ax_up["linewidth"] = _scaled(ax.linewidth)
            if ax_up:
                scene_updates[ax_name] = ax_up
        if scene_updates:
            fig.update_layout(scene=scene_updates)

    # ── 8. Polar axes (radar charts) ──────────────────────────────────
    if layout.polar:
        polar_updates: dict[str, Any] = {}
        if layout.polar.radialaxis:
            ra = layout.polar.radialaxis
            ra_up: dict[str, Any] = {}
            if ra.tickfont and ra.tickfont.size:
                ra_up["tickfont_size"] = _scaled(ra.tickfont.size)
            if ra.title and ra.title.font and ra.title.font.size:
                ra_up["title_font_size"] = _scaled(ra.title.font.size)
            if ra_up:
                polar_updates["radialaxis"] = ra_up
        if layout.polar.angularaxis:
            aa = layout.polar.angularaxis
            aa_up: dict[str, Any] = {}
            if aa.tickfont and aa.tickfont.size:
                aa_up["tickfont_size"] = _scaled(aa.tickfont.size)
            if aa_up:
                polar_updates["angularaxis"] = aa_up
        if polar_updates:
            fig.update_layout(polar=polar_updates)

    return fig


def apply_professional_polar_style(
    fig: go.Figure,
    title: str | None = None,
    height: int = 500,
    show_legend: bool = True,
    legend_position: str = "upper left",
    radial_range: tuple[float, float] = (0.0, 1.0),
) -> go.Figure:
    """Apply the AdsorbLab Pro house style to polar/radar charts."""
    position_map = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, xanchor, yanchor = position_map.get(legend_position, (0.02, 0.98, "left", "top"))

    layout_update: dict[str, Any] = {
        "height": height,
        "showlegend": show_legend,
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "font": {"family": FONT_FAMILY, "size": 12},
        "margin": {"l": 70, "r": 40, "t": 60, "b": 60},
        "legend": {
            "x": x,
            "y": y,
            "xanchor": xanchor,
            "yanchor": yanchor,
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "black",
            "borderwidth": 1,
            "font": {"size": 11, "family": FONT_FAMILY},
        },
        "polar": {
            "bgcolor": "white",
            "radialaxis": {
                "visible": True,
                "range": list(radial_range),
                "showgrid": False,
                "gridcolor": COLORS["grid"],
                "showline": True,
                "linewidth": 2,
                "linecolor": "black",
                "ticks": "outside",
                "tickfont": {"size": 11, "family": FONT_FAMILY, "color": COLORS["tick_text"]},
            },
            "angularaxis": {
                "showgrid": False,
                "gridcolor": COLORS["grid"],
                "showline": True,
                "linewidth": 2,
                "linecolor": "black",
                "ticks": "outside",
                "tickfont": {"size": 11, "family": FONT_FAMILY, "color": COLORS["tick_text"]},
            },
        },
    }

    if title:
        layout_update["title"] = {
            "text": f"<b>{title}</b>",
            "font": {"size": 16, "family": FONT_FAMILY},
        }

    fig.update_layout(**layout_update)
    return fig

def style_experimental_trace(name: str = "Experimental", use_small: bool = False) -> dict:
    """
    Get trace styling for experimental data points.

    Parameters
    ----------
    name : str
        Trace name for legend
    use_small : bool
        Use smaller markers

    Returns
    -------
    dict
        Keyword arguments for go.Scatter
    """
    marker_style = MARKERS["experimental_small"] if use_small else MARKERS["experimental"]
    return {
        "mode": "markers",
        "name": name,
        "marker": marker_style,
        "hovertemplate": "%{x:.2f}, %{y:.2f}<extra></extra>",
    }


def style_fit_trace(
    model_name: str, r_squared: float | None = None, is_primary: bool = True
) -> dict:
    """
    Get trace styling for fitted curve.

    Parameters
    ----------
    model_name : str
        Model name for legend and color selection
    r_squared : float, optional
        R² value to include in legend
    is_primary : bool
        Whether this is the primary (solid) or comparison (dashed) fit

    Returns
    -------
    dict
        Keyword arguments for go.Scatter
    """
    color = MODEL_COLORS.get(model_name, COLORS["fit_primary"])

    legend_name = f"{model_name} Fit"
    if r_squared is not None:
        legend_name = f"{model_name} (R²={r_squared:.4f})"

    line_style = {
        "color": color,
        "width": 2.5 if is_primary else 2,
        "dash": "solid" if is_primary else "dash",
    }

    return {
        "mode": "lines",
        "name": legend_name,
        "line": line_style,
        "hovertemplate": f"{model_name}: %{{y:.2f}}<extra></extra>",
    }


# =============================================================================
# MATPLOTLIB STYLE FOR STATIC EXPORTS
# =============================================================================

MATPLOTLIB_STYLE = {
    # Figure
    "figure.facecolor": "white",
    "figure.edgecolor": "white",
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    # Axes
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.5,
    "axes.grid": False,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.labelcolor": COLORS["tick_text"],
    # Grid
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.7,
    # Ticks
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.color": COLORS["tick_text"],
    "ytick.color": COLORS["tick_text"],
    # Legend
    "legend.fontsize": 11,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "black",
    # Lines
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
    # Scatter
    "scatter.marker": "o",
    # Font
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia"],
    "font.size": 12,
}



def apply_professional_3d_style(
    fig: go.Figure,
    title: str | None = None,
    height: int = 700,
    show_legend: bool = True,
    legend_position: str = "upper left",
    camera_eye: dict[str, float] | None = None,
    margin: dict[str, int] | None = None,
) -> go.Figure:
    """Apply house style to 3D figures (scene-based plots)."""
    position_map = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, xanchor, yanchor = position_map.get(legend_position, (0.02, 0.98, "left", "top"))

    # Preserve existing scene content (axis titles, ticktext/tickvals, etc.)
    scene = fig.layout.scene.to_plotly_json() if getattr(fig.layout, "scene", None) else {}

    def _norm_axis(ax: dict | None) -> dict:
        ax = dict(ax) if ax else {}
        # Preserve title text if present
        title_obj = ax.get("title", {}) or {}
        if isinstance(title_obj, str):
            title_obj = {"text": title_obj}
        title_obj.setdefault("font", {"size": 12, "family": FONT_FAMILY})
        ax["title"] = title_obj

        ax.setdefault("showbackground", True)
        ax.setdefault("backgroundcolor", "white")
        ax["gridcolor"] = COLORS["grid"]
        ax["showline"] = True
        ax["linecolor"] = "black"
        ax["linewidth"] = 2
        ax["ticks"] = "outside"
        ax["tickfont"] = {"size": 11, "family": FONT_FAMILY, "color": COLORS["tick_text"]}
        ax.setdefault("zeroline", False)
        return ax

    scene["xaxis"] = _norm_axis(scene.get("xaxis"))
    scene["yaxis"] = _norm_axis(scene.get("yaxis"))
    scene["zaxis"] = _norm_axis(scene.get("zaxis"))
    scene.setdefault("bgcolor", "white")
    scene.setdefault("aspectmode", "data")

    if camera_eye is not None:
        scene["camera"] = {"eye": camera_eye}
    else:
        scene.setdefault("camera", scene.get("camera") or {"eye": {"x": 1.6, "y": 1.6, "z": 1.2}})

    if margin is None:
        margin = {"l": 0, "r": 0, "b": 0, "t": 50}

    layout_update: dict[str, Any] = {
        "height": height,
        "showlegend": show_legend,
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"family": FONT_FAMILY, "size": 12},
        "margin": margin,
        "legend": {
            "x": x,
            "y": y,
            "xanchor": xanchor,
            "yanchor": yanchor,
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "black",
            "borderwidth": 1,
            "font": {"size": 11, "family": FONT_FAMILY},
        },
        "scene": scene,
    }

    if title:
        layout_update["title"] = {"text": f"<b>{title}</b>", "font": {"size": 16, "family": FONT_FAMILY}}

    fig.update_layout(**layout_update)

    # Harmonize colorbar fonts where present
    for tr in fig.data:
        try:
            if getattr(tr, "type", "") == "surface":
                tr.update(colorbar={"title": {"font": {"family": FONT_FAMILY, "size": 12}},
                                    "tickfont": {"family": FONT_FAMILY, "size": 11}})
            elif getattr(tr, "type", "") == "scatter3d":
                tr.update(marker={"colorbar": {"title": {"font": {"family": FONT_FAMILY, "size": 12}},
                                               "tickfont": {"family": FONT_FAMILY, "size": 11}}})
        except Exception:
            pass

    return fig

def infer_figure_kind(fig: go.Figure) -> str:
    """
    Infer figure kind for styling:
    - "3d" if fig.layout.scene has been configured
    - "polar" if fig.layout.polar has been configured
    - otherwise "2d"
    """
    try:
        if fig.layout.scene.to_plotly_json():
            return "3d"
    except Exception:
        pass
    try:
        if fig.layout.polar.to_plotly_json():
            return "polar"
    except Exception:
        pass
    return "2d"


def finalize_figure(
    fig: go.Figure | dict,
    *,
    kind: str | None = None,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int | None = None,
    show_legend: bool | None = None,
    legend_position: str = "upper left",
) -> go.Figure:
    """
    Single entry point to apply the AdsorbLabPro house style to any figure:
    - Works with go.Figure or serialized dict (saved figure)
    - Auto-detects kind if not provided: 2d / polar / 3d
    - Preserves existing titles/axis labels if not provided
    """
    if isinstance(fig, dict):
        fig = go.Figure(fig)

    if kind is None:
        kind = infer_figure_kind(fig)

    # Preserve existing titles if caller doesn't provide them
    if title is None and getattr(fig.layout, "title", None) is not None:
        try:
            title = fig.layout.title.text
        except Exception:
            title = None

    if x_title is None:
        try:
            x_title = fig.layout.xaxis.title.text
        except Exception:
            x_title = None

    if y_title is None:
        try:
            y_title = fig.layout.yaxis.title.text
        except Exception:
            y_title = None

    # Default height (if not passed)
    if height is None:
        height = 700 if kind == "3d" else 500

    # Default legend behavior
    if show_legend is None:
        show_legend = True

    if kind == "3d":
        return apply_professional_3d_style(
            fig,
            title=title,
            height=height,
            show_legend=show_legend,
            legend_position=legend_position,
        )

    if kind == "polar":
        return apply_professional_polar_style(
            fig,
            title=title,
            height=height,
            show_legend=show_legend,
            legend_position=legend_position,
            radial_range=(0.0, 1.0),
        )

    # Default: 2D
    return apply_professional_style(
        fig,
        title=title or "",
        x_title=x_title or "",
        y_title=y_title or "",
        height=height,
        show_legend=show_legend,
        legend_position=legend_position,
    )


def apply_matplotlib_style() -> None:
    """Apply the professional style to matplotlib."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(MATPLOTLIB_STYLE)