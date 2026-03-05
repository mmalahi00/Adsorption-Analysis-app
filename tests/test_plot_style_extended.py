"""
Extended tests for adsorblab_pro/plot_style.py - boosting coverage.

These tests require plotly. Tests are skipped if plotly is not available.
"""

import numpy as np
import pytest

go = pytest.importorskip("plotly.graph_objects")

from adsorblab_pro.plot_style import (
    _determine_legend_position,
    _get_axis_style,
    _get_base_layout,
    _get_legend_style,
    apply_professional_3d_style,
    apply_professional_polar_style,
    apply_professional_style,
    apply_standard_layout,
    create_effect_plot,
    create_isotherm_plot,
    create_kinetic_plot,
    create_model_comparison_plot,
    create_parity_plot,
    create_residual_plot,
    create_vant_hoff_plot,
    finalize_figure,
    get_axis_style,
    get_legend_style,
    get_study_color,
    hex_to_rgba,
    infer_figure_kind,
    prepare_figure_for_export,
    style_ci_traces,
    style_experimental_trace,
    style_fit_trace,
    style_study_trace,
)


# =============================================================================
# BASIC STYLE FUNCTIONS
# =============================================================================
class TestAxisAndLegendStyles:
    def test_get_axis_style(self):
        style = get_axis_style("X Axis")
        assert isinstance(style, dict)
        assert "title" in style or "showgrid" in style

    def test_internal_axis_style(self):
        style = _get_axis_style("Y Axis")
        assert isinstance(style, dict)

    def test_get_legend_style(self):
        style = get_legend_style(0.5, 0.5)
        assert isinstance(style, dict)

    def test_internal_legend_style(self):
        style = _get_legend_style(0.1, 0.9, xanchor="left", yanchor="top")
        assert isinstance(style, dict)

    def test_get_base_layout(self):
        layout = _get_base_layout("Test Title", height=500, show_legend=True)
        assert isinstance(layout, dict)
        assert "height" in layout or "title" in layout


class TestStudyColors:
    def test_get_study_color_basic(self):
        color = get_study_color(0)
        assert isinstance(color, str)
        assert color.startswith("#") or color.startswith("rgb")

    def test_get_study_color_wraps(self):
        # Should handle indices beyond list length
        color = get_study_color(100)
        assert isinstance(color, str)

    def test_hex_to_rgba_full_opacity(self):
        result = hex_to_rgba("#FF0000", alpha=1.0)
        assert "rgba" in result
        assert "255" in result

    def test_hex_to_rgba_half_opacity(self):
        result = hex_to_rgba("#0000FF", alpha=0.5)
        assert "0.5" in result


class TestDetermineLegendPosition:
    def test_increasing_curve(self):
        y = np.array([1, 5, 10, 20, 40])
        pos = _determine_legend_position(y)
        assert isinstance(pos, dict) or isinstance(pos, str)

    def test_decreasing_curve(self):
        y = np.array([40, 20, 10, 5, 1])
        pos = _determine_legend_position(y)
        assert isinstance(pos, dict) or isinstance(pos, str)


# =============================================================================
# TRACE STYLES
# =============================================================================
class TestStyleCITraces:
    def test_default(self):
        upper, lower = style_ci_traces()
        assert isinstance(upper, dict)
        assert isinstance(lower, dict)

    def test_with_model_name(self):
        upper, lower = style_ci_traces(model_name="Langmuir")
        assert isinstance(upper, dict)


class TestStyleExperimentalTraceExtended:
    def test_default(self):
        style = style_experimental_trace()
        assert isinstance(style, dict)
        assert "name" in style or "mode" in style

    def test_custom_name(self):
        style = style_experimental_trace(name="My Data")
        assert isinstance(style, dict)

    def test_small_markers(self):
        style = style_experimental_trace(use_small=True)
        assert isinstance(style, dict)


class TestStyleFitTraceExtended:
    def test_default(self):
        style = style_fit_trace("Langmuir")
        assert isinstance(style, dict)

    def test_with_r_squared(self):
        style = style_fit_trace("Langmuir", r_squared=0.98)
        assert isinstance(style, dict)

    def test_secondary(self):
        style = style_fit_trace("Freundlich", is_secondary=True)
        assert isinstance(style, dict)


class TestStyleStudyTrace:
    def test_basic(self):
        style = style_study_trace("Study 1", 0)
        assert isinstance(style, dict)


# =============================================================================
# PLOT CREATION
# =============================================================================
class TestCreateIsothermPlotExtended:
    def test_basic_plot(self):
        Ce = np.array([1, 5, 10, 20, 50])
        qe = np.array([5, 15, 22, 30, 40])
        Ce_fit = np.linspace(1, 50, 50)
        qe_fit = 45 * 0.05 * Ce_fit / (1 + 0.05 * Ce_fit)
        fig = create_isotherm_plot(Ce, qe, Ce_fit, qe_fit, model_name="Langmuir")
        assert isinstance(fig, go.Figure)

    def test_with_r_squared(self):
        Ce = np.array([1, 5, 10, 20, 50])
        qe = np.array([5, 15, 22, 30, 40])
        Ce_fit = np.linspace(1, 50, 50)
        qe_fit = 45 * 0.05 * Ce_fit / (1 + 0.05 * Ce_fit)
        fig = create_isotherm_plot(Ce, qe, Ce_fit, qe_fit, model_name="Langmuir", r_squared=0.98)
        assert isinstance(fig, go.Figure)


class TestCreateKineticPlotExtended:
    def test_basic_plot(self):
        t = np.array([0, 5, 10, 20, 60, 120])
        qt = np.array([0, 15, 25, 35, 45, 48])
        t_fit = np.linspace(0, 120, 50)
        qt_fit = 50 * (1 - np.exp(-0.05 * t_fit))
        fig = create_kinetic_plot(t, qt, t_fit, qt_fit, model_name="PFO")
        assert isinstance(fig, go.Figure)


class TestCreateModelComparisonPlotExtended:
    def test_basic_comparison(self):
        x_data = np.array([1, 5, 10, 20, 50])
        y_data = np.array([5, 15, 22, 30, 40])
        model_results = {
            "Langmuir": {
                "converged": True,
                "y_pred": np.array([4.5, 14.8, 22.5, 30.2, 39.5]),
                "r_squared": 0.98,
            },
            "Freundlich": {
                "converged": True,
                "y_pred": np.array([5.2, 15.5, 21.0, 29.5, 41.0]),
                "r_squared": 0.95,
            },
        }
        fig = create_model_comparison_plot(x_data, y_data, model_results)
        assert isinstance(fig, go.Figure)

    def test_with_unconverged(self):
        x_data = np.array([1, 5, 10, 20, 50])
        y_data = np.array([5, 15, 22, 30, 40])
        model_results = {
            "Langmuir": {
                "converged": True,
                "y_pred": np.array([4.5, 14.8, 22.5, 30.2, 39.5]),
                "r_squared": 0.98,
            },
            "Sips": {"converged": False},
        }
        fig = create_model_comparison_plot(x_data, y_data, model_results)
        assert isinstance(fig, go.Figure)


class TestCreateParityPlotExtended:
    def test_basic(self):
        y_obs = np.array([1, 2, 3, 4, 5], dtype=float)
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        fig = create_parity_plot(y_obs, y_pred)
        assert isinstance(fig, go.Figure)


class TestCreateResidualPlot:
    def test_basic(self):
        residuals = np.array([0.1, -0.2, 0.3, -0.1, 0.2])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fig = create_residual_plot(residuals, y_pred)
        assert isinstance(fig, go.Figure)


class TestCreateVantHoffPlot:
    def test_basic(self):
        inv_T = np.array([0.00330, 0.00325, 0.00320])
        ln_Kd = np.array([1.5, 1.3, 1.1])
        fig = create_vant_hoff_plot(inv_T, ln_Kd)
        assert isinstance(fig, go.Figure)


class TestCreateEffectPlot:
    def test_basic(self):
        x = np.array([2, 4, 6, 8, 10])
        y = np.array([60, 75, 90, 85, 70])
        fig = create_effect_plot(x, y, x_label="pH", y_label="Removal (%)")
        assert isinstance(fig, go.Figure)


# =============================================================================
# PROFESSIONAL STYLES
# =============================================================================
class TestApplyProfessionalStyleExtended:
    def test_basic(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
        styled = apply_professional_style(fig)
        assert isinstance(styled, go.Figure)

    def test_with_barmode(self):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[1, 2], y=[3, 4]))
        styled = apply_professional_style(fig, barmode="group")
        assert isinstance(styled, go.Figure)


class TestApplyProfessional3DStyle:
    def test_basic(self):
        fig = go.Figure()
        fig.add_trace(go.Surface(z=[[1, 2], [3, 4]]))
        styled = apply_professional_3d_style(fig)
        assert isinstance(styled, go.Figure)


class TestApplyProfessionalPolarStyle:
    def test_basic(self):
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[1, 2, 3], theta=[0, 120, 240]))
        styled = apply_professional_polar_style(fig)
        assert isinstance(styled, go.Figure)


class TestApplyStandardLayout:
    def test_basic(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
        styled = apply_standard_layout(fig, title="Test")
        assert isinstance(styled, go.Figure)


# =============================================================================
# FIGURE UTILITIES
# =============================================================================
class TestPrepareForExport:
    def test_basic(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
        exported = prepare_figure_for_export(fig)
        assert isinstance(exported, go.Figure)


class TestInferFigureKind:
    def test_scatter(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
        kind = infer_figure_kind(fig)
        assert isinstance(kind, str)

    def test_bar(self):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[1, 2], y=[3, 4]))
        kind = infer_figure_kind(fig)
        assert isinstance(kind, str)

    def test_surface(self):
        fig = go.Figure()
        fig.add_trace(go.Surface(z=[[1, 2], [3, 4]]))
        kind = infer_figure_kind(fig)
        assert isinstance(kind, str)


class TestFinalizeFigure:
    def test_basic(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
        finalized = finalize_figure(fig)
        assert isinstance(finalized, go.Figure)
