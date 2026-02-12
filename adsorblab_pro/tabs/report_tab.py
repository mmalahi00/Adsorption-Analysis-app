# tabs/report_tab.py
"""
Export Tab - AdsorbLab Pro
==========================

Centralized export system for all analysis results.

Features:
- Comprehensive list of all available figures
- Selectable items with preview
- Multiple export formats (TIFF, PNG, SVG, PDF)
- Organized ZIP structure
- Tables in CSV and Excel formats
"""

import io
import logging
import zipfile
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots

from adsorblab_pro.streamlit_compat import st

logger = logging.getLogger(__name__)

from ..config import EXPORT_DPI, FONT_FAMILY
from ..docx_report import (
    DOCX_AVAILABLE,
    DocxReportConfig,
    create_docx_report,
    _pt_to_px,
    _strip_user_headings_plotly,
    _fix_axis_gradient_overlaps_plotly,
)
from ..models import (
    elovich_model,
    freundlich_model,
    ipd_model,
    langmuir_model,
    pfo_model,
    pso_model,
    sips_model,
    temkin_model,
)
from ..plot_style import (
    COLORS,
    MARKERS,
    MODEL_COLORS,
    STUDY_COLORS,
    apply_professional_style,
    apply_professional_polar_style,
    apply_professional_3d_style,
    create_isotherm_plot,
    create_kinetic_plot,
    create_model_comparison_plot,
    create_effect_plot,
    create_dual_axis_effect_plot,
    create_residual_plot,
    get_axis_style,
    style_experimental_trace,
)

from ..utils import get_current_study_state

# Check for kaleido (used by Plotly for static image export)
try:
    import kaleido  # noqa: F401

    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False


# =============================================================================
# EXPORT RENDERING HELPERS
# =============================================================================

def style_figure_for_export(
    fig: go.Figure,
    *,
    width_px: int,
    height_px: int,
    target_width_in: float = 6.5,
    preset: str = "Manuscript (journal)",
    fig_id: str | None = None,
    study_state: dict | None = None,
    strip_user_headings: bool = True,
) -> go.Figure:
    """Return a copy of a Plotly figure with export-friendly typography.

    Export goals:
    - Publication-ready text sizing at a given intended physical width.
    - Remove user-added headings that can clutter exported figures.
    - Prevent overlaps between axis titles and gradient/colorbar elements.
    """
    if fig is None:
        return fig

    fig = go.Figure(fig)  # copy (don’t mutate UI figure)

    # Optional cleanup: remove user headings (export only)
    if strip_user_headings:
        _strip_user_headings_plotly(fig, fig_id=fig_id)

    # Force export canvas size
    try:
        fig.update_layout(width=int(width_px), height=int(height_px))
    except Exception:
        pass

    preset_norm = (preset or "Manuscript (journal)").strip().lower()
    if preset_norm.startswith("present"):
        base_pt, tick_pt, axis_title_pt, title_pt, legend_pt = 16.0, 15.0, 17.0, 20.0, 14.0
    elif preset_norm.startswith("poster"):
        base_pt, tick_pt, axis_title_pt, title_pt, legend_pt = 18.0, 18.0, 20.0, 24.0, 16.0
    else:
        base_pt, tick_pt, axis_title_pt, title_pt, legend_pt = 11.0, 10.5, 12.0, 14.0, 10.0

    base_px = _pt_to_px(base_pt, width_px=width_px, target_width_in=target_width_in)
    tick_px = _pt_to_px(tick_pt, width_px=width_px, target_width_in=target_width_in)
    axis_title_px = _pt_to_px(axis_title_pt, width_px=width_px, target_width_in=target_width_in)
    title_px = _pt_to_px(title_pt, width_px=width_px, target_width_in=target_width_in)
    legend_px = _pt_to_px(legend_pt, width_px=width_px, target_width_in=target_width_in)

    # Base font
    try:
        fig.update_layout(font={"family": FONT_FAMILY, "size": base_px})
    except Exception:
        pass

    # 2D axes typography
    try:
        fig.update_xaxes(
            tickfont={"size": tick_px, "family": FONT_FAMILY},
            title_font={"size": axis_title_px, "family": FONT_FAMILY},
        )
        fig.update_yaxes(
            tickfont={"size": tick_px, "family": FONT_FAMILY},
            title_font={"size": axis_title_px, "family": FONT_FAMILY},
        )
    except Exception:
        pass

    # Legend + title
    try:
        if getattr(fig.layout, "legend", None) is not None:
            fig.update_layout(legend={"font": {"size": legend_px, "family": FONT_FAMILY}})
    except Exception:
        pass

    # Make legend symbols a constant size (prevents huge markers from breaking legend layout)
    try:
        leg = fig.layout.legend.to_plotly_json() if getattr(fig.layout, "legend", None) else {}
        leg.update({"itemsizing": "constant"})
        fig.update_layout(legend=leg)
    except Exception:
        pass

    # If an equation-like trace name (e.g., "y = ax + b") is shown in the legend, move it to an annotation (export-only).
    try:
        import re as _re
        eq_re = _re.compile(r"^\s*y\s*=\s*[-+0-9.eE]+\s*x\s*[-+ ]\s*[-+0-9.eE]+\s*$", _re.IGNORECASE)
        existing_texts = set()
        try:
            if getattr(fig.layout, "annotations", None):
                for _a in fig.layout.annotations:
                    _t = getattr(_a, "text", None)
                    if _t:
                        existing_texts.add(str(_t))
        except Exception:
            pass

        eq_texts = []
        for _tr in getattr(fig, "data", []) or []:
            _name = str(getattr(_tr, "name", "") or "").strip()
            if _name and eq_re.match(_name):
                try:
                    _tr.showlegend = False
                except Exception:
                    pass
                if _name not in existing_texts:
                    eq_texts.append(_name)

        if eq_texts:
            # Place near top-left inside plot area
            fig.add_annotation(
                text=eq_texts[0],
                xref="paper",
                yref="paper",
                x=0.05,
                y=0.95,
                showarrow=False,
                font={"family": FONT_FAMILY, "size": base_px},
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="rgba(0,0,0,0.35)",
                borderwidth=1,
                borderpad=6,
            )
    except Exception:
        pass

    try:
        if getattr(fig.layout, "title", None) is not None:
            # If title is intentionally blank, keep it blank
            current_text = ""
            try:
                current_text = str(fig.layout.title.text or "")
            except Exception:
                current_text = ""
            if current_text.strip() != "":
                fig.update_layout(title={"font": {"size": title_px, "family": FONT_FAMILY}})
    except Exception:
        pass

    # Annotations: keep, but make typography consistent
    try:
        if getattr(fig.layout, "annotations", None):
            for ann in fig.layout.annotations:
                try:
                    ann_font = ann.font.to_plotly_json() if getattr(ann, "font", None) else {}
                    ann_font.update({"family": FONT_FAMILY, "size": base_px})
                    ann.font = ann_font
                except Exception:
                    pass
    except Exception:
        pass

    # Prevent axis-title vs gradient/colorbar overlaps
    _fix_axis_gradient_overlaps_plotly(fig, width_px=width_px, height_px=height_px, tick_px=tick_px, axis_title_px=axis_title_px, font_family=FONT_FAMILY)

    return fig


def _with_dpi_metadata(img_bytes: bytes, *, fmt: str, dpi: int) -> bytes:
    """Embed DPI metadata (useful for Office apps)."""
    fmt = (fmt or "").lower()
    if fmt not in {"png", "tiff", "tif"}:
        return img_bytes
    try:
        bio_in = io.BytesIO(img_bytes)
        img = Image.open(bio_in)
        bio_out = io.BytesIO()
        if fmt == "png":
            img.save(bio_out, format="PNG", dpi=(dpi, dpi))
        else:
            img.save(bio_out, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
        bio_out.seek(0)
        return bio_out.getvalue()
    except Exception:
        return img_bytes

# =============================================================================
# EXPORTABLE ITEM DEFINITIONS
# =============================================================================

FIGURE_CATEGORIES = {
    "calibration": {
        "name": "Calibration",
        "icon": "📐",
        "items": [
            ("calib_curve", "Calibration Curve", "Linear regression with equation"),
            ("calib_residuals", "Calibration Residuals", "Residual plot for calibration"),
        ],
    },
    "isotherm": {
        "name": "Isotherm Analysis",
        "icon": "📈",
        "items": [
            ("iso_overview", "Isotherm Overview", "Experimental Ce vs qe data"),
            ("iso_langmuir", "Langmuir Model", "Langmuir fit with CI"),
            ("iso_freundlich", "Freundlich Model", "Freundlich fit with CI"),
            ("iso_temkin", "Temkin Model", "Temkin fit with CI"),
            ("iso_sips", "Sips Model", "Sips fit with CI"),
            ("iso_comparison", "Model Comparison", "All models overlay"),
            ("iso_rl", "Separation Factor (RL)", "RL vs C0 plot"),
        ],
    },
    "kinetic": {
        "name": "Kinetic Analysis",
        "icon": "⏱️",
        "items": [
            ("kin_overview", "Kinetic Overview", "Time vs qt experimental data"),
            ("kin_pfo", "PFO Model", "Pseudo-first-order fit"),
            ("kin_pso", "PSO Model", "Pseudo-second-order fit"),
            ("kin_rpso", "rPSO Model", "Revised PSO with concentration correction"),
            ("kin_elovich", "Elovich Model", "Elovich fit"),
            ("kin_ipd", "IPD Model", "Intraparticle diffusion"),
            ("kin_comparison", "Model Comparison", "All kinetic models overlay"),
        ],
    },
    "thermodynamic": {
        "name": "Thermodynamic Analysis",
        "icon": "🌡️",
        "items": [
            ("thermo_vanthoff", "Van't Hoff Plot", "ln(Kd) vs 1/T"),
            ("thermo_gibbs", "Gibbs Energy Plot", "Delta G vs Temperature"),
        ],
    },
    "effects": {
        "name": "Effect Studies",
        "icon": "🔬",
        "items": [
            ("effect_ph", "pH Effect", "qe and removal vs pH"),
            ("effect_temp", "Temperature Effect", "qe vs temperature"),
            ("effect_dosage", "Dosage Effect", "qe vs adsorbent dosage"),
        ],
    },
    "statistical": {"name": "Statistical Summary", "icon": "📊", "items": []},
    "3d_explorer": {
        "name": "3D Explorer (Saved)",
        "icon": "🔮",
        "items": [],  # Dynamically populated from saved figures
    },
    "multi_study": {
        "name": "Multi-Study Comparison",
        "icon": "🆚",
        "items": [
            ("multi_iso_qm_bar", "qm Comparison Bar Chart", "Langmuir qm across studies"),
            ("multi_iso_radar", "Isotherm Radar Chart", "Multi-criteria isotherm comparison"),
            ("multi_kin_qe_bar", "qe Comparison Bar Chart", "PSO qe across studies"),
            ("multi_kin_radar", "Kinetic Radar Chart", "Multi-criteria kinetic comparison"),
            ("multi_thermo_bar", "Thermodynamic Bar Chart", "ΔH°, ΔS°, ΔG° comparison"),
            ("multi_ranking_bar", "Overall Ranking", "Combined performance ranking"),
        ],
    },
}

TABLE_CATEGORIES = {
    "calibration": {
        "name": "Calibration",
        "items": [
            ("tbl_calib_params", "Calibration Parameters", "Slope, intercept, R2"),
            ("tbl_calib_data", "Calibration Data", "Raw calibration data"),
        ],
    },
    "isotherm": {
        "name": "Isotherm",
        "items": [
            ("tbl_iso_data", "Isotherm Data", "Ce, qe, removal"),
            ("tbl_iso_params", "Isotherm Parameters", "All model parameters with CI"),
            ("tbl_iso_comparison", "Model Comparison", "R2, AIC, RMSE comparison"),
        ],
    },
    "kinetic": {
        "name": "Kinetic",
        "items": [
            ("tbl_kin_data", "Kinetic Data", "Time, qt, removal"),
            ("tbl_kin_params", "Kinetic Parameters", "All model parameters with CI"),
            ("tbl_kin_comparison", "Model Comparison", "R2, AIC, RMSE comparison"),
        ],
    },
    "thermodynamic": {
        "name": "Thermodynamic",
        "items": [
            ("tbl_thermo_params", "Thermodynamic Parameters", "Delta H, S, G"),
            ("tbl_thermo_data", "Temperature Data", "T, Kd, ln(Kd)"),
        ],
    },
    "effects": {
        "name": "Effect Studies",
        "items": [
            ("tbl_ph_data", "pH Effect Data", "pH vs qe data"),
            ("tbl_temp_data", "Temperature Effect Data", "T vs qe data"),
            ("tbl_dosage_data", "Dosage Effect Data", "Dosage vs qe data"),
        ],
    },
    "multi_study": {
        "name": "Multi-Study Comparison",
        "items": [
            (
                "tbl_multi_iso_params",
                "Isotherm Parameters Comparison",
                "All studies isotherm params",
            ),
            ("tbl_multi_kin_params", "Kinetic Parameters Comparison", "All studies kinetic params"),
            ("tbl_multi_thermo", "Thermodynamic Comparison", "All studies ΔH°, ΔS°, ΔG°"),
            ("tbl_multi_ranking", "Overall Study Ranking", "Combined performance scores"),
            ("tbl_multi_pub_summary", "Summary Table", "All key parameters in one table"),
            ("tbl_multi_mechanism", "Mechanism Interpretation", "Adsorption mechanism analysis"),
        ],
    },
}


# =============================================================================
# SAVED 3D FIGURES HELPERS
# =============================================================================


def get_saved_3d_figures(study_state: dict) -> list[tuple[str, str, str]]:
    """
    Get saved 3D figures from study state for export.

    Returns list of tuples: (fig_id, title, description)
    """
    if study_state is None:
        return []

    saved = study_state.get("saved_3d_figures", {})
    items = []

    for unique_id, fig_data in saved.items():
        title = fig_data.get("title", "Untitled 3D Figure")
        params = fig_data.get("params", {})

        # Create description from parameters
        params_str = ", ".join([f"{k}={v}" for k, v in list(params.items())[:2]])
        description = params_str if params_str else "Custom 3D visualization"

        items.append((unique_id, title, description))

    return items


# =============================================================================
# FIGURE GENERATION FUNCTIONS
# =============================================================================


def generate_figure(fig_id: str, study_state: dict) -> go.Figure | None:
    """Generate a specific figure by ID."""

    # Check if this is a saved 3D figure first
    if study_state:
        saved_3d = study_state.get("saved_3d_figures", {})
        if fig_id in saved_3d:
            fig_dict = saved_3d[fig_id].get("figure")
            if fig_dict:
                fig = go.Figure(fig_dict)
                fig = apply_professional_3d_style(fig, title=saved_3d[fig_id].get("title", "3D Figure"), height=700)
                return fig

    # Standard figure generators
    generators = {
        "calib_curve": _gen_calibration_curve,
        "calib_residuals": _gen_calibration_residuals,
        "iso_overview": _gen_isotherm_overview,
        "iso_langmuir": lambda s: _gen_isotherm_model(s, "Langmuir"),
        "iso_freundlich": lambda s: _gen_isotherm_model(s, "Freundlich"),
        "iso_temkin": lambda s: _gen_isotherm_model(s, "Temkin"),
        "iso_sips": lambda s: _gen_isotherm_model(s, "Sips"),
        "iso_comparison": _gen_isotherm_comparison,
        "iso_rl": _gen_separation_factor,
        "kin_overview": _gen_kinetic_overview,
        "kin_pfo": lambda s: _gen_kinetic_model(s, "PFO"),
        "kin_pso": lambda s: _gen_kinetic_model(s, "PSO"),
        "kin_rpso": lambda s: _gen_kinetic_model(s, "rPSO"),
        "kin_elovich": lambda s: _gen_kinetic_model(s, "Elovich"),
        "kin_ipd": lambda s: _gen_kinetic_model(s, "IPD"),
        "kin_comparison": _gen_kinetic_comparison,
        "thermo_vanthoff": _gen_vanthoff_plot,
        "thermo_gibbs": _gen_gibbs_plot,
        "effect_ph": _gen_ph_effect,
        "effect_temp": _gen_temperature_effect,
        "effect_dosage": _gen_dosage_effect,
        # Multi-study comparison figures
        "multi_iso_qm_bar": _gen_multi_iso_qm_bar,
        "multi_iso_radar": _gen_multi_iso_radar,
        "multi_kin_qe_bar": _gen_multi_kin_qe_bar,
        "multi_kin_radar": _gen_multi_kin_radar,
        "multi_thermo_bar": _gen_multi_thermo_bar,
        "multi_ranking_bar": _gen_multi_ranking_bar,
    }

    generator = generators.get(fig_id)
    if generator:
        try:
            return generator(study_state)
        except Exception as e:
            logger.warning(f"Failed to generate figure '{fig_id}': {e}")
            return None
    return None


def _gen_calibration_curve(study_state: dict) -> go.Figure | None:
    """Generate calibration curve figure."""
    calib_params = study_state.get("calibration_params")
    calib_df = study_state.get("calib_df_input")

    if calib_df is None or calib_params is None:
        return None

    conc = calib_df["Concentration"].values
    abs_val = calib_df["Absorbance"].values

    x_line = np.linspace(0, conc.max() * 1.1, 100)
    y_line = calib_params["slope"] * x_line + calib_params["intercept"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=conc,
            y=abs_val,
            **style_experimental_trace(name="Standards"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Linear Fit",
            line={"color": COLORS["fit_primary"], "width": 2},
        )
    )

    eq_text = f"y = {calib_params['slope']:.4f}x + {calib_params['intercept']:.4f}"
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=eq_text,
        showarrow=False,
        font={"size": 12},
        bgcolor="white",
        borderpad=4,
    )

    fig = apply_professional_style(fig, "Calibration Curve", "Concentration (mg/L)", "Absorbance")
    return fig


def _gen_calibration_residuals(study_state: dict) -> go.Figure | None:
    calib_params = study_state.get("calibration_params")
    calib_df = study_state.get("calib_df_input")

    if calib_df is None or calib_params is None:
        return None

    conc = calib_df["Concentration"].values
    abs_val = calib_df["Absorbance"].values
    predicted = calib_params["slope"] * conc + calib_params["intercept"]
    residuals = abs_val - predicted

    return create_residual_plot(predicted, residuals, model_name="Calibration")


def _gen_isotherm_overview(study_state: dict) -> go.Figure | None:
    """Generate isotherm overview figure."""
    iso_data = study_state.get("isotherm_results")

    if iso_data is None:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iso_data["Ce_mgL"],
            y=iso_data["qe_mg_g"],
            **style_experimental_trace(name="Experimental"),
        )
    )

    fig = apply_professional_style(
        fig,
        "Adsorption Isotherm - Experimental Data",
        "Equilibrium Concentration, Ce (mg/L)",
        "Adsorption Capacity, qe (mg/g)",
    )
    return fig


def _gen_isotherm_model(study_state: dict, model_name: str) -> go.Figure | None:
    """Generate individual isotherm model figure (case-study consistent)."""
    iso_results = study_state.get("isotherm_models_fitted", {})
    iso_data = study_state.get("isotherm_results")

    if iso_data is None or model_name not in iso_results:
        return None

    model_data = iso_results[model_name]
    if not model_data.get("converged"):
        return None

    Ce = iso_data["Ce_mgL"].values
    qe = iso_data["qe_mg_g"].values
    Ce_line = np.linspace(max(0.0, Ce.min() * 0.9), Ce.max() * 1.05, 200)

    params = model_data.get("params", {})
    qe_pred = None

    if model_name == "Langmuir" and "qm" in params and "KL" in params:
        qe_pred = langmuir_model(Ce_line, params["qm"], params["KL"])
    elif model_name == "Freundlich" and "KF" in params and "n_inv" in params:
        qe_pred = freundlich_model(Ce_line, params["KF"], params["n_inv"])
    elif model_name == "Temkin" and "B1" in params and "KT" in params:
        qe_pred = temkin_model(Ce_line, params["B1"], params["KT"])
    elif model_name == "Sips" and "qm" in params and "Ks" in params and "ns" in params:
        qe_pred = sips_model(Ce_line, params["qm"], params["Ks"], params["ns"])

    if qe_pred is None:
        return None

    r2 = model_data.get("r_squared")

    # Use the shared plotting helper so Report/Isotherm tab/export are identical.
    fig = create_isotherm_plot(
        Ce,
        qe,
        Ce_line,
        qe_pred,
        model_name=model_name,
        r_squared=float(r2) if isinstance(r2, (int, float)) else None,
        height=450,
    )
    return fig


def _gen_isotherm_comparison(study_state: dict) -> go.Figure | None:
    """Generate isotherm model comparison figure (case-study consistent)."""
    iso_results = study_state.get("isotherm_models_fitted", {})
    iso_data = study_state.get("isotherm_results")

    if iso_data is None or not iso_results:
        return None

    Ce = iso_data["Ce_mgL"].values
    qe = iso_data["qe_mg_g"].values

    model_functions = {
        "Langmuir": lambda x, p: langmuir_model(x, p["qm"], p["KL"]),
        "Freundlich": lambda x, p: freundlich_model(x, p["KF"], p["n_inv"]),
        "Temkin": lambda x, p: temkin_model(x, p["B1"], p["KT"]),
        "Sips": lambda x, p: sips_model(x, p["qm"], p["Ks"], p["ns"]),
    }

    fig = create_model_comparison_plot(
        Ce,
        qe,
        iso_results,
        model_functions,
        x_label="C<sub>e</sub> (mg/L)",
        y_label="q<sub>e</sub> (mg/g)",
        title="Isotherm Model Comparison",
        height=450,
    )
    return fig


def _gen_separation_factor(study_state: dict) -> go.Figure | None:
    """Generate RL separation factor plot (case-study consistent)."""
    iso_results = study_state.get("isotherm_models_fitted", {})
    iso_data = study_state.get("isotherm_results")

    langmuir = iso_results.get("Langmuir", {})
    if not langmuir.get("converged") or iso_data is None:
        return None

    RL = langmuir.get("RL")
    if RL is None:
        KL = langmuir.get("params", {}).get("KL")
        if KL is None:
            return None
        C0 = iso_data["C0_mgL"].values
        RL = 1 / (1 + KL * C0)

    C0 = iso_data["C0_mgL"].values

    color = MODEL_COLORS.get("Langmuir", "#1f77b4")

    fig = go.Figure()

    # Favourable region: 0 < RL < 1 (light green band)
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="y",
        x0=0,
        x1=1,
        y0=0,
        y1=1,
        fillcolor="rgba(0, 128, 0, 0.08)",
        line_width=0,
        layer="below",
    )

    fig.add_trace(
        go.Scatter(
            x=C0,
            y=RL,
            mode="markers+lines",
            name="RL",
            marker={"size": 10, "color": color},
            line={"width": 2, "color": color},
        )
    )

    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_hline(y=1, line_dash="dot", line_color="gray")

    fig = apply_professional_style(
        fig,
        "Langmuir Separation Factor (RL)",
        "Initial Concentration, C<sub>0</sub> (mg/L)",
        "Separation Factor, RL",
    )

    fig.update_yaxes(rangemode="tozero")

    return fig


def _gen_kinetic_overview(study_state: dict) -> go.Figure | None:
    """Generate kinetic overview figure (house style)."""
    kin_data = study_state.get("kinetic_results_df")
    if kin_data is None:
        return None

    t = np.asarray(kin_data["Time"], dtype=float)
    qt = np.asarray(kin_data["qt_mg_g"], dtype=float)

    fig = go.Figure()
    exp_kwargs = style_experimental_trace(name="Experimental")
    exp_kwargs["legendrank"] = 1
    fig.add_trace(go.Scatter(x=t, y=qt, **exp_kwargs))

    fig = apply_professional_style(
        fig,
        "Adsorption Kinetics (Experimental)",
        "Time (min)",
        "q<sub>t</sub> (mg/g)",
        legend_position="lower right",
    )
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    return fig

def _gen_kinetic_model(study_state: dict, model_name: str) -> go.Figure | None:
    """Generate individual kinetic model figure (house style)."""
    kin_results = study_state.get("kinetic_models_fitted", {})
    kin_data = study_state.get("kinetic_results_df")

    if kin_data is None or model_name not in kin_results:
        return None

    model_data = kin_results[model_name]
    if not model_data.get("converged"):
        return None

    t = kin_data["Time"].values
    qt = kin_data["qt_mg_g"].values
    t_line = np.linspace(0.1, float(np.nanmax(t)) * 1.1, 200)

    params = model_data.get("params", {})
    qt_pred = None

    if model_name == "PFO" and "qe" in params and "k1" in params:
        qt_pred = pfo_model(t_line, params["qe"], params["k1"])
    elif model_name == "PSO" and "qe" in params and "k2" in params:
        qt_pred = pso_model(t_line, params["qe"], params["k2"])
    elif model_name == "rPSO" and "qe" in params and "k2" in params:
        qt_pred = pso_model(t_line, params["qe"], params["k2"])
    elif model_name == "Elovich" and "alpha" in params and "beta" in params:
        qt_pred = elovich_model(t_line, params["alpha"], params["beta"])
    elif model_name == "IPD" and "kid" in params and "C" in params:
        qt_pred = ipd_model(t_line, params["kid"], params["C"])

    if qt_pred is None:
        return None

    r2 = float(model_data.get("r_squared", 0.0))

    fig = create_kinetic_plot(
        t=t,
        qt=qt,
        t_fit=t_line,
        qt_fit=qt_pred,
        model_name=model_name,
        r_squared=r2,
        title=f"{model_name} Kinetic Model (R² = {r2:.4f})",
        height=450,
    )
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    return fig

def _gen_kinetic_comparison(study_state: dict) -> go.Figure | None:
    """Generate kinetic model comparison figure (house style)."""
    kin_results = study_state.get("kinetic_models_fitted", {})
    kin_data = study_state.get("kinetic_results_df")

    if kin_data is None or not kin_results:
        return None

    t = kin_data["Time"].values
    qt = kin_data["qt_mg_g"].values

    model_functions = {
        "PFO": lambda x, p: pfo_model(x, p["qe"], p["k1"]),
        "PSO": lambda x, p: pso_model(x, p["qe"], p["k2"]),
        "rPSO": lambda x, p: pso_model(x, p["qe"], p["k2"]),
        "Elovich": lambda x, p: elovich_model(x, p["alpha"], p["beta"]),
        "IPD": lambda x, p: ipd_model(x, p["kid"], p["C"]),
    }

    fig = create_model_comparison_plot(
        x_exp=t,
        y_exp=qt,
        fitted_models=kin_results,
        model_functions=model_functions,
        x_label="Time (min)",
        y_label="q<sub>t</sub> (mg/g)",
        title="Kinetic Model Comparison",
        height=450,
    )

    fig = apply_professional_style(
        fig,
        "Kinetic Model Comparison",
        "Time (min)",
        "q<sub>t</sub> (mg/g)",
        legend_position="lower right",
    )
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    return fig

def _gen_vanthoff_plot(study_state: dict) -> go.Figure | None:
    """Generate Van't Hoff plot."""
    thermo_params = study_state.get("thermo_params")

    if not thermo_params:
        return None

    T_K = np.array(thermo_params.get("temperatures", []))
    Kd = np.array(thermo_params.get("Kd_values", []))

    if len(T_K) < 2 or len(Kd) < 2:
        return None

    x = 1 / T_K
    y = np.log(Kd)

    slope = thermo_params.get("slope", 0)
    intercept = thermo_params.get("intercept", 0)

    x_line = np.linspace(x.min() * 0.98, x.max() * 1.02, 100)
    y_line = slope * x_line + intercept

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x * 1000,
            y=y,
            mode="markers",
            marker=MARKERS["experimental"],
            name="Experimental",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line * 1000,
            y=y_line,
            mode="lines",
            name="Linear Fit",
            line={"color": COLORS["fit_primary"], "width": 2},
        )
    )

    delta_H = thermo_params.get("delta_H", 0)
    delta_S = thermo_params.get("delta_S", 0)
    fig.add_annotation(
        x=0.98,
        y=0.95,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="top",
        text=f"ΔH° = {delta_H:.2f} kJ/mol<br>ΔS° = {delta_S:.2f} J/(mol·K)",
        showarrow=False,
        font={"size": 12},
        bgcolor="white",
        borderpad=4,
    )

    fig = apply_professional_style(fig, "Van't Hoff Plot", "1000/T (1/K)", "ln(Kd)")
    
    # Reposition legend to upper left to avoid overlap with data points
    fig.update_layout(
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
        )
    )
    return fig


def _gen_gibbs_plot(study_state: dict) -> go.Figure | None:
    """Generate Gibbs free energy plot."""
    thermo_params = study_state.get("thermo_params")

    if not thermo_params:
        return None

    T_K = np.array(thermo_params.get("temperatures", []))
    delta_G = thermo_params.get("delta_G", {})

    if len(T_K) < 2 or not delta_G:
        return None

    G_values = [delta_G.get(T, 0) for T in T_K]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=T_K - 273.15,
            y=G_values,
            mode="markers+lines",
            marker=MARKERS["experimental"],
            line={"width": 2},
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="red")

    fig = apply_professional_style(
        fig, "Gibbs Free Energy vs Temperature", "Temperature (°C)", "ΔG° (kJ/mol)"
    )
    return fig


def _gen_ph_effect(study_state: dict) -> go.Figure | None:
    """Generate pH effect figure (dual-axis export, house style)."""
    ph_results = study_state.get("ph_effect_results")
    if ph_results is None or getattr(ph_results, "empty", False):
        return None

    if "pH" not in ph_results.columns or "qe_mg_g" not in ph_results.columns:
        return None

    y2 = ph_results["removal_%"] if "removal_%" in ph_results.columns else None

    return create_dual_axis_effect_plot(
        x=ph_results["pH"],
        y1=ph_results["qe_mg_g"],
        y2=y2,
        title="Effect of pH on Adsorption",
        x_title="pH",
        y1_title="qe (mg/g)",
        y2_title="Removal (%)",
        y1_name="qe (mg/g)",
        y2_name="Removal (%)",
        height=450,
        x_tozero=False,
        y1_tozero=True,
        y2_tozero=True,
    )

def _gen_temperature_effect(study_state: dict) -> go.Figure | None:
    """Generate temperature effect figure (house style)."""
    temp_results = study_state.get("temp_effect_results")
    if temp_results is None or getattr(temp_results, "empty", False):
        return None

    x_col = "Temperature" if "Temperature" in temp_results.columns else "Temperature_C"
    if x_col not in temp_results.columns or "qe_mg_g" not in temp_results.columns:
        return None

    return create_effect_plot(
        x=temp_results[x_col],
        y=temp_results["qe_mg_g"],
        title="Effect of Temperature on Adsorption",
        x_title="Temperature (°C)",
        y_title="qe (mg/g)",
        height=450,
        series_name="Temperature Effect",
        show_legend=False,
        x_tozero=False,
        y_tozero=True,
        hovertemplate="T: %{x:.1f}°C<br>qe: %{y:.2f}<extra></extra>",
    )

def _gen_dosage_effect(study_state: dict) -> go.Figure | None:
    """Generate dosage effect figure (dual-axis export, house style)."""
    dos_results = study_state.get("dosage_effect_results")
    if dos_results is None or getattr(dos_results, "empty", False):
        return None

    x_col = "Dosage_gL" if "Dosage_gL" in dos_results.columns else ("Mass_g" if "Mass_g" in dos_results.columns else None)
    if x_col is None or "qe_mg_g" not in dos_results.columns:
        return None

    y2 = dos_results["removal_%"] if "removal_%" in dos_results.columns else None

    return create_dual_axis_effect_plot(
        x=dos_results[x_col],
        y1=dos_results["qe_mg_g"],
        y2=y2,
        title="Effect of Adsorbent Dosage",
        x_title="Dosage (g/L)" if x_col == "Dosage_gL" else "Mass (g)",
        y1_title="qe (mg/g)",
        y2_title="Removal (%)",
        y1_name="qe (mg/g)",
        y2_name="Removal (%)",
        height=450,
        x_tozero=True,
        y1_tozero=True,
        y2_tozero=True,
    )

def _get_all_studies() -> tuple[dict, list[str]]:
    """Get all studies from session state."""
    studies = st.session_state.get("studies", {})
    study_names = list(studies.keys())
    return studies, study_names


def _gen_multi_iso_qm_bar(study_state: dict) -> go.Figure | None:
    """Generate multi-study Langmuir qm comparison bar chart."""
    studies, study_names = _get_all_studies()

    if len(study_names) < 2:
        return None

    qm_data = []
    for name in study_names:
        data = studies[name]
        langmuir = data.get("isotherm_models_fitted", {}).get("Langmuir", {})
        if langmuir.get("converged"):
            qm_data.append(
                {
                    "Study": name,
                    "qm": langmuir["params"].get("qm", 0),
                    "R²": langmuir.get("r_squared", 0),
                }
            )

    if not qm_data:
        return None

    df = pd.DataFrame(qm_data).sort_values("qm", ascending=False)
    colors = [STUDY_COLORS[i % len(STUDY_COLORS)] for i in range(len(df))]

    fig = go.Figure(
        data=[
            go.Bar(
                x=df["Study"],
                y=df["qm"],
                marker_color=colors,
                text=df["qm"].round(2),
                textposition="outside",
            )
        ]
    )

    fig = apply_professional_style(
        fig, "Maximum Adsorption Capacity (qm) Comparison", "Study", "qm (mg/g)", height=500
    )
    return fig


def _gen_multi_iso_radar(study_state: dict) -> go.Figure | None:
    """Generate multi-study isotherm radar chart."""
    studies, study_names = _get_all_studies()

    if len(study_names) < 2:
        return None

    metrics_data = []
    for name in study_names:
        data = studies[name]
        iso = data.get("isotherm_models_fitted", {})

        best_r2 = 0
        best_model = None
        for _model_name, results in iso.items():
            if results and results.get("converged"):
                r2 = results.get("r_squared", 0)
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = results

        if best_model:
            metrics_data.append(
                {
                    "Study": name,
                    "R²": best_r2,
                    "Adj-R²": best_model.get("adj_r_squared", best_r2),
                    "qm": iso.get("Langmuir", {}).get("params", {}).get("qm", 0)
                    if iso.get("Langmuir", {}).get("converged")
                    else 0,
                }
            )

    if len(metrics_data) < 2:
        return None

    df = pd.DataFrame(metrics_data)
    categories = ["R²", "Adj-R²", "qm (norm)"]

    fig = go.Figure()

    for i, row in df.iterrows():
        qm_max = df["qm"].max() if df["qm"].max() > 0 else 1
        values = [row["R²"], row["Adj-R²"], row["qm"] / qm_max]
        values.append(values[0])

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                name=row["Study"],
                line={"color": STUDY_COLORS[i % len(STUDY_COLORS)]},
            )
        )

    fig = apply_professional_polar_style(
        fig,
        title="Isotherm Performance Comparison",
        height=500,
        show_legend=True,
        legend_position="upper left",
        radial_range=(0.0, 1.0),
    )
    return fig


def _gen_multi_kin_qe_bar(study_state: dict) -> go.Figure | None:
    """Generate multi-study PSO qe comparison bar chart."""
    studies, study_names = _get_all_studies()

    if len(study_names) < 2:
        return None

    qe_data = []
    for name in study_names:
        data = studies[name]
        pso = data.get("kinetic_models_fitted", {}).get("PSO", {})
        if not pso.get("converged"):
            pso = data.get("kinetic_models_fitted", {}).get("rPSO", {})
        if pso.get("converged"):
            qe_data.append(
                {"Study": name, "qe": pso["params"].get("qe", 0), "R²": pso.get("r_squared", 0)}
            )

    if not qe_data:
        return None

    df = pd.DataFrame(qe_data).sort_values("qe", ascending=False)
    colors = [STUDY_COLORS[i % len(STUDY_COLORS)] for i in range(len(df))]

    fig = go.Figure(
        data=[
            go.Bar(
                x=df["Study"],
                y=df["qe"],
                marker_color=colors,
                text=df["qe"].round(2),
                textposition="outside",
            )
        ]
    )

    fig = apply_professional_style(
        fig, "Equilibrium Adsorption Capacity (qe) Comparison", "Study", "qe (mg/g)", height=500
    )
    return fig


def _gen_multi_kin_radar(study_state: dict) -> go.Figure | None:
    """Generate multi-study kinetic radar chart."""
    studies, study_names = _get_all_studies()

    if len(study_names) < 2:
        return None

    metrics_data = []
    for name in study_names:
        data = studies[name]
        kin = data.get("kinetic_models_fitted", {})

        best_r2 = 0
        best_model = None
        for _model_name, results in kin.items():
            if results and results.get("converged"):
                r2 = results.get("r_squared", 0)
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = results

        if best_model:
            pso = kin.get("PSO", {})
            if not pso.get("converged"):
                pso = kin.get("rPSO", {})
            metrics_data.append(
                {
                    "Study": name,
                    "R²": best_r2,
                    "Adj-R²": best_model.get("adj_r_squared", best_r2),
                    "qe": pso.get("params", {}).get("qe", 0) if pso.get("converged") else 0,
                }
            )

    if len(metrics_data) < 2:
        return None

    df = pd.DataFrame(metrics_data)
    categories = ["R²", "Adj-R²", "qe (norm)"]

    fig = go.Figure()

    for i, row in df.iterrows():
        qe_max = df["qe"].max() if df["qe"].max() > 0 else 1
        values = [row["R²"], row["Adj-R²"], row["qe"] / qe_max]
        values.append(values[0])

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                name=row["Study"],
                line={"color": STUDY_COLORS[i % len(STUDY_COLORS)]},
            )
        )

    fig = apply_professional_polar_style(
        fig,
        title="Kinetic Performance Comparison",
        height=500,
        show_legend=True,
        legend_position="upper left",
        radial_range=(0.0, 1.0),
    )
    return fig


def _gen_multi_thermo_bar(study_state: dict) -> go.Figure | None:
    """Generate multi-study thermodynamic parameters comparison."""
    studies, study_names = _get_all_studies()

    if len(study_names) < 2:
        return None

    thermo_data = []
    for name in study_names:
        data = studies[name]
        thermo = data.get("thermo_params", {})
        if thermo:
            thermo_data.append(
                {
                    "Study": name,
                    "ΔH° (kJ/mol)": thermo.get("delta_H", 0),
                    "ΔS° (J/mol·K)": thermo.get("delta_S", 0),
                    "ΔG° (kJ/mol)": thermo.get("delta_G_values", [0])[0]
                    if isinstance(thermo.get("delta_G_values"), list)
                    else thermo.get("delta_G", 0),
                }
            )

    if not thermo_data:
        return None

    df = pd.DataFrame(thermo_data)

    fig = make_subplots(
        rows=1, cols=3, subplot_titles=["ΔH° (kJ/mol)", "ΔS° (J/mol·K)", "ΔG° (kJ/mol)"]
    )

    colors = [STUDY_COLORS[i % len(STUDY_COLORS)] for i in range(len(df))]

    fig.add_trace(
        go.Bar(x=df["Study"], y=df["ΔH° (kJ/mol)"], marker_color=colors, showlegend=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=df["Study"], y=df["ΔS° (J/mol·K)"], marker_color=colors, showlegend=False),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(x=df["Study"], y=df["ΔG° (kJ/mol)"], marker_color=colors, showlegend=False),
        row=1,
        col=3,
    )

    fig.update_layout(
        title={
            "text": "<b>Thermodynamic Parameters Comparison</b>",
            "font": {"size": 16, "family": FONT_FAMILY},
        },
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": FONT_FAMILY, "size": 12},
        margin={"l": 70, "r": 40, "t": 60, "b": 60},
    )

    # Style all subplots via centralized helper
    for i in range(1, 4):
        fig.update_xaxes(**get_axis_style(""), row=1, col=i)
        fig.update_yaxes(**get_axis_style(""), row=1, col=i)

    # Match subplot title annotations to house style (bold + house font)
    for ann in fig.layout.annotations:
        ann.update(
            text=f"<b>{ann.text}</b>",
            font={"size": 14, "family": FONT_FAMILY},
        )

    return fig


def _gen_multi_ranking_bar(study_state: dict) -> go.Figure | None:
    """Generate overall study ranking bar chart."""
    studies, study_names = _get_all_studies()

    if len(study_names) < 2:
        return None

    ranking_data = []
    for name in study_names:
        data = studies[name]
        score = 0
        count = 0

        iso = data.get("isotherm_models_fitted", {})
        for results in iso.values():
            if results and results.get("converged"):
                score += results.get("r_squared", 0) * 100
                count += 1

        kin = data.get("kinetic_models_fitted", {})
        for results in kin.values():
            if results and results.get("converged"):
                score += results.get("r_squared", 0) * 100
                count += 1

        if count > 0:
            ranking_data.append({"Study": name, "Score": score / count})

    if not ranking_data:
        return None

    df = pd.DataFrame(ranking_data).sort_values("Score", ascending=False)
    colors = [STUDY_COLORS[i % len(STUDY_COLORS)] for i in range(len(df))]

    fig = go.Figure(
        data=[
            go.Bar(
                x=df["Study"],
                y=df["Score"],
                marker_color=colors,
                text=df["Score"].round(1),
                textposition="outside",
            )
        ]
    )

    fig = apply_professional_style(
        fig, "Overall Study Performance Ranking", "Study", "Average R² Score (%)", height=500
    )
    return fig


# =============================================================================
# TABLE GENERATION FUNCTIONS
# =============================================================================


def generate_table(tbl_id: str, study_state: dict) -> pd.DataFrame | None:
    """Generate a specific table by ID."""

    generators = {
        "tbl_calib_params": _gen_tbl_calib_params,
        "tbl_calib_data": _gen_tbl_calib_data,
        "tbl_iso_data": _gen_tbl_iso_data,
        "tbl_iso_params": _gen_tbl_iso_params,
        "tbl_iso_comparison": _gen_tbl_iso_comparison,
        "tbl_kin_data": _gen_tbl_kin_data,
        "tbl_kin_params": _gen_tbl_kin_params,
        "tbl_kin_comparison": _gen_tbl_kin_comparison,
        "tbl_thermo_params": _gen_tbl_thermo_params,
        "tbl_thermo_data": _gen_tbl_thermo_data,
        "tbl_ph_data": _gen_tbl_ph_data,
        "tbl_temp_data": _gen_tbl_temp_data,
        "tbl_dosage_data": _gen_tbl_dosage_data,
        "tbl_multi_iso_params": _gen_tbl_multi_iso_params,
        "tbl_multi_kin_params": _gen_tbl_multi_kin_params,
        "tbl_multi_thermo": _gen_tbl_multi_thermo,
        "tbl_multi_ranking": _gen_tbl_multi_ranking,
        "tbl_multi_pub_summary": _gen_tbl_multi_pub_summary,
        "tbl_multi_mechanism": _gen_tbl_multi_mechanism,
    }

    generator = generators.get(tbl_id)
    if generator:
        try:
            return generator(study_state)
        except Exception as e:
            logger.warning(f"Failed to generate table '{tbl_id}': {e}")
            return None
    return None


def _gen_tbl_calib_params(s: dict) -> pd.DataFrame | None:
    calib = s.get("calibration_params")
    if not calib:
        return None
    return pd.DataFrame(
        [
            {"Parameter": "Slope", "Value": f"{calib['slope']:.6f}", "Unit": "L/mg"},
            {"Parameter": "Intercept", "Value": f"{calib['intercept']:.6f}", "Unit": "AU"},
            {"Parameter": "R²", "Value": f"{calib['r_squared']:.6f}", "Unit": "-"},
        ]
    )


def _gen_tbl_calib_data(s: dict) -> pd.DataFrame | None:
    return s.get("calib_df_input")


def _gen_tbl_iso_data(s: dict) -> pd.DataFrame | None:
    return s.get("isotherm_results")


def _gen_tbl_iso_params(s: dict) -> pd.DataFrame | None:
    iso = s.get("isotherm_models_fitted", {})
    if not iso:
        return None
    rows = []
    for m, d in iso.items():
        if not d.get("converged"):
            continue
        for p, v in d.get("params", {}).items():
            ci = d.get("ci_95", {}).get(p, (None, None))
            rows.append(
                {
                    "Model": m,
                    "Parameter": p,
                    "Value": f"{v:.4f}" if isinstance(v, int | float) else str(v),
                    "CI_Lower": f"{ci[0]:.4f}" if ci[0] else "-",
                    "CI_Upper": f"{ci[1]:.4f}" if ci[1] else "-",
                }
            )
    return pd.DataFrame(rows) if rows else None


def _gen_tbl_iso_comparison(s: dict) -> pd.DataFrame | None:
    iso = s.get("isotherm_models_fitted", {})
    if not iso:
        return None
    rows = []
    for m, d in iso.items():
        if not d.get("converged"):
            continue
        rows.append(
            {
                "Model": m,
                "R²": d.get("r_squared", 0),
                "Adj_R²": d.get("adj_r_squared", 0),
                "RMSE": d.get("rmse", 0),
                "AIC": d.get("aicc", d.get("aic", 0)),
                "BIC": d.get("bic", 0),
                "PRESS": d.get("press", "-"),
                "Q²": d.get("q2", "-"),
            }
        )
    return pd.DataFrame(rows) if rows else None


def _gen_tbl_kin_data(s: dict) -> pd.DataFrame | None:
    return s.get("kinetic_results_df")


def _gen_tbl_kin_params(s: dict) -> pd.DataFrame | None:
    kin = s.get("kinetic_models_fitted", {})
    if not kin:
        return None
    rows = []
    for m, d in kin.items():
        if not d.get("converged"):
            continue
        for p, v in d.get("params", {}).items():
            ci = d.get("ci_95", {}).get(p, (None, None))
            rows.append(
                {
                    "Model": m,
                    "Parameter": p,
                    "Value": f"{v:.4f}" if isinstance(v, int | float) else str(v),
                    "CI_Lower": f"{ci[0]:.4f}" if ci[0] else "-",
                    "CI_Upper": f"{ci[1]:.4f}" if ci[1] else "-",
                }
            )
    return pd.DataFrame(rows) if rows else None


def _gen_tbl_kin_comparison(s: dict) -> pd.DataFrame | None:
    kin = s.get("kinetic_models_fitted", {})
    if not kin:
        return None
    rows = []
    for m, d in kin.items():
        if not d.get("converged"):
            continue
        rows.append(
            {
                "Model": m,
                "R²": d.get("r_squared", 0),
                "Adj_R²": d.get("adj_r_squared", 0),
                "RMSE": d.get("rmse", 0),
                "AIC": d.get("aicc", d.get("aic", 0)),
            }
        )
    return pd.DataFrame(rows) if rows else None


def _gen_tbl_thermo_params(s: dict) -> pd.DataFrame | None:
    t = s.get("thermo_params")
    if not t:
        return None
    rows = [
        {"Parameter": "ΔH°", "Value": f"{t.get('delta_H', 0):.2f}", "Unit": "kJ/mol"},
        {"Parameter": "ΔS°", "Value": f"{t.get('delta_S', 0):.2f}", "Unit": "J/(mol·K)"},
    ]
    for T, G in t.get("delta_G", {}).items():
        rows.append({"Parameter": f"ΔG° ({T}K)", "Value": f"{G:.2f}", "Unit": "kJ/mol"})
    return pd.DataFrame(rows)


def _gen_tbl_thermo_data(s: dict) -> pd.DataFrame | None:
    return s.get("temp_effect_results")


def _gen_tbl_ph_data(s: dict) -> pd.DataFrame | None:
    return s.get("ph_effect_results")


def _gen_tbl_temp_data(s: dict) -> pd.DataFrame | None:
    return s.get("temp_effect_results")


def _gen_tbl_dosage_data(s: dict) -> pd.DataFrame | None:
    return s.get("dosage_effect_results")


# =============================================================================
# MULTI-STUDY COMPARISON TABLE GENERATORS
# =============================================================================


def _gen_tbl_multi_iso_params(s: dict) -> pd.DataFrame | None:
    """Generate multi-study isotherm parameters comparison table."""
    studies, study_names = _get_all_studies()

    if len(study_names) < 2:
        return None

    rows = []
    for name in study_names:
        data = studies[name]
        iso = data.get("isotherm_models_fitted", {})

        for model_name, results in iso.items():
            if results and results.get("converged"):
                row = {
                    "Study": name,
                    "Model": model_name,
                    "R²": results.get("r_squared", np.nan),
                    "Adj-R²": results.get("adj_r_squared", np.nan),
                    "RMSE": results.get("rmse", np.nan),
                    "AIC": results.get("aicc", results.get("aic", np.nan)),
                }
                params = results.get("params", {})
                if model_name == "Langmuir":
                    row["qm (mg/g)"] = params.get("qm", np.nan)
                    row["KL (L/mg)"] = params.get("KL", np.nan)
                elif model_name == "Freundlich":
                    row["KF"] = params.get("KF", np.nan)
                    row["n"] = params.get("n", np.nan)
                rows.append(row)

    return pd.DataFrame(rows) if rows else None


def _gen_tbl_multi_kin_params(s: dict) -> pd.DataFrame | None:
    """Generate multi-study kinetic parameters comparison table."""
    studies, study_names = _get_all_studies()

    if len(study_names) < 2:
        return None

    rows = []
    for name in study_names:
        data = studies[name]
        kin = data.get("kinetic_models_fitted", {})

        for model_name, results in kin.items():
            if results and results.get("converged"):
                row = {
                    "Study": name,
                    "Model": model_name,
                    "R²": results.get("r_squared", np.nan),
                    "Adj-R²": results.get("adj_r_squared", np.nan),
                    "RMSE": results.get("rmse", np.nan),
                    "AIC": results.get("aicc", results.get("aic", np.nan)),
                }
                params = results.get("params", {})
                if model_name == "PSO":
                    row["qe (mg/g)"] = params.get("qe", np.nan)
                    row["k2 (g/mg·min)"] = params.get("k2", np.nan)
                elif model_name == "rPSO":
                    row["qe (mg/g)"] = params.get("qe", np.nan)
                    row["k2 (g/mg·min)"] = params.get("k2", np.nan)
                    row["φ (correction)"] = params.get("phi", np.nan)
                elif model_name == "PFO":
                    row["qe (mg/g)"] = params.get("qe", np.nan)
                    row["k1 (1/min)"] = params.get("k1", np.nan)
                rows.append(row)

    return pd.DataFrame(rows) if rows else None


def _gen_tbl_multi_thermo(s: dict) -> pd.DataFrame | None:
    """Generate multi-study thermodynamic comparison table."""
    studies, study_names = _get_all_studies()

    if len(study_names) < 2:
        return None

    rows = []
    for name in study_names:
        data = studies[name]
        thermo = data.get("thermo_params", {})

        if thermo:
            delta_G = thermo.get("delta_G_values", [])
            if isinstance(delta_G, list) and len(delta_G) > 0:
                delta_G_val = delta_G[0]
            else:
                delta_G_val = thermo.get("delta_G", np.nan)

            rows.append(
                {
                    "Study": name,
                    "ΔH° (kJ/mol)": thermo.get("delta_H", np.nan),
                    "ΔS° (J/mol·K)": thermo.get("delta_S", np.nan),
                    "ΔG° (kJ/mol)": delta_G_val,
                    "R²": thermo.get("r_squared", np.nan),
                    "Spontaneity": "Yes" if delta_G_val < 0 else "No",
                    "Mechanism": "Endothermic" if thermo.get("delta_H", 0) > 0 else "Exothermic",
                }
            )

    return pd.DataFrame(rows) if rows else None


def _gen_tbl_multi_ranking(s: dict) -> pd.DataFrame | None:
    """Generate multi-study overall ranking table."""
    studies, study_names = _get_all_studies()

    if len(study_names) < 2:
        return None

    rows = []
    for name in study_names:
        data = studies[name]

        iso_scores = []
        kin_scores = []

        iso = data.get("isotherm_models_fitted", {})
        for results in iso.values():
            if results and results.get("converged"):
                iso_scores.append(results.get("r_squared", 0))

        kin = data.get("kinetic_models_fitted", {})
        for results in kin.values():
            if results and results.get("converged"):
                kin_scores.append(results.get("r_squared", 0))

        langmuir = iso.get("Langmuir", {})
        qm = langmuir.get("params", {}).get("qm", 0) if langmuir.get("converged") else 0

        avg_iso = np.mean(iso_scores) * 100 if iso_scores else 0
        avg_kin = np.mean(kin_scores) * 100 if kin_scores else 0
        overall = (avg_iso + avg_kin) / 2 if (iso_scores or kin_scores) else 0

        rows.append(
            {
                "Study": name,
                "qm (mg/g)": qm,
                "Isotherm Avg R² (%)": avg_iso,
                "Kinetic Avg R² (%)": avg_kin,
                "Overall Score (%)": overall,
                "Models Fitted": len(iso_scores) + len(kin_scores),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Overall Score (%)", ascending=False)
        df["Rank"] = range(1, len(df) + 1)
        df = df[
            [
                "Rank",
                "Study",
                "qm (mg/g)",
                "Isotherm Avg R² (%)",
                "Kinetic Avg R² (%)",
                "Overall Score (%)",
                "Models Fitted",
            ]
        ]

    return df if rows else None


def _gen_tbl_multi_pub_summary(s: dict) -> pd.DataFrame | None:
    """Generate publication-ready summary table with all key parameters."""
    studies, study_names = _get_all_studies()

    if len(study_names) < 2:
        return None

    rows = []
    for name in study_names:
        data = studies[name]
        row: dict[str, Any] = {"Study/Adsorbent": name}

        langmuir = data.get("isotherm_models_fitted", {}).get("Langmuir", {})
        freundlich = data.get("isotherm_models_fitted", {}).get("Freundlich", {})

        if langmuir.get("converged"):
            row["qm (mg/g)"] = langmuir["params"].get("qm", np.nan)
            row["KL (L/mg)"] = langmuir["params"].get("KL", np.nan)
            row["R²_Langmuir"] = langmuir.get("r_squared", np.nan)
        else:
            row["qm (mg/g)"] = np.nan
            row["KL (L/mg)"] = np.nan
            row["R²_Langmuir"] = np.nan

        if freundlich.get("converged"):
            row["KF"] = freundlich["params"].get("KF", np.nan)
            row["n"] = freundlich["params"].get("n", np.nan)
        else:
            row["KF"] = np.nan
            row["n"] = np.nan

        pso = data.get("kinetic_models_fitted", {}).get("PSO", {})
        if not pso.get("converged"):
            pso = data.get("kinetic_models_fitted", {}).get("rPSO", {})
        if pso.get("converged"):
            row["qe (mg/g)"] = pso["params"].get("qe", np.nan)
            row["k2 (g/mg·min)"] = pso["params"].get("k2", np.nan)
            row["R²_PSO"] = pso.get("r_squared", np.nan)
        else:
            row["qe (mg/g)"] = np.nan
            row["k2 (g/mg·min)"] = np.nan
            row["R²_PSO"] = np.nan

        thermo = data.get("thermo_params")
        if thermo:
            row["ΔH° (kJ/mol)"] = thermo.get("delta_H", np.nan)
            row["ΔS° (J/mol·K)"] = thermo.get("delta_S", np.nan)
            delta_G_vals = thermo.get("delta_G_values", [])
            row["ΔG° (kJ/mol)"] = delta_G_vals[0] if delta_G_vals else np.nan
        else:
            row["ΔH° (kJ/mol)"] = np.nan
            row["ΔS° (J/mol·K)"] = np.nan
            row["ΔG° (kJ/mol)"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows) if rows else None


def _gen_tbl_multi_mechanism(s: dict) -> pd.DataFrame | None:
    """Generate mechanism interpretation table."""
    studies, study_names = _get_all_studies()

    if len(study_names) < 2:
        return None

    rows = []
    for name in study_names:
        data = studies[name]
        row: dict[str, Any] = {"Study": name}

        freundlich = data.get("isotherm_models_fitted", {}).get("Freundlich", {})
        if freundlich.get("converged"):
            n = freundlich["params"].get("n", 0)
            row["n (Freundlich)"] = n
            if n > 1:
                row["Favorability"] = "Favorable"
            elif n == 1:
                row["Favorability"] = "Linear"
            else:
                row["Favorability"] = "Unfavorable"
        else:
            row["n (Freundlich)"] = np.nan
            row["Favorability"] = "—"

        thermo = data.get("thermo_params")
        if thermo:
            delta_H = thermo.get("delta_H", 0)

            if delta_H > 0:
                row["Process"] = "Endothermic"
            else:
                row["Process"] = "Exothermic"

            abs_H = abs(delta_H)
            if abs_H < 40:
                row["Bonding"] = "Physical"
            elif abs_H < 80:
                row["Bonding"] = "Mixed"
            else:
                row["Bonding"] = "Chemical"
        else:
            row["Process"] = "—"
            row["Bonding"] = "—"

        rows.append(row)

    return pd.DataFrame(rows) if rows else None


# =============================================================================
# ZIP CREATION
# =============================================================================


def create_export_zip(
    study_state: dict, selected_figures: list[str], selected_tables: list[str], config: dict
) -> tuple[bytes, list[str]]:
    """Create a ZIP file with selected figures and tables.

    Returns:
        Tuple of (zip_bytes, list of error messages)
    """
    buffer = io.BytesIO()

    export_errors: list[str] = []

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        export_errors = []  # Collect errors for user feedback

        for fig_id in selected_figures:
            fig = generate_figure(fig_id, study_state)
            if fig:
                try:
                    # Make typography print-aware (avoid tiny tick labels in exports)
                    fig = style_figure_for_export(
                        fig,
                        width_px=int(config["width"]),
                        height_px=int(config["height"]),
                        target_width_in=float(config.get("target_width_in", 6.5)),
                        preset=str(config.get("text_preset", "Manuscript (journal)")),
                        fig_id=str(fig_id),
                        study_state=study_state,
                    )

                    img_bytes = fig.to_image(
                        format="png" if config["format"] == "tiff" else config["format"],
                        width=config["width"],
                        height=config["height"],
                        scale=config["scale"],
                    )

                    # Approximate effective DPI = (rendered pixel width) / (intended width in inches)
                    try:
                        eff_dpi = int(
                            round(
                                (float(config["width"]) * float(config["scale"]))
                                / float(config.get("target_width_in", 6.5))
                            )
                        )
                        if eff_dpi < 72:
                            eff_dpi = 72
                    except Exception:
                        eff_dpi = int(EXPORT_DPI)

                    if config["format"] == "png":
                        img_bytes = _with_dpi_metadata(img_bytes, fmt="png", dpi=eff_dpi)

                    if config["format"] == "tiff":
                        # Plotly cannot export TIFF directly; convert PNG -> TIFF and embed DPI
                        png_buffer = io.BytesIO(img_bytes)
                        img = Image.open(png_buffer)
                        tiff_buffer = io.BytesIO()
                        img.save(
                            tiff_buffer,
                            format="TIFF",
                            compression="tiff_lzw",
                            dpi=(eff_dpi, eff_dpi),
                        )
                        tiff_buffer.seek(0)
                        img_bytes = tiff_buffer.getvalue()


                    zf.writestr(f"figures/{fig_id}.{config['format']}", img_bytes)
                except Exception as e:
                    error_msg = f"Figure '{fig_id}': {str(e)}"
                    export_errors.append(error_msg)
                    logger.error(f"Failed to export figure: {error_msg}")

        for tbl_id in selected_tables:
            df = generate_table(tbl_id, study_state)
            if df is not None:
                zf.writestr(f"tables/{tbl_id}.csv", df.to_csv(index=False, sep=";"))
                excel_buf = io.BytesIO()
                df.to_excel(excel_buf, index=False, engine="openpyxl")
                excel_buf.seek(0)
                zf.writestr(f"tables/{tbl_id}.xlsx", excel_buf.getvalue())

        readme = f"""AdsorbLab Pro v2.0.0 Export
===========================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Contents:
- figures/: Figures ({config["format"].upper()} format)
- tables/: Data tables (CSV and Excel formats)

Figure Settings:
- Format: {config["format"].upper()}
- Dimensions: {config["width"]}x{config["height"]} pixels
- Scale: {config["scale"]}x
- Text preset: {config.get('text_preset', 'Manuscript (journal)')}
- Intended width: {config.get('target_width_in', 6.5)} in

Selected Figures: {len(selected_figures)}
Selected Tables: {len(selected_tables)}
"""
        zf.writestr("README.txt", readme)

    buffer.seek(0)
    return buffer.getvalue(), export_errors


# =============================================================================
# AVAILABILITY CHECK
# =============================================================================


def get_available_items(study_state: dict) -> tuple[dict, dict]:
    """Check which figures and tables are available based on data."""

    available_figures = {}
    available_tables = {}

    for category, cat_data in FIGURE_CATEGORIES.items():
        # Handle 3D Explorer specially - use saved figures
        if category == "3d_explorer":
            saved_3d = get_saved_3d_figures(study_state)
            if saved_3d:
                available_figures[category] = {
                    "name": "3D Explorer (Saved)",
                    "icon": "🔮",
                    "items": saved_3d,
                }
            continue

        # Standard categories
        available_in_cat = []
        for fig_id, name, desc in cat_data["items"]:
            fig = generate_figure(fig_id, study_state)
            if fig is not None:
                available_in_cat.append((fig_id, name, desc))
        if available_in_cat:
            available_figures[category] = {
                "name": cat_data["name"],
                "icon": cat_data["icon"],
                "items": available_in_cat,
            }

    for category, cat_data in TABLE_CATEGORIES.items():
        available_in_cat = []
        for tbl_id, name, desc in cat_data["items"]:
            df = generate_table(tbl_id, study_state)
            if df is not None and not df.empty:
                available_in_cat.append((tbl_id, name, desc))
        if available_in_cat:
            available_tables[category] = {"name": cat_data["name"], "items": available_in_cat}

    return available_figures, available_tables


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================


def render():
    """Render the comprehensive export tab."""
    st.header("📦 Export All Figures and Tables")
    st.markdown("*Download all your analysis results in a single compressed file*")

    if not KALEIDO_AVAILABLE:
        st.error("Kaleido not installed! Install with: pip install -U kaleido")
        return

    current_study_state = get_current_study_state()
    if not current_study_state:
        st.info("Please add or select a study from the sidebar to export results.")
        return

    available_figures, available_tables = get_available_items(current_study_state)

    total_figures = sum(len(cat["items"]) for cat in available_figures.values())
    total_tables = sum(len(cat["items"]) for cat in available_tables.values())

    if total_figures == 0 and total_tables == 0:
        st.warning("No exportable items found. Complete some analyses first!")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Available Figures", total_figures)
    with col2:
        st.metric("Available Tables", total_tables)
    with col3:
        st.metric("Categories", len(available_figures) + len(available_tables))

    # Auto-select all available items
    selected_figs = [fig_id for cat in available_figures.values() for fig_id, _, _ in cat["items"]]
    selected_tbls = [tbl_id for cat in available_tables.values() for tbl_id, _, _ in cat["items"]]

    # Show what will be exported
    st.markdown("---")
    st.markdown("### 📋 Items to Export")
    st.caption("All available figures and tables will be included in the export package.")

    col_fig, col_tbl = st.columns(2)

    with col_fig:
        st.markdown("#### 📊 Figures")
        for _category, cat_data in available_figures.items():
            with st.expander(
                f"{cat_data['icon']} {cat_data['name']} ({len(cat_data['items'])})", expanded=False
            ):
                for _fig_id, name, desc in cat_data["items"]:
                    st.markdown(f"• **{name}** - _{desc}_")

    with col_tbl:
        st.markdown("#### 📋 Tables")
        for _category, cat_data in available_tables.items():
            with st.expander(f"📋 {cat_data['name']} ({len(cat_data['items'])})", expanded=False):
                for _tbl_id, name, desc in cat_data["items"]:
                    st.markdown(f"• **{name}** - _{desc}_")

    st.markdown("---")
    st.markdown("### ⚙️ Export Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Image Format:**")
        fig_format = st.selectbox(
            "Format",
            options=["tiff", "png", "svg", "pdf"],
            index=0,
            help="TIFF: High quality | PNG: Web compatible | SVG: Vector | PDF: Print",
        )
        # Define options using EXPORT_DPI as reference
        dpi_options = ["Standard (150 DPI)", f"High ({EXPORT_DPI} DPI)", "High Quality (600 DPI)"]
        default_index = 1  # High quality as default

        quality_preset = st.selectbox("Quality Preset", options=dpi_options, index=default_index)
        scale_map = {dpi_options[0]: 150/96, dpi_options[1]: EXPORT_DPI/96, dpi_options[2]: 600/96}

    with col2:
        st.markdown("**Figure Dimensions:**")
        width = st.slider("Width (pixels)", 800, 2400, 1200, 100)
        height = st.slider("Height (pixels)", 600, 1800, 800, 100)
        st.markdown("**Typography (export only):**")
        text_preset = st.selectbox(
            "Text sizing preset",
            options=["Manuscript (journal)", "Presentation (slides)", "Poster"],
            index=0,
            help="Export figures with professional, readable typography. On-screen figures are unchanged.",
        )
        target_width_in = st.slider(
            "Intended figure width (inches)",
            min_value=3.0,
            max_value=10.0,
            value=6.5,
            step=0.25,
            help="Used to scale text so it looks correct when inserted in Word/LaTeX at this width.",
        )


    st.markdown("---")
    st.markdown("---")
    st.markdown("### 📥 Export")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.metric("Figures", len(selected_figs))

    with col3:
        st.metric("Tables", len(selected_tbls))

    with col2:
        export_type = st.radio(
            "Export type",
            options=["ZIP package (figures + tables)", "Word report (.docx)"],
            index=0,
            horizontal=True,
            help="ZIP includes selected figures + tables as files. Word report embeds figures + tables into a manuscript-ready .docx.",
        )

        # Report-only options
        report_fig_width_in = 6.5
        report_max_table_rows = 60
        if export_type.startswith("Word"):
            if not DOCX_AVAILABLE:
                st.error("python-docx is not installed. Install with: pip install python-docx")
            report_fig_width_in = st.slider(
                "Figure width in report (inches)",
                min_value=4.5,
                max_value=7.5,
                value=6.5,
                step=0.1,
                help="Controls how wide embedded figures appear on the page.",
            )
            report_max_table_rows = int(
                st.number_input(
                    "Max rows per table in report",
                    min_value=10,
                    max_value=500,
                    value=60,
                    step=10,
                    help="Large tables are truncated to keep the Word file responsive.",
                )
            )

        button_label = (
            "📦 Generate ZIP Package"
            if export_type.startswith("ZIP")
            else "📝 Generate Word Report"
        )
        disabled = export_type.startswith("Word") and (not DOCX_AVAILABLE)

        if st.button(button_label, type="primary", use_container_width=True, disabled=disabled):
            export_config = {
                "format": fig_format,
                "width": width,
                "height": height,
                "scale": scale_map[quality_preset],
                "text_preset": text_preset,
                "target_width_in": float(target_width_in),
            }

            try:
                if export_type.startswith("ZIP"):
                    with st.spinner(
                        f"Generating {len(selected_figs)} figures and {len(selected_tbls)} tables..."
                    ):
                        zip_bytes, export_errors = create_export_zip(
                            current_study_state, selected_figs, selected_tbls, export_config
                        )

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"adsorblab_export_{timestamp}.zip"

                    st.download_button(
                        label="⬇️ Download ZIP Package",
                        data=zip_bytes,
                        file_name=filename,
                        mime="application/zip",
                        use_container_width=True,
                    )

                    # Show export status
                    if export_errors:
                        st.warning(f"⚠️ {len(export_errors)} item(s) could not be exported:")
                        for err in export_errors[:5]:
                            st.caption(f"  • {err}")
                        if len(export_errors) > 5:
                            st.caption(f"  ... and {len(export_errors) - 5} more")
                        st.success(
                            f"✅ Partial export ready ({len(selected_figs) - len(export_errors)} figures, {len(selected_tbls)} tables)"
                        )
                    else:
                        st.success(
                            f"✅ Export package ready! ({len(selected_figs)} figures, {len(selected_tbls)} tables)"
                        )

                else:
                    # Build metadata maps for captions
                    fig_meta: dict[str, tuple[str, str]] = {}
                    for cat in available_figures.values():
                        for fid, name, desc in cat.get("items", []):
                            fig_meta[str(fid)] = (str(name), str(desc))

                    tbl_meta: dict[str, tuple[str, str]] = {}
                    for cat in available_tables.values():
                        for tid, name, desc in cat.get("items", []):
                            tbl_meta[str(tid)] = (str(name), str(desc))

                    # DOCX always embeds raster images
                    doc_cfg = DocxReportConfig(
                        img_format="png",
                        img_width_px=int(export_config["width"]),
                        img_height_px=int(export_config["height"]),
                        img_scale=float(export_config["scale"]),
                        figure_width_in=float(report_fig_width_in),
                        max_table_rows=int(report_max_table_rows),
                    )

                    study_title = (
                        f"AdsorbLab Pro — {st.session_state.get('current_study', 'Study Report')}"
                    )

                    def _docx_figure_generator(fid: str, st_state: dict) -> Any:
                        fig_obj = generate_figure(fid, st_state)
                        return style_figure_for_export(
                            fig_obj,
                            width_px=int(doc_cfg.img_width_px),
                            height_px=int(doc_cfg.img_height_px),
                            target_width_in=float(doc_cfg.figure_width_in),
                            preset=str(export_config.get("text_preset", "Manuscript (journal)")),
                            fig_id=str(fid),
                            study_state=st_state,
                        )


                    with st.spinner(
                        f"Generating Word report with {len(selected_figs)} figures and {len(selected_tbls)} tables..."
                    ):
                        docx_bytes, report_warnings = create_docx_report(
                            study_title=study_title,
                            study_state=current_study_state,
                            selected_figures=selected_figs,
                            selected_tables=selected_tbls,
                            figure_generator=_docx_figure_generator,
                            table_generator=generate_table,
                            figure_meta=fig_meta,
                            table_meta=tbl_meta,
                            config=doc_cfg,
                        )

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"adsorblab_report_{timestamp}.docx"

                    st.download_button(
                        label="⬇️ Download Word Report (.docx)",
                        data=docx_bytes,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                    )

                    if report_warnings:
                        st.warning(f"⚠️ {len(report_warnings)} note(s) while generating the report:")
                        for w in report_warnings[:8]:
                            st.caption(f"  • {w}")
                        if len(report_warnings) > 8:
                            st.caption(f"  ... and {len(report_warnings) - 8} more")
                        st.success("✅ Report generated (with notes).")
                    else:
                        st.success("✅ Report generated successfully!")

            except Exception as e:
                st.error(f"Error generating export: {str(e)}")