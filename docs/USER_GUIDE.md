# AdsorbLab Pro — User Guide

This guide focuses on running adsorption analyses and exporting publication-ready outputs.

## 1) Running the app

**Local**
1. Create/activate a virtualenv
2. Install dependencies:
   - `pip install -r requirements.txt`
   - (Reproducible) `pip install -r requirements-lock.txt`
3. Start (any one of these):
   - `streamlit run adsorption_app.py`     — recommended (root launcher)
   - `streamlit run adsorblab_pro/app.py`  — run the package entry point directly
   - `python -m adsorblab_pro`             — run as a Python module
   - `adsorblab`                           — shortcut (available after `pip install -e .`)

**Docker**
- `docker compose up --build`

## 2) Typical workflow

1. **Calibration tab**: Convert absorbance to concentration (Beer–Lambert).
2. **Isotherm tab**: Fit common isotherm models and compare fits.
3. **Kinetic tab**: Fit kinetic models and identify trends/mechanisms.
4. **Thermodynamics tab**: Van’t Hoff analysis across temperatures.
5. **Statistical Summary tab**: Review assumptions, diagnostics, and the data-quality checklist.
6. **Export tab**: Export figures/tables for publication.

## 3) Exporting results

You can export either a **ZIP package** (figures + tables) or a **Word report (.docx)**.

### A) ZIP package export (figures + tables)

In **Export**:
- Select figures and tables.
- Choose **image format** (PNG/SVG/PDF, depending on availability).
- Choose size and a quality preset.

Output is a ZIP containing:
- Image files (one per selected figure)
- Tables as CSV/XLSX (depending on selection/options)

### B) Word report export (.docx)

In **Export**:
- Select figures and tables.
- Set:
  - **Embedded figure width (in)**
  - **Max rows per table in report** (tables are truncated beyond this)
- Choose **Export type → Word report (.docx)** and generate.

The report includes:
- Title + timestamp
- A short methods summary (from current study state)
- Selected tables as Word tables
- Selected figures embedded as static images with captions
- A notes section listing anything that could not be embedded

**Notes**
- Figures are embedded as static images (best for publication).
- Very large exports may produce a large DOCX. Reduce the number of figures or lower export scale.

## 4) DOCX configuration options (advanced)

The Word export uses a `DocxReportConfig` object (see `adsorblab_pro/docx_report.py`).
Most users will only touch the UI settings, but you can tune defaults in code.

**Key options**
- `img_format` (default: `"png"`)
- `img_width_px`, `img_height_px` (default: 1400×900)
- `img_scale` (default: 2.0)
- `figure_width_in` (default: 6.5)
- `max_table_rows` (default: 60)
- `max_table_cols` (default: 12)
- `float_format` (default: `"{:.4g}"`)

## 5) Troubleshooting (DOCX export)

**DOCX export option is disabled**
- Install the dependency:
  - `pip install python-docx`
- Then restart Streamlit.

**ImportError related to `lxml`**
- Upgrade packaging tools:
  - `python -m pip install -U pip setuptools wheel`
- On minimal Linux images you may need system libs for lxml wheels/builds.

**Report is slow or the file is huge**
- Export fewer figures.
- Reduce image scale/size in export settings.
- Lower **max rows per table**.
