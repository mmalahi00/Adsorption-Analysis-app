# AdsorbLab Pro

**Professional Adsorption Data Analysis Platform**

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://adsorption.streamlit.app)
[![CI](https://github.com/mmalahi00/Adsorption-Analysis-app/actions/workflows/ci.yml/badge.svg)](https://github.com/mmalahi00/Adsorption-Analysis-app/actions/workflows/ci.yml)

AdsorbLab Pro is a comprehensive, browser-based tool for analyzing adsorption experiments. It fits isotherm and kinetic models using non-linear regression, provides bootstrap confidence intervals, performs rigorous model comparison (R², Adj-R², AIC, AICc, BIC, F-test), and generates publication-ready figures and Word reports — all without writing a single line of code.

> **Try it now →** [adsorption.streamlit.app](https://adsorption.streamlit.app)

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Equations](#model-equations)
- [Statistical Methods](#statistical-methods)
- [Project Structure](#project-structure)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Features

### Isotherm Models

| Model | Best for | Parameters |
|-------|----------|------------|
| **Langmuir** | Monolayer, homogeneous surfaces | qₘ, Kₗ |
| **Freundlich** | Heterogeneous surfaces, multilayer | Kf, n |
| **Temkin** | Adsorbate–adsorbate interactions | B₁, Kₜ |
| **Sips** | Heterogeneous at high C, Langmuir at low C | qₘ, Kₛ, nₛ |

### Kinetic Models

| Model | Mechanism | Parameters |
|-------|-----------|------------|
| **Pseudo-First Order** | Physisorption | qₑ, k₁ |
| **Pseudo-Second Order** | Chemisorption | qₑ, k₂, h |
| **Revised PSO** (Bullen et al. 2021) | Concentration-corrected PSO | qₑ, k₂, C₀ |
| **Elovich** | Heterogeneous chemisorption | α, β |
| **Intraparticle Diffusion** | Pore diffusion (Weber-Morris) | kᵢₚ, C |

### Multi-Component Competitive Adsorption

Predict how multiple adsorbates compete for the same binding sites — critical for real wastewater and multi-solute systems.

| Model | Use case |
|-------|----------|
| **Extended Langmuir** (Butler-Ockrent) | Binary/multi-solute systems with known single-component parameters |
| **Extended Freundlich** (SRS) | Heterogeneous surfaces with competition coefficients |

Includes selectivity coefficient (αᵢⱼ) calculation, the ability to link single-component fits or enter parameters manually, per-component bar charts, and automated interpretation of competitive effects.

### 3D Parameter Space Explorer

Visualise how adsorption responds to two variables at once (e.g. Cₑ × T → qₑ, or pH × T → Removal %). Fully interactive Plotly 3D surfaces that can be exported as static images or embedded in the Word report.

### Thermodynamics

Van't Hoff analysis across multiple temperatures yielding ΔG°, ΔH°, and ΔS°.

### Additional Capabilities

- **Calibration** — UV-Vis Beer–Lambert calibration with linearity diagnostics
- **Effect studies** — pH, adsorbent dosage, and temperature optimization
- **Diffusion analysis** — Biot number, Boyd plot, and Weber-Morris multilinearity for rate-limiting step identification
- **Multi-study comparison** across datasets

### Statistical Rigour

- Non-linear regression (not linearized transforms)
- 95 % confidence intervals on all parameters
- Bootstrap CI (500–1000 iterations)
- Model selection via Adj-R², AIC/AICc, BIC, and Akaike weights
- PRESS/Q² leave-one-out cross-validation
- Residual diagnostics (Shapiro-Wilk, Durbin-Watson)
- Weighted Least Squares (1/y, 1/y², √y schemes)

### Export

- **ZIP package** — selected figures (PNG/SVG/PDF) + tables (CSV/XLSX)
- **Word report (.docx)** — embedded figures, formatted tables, captions, and a methods summary

---

## Installation

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.11 or 3.12 |
| RAM | 4 GB | 8 GB |
| Storage | 500 MB | 1 GB |
| OS | Windows 10, macOS 10.14, Ubuntu 20.04 | Latest |

### Streamlit Cloud (no install)

Visit [adsorption.streamlit.app](https://adsorption.streamlit.app) and upload your data.

### Local Install

```bash
# 1. Clone the repository
git clone https://github.com/mmalahi00/Adsorption-Analysis-app.git
cd Adsorption-Analysis-app

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch the app
streamlit run adsorption_app.py
```

The app opens at `http://localhost:8501`.

Alternative launch methods:

```bash
streamlit run adsorblab_pro/app.py   # package entry point
python -m adsorblab_pro              # module mode
pip install -e . && adsorblab        # CLI shortcut after editable install
```

### Docker

```bash
docker compose up --build          # production on port 8501
docker compose --profile dev up    # dev mode with hot-reload on port 8502
```

---

## Quick Start

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Calibration │ -> │  Isotherm   │ -> │   Kinetic   │ -> │  Thermo-    │
│   Curve     │    │  Analysis   │    │  Analysis   │    │  dynamics   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       v                  v                  v                  v
   slope, R²         qₘ, Kₗ, n          qₑ, k₂, h         ΔH°, ΔS°, ΔG°
```

1. **Calibration** — Upload UV-Vis standards → Beer-Lambert parameters
2. **Isotherm** — Input C₀ and absorbances → Fit 4 models → Compare via AIC
3. **Kinetics** — Time-series data → Fit models → Identify mechanism
4. **Thermodynamics** — Multi-temperature data → Van't Hoff → ΔH°, ΔS°, ΔG°
5. **Statistical Summary** — Review diagnostics → Export report

For direct concentration data (e.g. from the literature), select **"Direct Concentration"** in the sidebar and skip calibration.

### Example Data

The `examples/` directory contains ready-to-use datasets for every tab, in both Standard (absorbance) and Direct (concentration) modes. `expected_results.json` provides validation benchmarks. Three fully documented case studies are under `case_studies/`.

---

## Model Equations

### Isotherm Models

**Langmuir (1918)** — monolayer adsorption on a homogeneous surface with finite identical sites.

```
qₑ = (qₘ · Kₗ · Cₑ) / (1 + Kₗ · Cₑ)
```

Separation factor Rₗ = 1/(1 + Kₗ·C₀): Rₗ = 0 irreversible, 0 < Rₗ < 1 favourable, Rₗ = 1 linear, Rₗ > 1 unfavourable.

**Freundlich (1906)** — heterogeneous surfaces with non-uniform energy distribution.

```
qₑ = Kf · Cₑ^(1/n)
```

n > 1 favourable, n = 1 linear, n < 1 unfavourable.

**Temkin (1940)** — heat of adsorption decreases linearly with coverage.

```
qₑ = B₁ · ln(Kₜ · Cₑ)        where B₁ = RT / bₜ
```

**Sips (Langmuir-Freundlich)** — hybrid; reduces to Langmuir when nₛ = 1.

```
qₑ = qₘ · (Kₛ · Cₑ)^nₛ / [1 + (Kₛ · Cₑ)^nₛ]
```

### Kinetic Models

**Pseudo-First Order (Lagergren 1898)**

```
qₜ = qₑ · (1 − e^(−k₁·t))
```

**Pseudo-Second Order (Ho & McKay 1999)**

```
qₜ = (qₑ² · k₂ · t) / (1 + qₑ · k₂ · t)        h = k₂ · qₑ²  (initial rate)
```

**Elovich**

```
qₜ = (1/β) · ln(1 + α·β·t)
```

**Intraparticle Diffusion (Weber-Morris)**

```
qₜ = kᵢₚ · √t + C
```

If C = 0, intraparticle diffusion is the sole rate-limiting step.

### Thermodynamics

**Van't Hoff:**  `ln(Kd) = ΔS°/R − ΔH°/(RT)` — plot ln(Kd) vs 1/T to obtain slope = −ΔH°/R and intercept = ΔS°/R.

**Gibbs free energy:**  `ΔG° = −RT·ln(Kd) = ΔH° − T·ΔS°`

---

## Statistical Methods

| Criterion | Purpose |
|-----------|---------|
| R² | Goodness of fit (0–1) |
| Adj. R² | Penalises extra parameters |
| AIC / AICc | Model selection (lower = better); AICc for small samples |
| BIC | Stricter parameter penalty than AIC |
| Q² (PRESS) | Predictive ability via leave-one-out cross-validation |

**Bootstrap confidence intervals** — residuals are resampled 500–1000 times; the model is refit each iteration and the 2.5th/97.5th percentiles are reported.

**PRESS / Q²:**

```
PRESS = Σ(yᵢ − ŷᵢ₍₋ᵢ₎)²
Q²    = 1 − PRESS / SStot
```

Q² > 0.5 indicates good predictive ability.

---

## Project Structure

```
Adsorption-Analysis-app/
├── adsorption_app.py              # Root Streamlit launcher
├── adsorblab_pro/
│   ├── app.py                     # Streamlit entry point
│   ├── app_main.py                # Main UI and routing
│   ├── config.py                  # Constants and configuration
│   ├── models.py                  # Isotherm and kinetic models
│   ├── utils.py                   # Calculations, bootstrap, statistics
│   ├── validation.py              # Input validation and diagnostics
│   ├── sidebar_ui.py              # Sidebar controls
│   ├── plot_style.py              # Publication-quality plot styling
│   ├── docx_report.py             # Word report generator
│   ├── streamlit_compat.py        # Streamlit compatibility shim
│   └── tabs/                      # One module per analysis tab
│       ├── home_tab.py
│       ├── calibration_tab.py
│       ├── isotherm_tab.py
│       ├── kinetic_tab.py
│       ├── thermodynamics_tab.py
│       ├── temperature_tab.py
│       ├── ph_effect_tab.py
│       ├── dosage_tab.py
│       ├── competitive_tab.py
│       ├── comparison_tab.py
│       ├── statistical_summary_tab.py
│       ├── threed_explorer_tab.py
│       └── report_tab.py
├── examples/                      # Sample datasets
├── case_studies/                   # Reproducible case studies
├── tests/                         # Test suite
├── docs/
│   └── USER_GUIDE.md
├── scripts/                       # Cleanup utilities
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── requirements-lock.txt
├── Dockerfile
├── docker-compose.yml
├── CITATION.cff
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
└── LICENSE
```

---

## Development

```bash
# Install runtime + dev dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Run the test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=adsorblab_pro --cov-report=html

# Lint
ruff check .

# Type checking
mypy adsorblab_pro/
```

For reproducible builds, use `requirements-lock.txt`. See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

### Production Deployment

Before packaging or deploying from a ZIP checkout, clean build/test artefacts:

```bash
bash scripts/clean_artifacts.sh                                       # macOS / Linux
powershell -ExecutionPolicy Bypass -File scripts/clean_artifacts.ps1  # Windows
```

---

## Troubleshooting

**pip install fails with compilation errors**

```bash
# Windows: install Visual C++ Build Tools
# macOS:   xcode-select --install
# Linux:   sudo apt-get install build-essential python3-dev
```

**Port 8501 already in use**

```bash
streamlit run adsorption_app.py --server.port 8502
```

**DOCX export option is disabled** — install the dependency and restart Streamlit: `pip install python-docx`. If you see an lxml ImportError, run `pip install -U pip setuptools wheel`.

**Fitting fails to converge** — check for outliers, verify Cₑ < C₀, try a simpler model first, or adjust initial parameter guesses.

**Bootstrap CI very wide** — ensure at least 6–8 data points, check for outliers, and consider whether the chosen model is appropriate.

### Data Quality Checklist

- Cₑ ≤ C₀ for all points
- No negative concentrations or capacities
- ≥ 5 points for isotherms, ≥ 8 for kinetics
- Consistent units (mg/L, g, L, min)
- Temperature in Kelvin for thermodynamic analysis

---

## Citation

If you use AdsorbLab Pro in your research, please cite:

```bibtex
@software{adsorblab_pro_2026,
  title   = {{AdsorbLab Pro}: Professional Adsorption Data Analysis Platform},
  author  = {{Mohamed EL MALLAHI}},
  year    = {2026},
  version = {2.0.0},
  url     = {https://github.com/mmalahi00/Adsorption-Analysis-app},
  license = {MIT}
}
```

---

## License

[MIT](LICENSE) © Mohamed EL MALLAHI

---

## Support

- [Report a bug](https://github.com/mmalahi00/Adsorption-Analysis-app/issues)
- [Request a feature](https://github.com/mmalahi00/Adsorption-Analysis-app/issues)