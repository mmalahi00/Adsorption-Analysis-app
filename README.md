# AdsorbLab Pro v2.0.0

## Publication-Ready Adsorption Data Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/mmalahi00/Adsorption-Analysis-app/actions/workflows/ci.yml/badge.svg)](https://github.com/mmalahi00/Adsorption-Analysis-app/actions/workflows/ci.yml)

AdsorbLab Pro is a comprehensive Streamlit-based application for analyzing adsorption equilibrium and kinetic data with statistical rigor. Designed by researchers, for researchers.

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Model Equations](#-model-equations)
- [Statistical Methods](#-statistical-methods)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Key Features

### Isotherm Models (4)
| Model | Best For | Parameters |
|-------|----------|------------|
| **Langmuir** | Monolayer, homogeneous surfaces | qₘ, Kₗ |
| **Freundlich** | Heterogeneous surfaces, multilayer | Kf, n |
| **Temkin** | Adsorbate-adsorbate interactions | B₁, Kₜ |
| **Sips** | Heterogeneous at high C, Langmuir at low C | qₘ, Kₛ, nₛ |

### Kinetic Models (4+)
| Model | Mechanism | Parameters |
|-------|-----------|------------|
| **Pseudo-First Order** | Physisorption | qₑ, k₁ |
| **Pseudo-Second Order** | Chemisorption | qₑ, k₂, h |
| **Elovich** | Heterogeneous chemisorption | α, β |
| **Intraparticle Diffusion** | Pore diffusion | kᵢₚ, C |

### Statistical Excellence
- ✅ **Non-linear regression** (not linearized transforms)
- ✅ **95% confidence intervals** on all parameters
- ✅ **Adjusted R²** for fair model comparison
- ✅ **AIC/BIC/AICc** for model selection with Akaike weights
- ✅ **PRESS/Q²** leave-one-out cross-validation
- ✅ **Bootstrap CI** (500-1000 iterations)
- ✅ **Residual diagnostics** (Shapiro-Wilk, Durbin-Watson)
- ✅ **Weighted Least Squares** (1/y, 1/y², √y schemes)

### 🔬 Multi-Component Competitive Adsorption *(new in v2)*

Predict how multiple adsorbates compete for the same binding sites — critical for real wastewater and multi-solute systems.

| Model | Equation | Use Case |
|-------|----------|----------|
| **Extended Langmuir** (Butler-Ockrent) | qₑ,ᵢ = qₘ,ᵢ Kₗ,ᵢ Cₑ,ᵢ / (1 + Σ Kₗ,ⱼ Cₑ,ⱼ) | Binary/multi-solute systems with known single-component parameters |
| **Extended Freundlich** (SRS) | qₑ,ᵢ = Kf,ᵢ Cₑ,ᵢ (Σ aᵢⱼ Cₑ,ⱼ)^(1/nᵢ − 1) | Heterogeneous surfaces with competition coefficients |

- **Selectivity coefficient** (αᵢⱼ) calculation for preferential uptake analysis
- Link single-component fits from existing studies **or** enter parameters manually
- Side-by-side per-component bar charts and a combined comparison plot
- Automated interpretation of competitive effects (suppression, enhancement, synergy)

### 📊 3D Parameter Space Explorer *(new in v2)*

Visualise how adsorption responds to **two variables at once** — no scripting required.

| Surface | X-axis | Y-axis | Z-axis |
|---------|--------|--------|--------|
| **Langmuir–Temperature** | Cₑ | T (K) | qₑ |
| **pH–Temperature Response** | pH | T (K) | Removal % |
| **Generic Parameter Sweep** | Any model param | Any model param | qₑ or qt |

- Fully interactive Plotly 3D: rotate, zoom, hover to read exact (x, y, z) values
- **Experimental design aid**: identify optimal (pH, T, dose) combinations before running costly batch tests
- Export surfaces as static images (PNG/SVG) or embed in the Word report

### Additional Advanced Features
- 📑 **Auto-Reports**: Word document generation with embedded figures, tables, and captions
- 🌡️ **Thermodynamics**: Van't Hoff analysis with Davies activity coefficient corrections
- 🧪 **Revised PSO (rPSO)**: Concentration-corrected kinetic model (Bullen et al., 2021) that addresses the well-known PSO artifact
- 📈 **Diffusion Analysis**: Biot number, Boyd plot, and Weber-Morris multilinearity for rate-limiting step identification

---

## 📥 Exporting for Publication

In the **Export** tab you can generate:

- **ZIP package**: selected figures + tables as files (PNG/SVG/PDF + CSV/XLSX depending on selections)
- **Word report (.docx)**: a manuscript-ready report with embedded figures, tables, captions, and notes

### Word report settings

When **Export type → Word report (.docx)** is selected:
- **Embedded figure width (in)** controls how wide figures appear in the document
- **Max rows per table in report** truncates very large tables to keep the report responsive

For advanced tuning (image size/scale, numeric formatting), see `DocxReportConfig` in `adsorblab_pro/docx_report.py`
or the full guide in `docs/USER_GUIDE.md`.

## 📦 Installation

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.11 or 3.12 |
| RAM | 4 GB | 8 GB |
| Storage | 500 MB | 1 GB |
| OS | Windows 10, macOS 10.14, Ubuntu 20.04 | Latest |

> **Supported Python: 3.10+**

### Step-by-Step Installation

```bash
# 1. Clone or download the repository
git clone https://github.com/mmalahi00/Adsorption-Analysis-app.git
cd Adsorption-Analysis-app

# 2. Create virtual environment (HIGHLY RECOMMENDED)
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run the application (either command works)
streamlit run adsorption_app.py          # recommended root launcher
# or
streamlit run adsorblab_pro/app.py       # package entry point
# or
python -m adsorblab_pro                  # module mode (no streamlit command needed)
```

---

## 🚀 Quick Start

### Workflow Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Calibration │ -> │  Isotherm   │ -> │   Kinetic   │ -> │  Thermo-    │
│   Curve     │    │  Analysis   │    │  Analysis   │    │  dynamics   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       v                  v                  v                  v
   slope, R²         qₘ, Kₗ, n          qₑ, k₂, h         ΔH°, ΔS°, ΔG°
```

1. **Calibration Tab**: Enter UV-Vis data → Get Beer-Lambert parameters
2. **Isotherm Tab**: Input C₀ and absorbances → Fit 4 models → Compare AIC
3. **Kinetic Tab**: Time-series data → Fit 4 models → Identify mechanism
4. **Thermodynamics Tab**: Multi-temperature → Van't Hoff → ΔH°, ΔS°, ΔG°
5. **Statistical Summary**: Review checklist → Export report

---

## 📐 Model Equations

### Isotherm Models

#### Langmuir (1918)
Monolayer adsorption on homogeneous surface with finite identical sites.

```
qₑ = (qₘ · Kₗ · Cₑ) / (1 + Kₗ · Cₑ)
```

| Parameter | Description | Units |
|-----------|-------------|-------|
| qₘ | Maximum monolayer capacity | mg/g |
| Kₗ | Langmuir constant (affinity) | L/mg |
| Rₗ | Separation factor = 1/(1+Kₗ·C₀) | dimensionless |

**Separation Factor Interpretation:**
- Rₗ = 0: Irreversible
- 0 < Rₗ < 1: Favorable ✓
- Rₗ = 1: Linear
- Rₗ > 1: Unfavorable

#### Freundlich (1906)
Heterogeneous surfaces with non-uniform energy distribution.

```
qₑ = Kf · Cₑ^(1/n)
```

| Parameter | Description | Units |
|-----------|-------------|-------|
| Kf | Freundlich constant | (mg/g)(L/mg)^(1/n) |
| n | Heterogeneity factor | dimensionless |

**Interpretation:** n > 1 = Favorable, n = 1 = Linear, n < 1 = Unfavorable

#### Temkin (1940)
Heat of adsorption decreases linearly with coverage.

```
qₑ = B₁ · ln(Kₜ · Cₑ)
```

where B₁ = RT/bₜ (bₜ = Temkin constant, J/mol)

#### Sips (Langmuir-Freundlich)
Hybrid: Freundlich at low C, Langmuir at high C.

```
qₑ = qₘ · (Kₛ · Cₑ)^nₛ / [1 + (Kₛ · Cₑ)^nₛ]
```

When nₛ = 1, reduces to Langmuir.

### Kinetic Models

#### Pseudo-First Order (Lagergren, 1898)
```
qₜ = qₑ · (1 - e^(-k₁·t))
```

#### Pseudo-Second Order (Ho & McKay, 1999)
```
qₜ = (qₑ² · k₂ · t) / (1 + qₑ · k₂ · t)
h = k₂ · qₑ²  (initial rate)
```

#### Elovich
```
qₜ = (1/β) · ln(1 + α·β·t)
```

#### Intraparticle Diffusion (Weber-Morris)
```
qₜ = kᵢₚ · √t + C
```

If C = 0, diffusion is sole rate-limiting step.

---

### Thermodynamic Equations

#### Van't Hoff
```
ln(Kd) = ΔS°/R - ΔH°/(RT)
```

Plot ln(Kd) vs 1/T: slope = -ΔH°/R, intercept = ΔS°/R

#### Gibbs Free Energy
```
ΔG° = -RT·ln(Kd) = ΔH° - T·ΔS°
```

---

## 📊 Statistical Methods

### Model Selection

| Criterion | Use |
|-----------|-----|
| **R²** | Goodness of fit (0-1) |
| **Adj. R²** | Penalizes extra parameters |
| **AIC** | Model selection (lower = better) |
| **AICc** | Small sample correction |
| **BIC** | Stricter parameter penalty |
| **Q²** | Predictive ability (PRESS-based) |

### Bootstrap CI
- Resample residuals 500-1000 times
- Refit model each iteration
- Report 2.5th and 97.5th percentiles

### PRESS/Q²
```
PRESS = Σ(yᵢ - ŷᵢ₍₋ᵢ₎)²
Q² = 1 - PRESS/SStot
```
Q² > 0.5 indicates good predictive ability.

---

## 📁 Project Structure

```
MonAppAdsorption/
├── adsorption_app.py              # Root Streamlit launcher (recommended)
├── adsorblab_pro/
│   ├── app.py                     # Streamlit entrypoint (package)
│   ├── app_main.py                # Main UI + routing
│   ├── config.py
│   ├── models.py
│   ├── utils.py
│   ├── validation.py
│   ├── sidebar_ui.py
│   ├── plot_style.py
│   ├── docx_report.py
│   ├── streamlit_compat.py
│   └── tabs/
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
├── docs/
│   └── USER_GUIDE.md
├── examples/
├── tests/
├── scripts/
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── requirements-lock.txt
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── CITATION.cff
└── README.md

```

---


## 🚀 Production Deployment Notes

Before packaging/deploying (especially when deploying from a ZIP checkout), clean build/test artifacts:

- macOS/Linux: `bash scripts/clean_artifacts.sh`
- Windows (PowerShell): `powershell -ExecutionPolicy Bypass -File scripts/clean_artifacts.ps1`

These remove `.coverage`, `.pytest_cache`, `__pycache__`, and other transient caches.


## 🔧 Troubleshooting

### Installation Issues

**"pip install fails with compilation errors"**
```bash
# Windows: Install Visual C++ Build Tools
# macOS: xcode-select --install
# Linux: sudo apt-get install build-essential python3-dev
```

**"ModuleNotFoundError: No module named 'streamlit'"**
```bash
# Activate venv first, then:
pip install -r requirements.txt
```

**"Port 8501 already in use"**
```bash
streamlit run adsorblab_pro/app.py --server.port 8502
```

### Runtime Issues


### Word Report (.docx) Issues

- **DOCX option is disabled**: install the dependency and restart Streamlit:
  - `pip install python-docx`
- **ImportError related to lxml**: upgrade build tooling:
  - `python -m pip install -U pip setuptools wheel`
- **Report is huge/slow**: export fewer figures, reduce image scale, and/or lower “Max rows per table in report”.

**"Fitting fails to converge"**
- Check data for outliers
- Verify Cₑ < C₀
- Try simpler model first
- Adjust initial parameter guesses

**"Bootstrap CI very wide"**
- Add more data points (6-8 minimum)
- Check for outliers
- Consider if model is appropriate

### Data Quality Checklist
- [ ] Cₑ ≤ C₀ for all points
- [ ] No negative values
- [ ] 5+ points for isotherm, 8+ for kinetics
- [ ] Consistent units (mg/L, g, L, min)
- [ ] Temperature in Kelvin for thermodynamics

---

## 🧪 Running Tests

```bash
# All tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=. --cov-report=html

# Specific file
python -m pytest tests/test_models.py -v
```

---

## 📝 Citation

```bibtex
@software{adsorblab_pro_2026,
  title = {{AdsorbLab Pro}: Publication-Ready Adsorption Data Analysis Platform},
  author = {{Mohamed EL MALLAHI}},
  year = {2026},
  version = {2.0.0},
  url = {https://github.com/mmalahi00/Adsorption-Analysis-app},
  license = {MIT}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🆘 Support

- 🐛 [Report Bug](https://github.com/mmalahi00/Adsorption-Analysis-app/issues)
- 💡 [Request Feature](https://github.com/mmalahi00/Adsorption-Analysis-app/issues)

---

**Made with ❤️ for the adsorption research community**
