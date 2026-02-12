# Example Datasets

This directory contains comprehensive sample datasets for testing and validating all features of AdsorbLab Pro.

## Quick Start

```bash
# Verify all example files
ls -la examples/

# Quick data check
python -c "
import pandas as pd
for f in ['calibration_data.csv', 'isotherm_data.csv', 'kinetic_data.csv']:
    df = pd.read_csv(f'examples/{f}')
    print(f'{f}: {len(df)} rows, columns: {list(df.columns)}')
"
```

## File Overview

| File | Purpose | Input Mode | Columns |
|------|---------|------------|---------|
| `calibration_data.csv` | UV-Vis calibration | Standard | Concentration, Absorbance |
| `isotherm_data.csv` | Equilibrium isotherm | Standard (Absorbance) | Concentration, Absorbance |
| `isotherm_direct.csv` | Equilibrium isotherm | Direct | C0, Ce |
| `kinetic_data.csv` | Kinetic study | Standard (Absorbance) | Time, Absorbance |
| `kinetic_direct.csv` | Kinetic study | Direct | Time, Ct |
| `ph_effect_data.csv` | pH optimization | Standard | pH, Absorbance |
| `ph_effect_direct.csv` | pH optimization | Direct | pH, Ce |
| `dosage_data.csv` | Dosage optimization | Standard | Mass, Absorbance |
| `dosage_direct.csv` | Dosage optimization | Direct | Mass, Ce |
| `temperature_data.csv` | Temperature effect | Standard | Temperature, Absorbance |
| `temperature_direct.csv` | Temperature effect | Direct | Temperature, Ce |
| `expected_results.json` | Validation benchmarks | - | All expected parameters |

---

## Detailed File Descriptions

### 1. Calibration Data (`calibration_data.csv`)

Standard calibration curve for UV-Vis spectrophotometry following Beer-Lambert law.

| Column | Description | Unit |
|--------|-------------|------|
| Concentration | Standard concentration | mg/L |
| Absorbance | Measured absorbance | AU |

**Expected calibration results:**
- Slope: ~0.01666 L/mg
- Intercept: ~0.002 AU
- R²: > 0.999

---

### 2. Isotherm Data

#### `isotherm_data.csv` (Standard Mode)
Pre-calculated equilibrium data with absorbance measurements.

| Column | Description | Unit |
|--------|-------------|------|
| Concentration | Concentration | mg/L |
| Absorbance | Measured absorbance | AU |

#### `isotherm_direct.csv` (Direct Input Mode)
Minimal input for literature data validation.

| Column | Description | Unit |
|--------|-------------|------|
| C0 | Initial concentration | mg/L |
| Ce | Equilibrium concentration | mg/L |

**Experimental conditions:**
- Volume: 50 mL
- Adsorbent mass: 0.1 g
- Temperature: 25°C (298.15 K)
- pH: 7.0

**Expected Best Model:** Langmuir (R² 0.995)
- qm: 78.5 mg/g
- KL: 0.018 L/mg

---

### 3. Kinetic Data

#### `kinetic_data.csv` (Standard Mode)
Time-resolved adsorption data with absorbance measurements.

| Column | Description | Unit |
|--------|-------------|------|
| Time | Contact time | min |
| Absorbance | Measured absorbance | AU |

#### `kinetic_direct.csv` (Direct Input Mode)

| Column | Description | Unit |
|--------|-------------|------|
| Time | Contact time | min |
| Ct | Concentration at time t | mg/L |

**Experimental conditions:**
- Initial concentration: 100 mg/L
- Volume: 50 mL
- Adsorbent mass: 0.1 g

**Expected Best Model:** Pseudo-Second-Order (R² 0.998)
- qe: 47.2 mg/g
- k2: 0.0012 g/(mg·min)
- Equilibrium time: ~240 minutes

---

### 4. pH Effect Data

#### `ph_effect_data.csv` (Standard Mode)

| Column | Description | Unit |
|--------|-------------|------|
| pH | Solution pH | - |
| Absorbance | Measured absorbance | AU |

#### `ph_effect_direct.csv` (Direct Input Mode)

| Column | Description | Unit |
|--------|-------------|------|
| pH | Solution pH | - |
| Ce | Equilibrium concentration | mg/L |

**Experimental conditions:**
- Initial concentration: 100 mg/L
- Volume: 50 mL
- Adsorbent mass: 0.1 g
- pH range: 2-12

**Expected results:**
- Optimal pH: ~7.0
- Maximum removal: ~80%
- Maximum qe: ~40 mg/g

---

### 5. Dosage Effect Data

#### `dosage_data.csv` (Standard Mode)

| Column | Description | Unit |
|--------|-------------|------|
| Mass | Adsorbent mass | g |
| Absorbance | Measured absorbance | AU |

#### `dosage_direct.csv` (Direct Input Mode)

| Column | Description | Unit |
|--------|-------------|------|
| Mass | Adsorbent mass | g |
| Ce | Equilibrium concentration | mg/L |

**Experimental conditions:**
- Initial concentration: 100 mg/L
- Volume: 50 mL
- Mass range: 0.025-0.5 g

**Expected results:**
- Optimal mass: 0.2 g (for 90% removal target)
- Removal at optimal: ~75%
- Trend: Diminishing returns at higher masses

---

### 6. Temperature Effect Data

#### `temperature_data.csv` (Standard Mode)

| Column | Description | Unit |
|--------|-------------|------|
| Temperature | Temperature | °C |
| Absorbance | Measured absorbance | AU |

#### `temperature_direct.csv` (Direct Input Mode)

| Column | Description | Unit |
|--------|-------------|------|
| Temperature | Temperature | °C |
| Ce | Equilibrium concentration | mg/L |

**Experimental conditions:**
- Initial concentration: 100 mg/L
- Volume: 50 mL
- Adsorbent mass: 0.1 g
- Temperature range: 25-65°C

**Expected results:**
- Process type: Endothermic
- Trend: qe increases with temperature
- Interpretation: Increasing capacity at higher temperatures

---

### 7. Expected Results (`expected_results.json`)

Comprehensive validation benchmarks containing:

- **Calibration parameters** (slope, intercept, R²)
- **Isotherm model parameters** (Langmuir, Freundlich, Temkin, Sips, Redlich-Peterson)
- **Kinetic model parameters** (PFO, PSO, Elovich, IPD, Avrami)
- **Effect study results** (pH, dosage, temperature optima)
- **Statistical validation criteria** (Bootstrap CI, PRESS Q², residual diagnostics)

---

## Usage Guide

### Standard Workflow (with Calibration)

1. Start application: `streamlit run adsorblab_pro/app.py`
2. Go to **Calibration** tab
3. Upload `calibration_data.csv`
4. Verify R² > 0.995
5. Upload experimental data (with Absorbance column)
6. Proceed to analysis tabs (Isotherm, Kinetics, pH Effect, Dosage, Temperature)

### Direct Input Workflow (without Calibration)

1. Start application: `streamlit run adsorblab_pro/app.py`
2. In sidebar, select **"📈 Direct Concentration"** input mode
3. Upload `*_direct.csv` files directly
4. Set experimental parameters (V, m, C0 if needed)
5. No calibration required!

### Validation Workflow

1. Load sample data into application
2. Fit all available models
3. Compare fitted parameters against `expected_results.json`
4. Verify R² values meet minimum thresholds
5. Check model selection (AIC/BIC) identifies best model

---

## Coverage Matrix

| Application Tab | Standard Mode File | Direct Mode File | Status |
|-----------------|-------------------|------------------|--------|
| Calibration | `calibration_data.csv` | N/A | ✅ |
| Isotherm | `isotherm_data.csv` | `isotherm_direct.csv` | ✅ |
| Kinetics | `kinetic_data.csv` | `kinetic_direct.csv` | ✅ |
| pH Effect | `ph_effect_data.csv` | `ph_effect_direct.csv` | ✅ |
| Dosage | `dosage_data.csv` | `dosage_direct.csv` | ✅ |
| Temperature | `temperature_data.csv` | `temperature_direct.csv` | ✅ |
| Statistical Summary | All of above | All of above | ✅ |
| Report Generation | All study data | All study data | ✅ |
| 3D Explorer | All compatible data | All compatible data | ✅ |
| Comparison | Multiple studies | Multiple studies | ✅ |

---

## Notes

- All datasets are synthetic but representative of real adsorption experiments
- Expected results assume non-linear fitting (not linearized transforms)
- Small variations in fitted parameters are normal due to algorithm differences
- Direct input mode is ideal for validating against published literature data
- Use tolerance values in `expected_results.json` to determine acceptable deviations
- Two input modes (Standard with calibration, Direct without) allow flexible data workflows
