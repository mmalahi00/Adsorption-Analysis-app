# Case Study 3 — Thermodynamic Analysis

## Published Source

**Sulaiman, N.S.; Mohamad Amini, M.H.; Danish, M.; Sulaiman, O.; Hashim, R.**
"Kinetics, Thermodynamics, and Isotherms of Methylene Blue Adsorption Study onto
Cassava Stem Activated Carbon." *Water* **2021**, *13*(20), 2936.

- **DOI:** [10.3390/w13202936](https://doi.org/10.3390/w13202936)
- **License:** CC BY 4.0 (open access)
- **Data location:** Table 1 (thermodynamics); Tables 2–3 (Langmuir/Freundlich at 4 T)

## System

| Property | Value |
|----------|-------|
| Adsorbate | Methylene Blue (MB) |
| Adsorbent | Activated cassava stem (ACS, pyrolyzed 787 °C, 146 min) |
| V | 50 mL |
| m | 0.075 g (1.5 g/L dosage) |
| C₀ | 100, 200, 300, 400 mg/L |
| T | 298.15, 308.15, 318.15, 328.15 K |
| pH | Natural (pHzpc = 9.20 for ACS) |
| Contact time | 1080 min (18 h, to ensure equilibrium) |
| Detection | UV-Vis at 660.11 nm |

## Thermodynamic Results (qualitative, from paper)

| Parameter | Sign | Interpretation |
|-----------|------|----------------|
| ΔH° | Positive | Endothermic adsorption |
| ΔS° | Positive | Increased randomness at solid–solute interface |
| ΔG° | Negative (all T) | Spontaneous process |

- ΔG° becomes more negative with increasing T → greater spontaneity at higher T
- Langmuir qm (ACS) = **384.61 mg/g** at 55 °C
- Kq definition: qe / Ce

## Data in `data.csv`

The file contains 16 rows (4 temperatures × 4 concentrations) with columns:

| Column | Description |
|--------|-------------|
| `Temperature_C` | Temperature (°C) — **used by script** |
| `C0_mg_L` | Initial concentration (mg/L) — **used by script** |
| `Ce_mg_L` | Equilibrium concentration (mg/L) — **used by script** |
| `Adsorbate` | Methylene Blue |
| `Adsorbent` | Activated cassava stem |
| `pH`, `V_mL`, `m_g` | Experimental conditions |
| `DOI` | Paper DOI |
| `Source` | Data provenance note |

> **Important:** The Ce values were **reconstructed from the published Langmuir
> parameters** (Tables 2–3) and mass balance, with ±1% noise. Download the PDF
> to extract exact tabulated values from Tables 1–3.

## Running

```bash
python run_case3.py
```

Outputs: `outputs/thermo_isotherms.png`, `outputs/vant_hoff.png`,
         `outputs/delta_G_vs_T.png`, `outputs/results.json`
