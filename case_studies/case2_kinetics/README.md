# Case Study 2 — Adsorption Kinetics

## Published Source

**Hasani, N.; Selimi, T.; Mele, A.; Thaçi, V.; Halili, J.; Berisha, A.; Sadiku, M.**
"Theoretical, Equilibrium, Kinetics and Thermodynamic Investigations of Methylene
Blue Adsorption onto Lignite Coal." *Molecules* **2022**, *27*(6), 1856.

- **DOI:** [10.3390/molecules27061856](https://doi.org/10.3390/molecules27061856)
- **PMC:** [PMC8950461](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8950461/)
- **License:** CC BY 4.0 (open access)
- **Data location:** Figure 1 (qt vs time); Table 4 (kinetic model parameters)

## System

| Property | Value |
|----------|-------|
| Adsorbate | Methylene Blue (MB) |
| Adsorbent | Natural lignite coal (Kosovo) |
| V | 25 mL |
| m | 0.1 g |
| C₀ | 50 mg/L (also studied at 10, 30 mg/L) |
| T | 299.15 K (26 °C) |
| pH | 6.35 (natural) |
| Equilibrium time | ~90 min |

## Kinetic Models Fitted (7 total, from Table 4)

| Model | qe,exp (mg/g) | R² |
|-------|---------------|-----|
| **PSO (best fit)** | **10.80** | **0.999** |
| PFO | 10.80 | varies |
| Elovich | — | varies |
| IPD (Weber-Morris) | — | varies |
| Liquid film diffusion | — | varies |
| General 1st order | — | varies |
| General 2nd order | — | varies |

## Data in `data.csv`

The file contains 11 time points with columns:

| Column | Description |
|--------|-------------|
| `time_min` | Contact time (min) — **used by script** |
| `Ct_mg_L` | Concentration at time t (mg/L) — **used by script** |
| `Adsorbate` | Methylene Blue |
| `Adsorbent` | Lignite coal (Kosovo) |
| `C0_mg_L`, `pH`, `T_K`, `V_mL`, `m_g` | Experimental conditions |
| `DOI` | Paper DOI |
| `Source` | Data provenance note |

> **Important:** The time-series data was **reconstructed from the published PSO
> model parameters** (qe = 10.80 mg/g, Table 4) with ±1.2% noise. The original
> raw qt vs. time data is in Figure 1 (not tabulated). Use
> [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/) to extract exact
> values from the PDF figure.

## Running

```bash
python run_case2.py
```

Outputs: `outputs/kinetics_fits.png`, `outputs/results.json`
