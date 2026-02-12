# Case Study 1 — Equilibrium Isotherm

## Published Source

**Suwannahong, K.; Wongcharee, S.; Kreetachart, T.; Sirilamduan, C.; Rioyo, J.;
Wongphat, A.** "Evaluation of the Microsoft Excel Solver Spreadsheet-Based Program
for Nonlinear Expressions of Adsorption Isotherm Models onto Magnetic Nanosorbent."
*Applied Sciences* **2021**, *11*(16), 7432.

- **DOI:** [10.3390/app11167432](https://doi.org/10.3390/app11167432)
- **License:** CC BY 4.0 (open access)
- **Data location:** Table 1 (11 Ce/qe pairs); Tables 2–8 (model parameters)

## System

| Property | Value |
|----------|-------|
| Adsorbate | Methylene Blue (MB) |
| Adsorbent | Magnetic nanosorbent (macadamia nut shell AC + Fe₃O₄) |
| BET surface area | ~70 m²/g |
| qe,max | ≈ 33 mg/g |

> **Note on V, m, C₀, T, pH:** These experimental conditions are defined in the
> original batch study (Ref [11]: Wongcharee et al. 2017, *Int. Biodeterioration
> Biodegradation*, 124, 276–287). The Suwannahong et al. paper focuses on the
> fitting methodology, not the experimental procedure.

## Isotherm Models Fitted (6 total)

| Category | Model | Key parameters |
|----------|-------|----------------|
| Two-parameter | Langmuir | qm = 34.48 mg/g, KL = 0.235 L/mg (best 2-param) |
| Two-parameter | Freundlich | KF = 10.76, n = 3.76 |
| Two-parameter | Temkin | bT = 6.05 J/mol, KT = 3.59 L/mol |
| Three-parameter | Khan | qm, KK, aK |
| Three-parameter | Toth | qm, Kth, nth (best 3-param fit) |
| Three-parameter | Liu | qm, Kg, ng |

## Data in `data.csv`

The file contains 11 data points with columns:

| Column | Description |
|--------|-------------|
| `Ce_mg_L` | Equilibrium concentration (mg/L) — **used by script** |
| `qe_exp_mg_g` | Experimental qe (mg/g) — **used by script** |
| `Adsorbate` | Methylene Blue |
| `Adsorbent` | Description of adsorbent |
| `pH`, `T_C`, `V_mL`, `m_g` | Conditions (see note above) |
| `DOI` | Paper DOI |
| `Source` | Data provenance note |

> **Important:** The data in `data.csv` was **reconstructed from the published
> Langmuir model parameters** (qm = 34.4765, KL = 0.2346) reported in Table 8,
> with ±1.5% Gaussian noise added for realism. To obtain the exact experimental
> values, download the PDF from MDPI and extract from Table 1.

## Running

```bash
python run_case1.py
```

Outputs: `outputs/isotherm_fits.png`, `outputs/results.json`
