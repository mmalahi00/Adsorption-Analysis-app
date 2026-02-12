#!/usr/bin/env python3
"""
Case Study 1 – Equilibrium Isotherm Modelling
==============================================
System : Methylene Blue on magnetic nanosorbent (macadamia shell AC + Fe₃O₄)
Source : Suwannahong et al., Appl. Sci. 2021, 11(16), 7432
DOI    : 10.3390/app11167432
Goal   : Fit four isotherm models, rank by AIC, and produce
         publication-ready figures and a summary table.

Models fitted
-------------
1. Langmuir       qe = qm·KL·Ce / (1 + KL·Ce)
2. Freundlich     qe = KF·Ce^(1/n)
3. Temkin         qe = B1·ln(KT·Ce)
4. Sips           qe = qm·(Ks·Ce)^ns / (1 + (Ks·Ce)^ns)

Requirements: numpy, scipy, pandas, matplotlib
"""

from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ── Paths ────────────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).resolve().parent
DATA_FILE = HERE / "data.csv"
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── Experimental conditions ──────────────────────────────────────────────
# Note: V, m, T, pH are defined in the original batch study
# (Wongcharee et al. 2017, Int. Biodeterioration Biodegradation, 124, 276-287).
# The Suwannahong paper provides Ce/qe data directly.
EPS = 1e-12  # numerical guard

# ── Load and derive ──────────────────────────────────────────────────────
df = pd.read_csv(DATA_FILE)
# CSV has extra metadata columns (Adsorbate, Adsorbent, DOI, etc.)
# We only need the core data columns:
Ce = df["Ce_mg_L"].values
qe = df["qe_exp_mg_g"].values

# ── Model definitions ────────────────────────────────────────────────────


def langmuir(Ce, qm, KL):
    return qm * KL * Ce / (1.0 + KL * Ce)


def freundlich(Ce, KF, n_inv):
    return KF * np.power(np.maximum(Ce, EPS), n_inv)


def temkin(Ce, B1, KT):
    return B1 * np.log(np.maximum(KT * Ce, EPS))


def sips(Ce, qm, Ks, ns):
    Ks_Ce_ns = np.power(np.maximum(Ks * Ce, EPS), ns)
    return qm * Ks_Ce_ns / (1.0 + Ks_Ce_ns)


# ── Fitting helper ───────────────────────────────────────────────────────


def fit_model(func, Ce, qe, p0, bounds, names):
    """Fit *func* and return a results dict with AIC."""
    popt, pcov = curve_fit(func, Ce, qe, p0=p0, bounds=bounds, maxfev=20_000)
    y_pred = func(Ce, *popt)
    ss_res = np.sum((qe - y_pred) ** 2)
    ss_tot = np.sum((qe - qe.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    n = len(qe)
    k = len(popt)
    aic = n * np.log(ss_res / n) + 2 * k
    perr = np.sqrt(np.diag(pcov))
    return {
        "params": dict(zip(names, popt)),
        "std_err": dict(zip(names, perr)),
        "R2": r2,
        "AIC": aic,
        "SS_res": ss_res,
        "n_params": k,
        "y_pred": y_pred,
    }


# ── Fit all models ───────────────────────────────────────────────────────
results = {}

results["Langmuir"] = fit_model(
    langmuir,
    Ce,
    qe,
    p0=[80, 0.02],
    bounds=([0, 0], [500, 10]),
    names=["qm (mg/g)", "KL (L/mg)"],
)

results["Freundlich"] = fit_model(
    freundlich,
    Ce,
    qe,
    p0=[5.0, 0.5],
    bounds=([0, 0.01], [200, 5]),
    names=["KF ((mg/g)(L/mg)^1/n)", "1/n"],
)

results["Temkin"] = fit_model(
    temkin,
    Ce,
    qe,
    p0=[15, 0.25],
    bounds=([0.01, 1e-6], [200, 100]),
    names=["B1 (J/mol)", "KT (L/mg)"],
)

results["Sips"] = fit_model(
    sips,
    Ce,
    qe,
    p0=[80, 0.02, 0.9],
    bounds=([0, 0, 0.1], [500, 10, 5]),
    names=["qm (mg/g)", "Ks (L/mg)", "ns"],
)

# ── Rank models by AIC ──────────────────────────────────────────────────
ranking = sorted(results.items(), key=lambda kv: kv[1]["AIC"])
best_name = ranking[0][0]

# ── Console summary ──────────────────────────────────────────────────────
print("=" * 65)
print("  CASE 1 – Isotherm Model Comparison  (MB / GAC)")
print("=" * 65)
for name, r in ranking:
    tag = " ◀ BEST" if name == best_name else ""
    print(f"\n  {name}{tag}")
    for pn, pv in r["params"].items():
        print(f"    {pn:30s} = {pv:.4g}  ± {r['std_err'][pn]:.3g}")
    print(f"    {'R²':30s} = {r['R2']:.5f}")
    print(f"    {'AIC':30s} = {r['AIC']:.2f}")
print()

# ── Separation factor RL for Langmuir ────────────────────────────────────
KL = results["Langmuir"]["params"]["KL (L/mg)"]
# RL = 1/(1+KL*C0); use Ce as lower-bound proxy (C0 > Ce always)
Ce_vals = df["Ce_mg_L"].values
RL = 1.0 / (1.0 + KL * Ce_vals)
print(f"  Langmuir separation factor RL range: {RL.min():.3f} – {RL.max():.3f}")
print("  (Computed using Ce as proxy for C0; actual RL slightly lower)")
print(f"  Interpretation: {'Favourable' if np.all((RL > 0) & (RL < 1)) else 'Check range'}\n")

# ── Figure 1: Experimental + fitted curves ───────────────────────────────
Ce_fine = np.linspace(0, Ce.max() * 1.05, 300)
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.scatter(Ce, qe, s=60, zorder=5, color="black", label="Experimental")

colours = {"Langmuir": "#1f77b4", "Freundlich": "#ff7f0e", "Temkin": "#2ca02c", "Sips": "#d62728"}
funcs = {"Langmuir": langmuir, "Freundlich": freundlich, "Temkin": temkin, "Sips": sips}

for name, r in results.items():
    y_fine = funcs[name](Ce_fine, *r["params"].values())
    style = "-" if name == best_name else "--"
    lbl = f"{name}  (R²={r['R2']:.4f})"
    ax1.plot(Ce_fine, y_fine, style, color=colours[name], lw=2, label=lbl)

ax1.set_xlabel("Ce  (mg / L)", fontsize=12)
ax1.set_ylabel("qe  (mg / g)", fontsize=12)
ax1.set_title("Isotherm Model Comparison – MB / Magnetic Nanosorbent", fontsize=13)
ax1.legend(fontsize=9, frameon=True)
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)
fig1.tight_layout()
fig1.savefig(OUT_DIR / "isotherm_fit_comparison.png", dpi=300)
print(f"  ✓ Saved {OUT_DIR / 'isotherm_fit_comparison.png'}")

# ── Figure 2: Residual plot ─────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
for ax, (name, r) in zip(axes2.flat, results.items()):
    residuals = qe - r["y_pred"]
    ax.bar(range(len(residuals)), residuals, color=colours[name], alpha=0.75)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"{name}  (R²={r['R2']:.4f})", fontsize=10)
    ax.set_ylabel("Residual (mg/g)", fontsize=9)
axes2[1, 0].set_xlabel("Data point index", fontsize=9)
axes2[1, 1].set_xlabel("Data point index", fontsize=9)
fig2.suptitle("Residual Analysis", fontsize=13, y=1.01)
fig2.tight_layout()
fig2.savefig(OUT_DIR / "isotherm_residuals.png", dpi=300)
print(f"  ✓ Saved {OUT_DIR / 'isotherm_residuals.png'}")

# ── Figure 3: RL separation factor ──────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.plot(Ce_vals, RL, "o-", color="#1f77b4", lw=2, markersize=7)
ax3.axhline(1, color="grey", ls=":", lw=0.8, label="RL = 1 (linear)")
ax3.axhline(0, color="grey", ls=":", lw=0.8, label="RL = 0 (irreversible)")
ax3.fill_between(Ce_vals, 0, 1, alpha=0.08, color="green", label="Favourable region")
ax3.set_xlabel("Ce  (mg / L)", fontsize=12)
ax3.set_ylabel("RL", fontsize=12)
ax3.set_title("Langmuir Separation Factor", fontsize=13)
ax3.legend(fontsize=9)
fig3.tight_layout()
fig3.savefig(OUT_DIR / "langmuir_RL.png", dpi=300)
print(f"  ✓ Saved {OUT_DIR / 'langmuir_RL.png'}")

# ── Summary table CSV ───────────────────────────────────────────────────
rows = []
for name, r in ranking:
    row = {
        "Model": name,
        "R²": r["R2"],
        "AIC": r["AIC"],
        "SS_res": r["SS_res"],
        "n_params": r["n_params"],
    }
    for pn, pv in r["params"].items():
        row[pn] = pv
    rows.append(row)

summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUT_DIR / "model_summary.csv", index=False, float_format="%.5g")
print(f"  ✓ Saved {OUT_DIR / 'model_summary.csv'}")

# ── Derived data table ──────────────────────────────────────────────────
df.to_csv(OUT_DIR / "derived_data.csv", index=False, float_format="%.4f")
print(f"  ✓ Saved {OUT_DIR / 'derived_data.csv'}")

# ── JSON results (machine-readable) ─────────────────────────────────────
json_out = {}
for name, r in results.items():
    json_out[name] = {
        "params": {k: float(v) for k, v in r["params"].items()},
        "std_err": {k: float(v) for k, v in r["std_err"].items()},
        "R2": float(r["R2"]),
        "AIC": float(r["AIC"]),
    }
json_out["best_model"] = best_name

with open(OUT_DIR / "results.json", "w") as f:
    json.dump(json_out, f, indent=2)
print(f"  ✓ Saved {OUT_DIR / 'results.json'}")

plt.close("all")
print("\n  Done – all outputs written to case_studies/case1_isotherm/outputs/")
