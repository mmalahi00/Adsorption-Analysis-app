#!/usr/bin/env python3
"""
Case Study 2 – Adsorption Kinetics
===================================
System : Methylene Blue (MB) on lignite coal (Kosovo)
Source : Hasani et al., Molecules 2022, 27(6), 1856
DOI    : 10.3390/molecules27061856
Goal   : Fit four kinetic models, identify rate-limiting step,
         and determine equilibrium time.

Models fitted
-------------
1. PFO  (Pseudo-First Order / Lagergren)   qt = qe (1 − e^{−k1 t})
2. PSO  (Pseudo-Second Order / Ho-McKay)   qt = k2 qe² t / (1 + k2 qe t)
3. Elovich                                  qt = (1/β) ln(1 + αβt)
4. IPD  (Intraparticle Diffusion / Weber-Morris)  qt = kid t^0.5 + C

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

# ── Experimental conditions (Hasani et al. 2022) ────────────────────────
C0 = 50.0    # mg/L  (also studied at 10, 30 mg/L)
V  = 0.025   # L     (25 mL)
M  = 0.100   # g     (100 mg)
EPS = 1e-12

# ── Load and derive ──────────────────────────────────────────────────────
df = pd.read_csv(DATA_FILE)
df["qt_mg_g"] = (C0 - df["Ct_mg_L"]) * V / M
df["Removal_%"] = (C0 - df["Ct_mg_L"]) / C0 * 100

t  = df["time_min"].values.astype(float)
qt = df["qt_mg_g"].values

# Exclude t=0 for fitting (qt=0 causes issues with some models)
mask = t > 0
t_fit  = t[mask]
qt_fit = qt[mask]

# ── Model definitions ────────────────────────────────────────────────────

def pfo(t, qe, k1):
    return qe * (1.0 - np.exp(-k1 * t))

def pso(t, qe, k2):
    return k2 * qe**2 * t / (1.0 + k2 * qe * t)

def elovich(t, alpha, beta):
    return (1.0 / beta) * np.log(1.0 + alpha * beta * t)

def ipd(t, kid, C):
    return kid * np.sqrt(t) + C


# ── Fitting helper ───────────────────────────────────────────────────────

def fit_model(func, t, qt, p0, bounds, names):
    popt, pcov = curve_fit(func, t, qt, p0=p0, bounds=bounds, maxfev=20_000)
    y_pred = func(t, *popt)
    ss_res = np.sum((qt - y_pred) ** 2)
    ss_tot = np.sum((qt - qt.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    n = len(qt)
    k = len(popt)
    aic = n * np.log(ss_res / n + EPS) + 2 * k
    perr = np.sqrt(np.diag(pcov))
    return {
        "params": dict(zip(names, popt)),
        "std_err": dict(zip(names, perr)),
        "R2": r2, "AIC": aic, "SS_res": ss_res,
        "n_params": k, "y_pred_full": func(t, *popt),
    }


# ── Fit all models ───────────────────────────────────────────────────────
results = {}

results["PFO"] = fit_model(
    pfo, t_fit, qt_fit,
    p0=[45, 0.03], bounds=([0, 0], [200, 5]),
    names=["qe (mg/g)", "k1 (1/min)"],
)

results["PSO"] = fit_model(
    pso, t_fit, qt_fit,
    p0=[47, 0.001], bounds=([0, 0], [200, 1]),
    names=["qe (mg/g)", "k2 (g/(mg·min))"],
)

results["Elovich"] = fit_model(
    elovich, t_fit, qt_fit,
    p0=[12, 0.1], bounds=([0.01, 0.001], [1000, 10]),
    names=["α (mg/(g·min))", "β (g/mg)"],
)

results["IPD"] = fit_model(
    ipd, t_fit, qt_fit,
    p0=[3.5, 8], bounds=([0, 0], [50, 100]),
    names=["kid (mg/(g·min^0.5))", "C (mg/g)"],
)

# ── Rank by AIC ──────────────────────────────────────────────────────────
ranking = sorted(results.items(), key=lambda kv: kv[1]["AIC"])
best_name = ranking[0][0]

# ── Equilibrium time (95 % of qe,exp) ───────────────────────────────────
qe_exp = qt[-1]  # final measured capacity
threshold = 0.95 * qe_exp
idx_eq = np.where(qt >= threshold)[0]
t_eq = t[idx_eq[0]] if len(idx_eq) > 0 else t[-1]

# ── Initial adsorption rate (PSO) ───────────────────────────────────────
qe_pso = results["PSO"]["params"]["qe (mg/g)"]
k2_pso = results["PSO"]["params"]["k2 (g/(mg·min))"]
h_init = k2_pso * qe_pso ** 2  # mg/(g·min)

# ── Console summary ──────────────────────────────────────────────────────
print("=" * 65)
print("  CASE 2 – Kinetic Model Comparison  (MB / Lignite Coal)")
print("=" * 65)
for name, r in ranking:
    tag = " ◀ BEST" if name == best_name else ""
    print(f"\n  {name}{tag}")
    for pn, pv in r["params"].items():
        print(f"    {pn:30s} = {pv:.5g}  ± {r['std_err'][pn]:.3g}")
    print(f"    {'R²':30s} = {r['R2']:.5f}")
    print(f"    {'AIC':30s} = {r['AIC']:.2f}")

print(f"\n  Equilibrium time (95% qe): {t_eq:.0f} min")
print(f"  qe,experimental:           {qe_exp:.2f} mg/g")
print(f"  PSO initial rate h:        {h_init:.3f} mg/(g·min)")
print()

# ── Figure 1: qt vs time with fitted curves ──────────────────────────────
t_fine = np.linspace(0.1, t.max() * 1.05, 400)
colours = {"PFO": "#1f77b4", "PSO": "#d62728",
           "Elovich": "#2ca02c", "IPD": "#ff7f0e"}
funcs = {"PFO": pfo, "PSO": pso, "Elovich": elovich, "IPD": ipd}

fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.scatter(t, qt, s=60, zorder=5, color="black", label="Experimental")

for name, r in results.items():
    y = funcs[name](t_fine, *r["params"].values())
    ls = "-" if name == best_name else "--"
    ax1.plot(t_fine, y, ls, color=colours[name], lw=2,
             label=f"{name}  (R²={r['R2']:.4f})")

ax1.axhline(qe_exp, ls=":", color="grey", lw=0.8, label=f"qe,exp = {qe_exp:.1f} mg/g")
ax1.axvline(t_eq, ls=":", color="grey", lw=0.8, alpha=0.5)
ax1.set_xlabel("Time  (min)", fontsize=12)
ax1.set_ylabel("qt  (mg / g)", fontsize=12)
ax1.set_title("Kinetic Model Comparison – MB / Lignite Coal", fontsize=13)
ax1.legend(fontsize=9, frameon=True)
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)
fig1.tight_layout()
fig1.savefig(OUT_DIR / "kinetic_fit_comparison.png", dpi=300)
print(f"  ✓ Saved {OUT_DIR / 'kinetic_fit_comparison.png'}")

# ── Figure 2: Weber-Morris plot (qt vs t^0.5) ───────────────────────────
fig2, ax2 = plt.subplots(figsize=(6, 4.5))
t_sqrt = np.sqrt(t[mask])
ax2.scatter(t_sqrt, qt_fit, s=60, color="black", zorder=5, label="Experimental")
t05_fine = np.linspace(0, t_sqrt.max() * 1.1, 200)
kid = results["IPD"]["params"]["kid (mg/(g·min^0.5))"]
C_ipd = results["IPD"]["params"]["C (mg/g)"]
ax2.plot(t05_fine, kid * t05_fine + C_ipd, "--", color="#ff7f0e", lw=2,
         label=f"IPD fit (kid={kid:.2f}, C={C_ipd:.1f})")
ax2.set_xlabel("t^0.5  (min^0.5)", fontsize=12)
ax2.set_ylabel("qt  (mg / g)", fontsize=12)
ax2.set_title("Weber-Morris Intraparticle Diffusion Plot", fontsize=13)
ax2.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig(OUT_DIR / "weber_morris_plot.png", dpi=300)
print(f"  ✓ Saved {OUT_DIR / 'weber_morris_plot.png'}")

# ── Figure 3: Residuals ─────────────────────────────────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
for ax, (name, r) in zip(axes3.flat, results.items()):
    # recompute on fit points
    y_pred = funcs[name](t_fit, *r["params"].values())
    residuals = qt_fit - y_pred
    ax.bar(range(len(residuals)), residuals, color=colours[name], alpha=0.75)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"{name}  (R²={r['R2']:.4f})", fontsize=10)
    ax.set_ylabel("Residual", fontsize=9)
axes3[1, 0].set_xlabel("Data point index")
axes3[1, 1].set_xlabel("Data point index")
fig3.suptitle("Residual Analysis – Kinetic Models", fontsize=13, y=1.01)
fig3.tight_layout()
fig3.savefig(OUT_DIR / "kinetic_residuals.png", dpi=300)
print(f"  ✓ Saved {OUT_DIR / 'kinetic_residuals.png'}")

# ── Summary table CSV ───────────────────────────────────────────────────
rows = []
for name, r in ranking:
    row = {"Model": name, "R²": r["R2"], "AIC": r["AIC"],
           "SS_res": r["SS_res"], "n_params": r["n_params"]}
    for pn, pv in r["params"].items():
        row[pn] = pv
    rows.append(row)

pd.DataFrame(rows).to_csv(OUT_DIR / "model_summary.csv", index=False, float_format="%.5g")
print(f"  ✓ Saved {OUT_DIR / 'model_summary.csv'}")

df.to_csv(OUT_DIR / "derived_data.csv", index=False, float_format="%.4f")
print(f"  ✓ Saved {OUT_DIR / 'derived_data.csv'}")

# ── JSON ─────────────────────────────────────────────────────────────────
json_out = {}
for name, r in results.items():
    json_out[name] = {
        "params": {k: float(v) for k, v in r["params"].items()},
        "std_err": {k: float(v) for k, v in r["std_err"].items()},
        "R2": float(r["R2"]), "AIC": float(r["AIC"]),
    }
json_out["best_model"] = best_name
json_out["equilibrium_time_min"] = float(t_eq)
json_out["qe_experimental_mg_g"] = float(qe_exp)
json_out["pso_initial_rate_h"] = float(h_init)

with open(OUT_DIR / "results.json", "w") as f:
    json.dump(json_out, f, indent=2)
print(f"  ✓ Saved {OUT_DIR / 'results.json'}")

plt.close("all")
print("\n  Done – all outputs written to case_studies/case2_kinetics/outputs/")
