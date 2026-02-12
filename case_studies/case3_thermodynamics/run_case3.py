#!/usr/bin/env python3
"""
Case Study 3 – Thermodynamic Analysis
======================================
System : Methylene Blue (MB) on activated cassava stem (ACS, 787 °C)
Source : Sulaiman et al., Water 2021, 13(20), 2936
DOI    : 10.3390/w13202936
Goal   : Fit Langmuir isotherms at four temperatures, calculate
         thermodynamic parameters (ΔH°, ΔS°, ΔG°) via Van't Hoff
         analysis, and classify the adsorption mechanism.

Workflow
--------
1. Fit Langmuir at each temperature → qm(T), KL(T)
2. Compute distribution coefficient  Kd = qe / Ce  (at C0 = 100 mg/L)
3. Van't Hoff plot: ln Kd  vs  1/T
4. Derive ΔH°, ΔS° from slope/intercept; ΔG° = ΔH° − TΔS°

Requirements: numpy, scipy, pandas, matplotlib
"""

from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress

# ── Paths ────────────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).resolve().parent
DATA_FILE = HERE / "data.csv"
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── Constants (Sulaiman et al. 2021) ─────────────────────────────────────
R = 8.314        # J/(mol·K)
V = 0.050        # L  (50 mL)
M = 0.075        # g  (1.5 g/L dosage × 50 mL)
EPS = 1e-12

# ── Load data ────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_FILE)
df["T_K"] = df["Temperature_C"] + 273.15
df["qe_mg_g"] = (df["C0_mg_L"] - df["Ce_mg_L"]) * V / M
df["Removal_%"] = (df["C0_mg_L"] - df["Ce_mg_L"]) / df["C0_mg_L"] * 100

temperatures = sorted(df["Temperature_C"].unique())
T_K_list = [t + 273.15 for t in temperatures]

# ── Langmuir definition ─────────────────────────────────────────────────

def langmuir(Ce, qm, KL):
    return qm * KL * Ce / (1.0 + KL * Ce)


# ── Fit Langmuir at each temperature ────────────────────────────────────
langmuir_fits = {}

print("=" * 65)
print("  CASE 3 – Thermodynamic Analysis  (MB / Cassava Stem AC)")
print("=" * 65)
print("\n  ── Langmuir fits per temperature ──")

for T_C in temperatures:
    sub = df[df["Temperature_C"] == T_C]
    Ce = sub["Ce_mg_L"].values
    qe = sub["qe_mg_g"].values
    popt, pcov = curve_fit(langmuir, Ce, qe, p0=[80, 0.02],
                           bounds=([0, 0], [500, 10]), maxfev=20_000)
    y_pred = langmuir(Ce, *popt)
    ss_res = np.sum((qe - y_pred) ** 2)
    ss_tot = np.sum((qe - qe.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    langmuir_fits[T_C] = {
        "qm": popt[0], "KL": popt[1], "R2": r2,
        "Ce": Ce, "qe": qe, "y_pred": y_pred,
    }
    print(f"    {T_C} °C ({T_C+273.15:.2f} K):  qm = {popt[0]:.2f} mg/g,  "
          f"KL = {popt[1]:.4f} L/mg,  R² = {r2:.5f}")

# ── Distribution coefficients ────────────────────────────────────────────
# Kd = (C0 − Ce) / Ce  at each temperature, averaged over all C0 values
# This is the standard dimensionless distribution coefficient used in
# Van't Hoff analysis (see Liu, 2009, J. Chem. Eng. Data).
print("\n  ── Distribution coefficients (averaged over all C0) ──")
Kd_list = []
for T_C in temperatures:
    sub = df[df["Temperature_C"] == T_C]
    kd_per_c0 = (sub["C0_mg_L"].values - sub["Ce_mg_L"].values) / sub["Ce_mg_L"].values
    Kd_avg = kd_per_c0.mean()
    Kd_list.append(Kd_avg)
    print(f"    {T_C} °C:  Kd (mean) = {Kd_avg:.4f}  "
          f"(range {kd_per_c0.min():.3f}–{kd_per_c0.max():.3f})")

Kd_arr = np.array(Kd_list)
T_K_arr = np.array(T_K_list)

# ── Van't Hoff regression ───────────────────────────────────────────────
inv_T = 1.0 / T_K_arr
ln_Kd = np.log(Kd_arr)

slope, intercept, r_value, p_value, std_err = linregress(inv_T, ln_Kd)
r2_vh = r_value ** 2

delta_H = -slope * R / 1000.0           # kJ/mol
delta_S = intercept * R                   # J/(mol·K)
delta_G = [delta_H - T * delta_S / 1000.0 for T in T_K_list]  # kJ/mol

print("\n  ── Van't Hoff Results ──")
print(f"    Slope     = {slope:.2f} K")
print(f"    Intercept = {intercept:.4f}")
print(f"    R²        = {r2_vh:.5f}")
print(f"\n    ΔH° = {delta_H:.2f} kJ/mol  {'(exothermic)' if delta_H < 0 else '(endothermic)'}")
print(f"    ΔS° = {delta_S:.2f} J/(mol·K)")
for T_C, T_K, dG in zip(temperatures, T_K_list, delta_G):
    print(f"    ΔG° at {T_C} °C = {dG:.2f} kJ/mol  "
          f"{'(spontaneous)' if dG < 0 else '(non-spontaneous)'}")

# ── Mechanism classification ─────────────────────────────────────────────
abs_dH = abs(delta_H)
if abs_dH < 40:
    mechanism = "physisorption"
elif abs_dH < 80:
    mechanism = "mixed (physi-/chemisorption)"
else:
    mechanism = "chemisorption"
print(f"\n    Mechanism: {mechanism}  (|ΔH°| = {abs_dH:.1f} kJ/mol)")
print()

# ── Figure 1: Multi-temperature isotherms ────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 5))
cmap = plt.cm.coolwarm
colors = [cmap(i / (len(temperatures) - 1)) for i in range(len(temperatures))]

for i, T_C in enumerate(temperatures):
    fit = langmuir_fits[T_C]
    ax1.scatter(fit["Ce"], fit["qe"], s=60, color=colors[i], zorder=5)
    Ce_fine = np.linspace(0, max(fit["Ce"]) * 1.05, 200)
    ax1.plot(Ce_fine, langmuir(Ce_fine, fit["qm"], fit["KL"]),
             "-", color=colors[i], lw=2,
             label=f"{T_C} °C  (qm={fit['qm']:.1f}, R²={fit['R2']:.4f})")

ax1.set_xlabel("Ce  (mg / L)", fontsize=12)
ax1.set_ylabel("qe  (mg / g)", fontsize=12)
ax1.set_title("Langmuir Isotherms at Multiple Temperatures", fontsize=13)
ax1.legend(fontsize=9, frameon=True)
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)
fig1.tight_layout()
fig1.savefig(OUT_DIR / "multi_temp_isotherms.png", dpi=300)
print(f"  ✓ Saved {OUT_DIR / 'multi_temp_isotherms.png'}")

# ── Figure 2: Van't Hoff plot ───────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(6, 4.5))
ax2.scatter(inv_T * 1000, ln_Kd, s=80, color="#d62728", zorder=5)
x_line = np.linspace(inv_T.min() * 0.998, inv_T.max() * 1.002, 100)
ax2.plot(x_line * 1000, slope * x_line + intercept, "--", color="#1f77b4",
         lw=2, label=f"Linear fit (R² = {r2_vh:.4f})")
ax2.set_xlabel("1000 / T  (K⁻¹)", fontsize=12)
ax2.set_ylabel("ln Kd", fontsize=12)
ax2.set_title("Van't Hoff Plot", fontsize=13)
ax2.legend(fontsize=10)

# Annotate ΔH, ΔS on plot
textstr = (f"ΔH° = {delta_H:.1f} kJ/mol\n"
           f"ΔS° = {delta_S:.1f} J/(mol·K)")
ax2.text(0.97, 0.97, textstr, transform=ax2.transAxes,
         fontsize=9, va="top", ha="right",
         bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.85))
fig2.tight_layout()
fig2.savefig(OUT_DIR / "vant_hoff_plot.png", dpi=300)
print(f"  ✓ Saved {OUT_DIR / 'vant_hoff_plot.png'}")

# ── Figure 3: ΔG° vs Temperature ────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.plot(temperatures, delta_G, "o-", color="#2ca02c", lw=2, markersize=8)
ax3.axhline(0, color="grey", ls=":", lw=0.8)
ax3.set_xlabel("Temperature  (°C)", fontsize=12)
ax3.set_ylabel("ΔG°  (kJ / mol)", fontsize=12)
ax3.set_title("Gibbs Free Energy vs Temperature", fontsize=13)
ax3.fill_between(temperatures, min(delta_G) * 1.3, 0,
                 alpha=0.08, color="green", label="Spontaneous region")
ax3.legend(fontsize=9)
fig3.tight_layout()
fig3.savefig(OUT_DIR / "delta_G_vs_T.png", dpi=300)
print(f"  ✓ Saved {OUT_DIR / 'delta_G_vs_T.png'}")

# ── Summary tables ───────────────────────────────────────────────────────
# Langmuir parameters per temperature
lang_rows = []
for T_C in temperatures:
    f = langmuir_fits[T_C]
    lang_rows.append({
        "T (°C)": T_C, "T (K)": T_C + 273.15,
        "qm (mg/g)": f["qm"], "KL (L/mg)": f["KL"], "R²": f["R2"],
    })
pd.DataFrame(lang_rows).to_csv(OUT_DIR / "langmuir_per_temperature.csv",
                                 index=False, float_format="%.5g")
print(f"  ✓ Saved {OUT_DIR / 'langmuir_per_temperature.csv'}")

# Thermodynamic parameters
thermo_rows = []
for T_C, T_K, Kd, dG in zip(temperatures, T_K_list, Kd_list, delta_G):
    thermo_rows.append({
        "T (°C)": T_C, "T (K)": T_K, "Kd": Kd, "ln(Kd)": np.log(Kd),
        "ΔG° (kJ/mol)": dG,
    })
thermo_df = pd.DataFrame(thermo_rows)
thermo_df.to_csv(OUT_DIR / "thermodynamic_parameters.csv",
                  index=False, float_format="%.5g")
print(f"  ✓ Saved {OUT_DIR / 'thermodynamic_parameters.csv'}")

df.to_csv(OUT_DIR / "derived_data.csv", index=False, float_format="%.4f")
print(f"  ✓ Saved {OUT_DIR / 'derived_data.csv'}")

# ── JSON ─────────────────────────────────────────────────────────────────
json_out = {
    "langmuir_per_temperature": {
        str(T_C): {"qm": float(langmuir_fits[T_C]["qm"]),
                    "KL": float(langmuir_fits[T_C]["KL"]),
                    "R2": float(langmuir_fits[T_C]["R2"])}
        for T_C in temperatures
    },
    "vant_hoff": {
        "slope_K": float(slope),
        "intercept": float(intercept),
        "R2": float(r2_vh),
    },
    "thermodynamics": {
        "delta_H_kJ_mol": float(delta_H),
        "delta_S_J_mol_K": float(delta_S),
        "delta_G_kJ_mol": {str(T_C): float(dG)
                            for T_C, dG in zip(temperatures, delta_G)},
        "mechanism": mechanism,
    },
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(json_out, f, indent=2)
print(f"  ✓ Saved {OUT_DIR / 'results.json'}")

plt.close("all")
print("\n  Done – all outputs written to case_studies/case3_thermodynamics/outputs/")
