# Detailed Adsorption Analysis App

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18501673.svg)](https://doi.org/10.5281/zenodo.18501673)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://adsorption.streamlit.app/)
![Version](https://img.shields.io/badge/version-v1.0.0-blue.svg)
![Language](https://img.shields.io/badge/Language-Python-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A comprehensive Streamlit web application designed for chemists, environmental scientists, and researchers to analyze adsorption experiment data. This tool automates the tedious and repetitive calculations and plotting involved in adsorption studies, allowing users to quickly process their data and visualize results.

---

## 🚀 Live Demo

**Experience the application live at:** https://adsorption.streamlit.app/

---

## 📌 Citation

If you use this software in your research, please cite:

**Detailed Adsorption Analysis App v1.0.0**. Zenodo. https://doi.org/10.5281/zenodo.18501673

### BibTeX
```bibtex
@software{el_mallahi_adsorption_2026,
  author  = {Mohamed EL MALLAHI},
  title   = {Detailed Adsorption Analysis App},
  version = {1.0.0},
  doi     = {10.5281/zenodo.18501673},
  url     = {https://doi.org/10.5281/zenodo.18501673}
}
```

---

## ✨ Features

This application provides a full suite of tools for analyzing different types of adsorption studies:

### General Features
- **⬆️ Data Upload:** Upload your experimental data directly via `.csv` or `.xlsx` files.
- **✏️ Interactive Data Editor:** Manually enter or edit data directly in the application's sidebar.
- **📊 Interactive Plots:** All graphs are generated using Plotly for an interactive experience (zoom, pan, inspect data points).
- **📥 Data & Plot Export:** Download calculated data as a CSV file and export publication-quality plots as PNG images.

### Analytical Modules
1. **🔬 Calibration**
   - Perform linear regression on your calibration standards (Concentration vs. Absorbance).
   - Automatically calculate the slope, intercept, and R² coefficient of determination.

2. **🧪 Isotherm Analysis**
   - Analyze experimental data to fit popular isotherm models.
   - **Linearized Models:** Langmuir, Freundlich, and Temkin.
   - **Non-Linear Model Fitting:** Langmuir, Freundlich, and Temkin.
   - Calculates key parameters (qm, KL, KF, n, B₁, KT).
   - Provides a model comparison table to easily see which model fits best based on R².

3. **⏳ Kinetic Analysis**
   - Analyze time-based adsorption data.
   - **Non-Linear Models:** Pseudo-First-Order (PFO) and Pseudo-Second-Order (PSO).
   - **Linearized Models:** PFO and PSO.
   - **Mechanism Analysis:** Intraparticle Diffusion (IPD) model (Weber–Morris plot).
   - Calculates kinetic parameters (qe, k₁, k₂, k_id, C).

4. **🌡️ Thermodynamics Analysis**
   - Calculates thermodynamic parameters from temperature-effect data.
   - Uses the Van’t Hoff equation to determine **Enthalpy (ΔH°)**, **Entropy (ΔS°)**, and **Gibbs Free Energy (ΔG°)**.
   - Indicates whether the adsorption process is exothermic/endothermic and spontaneous.

5. **📈 Parameter Effect Studies**
   - **pH Effect:** Visualize the effect of solution pH on adsorption capacity (qe).
   - **Dosage Effect:** Analyze the impact of adsorbent mass on qe.
   - **Temperature Effect:** Study how temperature influences adsorption.

---

## 🛠️ Installation & Local Run

### 1) Clone the repository
```bash
git clone https://github.com/mmalahi00/Adsorption-Analysis-app.git
cd Adsorption-Analysis-app
```

### 2) (Recommended) Create a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Run the app locally
```bash
streamlit run adsorption_app.py
```

If your entry file is named differently, replace `adsorption_app.py` with your main script.

---

## 📖 How to Use the App

1. **Enter Calibration Data:** In the sidebar, expand the **"1. Calibration"** section. Upload your data or enter it manually.
2. **Choose a Study:** Expand the section for the study you want to analyze (e.g., **"2. Isotherm Study"**).
3. **Input Data:** Enter the fixed conditions (e.g., mass, volume) and the experimental data (upload a file or use the data editor).
4. **Explore Results:** Navigate to the corresponding tab in the main window (e.g., **"Isotherms"**) to view tables and interactive plots.
5. **Download:** Use the download buttons to save your results and figures.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## 👤 Author

Developed by **[Mohamed EL MALLAHI](https://github.com/mmalahi00)**  
Live app: https://adsorption.streamlit.app/
