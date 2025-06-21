# Adsorption
# Detailed Adsorption Analysis App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://adsorption.streamlit.app/)
![Language](https://img.shields.io/badge/Language-Python-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A comprehensive Streamlit web application designed for chemists, environmental scientists, and researchers to analyze adsorption experiment data. This tool automates the tedious and repetitive calculations and plotting involved in adsorption studies, allowing users to quickly process their data and visualize results.

The app is fully internationalized, supporting both **English** and **French**.

## 🚀 Live Demo

**Experience the application live at: [https://adsorption.streamlit.app/](https://adsorption.streamlit.app/)**

![App Screenshot](./screenshot.png)
*(Replace this placeholder with a real screenshot of your app)*

## ✨ Features

This application provides a full suite of tools for analyzing different types of adsorption studies:

#### General Features
- **🌍 Multilingual Interface:** Switch between English and French seamlessly.
- **⬆️ Data Upload:** Upload your experimental data directly via `.csv` or `.xlsx` files.
- **✏️ Interactive Data Editor:** Manually enter or edit data directly in the application's sidebar.
- **📊 Interactive Plots:** All graphs are generated using Plotly for an interactive experience (zoom, pan, inspect data points).
- **📥 Data & Plot Export:** Download calculated data as a CSV file and export publication-quality plots as PNG images.

#### Analytical Modules
1.  **🔬 Calibration:**
    -   Perform linear regression on your calibration standards (Concentration vs. Absorbance).
    -   Automatically calculate the slope, intercept, and R² coefficient of determination.

2.  **🧪 Isotherm Analysis:**
    -   Analyze experimental data to fit popular isotherm models.
    -   **Linearized Models:** Langmuir, Freundlich, and Temkin.
    -   **Non-Linear Model Fitting:** Langmuir, Freundlich, and Temkin.
    -   Calculates key parameters (qm, KL, KF, n, B₁, KT).
    -   Provides a model comparison table to easily see which model fits best based on R².

3.  **⏳ Kinetic Analysis:**
    -   Analyze time-based adsorption data.
    -   **Non-Linear Models:** Pseudo-First-Order (PFO) and Pseudo-Second-Order (PSO).
    -   **Linearized Models:** PFO and PSO.
    -   **Mechanism Analysis:** Intraparticle Diffusion (IPD) model (Weber-Morris plot).
    -   Calculates kinetic parameters (qe, k₁, k₂, k_id, C).

4.  **🌡️ Thermodynamics Analysis:**
    -   Calculates thermodynamic parameters from temperature-effect data.
    -   Uses the Van't Hoff equation to determine **Enthalpy (ΔH°)**, **Entropy (ΔS°)**, and **Gibbs Free Energy (ΔG°)**.
    -   Indicates whether the adsorption process is exothermic/endothermic and spontaneous.

5.  **📈 Parameter Effect Studies:**
    -   **pH Effect:** Visualize the effect of solution pH on adsorption capacity (qe).
    -   **Dosage Effect:** Analyze the impact of adsorbent mass on qe.
    -   **Temperature Effect:** Study how temperature influences adsorption.

## 📖 How to Use the App

1.  **Select Language:** Choose your preferred language from the sidebar.
2.  **Enter Calibration Data:** In the sidebar, expand the "1. Calibration" section. Upload your data or enter it manually. The calibration parameters will be calculated automatically and used for all other analyses.
3.  **Choose a Study:** Expand the section for the study you want to analyze (e.g., "2. Isotherm Study").
4.  **Input Data:** Enter the fixed conditions (e.g., mass, volume) and the experimental data (either by uploading a file or using the data editor).
5.  **Explore Results:** Navigate to the corresponding tab in the main window (e.g., "Isotherms") to view the calculated data, tables, and interactive plots.
6.  **Download:** Use the download buttons to save your results and figures.

## 💻 Installation and Running Locally

To run this application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mmalahi00/my-streamlit-app.git
    cd my-streamlit-app
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    *(Ensure you have a `requirements.txt` file in your repository with the content listed below)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run adsorption_app.py
    ```
    The application will open in your default web browser.

#### `requirements.txt`
```
streamlit
pandas
numpy
scipy
plotly
openpyxl
```

## 📂 Project Structure

The project is organized to separate UI components, logic, and assets, making it scalable and easy to maintain.

```
.
├── adsorption_app.py        # Main application script
├── sidebar_ui.py            # Renders all sidebar input components
├── translations.py          # Contains English and French text translations
├── utils.py                 # Utility functions (e.g., data validation, CSV conversion)
├── models.py                # Mathematical models for isotherms and kinetics
├── tabs/
│   ├── calibration_tab.py     # UI and logic for the Calibration tab
│   ├── isotherm_tab.py        # UI and logic for the Isotherm tab
│   ├── kinetic_tab.py         # UI and logic for the Kinetic tab
│   ├── dosage_tab.py          # UI and logic for the Dosage Effect tab
│   ├── ph_effect_tab.py       # UI and logic for the pH Effect tab
│   ├── temperature_tab.py     # UI and logic for the Temperature Effect tab
│   └── thermodynamics_tab.py  # UI and logic for the Thermodynamics tab
├── requirements.txt         # List of Python dependencies
└── README.md                # This file
```

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or find any bugs, please feel free to open an issue or submit a pull request.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Commit your changes (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/YourFeature`).
5.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
*Developed by [Mohamed MALLAHI](https://github.com/mmalahi00)*
