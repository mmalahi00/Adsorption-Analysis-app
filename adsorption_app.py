# --- START OF FILE adsorption_app.py ---

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress
from scipy.optimize import curve_fit
import io
import plotly.io as pio

# --- Internationalization Setup ---
TRANSLATIONS = {
    "fr": {
        # General App
        "app_page_title": "Analyse Adsorption Détaillée",
        "app_title": "🔬 Analyse Détaillée de l'Adsorption",
        "app_subtitle": "Entrez vos données par type d'étude dans la barre latérale et explorez les résultats dans les onglets.",
        "language_select_label": "Langue / Language",

        # Sidebar
        "sidebar_header": "⚙️ Paramètres et Données d'Entrée",
        "sidebar_expander_calib": "1. Étalonnage",
        "sidebar_calib_intro": "Concentration vs. Absorbance:",
        "sidebar_expander_isotherm": "2. Étude Isotherme",
        "sidebar_isotherm_fixed_conditions": "Conditions fixes pour cette isotherme:",
        "sidebar_isotherm_mass_label": "Masse Ads. (g)",
        "sidebar_isotherm_mass_help": "Masse d'adsorbant utilisée",
        "sidebar_isotherm_volume_label": "Volume Sol. (L)",
        "sidebar_isotherm_volume_help": "Volume de la solution d'adsorbat",
        "sidebar_isotherm_c0_abs_intro": "Entrez C0 (variable) vs. Absorbance Eq.:",
        "sidebar_expander_kinetic": "5. Étude Cinétique",
        "sidebar_kinetic_fixed_conditions": "Conditions fixes pour UNE expérience cinétique:",
        "sidebar_kinetic_c0_help_const": "Concentration initiale constante",
        "sidebar_kinetic_mass_help_const": "Masse d'adsorbant constante",
        "sidebar_kinetic_volume_help_const": "Volume de solution constant",
        "sidebar_kinetic_time_abs_intro": "Entrez Temps (variable) vs. Absorbance(t):",
        "sidebar_expander_ph": "3. Étude Effet pH",
        "sidebar_ph_fixed_conditions": "Conditions fixes pour l'étude du pH:",
        "sidebar_ph_ph_abs_intro": "Entrez pH (variable) vs. Absorbance Eq.:",
        "sidebar_expander_temp": "4. Étude Effet Température",
        "sidebar_temp_fixed_conditions": "Conditions fixes pour l'étude de la T°:",
        "sidebar_temp_temp_abs_intro": "Entrez T° (variable) vs. Absorbance Eq.:",
        "sidebar_expander_dosage": "6. Étude Dosage (Effet Masse)",
        "sidebar_dosage_fixed_conditions": "Conditions fixes pour l'étude de la masse:",
        "sidebar_dosage_mass_abs_intro": "Entrez Masse Ads. (variable) vs. Absorbance Eq.:",

        # Data Editor Column Configs
        "col_concentration_help": "Concentration standard (mg/L ou unité équivalente)",
        "col_absorbance": "Absorbance",
        "col_absorbance_help": "Absorbance mesurée correspondante",
        "col_c0_help": "Concentration initiale de l'adsorbat",
        "col_abs_eq_help": "Absorbance mesurée à l'équilibre",
        "col_time_min": "Temps (min)",
        "col_time_min_help": "Temps écoulé depuis le début",
        "col_abs_t_help": "Absorbance mesurée au temps t",
        "col_ph_help": "pH variable de l'expérience",
        "col_temp_c_help": "Température variable de l'expérience",
        "col_mass_ads_g": "Masse Ads. (g)",
        "col_mass_ads_g_help_variable": "Masse variable d'adsorbant utilisée",

        # Validation Messages
        "validate_missing_cols_warning": "Colonnes requises manquantes: {missing_cols}. Veuillez les ajouter.",
        "validate_col_not_found_error": "Colonne requise '{col}' introuvable pendant la validation.",
        "validate_mass_non_positive_warning": "La colonne 'Masse_Adsorbant_g' contient des valeurs non positives. Ces lignes seront ignorées.",
        "validate_volume_non_positive_warning": "La colonne 'Volume_L' contient des valeurs non positives. Ces lignes seront ignorées.",
        "validate_n_valid_points_success": "{count} points valides.",
        "validate_no_valid_points_warning": "Aucun point valide après conversion numérique et filtrage. Vérifiez vos entrées.",
        "validate_error_message": "Erreur de validation: {error}",
        "validate_enter_data_info": "Entrez au moins un point de données.",

        # Calibration Specific
        "calib_slope_near_zero_warning": "Pente de calibration proche de zéro. Vérifiez les données.",
        "calib_error_calc_warning": "Erreur calcul calibration: {error}",

        # Main Area
        "main_results_header": "📊 Résultats et Analyses",
        "tab_calib": " Étalonnage ",
        "tab_isotherm": " Isothermes ",
        "tab_kinetic": " Cinétique ",
        "tab_ph": " Effet pH ",
        "tab_dosage": " Dosage ",
        "tab_temp": " Effet T° ",
        "tab_thermo": " Thermodynamique ",

        # Calibration Tab
        "calib_tab_subheader": "Courbe d'Étalonnage",
        "calib_tab_plot_title": "Absorbance en fonction de la concentration",
        "calib_tab_legend_exp": "Points expérimentaux",
        "calib_tab_legend_reg": "Régression linéaire",
        "calib_tab_styled_xaxis_label": "La concentration (mg/l)",
        "calib_tab_download_styled_png_button": "📥 Télécharger la Figure Stylisée (PNG)",
        "calib_tab_download_styled_png_filename": "etalonnage_stylise.png",
        "calib_tab_params_header": "Paramètres de Calibration",
        "calib_tab_slope_metric": "Pente (A / (mg/L))",
        "calib_tab_r2_metric": "Coefficient de Détermination (R²)",
        "calib_tab_min_2_points_warning": "Au moins 2 points de données valides sont requis pour la calibration.",
        "calib_tab_params_not_calc_warning": "Les paramètres de calibration n'ont pas pu être calculés. Vérifiez les données.",
        "calib_tab_raw_data_plot_title": "Données d'Étalonnage Brutes",
        "calib_tab_enter_data_info": "Entrez des données d'étalonnage valides (min 2 points) dans la barre latérale.",
        "calib_plot_error_warning": "Erreur lors de la création du graphique d'étalonnage: {e_plot}",


        # Isotherm Tab
        "isotherm_tab_subheader": "Analyse de l'Isotherme d'Adsorption",
        "isotherm_spinner_ce_qe": "Calcul Ce/qe pour isothermes...",
        "isotherm_error_slope_zero": "Erreur lors du calcul Ce/qe: Pente de calibration nulle ou proche de zéro.",
        "isotherm_error_mass_non_positive": "Masse d'adsorbant non positive ({m_adsorbant}g) pour C0={c0}. Ce point sera ignoré.",
        "isotherm_warning_no_valid_points": "Aucun point de données isotherme valide après calcul Ce/qe et vérification de la masse.",
        "isotherm_success_ce_qe_calc": "Calcul Ce/qe pour isothermes terminé.",
        "isotherm_error_div_by_zero": "Erreur lors du calcul Ce/qe: Division par zéro détectée (pente de calibration nulle?).",
        "isotherm_error_ce_qe_calc_general": "Erreur lors du calcul Ce/qe pour l'isotherme: {e}",
        "isotherm_calculated_data_header": "##### Données Calculées (Ce vs qe)",
        "isotherm_download_data_button": "📥 DL Données Isotherme (Ce/qe)",
        "isotherm_download_data_filename": "isotherme_results.csv",
        "isotherm_exp_plot_header": "##### Courbe expérimentale d'adsorption (qe vs Ce)",
        "isotherm_exp_plot_legend": "Données expérimentales",
        "isotherm_download_exp_plot_button": "📥 Télécharger la Courbe Expérimentale (PNG)",
        "isotherm_download_exp_plot_filename": "courbe_experimentale_qe_Ce.png",
        "isotherm_exp_plot_error": "Erreur lors du traçage de la courbe expérimentale : {e_exp_plot}",
        "isotherm_linearization_header": "##### Linéarisation des Modèles",
        "isotherm_linearization_caption": "Les paramètres (qm, KL, KF, n) sont déterminés à partir de ces régressions linéaires.",
        "isotherm_langmuir_lin_header": "###### Langmuir Linéarisé",
        "isotherm_insufficient_variation_warning": "Variation insuffisante dans {var1} ou {var2} pour la régression {model}.",
        "isotherm_lin_plot_legend_fit": "Ajustement Linéaire",
        "isotherm_langmuir_lin_caption": "Pente = {slope_L_lin:.4f} (1 / (qm·KL)), Intercept = {intercept_L_lin:.4f} (1 / qm)",
        "isotherm_download_langmuir_lin_inv_button": "📥 Télécharger 1/qe vs 1/Ce (PNG)",
        "isotherm_download_langmuir_lin_inv_filename": "langmuir_lineaire_inv.png",
        "isotherm_error_export_langmuir_lin_inv": "Erreur export 1/qe vs 1/Ce : {e}",
        "isotherm_error_langmuir_lin_regression": "Erreur régression Langmuir linéarisé: {ve}",
        "isotherm_error_langmuir_lin_plot_creation": "Erreur lors de la création du graphique Langmuir linéarisé: {e_lin_L}",
        "isotherm_no_valid_data_langmuir_lin": "Pas de données valides (Ce>0, qe>0) pour le graphique Langmuir linéarisé.",
        "isotherm_freundlich_lin_header": "###### Freundlich Linéarisé",
        "isotherm_warning_filter_log_freundlich": "Filtrage Ce>0 et qe>0 appliqué pour le log Freundlich.",
        "isotherm_info_no_valid_data_freundlich_lin": "Pas de données valides (Ce>0, qe>0) pour le graphique Freundlich linéarisé.",
        "isotherm_freundlich_lin_caption": "Pente = {slope_F_lin:.4f} (1/n), Intercept = {intercept_F_lin:.4f} (log₁₀ KF)",
        "isotherm_download_freundlich_lin_button": "📥 Télécharger Freundlich Linéarisé (PNG)",
        "isotherm_download_freundlich_lin_filename": "freundlich_lineaire.png",
        "isotherm_error_export_freundlich_lin": "Erreur export Freundlich linéarisé : {e}",
        "isotherm_error_freundlich_lin_regression": "Erreur régression Freundlich linéarisé: {ve}",
        "isotherm_error_freundlich_lin_plot_creation": "Erreur lors de la création du graphique Freundlich linéarisé: {e_lin_F}",
        "isotherm_derived_params_header": "##### Paramètres Dérivés des Modèles Linéarisés",
        "isotherm_info_params_not_calculated": "Les paramètres n'ont pas pu être calculés à partir des ajustements linéarisés (vérifiez les données, les graphiques et les messages d'erreur).",
        "isotherm_warning_less_than_2_points_lin_fit": "Moins de 2 points de données avec Ce > 0 et qe > 0. Impossible de réaliser les ajustements linéarisés.",
        "isotherm_warning_ce_qe_no_results": "Le calcul Ce/qe n'a produit aucun résultat valide (vérifiez la calibration et les données d'entrée).",
        "isotherm_warning_provide_calib_data": "Veuillez d'abord fournir des données d'étalonnage valides et calculer les paramètres.",
        "isotherm_info_enter_isotherm_data": "Veuillez entrer des données pour l'étude isotherme dans la barre latérale.",

        # Kinetic Tab
        "kinetic_tab_main_header": "Analyse Cinétique d'Adsorption",
        "kinetic_spinner_qt_calc": "Calcul qt pour cinétique...",
        "kinetic_error_qt_calc_slope_zero": "Erreur lors du calcul Ct/qt: Pente de calibration nulle ou proche de zéro.",
        "kinetic_success_qt_calc": "Calcul qt pour cinétique terminé.",
        "kinetic_error_qt_calc_div_by_zero": "Erreur lors du calcul Ct/qt: Division par zéro détectée (pente de calibration nulle?).",
        "kinetic_error_qt_calc_general": "Erreur lors du calcul qt pour la cinétique: {e_qt}",
        "kinetic_error_missing_abs_t_col": "Colonne 'Absorbance_t' manquante dans les données cinétiques.",
        "kinetic_error_mass_volume_non_positive": "La masse d'adsorbant et le volume doivent être positifs pour le calcul de qt.",
        "kinetic_calculated_data_subheader": "Données Calculées (qt vs t)",
        "kinetic_download_data_button": "📥 DL Données Cinétiques (qt)",
        "kinetic_download_data_filename": "resultats_cinetique.csv",
        "kinetic_plot_qt_vs_t_subheader": "1. Effet du Temps de Contact (qt vs t)",
        "kinetic_plot_qt_vs_t_title": "Évolution de l'adsorption au cours du temps",
        "kinetic_plot_qt_vs_t_xaxis": "Temps (min)",
        "kinetic_plot_qt_vs_t_legend": "Données Exp.",
        "kinetic_download_qt_vs_t_button": "📥 Télécharger qt vs Temps (PNG)",
        "kinetic_download_qt_vs_t_filename": "qt_vs_temps.png",
        "kinetic_error_export_qt_vs_t": "Erreur export qt vs t : {e}",
        "kinetic_error_plot_qt_vs_t": "Erreur lors du traçage qt vs t : {e_qt_plot}",
        "kinetic_spinner_pfo_nl_calc": "Calcul des paramètres PFO (non-linéaire)...",
        "kinetic_warning_pfo_nl_calc_failed": "Calcul paramètres PFO (non-linéaire) échoué: {e}",
        "kinetic_spinner_pso_nl_calc": "Calcul des paramètres PSO (non-linéaire)...", # Hidden, but keys for completeness
        "kinetic_warning_pso_nl_calc_failed": "Calcul paramètres PSO (non-linéaire) échoué: {e}", # Hidden
        "kinetic_spinner_ipd_analysis": "Analyse Diffusion Intraparticulaire (IPD)...",
        "kinetic_warning_ipd_insufficient_variation": "Variation insuffisante dans √t ou qt pour la régression IPD.",
        "kinetic_warning_ipd_calc_failed_regression": "Calcul IPD échoué (régression linéaire): {ve}",
        "kinetic_warning_ipd_calc_failed_general": "Calcul IPD échoué: {e}",
        "kinetic_warning_ipd_not_enough_points": "Pas assez de points (Temps > 0) pour l'analyse IPD.",
        "kinetic_linearized_models_subheader": "Analyse des Modèles Linéarisés",
        "kinetic_pfo_lin_header": "###### 2. PFO Linéarisé: ln(qe-qt) vs t",
        "kinetic_warning_pfo_lin_insufficient_variation": "Variation insuffisante pour régression PFO linéarisée.",
        "kinetic_pfo_lin_plot_title": "PFO Linéarisé (R²={r2_pfo_lin:.4f})",
        "kinetic_pfo_lin_caption": "Pente = {slope_pfo_lin:.4f} (-k1), Intercept = {intercept_pfo_lin:.4f} (ln qe)\nR² = {r2_pfo_lin:.4f}, k1_lin = {k1_lin:.4f} min⁻¹",
        "kinetic_download_pfo_lin_button": "📥 Télécharger PFO Linéarisé (PNG)",
        "kinetic_download_pfo_lin_filename": "pfo_lineaire.png",
        "kinetic_error_export_pfo_lin": "Erreur export PFO linéarisé : {e}",
        "kinetic_error_pfo_lin_plot_regression": "Erreur graphique PFO linéarisé (régression): {ve}",
        "kinetic_error_pfo_lin_plot_general": "Erreur graphique PFO linéarisé: {e_pfo_lin}",
        "kinetic_warning_pfo_lin_not_enough_points": "Pas assez de points valides (t>0, qt < qe_calc) pour le graphique PFO linéarisé.",
        "kinetic_warning_pfo_nl_calc_required": "Calcul des paramètres PFO non-linéaires (pour qe) requis mais échoué.",
        "kinetic_info_pfo_lin_uses_nl_qe": "Le graphique PFO linéarisé utilise la valeur `qe` déterminée par l'ajustement non-linéaire.",
        "kinetic_pso_lin_header": "###### 3. PSO Linéarisé: t/qt vs t",
        "kinetic_warning_pso_lin_insufficient_variation": "Variation insuffisante pour régression PSO linéarisée.",
        "kinetic_pso_lin_plot_title": "PSO Linéarisé (R²={r2_pso_lin:.4f})",
        "kinetic_pso_lin_caption": "Pente = {slope_pso_lin:.4f} (1/qe), Intercept = {intercept_pso_lin:.4f} (1/(k2·qe²))\nR² = {r2_pso_lin:.4f}, qe_lin = {qe_lin:.3f} mg/g, k2_lin = {k2_lin:.4f} g·mg⁻¹·min⁻¹",
        "kinetic_download_pso_lin_button": "📥 Télécharger PSO Linéarisé (PNG)",
        "kinetic_download_pso_lin_filename": "pso_lineaire.png",
        "kinetic_error_export_pso_lin": "Erreur export PSO linéarisé : {e}",
        "kinetic_error_pso_lin_plot_regression": "Erreur graphique PSO linéarisé (régression): {ve}",
        "kinetic_error_pso_lin_plot_general": "Erreur graphique PSO linéarisé: {e_pso_lin}",
        "kinetic_warning_pso_lin_not_enough_points": "Pas assez de points valides (t>0 et qt>0) pour le graphique PSO linéarisé.",
        "kinetic_ipd_subheader": "4. Analyse de Diffusion Intraparticulaire (IPD)",
        "kinetic_ipd_plot_title": "qt vs √Temps",
        "kinetic_ipd_plot_xaxis": "√Temps (min⁰⁵)",
        "kinetic_ipd_plot_legend_fit": "IPD Fit Global (R²={r2_ipd:.3f})",
        "kinetic_ipd_caption_params": "Paramètres IPD (global) : k_id = {k_id:.4f} mg·g⁻¹·min⁻⁰⁵, C = {C_ipd:.3f} mg·g⁻¹, R² = {R2_IPD:.4f}",
        "kinetic_ipd_caption_interp": "Si la droite ne passe pas par l'origine (C ≠ 0), la diffusion intraparticulaire n'est pas la seule étape limitante.",
        "kinetic_download_ipd_lin_button": "📥 Télécharger IPD Linéarisé (PNG)",
        "kinetic_download_ipd_lin_filename": "ipd_lineaire.png",
        "kinetic_error_export_ipd_lin": "Erreur export IPD linéarisé : {e}",
        "kinetic_warning_ipd_params_calc_failed": "Calcul des paramètres IPD échoué.",
        "kinetic_info_ipd_fit_unavailable": "Ajustement IPD non disponible (pas assez de points ou erreur de calcul).",
        "kinetic_warning_no_data_for_ipd": "Pas de données cinétiques à tracer pour IPD.",
        "kinetic_warning_less_than_3_points": "Moins de 3 points de données cinétiques disponibles. Impossible d'analyser les modèles.",
        "kinetic_warning_qt_calc_no_results": "Le calcul qt n'a produit aucun résultat valide.",
        "kinetic_warning_provide_calib_data": "Veuillez d'abord fournir des données d'étalonnage valides.",
        "kinetic_info_enter_kinetic_data": "Veuillez entrer des données pour l'étude cinétique.",
        
        # pH Effect Tab
        "ph_effect_tab_header": "Effet du pH sur la Capacité d'Adsorption (qe)",
        "ph_effect_spinner_ce_qe": "Calcul Ce/qe pour effet pH...",
        "ph_effect_error_slope_zero": "Erreur calcul Ce/qe: Pente de calibration nulle.",
        "ph_effect_error_mass_non_positive": "Erreur calcul qe: Masse fixe non positive ({m_fixed}g).",
        "ph_effect_success_ce_qe_calc": "Calcul Ce/qe pour effet pH terminé.",
        "ph_effect_warning_no_valid_points": "Aucun point pH valide après calcul Ce/qe.",
        "ph_effect_error_ce_qe_calc_general": "Erreur lors du calcul Ce/qe pour l'effet pH: {e}",
        "ph_effect_calculated_data_header": "##### Données Calculées (qe vs pH)",
        "ph_effect_conditions_caption": "Conditions fixes: C0={C0}mg/L, m={m}g, V={V}L",
        "ph_effect_download_data_button": "📥 DL Données Effet pH",
        "ph_effect_download_data_filename": "effet_ph_results.csv",
        "ph_effect_plot_header": "##### Graphique qe vs pH",
        "ph_effect_plot_title": "Effet du pH sur qe",
        "ph_effect_plot_legend_trend": "Tendance",
        "ph_effect_download_styled_plot_button": "📥 Télécharger Figure pH Stylisée (PNG)",
        "ph_effect_download_styled_plot_filename": "effet_ph_stylise.png",
        "ph_effect_error_export_styled_plot": "Erreur export figure pH stylisée : {e_export_ph}",
        "ph_effect_error_plot_general": "Erreur lors du traçage Effet pH: {e_ph_plot}",
        "ph_effect_warning_ce_qe_no_results": "Le calcul Ce/qe n'a produit aucun résultat valide pour l'effet pH.",
        "ph_effect_info_enter_ph_data": "Veuillez entrer des données pour l'étude de l'effet du pH.",

        # Dosage Tab
        "dosage_tab_header": "Effet de la Dose d'Adsorbant (Masse)",
        "dosage_spinner_ce_qe": "Calcul Ce/qe pour effet dosage...",
        "dosage_error_slope_zero": "Erreur calcul Ce/qe: Pente de calibration nulle.",
        "dosage_error_volume_non_positive": "Erreur calcul qe: Volume fixe ({v_fixed}L) non valide.",
        "dosage_success_ce_qe_calc": "Calcul Ce/qe pour effet dosage terminé.",
        "dosage_warning_no_valid_points": "Aucun point de dosage valide après calcul Ce/qe.",
        "dosage_error_ce_qe_calc_general": "Erreur lors du calcul Ce/qe pour l'effet dosage: {calc_err_dos}",
        "dosage_error_ce_qe_calc_unexpected": "Erreur inattendue lors du calcul Ce/qe pour l'effet dosage: {e}",
        "dosage_calculated_data_header": "##### Données Calculées (qe vs Masse Adsorbant)",
        "dosage_conditions_caption": "Conditions fixes: C0={C0}mg/L, V={V}L",
        "dosage_download_data_button": "📥 DL Données Effet Dosage",
        "dosage_download_data_filename": "effet_dosage_results.csv",
        "dosage_plot_header": "##### Graphique qe vs Masse Adsorbant",
        "dosage_plot_title": "Effet de la Masse d'Adsorbant sur qe",
        "dosage_plot_xaxis": "Masse Adsorbant (g)",
        "dosage_plot_legend_trend": "Tendance",
        "dosage_download_styled_plot_button": "📥 Télécharger Figure Dosage Stylisée (PNG)",
        "dosage_download_styled_plot_filename": "effet_dosage_stylise.png",
        "dosage_error_export_styled_plot": "Erreur export figure dosage stylisée : {e_export_dos}",
        "dosage_error_plot_general": "Erreur lors du traçage Effet Dosage: {e_dos_plot}",
        "dosage_warning_ce_qe_no_results": "Le calcul Ce/qe n'a produit aucun résultat valide pour l'effet dosage.",
        "dosage_info_enter_dosage_data": "Veuillez entrer des données pour l'étude de l'effet de la dose (masse adsorbant).",

        # Temperature Effect Tab
        "temp_effect_tab_header": "Effet de la Température sur la Capacité d'Adsorption (qe)",
        "temp_effect_spinner_ce_qe": "Calcul Ce/qe pour effet T°...",
        "temp_effect_error_slope_zero": "Erreur calcul Ce/qe: Pente de calibration nulle.",
        "temp_effect_error_mass_non_positive": "Erreur calcul qe: Masse fixe non positive ({m_fixed}g).",
        "temp_effect_success_ce_qe_calc": "Calcul Ce/qe pour effet T° terminé.",
        "temp_effect_warning_no_valid_points": "Aucun point T° valide après calcul Ce/qe.",
        "temp_effect_error_ce_qe_calc_general": "Erreur lors du calcul Ce/qe pour l'effet T°: {e}",
        "temp_effect_calculated_data_header": "##### Données Calculées (qe vs T°)",
        "temp_effect_conditions_caption": "Conditions fixes: C0={C0}mg/L, m={m}g, V={V}L",
        "temp_effect_download_data_button": "📥 DL Données Effet T°",
        "temp_effect_download_data_filename": "effet_temp_results.csv",
        "temp_effect_plot_header": "##### Graphique qe vs T°",
        "temp_effect_plot_title": "Effet de la T° sur qe",
        "temp_effect_plot_xaxis": "Température (°C)",
        "temp_effect_plot_legend_trend": "Tendance",
        "temp_effect_download_styled_plot_button": "📥 Télécharger Figure Température Stylisée (PNG)",
        "temp_effect_download_styled_plot_filename": "effet_temperature_stylise.png",
        "temp_effect_error_export_styled_plot": "Erreur export figure température stylisée : {e_export_temp}",
        "temp_effect_error_plot_general": "Erreur lors du traçage Effet T°: {e_t_plot}",
        "temp_effect_warning_ce_qe_no_results": "Le calcul Ce/qe n'a produit aucun résultat valide pour l'effet T°.",
        "temp_effect_info_enter_temp_data": "Veuillez entrer des données pour l'étude de l'effet de la température.",

        # Thermodynamics Tab
        "thermo_tab_header": "Analyse Thermodynamique",
        "thermo_tab_intro_markdown": """
        Cette analyse utilise les données de l'étude **Effet Température**.
        Elle calcule Kd = qe / Ce (L/g) puis utilise **Van't Hoff** (ln(Kd) vs 1/T) pour déterminer ΔH° et ΔS°.
        """,
        "thermo_spinner_analysis": "Analyse thermodynamique basée sur Kd...",
        "thermo_warning_insufficient_variation_vant_hoff": "Analyse Van't Hoff impossible: variation insuffisante dans 1/T ou ln(Kd).",
        "thermo_success_analysis_kd": "Analyse thermodynamique basée sur Kd terminée.",
        "thermo_warning_not_enough_distinct_temps_kd": "Pas assez de T° distinctes avec Kd > 0 pour Van't Hoff.",
        "thermo_error_vant_hoff_kd": "Erreur analyse Van't Hoff (Kd): {e_vth}",
        "thermo_warning_not_enough_distinct_temps_ce": "Moins de 2 T° distinctes avec Ce > 0 pour l'analyse thermo.",
        "thermo_calculated_params_header": "#### Paramètres Thermodynamiques Calculés",
        "thermo_delta_h_help": "< 0: Exothermique, > 0: Endothermique.",
        "thermo_delta_s_help": "> 0: Augmentation désordre.",
        "thermo_r2_vant_hoff_help": "Qualité ajustement ln(Kd) vs 1/T.",
        "thermo_delta_g_header": "ΔG° (kJ/mol) à différentes T°:",
        "thermo_delta_g_spontaneous_caption": "ΔG° < 0 : Spontané.",
        "thermo_delta_g_not_calculated": "Non calculé.",
        "thermo_vant_hoff_plot_header": "#### Graphique de Van't Hoff (ln(Kd) vs 1/T)",
        "thermo_vant_hoff_plot_title": "Graphique de Van't Hoff",
        "thermo_vant_hoff_plot_legend_fit": "Ajustement Linéaire (R²={r2_vt:.3f})",
        "thermo_download_vant_hoff_styled_button": "📥 Télécharger Van’t Hoff Stylisé (PNG)",
        "thermo_download_vant_hoff_styled_filename": "vant_hoff_stylise.png",
        "thermo_error_export_vant_hoff_styled": "Erreur export Van’t Hoff stylisé : {e}",
        "thermo_error_plot_vant_hoff": "Erreur traçage Van't Hoff: {e_vt_plot}",
        "thermo_kd_coeffs_header": "##### Coefficients de Distribution (Kd) utilisés",
        "thermo_kd_table_temp_c": "Température (°C)",
        "thermo_kd_unavailable": "Non disponible.",
        "thermo_download_params_kd_button": "📥 DL Paramètres Thermo (Kd)",
        "thermo_download_params_kd_filename": "params_thermo_kd.csv",
        "thermo_download_data_vant_hoff_kd_button": "📥 DL Données Van't Hoff (Kd)",
        "thermo_download_data_vant_hoff_kd_filename": "vant_hoff_data_kd.csv",
        "thermo_info_provide_temp_data": "Veuillez fournir des données valides pour l'étude Effet T°.",
        "thermo_warning_less_than_2_distinct_temps": "Moins de 2 T° distinctes pour l'analyse thermo.",
        "thermo_warning_analysis_not_done_kd": "Analyse thermo basée sur Kd non réalisée (vérifiez messages/données).",
        "thermo_warning_params_calculated_differently": "Paramètres thermo existants calculés différemment. Réinitialisez si besoin.",
    },
    "en": {
        # General App
        "app_page_title": "Detailed Adsorption Analysis",
        "app_title": "🔬 Detailed Adsorption Analysis",
        "app_subtitle": "Enter your data by study type in the sidebar and explore the results in the tabs.",
        "language_select_label": "Language / Langue",

        # Sidebar
        "sidebar_header": "⚙️ Settings and Input Data",
        "sidebar_expander_calib": "1. Calibration",
        "sidebar_calib_intro": "Concentration vs. Absorbance:",
        "sidebar_expander_isotherm": "2. Isotherm Study",
        "sidebar_isotherm_fixed_conditions": "Fixed conditions for this isotherm:",
        "sidebar_isotherm_mass_label": "Ads. Mass (g)",
        "sidebar_isotherm_mass_help": "Mass of adsorbent used",
        "sidebar_isotherm_volume_label": "Sol. Volume (L)",
        "sidebar_isotherm_volume_help": "Volume of adsorbate solution",
        "sidebar_isotherm_c0_abs_intro": "Enter C0 (variable) vs. Absorbance Eq.:",
        "sidebar_expander_kinetic": "5. Kinetic Study",
        "sidebar_kinetic_fixed_conditions": "Fixed conditions for ONE kinetic experiment:",
        "sidebar_kinetic_c0_help_const": "Constant initial concentration",
        "sidebar_kinetic_mass_help_const": "Constant adsorbent mass",
        "sidebar_kinetic_volume_help_const": "Constant solution volume",
        "sidebar_kinetic_time_abs_intro": "Enter Time (variable) vs. Absorbance(t):",
        "sidebar_expander_ph": "3. pH Effect Study",
        "sidebar_ph_fixed_conditions": "Fixed conditions for pH study:",
        "sidebar_ph_ph_abs_intro": "Enter pH (variable) vs. Absorbance Eq.:",
        "sidebar_expander_temp": "4. Temperature Effect Study",
        "sidebar_temp_fixed_conditions": "Fixed conditions for T° study:",
        "sidebar_temp_temp_abs_intro": "Enter T° (variable) vs. Absorbance Eq.:",
        "sidebar_expander_dosage": "6. Dosage Study (Mass Effect)",
        "sidebar_dosage_fixed_conditions": "Fixed conditions for mass study:",
        "sidebar_dosage_mass_abs_intro": "Enter Ads. Mass (variable) vs. Absorbance Eq.:",

        # Data Editor Column Configs
        "col_concentration_help": "Standard concentration (mg/L or equivalent unit)",
        "col_absorbance": "Absorbance",
        "col_absorbance_help": "Corresponding measured absorbance",
        "col_c0_help": "Initial adsorbate concentration",
        "col_abs_eq_help": "Measured absorbance at equilibrium",
        "col_time_min": "Time (min)",
        "col_time_min_help": "Time elapsed since start",
        "col_abs_t_help": "Measured absorbance at time t",
        "col_ph_help": "Variable pH of the experiment",
        "col_temp_c_help": "Variable temperature of the experiment",
        "col_mass_ads_g": "Ads. Mass (g)",
        "col_mass_ads_g_help_variable": "Variable mass of adsorbent used",

        # Validation Messages
        "validate_missing_cols_warning": "Missing required columns: {missing_cols}. Please add them.",
        "validate_col_not_found_error": "Required column '{col}' not found during validation.",
        "validate_mass_non_positive_warning": "Column 'Masse_Adsorbant_g' (Adsorbent Mass) contains non-positive values. These rows will be ignored.",
        "validate_volume_non_positive_warning": "Column 'Volume_L' (Volume) contains non-positive values. These rows will be ignored.",
        "validate_n_valid_points_success": "{count} valid points.",
        "validate_no_valid_points_warning": "No valid points after numerical conversion and filtering. Check your inputs.",
        "validate_error_message": "Validation error: {error}",
        "validate_enter_data_info": "Enter at least one data point.",

        # Calibration Specific
        "calib_slope_near_zero_warning": "Calibration slope close to zero. Check data.",
        "calib_error_calc_warning": "Calibration calculation error: {error}",

        # Main Area
        "main_results_header": "📊 Results and Analyses",
        "tab_calib": " Calibration ",
        "tab_isotherm": " Isotherms ",
        "tab_kinetic": " Kinetics ",
        "tab_ph": " pH Effect ",
        "tab_dosage": " Dosage ",
        "tab_temp": " T° Effect ",
        "tab_thermo": " Thermodynamics ",

        # Calibration Tab
        "calib_tab_subheader": "Calibration Curve",
        "calib_tab_plot_title": "Absorbance vs. Concentration",
        "calib_tab_legend_exp": "Experimental points",
        "calib_tab_legend_reg": "Linear regression",
        "calib_tab_styled_xaxis_label": "Concentration (mg/L)",
        "calib_tab_download_styled_png_button": "📥 Download Styled Figure (PNG)",
        "calib_tab_download_styled_png_filename": "calibration_styled.png",
        "calib_tab_params_header": "Calibration Parameters",
        "calib_tab_slope_metric": "Slope (A / (mg/L))",
        "calib_tab_r2_metric": "Coefficient of Determination (R²)",
        "calib_tab_min_2_points_warning": "At least 2 valid data points are required for calibration.",
        "calib_tab_params_not_calc_warning": "Calibration parameters could not be calculated. Check data.",
        "calib_tab_raw_data_plot_title": "Raw Calibration Data",
        "calib_tab_enter_data_info": "Enter valid calibration data (min 2 points) in the sidebar.",
        "calib_plot_error_warning": "Error creating calibration plot: {e_plot}",

        # Isotherm Tab
        "isotherm_tab_subheader": "Adsorption Isotherm Analysis",
        "isotherm_spinner_ce_qe": "Calculating Ce/qe for isotherms...",
        "isotherm_error_slope_zero": "Error calculating Ce/qe: Calibration slope is zero or near zero.",
        "isotherm_error_mass_non_positive": "Non-positive adsorbent mass ({m_adsorbant}g) for C0={c0}. This point will be ignored.",
        "isotherm_warning_no_valid_points": "No valid isotherm data points after Ce/qe calculation and mass check.",
        "isotherm_success_ce_qe_calc": "Ce/qe calculation for isotherms complete.",
        "isotherm_error_div_by_zero": "Error calculating Ce/qe: Division by zero detected (calibration slope zero?).",
        "isotherm_error_ce_qe_calc_general": "Error calculating Ce/qe for isotherm: {e}",
        "isotherm_calculated_data_header": "##### Calculated Data (Ce vs qe)",
        "isotherm_download_data_button": "📥 DL Isotherm Data (Ce/qe)",
        "isotherm_download_data_filename": "isotherm_results.csv",
        "isotherm_exp_plot_header": "##### Experimental Adsorption Curve (qe vs Ce)",
        "isotherm_exp_plot_legend": "Experimental data",
        "isotherm_download_exp_plot_button": "📥 Download Experimental Curve (PNG)",
        "isotherm_download_exp_plot_filename": "experimental_curve_qe_Ce.png",
        "isotherm_exp_plot_error": "Error plotting experimental curve: {e_exp_plot}",
        "isotherm_linearization_header": "##### Model Linearization",
        "isotherm_linearization_caption": "Parameters (qm, KL, KF, n) are determined from these linear regressions.",
        "isotherm_langmuir_lin_header": "###### Linearized Langmuir",
        "isotherm_insufficient_variation_warning": "Insufficient variation in {var1} or {var2} for {model} regression.",
        "isotherm_lin_plot_legend_fit": "Linear Fit",
        "isotherm_langmuir_lin_caption": "Slope = {slope_L_lin:.4f} (1 / (qm·KL)), Intercept = {intercept_L_lin:.4f} (1 / qm)",
        "isotherm_download_langmuir_lin_inv_button": "📥 Download 1/qe vs 1/Ce (PNG)",
        "isotherm_download_langmuir_lin_inv_filename": "langmuir_linear_inv.png",
        "isotherm_error_export_langmuir_lin_inv": "Error exporting 1/qe vs 1/Ce: {e}",
        "isotherm_error_langmuir_lin_regression": "Error in linearized Langmuir regression: {ve}",
        "isotherm_error_langmuir_lin_plot_creation": "Error creating linearized Langmuir plot: {e_lin_L}",
        "isotherm_no_valid_data_langmuir_lin": "No valid data (Ce>0, qe>0) for linearized Langmuir plot.",
        "isotherm_freundlich_lin_header": "###### Linearized Freundlich",
        "isotherm_warning_filter_log_freundlich": "Filtering Ce>0 and qe>0 applied for Freundlich log.",
        "isotherm_info_no_valid_data_freundlich_lin": "No valid data (Ce>0, qe>0) for linearized Freundlich plot.",
        "isotherm_freundlich_lin_caption": "Slope = {slope_F_lin:.4f} (1/n), Intercept = {intercept_F_lin:.4f} (log₁₀ KF)",
        "isotherm_download_freundlich_lin_button": "📥 Download Linearized Freundlich (PNG)",
        "isotherm_download_freundlich_lin_filename": "freundlich_linear.png",
        "isotherm_error_export_freundlich_lin": "Error exporting linearized Freundlich: {e}",
        "isotherm_error_freundlich_lin_regression": "Error in linearized Freundlich regression: {ve}",
        "isotherm_error_freundlich_lin_plot_creation": "Error creating linearized Freundlich plot: {e_lin_F}",
        "isotherm_derived_params_header": "##### Parameters Derived from Linearized Models",
        "isotherm_info_params_not_calculated": "Parameters could not be calculated from linearized fits (check data, plots, and error messages).",
        "isotherm_warning_less_than_2_points_lin_fit": "Fewer than 2 data points with Ce > 0 and qe > 0. Cannot perform linearized fits.",
        "isotherm_warning_ce_qe_no_results": "Ce/qe calculation produced no valid results (check calibration and input data).",
        "isotherm_warning_provide_calib_data": "Please provide valid calibration data and calculate parameters first.",
        "isotherm_info_enter_isotherm_data": "Please enter data for the isotherm study in the sidebar.",

        # Kinetic Tab
        "kinetic_tab_main_header": "Adsorption Kinetic Analysis",
        "kinetic_spinner_qt_calc": "Calculating qt for kinetics...",
        "kinetic_error_qt_calc_slope_zero": "Error calculating Ct/qt: Calibration slope is zero or near zero.",
        "kinetic_success_qt_calc": "qt calculation for kinetics complete.",
        "kinetic_error_qt_calc_div_by_zero": "Error calculating Ct/qt: Division by zero detected (calibration slope zero?).",
        "kinetic_error_qt_calc_general": "Error calculating qt for kinetics: {e_qt}",
        "kinetic_error_missing_abs_t_col": "'Absorbance_t' column missing in kinetic data.",
        "kinetic_error_mass_volume_non_positive": "Adsorbent mass and volume must be positive for qt calculation.",
        "kinetic_calculated_data_subheader": "Calculated Data (qt vs t)",
        "kinetic_download_data_button": "📥 DL Kinetic Data (qt)",
        "kinetic_download_data_filename": "kinetic_results.csv",
        "kinetic_plot_qt_vs_t_subheader": "1. Effect of Contact Time (qt vs t)",
        "kinetic_plot_qt_vs_t_title": "Evolution of Adsorption Over Time",
        "kinetic_plot_qt_vs_t_xaxis": "Time (min)",
        "kinetic_plot_qt_vs_t_legend": "Exp. Data",
        "kinetic_download_qt_vs_t_button": "📥 Download qt vs Time (PNG)",
        "kinetic_download_qt_vs_t_filename": "qt_vs_time.png",
        "kinetic_error_export_qt_vs_t": "Error exporting qt vs t: {e}",
        "kinetic_error_plot_qt_vs_t": "Error plotting qt vs t: {e_qt_plot}",
        "kinetic_spinner_pfo_nl_calc": "Calculating PFO parameters (non-linear)...",
        "kinetic_warning_pfo_nl_calc_failed": "PFO parameter calculation (non-linear) failed: {e}",
        "kinetic_spinner_pso_nl_calc": "Calculating PSO parameters (non-linear)...",
        "kinetic_warning_pso_nl_calc_failed": "PSO parameter calculation (non-linear) failed: {e}",
        "kinetic_spinner_ipd_analysis": "Intraparticle Diffusion (IPD) Analysis...",
        "kinetic_warning_ipd_insufficient_variation": "Insufficient variation in √t or qt for IPD regression.",
        "kinetic_warning_ipd_calc_failed_regression": "IPD calculation failed (linear regression): {ve}",
        "kinetic_warning_ipd_calc_failed_general": "IPD calculation failed: {e}",
        "kinetic_warning_ipd_not_enough_points": "Not enough points (Time > 0) for IPD analysis.",
        "kinetic_linearized_models_subheader": "Analysis of Linearized Models",
        "kinetic_pfo_lin_header": "###### 2. Linearized PFO: ln(qe-qt) vs t",
        "kinetic_warning_pfo_lin_insufficient_variation": "Insufficient variation for linearized PFO regression.",
        "kinetic_pfo_lin_plot_title": "Linearized PFO (R²={r2_pfo_lin:.4f})",
        "kinetic_pfo_lin_plot_xaxis": "Time (min)",
        "kinetic_pfo_lin_caption": "Slope = {slope_pfo_lin:.4f} (-k1), Intercept = {intercept_pfo_lin:.4f} (ln qe)\nR² = {r2_pfo_lin:.4f}, k1_lin = {k1_lin:.4f} min⁻¹",
        "kinetic_download_pfo_lin_button": "📥 Download Linearized PFO (PNG)",
        "kinetic_download_pfo_lin_filename": "pfo_linear.png",
        "kinetic_error_export_pfo_lin": "Error exporting linearized PFO: {e}",
        "kinetic_error_pfo_lin_plot_regression": "Error in linearized PFO plot (regression): {ve}",
        "kinetic_error_pfo_lin_plot_general": "Error in linearized PFO plot: {e_pfo_lin}",
        "kinetic_warning_pfo_lin_not_enough_points": "Not enough valid points (t>0, qt < qe_calc) for linearized PFO plot.",
        "kinetic_warning_pfo_nl_calc_required": "Non-linear PFO parameter calculation (for qe) required but failed.",
        "kinetic_info_pfo_lin_uses_nl_qe": "The linearized PFO plot uses the `qe` value determined by non-linear fitting.",
        "kinetic_pso_lin_header": "###### 3. Linearized PSO: t/qt vs t",
        "kinetic_warning_pso_lin_insufficient_variation": "Insufficient variation for linearized PSO regression.",
        "kinetic_pso_lin_plot_title": "Linearized PSO (R²={r2_pso_lin:.4f})",
        "kinetic_pso_lin_caption": "Slope = {slope_pso_lin:.4f} (1/qe), Intercept = {intercept_pso_lin:.4f} (1/(k2·qe²))\nR² = {r2_pso_lin:.4f}, qe_lin = {qe_lin:.3f} mg/g, k2_lin = {k2_lin:.4f} g·mg⁻¹·min⁻¹",
        "kinetic_download_pso_lin_button": "📥 Download Linearized PSO (PNG)",
        "kinetic_download_pso_lin_filename": "pso_linear.png",
        "kinetic_error_export_pso_lin": "Error exporting linearized PSO: {e}",
        "kinetic_error_pso_lin_plot_regression": "Error in linearized PSO plot (regression): {ve}",
        "kinetic_error_pso_lin_plot_general": "Error in linearized PSO plot: {e_pso_lin}",
        "kinetic_warning_pso_lin_not_enough_points": "Not enough valid points (t>0 and qt>0) for linearized PSO plot.",
        "kinetic_ipd_subheader": "4. Intraparticle Diffusion (IPD) Analysis",
        "kinetic_ipd_plot_title": "qt vs √Time",
        "kinetic_ipd_plot_xaxis": "√Time (min⁰⁵)",
        "kinetic_ipd_plot_legend_fit": "IPD Fit Global (R²={r2_ipd:.3f})",
        "kinetic_ipd_caption_params": "IPD Parameters (global): k_id = {k_id:.4f} mg·g⁻¹·min⁻⁰⁵, C = {C_ipd:.3f} mg·g⁻¹, R² = {R2_IPD:.4f}",
        "kinetic_ipd_caption_interp": "If the line does not pass through the origin (C ≠ 0), intraparticle diffusion is not the sole rate-limiting step.",
        "kinetic_download_ipd_lin_button": "📥 Download Linearized IPD (PNG)",
        "kinetic_download_ipd_lin_filename": "ipd_linear.png",
        "kinetic_error_export_ipd_lin": "Error exporting linearized IPD: {e}",
        "kinetic_warning_ipd_params_calc_failed": "IPD parameter calculation failed.",
        "kinetic_info_ipd_fit_unavailable": "IPD fit unavailable (not enough points or calculation error).",
        "kinetic_warning_no_data_for_ipd": "No kinetic data to plot for IPD.",
        "kinetic_warning_less_than_3_points": "Fewer than 3 kinetic data points available. Cannot analyze models.",
        "kinetic_warning_qt_calc_no_results": "qt calculation produced no valid results.",
        "kinetic_warning_provide_calib_data": "Please provide valid calibration data first.",
        "kinetic_info_enter_kinetic_data": "Please enter data for the kinetic study.",

        # pH Effect Tab
        "ph_effect_tab_header": "Effect of pH on Adsorption Capacity (qe)",
        "ph_effect_spinner_ce_qe": "Calculating Ce/qe for pH effect...",
        "ph_effect_error_slope_zero": "Error calculating Ce/qe: Calibration slope is zero.",
        "ph_effect_error_mass_non_positive": "Error calculating qe: Fixed mass is non-positive ({m_fixed}g).",
        "ph_effect_success_ce_qe_calc": "Ce/qe calculation for pH effect complete.",
        "ph_effect_warning_no_valid_points": "No valid pH points after Ce/qe calculation.",
        "ph_effect_error_ce_qe_calc_general": "Error calculating Ce/qe for pH effect: {e}",
        "ph_effect_calculated_data_header": "##### Calculated Data (qe vs pH)",
        "ph_effect_conditions_caption": "Fixed conditions: C0={C0}mg/L, m={m}g, V={V}L",
        "ph_effect_download_data_button": "📥 DL pH Effect Data",
        "ph_effect_download_data_filename": "ph_effect_results.csv",
        "ph_effect_plot_header": "##### qe vs pH Plot",
        "ph_effect_plot_title": "Effect of pH on qe",
        "ph_effect_plot_legend_trend": "Trend",
        "ph_effect_download_styled_plot_button": "📥 Download Styled pH Figure (PNG)",
        "ph_effect_download_styled_plot_filename": "ph_effect_styled.png",
        "ph_effect_error_export_styled_plot": "Error exporting styled pH figure: {e_export_ph}",
        "ph_effect_error_plot_general": "Error plotting pH Effect: {e_ph_plot}",
        "ph_effect_warning_ce_qe_no_results": "Ce/qe calculation produced no valid results for pH effect.",
        "ph_effect_info_enter_ph_data": "Please enter data for the pH effect study.",

        # Dosage Tab
        "dosage_tab_header": "Effect of Adsorbent Dose (Mass)",
        "dosage_spinner_ce_qe": "Calculating Ce/qe for dosage effect...",
        "dosage_error_slope_zero": "Error calculating Ce/qe: Calibration slope is zero.",
        "dosage_error_volume_non_positive": "Error calculating qe: Fixed volume ({v_fixed}L) is invalid.",
        "dosage_success_ce_qe_calc": "Ce/qe calculation for dosage effect complete.",
        "dosage_warning_no_valid_points": "No valid dosage points after Ce/qe calculation.",
        "dosage_error_ce_qe_calc_general": "Error calculating Ce/qe for dosage effect: {calc_err_dos}",
        "dosage_error_ce_qe_calc_unexpected": "Unexpected error calculating Ce/qe for dosage effect: {e}",
        "dosage_calculated_data_header": "##### Calculated Data (qe vs Adsorbent Mass)",
        "dosage_conditions_caption": "Fixed conditions: C0={C0}mg/L, V={V}L",
        "dosage_download_data_button": "📥 DL Dosage Effect Data",
        "dosage_download_data_filename": "dosage_effect_results.csv",
        "dosage_plot_header": "##### qe vs Adsorbent Mass Plot",
        "dosage_plot_title": "Effect of Adsorbent Mass on qe",
        "dosage_plot_xaxis": "Adsorbent Mass (g)",
        "dosage_plot_legend_trend": "Trend",
        "dosage_download_styled_plot_button": "📥 Download Styled Dosage Figure (PNG)",
        "dosage_download_styled_plot_filename": "dosage_effect_styled.png",
        "dosage_error_export_styled_plot": "Error exporting styled dosage figure: {e_export_dos}",
        "dosage_error_plot_general": "Error plotting Dosage Effect: {e_dos_plot}",
        "dosage_warning_ce_qe_no_results": "Ce/qe calculation produced no valid results for dosage effect.",
        "dosage_info_enter_dosage_data": "Please enter data for the dosage effect study (adsorbent mass).",

        # Temperature Effect Tab
        "temp_effect_tab_header": "Effect of Temperature on Adsorption Capacity (qe)",
        "temp_effect_spinner_ce_qe": "Calculating Ce/qe for T° effect...",
        "temp_effect_error_slope_zero": "Error calculating Ce/qe: Calibration slope is zero.",
        "temp_effect_error_mass_non_positive": "Error calculating qe: Fixed mass is non-positive ({m_fixed}g).",
        "temp_effect_success_ce_qe_calc": "Ce/qe calculation for T° effect complete.",
        "temp_effect_warning_no_valid_points": "No valid T° points after Ce/qe calculation.",
        "temp_effect_error_ce_qe_calc_general": "Error calculating Ce/qe for T° effect: {e}",
        "temp_effect_calculated_data_header": "##### Calculated Data (qe vs T°)",
        "temp_effect_conditions_caption": "Fixed conditions: C0={C0}mg/L, m={m}g, V={V}L",
        "temp_effect_download_data_button": "📥 DL T° Effect Data",
        "temp_effect_download_data_filename": "temp_effect_results.csv",
        "temp_effect_plot_header": "##### qe vs T° Plot",
        "temp_effect_plot_title": "Effect of T° on qe",
        "temp_effect_plot_xaxis": "Temperature (°C)",
        "temp_effect_plot_legend_trend": "Trend",
        "temp_effect_download_styled_plot_button": "📥 Download Styled Temperature Figure (PNG)",
        "temp_effect_download_styled_plot_filename": "temperature_effect_styled.png",
        "temp_effect_error_export_styled_plot": "Error exporting styled temperature figure: {e_export_temp}",
        "temp_effect_error_plot_general": "Error plotting T° Effect: {e_t_plot}",
        "temp_effect_warning_ce_qe_no_results": "Ce/qe calculation produced no valid results for T° effect.",
        "temp_effect_info_enter_temp_data": "Please enter data for the temperature effect study.",

        # Thermodynamics Tab
        "thermo_tab_header": "Thermodynamic Analysis",
        "thermo_tab_intro_markdown": """
        This analysis uses data from the **Temperature Effect** study.
        It calculates Kd = qe / Ce (L/g) then uses **Van't Hoff** (ln(Kd) vs 1/T) to determine ΔH° and ΔS°.
        """,
        "thermo_spinner_analysis": "Thermodynamic analysis based on Kd...",
        "thermo_warning_insufficient_variation_vant_hoff": "Van't Hoff analysis impossible: insufficient variation in 1/T or ln(Kd).",
        "thermo_success_analysis_kd": "Thermodynamic analysis based on Kd complete.",
        "thermo_warning_not_enough_distinct_temps_kd": "Not enough distinct T° with Kd > 0 for Van't Hoff.",
        "thermo_error_vant_hoff_kd": "Van't Hoff analysis error (Kd): {e_vth}",
        "thermo_warning_not_enough_distinct_temps_ce": "Fewer than 2 distinct T° with Ce > 0 for thermo analysis.",
        "thermo_calculated_params_header": "#### Calculated Thermodynamic Parameters",
        "thermo_delta_h_help": "< 0: Exothermic, > 0: Endothermic.",
        "thermo_delta_s_help": "> 0: Increased disorder.",
        "thermo_r2_vant_hoff_help": "Goodness of fit for ln(Kd) vs 1/T.",
        "thermo_delta_g_header": "ΔG° (kJ/mol) at different T°:",
        "thermo_delta_g_spontaneous_caption": "ΔG° < 0 : Spontaneous.",
        "thermo_delta_g_not_calculated": "Not calculated.",
        "thermo_vant_hoff_plot_header": "#### Van't Hoff Plot (ln(Kd) vs 1/T)",
        "thermo_vant_hoff_plot_title": "Van't Hoff Plot",
        "thermo_vant_hoff_plot_legend_fit": "Linear Fit (R²={r2_vt:.3f})",
        "thermo_download_vant_hoff_styled_button": "📥 Download Styled Van’t Hoff (PNG)",
        "thermo_download_vant_hoff_styled_filename": "vant_hoff_styled.png",
        "thermo_error_export_vant_hoff_styled": "Error exporting styled Van’t Hoff: {e}",
        "thermo_error_plot_vant_hoff": "Error plotting Van't Hoff: {e_vt_plot}",
        "thermo_kd_coeffs_header": "##### Distribution Coefficients (Kd) Used",
        "thermo_kd_table_temp_c": "Temperature (°C)",
        "thermo_kd_unavailable": "Not available.",
        "thermo_download_params_kd_button": "📥 DL Thermo Parameters (Kd)",
        "thermo_download_params_kd_filename": "thermo_params_kd.csv",
        "thermo_download_data_vant_hoff_kd_button": "📥 DL Van't Hoff Data (Kd)",
        "thermo_download_data_vant_hoff_kd_filename": "vant_hoff_data_kd.csv",
        "thermo_info_provide_temp_data": "Please provide valid data for the T° Effect study.",
        "thermo_warning_less_than_2_distinct_temps": "Fewer than 2 distinct T° for thermo analysis.",
        "thermo_warning_analysis_not_done_kd": "Thermo analysis based on Kd not performed (check messages/data).",
        "thermo_warning_params_calculated_differently": "Existing thermo parameters calculated differently. Reset if needed.",
    }
}

def _t(key, **kwargs):
    lang = st.session_state.get('language', 'fr')
    # Fallback to French if a key is missing in the current language, then to a default message
    text_template = TRANSLATIONS.get(lang, {}).get(key)
    if text_template is None and lang != 'fr': # Try French as fallback
        text_template = TRANSLATIONS.get('fr', {}).get(key, f"FR_LT__{key}") # LT for Missing Translation Key in French
    if text_template is None: # Ultimate fallback
         return f"LT__{key}" # LT for Missing Translation Key
    if kwargs:
        return text_template.format(**kwargs)
    return text_template

# --- Initialize session state for language ---
if 'language' not in st.session_state:
    st.session_state.language = 'fr' # Default language

# --- Configuration de la Page ---
st.set_page_config(
    page_title="app_page_title",
    page_icon="🔬",
    layout="wide"
)



# --- TITRE PRINCIPAL ---
st.title(_t("app_title"))
st.markdown(_t("app_subtitle"))
st.markdown("---")

# --- Language Selector ---
current_lang_index = 0 if st.session_state.language == 'fr' else 1
selected_lang = st.sidebar.selectbox(
    _t("language_select_label"),
    options=['fr', 'en'],
    format_func=lambda x: "Français" if x == 'fr' else "English",
    index=current_lang_index,
    key='lang_selector'
)
if selected_lang != st.session_state.language:
    st.session_state.language = selected_lang
    st.rerun()

# --- Définition des Fonctions Modèles (Gardées pour Cinétique, etc.) ---
def langmuir_model(Ce, qm, KL):
    KL = max(KL, 0)
    Ce_safe = np.maximum(0, Ce)
    epsilon = 1e-12
    denominator = 1 + KL * Ce_safe
    denominator = np.where(np.abs(denominator) < epsilon, np.sign(denominator + epsilon) * epsilon, denominator)
    return np.where(denominator != 0, (qm * KL * Ce_safe) / denominator, 0)

def freundlich_model(Ce, KF, n_inv):
    KF = max(KF, 0)
    n_inv = max(n_inv, 0)
    epsilon = 1e-12
    Ce_safe = np.maximum(Ce, epsilon)
    return KF * Ce_safe**n_inv

def pfo_model(t, qe, k1):
    k1 = max(k1, 0)
    k1_safe = max(k1, 1e-12)
    t_safe = np.clip(t, 0, None)
    exp_arg = -k1 * np.clip(t_safe, 0, 700 / k1_safe)
    exp_term = np.exp(exp_arg)
    return qe * (1 - exp_term)

def pso_model(t, qe, k2):
    qe = max(qe, 0)
    k2 = max(k2, 0)
    qe_safe = max(qe, 1e-12)
    k2_safe = max(k2, 1e-12)
    t_safe = np.clip(t, 0, None)
    epsilon = 1e-12
    denominator = 1 + k2_safe * qe_safe * t_safe
    denominator = np.where(np.abs(denominator) < epsilon, 1e-9 * np.sign(denominator + epsilon), denominator)
    return (k2_safe * qe_safe**2 * t_safe) / denominator

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False, sep=';').encode('utf-8')

# --- Initialisation de l'état de session (Corrigée) ---
default_keys = {
    'calib_df_input': None, 'calibration_params': None,
    'isotherm_input': None, 'isotherm_results': None,
    'langmuir_params_lin': None, 'freundlich_params_lin': None, # Clés pour les paramètres linéaires
    'ph_effect_input': None, 'ph_effect_results': None,
    'temp_effect_input': None, 'temp_effect_results': None, 'thermo_params': None,
    'kinetic_input': None, 'kinetic_results_df': None, 'pfo_params': None, 'pso_params': None, 'dosage_input': None, 'dosage_results': None, 'ipd_params_list': []
}
for key, default_value in default_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Fonction de Validation Générique pour Data Editor ---
def validate_data_editor(edited_df, required_cols, success_message_key, error_message_key):
    validated_df = None
    if edited_df is not None and not edited_df.empty:
        try:
            temp_df = edited_df.copy()
            all_cols_present = all(col in temp_df.columns for col in required_cols)
            if not all_cols_present:
                 missing_cols = [col for col in required_cols if col not in temp_df.columns]
                 st.sidebar.warning(_t("validate_missing_cols_warning", missing_cols=', '.join(missing_cols)), icon="⚠️")
                 return None

            for col in required_cols:
                 if col in temp_df.columns: # Vérifier si la colonne existe avant conversion
                    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
                 else: # Ne devrait pas arriver grace a la verification all_cols_present, mais securite
                     st.sidebar.error(_t("validate_col_not_found_error", col=col))
                     return None

            temp_df.dropna(subset=required_cols, inplace=True)

            if 'Masse_Adsorbant_g' in temp_df.columns:
                 if (temp_df['Masse_Adsorbant_g'] <= 0).any():
                     st.sidebar.warning(_t("validate_mass_non_positive_warning"), icon="⚠️")
                     temp_df = temp_df[temp_df['Masse_Adsorbant_g'] > 0]

            if 'Volume_L' in temp_df.columns:
                 if (temp_df['Volume_L'] <= 0).any():
                     st.sidebar.warning(_t("validate_volume_non_positive_warning"), icon="⚠️")
                     temp_df = temp_df[temp_df['Volume_L'] > 0]

            if not temp_df.empty:
                validated_df = temp_df
                st.sidebar.success(_t("validate_n_valid_points_success", count=len(validated_df)))
        
            else:
                st.sidebar.warning(_t("validate_no_valid_points_warning"))
        except Exception as e:
            st.sidebar.error(_t("validate_error_message", error=e))
    else:
        st.sidebar.info(_t("validate_enter_data_info"))
    return validated_df


# --- SIDEBAR (Entrées Organisées par Expander) ---
st.sidebar.header(_t("sidebar_header"))

with st.sidebar.expander(_t("sidebar_expander_calib"), expanded=False):
    st.write(_t("sidebar_calib_intro"))
    initial_calib_data = pd.DataFrame({'Concentration': [], 'Absorbance': []})
    edited_calib = st.data_editor(
        initial_calib_data.astype(float), num_rows="dynamic", key="calib_editor",
        column_config={
            "Concentration": st.column_config.NumberColumn(format="%.4f", required=True, help=_t("col_concentration_help")),
            "Absorbance": st.column_config.NumberColumn(format="%.4f", required=True, help=_t("col_absorbance_help"))
        })
    st.session_state['calib_df_input'] = validate_data_editor(edited_calib, ['Concentration', 'Absorbance'], 'calib_ok', 'calib_err')

with st.sidebar.expander(_t("sidebar_expander_isotherm"), expanded=False):
    st.write(_t("sidebar_isotherm_fixed_conditions"))
    m_iso = st.number_input(_t("sidebar_isotherm_mass_label"), min_value=1e-9, value=0.02, format="%.4f", key="m_iso_input", help=_t("sidebar_isotherm_mass_help"))
    V_iso = st.number_input(_t("sidebar_isotherm_volume_label"), min_value=1e-6, value=0.05, format="%.4f", key="V_iso_input", help=_t("sidebar_isotherm_volume_help"))
    st.write(_t("sidebar_isotherm_c0_abs_intro"))
    initial_iso_data = pd.DataFrame({'Concentration_Initiale_C0': [], 'Absorbance_Equilibre': []})
    edited_iso = st.data_editor(
        initial_iso_data.astype(float), num_rows="dynamic", key="iso_editor",
        column_config={
            "Concentration_Initiale_C0": st.column_config.NumberColumn("C0 (mg/L)", format="%.4f", required=True, help=_t("col_c0_help")),
            "Absorbance_Equilibre": st.column_config.NumberColumn("Abs Eq.", format="%.4f", required=True, help=_t("col_abs_eq_help"))
        })

    iso_df_validated = validate_data_editor(edited_iso, ['Concentration_Initiale_C0', 'Absorbance_Equilibre'], 'iso_ok', 'iso_err')
    current_iso_input_in_state = st.session_state.get('isotherm_input') # Get current state

    if iso_df_validated is not None:
        new_iso_input = {'data': iso_df_validated, 'params': {'m': m_iso, 'V': V_iso}}
        needs_update = False
        if current_iso_input_in_state is None: # First valid run or after reset/invalid
             needs_update = True
        elif isinstance(current_iso_input_in_state, dict): # Only compare if state holds a dict
            # Check data equality
            data_equal = iso_df_validated.equals(current_iso_input_in_state.get('data'))
            # Check params equality
            params_equal = (m_iso == current_iso_input_in_state.get('params', {}).get('m') and
                            V_iso == current_iso_input_in_state.get('params', {}).get('V'))
            if not data_equal or not params_equal:
                needs_update = True
        # else: current state is something unexpected, treat as needing update maybe? Or ignore. Let's update.
        # elif not isinstance(current_iso_input_in_state, dict): needs_update = True # Optionally handle weird states

        if needs_update:
            st.session_state['isotherm_input'] = new_iso_input
            st.session_state['isotherm_results'] = None
            st.session_state['langmuir_params_lin'] = None # Effacer params linéaires
            st.session_state['freundlich_params_lin'] = None # Effacer params linéaires

    elif current_iso_input_in_state is not None: # Input became invalid, clear state if it was previously valid
        st.session_state['isotherm_input'] = None
        st.session_state['isotherm_results'] = None
        st.session_state['langmuir_params_lin'] = None
        st.session_state['freundlich_params_lin'] = None

with st.sidebar.expander(_t("sidebar_expander_kinetic"), expanded=False):
    st.write(_t("sidebar_kinetic_fixed_conditions"))
    C0_k = st.number_input("C0 (mg/L)", min_value=0.0, value=10.0, format="%.4f", key="C0_k_cin", help=_t("sidebar_kinetic_c0_help_const"))
    m_k = st.number_input(_t("sidebar_isotherm_mass_label"), min_value=1e-9, value=0.02, format="%.4f", key="m_k_cin", help=_t("sidebar_kinetic_mass_help_const"))
    V_k = st.number_input(_t("sidebar_isotherm_volume_label"), min_value=1e-6, value=0.05, format="%.4f", key="V_k_cin", help=_t("sidebar_kinetic_volume_help_const"))
    st.write(_t("sidebar_kinetic_time_abs_intro"))
    initial_kinetic_data = pd.DataFrame({'Temps_min': [], 'Absorbance_t': []})
    edited_kin = st.data_editor(
        initial_kinetic_data.astype(float), num_rows="dynamic", key="kin_editor",
        column_config={
            "Temps_min": st.column_config.NumberColumn(_t("col_time_min"), format="%.2f", required=True, min_value=0, help=_t("col_time_min_help")),
            "Absorbance_t": st.column_config.NumberColumn("Abs(t)", format="%.4f", required=True, help=_t("col_abs_t_help"))
        })

    kin_df_validated = validate_data_editor(edited_kin, ['Temps_min', 'Absorbance_t'], 'kin_ok', 'kin_err')
    current_kin_input_in_state = st.session_state.get('kinetic_input') # Get current state

    if kin_df_validated is not None:
        kin_df_validated.sort_values(by='Temps_min', inplace=True) # Trier par temps
        new_kin_input = {'data': kin_df_validated, 'params': {'C0': C0_k, 'm': m_k, 'V': V_k}}
        needs_update = False
        if current_kin_input_in_state is None:
            needs_update = True
        elif isinstance(current_kin_input_in_state, dict):
            # Need to sort the stored data as well for accurate comparison
            stored_data = current_kin_input_in_state.get('data')
            if stored_data is not None:
                try: # Handle potential error if stored data isn't a DataFrame
                   stored_data_sorted = stored_data.sort_values(by='Temps_min').reset_index(drop=True)
                   kin_df_validated_sorted = kin_df_validated.reset_index(drop=True)
                   data_equal = kin_df_validated_sorted.equals(stored_data_sorted)
                except AttributeError:
                   data_equal = False # Treat as not equal if stored data isn't DataFrame like
            else:
                data_equal = False # No stored data to compare

            params_equal = (C0_k == current_kin_input_in_state.get('params', {}).get('C0') and
                            m_k == current_kin_input_in_state.get('params', {}).get('m') and
                            V_k == current_kin_input_in_state.get('params', {}).get('V'))
            if not data_equal or not params_equal:
                needs_update = True

        if needs_update:
            st.session_state['kinetic_input'] = new_kin_input
            st.session_state['kinetic_results_df'] = None
            st.session_state['pfo_params'] = None
            st.session_state['pso_params'] = None
            st.session_state['ipd_params_list'] = []

    elif current_kin_input_in_state is not None: # Input became invalid
        st.session_state['kinetic_input'] = None
        st.session_state['kinetic_results_df'] = None
        st.session_state['pfo_params'] = None
        st.session_state['pso_params'] = None
        st.session_state['ipd_params_list'] = []

with st.sidebar.expander(_t("sidebar_expander_ph"), expanded=False):
    st.write(_t("sidebar_ph_fixed_conditions"))
    C0_ph = st.number_input("C0 (mg/L)", min_value=0.0, value=20.0, format="%.4f", key="C0_ph_input", help=_t("sidebar_kinetic_c0_help_const"))
    m_ph = st.number_input(_t("sidebar_isotherm_mass_label"), min_value=1e-9, value=0.02, format="%.4f", key="m_ph_input", help=_t("sidebar_kinetic_mass_help_const"))
    V_ph = st.number_input(_t("sidebar_isotherm_volume_label"), min_value=1e-6, value=0.05, format="%.4f", key="V_ph_input", help=_t("sidebar_kinetic_volume_help_const"))
    st.write(_t("sidebar_ph_ph_abs_intro"))
    initial_ph_data = pd.DataFrame({'pH': [], 'Absorbance_Equilibre': []})
    edited_ph = st.data_editor(
        initial_ph_data.astype(float), num_rows="dynamic", key="ph_editor",
        column_config={
            "pH": st.column_config.NumberColumn("pH", format="%.2f", required=True, help=_t("col_ph_help")),
            "Absorbance_Equilibre": st.column_config.NumberColumn("Abs Eq.", format="%.4f", required=True, help=_t("col_abs_eq_help"))
        })

    ph_df_validated = validate_data_editor(edited_ph, ['pH', 'Absorbance_Equilibre'], 'ph_ok', 'ph_err')
    current_ph_input_in_state = st.session_state.get('ph_effect_input') # Get current state

    if ph_df_validated is not None:
        new_ph_input = {'data': ph_df_validated, 'params': {'C0': C0_ph, 'm': m_ph, 'V': V_ph}}
        needs_update = False
        if current_ph_input_in_state is None:
             needs_update = True
        elif isinstance(current_ph_input_in_state, dict):
            data_equal = ph_df_validated.equals(current_ph_input_in_state.get('data'))
            params_equal = (C0_ph == current_ph_input_in_state.get('params', {}).get('C0') and
                            m_ph == current_ph_input_in_state.get('params', {}).get('m') and
                            V_ph == current_ph_input_in_state.get('params', {}).get('V'))
            if not data_equal or not params_equal:
                needs_update = True

        if needs_update:
            st.session_state['ph_effect_input'] = new_ph_input
            st.session_state['ph_effect_results'] = None # Clear results

    elif current_ph_input_in_state is not None: # Input became invalid
        st.session_state['ph_effect_input'] = None
        st.session_state['ph_effect_results'] = None


with st.sidebar.expander(_t("sidebar_expander_temp"), expanded=False):
    st.write(_t("sidebar_temp_fixed_conditions"))
    C0_t = st.number_input("C0 (mg/L)", min_value=0.0, value=50.0, format="%.4f", key="C0_t_input", help=_t("sidebar_kinetic_c0_help_const"))
    m_t = st.number_input(_t("sidebar_isotherm_mass_label"), min_value=1e-9, value=0.02, format="%.4f", key="m_t_input", help=_t("sidebar_kinetic_mass_help_const"))
    V_t = st.number_input(_t("sidebar_isotherm_volume_label"), min_value=1e-6, value=0.05, format="%.4f", key="V_t_input", help=_t("sidebar_kinetic_volume_help_const"))
    st.write(_t("sidebar_temp_temp_abs_intro"))
    initial_t_data = pd.DataFrame({'Temperature_C': [], 'Absorbance_Equilibre': []})
    edited_t = st.data_editor(
        initial_t_data.astype(float), num_rows="dynamic", key="temp_editor",
        column_config={
            "Temperature_C": st.column_config.NumberColumn("T (°C)", format="%.1f", required=True, help=_t("col_temp_c_help")),
            "Absorbance_Equilibre": st.column_config.NumberColumn("Abs Eq.", format="%.4f", required=True, help=_t("col_abs_eq_help"))
        })

    temp_df_validated = validate_data_editor(edited_t, ['Temperature_C', 'Absorbance_Equilibre'], 'temp_ok', 'temp_err')
    current_temp_input_in_state = st.session_state.get('temp_effect_input') # Get current state

    if temp_df_validated is not None:
        new_temp_input = {'data': temp_df_validated, 'params': {'C0': C0_t, 'm': m_t, 'V': V_t}}
        needs_update = False
        if current_temp_input_in_state is None:
            needs_update = True
        elif isinstance(current_temp_input_in_state, dict):
            data_equal = temp_df_validated.equals(current_temp_input_in_state.get('data'))
            params_equal = (C0_t == current_temp_input_in_state.get('params', {}).get('C0') and
                            m_t == current_temp_input_in_state.get('params', {}).get('m') and
                            V_t == current_temp_input_in_state.get('params', {}).get('V'))
            if not data_equal or not params_equal:
                needs_update = True

        if needs_update:
            st.session_state['temp_effect_input'] = new_temp_input
            st.session_state['temp_effect_results'] = None
            st.session_state['thermo_params'] = None # Thermo depends on this

    elif current_temp_input_in_state is not None: # Input became invalid
        st.session_state['temp_effect_input'] = None
        st.session_state['temp_effect_results'] = None
        st.session_state['thermo_params'] = None

with st.sidebar.expander(_t("sidebar_expander_dosage"), expanded=False):
    st.write(_t("sidebar_dosage_fixed_conditions"))
    C0_dos = st.number_input("C0 (mg/L)", min_value=0.0, value=20.0, format="%.4f", key="C0_dos_input", help=_t("sidebar_kinetic_c0_help_const"))
    V_dos = st.number_input(_t("sidebar_isotherm_volume_label"), min_value=1e-6, value=0.05, format="%.4f", key="V_dos_input", help=_t("sidebar_kinetic_volume_help_const"))
    st.write(_t("sidebar_dosage_mass_abs_intro"))
    initial_dos_data = pd.DataFrame({'Masse_Adsorbant_g': [], 'Absorbance_Equilibre': []})
    edited_dos = st.data_editor(
        initial_dos_data.astype(float), num_rows="dynamic", key="dos_editor",
        column_config={
            "Masse_Adsorbant_g": st.column_config.NumberColumn(_t("col_mass_ads_g"), format="%.4f", required=True, min_value=1e-9, help=_t("col_mass_ads_g_help_variable")),
            "Absorbance_Equilibre": st.column_config.NumberColumn("Abs Eq.", format="%.4f", required=True, help=_t("col_abs_eq_help"))
        })

    # --- Logique de validation et mise à jour de l'état pour Dosage ---
    dos_df_validated = validate_data_editor(edited_dos, ['Masse_Adsorbant_g', 'Absorbance_Equilibre'], 'dos_ok', 'dos_err')
    current_dos_input_in_state = st.session_state.get('dosage_input') # Get current state

    if dos_df_validated is not None:
        new_dos_input = {'data': dos_df_validated, 'params': {'C0': C0_dos, 'V': V_dos}}
        needs_update = False
        if current_dos_input_in_state is None:
            needs_update = True
        elif isinstance(current_dos_input_in_state, dict):
            # Compare dataframes
            data_equal = dos_df_validated.equals(current_dos_input_in_state.get('data'))
            # Compare parameters
            current_params = current_dos_input_in_state.get('params', {})
            params_equal = (C0_dos == current_params.get('C0') and
                            V_dos == current_params.get('V'))
            if not data_equal or not params_equal:
                needs_update = True

        if needs_update:
            st.session_state['dosage_input'] = new_dos_input
            st.session_state['dosage_results'] = None # Clear results

    elif current_dos_input_in_state is not None: # Input became invalid
        st.session_state['dosage_input'] = None
        st.session_state['dosage_results'] = None


# --- CALIBRATION AUTOMATIQUE ---
new_calib_df = st.session_state.get('calib_df_input')
old_calib_df = st.session_state.get('previous_calib_df')

if new_calib_df is not None and len(new_calib_df) >= 2:
    if old_calib_df is None or not new_calib_df.equals(old_calib_df):
        try:
            slope, intercept, r_value, _, _ = linregress(new_calib_df['Concentration'], new_calib_df['Absorbance'])
            if abs(slope) > 1e-9:
                st.session_state['calibration_params'] = {'slope': slope, 'intercept': intercept, 'r_squared': r_value**2}
            else:
                st.session_state['calibration_params'] = None
                st.sidebar.warning(_t("calib_slope_near_zero_warning"), icon="⚠️")
        except Exception as e:
            st.session_state['calibration_params'] = None
            st.sidebar.error(_t("calib_error_calc_warning", error=e), icon="🔥")

        # Sauvegarde les données actuelles pour comparaison future
        st.session_state['previous_calib_df'] = new_calib_df.copy()

elif new_calib_df is None or len(new_calib_df) < 2:
    st.session_state['calibration_params'] = None
    st.session_state['previous_calib_df'] = None

# --- ZONE PRINCIPALE AVEC ONGLES ---
st.header(_t("main_results_header"))

tab_calib, tab_iso, tab_kin, tab_ph, tab_dosage, tab_temp, tab_thermo = st.tabs([
    _t("tab_calib"), _t("tab_isotherm"), _t("tab_kinetic"), _t("tab_ph"), 
    _t("tab_dosage"), _t("tab_temp"), _t("tab_thermo")
])
# --- Onglet 1: Étalonnage ---
with tab_calib:
    st.subheader(_t("calib_tab_subheader"))
    calib_params = st.session_state.get('calibration_params')
    calib_data = st.session_state.get('calib_df_input')

    if calib_data is not None and not calib_data.empty:
        if calib_params and len(calib_data) >= 2: # Ensure we have params and enough data
            col_plot, col_param = st.columns([2, 1]) # Keep the two-column layout

            with col_plot:
                # --- Create the Figure using Plotly Graph Objects for more control ---
                fig = go.Figure()

                # 1. Add Experimental Data Points (Scatter Trace)
                fig.add_trace(go.Scatter(
                    x=calib_data['Concentration'],
                    y=calib_data['Absorbance'],
                    mode='markers',
                    marker=dict(
                        color='blue',    # Blue markers
                        symbol='circle', # Circle shape
                        size=8           # Adjust size as needed
                    ),
                    name=_t("calib_tab_legend_exp") # Legend entry
                ))

                # 2. Add Linear Regression Line (Line Trace)
                try:
                    slope = calib_params['slope']
                    intercept = calib_params['intercept']
                    r_squared = calib_params['r_squared']

                    # Determine line range (slightly beyond data)
                    x_min_data = calib_data['Concentration'].min()
                    x_max_data = calib_data['Concentration'].max()
                    x_range_ext = (x_max_data - x_min_data) * 0.1 if x_max_data > x_min_data else 0.5
                    # Ensure line starts from 0 or slightly before min data point
                    x_start = max(0, x_min_data - x_range_ext)
                    x_end = x_max_data + x_range_ext

                    x_line = np.array([x_start, x_end])
                    y_line = slope * x_line + intercept

                    fig.add_trace(go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode='lines',
                        line=dict(
                            color='red', # Red line
                            width=1.5    # Adjust thickness
                        ),
                        name=_t("calib_tab_legend_reg") # Legend entry
                    ))

                    # 3. Add Equation Annotation
                    # Note: Using the calculated intercept sign, not forcing positive like the target image
                    equation_text = f"y = {slope:.4f}x {intercept:+.4f}"
                    fig.add_annotation(
                        # Position relative to data (adjust x,y as needed)
                        x=x_max_data * 0.95, # Position towards the right
                        y=y_line[-1] * 0.1 + intercept * 0.9 , # Position towards the bottom (heuristic)
                        text=equation_text,
                        showarrow=False,
                        font=dict(
                            family="Times New Roman, serif", # Try serif font
                            size=12,
                            color="red"
                        ),
                        align='right'
                    )

                    # 4. Configure Layout to match target style
                    y_max_data = calib_data['Absorbance'].max()
                    fig.update_layout(
                        title=_t("calib_tab_plot_title"), 
                        xaxis_title="Concentration (mg/L)", # French X Label (assuming mg/L)
                        yaxis_title="Absorbance (A)",      # French Y Label
                        plot_bgcolor='white',              # White background
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='LightGrey',
                            gridwidth=1,
                            zeroline=False, # Hide thicker zero line if not needed
                            range=[0, x_end * 1.05] # Start x-axis at 0
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='LightGrey',
                            gridwidth=1,
                            zeroline=False,
                            range=[0, y_max_data * 1.1] # Start y-axis at 0
                        ),
                        legend=dict(
                            x=0.02, # Position legend: 0 is left, 1 is right
                            y=0.98, # Position legend: 0 is bottom, 1 is top
                            traceorder='normal',
                            bgcolor='rgba(255,255,255,0.8)', # Semi-transparent white background
                            bordercolor='Black',
                            borderwidth=1
                        ),
                        font=dict(
                            family="Times New Roman, serif", # Apply serif font globally
                            size=12
                        ),
                        margin=dict(l=40, r=30, t=50, b=40) # Adjust margins
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    # Génération d'une figure Plotly stylisée pour téléchargement
                    fig_styled = go.Figure()

                    # Points expérimentaux – carrés noirs
                    fig_styled.add_trace(go.Scatter(
                        x=calib_data['Concentration'],
                        y=calib_data['Absorbance'],
                        mode='markers',
                        marker=dict(symbol='square', color='black', size=10),
                        name=_t("calib_tab_legend_exp")
                    ))

                    # Régression linéaire – ligne rouge épaisse
                    x_vals = np.array([calib_data['Concentration'].min(), calib_data['Concentration'].max()])
                    slope = st.session_state['calibration_params']['slope']
                    intercept = st.session_state['calibration_params']['intercept']
                    r_squared = st.session_state['calibration_params']['r_squared']
                    y_vals = slope * x_vals + intercept

                    fig_styled.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines',
                        line=dict(color='red', width=3),
                        name=_t("calib_tab_legend_reg")
                    ))

                    # Style global – imitation OriginLab
                    fig_styled.update_layout(
                        width=1000,
                        height=800,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family="Times New Roman", size=22, color="black"),
                        margin=dict(l=80, r=40, t=60, b=80),
                        xaxis=dict(
                            title="La concentration (mg/l)",
                            linecolor='black',
                            mirror=True,
                            ticks='outside',
                            showline=True,
                            showgrid=False
                        ),
                        yaxis=dict(
                            title="Absorbance (%)",
                            linecolor='black',
                            mirror=True,
                            ticks='outside',
                            showline=True,
                            showgrid=False
                        ),
                        showlegend=False
                    )

                    # Ajout de l'équation et R²
                    fig_styled.add_annotation(
                        xref="paper", yref="paper",
                        x=0.05, y=0.95,
                        text=f"y = {slope:.4f}x + {intercept:.4f}<br>R² = {r_squared:.4f}",
                        showarrow=False,
                        font=dict(size=20, color="black"),
                        align="left"
                    )

                    # Exporter au format PNG (version statique stylée)
                    img_buffer = io.BytesIO()
                    fig_styled.write_image(img_buffer, format="png", width=1000, height=800, scale=2)
                    img_buffer.seek(0)

                    # Bouton de téléchargement
                    st.download_button(
                        label=_t("calib_tab_download_styled_png_button"),
                        data=img_buffer,
                        file_name=_t("calib_tab_download_styled_png_filename"),
                        mime="image/png"
                    )


                except Exception as e_plot:
                    st.warning(_t("calib_plot_error_warning", e_plot=e_plot))

            with col_param:
                st.markdown(f"##### {_t('calib_tab_params_header')}")
                st.metric(_t("calib_tab_slope_metric"), f"{calib_params['slope']:.4f}")
                st.metric("Intercept (A)", f"{calib_params['intercept']:.4f}")
                st.metric(_t("calib_tab_r2_metric"), f"{calib_params['r_squared']:.4f}")
                # Display equation text separately as well, maybe clearer
                

        elif len(calib_data) < 2:
             st.warning(_t("calib_tab_min_2_points_warning"))
        else: # calib_params is None but data is present
             st.warning(_t("calib_tab_params_not_calc_warning"), icon="⚠️")
             # Optionally display the raw data plot without regression
             fig_raw = px.scatter(calib_data, x='Concentration', y='Absorbance',
                                 title=_t("calib_tab_params_not_calc_warning"),
                                 labels={'Concentration': 'Concentration (unités)', 'Absorbance': 'Absorbance (UA)'})
             fig_raw.update_layout(template="simple_white")
             st.plotly_chart(fig_raw, use_container_width=True)

    else:
        st.info(_t("calib_tab_enter_data_info"))


# --- Onglet 2: Isothermes (REFACTORED AND CORRECTED) ---
with tab_iso:
    st.subheader(_t("isotherm_tab_subheader"))
    iso_input = st.session_state.get('isotherm_input')
    calib_params = st.session_state.get('calibration_params')
    iso_results = st.session_state.get('isotherm_results')
    # References to non-linear params REMOVED
    # lang_params = st.session_state.get('langmuir_params') # REMOVED
    # freund_params = st.session_state.get('freundlich_params') # REMOVED

    if iso_input and calib_params:
        # Calculate Ce/qe specifically for this tab if not already done
        if iso_results is None:
            with st.spinner(_t("isotherm_spinner_ce_qe")):
                results_list = []
                df_iso = iso_input['data'].copy() # Use validated data
                params_iso = iso_input['params'] # Get fixed parameters
                try:
                    # Check if slope is valid before proceeding
                    if abs(calib_params['slope']) < 1e-9:
                         st.error(_t("isotherm_error_slope_zero"))
                         st.session_state['isotherm_results'] = None
                         raise ZeroDivisionError("Calibration slope is zero.")

                    for _, row in df_iso.iterrows():
                        c0 = row['Concentration_Initiale_C0']
                        abs_eq = row['Absorbance_Equilibre']
                        # Calculate Ce using calibration parameters
                        ce = (abs_eq - calib_params['intercept']) / calib_params['slope']
                        ce = max(0, ce) # Ensure non-negative concentration

                        # Calculate qe using the mass and volume for this experiment
                        m_adsorbant = params_iso['m']
                        volume = params_iso['V']
                        if m_adsorbant <= 0:
                            st.warning(_t("isotherm_error_mass_non_positive", m_adsorbant=m_adsorbant, c0=c0), icon="⚠️")
                            continue # Skip this data point
                        qe = (c0 - ce) * volume / m_adsorbant
                        qe = max(0, qe) # Ensure non-negative adsorption

                        results_list.append({
                            'C0': c0, 'Abs_Eq': abs_eq, 'Ce': ce, 'qe': qe,
                            'Masse_Adsorbant_g': m_adsorbant, 'Volume_L': volume,
                        })

                    if not results_list:
                         st.warning(_t("isotherm_warning_no_valid_points"))
                         st.session_state['isotherm_results'] = pd.DataFrame(columns=['C0', 'Abs_Eq', 'Ce', 'qe', 'Masse_Adsorbant_g', 'Volume_L'])
                    else:
                        st.session_state['isotherm_results'] = pd.DataFrame(results_list)
                        st.success(_t("isotherm_success_ce_qe_calc"))

                    iso_results = st.session_state['isotherm_results'] # Update local variable
                    # Reset linear parameter states since data changed
                    st.session_state['langmuir_params_lin'] = None
                    st.session_state['freundlich_params_lin'] = None

                except ZeroDivisionError:
                     if 'isotherm_results' not in st.session_state or st.session_state['isotherm_results'] is not None:
                         st.error(_t("isotherm_error_div_by_zero"))
                     st.session_state['isotherm_results'] = None
                except Exception as e:
                    st.error(_t("isotherm_error_ce_qe_calc_general", e=e))
                    st.session_state['isotherm_results'] = None


        # --- Display Calculated Data and Experimental Plot FIRST ---
        if iso_results is not None and not iso_results.empty:
            st.markdown(_t("isotherm_calculated_data_header"))
            st.dataframe(iso_results[['C0', 'Abs_Eq', 'Ce', 'qe']].style.format("{:.4f}"))
            csv_iso_res = convert_df_to_csv(iso_results)
            st.download_button(_t("isotherm_download_data_button"), csv_iso_res, _t("isotherm_download_data_filename"), "text/csv", key="dl_iso_res")
            st.caption(f"Conditions: m={iso_input['params']['m']}g, V={iso_input['params']['V']}L")
            st.markdown("---")

            
        # The rest of the isotherm tab (linearization plots etc.) follows here...
            st.markdown(_t("isotherm_exp_plot_header"))
            try:
                # Sort by Ce for a cleaner line plot
                
                iso_results_sorted = iso_results.sort_values(by='Ce')
                fig_exp = go.Figure()
                fig_exp.add_trace(go.Scatter(
                    x=iso_results_sorted['Ce'],
                    y=iso_results_sorted['qe'],
                    mode='lines+markers', # Connect points with lines
                    name=_t("isotherm_exp_plot_legend"),
                    line=dict(color='blue'),
                    marker=dict(size=8)
                ))
                fig_exp.update_layout(
                    xaxis_title="Ce (mg/L)",
                    yaxis_title="qe (mg/g)",
                    template="simple_white" # Cleaner look
                )
                st.plotly_chart(fig_exp, use_container_width=True)
                # Courbe expérimentale stylisée (qe vs Ce) pour téléchargement
                fig_exp_styled = go.Figure()

                # Points expérimentaux : carrés noirs
                iso_sorted = st.session_state['isotherm_results'].sort_values(by='Ce')
                fig_exp_styled.add_trace(go.Scatter(
                    x=iso_sorted['Ce'],
                    y=iso_sorted['qe'],
                    mode='markers+lines',
                    marker=dict(symbol='square', color='black', size=10),
                    line=dict(color='red', width=3),
                    name=_t("isotherm_exp_plot_legend")
                ))

                # Mise en forme style OriginLab
                fig_exp_styled.update_layout(
                    width=1000,
                    height=800,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Times New Roman", size=22, color="black"),
                    margin=dict(l=80, r=40, t=60, b=80),
                    xaxis=dict(
                        title="Ce (mg/L)",
                        linecolor='black',
                        mirror=True,
                        ticks='outside',
                        showline=True,
                        showgrid=False
                    ),
                    yaxis=dict(
                        title="qe (mg/g)",
                        linecolor='black',
                        mirror=True,
                        ticks='outside',
                        showline=True,
                        showgrid=False
                    ),
                    showlegend=False
                )

                # Sauvegarde de la figure en image haute qualité
                exp_img_buffer = io.BytesIO()
                fig_exp_styled.write_image(exp_img_buffer, format="png", width=1000, height=800, scale=2)
                exp_img_buffer.seek(0)

                # Bouton de téléchargement
                st.download_button(
                    label=_t("isotherm_download_exp_plot_button"),
                    data=exp_img_buffer,
                    file_name=_t("isotherm_download_exp_plot_filename"),
                    mime="image/png"
                )

            except Exception as e_exp_plot:
                 st.warning(_t("isotherm_exp_plot_error", e_exp_plot=e_exp_plot))
            st.markdown("---")
            
            st.markdown(_t("isotherm_linearization_header"))
            st.caption(_t("isotherm_linearization_caption"))

            # Filter data for fitting linear models (positive Ce and qe)
            # Define iso_filtered here, before the columns
            iso_filtered = iso_results[(iso_results['Ce'] > 1e-9) & (iso_results['qe'] > 1e-9)].copy()

            if len(iso_filtered) >= 2: # Need at least 2 points for linear regression
                
                st.markdown(_t("isotherm_langmuir_lin_header"))
                if not iso_filtered.empty:
                        try:
                            iso_filtered['inv_Ce'] = 1 / iso_filtered['Ce']
                            iso_filtered['inv_qe'] = 1 / iso_filtered['qe']

                            # Check for sufficient variation before linregress
                            if iso_filtered['inv_Ce'].nunique() < 2 or iso_filtered['inv_qe'].nunique() < 2:
                                st.warning(_t("isotherm_insufficient_variation_warning", var1="1/Ce", var2="1/qe", model="Langmuir"))
                                raise ValueError("Insufficient variation for linregress")

                            slope_L_lin, intercept_L_lin, r_val_L_lin, p_val_L_lin, std_err_L_lin = linregress(iso_filtered['inv_Ce'], iso_filtered['inv_qe'])
                            r2_L_lin = r_val_L_lin**2

                            # Calculate params from linear fit
                            qm_L_lin = 1 / intercept_L_lin if abs(intercept_L_lin) > 1e-12 else np.nan # Increased precision check
                            KL_L_lin = intercept_L_lin / slope_L_lin if abs(slope_L_lin) > 1e-12 and abs(intercept_L_lin) > 1e-12 else np.nan

                            # Store params in session state
                            st.session_state['langmuir_params_lin'] = {'qm': qm_L_lin, 'KL': KL_L_lin, 'r_squared': r2_L_lin}

                            fig_L_lin = px.scatter(iso_filtered, x='inv_Ce', y='inv_qe',
                                                 title=f"1/qe vs 1/Ce (R²={r2_L_lin:.4f})",
                                                 labels={'inv_Ce': '1 / Ce (L/mg)', 'inv_qe': '1 / qe (g/mg)'})

                            x_min_L_lin = iso_filtered['inv_Ce'].min()
                            x_max_L_lin = iso_filtered['inv_Ce'].max()
                            x_range_L_lin = x_max_L_lin - x_min_L_lin if x_max_L_lin > x_min_L_lin else 1.0
                            x_line_L_lin = np.linspace(x_min_L_lin - 0.1 * x_range_L_lin, x_max_L_lin + 0.1 * x_range_L_lin, 100)
                            x_line_L_lin = np.maximum(x_line_L_lin, 0 if x_min_L_lin >= 0 else x_line_L_lin[0])
                            y_line_L_lin = intercept_L_lin + slope_L_lin * x_line_L_lin
                            fig_L_lin.add_trace(go.Scatter(x=x_line_L_lin, y=y_line_L_lin, mode='lines', name=_t("isotherm_lin_plot_legend_fit")))

                            fig_L_lin.update_layout(template="simple_white")
                            st.plotly_chart(fig_L_lin, use_container_width=True)
                            st.caption(_t("isotherm_langmuir_lin_caption", slope_L_lin=slope_L_lin, intercept_L_lin=intercept_L_lin))
                            # --- Création figure stylisée pour Langmuir linéarisé ---
# --- Figure stylisée pour téléchargement : 1/qe vs 1/Ce ---
                            try:
                                x_vals = np.linspace(iso_filtered['inv_Ce'].min(), iso_filtered['inv_Ce'].max(), 100)
                                y_vals = intercept_L_lin + slope_L_lin * x_vals

                                fig_lang_1_over = go.Figure()

                                # Données expérimentales : carrés noirs
                                fig_lang_1_over.add_trace(go.Scatter(
                                    x=iso_filtered['inv_Ce'],
                                    y=iso_filtered['inv_qe'],
                                    mode='markers',
                                    marker=dict(symbol='square', color='black', size=10),
                                    name=_t("isotherm_exp_plot_legend")
                                ))

                                # Ligne de régression : rouge épaisse
                                fig_lang_1_over.add_trace(go.Scatter(
                                    x=x_vals,
                                    y=y_vals,
                                    mode='lines',
                                    line=dict(color='red', width=3),
                                    name=_t("calib_tab_legend_reg")
                                ))

                                # Mise en forme style OriginLab
                                fig_lang_1_over.update_layout(
                                    width=1000,
                                    height=800,
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    font=dict(family="Times New Roman", size=22, color="black"),
                                    margin=dict(l=80, r=40, t=60, b=80),
                                    xaxis=dict(
                                        title="1 / Ce (L/mg)",
                                        linecolor='black',
                                        mirror=True,
                                        ticks='outside',
                                        showline=True,
                                        showgrid=False
                                    ),
                                    yaxis=dict(
                                        title="1 / qe (g/mg)",
                                        linecolor='black',
                                        mirror=True,
                                        ticks='outside',
                                        showline=True,
                                        showgrid=False
                                    ),
                                    showlegend=False
                                )

                                # Annotation équation + R²
                                fig_lang_1_over.add_annotation(
                                    xref="paper", yref="paper",
                                    x=0.05, y=0.95,
                                    text=f"y = {slope_L_lin:.4f}x + {intercept_L_lin:.4f}<br>R² = {r2_L_lin:.4f}",
                                    showarrow=False,
                                    font=dict(size=20, color="black"),
                                    align="left"
                                )

                                # Export PNG
                                img_buffer_lang_1_over = io.BytesIO()
                                fig_lang_1_over.write_image(img_buffer_lang_1_over, format="png", width=1000, height=800, scale=2)
                                img_buffer_lang_1_over.seek(0)

                                st.download_button(
                                    label=_t("isotherm_download_langmuir_lin_inv_button"),
                                    data=img_buffer_lang_1_over,
                                    file_name=_t("isotherm_download_langmuir_lin_inv_filename"),
                                    mime="image/png"
                                )
                            except Exception as e:
                                st.warning(_t("isotherm_error_export_langmuir_lin_inv", e=e))



                        except ValueError as ve: # Catch variation error specifically
                             if "Insufficient variation" not in str(ve): # Avoid duplicating message
                                st.warning(_t("isotherm_error_langmuir_lin_regression", ve=ve))
                             st.session_state['langmuir_params_lin'] = None # Clear params on error
                        except Exception as e_lin_L:
                            st.warning(_t("isotherm_error_langmuir_lin_plot_creation", e_lin_L=e_lin_L))
                            st.session_state['langmuir_params_lin'] = None # Clear params on error
                else:
                        st.info(_t("isotherm_no_valid_data_langmuir_lin"))
                    # Check iso_filtered is not empty here
                        
                st.markdown("---")
                # --- Linearized Freundlich (log qe vs log Ce) ---
                
                st.markdown(_t("isotherm_freundlich_lin_header"))
                if not iso_filtered.empty:
                        try:
                            # Use np.log10 for base-10 logarithm
                            # Add checks for potential issues with log
                            if (iso_filtered['Ce'] <= 0).any() or (iso_filtered['qe'] <= 0).any():
                                st.warning(_t("isotherm_warning_filter_log_freundlich"))
                            iso_filtered_log = iso_filtered[(iso_filtered['Ce'] > 0) & (iso_filtered['qe'] > 0)].copy()

                            if iso_filtered_log.empty:
                                st.info(_t("isotherm_info_no_valid_data_freundlich_lin"))
                                raise ValueError("No valid data for log.")

                            iso_filtered_log['log_Ce'] = np.log10(iso_filtered_log['Ce'])
                            iso_filtered_log['log_qe'] = np.log10(iso_filtered_log['qe'])

                            # Check for sufficient variation
                            if iso_filtered_log['log_Ce'].nunique() < 2 or iso_filtered_log['log_qe'].nunique() < 2:
                                st.warning(_t("isotherm_insufficient_variation_warning", var1="log(Ce)", var2="log(qe)", model="Freundlich"))
                                raise ValueError("Insufficient variation for linregress")

                            slope_F_lin, intercept_F_lin, r_val_F_lin, p_val_F_lin, std_err_F_lin = linregress(iso_filtered_log['log_Ce'], iso_filtered_log['log_qe'])
                            r2_F_lin = r_val_F_lin**2

                            # Calculate params from linear fit
                            n_F_lin = 1 / slope_F_lin if abs(slope_F_lin) > 1e-12 else np.nan
                            KF_F_lin = 10**intercept_F_lin # Since we used log10

                            # Store params in session state
                            st.session_state['freundlich_params_lin'] = {'KF': KF_F_lin, 'n': n_F_lin, 'r_squared': r2_F_lin}

                            fig_F_lin = px.scatter(iso_filtered_log, x='log_Ce', y='log_qe',
                                                 title=f"log(qe) vs log(Ce) (R²={r2_F_lin:.4f})",
                                                 labels={'log_Ce': 'log₁₀(Ce)', 'log_qe': 'log₁₀(qe)'})

                            x_min_F_lin = iso_filtered_log['log_Ce'].min()
                            x_max_F_lin = iso_filtered_log['log_Ce'].max()
                            x_range_F_lin = x_max_F_lin - x_min_F_lin if x_max_F_lin > x_min_F_lin else 1.0
                            x_line_F_lin = np.linspace(x_min_F_lin - 0.1 * abs(x_range_F_lin) - 0.01, x_max_F_lin + 0.1 * abs(x_range_F_lin) + 0.01, 100)
                            y_line_F_lin = intercept_F_lin + slope_F_lin * x_line_F_lin
                            fig_F_lin.add_trace(go.Scatter(x=x_line_F_lin, y=y_line_F_lin, mode='lines', name=_t("isotherm_lin_plot_legend_fit")))

                            fig_F_lin.update_layout(template="simple_white")
                            st.plotly_chart(fig_F_lin, use_container_width=True)
                            st.caption(_t("isotherm_freundlich_lin_caption", slope_F_lin=slope_F_lin, intercept_F_lin=intercept_F_lin))
                            # --- Figure stylisée pour Freundlich linéarisé (log(qe) vs log(Ce)) ---
                            try:
                                x_vals = np.linspace(iso_filtered_log['log_Ce'].min(), iso_filtered_log['log_Ce'].max(), 100)
                                y_vals = intercept_F_lin + slope_F_lin * x_vals

                                fig_freund_lin = go.Figure()

                                # Données expérimentales : carrés noirs
                                fig_freund_lin.add_trace(go.Scatter(
                                    x=iso_filtered_log['log_Ce'],
                                    y=iso_filtered_log['log_qe'],
                                    mode='markers',
                                    marker=dict(symbol='square', color='black', size=10),
                                    name=_t("isotherm_exp_plot_legend")
                                ))

                                # Ligne de régression : rouge épaisse
                                fig_freund_lin.add_trace(go.Scatter(
                                    x=x_vals,
                                    y=y_vals,
                                    mode='lines',
                                    line=dict(color='red', width=3),
                                    name=_t("calib_tab_legend_reg")
                                ))

                                # Mise en forme style OriginLab
                                fig_freund_lin.update_layout(
                                    width=1000,
                                    height=800,
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    font=dict(family="Times New Roman", size=22, color="black"),
                                    margin=dict(l=80, r=40, t=60, b=80),
                                    xaxis=dict(
                                        title="log₁₀(Ce)",
                                        linecolor='black',
                                        mirror=True,
                                        ticks='outside',
                                        showline=True,
                                        showgrid=False
                                    ),
                                    yaxis=dict(
                                        title="log₁₀(qe)",
                                        linecolor='black',
                                        mirror=True,
                                        ticks='outside',
                                        showline=True,
                                        showgrid=False
                                    ),
                                    showlegend=False
                                )

                                # Annotation équation + R²
                                fig_freund_lin.add_annotation(
                                    xref="paper", yref="paper",
                                    x=0.05, y=0.95,
                                    text=f"y = {slope_F_lin:.4f}x + {intercept_F_lin:.4f}<br>R² = {r2_F_lin:.4f}",
                                    showarrow=False,
                                    font=dict(size=20, color="black"),
                                    align="left"
                                )

                                # Export PNG
                                freund_img_buffer = io.BytesIO()
                                fig_freund_lin.write_image(freund_img_buffer, format="png", width=1000, height=800, scale=2)
                                freund_img_buffer.seek(0)

                                # Bouton de téléchargement
                                st.download_button(
                                    label=_t("isotherm_download_freundlich_lin_button"),
                                    data=freund_img_buffer,
                                    file_name=_t("isotherm_download_freundlich_lin_filename"),
                                    mime="image/png"
                                )

                            except Exception as e:
                                st.warning(_t("isotherm_error_export_freundlich_lin", e=e))


                        except ValueError as ve:
                             if "Insufficient variation" not in str(ve) and "No valid data for log" not in str(ve):
                                st.warning(_t("isotherm_error_freundlich_lin_regression", ve=ve))
                             st.session_state['freundlich_params_lin'] = None # Clear params on error
                        except Exception as e_lin_F:
                            st.warning(_t("isotherm_error_freundlich_lin_plot_creation", e_lin_F=e_lin_F))
                            st.session_state['freundlich_params_lin'] = None # Clear params on error
                    # else: Handled by empty check earlier    

                # --- Display Parameters from Linear Fits --- ADDED BACK ---
                st.markdown("---")
                st.markdown(_t("isotherm_derived_params_header"))

                params_data = {'Modèle': [], 'Paramètre': [], 'Valeur': [], 'R² (Linéarisé)': []}
                # Retrieve calculated linear params from session state
                params_L = st.session_state.get('langmuir_params_lin')
                params_F = st.session_state.get('freundlich_params_lin')

                if params_L and not np.isnan(params_L.get('qm', np.nan)): # Check if calculation was successful
                    params_data['Modèle'].extend(['Langmuir', 'Langmuir'])
                    params_data['Paramètre'].extend(['qm (mg/g)', 'KL (L/mg)'])
                    params_data['Valeur'].extend([f"{params_L.get('qm', np.nan):.4f}", f"{params_L.get('KL', np.nan):.4f}"])
                    params_data['R² (Linéarisé)'].extend([f"{params_L.get('r_squared', np.nan):.4f}"] * 2)

                if params_F and not np.isnan(params_F.get('KF', np.nan)): # Check if calculation was successful
                    params_data['Modèle'].extend(['Freundlich', 'Freundlich'])
                    params_data['Paramètre'].extend(['KF ((mg/g)(L/mg)¹/ⁿ)', 'n'])
                    params_data['Valeur'].extend([f"{params_F.get('KF', np.nan):.4f}", f"{params_F.get('n', np.nan):.4f}"])
                    params_data['R² (Linéarisé)'].extend([f"{params_F.get('r_squared', np.nan):.4f}"] * 2)

                if params_data['Modèle']:
                    params_df = pd.DataFrame(params_data)
                    st.dataframe(params_df.set_index('Modèle'), use_container_width=True)
                else:
                    st.info(_t("isotherm_info_params_not_calculated"))

            elif not iso_results.empty: # Have results, but < 2 points with Ce>0, qe>0
                 st.warning(_t("isotherm_warning_less_than_2_points_lin_fit"))
            # else: iso_filtered was empty, handled within linear plots

        elif iso_results is not None and iso_results.empty:
             st.warning(_t("isotherm_warning_ce_qe_no_results"))
        # else: Handled by spinner or error message during calculation

    elif not calib_params:
        st.warning(_t("isotherm_warning_provide_calib_data"))
    else: # iso_input is None
        st.info(_t("isotherm_info_enter_isotherm_data"))

# --- Onglet 3: Cinétique  ---

with tab_kin:
    st.header(_t("kinetic_tab_main_header"))
    kinetic_input = st.session_state.get('kinetic_input')
    calib_params = st.session_state.get('calibration_params')
    kinetic_results = st.session_state.get('kinetic_results_df')
    # Retrieve potentially calculated non-linear params (needed for PFO linear)
    pfo_params_nl = st.session_state.get('pfo_params_nonlinear')
    pso_params_nl = st.session_state.get('pso_params_nonlinear') # Calculation kept, but plot/results hidden
    ipd_params_list = st.session_state.get('ipd_params_list', [])

    if kinetic_input and calib_params:
        # --- Calculate qt if not already done ---
        if kinetic_results is None:
            with st.spinner(_t("kinetic_spinner_qt_calc")):
                df_kin = kinetic_input['data'].copy()
                params_kin = kinetic_input['params']
                C0_k, V_k, m_k = params_kin['C0'], params_kin['V'], params_kin['m']

                if 'Absorbance_t' in df_kin.columns and m_k > 0 and V_k > 0:
                    try:
                        if abs(calib_params['slope']) < 1e-9:
                            st.error(_t("kinetic_error_qt_calc_slope_zero"))
                            st.session_state['kinetic_results_df'] = None
                            raise ZeroDivisionError("Calibration slope is zero.")

                        df_kin['Ct'] = (df_kin['Absorbance_t'] - calib_params['intercept']) / calib_params['slope']
                        df_kin['Ct'] = df_kin['Ct'].clip(lower=0)
                        df_kin['qt'] = (C0_k - df_kin['Ct']) * V_k / m_k
                        df_kin['qt'] = df_kin['qt'].clip(lower=0)
                        df_kin['sqrt_t'] = np.sqrt(df_kin['Temps_min'])

                        st.session_state['kinetic_results_df'] = df_kin[['Temps_min', 'Absorbance_t', 'Ct', 'qt', 'sqrt_t']].copy()
                        kinetic_results = st.session_state['kinetic_results_df']
                        # Reset params as data is recalculated
                        st.session_state['pfo_params_nonlinear'] = None
                        st.session_state['pso_params_nonlinear'] = None
                        st.session_state['ipd_params_list'] = []
                        pfo_params_nl = None
                        pso_params_nl = None
                        ipd_params_list = []
                        st.success(_t("kinetic_success_qt_calc"))
                    except ZeroDivisionError:
                        if 'kinetic_results_df' not in st.session_state or st.session_state['kinetic_results_df'] is not None:
                            st.error(_t("kinetic_error_qt_calc_div_by_zero"))
                        st.session_state['kinetic_results_df'] = None
                    except Exception as e_qt:
                        st.error(_t("kinetic_error_qt_calc_general", e_qt=e_qt))
                        st.session_state['kinetic_results_df'] = None
                elif not ('Absorbance_t' in df_kin.columns):
                     st.error(_t("kinetic_error_missing_abs_t_col"))
                     st.session_state['kinetic_results_df'] = None
                else:
                     st.error(_t("kinetic_error_mass_volume_non_positive"))
                     st.session_state['kinetic_results_df'] = None


        # --- Display results if available ---
        if kinetic_results is not None and not kinetic_results.empty:
            st.subheader(_t("kinetic_calculated_data_subheader"))
            st.dataframe(kinetic_results.style.format(precision=4))
            csv_kin_res = convert_df_to_csv(kinetic_results)
            st.download_button(_t("kinetic_download_data_button"), csv_kin_res, _t("kinetic_download_data_filename"), "text/csv", key='dl_kin_res_tab_kin')
            st.caption(f"Conditions: C0={kinetic_input['params']['C0']}mg/L, m={kinetic_input['params']['m']}g, V={kinetic_input['params']['V']}L")
            st.markdown("---")

            # --- 1. Plot qt vs t ---
            st.subheader(_t("kinetic_plot_qt_vs_t_subheader"))
            try:
                fig_qt_vs_t = px.scatter(kinetic_results, x='Temps_min', y='qt',
                                         title=_t("kinetic_plot_qt_vs_t_title"),
                                         labels={'Temps_min': _t("kinetic_plot_qt_vs_t_xaxis"), 'qt': 'qt (mg/g)'})
                fig_qt_vs_t.add_trace(go.Scatter(x=kinetic_results['Temps_min'], y=kinetic_results['qt'], mode='lines', name=_t("kinetic_plot_qt_vs_t_legend"), showlegend=False)) # Add connecting line
                fig_qt_vs_t.update_layout(template="simple_white")
                st.plotly_chart(fig_qt_vs_t, use_container_width=True)
                # --- Figure stylisée pour Effet du Temps de Contact (qt vs t) ---
                try:
                    df_qt = kinetic_results.copy()

                    fig_qt_styled = go.Figure()

                    # Données expérimentales : carrés noirs
                    fig_qt_styled.add_trace(go.Scatter(
                        x=df_qt['Temps_min'],
                        y=df_qt['qt'],
                        mode='markers+lines',
                        marker=dict(symbol='square', color='black', size=10),
                        line=dict(color='red', width=3),
                        name=_t("isotherm_exp_plot_legend")
                    ))

                    # Mise en forme style OriginLab
                    fig_qt_styled.update_layout(
                        width=1000,
                        height=800,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family="Times New Roman", size=22, color="black"),
                        margin=dict(l=80, r=40, t=60, b=80),
                        xaxis=dict(
                            title=_t("kinetic_plot_qt_vs_t_xaxis"),
                            linecolor='black',
                            mirror=True,
                            ticks='outside',
                            showline=True,
                            showgrid=False
                        ),
                        yaxis=dict(
                            title="qt (mg/g)",
                            linecolor='black',
                            mirror=True,
                            ticks='outside',
                            showline=True,
                            showgrid=False
                        ),
                        showlegend=False
                    )

                    # Export PNG
                    qt_img_buffer = io.BytesIO()
                    fig_qt_styled.write_image(qt_img_buffer, format="png", width=1000, height=800, scale=2)
                    qt_img_buffer.seek(0)

                    # Bouton de téléchargement
                    st.download_button(
                        label=_t("kinetic_download_qt_vs_t_button"),
                        data=qt_img_buffer,
                        file_name=_t("kinetic_download_qt_vs_t_filename"),
                        mime="image/png"
                    )
                except Exception as e:
                    st.warning(_t("kinetic_error_export_qt_vs_t", e=e))

            except Exception as e_qt_plot:
                st.warning(_t("kinetic_error_plot_qt_vs_t", e_qt_plot=e_qt_plot))
            st.markdown("---")


            # --- Calculate Kinetic Parameters (needed for plots) ---
            if len(kinetic_results) >= 3:
                t_data = kinetic_results['Temps_min'].values
                qt_data = kinetic_results['qt'].values
                sqrt_t_data = kinetic_results['sqrt_t'].values
                qe_exp = qt_data[-1] if len(qt_data) > 0 else np.nan

                # --- Perform PFO non-linear fit (needed for qe in linearized plot) ---
                if pfo_params_nl is None:
                     with st.spinner(_t("kinetic_spinner_pfo_nl_calc")):
                         try:
                             p0_PFO = [qe_exp if not np.isnan(qe_exp) and qe_exp > 1e-6 else 1.0, 0.01]
                             params_PFO, _ = curve_fit(pfo_model, t_data, qt_data, p0=p0_PFO, maxfev=5000, bounds=([0, 0], [np.inf, np.inf]))
                             qe_PFO_nl, k1_PFO_nl = params_PFO
                             # Calculate R2 for non-linear PFO fit (optional, not displayed by default now)
                             qt_pred_PFO = pfo_model(t_data, qe_PFO_nl, k1_PFO_nl)
                             mean_qt = np.mean(qt_data)
                             ss_tot = np.sum((qt_data - mean_qt)**2)
                             ss_res = np.sum((qt_data - qt_pred_PFO)**2)
                             r2_PFO_nl = 1 - (ss_res / ss_tot) if ss_tot > 1e-9 else (1.0 if ss_res < 1e-9 else 0.0)
                             st.session_state['pfo_params_nonlinear'] = {'qe_PFO_nl': qe_PFO_nl, 'k1_nl': k1_PFO_nl, 'R2_PFO_nl': r2_PFO_nl}
                             pfo_params_nl = st.session_state['pfo_params_nonlinear']
                         except Exception as e:
                             st.warning(_t("kinetic_warning_pfo_nl_calc_failed", e=e))
                             st.session_state['pfo_params_nonlinear'] = None

                # --- Perform PSO non-linear fit (calculation kept, but results/plot hidden) ---
                if pso_params_nl is None:
                     with st.spinner(_t("kinetic_spinner_pso_nl_calc")):
                         try:
                             qe_guess_pso = qe_exp if not np.isnan(qe_exp) and qe_exp > 1e-6 else 1.0
                             k2_guess = 0.01 / qe_guess_pso if qe_guess_pso > 1e-6 else 0.01
                             p0_PSO = [qe_guess_pso, k2_guess]
                             params_PSO, _ = curve_fit(pso_model, t_data, qt_data, p0=p0_PSO, maxfev=5000, bounds=([0, 0], [np.inf, np.inf]))
                             qe_PSO_nl, k2_PSO_nl = params_PSO
                             # Calculate R2 for non-linear PSO fit (optional, not displayed by default now)
                             qt_pred_PSO = pso_model(t_data, qe_PSO_nl, k2_PSO_nl)
                             mean_qt = np.mean(qt_data)
                             ss_tot = np.sum((qt_data - mean_qt)**2)
                             ss_res = np.sum((qt_data - qt_pred_PSO)**2)
                             r2_PSO_nl = 1 - (ss_res / ss_tot) if ss_tot > 1e-9 else (1.0 if ss_res < 1e-9 else 0.0)
                             st.session_state['pso_params_nonlinear'] = {'qe_PSO_nl': qe_PSO_nl, 'k2_nl': k2_PSO_nl, 'R2_PSO_nl': r2_PSO_nl}
                             pso_params_nl = st.session_state['pso_params_nonlinear']
                         except Exception as e:
                             st.warning(f"Calcul paramètres PSO (non-linéaire) échoué: {e}")
                             st.session_state['pso_params_nonlinear'] = None

                # --- IPD Fit ---
                if not ipd_params_list:
                     with st.spinner(_t("kinetic_spinner_ipd_analysis")):
                         ipd_df = kinetic_results[kinetic_results['Temps_min'] > 1e-9].copy()
                         if len(ipd_df) >= 2:
                             try:
                                 if ipd_df['sqrt_t'].nunique() < 2 or ipd_df['qt'].nunique() < 2:
                                     st.warning(_t("kinetic_warning_ipd_insufficient_variation"))
                                     slope_ipd, intercept_ipd, r_val_ipd = np.nan, np.nan, np.nan
                                 else:
                                     slope_ipd, intercept_ipd, r_val_ipd, _, _ = linregress(ipd_df['sqrt_t'], ipd_df['qt'])

                                 if not np.isnan(r_val_ipd):
                                     st.session_state['ipd_params_list'] = [{'k_id': slope_ipd, 'C_ipd': intercept_ipd, 'R2_IPD': r_val_ipd**2, 'stage': 'Global'}]
                                     ipd_params_list = st.session_state['ipd_params_list']

                                 else:
                                     st.session_state['ipd_params_list'] = []
                             except ValueError as ve:
                                st.warning(_t("kinetic_warning_ipd_calc_failed_regression", ve=ve))
                                st.session_state['ipd_params_list'] = []
                             except Exception as e:
                                 st.warning(_t("kinetic_warning_ipd_calc_failed_general", e=e))
                                 st.session_state['ipd_params_list'] = []
                         else:
                             st.warning(_t("kinetic_warning_ipd_not_enough_points"))
                             st.session_state['ipd_params_list'] = []


                # --- Display Linearized and IPD Plots ---
                st.subheader(_t("kinetic_linearized_models_subheader"))
                

                # --- 2. PFO Linearized ---
                
                st.markdown(_t("kinetic_pfo_lin_header"))
                    # Requires qe from the non-linear fit
                if pfo_params_nl:
                        qe_calc_pfo = pfo_params_nl['qe_PFO_nl']
                        # Filter data: qt must be less than qe_calc_pfo for log, and t>0 often preferred
                        df_pfo_lin = kinetic_results[(kinetic_results['qt'] < qe_calc_pfo - 1e-9) & (kinetic_results['qt'] >= 0) & (kinetic_results['Temps_min'] > 1e-9)].copy()
                        if len(df_pfo_lin) >= 2:
                            try:
                                df_pfo_lin['ln_qe_qt'] = np.log(qe_calc_pfo - df_pfo_lin['qt'])
                                t_pfo_lin = df_pfo_lin['Temps_min']
                                y_pfo_lin = df_pfo_lin['ln_qe_qt']

                                if t_pfo_lin.nunique() < 2 or y_pfo_lin.nunique() < 2 :
                                     st.warning(_t("kinetic_warning_pfo_lin_insufficient_variation"))
                                     raise ValueError("Insufficient variation for PFO linregress")

                                slope_pfo_lin, intercept_pfo_lin, r_val_pfo, _, _ = linregress(t_pfo_lin, y_pfo_lin)
                                r2_pfo_lin = r_val_pfo**2
                                k1_lin = -slope_pfo_lin # k1 from linear plot

                                fig_pfo_lin = px.scatter(df_pfo_lin, x='Temps_min', y='ln_qe_qt', title=_t("kinetic_pfo_lin_plot_title", r2_pfo_lin=r2_pfo_lin), labels={'Temps_min': _t("kinetic_plot_qt_vs_t_xaxis"), 'ln_qe_qt': 'ln(qe - qt)'})
                                # Extend line slightly beyond data range for clarity
                                t_min_plot, t_max_plot = t_pfo_lin.min(), t_pfo_lin.max()
                                t_range_plot = max(1.0, t_max_plot - t_min_plot) # Avoid zero range
                                t_line_lin = np.linspace(t_min_plot - 0.05 * t_range_plot, t_max_plot + 0.05 * t_range_plot, 50)
                                t_line_lin = np.maximum(0, t_line_lin) # Ensure time >= 0
                                y_line_lin = intercept_pfo_lin + slope_pfo_lin * t_line_lin
                                fig_pfo_lin.add_trace(go.Scatter(x=t_line_lin, y=y_line_lin, mode='lines', name=_t("isotherm_lin_plot_legend_fit")))
                                fig_pfo_lin.update_layout(template="simple_white")
                                st.plotly_chart(fig_pfo_lin, use_container_width=True)
                                st.caption(_t("kinetic_pfo_lin_caption", slope_pfo_lin=slope_pfo_lin, intercept_pfo_lin=intercept_pfo_lin, r2_pfo_lin=r2_pfo_lin, k1_lin=k1_lin))
                                # --- Figure stylisée pour PFO Linéarisé (ln(qe - qt) vs t) ---
                                try:
                                    x_vals = np.linspace(t_pfo_lin.min(), t_pfo_lin.max(), 100)
                                    y_vals = intercept_pfo_lin + slope_pfo_lin * x_vals

                                    fig_pfo_styled = go.Figure()

                                    # Données expérimentales : carrés noirs
                                    fig_pfo_styled.add_trace(go.Scatter(
                                        x=t_pfo_lin,
                                        y=y_pfo_lin,
                                        mode='markers',
                                        marker=dict(symbol='square', color='black', size=10),
                                        name=_t("isotherm_exp_plot_legend")
                                    ))

                                    # Ligne de régression : rouge épaisse
                                    fig_pfo_styled.add_trace(go.Scatter(
                                        x=x_vals,
                                        y=y_vals,
                                        mode='lines',
                                        line=dict(color='red', width=3),
                                        name=_t("calib_tab_legend_reg")
                                    ))

                                    # Mise en forme style OriginLab
                                    fig_pfo_styled.update_layout(
                                        width=1000,
                                        height=800,
                                        plot_bgcolor='white',
                                        paper_bgcolor='white',
                                        font=dict(family="Times New Roman", size=22, color="black"),
                                        margin=dict(l=80, r=40, t=60, b=80),
                                        xaxis=dict(
                                            title=_t("kinetic_plot_qt_vs_t_xaxis"),
                                            linecolor='black',
                                            mirror=True,
                                            ticks='outside',
                                            showline=True,
                                            showgrid=False
                                        ),
                                        yaxis=dict(
                                            title="ln(qe - qt)",
                                            linecolor='black',
                                            mirror=True,
                                            ticks='outside',
                                            showline=True,
                                            showgrid=False
                                        ),
                                        showlegend=False
                                    )

                                    # Annotation équation + R²
                                    fig_pfo_styled.add_annotation(
                                        xref="paper", yref="paper",
                                        x=0.05, y=0.95,
                                        text=f"y = {slope_pfo_lin:.4f}x + {intercept_pfo_lin:.4f}<br>R² = {r2_pfo_lin:.4f}",
                                        showarrow=False,
                                        font=dict(size=20, color="black"),
                                        align="left"
                                    )

                                    # Export PNG
                                    pfo_img_buffer = io.BytesIO()
                                    fig_pfo_styled.write_image(pfo_img_buffer, format="png", width=1000, height=800, scale=2)
                                    pfo_img_buffer.seek(0)

                                    # Bouton de téléchargement
                                    st.download_button(
                                        label=_t("kinetic_download_pfo_lin_button"),
                                        data=pfo_img_buffer,
                                        file_name=_t("kinetic_download_pfo_lin_filename"),
                                        mime="image/png"
                                    )
                                except Exception as e:
                                    st.warning(_t("kinetic_error_export_pfo_lin", e=e))


                            except ValueError as ve:
                                if "Insufficient variation" not in str(ve): st.warning(_t("kinetic_error_pfo_lin_plot_regression", ve=ve))
                            except Exception as e_pfo_lin:
                                st.warning(_t("kinetic_error_pfo_lin_plot_general", e_pfo_lin=e_pfo_lin))
                        else:
                            st.warning(_t("kinetic_warning_pfo_lin_not_enough_points"))
                else:
                        st.warning(_t("kinetic_warning_pfo_nl_calc_required"))
                        st.info(_t("kinetic_info_pfo_lin_uses_nl_qe"))
                st.markdown("---")
                # --- 3. PSO Linearized ---
                
                st.markdown(_t("kinetic_pso_lin_header"))
                    # Filter data: t > 0 and qt > 0 for division
                df_pso_lin = kinetic_results[(kinetic_results['Temps_min'] > 1e-9) & (kinetic_results['qt'] > 1e-9)].copy()
                if len(df_pso_lin) >= 2:
                        try:
                            df_pso_lin['t_div_qt'] = df_pso_lin['Temps_min'] / df_pso_lin['qt']
                            t_pso_lin = df_pso_lin['Temps_min']
                            y_pso_lin = df_pso_lin['t_div_qt']

                            if t_pso_lin.nunique() < 2 or y_pso_lin.nunique() < 2 :
                                st.warning(_t("kinetic_warning_pso_lin_insufficient_variation"))
                                raise ValueError("Insufficient variation for PSO linregress")

                            slope_pso_lin, intercept_pso_lin, r_val_pso, _, _ = linregress(t_pso_lin, y_pso_lin)
                            r2_pso_lin = r_val_pso**2

                            # Calculate parameters from the linear fit
                            qe_lin = 1 / slope_pso_lin if abs(slope_pso_lin) > 1e-12 else np.nan
                            k2_lin = slope_pso_lin**2 / intercept_pso_lin if abs(intercept_pso_lin) > 1e-12 and not np.isnan(qe_lin) else np.nan # Correct formula: k2 = slope^2 / intercept = (1/qe)^2 / (1/(k2*qe^2)) = k2

                            fig_pso_lin = px.scatter(df_pso_lin, x='Temps_min', y='t_div_qt', title=_t("kinetic_pso_lin_plot_title", r2_pso_lin=r2_pso_lin), labels={'Temps_min': _t("kinetic_plot_qt_vs_t_xaxis"), 't_div_qt': 't / qt (min·g/mg)'})
                            # Extend line slightly beyond data range
                            t_min_plot, t_max_plot = t_pso_lin.min(), t_pso_lin.max()
                            t_range_plot = max(1.0, t_max_plot - t_min_plot)
                            t_line_lin_pso = np.linspace(t_min_plot - 0.05 * t_range_plot, t_max_plot + 0.05 * t_range_plot, 50)
                            t_line_lin_pso = np.maximum(0, t_line_lin_pso) # Ensure time >= 0
                            y_line_lin_pso = intercept_pso_lin + slope_pso_lin * t_line_lin_pso
                            fig_pso_lin.add_trace(go.Scatter(x=t_line_lin_pso, y=y_line_lin_pso, mode='lines', name=_t("isotherm_lin_plot_legend_fit")))
                            fig_pso_lin.update_layout(template="simple_white")
                            st.plotly_chart(fig_pso_lin, use_container_width=True)
                            st.caption(_t("kinetic_pso_lin_caption", slope_pso_lin=slope_pso_lin, intercept_pso_lin=intercept_pso_lin, r2_pso_lin=r2_pso_lin, qe_lin=qe_lin, k2_lin=k2_lin))
                            # --- Figure stylisée pour PSO Linéarisé (t/qt vs t) ---
                            try:
                                x_vals = np.linspace(t_pso_lin.min(), t_pso_lin.max(), 100)
                                y_vals = intercept_pso_lin + slope_pso_lin * x_vals

                                fig_pso_styled = go.Figure()

                                # Données expérimentales : carrés noirs
                                fig_pso_styled.add_trace(go.Scatter(
                                    x=t_pso_lin,
                                    y=y_pso_lin,
                                    mode='markers',
                                    marker=dict(symbol='square', color='black', size=10),
                                    name=_t("isotherm_exp_plot_legend")
                                ))

                                # Ligne de régression : rouge épaisse
                                fig_pso_styled.add_trace(go.Scatter(
                                    x=x_vals,
                                    y=y_vals,
                                    mode='lines',
                                    line=dict(color='red', width=3),
                                    name=_t("calib_tab_legend_reg")
                                ))

                                # Mise en forme style OriginLab
                                fig_pso_styled.update_layout(
                                    width=1000,
                                    height=800,
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    font=dict(family="Times New Roman", size=22, color="black"),
                                    margin=dict(l=80, r=40, t=60, b=80),
                                    xaxis=dict(
                                        title=_t("kinetic_plot_qt_vs_t_xaxis"),
                                        linecolor='black',
                                        mirror=True,
                                        ticks='outside',
                                        showline=True,
                                        showgrid=False
                                    ),
                                    yaxis=dict(
                                        title="t / qt (min·g/mg)",
                                        linecolor='black',
                                        mirror=True,
                                        ticks='outside',
                                        showline=True,
                                        showgrid=False
                                    ),
                                    showlegend=False
                                )

                                # Annotation équation + R²
                                fig_pso_styled.add_annotation(
                                    xref="paper", yref="paper",
                                    x=0.05, y=0.95,
                                    text=f"y = {slope_pso_lin:.4f}x + {intercept_pso_lin:.4f}<br>R² = {r2_pso_lin:.4f}",
                                    showarrow=False,
                                    font=dict(size=20, color="black"),
                                    align="left"
                                )

                                # Export PNG
                                pso_img_buffer = io.BytesIO()
                                fig_pso_styled.write_image(pso_img_buffer, format="png", width=1000, height=800, scale=2)
                                pso_img_buffer.seek(0)

                                # Bouton de téléchargement
                                st.download_button(
                                    label=_t("kinetic_download_pso_lin_button"),
                                    data=pso_img_buffer,
                                    file_name=_t("kinetic_download_pso_lin_filename"),
                                    mime="image/png"
                                )
                            except Exception as e:
                                st.warning(_t("kinetic_error_export_pso_lin", e=e))


                        except ValueError as ve:
                            if "Insufficient variation" not in str(ve): st.warning(_t("kinetic_error_pso_lin_plot_regression", ve=ve))
                        except Exception as e_pso_lin:
                            st.warning(_t("kinetic_error_pso_lin_plot_general", e_pso_lin=e_pso_lin))
                else:
                        st.warning(_t("kinetic_warning_pso_lin_not_enough_points"))

                st.markdown("---") # Separator after linearized plots

                # --- 4. Intraparticle Diffusion ---
                st.subheader(_t("kinetic_ipd_subheader"))
                col_ipd_graph, _ = st.columns([2,1]) # Keep layout column
                with col_ipd_graph:
                    if not kinetic_results.empty:
                        fig_i = px.scatter(kinetic_results, x='sqrt_t', y='qt', title=_t("kinetic_ipd_plot_title"), labels={'sqrt_t': _t("kinetic_ipd_plot_xaxis"), 'qt': 'qt (mg/g)'})
                        # Plot line if IPD parameters were calculated
                        if ipd_params_list:
                            ipd_param = ipd_params_list[0] # Assuming global fit for now
                            if not np.isnan(ipd_param.get('k_id', np.nan)) and not np.isnan(ipd_param.get('C_ipd', np.nan)):
                                sqrt_t_min = kinetic_results['sqrt_t'].min()
                                sqrt_t_max = kinetic_results['sqrt_t'].max()
                                sqrt_t_range = max(1.0, sqrt_t_max - sqrt_t_min)
                                # Generate line points, ensuring it starts near or at 0
                                sqrt_t_line = np.linspace(max(0, sqrt_t_min - 0.05*sqrt_t_range), sqrt_t_max + 0.05*sqrt_t_range, 100)
                                qt_ipd_line = ipd_param['k_id'] * sqrt_t_line + ipd_param['C_ipd']
                                qt_ipd_line = np.maximum(0, qt_ipd_line) # Ensure qt >= 0
                                fig_i.add_trace(go.Scatter(x=sqrt_t_line, y=qt_ipd_line, mode='lines', name=_t("kinetic_ipd_plot_legend_fit", r2_ipd=ipd_param.get('R2_IPD', np.nan))))
                                fig_i.update_layout(template="simple_white")
                                st.plotly_chart(fig_i, use_container_width=True)
                                st.caption(_t("kinetic_ipd_caption_params", k_id=ipd_param.get('k_id', np.nan), C_ipd=ipd_param.get('C_ipd', np.nan), R2_IPD=ipd_param.get('R2_IPD', np.nan)))
                                st.caption(_t("kinetic_ipd_caption_interp"))
                                # --- Figure stylisée pour Diffusion Intraparticulaire (qt vs √t) ---
                                try:
                                    ipd_df = kinetic_results[(kinetic_results['Temps_min'] > 0)].copy()

                                    fig_ipd_styled = go.Figure()

                                    # Données expérimentales : carrés noirs
                                    fig_ipd_styled.add_trace(go.Scatter(
                                        x=ipd_df['sqrt_t'],
                                        y=ipd_df['qt'],
                                        mode='markers',
                                        marker=dict(symbol='square', color='black', size=10),
                                        name=_t("isotherm_exp_plot_legend")
                                    ))

                                    # Ligne de régression : rouge épaisse (si disponible)
                                    if ipd_params_list and not np.isnan(ipd_params_list[0]['k_id']):
                                        slope_ipd = ipd_params_list[0]['k_id']
                                        intercept_ipd = ipd_params_list[0]['C_ipd']
                                        r2_ipd = ipd_params_list[0]['R2_IPD']
                                        x_line = np.linspace(ipd_df['sqrt_t'].min(), ipd_df['sqrt_t'].max(), 100)
                                        y_line = slope_ipd * x_line + intercept_ipd

                                        fig_ipd_styled.add_trace(go.Scatter(
                                            x=x_line,
                                            y=y_line,
                                            mode='lines',
                                            line=dict(color='red', width=3),
                                            name=_t("calib_tab_legend_reg")
                                        ))

                                        # Annotation équation + R²
                                        fig_ipd_styled.add_annotation(
                                            xref="paper", yref="paper",
                                            x=0.05, y=0.95,
                                            text=f"y = {slope_ipd:.4f}x + {intercept_ipd:.4f}<br>R² = {r2_ipd:.4f}",
                                            showarrow=False,
                                            font=dict(size=20, color="black"),
                                            align="left"
                                        )

                                    # Mise en forme style OriginLab
                                    fig_ipd_styled.update_layout(
                                        width=1000,
                                        height=800,
                                        plot_bgcolor='white',
                                        paper_bgcolor='white',
                                        font=dict(family="Times New Roman", size=22, color="black"),
                                        margin=dict(l=80, r=40, t=60, b=80),
                                        xaxis=dict(
                                            title="√t (min¹ᐟ²)",
                                            linecolor='black',
                                            mirror=True,
                                            ticks='outside',
                                            showline=True,
                                            showgrid=False
                                        ),
                                        yaxis=dict(
                                            title="qt (mg/g)",
                                            linecolor='black',
                                            mirror=True,
                                            ticks='outside',
                                            showline=True,
                                            showgrid=False
                                        ),
                                        showlegend=False
                                    )

                                    # Export PNG
                                    ipd_img_buffer = io.BytesIO()
                                    fig_ipd_styled.write_image(ipd_img_buffer, format="png", width=1000, height=800, scale=2)
                                    ipd_img_buffer.seek(0)

                                    # Bouton de téléchargement
                                    st.download_button(
                                        label=_t("kinetic_download_ipd_lin_button"),
                                        data=ipd_img_buffer,
                                        file_name=_t("kinetic_download_ipd_lin_filename"),
                                        mime="image/png"
                                    )
                                except Exception as e:
                                    st.warning(_t("kinetic_error_export_ipd_lin", e=e))

                            else: # Parameters calculated but invalid
                                fig_i.update_layout(template="simple_white")
                                st.plotly_chart(fig_i, use_container_width=True)
                                st.warning(_t("kinetic_warning_ipd_params_calc_failed"))
                        else: # IPD calculation failed or not enough data
                            fig_i.update_layout(template="simple_white")
                            st.plotly_chart(fig_i, use_container_width=True)
                            st.write(_t("kinetic_info_ipd_fit_unavailable"))
                    else:
                         st.warning(_t("kinetic_warning_no_data_for_ipd"))


            elif not kinetic_results.empty:
                st.warning(_t("kinetic_warning_less_than_3_points"))

        elif kinetic_results is not None and kinetic_results.empty:
             st.warning(_t("kinetic_warning_qt_calc_no_results"))

    # --- Handling missing calibration or kinetic input ---
    elif not calib_params:
        st.warning(_t("kinetic_warning_provide_calib_data"))
    else: # kinetic_input is None
        st.info(_t("kinetic_info_enter_kinetic_data"))
# --- Onglet 4: Effet pH ---
with tab_ph:
    st.header(_t("ph_effect_tab_header"))
    ph_input = st.session_state.get('ph_effect_input')
    calib_params = st.session_state.get('calibration_params')
    ph_results = st.session_state.get('ph_effect_results')

    if ph_input and calib_params:
        if ph_results is None:
            with st.spinner(_t("ph_effect_spinner_ce_qe")):
                results_list_ph = []
                df_ph_data = ph_input['data'].copy()
                params_ph = ph_input['params']
                try:
                    if abs(calib_params['slope']) < 1e-9:
                        st.error(_t("ph_effect_error_slope_zero"))
                        raise ZeroDivisionError("Calibration slope is zero.")
                    m_fixed = params_ph['m']
                    if m_fixed <= 0:
                        st.error(_t("ph_effect_error_mass_non_positive", m_fixed=m_fixed))
                        raise ValueError("Fixed mass is non-positive.")

                    for _, row in df_ph_data.iterrows():
                        ph_val = row['pH']
                        abs_eq = row['Absorbance_Equilibre']
                        ce = max(0, (abs_eq - calib_params['intercept']) / calib_params['slope'])
                        c0_fixed = params_ph['C0']
                        v_fixed = params_ph['V']
                        qe = max(0, (c0_fixed - ce) * v_fixed / m_fixed)
                        results_list_ph.append({'pH': ph_val, 'Abs_Eq': abs_eq, 'Ce': ce, 'qe': qe, 'C0_fixe': c0_fixed, 'Masse_fixe_g': m_fixed, 'Volume_fixe_L': v_fixed})

                    if results_list_ph:
                        st.session_state['ph_effect_results'] = pd.DataFrame(results_list_ph)
                        ph_results = st.session_state['ph_effect_results']
                        st.success(_t("ph_effect_success_ce_qe_calc"))
                    else:
                        st.warning(_t("ph_effect_warning_no_valid_points"))
                        st.session_state['ph_effect_results'] = pd.DataFrame(columns=['pH', 'Abs_Eq', 'Ce', 'qe', 'C0_fixe', 'Masse_fixe_g', 'Volume_fixe_L'])

                except (ZeroDivisionError, ValueError):
                     st.session_state['ph_effect_results'] = None # Error already shown
                except Exception as e:
                    st.error(_t("ph_effect_error_ce_qe_calc_general", e=e))
                    st.session_state['ph_effect_results'] = None

        if ph_results is not None and not ph_results.empty:
            st.markdown(_t("ph_effect_calculated_data_header"))
            st.dataframe(ph_results[['pH', 'Abs_Eq', 'Ce', 'qe']].style.format({'pH': '{:.2f}', 'Abs_Eq': '{:.4f}', 'Ce': '{:.4f}', 'qe': '{:.4f}'}))
            st.caption(_t("ph_effect_conditions_caption", C0=ph_input['params']['C0'], m=ph_input['params']['m'], V=ph_input['params']['V']))

            # --- Download Button for Data ---
            csv_ph_res = convert_df_to_csv(ph_results)
            st.download_button(_t("ph_effect_download_data_button"), csv_ph_res, _t("ph_effect_download_data_filename"), "text/csv", key='dl_ph_eff_data')

            st.markdown(_t("ph_effect_plot_header"))
            try: # Plotting code starts here
                fig_ph = px.scatter(ph_results, x='pH', y='qe', title=_t("ph_effect_plot_title"), labels={'pH': 'pH', 'qe': 'qe (mg/g)'}, hover_data=ph_results.columns)
                ph_results_sorted = ph_results.sort_values('pH')
                fig_ph.add_trace(go.Scatter(x=ph_results_sorted['pH'], y=ph_results_sorted['qe'], mode='lines', name=_t("ph_effect_plot_legend_trend"), showlegend=False))
                fig_ph.update_layout(template="simple_white")
                st.plotly_chart(fig_ph, use_container_width=True)

                # --- START: Figure stylisée pour Effet pH (qe vs pH) ---
                try:
                    df_ph_styled = ph_results.sort_values(by='pH').copy()

                    fig_ph_styled = go.Figure()

                    # Données expérimentales : carrés noirs + ligne rouge épaisse
                    fig_ph_styled.add_trace(go.Scatter(
                        x=df_ph_styled['pH'],
                        y=df_ph_styled['qe'],
                        mode='markers+lines', # Combine markers and lines
                        marker=dict(symbol='square', color='black', size=10),
                        line=dict(color='red', width=3),
                        name=_t("isotherm_exp_plot_legend")
                    ))

                    # Mise en forme style OriginLab
                    fig_ph_styled.update_layout(
                        width=1000,
                        height=800,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family="Times New Roman", size=22, color="black"),
                        margin=dict(l=80, r=40, t=60, b=80),
                        xaxis=dict(
                            title="pH", # Updated axis title
                            linecolor='black',
                            mirror=True,
                            ticks='outside',
                            showline=True,
                            showgrid=False
                        ),
                        yaxis=dict(
                            title="qe (mg/g)", # Updated axis title
                            linecolor='black',
                            mirror=True,
                            ticks='outside',
                            showline=True,
                            showgrid=False
                        ),
                        showlegend=False # No legend needed for single trace
                    )

                    # Export PNG
                    ph_img_buffer = io.BytesIO()
                    fig_ph_styled.write_image(ph_img_buffer, format="png", width=1000, height=800, scale=2)
                    ph_img_buffer.seek(0)

                    # Bouton de téléchargement
                    st.download_button(
                        label=_t("ph_effect_download_styled_plot_button"),
                        data=ph_img_buffer,
                        file_name=_t("ph_effect_download_styled_plot_filename"),
                        mime="image/png",
                        key='dl_ph_fig_stylisee' # Added a unique key
                    )
                except Exception as e_export_ph:
                    st.warning(_t("ph_effect_error_export_styled_plot", e_export_ph=e_export_ph))
                # --- END: Figure stylisée ---

            except Exception as e_ph_plot:
                 st.warning(_t("ph_effect_error_plot_general", e_ph_plot=e_ph_plot))

        elif ph_results is not None and ph_results.empty:
            st.warning(_t("ph_effect_warning_ce_qe_no_results"))

    elif not calib_params:
        st.warning(_t("isotherm_warning_provide_calib_data"))
    else:
        st.info(_t("ph_effect_info_enter_ph_data"))
# --- NOUVEAU: Onglet 6: Dosage (Ajouter AVANT l'onglet Thermodynamique) ---
with tab_dosage:
    st.header(_t("dosage_tab_header"))
    dosage_input = st.session_state.get('dosage_input')
    calib_params = st.session_state.get('calibration_params')
    dosage_results = st.session_state.get('dosage_results')

    if dosage_input and calib_params:
        if dosage_results is None:
            with st.spinner(_t("dosage_spinner_ce_qe")):
                results_list_dos = []
                df_dos_data = dosage_input['data'].copy()
                params_dos = dosage_input['params']
                try:
                    if abs(calib_params['slope']) < 1e-9:
                        st.error(_t("dosage_error_slope_zero"))
                        raise ZeroDivisionError("Calibration slope is zero.")

                    v_fixed = params_dos.get('V', 0)
                    c0_fixed = params_dos.get('C0', 0)
                    if v_fixed <= 0:
                        st.error(_t("dosage_error_volume_non_positive", v_fixed=v_fixed))
                        raise ValueError("Fixed volume is non-positive.")

                    for _, row in df_dos_data.iterrows():
                        # Masse is variable here, already validated > 0 in validate_data_editor
                        m_adsorbant = row['Masse_Adsorbant_g']
                        abs_eq = row['Absorbance_Equilibre']

                        ce = max(0, (abs_eq - calib_params['intercept']) / calib_params['slope'])
                        # qe calculation uses VARIABLE mass
                        qe = max(0, (c0_fixed - ce) * v_fixed / m_adsorbant)

                        results_list_dos.append({
                            'Masse_Adsorbant_g': m_adsorbant,
                            'Abs_Eq': abs_eq, 'Ce': ce, 'qe': qe,
                            'C0_fixe': c0_fixed, 'Volume_fixe_L': v_fixed
                        })

                    if results_list_dos:
                        st.session_state['dosage_results'] = pd.DataFrame(results_list_dos)
                        dosage_results = st.session_state['dosage_results'] # Update local variable
                        st.success(_t("dosage_success_ce_qe_calc"))
                    else:
                        st.warning(_t("dosage_warning_no_valid_points"))
                        st.session_state['dosage_results'] = pd.DataFrame(columns=['Masse_Adsorbant_g', 'Abs_Eq', 'Ce', 'qe', 'C0_fixe', 'Volume_fixe_L'])

                except (ZeroDivisionError, ValueError) as calc_err_dos:
                     # Error message displayed above
                     if 'dosage_results' not in st.session_state or st.session_state['dosage_results'] is not None:
                          st.error(_t("dosage_error_ce_qe_calc_general", calc_err_dos=calc_err_dos))
                     st.session_state['dosage_results'] = None
                except Exception as e:
                    st.error(_t("dosage_error_ce_qe_calc_unexpected", e=e))
                    st.session_state['dosage_results'] = None

        # --- Affichage des résultats et graphique ---
        if dosage_results is not None and not dosage_results.empty:
            st.markdown(_t("dosage_calculated_data_header"))
            # Display relevant columns
            display_cols = ['Masse_Adsorbant_g', 'Abs_Eq', 'Ce', 'qe']
            st.dataframe(dosage_results[display_cols].style.format({'Masse_Adsorbant_g': '{:.4f}', 'Abs_Eq': '{:.4f}', 'Ce': '{:.4f}', 'qe': '{:.4f}'}))
            st.caption(_t("dosage_conditions_caption", C0=dosage_input.get('params', {}).get('C0', 'N/A'), V=dosage_input.get('params', {}).get('V', 'N/A')))

             # Download Button for Data
            csv_dos_res = convert_df_to_csv(dosage_results)
            st.download_button(_t("dosage_download_data_button"), csv_dos_res, _t("dosage_download_data_filename"), "text/csv", key='dl_dos_eff_data')


            st.markdown(_t("dosage_plot_header"))
            try:
                # Sort by mass for plotting the line correctly
                dosage_results_sorted = dosage_results.sort_values('Masse_Adsorbant_g')
                fig_dos = px.scatter(dosage_results_sorted, x='Masse_Adsorbant_g', y='qe',
                                     title=_t("dosage_plot_title"),
                                     labels={'Masse_Adsorbant_g': _t("dosage_plot_xaxis"), 'qe': 'qe (mg/g)'},
                                     hover_data=dosage_results_sorted.columns)
                # Add a line trace connecting the points
                fig_dos.add_trace(go.Scatter(x=dosage_results_sorted['Masse_Adsorbant_g'], y=dosage_results_sorted['qe'],
                                             mode='lines', name=_t("dosage_plot_legend_trend"), showlegend=False))
                fig_dos.update_layout(template="simple_white")
                st.plotly_chart(fig_dos, use_container_width=True)

                # --- START: Figure stylisée pour Effet Dosage (qe vs Masse) ---
                try:
                    df_dos_styled = dosage_results_sorted.copy() # Use the sorted data

                    fig_dos_styled = go.Figure()

                    # Données expérimentales : carrés noirs + ligne rouge épaisse
                    fig_dos_styled.add_trace(go.Scatter(
                        x=df_dos_styled['Masse_Adsorbant_g'],
                        y=df_dos_styled['qe'],
                        mode='markers+lines', # Combine markers and lines
                        marker=dict(symbol='square', color='black', size=10),
                        line=dict(color='red', width=3),
                        name=_t("isotherm_exp_plot_legend")
                    ))

                    # Mise en forme style OriginLab
                    fig_dos_styled.update_layout(
                        width=1000,
                        height=800,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family="Times New Roman", size=22, color="black"),
                        margin=dict(l=80, r=40, t=60, b=80),
                        xaxis=dict(
                            title=_t("dosage_plot_xaxis"), # Updated axis title
                            linecolor='black',
                            mirror=True,
                            ticks='outside',
                            showline=True,
                            showgrid=False
                        ),
                        yaxis=dict(
                            title="qe (mg/g)", # Updated axis title
                            linecolor='black',
                            mirror=True,
                            ticks='outside',
                            showline=True,
                            showgrid=False
                        ),
                        showlegend=False # No legend needed for single trace
                    )

                    # Export PNG
                    dos_img_buffer = io.BytesIO()
                    fig_dos_styled.write_image(dos_img_buffer, format="png", width=1000, height=800, scale=2)
                    dos_img_buffer.seek(0)

                    # Bouton de téléchargement
                    st.download_button(
                        label=_t("dosage_download_styled_plot_button"),
                        data=dos_img_buffer,
                        file_name=_t("dosage_download_styled_plot_filename"),
                        mime="image/png",
                        key='dl_dos_fig_stylisee' # Added a unique key
                    )
                except Exception as e_export_dos:
                    st.warning(_t("dosage_error_export_styled_plot", e_export_dos=e_export_dos))
                # --- END: Figure stylisée ---

            except Exception as e_dos_plot:
                st.warning(_t("dosage_error_plot_general", e_dos_plot=e_dos_plot))

        elif dosage_results is not None and dosage_results.empty:
            st.warning(_t("dosage_warning_ce_qe_no_results"))

    elif not calib_params:
        st.warning(_t("isotherm_warning_provide_calib_data"))
    else: # dosage_input is None
        st.info(_t("dosage_info_enter_dosage_data"))

# --- FIN de l'onglet Dosage ---

# --- Onglet 5: Effet T° ---
with tab_temp:
    st.header(_t("temp_effect_tab_header"))
    temp_input = st.session_state.get('temp_effect_input')
    calib_params = st.session_state.get('calibration_params')
    temp_results = st.session_state.get('temp_effect_results')

    if temp_input and calib_params:
        if temp_results is None:
            with st.spinner(_t("temp_effect_spinner_ce_qe")):
                results_list_temp = []
                df_temp_data = temp_input['data'].copy()
                params_temp = temp_input['params']
                try:
                    if abs(calib_params['slope']) < 1e-9:
                        st.error(_t("temp_effect_error_slope_zero"))
                        raise ZeroDivisionError("Calibration slope is zero.")
                    m_fixed = params_temp['m']
                    if m_fixed <= 0:
                        st.error(_t("temp_effect_error_mass_non_positive", m_fixed=m_fixed))
                        raise ValueError("Fixed mass is non-positive.")

                    for _, row in df_temp_data.iterrows():
                        T_val = row['Temperature_C']
                        abs_eq = row['Absorbance_Equilibre']
                        ce = max(0, (abs_eq - calib_params['intercept']) / calib_params['slope'])
                        c0_fixed = params_temp['C0']
                        v_fixed = params_temp['V']
                        qe = max(0, (c0_fixed - ce) * v_fixed / m_fixed)
                        results_list_temp.append({'Temperature_C': T_val, 'Abs_Eq': abs_eq, 'Ce': ce, 'qe': qe, 'C0_fixe': c0_fixed, 'Masse_fixe_g': m_fixed, 'Volume_fixe_L': v_fixed})

                    if results_list_temp:
                        st.session_state['temp_effect_results'] = pd.DataFrame(results_list_temp)
                        temp_results = st.session_state['temp_effect_results']
                        st.success(_t("temp_effect_success_ce_qe_calc"))
                        st.session_state['thermo_params'] = None # Reset thermo params
                    else:
                        st.warning(_t("temp_effect_warning_no_valid_points"))
                        st.session_state['temp_effect_results'] = pd.DataFrame(columns=['Temperature_C', 'Abs_Eq', 'Ce', 'qe', 'C0_fixe', 'Masse_fixe_g', 'Volume_fixe_L'])

                except (ZeroDivisionError, ValueError):
                    st.session_state['temp_effect_results'] = None # Error already shown
                except Exception as e:
                    st.error(_t("temp_effect_error_ce_qe_calc_general", e=e))
                    st.session_state['temp_effect_results'] = None

        if temp_results is not None and not temp_results.empty:
            st.markdown(_t("temp_effect_calculated_data_header"))
            st.dataframe(temp_results[['Temperature_C', 'Abs_Eq', 'Ce', 'qe']].style.format({'Temperature_C': '{:.1f}', 'Abs_Eq': '{:.4f}', 'Ce': '{:.4f}', 'qe': '{:.4f}'}))
            st.caption(_t("temp_effect_conditions_caption", C0=temp_input['params']['C0'], m=temp_input['params']['m'], V=temp_input['params']['V']))

            # --- Download Button for Data ---
            csv_t_res = convert_df_to_csv(temp_results)
            st.download_button(_t("temp_effect_download_data_button"), csv_t_res, _t("temp_effect_download_data_filename"), "text/csv", key='dl_t_eff_data')

            st.markdown(_t("temp_effect_plot_header"))
            try:
                temp_results_sorted = temp_results.sort_values('Temperature_C')
                fig_t = px.scatter(temp_results_sorted, x='Temperature_C', y='qe', title=_t("temp_effect_plot_title"), labels={'Temperature_C': _t("temp_effect_plot_xaxis"), 'qe': 'qe (mg/g)'}, hover_data=temp_results.columns)
                fig_t.add_trace(go.Scatter(x=temp_results_sorted['Temperature_C'], y=temp_results_sorted['qe'], mode='lines', name=_t("temp_effect_plot_legend_trend"), showlegend=False))
                fig_t.update_layout(template="simple_white")
                st.plotly_chart(fig_t, use_container_width=True)

                # --- START: Figure stylisée pour Effet Température (qe vs T) ---
                try:
                    df_temp_styled = temp_results_sorted.copy() # Use the sorted data

                    fig_temp_styled = go.Figure()

                    # Données expérimentales : carrés noirs + ligne rouge épaisse
                    fig_temp_styled.add_trace(go.Scatter(
                        x=df_temp_styled['Temperature_C'],
                        y=df_temp_styled['qe'],
                        mode='markers+lines', # Combine markers and lines
                        marker=dict(symbol='square', color='black', size=10),
                        line=dict(color='red', width=3),
                        name=_t("isotherm_exp_plot_legend")
                    ))

                    # Mise en forme style OriginLab
                    fig_temp_styled.update_layout(
                        width=1000,
                        height=800,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family="Times New Roman", size=22, color="black"),
                        margin=dict(l=80, r=40, t=60, b=80),
                        xaxis=dict(
                            title=_t("temp_effect_plot_xaxis"), # Updated axis title
                            linecolor='black',
                            mirror=True,
                            ticks='outside',
                            showline=True,
                            showgrid=False
                        ),
                        yaxis=dict(
                            title="qe (mg/g)", # Updated axis title
                            linecolor='black',
                            mirror=True,
                            ticks='outside',
                            showline=True,
                            showgrid=False
                        ),
                        showlegend=False # No legend needed for single trace
                    )

                    # Export PNG
                    temp_img_buffer = io.BytesIO()
                    fig_temp_styled.write_image(temp_img_buffer, format="png", width=1000, height=800, scale=2)
                    temp_img_buffer.seek(0)

                    # Bouton de téléchargement
                    st.download_button(
                        label=_t("temp_effect_download_styled_plot_button"),
                        data=temp_img_buffer,
                        file_name=_t("temp_effect_download_styled_plot_filename"),
                        mime="image/png",
                        key='dl_temp_fig_stylisee' # Added a unique key
                    )
                except Exception as e_export_temp:
                    st.warning(_t("temp_effect_error_export_styled_plot", e_export_temp=e_export_temp))
                # --- END: Figure stylisée ---

            except Exception as e_t_plot:
                st.warning(_t("temp_effect_error_plot_general", e_t_plot=e_t_plot))

        elif temp_results is not None and temp_results.empty:
            st.warning(_t("temp_effect_warning_ce_qe_no_results"))

    elif not calib_params:
        st.warning(_t("isotherm_warning_provide_calib_data"))
    else:
        st.info(_t("temp_effect_info_enter_temp_data"))

# --- Onglet 6: Thermodynamique ---
with tab_thermo:
    st.header(_t("thermo_tab_header"))
    st.markdown(_t("thermo_tab_intro_markdown"))
    temp_results_for_thermo = st.session_state.get('temp_effect_results')
    thermo_params = st.session_state.get('thermo_params')

    if temp_results_for_thermo is not None and not temp_results_for_thermo.empty and thermo_params is None:
        with st.spinner(_t("thermo_spinner_analysis")):
            R = 8.314 # J/mol·K
            df_thermo = temp_results_for_thermo.copy()
            df_thermo = df_thermo[(df_thermo['Ce'] > 1e-9) & (df_thermo['qe'] >= 0)].copy() # qe >= 0 is fine here

            if len(df_thermo['Temperature_C'].unique()) >= 2:
                try:
                    df_thermo['T_K'] = df_thermo['Temperature_C'] + 273.15
                    df_thermo['inv_T_K'] = 1 / df_thermo['T_K']
                    df_thermo['Kd'] = df_thermo['qe'] / df_thermo['Ce']
                    df_thermo_valid_kd = df_thermo[df_thermo['Kd'] > 1e-9].copy() # Filter for log

                    if len(df_thermo_valid_kd['T_K'].unique()) >= 2:
                        df_thermo_valid_kd['ln_Kd'] = np.log(df_thermo_valid_kd['Kd'])
                        inv_T_valid = df_thermo_valid_kd['inv_T_K'].values
                        ln_Kd_valid = df_thermo_valid_kd['ln_Kd'].values
                        temps_K_valid = df_thermo_valid_kd['T_K'].values
                        temps_C_valid = df_thermo_valid_kd['Temperature_C'].values
                        kd_values_valid = df_thermo_valid_kd['Kd'].values

                        if df_thermo_valid_kd['inv_T_K'].nunique() < 2 or df_thermo_valid_kd['ln_Kd'].nunique() < 2:
                             st.warning(_t("thermo_warning_insufficient_variation_vant_hoff"))
                             raise ValueError("Insufficient variation for Van't Hoff")

                        slope, intercept, r_val, p_val, std_err = linregress(inv_T_valid, ln_Kd_valid)
                        delta_H_J_mol = -slope * R
                        delta_H_kJ_mol = delta_H_J_mol / 1000
                        delta_S_J_mol_K = intercept * R
                        r_squared_vt = r_val**2
                        delta_G_kJ_mol_dict = {round(T_c, 1): (delta_H_J_mol - T_k * delta_S_J_mol_K) / 1000
                                                for T_k, T_c in zip(temps_K_valid, temps_C_valid)}

                        st.session_state['thermo_params'] = {
                            'Delta_H_kJ_mol': delta_H_kJ_mol, 'Delta_S_J_mol_K': delta_S_J_mol_K,
                            'Delta_G_kJ_mol': delta_G_kJ_mol_dict, 'R2_Van_t_Hoff': r_squared_vt,
                            'ln_K': ln_Kd_valid.tolist(), 'inv_T': inv_T_valid.tolist(),
                            'temps_K_valid': temps_K_valid.tolist(), 'K_values': dict(zip(temps_K_valid, kd_values_valid)),
                            'Analysis_Type': 'Kd' }
                        thermo_params = st.session_state['thermo_params']
                        st.success(_t("thermo_success_analysis_kd"))

                    else:
                        st.warning(_t("thermo_warning_not_enough_distinct_temps_kd"))
                        st.session_state['thermo_params'] = None

                except ValueError as ve: # Catch variation errors
                     st.session_state['thermo_params'] = None # Message already shown
                except Exception as e_vth:
                    st.error(_t("thermo_error_vant_hoff_kd", e_vth=e_vth))
                    st.session_state['thermo_params'] = None
            else:
                st.warning(_t("thermo_warning_not_enough_distinct_temps_ce"))
                st.session_state['thermo_params'] = None

    # --- Affichage Thermo ---
    if thermo_params and thermo_params.get('Analysis_Type') == 'Kd':
        st.markdown(_t("thermo_calculated_params_header"))
        col_th1, col_th2 = st.columns(2)
        with col_th1:
            st.metric("ΔH° (kJ/mol)", f"{thermo_params['Delta_H_kJ_mol']:.2f}", help=_t("thermo_delta_h_help"))
            st.metric("ΔS° (J/mol·K)", f"{thermo_params['Delta_S_J_mol_K']:.2f}", help=_t("thermo_delta_s_help"))
            st.metric("R² (Van't Hoff)", f"{thermo_params['R2_Van_t_Hoff']:.3f}", help=_t("thermo_r2_vant_hoff_help"))
        with col_th2:
            st.write(_t("thermo_delta_g_header"))
            if thermo_params['Delta_G_kJ_mol']:
                 dG_df = pd.DataFrame(list(thermo_params['Delta_G_kJ_mol'].items()), columns=['T (°C)', 'ΔG° (kJ/mol)'])
                 dG_df = dG_df.sort_values(by='T (°C)').reset_index(drop=True)
                 st.dataframe(dG_df.style.format({'T (°C)': '{:.1f}','ΔG° (kJ/mol)': '{:.2f}'}), height=min(200, (len(dG_df)+1)*35 + 3))
                 st.caption(_t("thermo_delta_g_spontaneous_caption"))
            else: st.write(_t("thermo_delta_g_not_calculated"))

        st.markdown(_t("thermo_vant_hoff_plot_header"))
        try:
            if thermo_params.get('inv_T') and thermo_params.get('ln_K'):
                df_vt = pd.DataFrame({'1/T (1/K)': thermo_params['inv_T'], 'ln(Kd)': thermo_params['ln_K']})
                fig_vt = px.scatter(df_vt, x='1/T (1/K)', y='ln(Kd)', title=_t("thermo_vant_hoff_plot_title"), labels={'1/T (1/K)': '1 / T (1/K)', 'ln(Kd)': 'ln(Kd)'})
                R_gas = 8.314
                slope_vt = -thermo_params['Delta_H_kJ_mol'] * 1000 / R_gas
                intercept_vt = thermo_params['Delta_S_J_mol_K'] / R_gas
                inv_T_line = np.linspace(min(thermo_params['inv_T']), max(thermo_params['inv_T']), 50)
                ln_K_line = slope_vt * inv_T_line + intercept_vt
                fig_vt.add_trace(go.Scatter(x=inv_T_line, y=ln_K_line, mode='lines', name=_t("thermo_vant_hoff_plot_legend_fit", r2_vt=thermo_params["R2_Van_t_Hoff"])))
                fig_vt.update_layout(template="simple_white")
                st.plotly_chart(fig_vt, use_container_width=True)

            # --- Export PNG stylisé Van’t Hoff ---
            try:
                fig_vt_styled = go.Figure()

                fig_vt_styled.add_trace(go.Scatter(
                    x=thermo_params['inv_T'],
                    y=thermo_params['ln_K'],
                    mode='markers',
                    marker=dict(symbol='square', color='black', size=10),
                    
                    name=_t("isotherm_exp_plot_legend")
                ))

                fig_vt_styled.update_layout(
                    width=1000,
                    height=800,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Times New Roman", size=22, color="black"),
                    margin=dict(l=80, r=40, t=60, b=80),
                    xaxis=dict(
                        title="1 / T (1/K)",
                        linecolor='black',
                        mirror=True,
                        ticks='outside',
                        showline=True,
                        showgrid=False
                    ),
                    yaxis=dict(
                        title="ln(Kd)",
                        linecolor='black',
                        mirror=True,
                        ticks='outside',
                        showline=True,
                        showgrid=False
                    ),
                    showlegend=False
                )

                # Ligne de régression
                x_vals = np.linspace(min(thermo_params['inv_T']), max(thermo_params['inv_T']), 100)
                slope_vt = -thermo_params['Delta_H_kJ_mol'] * 1000 / 8.314
                intercept_vt = thermo_params['Delta_S_J_mol_K'] / 8.314
                y_vals = slope_vt * x_vals + intercept_vt

                fig_vt_styled.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    line=dict(color='red', width=3),
                    name=_t("calib_tab_legend_reg")
                ))

                # Annotation équation
                fig_vt_styled.add_annotation(
                    xref="paper", yref="paper",
                    x=0.05, y=0.95,
                    text=f"y = {slope_vt:.4f}x + {intercept_vt:.4f}<br>R² = {thermo_params['R2_Van_t_Hoff']:.4f}",
                    showarrow=False,
                    font=dict(size=20, color="black"),
                    align="left"
                )

                # Export PNG
                vt_img_buffer = io.BytesIO()
                fig_vt_styled.write_image(vt_img_buffer, format="png", width=1000, height=800, scale=2)
                vt_img_buffer.seek(0)

                st.download_button(
                    label=_t("thermo_download_vant_hoff_styled_button"),
                    data=vt_img_buffer,
                    file_name=_t("thermo_download_vant_hoff_styled_filename"),
                    mime=_t("thermo_error_export_vant_hoff_styled", e=e),
                    key="dl_vt_stylise"
                )
            except Exception as e:
                st.warning(_t("thermo_error_export_vant_hoff_styled", e=e))    
        except Exception as e_vt_plot:
            st.warning(_t("thermo_error_plot_vant_hoff", e_vt_plot=e_vt_plot))

        st.markdown(_t("thermo_kd_coeffs_header"))
        if thermo_params.get('K_values'):
            k_vals_list = [{'T_K': T_k, 'Kd (L/g)': Kd} for T_k, Kd in thermo_params['K_values'].items()]
            k_vals_df = pd.DataFrame(k_vals_list)
            k_vals_df['Température (°C)'] = k_vals_df['T_K'] - 273.15
            k_vals_df = k_vals_df[['Température (°C)', 'Kd (L/g)']].sort_values(by='Température (°C)').reset_index(drop=True)
            st.dataframe(k_vals_df.style.format({'Température (°C)': '{:.1f}','Kd (L/g)': '{:.4g}'}))
        else: st.write(_t("thermo_kd_unavailable"))

        # --- Download Thermo Data ---
        col_dlt1, col_dlt2 = st.columns(2)
        with col_dlt1:
            thermo_res_export = {'Delta_H_kJ_mol': thermo_params['Delta_H_kJ_mol'], 'Delta_S_J_mol_K': thermo_params['Delta_S_J_mol_K'], 'R2_Van_t_Hoff': thermo_params['R2_Van_t_Hoff'], **{f'Delta_G_kJ_mol_{T_C}C': G for T_C, G in thermo_params['Delta_G_kJ_mol'].items()}}
            thermo_df_export = pd.DataFrame([thermo_res_export])
            csv_t_params = convert_df_to_csv(thermo_df_export)
            st.download_button(_t("thermo_download_params_kd_button"), csv_t_params, _t("thermo_download_params_kd_filename"), "text/csv", key="dl_t_p_kd_tab")
        with col_dlt2:
             if thermo_params.get('inv_T') and thermo_params.get('ln_K'):
                df_vt_export = pd.DataFrame({'1/T (1/K)': thermo_params['inv_T'], 'ln(Kd)': thermo_params['ln_K']})
                csv_vt_data = convert_df_to_csv(df_vt_export)
                st.download_button(_t("thermo_download_data_vant_hoff_kd_button"), csv_vt_data, _t("thermo_download_data_vant_hoff_kd_filename"), "text/csv", key="dl_vt_d_kd_tab")

    elif temp_results_for_thermo is None or temp_results_for_thermo.empty:
         st.info(_t("thermo_info_provide_temp_data"))
    elif len(temp_results_for_thermo['Temperature_C'].unique()) < 2:
         st.warning(_t("thermo_warning_less_than_2_distinct_temps"))
    elif thermo_params is None and st.session_state.get('temp_effect_results') is not None:
         st.warning(_t("thermo_warning_analysis_not_done_kd"))
    elif thermo_params and thermo_params.get('Analysis_Type') != 'Kd':
        st.warning(_t("thermo_warning_params_calculated_differently"))


# --- Pied de page ---
st.markdown("---")
st.caption("Analyse Adsorption v2.4")

# --- FIN DU FICHIER ---
