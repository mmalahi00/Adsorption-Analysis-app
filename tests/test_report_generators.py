# tests/test_report_generators.py
"""
Tests for Report Tab Figure and Table Generators
==================================================

Tests ``report_tab.generate_figure``, ``report_tab.generate_table``, and the
individual ``_gen_*`` helper functions using mock study-state dictionaries.

Multi-study generators that call ``_get_all_studies()`` (which reads
``st.session_state``) are tested with patched session state.
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

go = pytest.importorskip("plotly.graph_objects", reason="plotly required for report_tab")


# =============================================================================
# FIXTURES — mock study states with realistic data
# =============================================================================


@pytest.fixture
def calibration_state():
    """Study state containing calibration data and params."""
    conc = np.array([5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    absorbance = 0.05 * conc + 0.01 + np.random.default_rng(42).normal(0, 0.005, len(conc))
    return {
        "calib_df_input": pd.DataFrame(
            {"Concentration": conc, "Absorbance": absorbance}
        ),
        "calibration_params": {
            "slope": 0.05,
            "intercept": 0.01,
            "r_squared": 0.999,
            "std_err_slope": 0.001,
            "std_err_intercept": 0.002,
        },
    }


@pytest.fixture
def isotherm_state(calibration_state):
    """Study state containing isotherm results and fitted models."""
    Ce = np.array([2.0, 5.0, 10.0, 20.0, 35.0, 50.0])
    qe = np.array([24.0, 22.5, 19.0, 14.0, 9.0, 6.5])
    C0 = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
    state = dict(calibration_state)
    state.update(
        {
            "isotherm_results": pd.DataFrame(
                {
                    "C0_mgL": C0,
                    "Ce_mgL": Ce,
                    "qe_mg_g": qe,
                    "removal_%": ((C0 - Ce) / C0) * 100,
                    "Ce_error": np.zeros_like(Ce),
                    "qe_error": np.zeros_like(qe),
                }
            ),
            "isotherm_models_fitted": {
                "Langmuir": {
                    "converged": True,
                    "params": {"qm": 28.0, "KL": 0.15},
                    "r_squared": 0.985,
                    "adj_r_squared": 0.981,
                    "rmse": 0.8,
                    "aicc": 12.5,
                    "aic": 11.0,
                    "bic": 13.0,
                    "press": 5.0,
                    "q2": 0.95,
                    "ci_95": {"qm": (25.0, 31.0), "KL": (0.10, 0.20)},
                    "RL": 1 / (1 + 0.15 * C0),
                },
                "Freundlich": {
                    "converged": True,
                    "params": {"KF": 8.5, "n_inv": 0.45},
                    "r_squared": 0.972,
                    "adj_r_squared": 0.965,
                    "rmse": 1.2,
                    "aicc": 18.3,
                    "aic": 17.0,
                    "bic": 19.0,
                    "press": 8.0,
                    "q2": 0.91,
                    "ci_95": {"KF": (7.0, 10.0), "n_inv": (0.35, 0.55)},
                },
                "Temkin": {
                    "converged": True,
                    "params": {"B1": 5.0, "KT": 1.2},
                    "r_squared": 0.960,
                    "adj_r_squared": 0.950,
                    "rmse": 1.5,
                    "aicc": 22.0,
                    "aic": 21.0,
                    "bic": 23.0,
                    "ci_95": {"B1": (4.0, 6.0), "KT": (0.8, 1.6)},
                },
                "Sips": {
                    "converged": True,
                    "params": {"qm": 30.0, "Ks": 0.12, "ns": 0.8},
                    "r_squared": 0.990,
                    "adj_r_squared": 0.985,
                    "rmse": 0.6,
                    "aicc": 10.0,
                    "aic": 8.0,
                    "bic": 12.0,
                    "ci_95": {
                        "qm": (27.0, 33.0),
                        "Ks": (0.08, 0.16),
                        "ns": (0.6, 1.0),
                    },
                },
                "UnconvergedModel": {"converged": False},
            },
        }
    )
    return state


@pytest.fixture
def kinetic_state(calibration_state):
    """Study state containing kinetic results and fitted models."""
    t = np.array([0, 5, 10, 20, 30, 60, 90, 120], dtype=float)
    qt = np.array([0, 3.5, 6.0, 9.5, 11.5, 13.5, 14.0, 14.2])
    state = dict(calibration_state)
    state.update(
        {
            "kinetic_results_df": pd.DataFrame(
                {
                    "Time": t,
                    "qt_mg_g": qt,
                    "Ct_mgL": 50.0 - qt * (0.1 / 0.05),
                    "removal_%": (qt * (0.1 / 0.05) / 50.0) * 100,
                }
            ),
            "kinetic_models_fitted": {
                "PFO": {
                    "converged": True,
                    "params": {"qe": 14.5, "k1": 0.05},
                    "r_squared": 0.965,
                    "adj_r_squared": 0.958,
                    "rmse": 0.7,
                    "aicc": 8.0,
                    "ci_95": {"qe": (13.0, 16.0), "k1": (0.03, 0.07)},
                },
                "PSO": {
                    "converged": True,
                    "params": {"qe": 15.0, "k2": 0.008},
                    "r_squared": 0.992,
                    "adj_r_squared": 0.990,
                    "rmse": 0.3,
                    "aicc": 4.0,
                    "ci_95": {"qe": (14.0, 16.0), "k2": (0.005, 0.011)},
                },
            },
        }
    )
    return state


@pytest.fixture
def thermo_state():
    """Study state with thermodynamic parameters."""
    T_K = np.array([298.15, 308.15, 318.15, 328.15])
    return {
        "thermo_params": {
            "temperatures": T_K.tolist(),
            "Kd_values": [5.0, 4.2, 3.5, 3.0],
            "slope": -2500.0,
            "intercept": 10.0,
            "delta_H": -20.8,
            "delta_S": 83.1,
            "delta_G": {
                298.15: -4.5,
                308.15: -3.7,
                318.15: -2.8,
                328.15: -2.0,
            },
        },
    }


@pytest.fixture
def effect_state():
    """Study state with pH, temperature, and dosage effect results."""
    return {
        "ph_effect_results": pd.DataFrame(
            {
                "pH": [2, 4, 6, 7, 8, 10],
                "qe_mg_g": [5, 12, 18, 20, 17, 8],
                "removal_%": [10, 24, 36, 40, 34, 16],
            }
        ),
        "temp_effect_results": pd.DataFrame(
            {
                "Temperature": [25, 35, 45, 55],
                "qe_mg_g": [15, 17, 19, 20],
                "removal_%": [30, 34, 38, 40],
            }
        ),
        "dosage_effect_results": pd.DataFrame(
            {
                "Mass_g": [0.02, 0.05, 0.1, 0.2, 0.5],
                "qe_mg_g": [40, 25, 15, 8, 3],
                "removal_%": [20, 35, 55, 75, 92],
            }
        ),
    }


@pytest.fixture
def full_state(isotherm_state, kinetic_state, thermo_state, effect_state):
    """Merged study state with all analysis sections populated."""
    merged = {}
    merged.update(isotherm_state)
    merged.update(kinetic_state)
    merged.update(thermo_state)
    merged.update(effect_state)
    return merged


@pytest.fixture
def empty_state():
    """Completely empty study state."""
    return {}


# =============================================================================
# generate_figure DISPATCHER
# =============================================================================


class TestGenerateFigure:
    """Test report_tab.generate_figure dispatcher."""

    def test_unknown_id_returns_none(self, full_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        assert generate_figure("nonexistent_id", full_state) is None

    def test_none_state_returns_none(self):
        from adsorblab_pro.tabs.report_tab import generate_figure

        assert generate_figure("calib_curve", None) is None

    def test_empty_state_returns_none(self, empty_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        assert generate_figure("calib_curve", empty_state) is None

    def test_calib_curve(self, calibration_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        fig = generate_figure("calib_curve", calibration_state)
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_calib_residuals(self, calibration_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        fig = generate_figure("calib_residuals", calibration_state)
        assert fig is not None

    def test_iso_overview(self, isotherm_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        fig = generate_figure("iso_overview", isotherm_state)
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_iso_langmuir(self, isotherm_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        fig = generate_figure("iso_langmuir", isotherm_state)
        assert fig is not None

    def test_iso_freundlich(self, isotherm_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        fig = generate_figure("iso_freundlich", isotherm_state)
        assert fig is not None

    def test_iso_temkin(self, isotherm_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        fig = generate_figure("iso_temkin", isotherm_state)
        assert fig is not None

    def test_iso_sips(self, isotherm_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        fig = generate_figure("iso_sips", isotherm_state)
        assert fig is not None

    def test_iso_comparison(self, isotherm_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        fig = generate_figure("iso_comparison", isotherm_state)
        assert fig is not None

    def test_iso_rl(self, isotherm_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        fig = generate_figure("iso_rl", isotherm_state)
        assert fig is not None

    def test_kin_overview(self, kinetic_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        fig = generate_figure("kin_overview", kinetic_state)
        assert fig is not None

    def test_kin_pfo(self, kinetic_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        fig = generate_figure("kin_pfo", kinetic_state)
        assert fig is not None

    def test_kin_pso(self, kinetic_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        fig = generate_figure("kin_pso", kinetic_state)
        assert fig is not None

    def test_kin_comparison(self, kinetic_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        fig = generate_figure("kin_comparison", kinetic_state)
        assert fig is not None

    def test_thermo_vanthoff(self, thermo_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        fig = generate_figure("thermo_vanthoff", thermo_state)
        assert fig is not None

    def test_thermo_gibbs(self, thermo_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        fig = generate_figure("thermo_gibbs", thermo_state)
        assert fig is not None

    def test_effect_ph(self, effect_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        fig = generate_figure("effect_ph", effect_state)
        assert fig is not None

    def test_effect_temp(self, effect_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        fig = generate_figure("effect_temp", effect_state)
        assert fig is not None

    def test_effect_dosage(self, effect_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        fig = generate_figure("effect_dosage", effect_state)
        assert fig is not None

    def test_saved_3d_figure_lookup(self, full_state):
        from adsorblab_pro.tabs.report_tab import generate_figure

        dummy_fig = go.Figure()
        dummy_fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
        full_state["saved_3d_figures"] = {
            "custom_3d_1": {
                "figure": dummy_fig.to_dict(),
                "title": "Custom 3D Figure",
            }
        }
        fig = generate_figure("custom_3d_1", full_state)
        assert fig is not None
        assert isinstance(fig, go.Figure)


# =============================================================================
# INDIVIDUAL FIGURE GENERATORS
# =============================================================================


class TestCalibrationFigures:
    """Test calibration figure generators with edge cases."""

    def test_calib_curve_missing_params(self):
        from adsorblab_pro.tabs.report_tab import _gen_calibration_curve

        assert _gen_calibration_curve({"calibration_params": None}) is None
        assert _gen_calibration_curve({"calib_df_input": None}) is None

    def test_calib_residuals_returns_figure(self, calibration_state):
        from adsorblab_pro.tabs.report_tab import _gen_calibration_residuals

        fig = _gen_calibration_residuals(calibration_state)
        assert isinstance(fig, go.Figure)

    def test_calib_residuals_missing_data(self):
        from adsorblab_pro.tabs.report_tab import _gen_calibration_residuals

        assert _gen_calibration_residuals({}) is None


class TestIsothermFigures:
    """Test isotherm figure generators."""

    def test_overview_missing_data(self):
        from adsorblab_pro.tabs.report_tab import _gen_isotherm_overview

        assert _gen_isotherm_overview({}) is None
        assert _gen_isotherm_overview({"isotherm_results": None}) is None

    def test_model_unconverged_returns_none(self, isotherm_state):
        from adsorblab_pro.tabs.report_tab import _gen_isotherm_model

        assert _gen_isotherm_model(isotherm_state, "UnconvergedModel") is None

    def test_model_missing_returns_none(self, isotherm_state):
        from adsorblab_pro.tabs.report_tab import _gen_isotherm_model

        assert _gen_isotherm_model(isotherm_state, "NonexistentModel") is None

    def test_comparison_empty_models(self):
        from adsorblab_pro.tabs.report_tab import _gen_isotherm_comparison

        state = {
            "isotherm_results": pd.DataFrame({"Ce_mgL": [1], "qe_mg_g": [1]}),
            "isotherm_models_fitted": {},
        }
        result = _gen_isotherm_comparison(state)
        # Empty models dict → returns None or empty fig
        assert result is None or isinstance(result, go.Figure)

    def test_separation_factor_no_langmuir(self):
        from adsorblab_pro.tabs.report_tab import _gen_separation_factor

        state = {
            "isotherm_results": pd.DataFrame({"C0_mgL": [10], "Ce_mgL": [5]}),
            "isotherm_models_fitted": {},
        }
        assert _gen_separation_factor(state) is None


class TestKineticFigures:
    """Test kinetic figure generators."""

    def test_overview_missing_data(self):
        from adsorblab_pro.tabs.report_tab import _gen_kinetic_overview

        assert _gen_kinetic_overview({}) is None
        assert _gen_kinetic_overview({"kinetic_results_df": None}) is None

    def test_model_unconverged(self, kinetic_state):
        from adsorblab_pro.tabs.report_tab import _gen_kinetic_model

        kinetic_state["kinetic_models_fitted"]["PFO"]["converged"] = False
        assert _gen_kinetic_model(kinetic_state, "PFO") is None

    def test_model_missing(self, kinetic_state):
        from adsorblab_pro.tabs.report_tab import _gen_kinetic_model

        assert _gen_kinetic_model(kinetic_state, "NonexistentModel") is None


class TestThermoFigures:
    """Test thermodynamic figure generators."""

    def test_vanthoff_missing_data(self):
        from adsorblab_pro.tabs.report_tab import _gen_vanthoff_plot

        assert _gen_vanthoff_plot({}) is None
        assert _gen_vanthoff_plot({"thermo_params": None}) is None

    def test_vanthoff_insufficient_points(self):
        from adsorblab_pro.tabs.report_tab import _gen_vanthoff_plot

        state = {"thermo_params": {"temperatures": [300], "Kd_values": [5.0]}}
        assert _gen_vanthoff_plot(state) is None

    def test_gibbs_missing_data(self):
        from adsorblab_pro.tabs.report_tab import _gen_gibbs_plot

        assert _gen_gibbs_plot({}) is None

    def test_gibbs_insufficient_temps(self):
        from adsorblab_pro.tabs.report_tab import _gen_gibbs_plot

        state = {"thermo_params": {"temperatures": [300], "delta_G": {}}}
        assert _gen_gibbs_plot(state) is None


class TestEffectFigures:
    """Test effect study figure generators."""

    def test_ph_effect_missing(self):
        from adsorblab_pro.tabs.report_tab import _gen_ph_effect

        assert _gen_ph_effect({}) is None
        assert _gen_ph_effect({"ph_effect_results": None}) is None

    def test_ph_effect_empty_df(self):
        from adsorblab_pro.tabs.report_tab import _gen_ph_effect

        state = {"ph_effect_results": pd.DataFrame()}
        assert _gen_ph_effect(state) is None

    def test_ph_effect_missing_columns(self):
        from adsorblab_pro.tabs.report_tab import _gen_ph_effect

        state = {"ph_effect_results": pd.DataFrame({"x": [1, 2]})}
        assert _gen_ph_effect(state) is None

    def test_temp_effect_missing(self):
        from adsorblab_pro.tabs.report_tab import _gen_temperature_effect

        assert _gen_temperature_effect({}) is None

    def test_temp_effect_valid(self, effect_state):
        from adsorblab_pro.tabs.report_tab import _gen_temperature_effect

        fig = _gen_temperature_effect(effect_state)
        assert isinstance(fig, go.Figure)

    def test_dosage_effect_missing(self):
        from adsorblab_pro.tabs.report_tab import _gen_dosage_effect

        assert _gen_dosage_effect({}) is None

    def test_dosage_effect_valid(self, effect_state):
        from adsorblab_pro.tabs.report_tab import _gen_dosage_effect

        fig = _gen_dosage_effect(effect_state)
        assert isinstance(fig, go.Figure)


# =============================================================================
# generate_table DISPATCHER
# =============================================================================


class TestGenerateTable:
    """Test report_tab.generate_table dispatcher."""

    def test_unknown_id_returns_none(self, full_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        assert generate_table("nonexistent_table", full_state) is None

    def test_empty_state_returns_none(self, empty_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        assert generate_table("tbl_calib_params", empty_state) is None


# =============================================================================
# TABLE GENERATORS
# =============================================================================


class TestCalibrationTables:
    """Test calibration table generators."""

    def test_calib_params_table(self, calibration_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        df = generate_table("tbl_calib_params", calibration_state)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # Slope, Intercept, R²
        assert "Parameter" in df.columns
        assert "Value" in df.columns

    def test_calib_data_table(self, calibration_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        df = generate_table("tbl_calib_data", calibration_state)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 7
        assert "Concentration" in df.columns

    def test_calib_params_missing(self, empty_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        assert generate_table("tbl_calib_params", empty_state) is None


class TestIsothermTables:
    """Test isotherm table generators."""

    def test_iso_data_table(self, isotherm_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        df = generate_table("tbl_iso_data", isotherm_state)
        assert isinstance(df, pd.DataFrame)

    def test_iso_params_table(self, isotherm_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        df = generate_table("tbl_iso_params", isotherm_state)
        assert isinstance(df, pd.DataFrame)
        assert "Model" in df.columns
        assert "Parameter" in df.columns
        assert "Value" in df.columns
        # Unconverged model should NOT appear
        assert "UnconvergedModel" not in df["Model"].values

    def test_iso_params_has_ci(self, isotherm_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        df = generate_table("tbl_iso_params", isotherm_state)
        assert "CI_Lower" in df.columns
        assert "CI_Upper" in df.columns

    def test_iso_comparison_table(self, isotherm_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        df = generate_table("tbl_iso_comparison", isotherm_state)
        assert isinstance(df, pd.DataFrame)
        assert "Model" in df.columns
        assert "R²" in df.columns
        assert "RMSE" in df.columns
        # Only converged models
        assert "UnconvergedModel" not in df["Model"].values

    def test_iso_comparison_empty(self, empty_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        assert generate_table("tbl_iso_comparison", empty_state) is None

    def test_iso_params_all_unconverged(self):
        from adsorblab_pro.tabs.report_tab import generate_table

        state = {
            "isotherm_models_fitted": {
                "Langmuir": {"converged": False},
                "Freundlich": {"converged": False},
            }
        }
        assert generate_table("tbl_iso_params", state) is None


class TestKineticTables:
    """Test kinetic table generators."""

    def test_kin_data_table(self, kinetic_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        df = generate_table("tbl_kin_data", kinetic_state)
        assert isinstance(df, pd.DataFrame)

    def test_kin_params_table(self, kinetic_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        df = generate_table("tbl_kin_params", kinetic_state)
        assert isinstance(df, pd.DataFrame)
        assert "Model" in df.columns
        assert "Parameter" in df.columns
        models_present = set(df["Model"])
        assert "PFO" in models_present
        assert "PSO" in models_present

    def test_kin_comparison_table(self, kinetic_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        df = generate_table("tbl_kin_comparison", kinetic_state)
        assert isinstance(df, pd.DataFrame)
        assert "R²" in df.columns

    def test_kin_params_missing(self, empty_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        assert generate_table("tbl_kin_params", empty_state) is None


class TestThermodynamicTables:
    """Test thermodynamic table generators."""

    def test_thermo_params_table(self, thermo_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        df = generate_table("tbl_thermo_params", thermo_state)
        assert isinstance(df, pd.DataFrame)
        params = set(df["Parameter"])
        assert "ΔH°" in params
        assert "ΔS°" in params
        # Should have ΔG° rows for each temperature
        delta_g_rows = df[df["Parameter"].str.startswith("ΔG°")]
        assert len(delta_g_rows) == 4

    def test_thermo_params_missing(self, empty_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        assert generate_table("tbl_thermo_params", empty_state) is None

    def test_thermo_data_table(self, thermo_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        # No temp_effect_results in thermo_state → None
        assert generate_table("tbl_thermo_data", thermo_state) is None


class TestEffectDataTables:
    """Test effect study data table generators."""

    def test_ph_data_table(self, effect_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        df = generate_table("tbl_ph_data", effect_state)
        assert isinstance(df, pd.DataFrame)
        assert "pH" in df.columns

    def test_temp_data_table(self, effect_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        df = generate_table("tbl_temp_data", effect_state)
        assert isinstance(df, pd.DataFrame)

    def test_dosage_data_table(self, effect_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        df = generate_table("tbl_dosage_data", effect_state)
        assert isinstance(df, pd.DataFrame)
        assert "Mass_g" in df.columns

    def test_all_effect_tables_missing(self, empty_state):
        from adsorblab_pro.tabs.report_tab import generate_table

        for tbl_id in ("tbl_ph_data", "tbl_temp_data", "tbl_dosage_data"):
            assert generate_table(tbl_id, empty_state) is None


# =============================================================================
# MULTI-STUDY TABLE GENERATORS (require patched st.session_state)
# =============================================================================


class TestMultiStudyTables:
    """Test multi-study table generators with mocked session state."""

    @pytest.fixture
    def mock_studies(self, isotherm_state, kinetic_state, thermo_state):
        """Two studies for multi-study comparison."""
        study_a = {}
        study_a.update(isotherm_state)
        study_a.update(kinetic_state)
        study_a.update(thermo_state)

        # Slightly different params for study B
        study_b = {}
        study_b.update(isotherm_state)
        study_b["isotherm_models_fitted"] = {
            "Langmuir": {
                "converged": True,
                "params": {"qm": 35.0, "KL": 0.20},
                "r_squared": 0.978,
                "adj_r_squared": 0.972,
                "rmse": 1.0,
                "aicc": 14.0,
            },
        }
        study_b.update(kinetic_state)
        study_b.update(thermo_state)

        return {"Study A": study_a, "Study B": study_b}

    def test_multi_iso_params_with_mock(self, mock_studies):
        from unittest.mock import patch

        from adsorblab_pro.tabs.report_tab import _gen_tbl_multi_iso_params

        with patch(
            "adsorblab_pro.tabs.report_tab._get_all_studies",
            return_value=(mock_studies, list(mock_studies.keys())),
        ):
            df = _gen_tbl_multi_iso_params({})
            assert isinstance(df, pd.DataFrame)
            assert "Study" in df.columns
            assert "Model" in df.columns
            assert len(df) > 0

    def test_multi_kin_params_with_mock(self, mock_studies):
        from unittest.mock import patch

        from adsorblab_pro.tabs.report_tab import _gen_tbl_multi_kin_params

        with patch(
            "adsorblab_pro.tabs.report_tab._get_all_studies",
            return_value=(mock_studies, list(mock_studies.keys())),
        ):
            df = _gen_tbl_multi_kin_params({})
            assert isinstance(df, pd.DataFrame)
            assert "Study" in df.columns

    def test_multi_iso_returns_none_single_study(self, isotherm_state):
        from unittest.mock import patch

        from adsorblab_pro.tabs.report_tab import _gen_tbl_multi_iso_params

        single = {"Study A": isotherm_state}
        with patch(
            "adsorblab_pro.tabs.report_tab._get_all_studies",
            return_value=(single, ["Study A"]),
        ):
            assert _gen_tbl_multi_iso_params({}) is None


# =============================================================================
# get_saved_3d_figures
# =============================================================================


class TestGetSaved3DFigures:
    """Test report_tab.get_saved_3d_figures."""

    def test_no_saved_figures(self):
        from adsorblab_pro.tabs.report_tab import get_saved_3d_figures

        result = get_saved_3d_figures({})
        assert result == []

    def test_with_saved_figures(self):
        from adsorblab_pro.tabs.report_tab import get_saved_3d_figures

        state = {
            "saved_3d_figures": {
                "fig_1": {"title": "Surface A", "figure": {}, "params": {}},
                "fig_2": {"title": "Surface B", "figure": {}, "params": {}},
            }
        }
        result = get_saved_3d_figures(state)
        assert len(result) == 2
        # Each entry is a tuple (unique_id, title, description)
        assert all(isinstance(r, tuple) and len(r) == 3 for r in result)
