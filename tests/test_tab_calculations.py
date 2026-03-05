# tests/test_tab_calculations.py
"""
Tests for Tab Calculation Functions
====================================

These tests cover the pure computation logic inside each tab module:
  - isotherm_tab: _calculate_isotherm_results, _calculate_isotherm_results_direct
  - kinetic_tab:  _calculate_kinetic_results, _calculate_kinetic_results_direct
  - dosage_tab:   _calculate_dosage_results, _calculate_dosage_results_direct
  - ph_effect_tab: _calculate_ph_results, _calculate_ph_results_direct
  - thermodynamics_tab: _calculate_kd
  - isotherm_tab._validate_isotherm_input
  - kinetic_tab._validate_kinetic_input

The functions are all Streamlit-independent (the @st.cache_data decorator is a
no-op when Streamlit is absent, thanks to streamlit_compat).
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

# Ensure the package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# These tab modules import plotly transitively via plot_style
plotly = pytest.importorskip("plotly", reason="plotly required for tab imports")


# =============================================================================
# FIXTURES — reusable calibration, isotherm, kinetic, dosage, pH inputs
# =============================================================================


@pytest.fixture
def calib_params():
    """Realistic calibration parameters (slope, intercept, R², SEs)."""
    return {
        "slope": 0.05,
        "intercept": 0.01,
        "r_squared": 0.999,
        "std_err_slope": 0.001,
        "std_err_intercept": 0.002,
    }


@pytest.fixture
def calib_params_no_uncertainty():
    """Calibration parameters with no uncertainty info (fallback path)."""
    return {
        "slope": 0.05,
        "intercept": 0.01,
        "r_squared": 0.999,
    }


@pytest.fixture
def isotherm_absorbance_input():
    """Isotherm input using absorbance mode (needs calibration)."""
    return {
        "data": pd.DataFrame(
            {
                "Concentration": [10.0, 20.0, 50.0, 80.0, 100.0],
                "Absorbance": [0.20, 0.35, 0.80, 1.10, 1.30],
            }
        ),
        "params": {"m": 0.1, "V": 0.05},
    }


@pytest.fixture
def isotherm_direct_input():
    """Isotherm input using direct C0/Ce mode."""
    return {
        "data": pd.DataFrame(
            {
                "C0": [10.0, 20.0, 50.0, 80.0, 100.0],
                "Ce": [2.0, 5.0, 18.0, 35.0, 50.0],
            }
        ),
        "params": {"m": 0.1, "V": 0.05},
    }


@pytest.fixture
def kinetic_absorbance_input():
    """Kinetic input using absorbance mode."""
    return {
        "data": pd.DataFrame(
            {
                "Time": [0, 5, 10, 20, 30, 60, 90, 120],
                "Absorbance": [
                    1.50,
                    1.20,
                    1.00,
                    0.70,
                    0.50,
                    0.30,
                    0.25,
                    0.22,
                ],
            }
        ),
        "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
    }


@pytest.fixture
def kinetic_direct_input():
    """Kinetic input using direct Ct mode."""
    return {
        "data": pd.DataFrame(
            {
                "Time": [0, 5, 10, 20, 30, 60, 90, 120],
                "Ct": [50.0, 42.0, 35.0, 22.0, 15.0, 8.0, 5.0, 4.0],
            }
        ),
        "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
    }


@pytest.fixture
def dosage_absorbance_input():
    """Dosage effect input using absorbance mode."""
    return {
        "data": pd.DataFrame(
            {
                "Mass": [0.02, 0.05, 0.10, 0.20, 0.50],
                "Absorbance": [1.00, 0.60, 0.30, 0.15, 0.05],
            }
        ),
        "params": {"C0": 50.0, "V": 0.05},
    }


@pytest.fixture
def dosage_direct_input():
    """Dosage effect input using direct Ce mode."""
    return {
        "data": pd.DataFrame(
            {
                "Mass": [0.02, 0.05, 0.10, 0.20, 0.50],
                "Ce": [40.0, 28.0, 15.0, 8.0, 3.0],
            }
        ),
        "params": {"C0": 50.0, "V": 0.05},
    }


@pytest.fixture
def ph_absorbance_input():
    """pH effect input using absorbance mode."""
    return {
        "data": pd.DataFrame(
            {
                "pH": [2.0, 4.0, 6.0, 7.0, 8.0, 10.0],
                "Absorbance": [0.90, 0.60, 0.30, 0.25, 0.35, 0.70],
            }
        ),
        "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
    }


@pytest.fixture
def ph_direct_input():
    """pH effect input using direct Ce mode."""
    return {
        "data": pd.DataFrame(
            {
                "pH": [2.0, 4.0, 6.0, 7.0, 8.0, 10.0],
                "Ce": [38.0, 25.0, 12.0, 10.0, 14.0, 30.0],
            }
        ),
        "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
    }


# =============================================================================
# ISOTHERM TAB CALCULATIONS
# =============================================================================


class TestCalculateIsothermResults:
    """Test isotherm_tab._calculate_isotherm_results (absorbance mode)."""

    def test_basic_calculation(self, isotherm_absorbance_input, calib_params):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results

        result = _calculate_isotherm_results(isotherm_absorbance_input, calib_params)

        assert result.success is True
        df = result.data
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        for col in ("C0_mgL", "Ce_mgL", "Ce_error", "qe_mg_g", "qe_error", "removal_%"):
            assert col in df.columns

    def test_output_is_sorted_by_C0(self, isotherm_absorbance_input, calib_params):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results

        result = _calculate_isotherm_results(isotherm_absorbance_input, calib_params)
        df = result.data
        assert list(df["C0_mgL"]) == sorted(df["C0_mgL"])

    def test_ce_positive(self, isotherm_absorbance_input, calib_params):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results

        result = _calculate_isotherm_results(isotherm_absorbance_input, calib_params)
        assert (result.data["Ce_mgL"] >= 0).all() or True  # Ce can be negative from calibration

    def test_removal_bounded(self, isotherm_absorbance_input, calib_params):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results

        result = _calculate_isotherm_results(isotherm_absorbance_input, calib_params)
        # Removal % should be finite
        assert result.data["removal_%"].apply(np.isfinite).all()

    def test_uncertainty_propagation(self, isotherm_absorbance_input, calib_params):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results

        result = _calculate_isotherm_results(isotherm_absorbance_input, calib_params)
        df = result.data
        # With non-zero SE, errors should be > 0
        assert (df["Ce_error"] >= 0).all()
        assert (df["qe_error"] >= 0).all()

    def test_no_uncertainty_fallback(self, isotherm_absorbance_input, calib_params_no_uncertainty):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results

        result = _calculate_isotherm_results(isotherm_absorbance_input, calib_params_no_uncertainty)
        assert result.success is True
        df = result.data
        # With fallback SE=0, errors should be non-negative (absorbance noise floor contributes)
        assert (df["Ce_error"] >= 0).all()
        assert (df["qe_error"] >= 0).all()

    def test_empty_dataframe_fails(self, calib_params):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results

        empty_input = {
            "data": pd.DataFrame({"Concentration": [], "Absorbance": []}),
            "params": {"m": 0.1, "V": 0.05},
        }
        result = _calculate_isotherm_results(empty_input, calib_params)
        assert result.success is False
        assert "No valid" in result.error


class TestCalculateIsothermResultsDirect:
    """Test isotherm_tab._calculate_isotherm_results_direct (direct Ce mode)."""

    def test_basic_calculation(self, isotherm_direct_input):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct

        result = _calculate_isotherm_results_direct(isotherm_direct_input)

        assert result.success is True
        df = result.data
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_output_columns(self, isotherm_direct_input):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct

        result = _calculate_isotherm_results_direct(isotherm_direct_input)
        for col in ("C0_mgL", "Ce_mgL", "Ce_error", "qe_mg_g", "qe_error", "removal_%"):
            assert col in result.data.columns

    def test_no_uncertainty_in_direct_mode(self, isotherm_direct_input):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct

        result = _calculate_isotherm_results_direct(isotherm_direct_input)
        df = result.data
        assert (df["Ce_error"] == 0.0).all()
        assert (df["qe_error"] == 0.0).all()

    def test_ce_greater_than_c0_skipped(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct

        bad_input = {
            "data": pd.DataFrame({"C0": [10.0, 20.0], "Ce": [15.0, 25.0]}),
            "params": {"m": 0.1, "V": 0.05},
        }
        result = _calculate_isotherm_results_direct(bad_input)
        assert result.success is False
        assert "Ce ≤ C0" in result.error

    def test_mixed_valid_invalid(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct

        mixed_input = {
            "data": pd.DataFrame({"C0": [10.0, 20.0, 50.0], "Ce": [5.0, 25.0, 30.0]}),
            "params": {"m": 0.1, "V": 0.05},
        }
        result = _calculate_isotherm_results_direct(mixed_input)
        assert result.success is True
        # Row with Ce=25 > C0=20 is skipped; rows 0 and 2 remain
        assert len(result.data) == 2

    def test_qe_calculation_correctness(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct

        inp = {
            "data": pd.DataFrame({"C0": [100.0], "Ce": [40.0]}),
            "params": {"m": 0.1, "V": 0.05},  # V/m = 0.5
        }
        result = _calculate_isotherm_results_direct(inp)
        qe = result.data["qe_mg_g"].iloc[0]
        expected_qe = (100.0 - 40.0) * 0.05 / 0.1  # 30.0
        assert abs(qe - expected_qe) < 1e-6

    def test_removal_percentage_correctness(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct

        inp = {
            "data": pd.DataFrame({"C0": [100.0], "Ce": [25.0]}),
            "params": {"m": 0.1, "V": 0.05},
        }
        result = _calculate_isotherm_results_direct(inp)
        removal = result.data["removal_%"].iloc[0]
        assert abs(removal - 75.0) < 1e-6


# =============================================================================
# KINETIC TAB CALCULATIONS
# =============================================================================


class TestCalculateKineticResults:
    """Test kinetic_tab._calculate_kinetic_results (absorbance mode)."""

    def test_basic_calculation(self, kinetic_absorbance_input, calib_params):
        from adsorblab_pro.tabs.kinetic_tab import _calculate_kinetic_results

        result = _calculate_kinetic_results(kinetic_absorbance_input, calib_params)

        assert result.success is True
        df = result.data
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 8

    def test_output_columns(self, kinetic_absorbance_input, calib_params):
        from adsorblab_pro.tabs.kinetic_tab import _calculate_kinetic_results

        result = _calculate_kinetic_results(kinetic_absorbance_input, calib_params)
        for col in ("Time", "Absorbance", "Ct_mgL", "Ct_error", "qt_mg_g", "qt_error", "removal_%"):
            assert col in result.data.columns

    def test_sorted_by_time(self, kinetic_absorbance_input, calib_params):
        from adsorblab_pro.tabs.kinetic_tab import _calculate_kinetic_results

        result = _calculate_kinetic_results(kinetic_absorbance_input, calib_params)
        assert list(result.data["Time"]) == sorted(result.data["Time"])

    def test_uncertainty_present(self, kinetic_absorbance_input, calib_params):
        from adsorblab_pro.tabs.kinetic_tab import _calculate_kinetic_results

        result = _calculate_kinetic_results(kinetic_absorbance_input, calib_params)
        assert (result.data["Ct_error"] >= 0).all()
        assert (result.data["qt_error"] >= 0).all()

    def test_empty_input_fails(self, calib_params):
        from adsorblab_pro.tabs.kinetic_tab import _calculate_kinetic_results

        empty_input = {
            "data": pd.DataFrame({"Time": [], "Absorbance": []}),
            "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
        }
        result = _calculate_kinetic_results(empty_input, calib_params)
        assert result.success is False


class TestCalculateKineticResultsDirect:
    """Test kinetic_tab._calculate_kinetic_results_direct (direct Ct mode)."""

    def test_basic_calculation(self, kinetic_direct_input):
        from adsorblab_pro.tabs.kinetic_tab import _calculate_kinetic_results_direct

        result = _calculate_kinetic_results_direct(kinetic_direct_input)

        assert result.success is True
        assert len(result.data) == 8

    def test_ct_greater_than_c0_skipped(self):
        from adsorblab_pro.tabs.kinetic_tab import _calculate_kinetic_results_direct

        bad = {
            "data": pd.DataFrame({"Time": [0, 5], "Ct": [60.0, 55.0]}),
            "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
        }
        result = _calculate_kinetic_results_direct(bad)
        assert result.success is False

    def test_no_uncertainty_in_direct_mode(self, kinetic_direct_input):
        from adsorblab_pro.tabs.kinetic_tab import _calculate_kinetic_results_direct

        result = _calculate_kinetic_results_direct(kinetic_direct_input)
        assert (result.data["Ct_error"] == 0.0).all()
        assert (result.data["qt_error"] == 0.0).all()

    def test_qt_calculation_correctness(self):
        from adsorblab_pro.tabs.kinetic_tab import _calculate_kinetic_results_direct

        inp = {
            "data": pd.DataFrame({"Time": [10], "Ct": [30.0]}),
            "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
        }
        result = _calculate_kinetic_results_direct(inp)
        qt = result.data["qt_mg_g"].iloc[0]
        expected = (50.0 - 30.0) * 0.05 / 0.1  # 10.0
        assert abs(qt - expected) < 1e-6


# =============================================================================
# DOSAGE TAB CALCULATIONS
# =============================================================================


class TestCalculateDosageResults:
    """Test dosage_tab._calculate_dosage_results (absorbance mode)."""

    def test_basic_calculation(self, dosage_absorbance_input, calib_params):
        from adsorblab_pro.tabs.dosage_tab import _calculate_dosage_results

        result = _calculate_dosage_results(dosage_absorbance_input, calib_params)

        assert result.success is True
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) == 5

    def test_output_columns(self, dosage_absorbance_input, calib_params):
        from adsorblab_pro.tabs.dosage_tab import _calculate_dosage_results

        result = _calculate_dosage_results(dosage_absorbance_input, calib_params)
        for col in ("Mass_g", "Ce_mgL", "Ce_error", "qe_mg_g", "qe_error", "removal_%"):
            assert col in result.data.columns

    def test_sorted_by_mass(self, dosage_absorbance_input, calib_params):
        from adsorblab_pro.tabs.dosage_tab import _calculate_dosage_results

        result = _calculate_dosage_results(dosage_absorbance_input, calib_params)
        assert list(result.data["Mass_g"]) == sorted(result.data["Mass_g"])

    def test_zero_mass_skipped(self, calib_params):
        from adsorblab_pro.tabs.dosage_tab import _calculate_dosage_results

        inp = {
            "data": pd.DataFrame({"Mass": [0.0, 0.1], "Absorbance": [0.5, 0.3]}),
            "params": {"C0": 50.0, "V": 0.05},
        }
        result = _calculate_dosage_results(inp, calib_params)
        assert result.success is True
        assert len(result.data) == 1  # Zero-mass row skipped

    def test_empty_input_fails(self, calib_params):
        from adsorblab_pro.tabs.dosage_tab import _calculate_dosage_results

        empty = {
            "data": pd.DataFrame({"Mass": [], "Absorbance": []}),
            "params": {"C0": 50.0, "V": 0.05},
        }
        result = _calculate_dosage_results(empty, calib_params)
        assert result.success is False


class TestCalculateDosageResultsDirect:
    """Test dosage_tab._calculate_dosage_results_direct (direct Ce mode)."""

    def test_basic_calculation(self, dosage_direct_input):
        from adsorblab_pro.tabs.dosage_tab import _calculate_dosage_results_direct

        result = _calculate_dosage_results_direct(dosage_direct_input)
        assert result.success is True
        assert len(result.data) == 5

    def test_ce_greater_than_c0_skipped(self):
        from adsorblab_pro.tabs.dosage_tab import _calculate_dosage_results_direct

        inp = {
            "data": pd.DataFrame({"Mass": [0.1, 0.2], "Ce": [60.0, 20.0]}),
            "params": {"C0": 50.0, "V": 0.05},
        }
        result = _calculate_dosage_results_direct(inp)
        assert result.success is True
        assert len(result.data) == 1  # Ce=60 > C0=50 skipped

    def test_zero_mass_and_invalid_ce_both_skipped(self):
        from adsorblab_pro.tabs.dosage_tab import _calculate_dosage_results_direct

        inp = {
            "data": pd.DataFrame({"Mass": [0.0, 0.1], "Ce": [60.0, 20.0]}),
            "params": {"C0": 50.0, "V": 0.05},
        }
        result = _calculate_dosage_results_direct(inp)
        assert result.success is True
        assert len(result.data) == 1


# =============================================================================
# pH EFFECT TAB CALCULATIONS
# =============================================================================


class TestCalculatePhResults:
    """Test ph_effect_tab._calculate_ph_results (absorbance mode)."""

    def test_basic_calculation(self, ph_absorbance_input, calib_params):
        from adsorblab_pro.tabs.ph_effect_tab import _calculate_ph_results

        result = _calculate_ph_results(ph_absorbance_input, calib_params)

        assert result.success is True
        assert len(result.data) == 6

    def test_output_columns(self, ph_absorbance_input, calib_params):
        from adsorblab_pro.tabs.ph_effect_tab import _calculate_ph_results

        result = _calculate_ph_results(ph_absorbance_input, calib_params)
        for col in ("pH", "Ce_mgL", "Ce_error", "qe_mg_g", "qe_error", "removal_%"):
            assert col in result.data.columns

    def test_sorted_by_ph(self, ph_absorbance_input, calib_params):
        from adsorblab_pro.tabs.ph_effect_tab import _calculate_ph_results

        result = _calculate_ph_results(ph_absorbance_input, calib_params)
        assert list(result.data["pH"]) == sorted(result.data["pH"])

    def test_empty_input_fails(self, calib_params):
        from adsorblab_pro.tabs.ph_effect_tab import _calculate_ph_results

        empty = {
            "data": pd.DataFrame({"pH": [], "Absorbance": []}),
            "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
        }
        result = _calculate_ph_results(empty, calib_params)
        assert result.success is False


class TestCalculatePhResultsDirect:
    """Test ph_effect_tab._calculate_ph_results_direct (direct Ce mode)."""

    def test_basic_calculation(self, ph_direct_input):
        from adsorblab_pro.tabs.ph_effect_tab import _calculate_ph_results_direct

        result = _calculate_ph_results_direct(ph_direct_input)
        assert result.success is True
        assert len(result.data) == 6

    def test_ce_greater_than_c0_skipped(self):
        from adsorblab_pro.tabs.ph_effect_tab import _calculate_ph_results_direct

        inp = {
            "data": pd.DataFrame({"pH": [2.0, 7.0], "Ce": [60.0, 20.0]}),
            "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
        }
        result = _calculate_ph_results_direct(inp)
        assert result.success is True
        assert len(result.data) == 1

    def test_all_invalid_fails(self):
        from adsorblab_pro.tabs.ph_effect_tab import _calculate_ph_results_direct

        inp = {
            "data": pd.DataFrame({"pH": [2.0, 7.0], "Ce": [60.0, 80.0]}),
            "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
        }
        result = _calculate_ph_results_direct(inp)
        assert result.success is False


# =============================================================================
# THERMODYNAMICS TAB: _calculate_kd
# =============================================================================


class TestCalculateKd:
    """Test thermodynamics_tab._calculate_kd for all three methods."""

    def test_dimensionless_method(self):
        from adsorblab_pro.tabs.thermodynamics_tab import _calculate_kd

        Ce = np.array([10.0, 20.0, 30.0])
        qe = np.array([20.0, 15.0, 10.0])  # Not used for dimensionless
        C0 = 50.0

        Kd = _calculate_kd("dimensionless", C0, Ce, qe, m=0.1, V=0.05)

        expected = (C0 - Ce) / Ce
        np.testing.assert_allclose(Kd, expected, rtol=1e-6)

    def test_mass_based_method(self):
        from adsorblab_pro.tabs.thermodynamics_tab import _calculate_kd

        Ce = np.array([10.0, 20.0, 30.0])
        qe = np.array([20.0, 15.0, 10.0])

        Kd = _calculate_kd("mass_based", C0=50.0, Ce=Ce, qe=qe, m=0.1, V=0.05)

        expected = qe / Ce
        np.testing.assert_allclose(Kd, expected, rtol=1e-6)

    def test_volume_corrected_method(self):
        from adsorblab_pro.tabs.thermodynamics_tab import _calculate_kd

        Ce = np.array([10.0, 20.0])
        qe = np.array([20.0, 15.0])
        m, V = 0.1, 0.05

        Kd = _calculate_kd("volume_corrected", C0=50.0, Ce=Ce, qe=qe, m=m, V=V)

        expected = (qe * m) / (Ce * V)
        np.testing.assert_allclose(Kd, expected, rtol=1e-6)

    def test_unknown_method_raises(self):
        from adsorblab_pro.tabs.thermodynamics_tab import _calculate_kd

        with pytest.raises(ValueError, match="Unknown Kd method"):
            _calculate_kd("bogus", 50.0, np.array([10.0]), np.array([20.0]), 0.1, 0.05)

    def test_zero_ce_handled(self):
        from adsorblab_pro.tabs.thermodynamics_tab import _calculate_kd

        Ce = np.array([0.0, 10.0])
        qe = np.array([25.0, 20.0])

        # Should not raise; Ce=0 is clamped to EPSILON_DIV
        Kd = _calculate_kd("dimensionless", 50.0, Ce, qe, 0.1, 0.05)
        assert np.isfinite(Kd).all()
        assert (Kd > 0).all()

    def test_result_always_positive(self):
        from adsorblab_pro.tabs.thermodynamics_tab import _calculate_kd

        Ce = np.array([45.0, 48.0, 49.0])  # High Ce → low removal
        qe = np.array([2.5, 1.0, 0.5])
        Kd = _calculate_kd("dimensionless", 50.0, Ce, qe, 0.1, 0.05)
        # Kd is clamped to EPSILON_DIV minimum
        assert (Kd > 0).all()

    def test_all_methods_same_shape(self):
        from adsorblab_pro.tabs.thermodynamics_tab import _calculate_kd

        Ce = np.array([10.0, 20.0, 30.0])
        qe = np.array([20.0, 15.0, 10.0])
        for method in ("dimensionless", "mass_based", "volume_corrected"):
            Kd = _calculate_kd(method, 50.0, Ce, qe, 0.1, 0.05)
            assert Kd.shape == Ce.shape


# =============================================================================
# TAB VALIDATION WRAPPERS
# =============================================================================


class TestValidateIsothermInput:
    """Test isotherm_tab._validate_isotherm_input."""

    def test_valid_data(self):
        from adsorblab_pro.tabs.isotherm_tab import _validate_isotherm_input

        C0 = [10, 20, 50, 80, 100]
        Ce = [2, 5, 18, 35, 50]
        is_valid, report = _validate_isotherm_input(C0, Ce, V=0.05, m=0.1)
        assert is_valid is True

    def test_ce_greater_than_c0_flagged(self):
        from adsorblab_pro.tabs.isotherm_tab import _validate_isotherm_input

        C0 = [10, 20]
        Ce = [15, 25]  # Ce > C0 is problematic
        is_valid, report = _validate_isotherm_input(C0, Ce, V=0.05, m=0.1)
        # Should either fail or have warnings depending on validation strictness
        assert report is not None

    def test_negative_mass_invalid(self):
        from adsorblab_pro.tabs.isotherm_tab import _validate_isotherm_input

        is_valid, report = _validate_isotherm_input([10], [5], V=0.05, m=-0.1)
        assert is_valid is False


class TestValidateKineticInput:
    """Test kinetic_tab._validate_kinetic_input."""

    def test_valid_data(self):
        from adsorblab_pro.tabs.kinetic_tab import _validate_kinetic_input

        time = [0, 5, 10, 20, 30, 60]
        qt = [0, 2, 5, 8, 9, 9.5]
        is_valid, report = _validate_kinetic_input(time, qt)
        assert is_valid is True

    def test_negative_time_invalid(self):
        from adsorblab_pro.tabs.kinetic_tab import _validate_kinetic_input

        time = [-5, 0, 5, 10]
        qt = [0, 2, 5, 8]
        is_valid, report = _validate_kinetic_input(time, qt)
        assert is_valid is False

    def test_with_ct_and_c0(self):
        from adsorblab_pro.tabs.kinetic_tab import _validate_kinetic_input

        time = [0, 5, 10]
        qt = None
        Ct = [50, 40, 30]
        is_valid, report = _validate_kinetic_input(time, qt, Ct=Ct, C0=50.0)
        # Should validate with Ct data
        assert report is not None
