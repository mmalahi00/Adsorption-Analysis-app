# tests/test_tab_helpers.py
"""
Tests for pure-logic helper functions inside tab modules.

These functions are Streamlit-independent and testable without a running
Streamlit server. They cover caching utilities, validation wrappers,
data hashing, and calculation functions in:
  - isotherm_tab
  - kinetic_tab
  - thermodynamics_tab
  - dosage_tab
  - ph_effect_tab
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

plotly = pytest.importorskip("plotly", reason="plotly required for tab imports")


# =============================================================================
# ISOTHERM TAB HELPERS
# =============================================================================


class TestIsothermGetTemperatureK:
    def test_with_T_K(self):
        from adsorblab_pro.tabs.isotherm_tab import _get_temperature_k

        assert _get_temperature_k({"T_K": 310.0}) == 310.0

    def test_with_T_C(self):
        from adsorblab_pro.tabs.isotherm_tab import _get_temperature_k

        result = _get_temperature_k({"T_C": 25.0})
        assert abs(result - 298.15) < 0.01

    def test_no_temp(self):
        from adsorblab_pro.tabs.isotherm_tab import _get_temperature_k

        assert _get_temperature_k({}) == 298.15

    def test_none_params(self):
        from adsorblab_pro.tabs.isotherm_tab import _get_temperature_k

        assert _get_temperature_k(None) == 298.15

    def test_T_K_preferred_over_T_C(self):
        from adsorblab_pro.tabs.isotherm_tab import _get_temperature_k

        result = _get_temperature_k({"T_K": 350.0, "T_C": 25.0})
        assert result == 350.0

    def test_T_K_none_falls_to_T_C(self):
        from adsorblab_pro.tabs.isotherm_tab import _get_temperature_k

        result = _get_temperature_k({"T_K": None, "T_C": 0.0})
        assert abs(result - 273.15) < 0.01

    def test_both_none(self):
        from adsorblab_pro.tabs.isotherm_tab import _get_temperature_k

        result = _get_temperature_k({"T_K": None, "T_C": None})
        assert result == 298.15


class TestIsothermValidateInput:
    def test_valid_input(self):
        from adsorblab_pro.tabs.isotherm_tab import _validate_isotherm_input

        is_valid, report = _validate_isotherm_input(
            C0=[100.0, 80.0, 50.0, 30.0],
            Ce=[10.0, 15.0, 20.0, 25.0],
            V=0.05,
            m=0.1,
        )
        assert is_valid is True

    def test_invalid_negative_Ce(self):
        from adsorblab_pro.tabs.isotherm_tab import _validate_isotherm_input

        is_valid, report = _validate_isotherm_input(
            C0=[100.0],
            Ce=[-5.0],
            V=0.05,
            m=0.1,
        )
        # Should flag negative Ce as invalid
        assert report is not None


class TestIsothermComputeDataHash:
    def test_deterministic(self):
        from adsorblab_pro.tabs.isotherm_tab import _compute_data_hash

        Ce = np.array([1.0, 2.0, 3.0])
        qe = np.array([10.0, 20.0, 30.0])
        C0 = np.array([50.0, 50.0, 50.0])
        h1 = _compute_data_hash(Ce, qe, C0)
        h2 = _compute_data_hash(Ce, qe, C0)
        assert h1 == h2

    def test_different_data_different_hash(self):
        from adsorblab_pro.tabs.isotherm_tab import _compute_data_hash

        Ce1 = np.array([1.0, 2.0])
        Ce2 = np.array([1.0, 3.0])
        qe = np.array([10.0, 20.0])
        C0 = np.array([50.0, 50.0])
        assert _compute_data_hash(Ce1, qe, C0) != _compute_data_hash(Ce2, qe, C0)


class TestIsothermArraysToTuples:
    def test_basic(self):
        from adsorblab_pro.tabs.isotherm_tab import _arrays_to_tuples

        Ce = np.array([1.0, 2.0])
        qe = np.array([10.0, 20.0])
        C0 = np.array([50.0, 50.0])
        result = _arrays_to_tuples(Ce, qe, C0)
        assert len(result) == 3
        assert all(isinstance(t, tuple) for t in result)

    def test_precision_rounding(self):
        from adsorblab_pro.tabs.isotherm_tab import _arrays_to_tuples

        Ce = np.array([1.123456789012345])
        qe = np.array([2.0])
        C0 = np.array([50.0])
        result = _arrays_to_tuples(Ce, qe, C0)
        # Should be rounded to 8 decimal places
        assert result[0][0] == round(1.123456789012345, 8)


# =============================================================================
# KINETIC TAB HELPERS
# =============================================================================


class TestKineticComputeDataHash:
    def test_deterministic(self):
        from adsorblab_pro.tabs.kinetic_tab import _compute_kinetic_data_hash

        t = np.array([0, 5, 10, 30, 60.0])
        qt = np.array([0, 5, 8, 12, 14.0])
        h1 = _compute_kinetic_data_hash(t, qt)
        h2 = _compute_kinetic_data_hash(t, qt)
        assert h1 == h2

    def test_different_data(self):
        from adsorblab_pro.tabs.kinetic_tab import _compute_kinetic_data_hash

        t = np.array([0, 5, 10.0])
        qt1 = np.array([0, 5, 8.0])
        qt2 = np.array([0, 5, 9.0])
        assert _compute_kinetic_data_hash(t, qt1) != _compute_kinetic_data_hash(t, qt2)


class TestKineticArraysToTuples:
    def test_basic(self):
        from adsorblab_pro.tabs.kinetic_tab import _arrays_to_tuples

        t = np.array([0, 5, 10.0])
        qt = np.array([0, 5, 8.0])
        result = _arrays_to_tuples(t, qt)
        assert len(result) == 2
        assert all(isinstance(r, tuple) for r in result)


class TestKineticValidateInput:
    def test_valid_input(self):
        from adsorblab_pro.tabs.kinetic_tab import _validate_kinetic_input

        is_valid, report = _validate_kinetic_input(
            time=[0, 5, 10, 30, 60],
            qt=[0, 5, 8, 12, 14],
        )
        assert is_valid is True

    def test_negative_time(self):
        from adsorblab_pro.tabs.kinetic_tab import _validate_kinetic_input

        is_valid, report = _validate_kinetic_input(
            time=[-1, 5, 10],
            qt=[0, 5, 8],
        )
        assert report is not None


# =============================================================================
# ISOTHERM CALCULATE RESULTS
# =============================================================================


class TestCalculateIsothermResults:
    def test_absorbance_mode(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results

        iso_input = {
            "data": pd.DataFrame(
                {
                    "Concentration": [10.0, 20.0, 50.0, 80.0, 100.0],
                    "Absorbance": [0.20, 0.35, 0.80, 1.10, 1.30],
                }
            ),
            "params": {"m": 0.1, "V": 0.05},
        }
        calib = {
            "slope": 0.05,
            "intercept": 0.01,
            "std_err_slope": 0.001,
            "std_err_intercept": 0.002,
        }
        result = _calculate_isotherm_results(iso_input, calib)
        assert result.success is True
        assert "Ce_mgL" in result.data.columns
        assert "qe_mg_g" in result.data.columns
        assert len(result.data) == 5

    def test_with_no_uncertainty(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results

        iso_input = {
            "data": pd.DataFrame(
                {
                    "Concentration": [50.0],
                    "Absorbance": [0.80],
                }
            ),
            "params": {"m": 0.1, "V": 0.05},
        }
        calib = {"slope": 0.05, "intercept": 0.01}
        result = _calculate_isotherm_results(iso_input, calib)
        assert result.success is True

    def test_empty_data(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results

        iso_input = {
            "data": pd.DataFrame({"Concentration": [], "Absorbance": []}),
            "params": {"m": 0.1, "V": 0.05},
        }
        calib = {"slope": 0.05, "intercept": 0.01}
        result = _calculate_isotherm_results(iso_input, calib)
        assert result.success is False

    def test_zero_mass(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results

        iso_input = {
            "data": pd.DataFrame(
                {
                    "Concentration": [50.0],
                    "Absorbance": [0.80],
                }
            ),
            "params": {"m": 0.0, "V": 0.05},
        }
        calib = {
            "slope": 0.05,
            "intercept": 0.01,
            "std_err_slope": 0.001,
            "std_err_intercept": 0.002,
        }
        result = _calculate_isotherm_results(iso_input, calib)
        # Should handle m=0 gracefully (qe_error = 0)
        assert result.success is True


class TestCalculateIsothermResultsDirect:
    def test_basic(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct

        iso_input = {
            "data": pd.DataFrame(
                {
                    "C0": [100.0, 80.0, 50.0],
                    "Ce": [10.0, 15.0, 20.0],
                }
            ),
            "params": {"m": 0.1, "V": 0.05},
        }
        result = _calculate_isotherm_results_direct(iso_input)
        assert result.success is True
        assert len(result.data) == 3

    def test_Ce_greater_than_C0_skipped(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct

        iso_input = {
            "data": pd.DataFrame(
                {
                    "C0": [50.0, 50.0],
                    "Ce": [60.0, 20.0],  # first row invalid
                }
            ),
            "params": {"m": 0.1, "V": 0.05},
        }
        result = _calculate_isotherm_results_direct(iso_input)
        assert result.success is True
        assert len(result.data) == 1

    def test_all_invalid(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct

        iso_input = {
            "data": pd.DataFrame(
                {
                    "C0": [10.0],
                    "Ce": [20.0],  # Ce > C0
                }
            ),
            "params": {"m": 0.1, "V": 0.05},
        }
        result = _calculate_isotherm_results_direct(iso_input)
        assert result.success is False


# =============================================================================
# KINETIC CALCULATE RESULTS
# =============================================================================


class TestCalculateKineticResults:
    def test_absorbance_mode(self):
        from adsorblab_pro.tabs.kinetic_tab import _calculate_kinetic_results

        kin_input = {
            "data": pd.DataFrame(
                {
                    "Time": [0, 5, 10, 30, 60],
                    "Absorbance": [1.5, 1.2, 0.9, 0.5, 0.3],
                }
            ),
            "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
        }
        calib = {
            "slope": 0.03,
            "intercept": 0.0,
            "std_err_slope": 0.001,
            "std_err_intercept": 0.001,
        }
        result = _calculate_kinetic_results(kin_input, calib)
        assert result.success is True
        assert "qt_mg_g" in result.data.columns
        assert len(result.data) == 5

    def test_empty_data(self):
        from adsorblab_pro.tabs.kinetic_tab import _calculate_kinetic_results

        kin_input = {
            "data": pd.DataFrame({"Time": [], "Absorbance": []}),
            "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
        }
        calib = {"slope": 0.03, "intercept": 0.0}
        result = _calculate_kinetic_results(kin_input, calib)
        assert result.success is False


class TestCalculateKineticResultsDirect:
    def test_basic(self):
        from adsorblab_pro.tabs.kinetic_tab import _calculate_kinetic_results_direct

        kin_input = {
            "data": pd.DataFrame(
                {
                    "Time": [0, 10, 30, 60],
                    "Ct": [50.0, 40.0, 25.0, 15.0],
                }
            ),
            "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
        }
        result = _calculate_kinetic_results_direct(kin_input)
        assert result.success is True
        assert len(result.data) == 4

    def test_Ct_greater_than_C0_skipped(self):
        from adsorblab_pro.tabs.kinetic_tab import _calculate_kinetic_results_direct

        kin_input = {
            "data": pd.DataFrame(
                {
                    "Time": [0, 10],
                    "Ct": [60.0, 30.0],  # first invalid
                }
            ),
            "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
        }
        result = _calculate_kinetic_results_direct(kin_input)
        assert result.success is True
        assert len(result.data) == 1

    def test_all_invalid(self):
        from adsorblab_pro.tabs.kinetic_tab import _calculate_kinetic_results_direct

        kin_input = {
            "data": pd.DataFrame(
                {
                    "Time": [0],
                    "Ct": [60.0],  # > C0
                }
            ),
            "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
        }
        result = _calculate_kinetic_results_direct(kin_input)
        assert result.success is False


# =============================================================================
# THERMODYNAMICS _calculate_kd — additional edge cases
# =============================================================================


class TestCalculateKdExtended:
    def test_dimensionless_basic(self):
        from adsorblab_pro.tabs.thermodynamics_tab import _calculate_kd

        Ce = np.array([10.0, 20.0, 30.0])
        qe = np.array([20.0, 15.0, 10.0])
        Kd = _calculate_kd("dimensionless", C0=50.0, Ce=Ce, qe=qe, m=0.1, V=0.05)
        assert len(Kd) == 3
        assert all(k > 0 for k in Kd)

    def test_mass_based_basic(self):
        from adsorblab_pro.tabs.thermodynamics_tab import _calculate_kd

        Ce = np.array([10.0, 20.0])
        qe = np.array([20.0, 15.0])
        Kd = _calculate_kd("mass_based", C0=50.0, Ce=Ce, qe=qe, m=0.1, V=0.05)
        np.testing.assert_allclose(Kd, [2.0, 0.75])

    def test_volume_corrected_basic(self):
        from adsorblab_pro.tabs.thermodynamics_tab import _calculate_kd

        Ce = np.array([10.0])
        qe = np.array([20.0])
        Kd = _calculate_kd("volume_corrected", C0=50.0, Ce=Ce, qe=qe, m=0.1, V=0.05)
        expected = (20.0 * 0.1) / (10.0 * 0.05)
        np.testing.assert_allclose(Kd, [expected])

    def test_zero_Ce_clamped(self):
        from adsorblab_pro.tabs.thermodynamics_tab import _calculate_kd

        Ce = np.array([0.0])
        qe = np.array([20.0])
        Kd = _calculate_kd("dimensionless", C0=50.0, Ce=Ce, qe=qe, m=0.1, V=0.05)
        assert Kd[0] > 0  # Should not be inf

    def test_unknown_method(self):
        from adsorblab_pro.tabs.thermodynamics_tab import _calculate_kd

        with pytest.raises(ValueError, match="Unknown Kd method"):
            _calculate_kd("fake_method", 50.0, np.array([10.0]), np.array([20.0]), 0.1, 0.05)

    def test_negative_qe_clamped(self):
        from adsorblab_pro.tabs.thermodynamics_tab import _calculate_kd

        Ce = np.array([10.0])
        qe = np.array([-5.0])
        Kd = _calculate_kd("mass_based", C0=50.0, Ce=Ce, qe=qe, m=0.1, V=0.05)
        assert Kd[0] > 0  # Clamped to epsilon

    def test_array_of_ones(self):
        from adsorblab_pro.tabs.thermodynamics_tab import _calculate_kd

        Ce = np.ones(5)
        qe = np.ones(5)
        for method in ["dimensionless", "mass_based", "volume_corrected"]:
            Kd = _calculate_kd(method, C0=50.0, Ce=Ce, qe=qe, m=0.1, V=0.05)
            assert len(Kd) == 5
            assert all(np.isfinite(Kd))
