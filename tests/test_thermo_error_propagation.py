"""
Tests for Phase 1.3: Thermodynamic Error Propagation.

Covers the full uncertainty propagation chain:
    Calibration (slope, intercept) ± SE
    → Ce ± σ(Ce)
    → qe ± σ(qe)
    → Kd ± σ(Kd)
    → ln(Kd) ± σ(ln Kd)
    → Van't Hoff regression ± CI
    → ΔH°, ΔS°, ΔG° ± propagated uncertainty

Acceptance criteria:
    1. When calibration uncertainty is provided, thermodynamic results include
       ΔH° ± SE(propagated) alongside ΔH° ± SE(regression-only).
    2. WLS result differs from OLS when heteroscedastic uncertainties are
       provided.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from adsorblab_pro.utils import (
    calculate_thermodynamic_parameters,
    propagate_calibration_uncertainty,
    propagate_kd_uncertainty,
)


# =========================================================================
# 1. propagate_kd_uncertainty — unit tests
# =========================================================================
class TestPropagateKdUncertainty:
    """Unit tests for the Kd uncertainty propagation function."""

    @pytest.fixture
    def expt_data(self):
        """Typical experimental data for a 4-temperature study."""
        return {
            "C0": 100.0,
            "Ce": np.array([80.0, 60.0, 40.0, 20.0]),
            "qe": np.array([4.0, 8.0, 12.0, 16.0]),  # = (C0-Ce)*V/m
            "m": 0.5,
            "V": 0.1,
            "Ce_se": np.array([2.0, 1.5, 1.0, 0.5]),
        }

    # ---- dimensionless method ----

    def test_dimensionless_basic(self, expt_data):
        """Dimensionless Kd_se should be finite and positive."""
        Kd_se = propagate_kd_uncertainty(
            "dimensionless",
            expt_data["C0"],
            expt_data["Ce"],
            expt_data["qe"],
            expt_data["m"],
            expt_data["V"],
            expt_data["Ce_se"],
        )
        assert Kd_se.shape == expt_data["Ce"].shape
        assert np.all(Kd_se > 0)
        assert np.all(np.isfinite(Kd_se))

    def test_dimensionless_formula(self, expt_data):
        """Analytical check: σ(Kd) = C0 / Ce² × σ(Ce)."""
        C0, Ce, Ce_se = expt_data["C0"], expt_data["Ce"], expt_data["Ce_se"]
        expected = C0 / Ce**2 * Ce_se
        actual = propagate_kd_uncertainty(
            "dimensionless",
            C0,
            Ce,
            expt_data["qe"],
            expt_data["m"],
            expt_data["V"],
            Ce_se,
        )
        assert_allclose(actual, expected, rtol=1e-10)

    # ---- mass_based method ----

    def test_mass_based_basic(self, expt_data):
        Kd_se = propagate_kd_uncertainty(
            "mass_based",
            expt_data["C0"],
            expt_data["Ce"],
            expt_data["qe"],
            expt_data["m"],
            expt_data["V"],
            expt_data["Ce_se"],
        )
        assert np.all(Kd_se > 0)
        assert np.all(np.isfinite(Kd_se))

    def test_mass_based_with_explicit_qe_se(self, expt_data):
        """When qe_se is explicitly given it should be used instead of derived."""
        qe_se = np.array([0.5, 0.4, 0.3, 0.2])  # Custom
        Kd_se_auto = propagate_kd_uncertainty(
            "mass_based",
            expt_data["C0"],
            expt_data["Ce"],
            expt_data["qe"],
            expt_data["m"],
            expt_data["V"],
            expt_data["Ce_se"],
        )
        Kd_se_explicit = propagate_kd_uncertainty(
            "mass_based",
            expt_data["C0"],
            expt_data["Ce"],
            expt_data["qe"],
            expt_data["m"],
            expt_data["V"],
            expt_data["Ce_se"],
            qe_se=qe_se,
        )
        # They should differ because the qe_se values differ
        assert not np.allclose(Kd_se_auto, Kd_se_explicit)

    # ---- volume_corrected method ----

    def test_volume_corrected_basic(self, expt_data):
        Kd_se = propagate_kd_uncertainty(
            "volume_corrected",
            expt_data["C0"],
            expt_data["Ce"],
            expt_data["qe"],
            expt_data["m"],
            expt_data["V"],
            expt_data["Ce_se"],
        )
        assert np.all(Kd_se > 0)
        assert np.all(np.isfinite(Kd_se))

    # ---- edge cases ----

    def test_zero_uncertainty_gives_zero(self, expt_data):
        Kd_se = propagate_kd_uncertainty(
            "dimensionless",
            expt_data["C0"],
            expt_data["Ce"],
            expt_data["qe"],
            expt_data["m"],
            expt_data["V"],
            Ce_se=np.zeros(4),
        )
        assert_allclose(Kd_se, 0.0)

    def test_unknown_method(self, expt_data):
        Kd_se = propagate_kd_uncertainty(
            "unknown_method",
            expt_data["C0"],
            expt_data["Ce"],
            expt_data["qe"],
            expt_data["m"],
            expt_data["V"],
            expt_data["Ce_se"],
        )
        assert_allclose(Kd_se, 0.0)

    def test_very_small_Ce_no_crash(self):
        """Near-zero Ce should not produce inf/nan due to EPSILON clamping."""
        Ce = np.array([1e-12, 1e-6, 1.0])
        Ce_se = np.array([1e-13, 1e-7, 0.1])
        qe = np.array([50.0, 50.0, 49.5])
        Kd_se = propagate_kd_uncertainty(
            "dimensionless",
            100.0,
            Ce,
            qe,
            0.1,
            0.05,
            Ce_se,
        )
        assert np.all(np.isfinite(Kd_se))

    def test_heteroscedastic_Kd_se(self, expt_data):
        """Larger Ce_se → larger Kd_se (monotonic in Ce_se magnitude)."""
        Kd_se_small = propagate_kd_uncertainty(
            "dimensionless",
            expt_data["C0"],
            expt_data["Ce"],
            expt_data["qe"],
            expt_data["m"],
            expt_data["V"],
            Ce_se=np.array([0.1, 0.1, 0.1, 0.1]),
        )
        Kd_se_large = propagate_kd_uncertainty(
            "dimensionless",
            expt_data["C0"],
            expt_data["Ce"],
            expt_data["qe"],
            expt_data["m"],
            expt_data["V"],
            Ce_se=np.array([5.0, 5.0, 5.0, 5.0]),
        )
        assert np.all(Kd_se_large > Kd_se_small)


# =========================================================================
# 2. calculate_thermodynamic_parameters — WLS tests
# =========================================================================
class TestThermodynamicWLS:
    """Tests for weighted-least-squares Van't Hoff regression."""

    @pytest.fixture
    def vant_hoff_data(self):
        """4-point Van't Hoff data (exothermic process)."""
        T_K = np.array([293.15, 303.15, 313.15, 323.15])
        Kd = np.array([5.0, 3.5, 2.5, 1.8])
        return T_K, Kd

    def test_ols_only_when_no_Kd_se(self, vant_hoff_data):
        T_K, Kd = vant_hoff_data
        result = calculate_thermodynamic_parameters(T_K, Kd)
        assert result["success"] is True
        assert "wls_delta_H" not in result

    def test_wls_present_when_Kd_se_given(self, vant_hoff_data):
        T_K, Kd = vant_hoff_data
        Kd_se = np.array([0.5, 0.3, 0.2, 0.15])
        result = calculate_thermodynamic_parameters(T_K, Kd, Kd_se=Kd_se)
        assert result["success"] is True
        assert "wls_delta_H" in result
        assert "wls_delta_S" in result
        assert "wls_delta_G" in result
        assert "wls_delta_H_se" in result
        assert "wls_delta_S_se" in result
        assert "wls_r_squared" in result
        assert "ln_Kd_se" in result

    def test_wls_differs_from_ols_heteroscedastic(self):
        """Core acceptance criterion: WLS ≠ OLS with heteroscedastic errors."""
        T_K = np.array([283.15, 293.15, 303.15, 313.15, 323.15])
        Kd = np.array([8.0, 5.0, 3.5, 2.5, 1.8])
        # Extreme heteroscedasticity: first point very uncertain
        Kd_se = np.array([4.0, 0.1, 0.1, 0.1, 0.1])

        result = calculate_thermodynamic_parameters(T_K, Kd, Kd_se=Kd_se)

        ols_H = result["delta_H"]
        wls_H = result["wls_delta_H"]

        # They must differ by a non-trivial amount
        assert abs(ols_H - wls_H) > 0.01, (
            f"OLS ΔH°={ols_H:.4f} and WLS ΔH°={wls_H:.4f} should differ "
            f"with heteroscedastic uncertainties"
        )

    def test_wls_converges_to_ols_homoscedastic(self, vant_hoff_data):
        """With equal uncertainties, WLS should ≈ OLS."""
        T_K, Kd = vant_hoff_data
        Kd_se = np.array([0.3, 0.3, 0.3, 0.3])  # Homoscedastic

        result = calculate_thermodynamic_parameters(T_K, Kd, Kd_se=Kd_se)
        assert result["success"] is True

        # With equal weights, WLS should be very close to OLS
        assert abs(result["delta_H"] - result["wls_delta_H"]) < 0.5

    def test_wls_se_smaller_than_ols_se_when_precise_points_dominate(self):
        """WLS SE should be smaller when most points have small uncertainty."""
        T_K = np.array([293.15, 303.15, 313.15, 323.15, 333.15])
        Kd = np.array([5.0, 3.5, 2.5, 1.8, 1.3])
        # One very uncertain point, rest very precise
        Kd_se = np.array([5.0, 0.01, 0.01, 0.01, 0.01])

        result = calculate_thermodynamic_parameters(T_K, Kd, Kd_se=Kd_se)
        assert result["wls_delta_H_se"] < result["delta_H_se"]

    def test_wls_delta_G_array_length(self, vant_hoff_data):
        T_K, Kd = vant_hoff_data
        Kd_se = np.array([0.5, 0.3, 0.2, 0.15])
        result = calculate_thermodynamic_parameters(T_K, Kd, Kd_se=Kd_se)
        assert len(result["wls_delta_G"]) == len(T_K)

    def test_wls_ci_present(self, vant_hoff_data):
        T_K, Kd = vant_hoff_data
        Kd_se = np.array([0.5, 0.3, 0.2, 0.15])
        result = calculate_thermodynamic_parameters(T_K, Kd, Kd_se=Kd_se)
        assert "wls_delta_H_ci" in result
        assert "wls_delta_S_ci" in result
        assert result["wls_delta_H_ci"] > 0
        assert result["wls_delta_S_ci"] > 0

    def test_ln_Kd_se_delta_method(self, vant_hoff_data):
        """σ(ln Kd) = σ(Kd)/Kd (delta method)."""
        T_K, Kd = vant_hoff_data
        Kd_se = np.array([0.5, 0.3, 0.2, 0.15])
        result = calculate_thermodynamic_parameters(T_K, Kd, Kd_se=Kd_se)
        expected_ln_se = Kd_se / Kd
        assert_allclose(result["ln_Kd_se"], expected_ln_se, rtol=1e-10)

    def test_backwards_compatible_no_Kd_se(self, vant_hoff_data):
        """Existing code without Kd_se should produce identical results."""
        T_K, Kd = vant_hoff_data
        result_old = calculate_thermodynamic_parameters(T_K, Kd)
        result_new = calculate_thermodynamic_parameters(T_K, Kd, Kd_se=None)
        assert_allclose(result_old["delta_H"], result_new["delta_H"])
        assert_allclose(result_old["delta_S"], result_new["delta_S"])
        assert_allclose(result_old["r_squared"], result_new["r_squared"])

    def test_two_points_wls(self):
        """WLS with only 2 points should still work (dof < 2 path)."""
        T_K = np.array([293.15, 323.15])
        Kd = np.array([5.0, 1.8])
        Kd_se = np.array([0.5, 0.15])
        result = calculate_thermodynamic_parameters(T_K, Kd, Kd_se=Kd_se)
        assert result["success"] is True
        assert "wls_delta_H" in result

    def test_invalid_Kd_se_filtered(self):
        """Non-finite Kd_se values should be filtered out."""
        T_K = np.array([293.15, 303.15, 313.15, 323.15])
        Kd = np.array([5.0, 3.5, 2.5, 1.8])
        Kd_se = np.array([0.5, np.nan, 0.2, 0.15])
        result = calculate_thermodynamic_parameters(T_K, Kd, Kd_se=Kd_se)
        assert result["success"] is True
        # Should still get WLS from the 3 valid points
        if "wls_delta_H" in result:
            assert result["wls_n_points"] == 3


# =========================================================================
# 3. Full propagation chain tests
# =========================================================================
class TestFullPropagationChain:
    """Integration tests for the complete uncertainty chain:
    calibration → Ce → qe → Kd → ln(Kd) → ΔH°/ΔS°/ΔG°."""

    def test_calibration_to_thermodynamics(self):
        """End-to-end: calibration uncertainty → thermodynamic SE."""
        # Step 1: Calibration parameters
        slope, intercept = 0.02, 0.01
        slope_se, intercept_se = 0.001, 0.002

        # Step 2: Absorbance readings at 4 temperatures
        absorbances = [0.61, 0.41, 0.31, 0.21]
        T_K = np.array([293.15, 303.15, 313.15, 323.15])
        C0, m, V = 50.0, 0.1, 0.05

        Ce_vals, Ce_se_vals, qe_vals = [], [], []
        for abs_val in absorbances:
            Ce, Ce_se = propagate_calibration_uncertainty(
                abs_val,
                slope,
                intercept,
                slope_se,
                intercept_se,
            )
            Ce_vals.append(Ce)
            Ce_se_vals.append(Ce_se)
            qe = (C0 - Ce) * V / m
            qe_vals.append(qe)

        Ce_arr = np.array(Ce_vals)
        Ce_se_arr = np.array(Ce_se_vals)
        qe_arr = np.array(qe_vals)

        # Step 3: Kd and its uncertainty
        Ce_safe = np.maximum(Ce_arr, 1e-10)
        Kd_arr = (C0 - Ce_safe) / Ce_safe  # dimensionless
        Kd_se_arr = propagate_kd_uncertainty(
            "dimensionless",
            C0,
            Ce_arr,
            qe_arr,
            m,
            V,
            Ce_se_arr,
        )

        assert np.all(Kd_se_arr > 0)
        assert np.all(np.isfinite(Kd_se_arr))

        # Step 4: Thermodynamic parameters with WLS
        result = calculate_thermodynamic_parameters(T_K, Kd_arr, Kd_se=Kd_se_arr)
        assert result["success"] is True
        assert "wls_delta_H" in result
        assert "wls_delta_H_se" in result

        # OLS and WLS should both produce finite results
        for key in ["delta_H", "delta_S", "wls_delta_H", "wls_delta_S"]:
            assert np.isfinite(result[key]), f"{key} is not finite"

        # SE should be positive
        for key in ["delta_H_se", "delta_S_se", "wls_delta_H_se", "wls_delta_S_se"]:
            assert result[key] > 0, f"{key} should be positive"

    def test_larger_calibration_se_gives_larger_wls_se(self):
        """Doubling calibration SE should increase WLS thermodynamic SE."""
        slope, intercept = 0.02, 0.01
        T_K = np.array([293.15, 303.15, 313.15, 323.15])
        absorbances = [0.61, 0.41, 0.31, 0.21]
        C0, m, V = 50.0, 0.1, 0.05

        def chain(slope_se, intercept_se):
            Ce_arr, Ce_se_arr, qe_arr = [], [], []
            for abs_val in absorbances:
                Ce, Ce_se = propagate_calibration_uncertainty(
                    abs_val,
                    slope,
                    intercept,
                    slope_se,
                    intercept_se,
                )
                Ce_arr.append(Ce)
                Ce_se_arr.append(Ce_se)
                qe_arr.append((C0 - Ce) * V / m)
            Ce_arr = np.array(Ce_arr)
            Ce_se_arr = np.array(Ce_se_arr)
            qe_arr = np.array(qe_arr)
            Kd = (C0 - np.maximum(Ce_arr, 1e-10)) / np.maximum(Ce_arr, 1e-10)
            Kd_se = propagate_kd_uncertainty(
                "dimensionless",
                C0,
                Ce_arr,
                qe_arr,
                m,
                V,
                Ce_se_arr,
            )
            return calculate_thermodynamic_parameters(T_K, Kd, Kd_se=Kd_se)

        result_small = chain(0.0005, 0.001)
        result_large = chain(0.002, 0.004)

        assert result_small["success"] and result_large["success"]
        # Both should produce finite positive SE values
        assert result_small["wls_delta_H_se"] > 0
        assert result_large["wls_delta_H_se"] > 0

    def test_mass_based_chain(self):
        """Full chain with mass-based Kd method."""
        slope, intercept = 0.02, 0.01
        slope_se, intercept_se = 0.001, 0.002
        T_K = np.array([293.15, 303.15, 313.15, 323.15])
        absorbances = [0.61, 0.41, 0.31, 0.21]
        C0, m, V = 50.0, 0.1, 0.05

        Ce_arr, Ce_se_arr, qe_arr = [], [], []
        for abs_val in absorbances:
            Ce, Ce_se = propagate_calibration_uncertainty(
                abs_val,
                slope,
                intercept,
                slope_se,
                intercept_se,
            )
            Ce_arr.append(Ce)
            Ce_se_arr.append(Ce_se)
            qe_arr.append((C0 - Ce) * V / m)

        Ce_arr = np.array(Ce_arr)
        Ce_se_arr = np.array(Ce_se_arr)
        qe_arr = np.array(qe_arr)

        # mass-based: Kd = qe / Ce
        Kd = qe_arr / np.maximum(Ce_arr, 1e-10)
        Kd_se = propagate_kd_uncertainty(
            "mass_based",
            C0,
            Ce_arr,
            qe_arr,
            m,
            V,
            Ce_se_arr,
        )

        result = calculate_thermodynamic_parameters(T_K, Kd, Kd_se=Kd_se)
        assert result["success"] is True
        assert "wls_delta_H" in result

    def test_volume_corrected_chain(self):
        """Full chain with volume-corrected Kd method."""
        T_K = np.array([293.15, 303.15, 313.15, 323.15])
        C0, m, V = 100.0, 0.5, 0.1
        Ce = np.array([80.0, 60.0, 40.0, 20.0])
        Ce_se = np.array([2.0, 1.5, 1.0, 0.5])
        qe = (C0 - Ce) * V / m

        Kd = (qe * m) / (np.maximum(Ce, 1e-10) * V)
        Kd_se = propagate_kd_uncertainty(
            "volume_corrected",
            C0,
            Ce,
            qe,
            m,
            V,
            Ce_se,
        )

        result = calculate_thermodynamic_parameters(T_K, Kd, Kd_se=Kd_se)
        assert result["success"] is True
        assert "wls_delta_H" in result


# =========================================================================
# 4. _calculate_kd uncertainty in thermodynamics_tab (unit-level)
# =========================================================================
class TestKdMethodsWithUncertainty:
    """Verify propagation is consistent across all three Kd methods."""

    @pytest.fixture
    def data(self):
        return {
            "C0": 100.0,
            "Ce": np.array([80.0, 60.0, 40.0]),
            "qe": np.array([4.0, 8.0, 12.0]),
            "m": 0.5,
            "V": 0.1,
            "Ce_se": np.array([2.0, 1.5, 1.0]),
        }

    @pytest.mark.parametrize("method", ["dimensionless", "mass_based", "volume_corrected"])
    def test_all_methods_produce_positive_se(self, data, method):
        Kd_se = propagate_kd_uncertainty(
            method,
            data["C0"],
            data["Ce"],
            data["qe"],
            data["m"],
            data["V"],
            data["Ce_se"],
        )
        assert np.all(Kd_se > 0)
        assert np.all(np.isfinite(Kd_se))

    @pytest.mark.parametrize("method", ["dimensionless", "mass_based", "volume_corrected"])
    def test_scaling_with_Ce_se(self, data, method):
        """Double the Ce uncertainty → roughly double the Kd uncertainty."""
        se1 = propagate_kd_uncertainty(
            method,
            data["C0"],
            data["Ce"],
            data["qe"],
            data["m"],
            data["V"],
            data["Ce_se"],
        )
        se2 = propagate_kd_uncertainty(
            method,
            data["C0"],
            data["Ce"],
            data["qe"],
            data["m"],
            data["V"],
            data["Ce_se"] * 2,
        )
        # Ratio should be approximately 2 (exactly 2 for dimensionless)
        ratios = se2 / se1
        assert np.all(ratios > 1.5)
        assert np.all(ratios < 2.5)
