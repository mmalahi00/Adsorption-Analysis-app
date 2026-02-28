# tests/test_models.py
"""
Unit Tests for Adsorption Models
================================

Comprehensive test suite for isotherm and kinetic models.
Tests include:
- Model function correctness
- Parameter bounds handling
- Numerical stability
- Fitting convergence
- Multi-component competitive models
- Diffusion analysis (Biot number, rate-limiting step)
- Helper functions (initial rate, equilibrium time)

Author: AdsorbLab Team
"""

import os
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adsorblab_pro.models import (
    # Diagnostics
    calculate_biot_number,
    # Helper functions
    calculate_initial_rate,
    calculate_selectivity_coefficient,
    elovich_model,
    extended_freundlich_multicomponent,
    # Multi-component models
    extended_langmuir_multicomponent,
    # Fitting functions
    fit_model_with_ci,
    freundlich_model,
    get_model_info,
    identify_equilibrium_time,
    identify_rate_limiting_step,
    ipd_model,
    # Isotherm models
    langmuir_model,
    # Kinetic models
    pfo_model,
    pso_model,
    # Kinetic models - diffusion
    revised_pso_model,
    revised_pso_model_fixed_conditions,
    sips_model,
    temkin_model,
)

# =============================================================================
# FIXTURES - Reusable test data
# =============================================================================


@pytest.fixture
def isotherm_data():
    """Standard isotherm test data (Ce vs qe)."""
    Ce = np.array([5, 10, 20, 40, 60, 80, 100])
    qe = np.array([15.2, 25.8, 38.5, 52.1, 58.3, 62.5, 65.0])
    return Ce, qe


@pytest.fixture
def kinetic_data():
    """Standard kinetic test data (time vs qt)."""
    t = np.array([0, 5, 10, 20, 30, 60, 90, 120, 180, 240, 300])
    qt = np.array([0, 12.5, 22.0, 35.5, 44.0, 55.2, 60.1, 62.8, 64.5, 65.0, 65.2])
    return t, qt


@pytest.fixture
def langmuir_params():
    """Known Langmuir parameters for validation."""
    return {"qm": 70.0, "KL": 0.05}


@pytest.fixture
def pso_params():
    """Known PSO parameters for validation."""
    return {"qe": 65.0, "k2": 0.001}


# =============================================================================
# ISOTHERM MODEL TESTS
# =============================================================================


class TestLangmuirModel:
    """Tests for Langmuir isotherm model."""

    def test_langmuir_basic(self):
        """Test Langmuir model with known parameters."""
        Ce = np.array([10, 50, 100])
        qm, KL = 100, 0.1
        qe = langmuir_model(Ce, qm, KL)

        # Manual calculation: qe = (qm * KL * Ce) / (1 + KL * Ce)
        expected = (qm * KL * Ce) / (1 + KL * Ce)
        assert_allclose(qe, expected, rtol=1e-10)

    def test_langmuir_zero_concentration(self):
        """Test Langmuir at Ce = 0 returns qe = 0."""
        Ce = np.array([0.0])
        qe = langmuir_model(Ce, qm=100, KL=0.1)
        assert qe[0] == pytest.approx(0.0, abs=1e-10)

    def test_langmuir_saturation(self):
        """Test Langmuir approaches qm at high Ce."""
        Ce = np.array([1e6])  # Very high concentration
        qm = 100.0
        qe = langmuir_model(Ce, qm=qm, KL=0.1)
        assert qe[0] == pytest.approx(qm, rel=0.01)

    def test_langmuir_negative_protection(self):
        """Test Langmuir handles negative Ce gracefully."""
        Ce = np.array([-5, 0, 10])
        qe = langmuir_model(Ce, qm=100, KL=0.1)
        assert all(qe >= 0), "Langmuir should not produce negative values"

    def test_langmuir_array_output(self):
        """Test Langmuir returns array of correct shape."""
        Ce = np.linspace(1, 100, 50)
        qe = langmuir_model(Ce, qm=100, KL=0.1)
        assert qe.shape == Ce.shape

    def test_langmuir_monotonic_increasing(self):
        """Test Langmuir is monotonically increasing."""
        Ce = np.linspace(0.1, 100, 100)
        qe = langmuir_model(Ce, qm=100, KL=0.1)
        assert all(np.diff(qe) >= 0), "Langmuir should be monotonically increasing"


class TestFreundlichModel:
    """Tests for Freundlich isotherm model."""

    def test_freundlich_basic(self):
        """Test Freundlich model with known parameters."""
        Ce = np.array([10, 50, 100])
        KF, n_inv = 10.0, 0.5
        qe = freundlich_model(Ce, KF, n_inv)

        # Manual calculation: qe = KF * Ce^(1/n) = KF * Ce^n_inv
        expected = KF * np.power(Ce, n_inv)
        assert_allclose(qe, expected, rtol=1e-10)

    def test_freundlich_favorable(self):
        """Test Freundlich with n > 1 (favorable adsorption)."""
        Ce = np.array([10, 50, 100])
        KF, n_inv = 10.0, 0.3  # n = 1/0.3 ≈ 3.33 > 1
        qe = freundlich_model(Ce, KF, n_inv)
        assert all(qe > 0), "Freundlich should produce positive values"

    def test_freundlich_positive_output(self):
        """Test Freundlich always returns positive values."""
        Ce = np.linspace(0.1, 100, 50)
        qe = freundlich_model(Ce, KF=10, n_inv=0.5)
        assert all(qe > 0), "Freundlich should always be positive"


class TestTemkinModel:
    """Tests for Temkin isotherm model."""

    def test_temkin_basic(self):
        """Test Temkin model with known parameters."""
        Ce = np.array([10, 50, 100])
        B1, KT = 20.0, 0.5
        qe = temkin_model(Ce, B1, KT)

        # Manual calculation: qe = B1 * ln(KT * Ce)
        expected = B1 * np.log(KT * Ce)
        assert_allclose(qe, expected, rtol=1e-10)

    def test_temkin_positive_output(self):
        """Test Temkin returns non-negative values for valid inputs."""
        Ce = np.array([10, 50, 100])  # KT * Ce > 1
        qe = temkin_model(Ce, B1=20.0, KT=0.5)
        assert all(qe >= 0), "Temkin should be non-negative for KT*Ce > 1"

    def test_temkin_logarithmic_shape(self):
        """Test Temkin has logarithmic (concave) shape."""
        Ce = np.linspace(10, 100, 50)
        qe = temkin_model(Ce, B1=20.0, KT=0.5)
        # Second derivative should be negative (concave)
        second_deriv = np.diff(qe, 2)
        assert all(second_deriv < 0), "Temkin should be concave"


class TestSipsModel:
    """Tests for Sips (Langmuir-Freundlich) isotherm model."""

    def test_sips_basic(self):
        """Test Sips model with known parameters."""
        Ce = np.array([10, 50, 100])
        qe = sips_model(Ce, qm=100, Ks=0.1, ns=0.8)
        assert all(qe > 0), "Sips should produce positive values"
        assert all(qe <= 100), "Sips should not exceed qm"

    def test_sips_reduces_to_langmuir(self):
        """Test Sips reduces to Langmuir when ns = 1."""
        Ce = np.array([10, 50, 100])
        qm, Ks = 100, 0.1

        qe_sips = sips_model(Ce, qm=qm, Ks=Ks, ns=1.0)
        qe_lang = langmuir_model(Ce, qm=qm, KL=Ks)

        assert_allclose(qe_sips, qe_lang, rtol=0.01)

    def test_sips_saturation(self):
        """Test Sips approaches qm at high Ce."""
        Ce = np.array([1e6])
        qm = 100.0
        qe = sips_model(Ce, qm=qm, Ks=0.1, ns=0.8)
        assert qe[0] == pytest.approx(qm, rel=0.05)


# =============================================================================
# KINETIC MODEL TESTS
# =============================================================================


class TestPFOModel:
    """Tests for Pseudo-First Order kinetic model."""

    def test_pfo_basic(self):
        """Test PFO model with known parameters."""
        t = np.array([0, 30, 60, 120])
        qe, k1 = 50.0, 0.05
        qt = pfo_model(t, qe, k1)

        # Manual calculation: qt = qe * (1 - exp(-k1 * t))
        expected = qe * (1 - np.exp(-k1 * t))
        assert_allclose(qt, expected, rtol=1e-10)

    def test_pfo_initial_zero(self):
        """Test PFO starts at qt = 0 when t = 0."""
        t = np.array([0.0])
        qt = pfo_model(t, qe=50, k1=0.05)
        assert qt[0] == pytest.approx(0.0, abs=1e-10)

    def test_pfo_equilibrium(self):
        """Test PFO approaches qe at large t."""
        t = np.array([1e6])  # Very large time
        qe = 50.0
        qt = pfo_model(t, qe=qe, k1=0.05)
        assert qt[0] == pytest.approx(qe, rel=0.001)

    def test_pfo_monotonic(self):
        """Test PFO is monotonically increasing."""
        t = np.linspace(0, 300, 100)
        qt = pfo_model(t, qe=50, k1=0.05)
        assert all(np.diff(qt) >= 0), "PFO should be monotonically increasing"


class TestPSOModel:
    """Tests for Pseudo-Second Order kinetic model."""

    def test_pso_basic(self):
        """Test PSO model with known parameters."""
        t = np.array([0, 30, 60, 120])
        qe, k2 = 50.0, 0.001
        qt = pso_model(t, qe, k2)

        # Manual calculation: qt = (qe^2 * k2 * t) / (1 + qe * k2 * t)
        expected = (qe**2 * k2 * t) / (1 + qe * k2 * t)
        assert_allclose(qt, expected, rtol=1e-10)

    def test_pso_initial_zero(self):
        """Test PSO starts at qt = 0 when t = 0."""
        t = np.array([0.0])
        qt = pso_model(t, qe=50, k2=0.001)
        assert qt[0] == pytest.approx(0.0, abs=1e-10)

    def test_pso_equilibrium(self):
        """Test PSO approaches qe at large t."""
        t = np.array([1e6])
        qe = 50.0
        qt = pso_model(t, qe=qe, k2=0.001)
        assert qt[0] == pytest.approx(qe, rel=0.001)

    def test_pso_initial_rate(self):
        """Test PSO initial rate h = k2 * qe^2."""
        qe, k2 = 50.0, 0.001
        h_expected = k2 * qe**2  # Initial rate

        # At very small t, qt ≈ h * t
        t_small = np.array([0.001])
        qt = pso_model(t_small, qe, k2)
        h_actual = qt[0] / t_small[0]

        assert h_actual == pytest.approx(h_expected, rel=0.01)


class TestElovichModel:
    """Tests for Elovich kinetic model."""

    def test_elovich_basic(self):
        """Test Elovich model returns positive values."""
        t = np.array([1, 30, 60, 120])  # t > 0 required
        qt = elovich_model(t, alpha=10, beta=0.1)
        assert all(qt > 0), "Elovich should produce positive values"

    def test_elovich_monotonic(self):
        """Test Elovich is monotonically increasing."""
        t = np.linspace(1, 300, 100)
        qt = elovich_model(t, alpha=10, beta=0.1)
        assert all(np.diff(qt) >= 0), "Elovich should be monotonically increasing"

    def test_elovich_logarithmic_shape(self):
        """Test Elovich has logarithmic growth (decreasing rate)."""
        t = np.linspace(1, 300, 100)
        qt = elovich_model(t, alpha=10, beta=0.1)
        # Rate should decrease over time
        rates = np.diff(qt)
        assert all(np.diff(rates) <= 0), "Elovich rate should decrease"


class TestIPDModel:
    """Tests for Intraparticle Diffusion model."""

    def test_ipd_basic(self):
        """Test IPD model with known parameters."""
        t = np.array([1, 4, 9, 16, 25])
        kid, C = 5.0, 10.0
        qt = ipd_model(t, kid, C)

        # Manual calculation: qt = kid * sqrt(t) + C
        expected = kid * np.sqrt(t) + C
        assert_allclose(qt, expected, rtol=1e-10)

    def test_ipd_intercept(self):
        """Test IPD intercept C indicates boundary layer effect."""
        t = np.array([0.0])
        kid, C = 5.0, 10.0
        qt = ipd_model(t, kid, C)
        assert qt[0] == pytest.approx(C, abs=1e-10)

    def test_ipd_sqrt_dependence(self):
        """Test IPD varies with sqrt(t)."""
        t = np.array([1, 4, 9, 16])
        qt = ipd_model(t, kid=5.0, C=0)
        sqrt_t = np.sqrt(t)

        # qt should be linear with sqrt(t)
        coeffs = np.polyfit(sqrt_t, qt, 1)
        assert coeffs[0] == pytest.approx(5.0, rel=0.001)


# =============================================================================
# MODEL FITTING TESTS
# =============================================================================


class TestModelFitting:
    """Tests for model fitting with confidence intervals."""

    def test_langmuir_fit(self, isotherm_data, langmuir_params):
        """Test Langmuir model fitting converges."""
        Ce, qe = isotherm_data
        result = fit_model_with_ci(langmuir_model, Ce, qe, p0=[50, 0.1], param_names=["qm", "KL"])

        assert result is not None, "Fit should converge"
        assert result["converged"], "Fit should report convergence"
        assert "params" in result, "Result should contain params"
        assert "qm" in result["params"], "Should have qm parameter"
        assert "KL" in result["params"], "Should have KL parameter"

    def test_langmuir_fit_reasonable_params(self, isotherm_data):
        """Test Langmuir fit returns physically reasonable parameters."""
        Ce, qe = isotherm_data
        result = fit_model_with_ci(langmuir_model, Ce, qe, p0=[50, 0.1], param_names=["qm", "KL"])

        assert result["params"]["qm"] > 0, "qm should be positive"
        assert result["params"]["KL"] > 0, "KL should be positive"
        assert result["params"]["qm"] >= max(qe), "qm should be >= max observed qe"

    def test_langmuir_fit_r_squared(self, isotherm_data):
        """Test Langmuir fit achieves good R²."""
        Ce, qe = isotherm_data
        result = fit_model_with_ci(langmuir_model, Ce, qe, p0=[50, 0.1], param_names=["qm", "KL"])

        assert "r_squared" in result, "Should have R² value"
        assert result["r_squared"] > 0.9, "Should achieve R² > 0.9 on test data"

    def test_langmuir_fit_confidence_intervals(self, isotherm_data):
        """Test Langmuir fit includes confidence intervals."""
        Ce, qe = isotherm_data
        result = fit_model_with_ci(
            langmuir_model, Ce, qe, p0=[50, 0.1], param_names=["qm", "KL"], confidence=0.95
        )

        assert "ci_95" in result, "Should have 95% CI"
        assert "qm" in result["ci_95"], "Should have CI for qm"
        assert "KL" in result["ci_95"], "Should have CI for KL"

        # CI should bracket the fitted value
        qm_ci = result["ci_95"]["qm"]
        assert qm_ci[0] <= result["params"]["qm"] <= qm_ci[1]

    def test_pso_fit(self, kinetic_data, pso_params):
        """Test PSO model fitting converges."""
        t, qt = kinetic_data
        result = fit_model_with_ci(pso_model, t, qt, p0=[60, 0.0005], param_names=["qe", "k2"])

        assert result is not None, "Fit should converge"
        assert result["converged"], "Fit should report convergence"
        assert result["params"]["qe"] > 0, "qe should be positive"
        assert result["params"]["k2"] > 0, "k2 should be positive"

    def test_fit_with_insufficient_data(self):
        """Test fitting handles insufficient data gracefully."""
        Ce = np.array([10, 20])  # Only 2 points
        qe = np.array([15, 25])

        result = fit_model_with_ci(langmuir_model, Ce, qe, p0=[50, 0.1], param_names=["qm", "KL"])

        # Should either fail gracefully or fit with warning
        # Not crash
        assert result is None or isinstance(result, dict)


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_calculate_initial_rate(self, kinetic_data):
        """Test initial rate calculation."""
        t, qt = kinetic_data
        rate = calculate_initial_rate(t, qt)

        assert rate >= 0, "Initial rate should be non-negative"
        assert isinstance(rate, float)

    def test_identify_equilibrium_time(self, kinetic_data):
        """Test equilibrium time identification."""
        t, qt = kinetic_data
        t_eq = identify_equilibrium_time(t, qt, threshold=0.95)

        assert t_eq > 0, "Equilibrium time should be positive"
        assert t_eq <= t[-1], "Equilibrium time should be within data range"

    def test_get_model_info(self):
        """Test model info retrieval."""
        info = get_model_info()

        assert "isotherms" in info
        assert "kinetics" in info
        assert "Langmuir" in info["isotherms"]
        assert "PSO" in info["kinetics"]

        # Check model info structure
        lang_info = info["isotherms"]["Langmuir"]
        assert "equation" in lang_info
        assert "params" in lang_info
        assert "description" in lang_info


# =============================================================================
# EDGE CASE AND NUMERICAL STABILITY TESTS
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability and edge cases."""

    def test_langmuir_extreme_KL(self):
        """Test Langmuir with very small and large KL."""
        Ce = np.array([10, 50, 100])

        # Very small KL (low affinity)
        qe_small = langmuir_model(Ce, qm=100, KL=1e-10)
        assert all(np.isfinite(qe_small)), "Should handle very small KL"

        # Very large KL (high affinity)
        qe_large = langmuir_model(Ce, qm=100, KL=1e6)
        assert all(np.isfinite(qe_large)), "Should handle very large KL"

    def test_freundlich_extreme_n(self):
        """Test Freundlich with extreme n values."""
        Ce = np.array([10, 50, 100])

        qe_small_n = freundlich_model(Ce, KF=10, n_inv=0.1)
        qe_large_n = freundlich_model(Ce, KF=10, n_inv=2.0)

        assert all(np.isfinite(qe_small_n)), "Should handle small n_inv"
        assert all(np.isfinite(qe_large_n)), "Should handle large n_inv"

    def test_pso_extreme_k2(self):
        """Test PSO with extreme k2 values."""
        t = np.array([0, 30, 60, 120])

        qt_small = pso_model(t, qe=50, k2=1e-10)
        qt_large = pso_model(t, qe=50, k2=1e3)

        assert all(np.isfinite(qt_small)), "Should handle very small k2"
        assert all(np.isfinite(qt_large)), "Should handle very large k2"

    def test_models_with_zeros(self):
        """Test models handle zero values appropriately."""
        Ce = np.array([0, 10, 50])
        t = np.array([0, 30, 60])

        # These should not raise exceptions
        qe_lang = langmuir_model(Ce, qm=100, KL=0.1)
        qt_pso = pso_model(t, qe=50, k2=0.001)

        assert all(np.isfinite(qe_lang))
        assert all(np.isfinite(qt_pso))

    def test_models_with_nan_protection(self):
        """Test models don't produce NaN for edge cases."""
        Ce = np.array([1e-15, 1e15])
        t = np.array([1e-15, 1e15])

        # All models should return finite values
        assert all(np.isfinite(langmuir_model(Ce, 100, 0.1)))
        assert all(np.isfinite(freundlich_model(Ce, 10, 0.5)))
        assert all(np.isfinite(pfo_model(t, 50, 0.05)))
        assert all(np.isfinite(pso_model(t, 50, 0.001)))


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# =============================================================================
# EXTENDED TESTS (Additional coverage for models.py)
# =============================================================================

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def multicomponent_data():
    """Test data for multi-component adsorption."""
    Ce = np.array(
        [
            [10, 5],  # Component 1: 10 mg/L, Component 2: 5 mg/L
            [20, 10],
            [30, 15],
            [40, 20],
        ]
    )
    return Ce


@pytest.fixture
def kinetic_data_extended():
    """Extended kinetic data for diffusion tests."""
    t = np.array([0, 1, 2, 5, 10, 20, 30, 60, 90, 120, 180, 240, 300, 360])
    qt = np.array([0, 5, 10, 20, 32, 45, 52, 60, 63, 64.5, 65.5, 66, 66.2, 66.3])
    return t, qt


# =============================================================================
# MULTI-COMPONENT COMPETITIVE ADSORPTION TESTS
# =============================================================================


class TestExtendedLangmuir:
    """Tests for extended Langmuir multi-component model."""

    def test_extended_langmuir_basic(self):
        """Test extended Langmuir for component i with competing species."""
        # Component i concentrations
        Ce_i = np.array([10, 20, 30, 40])
        qm_i = 100  # Max capacity for component i
        KL_i = 0.1  # Langmuir constant for component i

        # All components (including i) for competition term
        Ce_all = [Ce_i, np.array([5, 10, 15, 20])]  # Two components
        KL_all = [0.1, 0.05]

        qe = extended_langmuir_multicomponent(Ce_i, qm_i, KL_i, Ce_all, KL_all)

        assert len(qe) == len(Ce_i)
        assert np.all(qe >= 0)
        assert np.all(qe <= qm_i)

    def test_extended_langmuir_competition_reduces_uptake(self):
        """Test that competition reduces individual component uptake."""
        Ce_i = np.array([20, 40, 60])
        qm_i = 100
        KL_i = 0.1

        # Without competition (only component i)
        Ce_all_alone = [Ce_i]
        KL_all_alone = [KL_i]
        qe_alone = extended_langmuir_multicomponent(Ce_i, qm_i, KL_i, Ce_all_alone, KL_all_alone)

        # With competition (component i + competitor)
        Ce_all_compete = [Ce_i, np.array([20, 40, 60])]
        KL_all_compete = [KL_i, 0.1]
        qe_compete = extended_langmuir_multicomponent(
            Ce_i, qm_i, KL_i, Ce_all_compete, KL_all_compete
        )

        # Competition should reduce uptake
        assert np.all(qe_compete < qe_alone)

    def test_extended_langmuir_positive_output(self):
        """Test all outputs are positive."""
        Ce_i = np.array([5, 10, 20, 40])
        qe = extended_langmuir_multicomponent(
            Ce_i, qm_i=80, KL_i=0.05, Ce_all=[Ce_i, np.array([10, 20, 40, 80])], KL_all=[0.05, 0.1]
        )
        assert np.all(qe > 0)


class TestExtendedFreundlich:
    """Tests for extended Freundlich multi-component model."""

    def test_extended_freundlich_basic(self):
        """Test extended Freundlich for component i."""
        Ce_i = np.array([10, 20, 30, 40])
        Kf_i = 5.0
        n_i = 2.0

        Ce_all = [Ce_i, np.array([5, 10, 15, 20])]
        Kf_all = [5.0, 3.0]
        n_all = [2.0, 2.5]

        qe = extended_freundlich_multicomponent(Ce_i, Kf_i, n_i, Ce_all, Kf_all, n_all)

        assert len(qe) == len(Ce_i)
        assert np.all(qe >= 0)

    def test_extended_freundlich_positive_output(self):
        """Test all outputs are positive."""
        Ce_i = np.array([5, 10, 20])
        qe = extended_freundlich_multicomponent(
            Ce_i,
            Kf_i=10.0,
            n_i=1.5,
            Ce_all=[Ce_i, np.array([5, 10, 20])],
            Kf_all=[10.0, 8.0],
            n_all=[1.5, 2.0],
        )
        assert np.all(qe > 0)


class TestSelectivityCoefficient:
    """Tests for selectivity coefficient calculation."""

    def test_selectivity_basic(self):
        """Test basic selectivity calculation."""
        alpha = calculate_selectivity_coefficient(qe_i=80, Ce_i=10, qe_j=20, Ce_j=20)
        # α = (80 * 20) / (20 * 10) = 8
        assert alpha == pytest.approx(8.0, rel=0.01)

    def test_selectivity_equal_preference(self):
        """Test equal preference gives α = 1."""
        alpha = calculate_selectivity_coefficient(qe_i=50, Ce_i=10, qe_j=50, Ce_j=10)
        assert alpha == pytest.approx(1.0, rel=0.01)

    def test_selectivity_array_input(self):
        """Test with array inputs."""
        qe_i = np.array([80, 70, 60])
        Ce_i = np.array([10, 15, 20])
        qe_j = np.array([20, 30, 40])
        Ce_j = np.array([20, 15, 10])

        alpha = calculate_selectivity_coefficient(qe_i, Ce_i, qe_j, Ce_j)
        assert len(alpha) == 3
        assert np.all(np.isfinite(alpha))

    def test_selectivity_invalid_inputs(self):
        """Test handling of invalid inputs."""
        alpha = calculate_selectivity_coefficient(
            qe_i=50,
            Ce_i=0,  # Invalid: Ce_i = 0
            qe_j=50,
            Ce_j=10,
        )
        assert np.isnan(alpha)


# =============================================================================
# REVISED PSO MODEL TESTS
# =============================================================================


class TestRevisedPSO:
    """Tests for revised PSO model (Bullen et al., 2021)."""

    def test_rpso_basic(self):
        """Test rPSO model basic calculation."""
        t = np.array([0, 10, 30, 60, 120, 240])
        qt = revised_pso_model(t, qe=65, k2=0.001, C0=100, m=0.1, V=0.1)

        assert len(qt) == len(t)
        assert qt[0] == pytest.approx(0, abs=1e-6)
        assert np.all(np.diff(qt) >= 0)  # Monotonically increasing

    def test_rpso_approaches_equilibrium(self):
        """Test rPSO approaches qe at long times."""
        t = np.array([0, 60, 120, 240, 480, 960, 1920, 10000])
        qe = 65
        qt = revised_pso_model(t, qe=qe, k2=0.001, C0=100, m=0.1, V=0.1)

        # At very long time, should approach qe
        assert qt[-1] < qe  # Should be below due to concentration correction

    def test_rpso_fixed_conditions(self):
        """Test rPSO with fixed experimental conditions."""
        model_func = revised_pso_model_fixed_conditions(C0=100, m=0.1, V=0.1)

        t = np.array([0, 30, 60, 120])
        qt = model_func(t, qe=65, k2=0.001)

        assert len(qt) == len(t)
        assert qt[0] == pytest.approx(0, abs=1e-6)


# =============================================================================
# BIOT NUMBER AND RATE-LIMITING STEP TESTS
# =============================================================================


class TestBiotNumber:
    """Tests for Biot number calculation."""

    def test_biot_number_basic(self):
        """Test basic Biot number calculation."""
        Bi = calculate_biot_number(kf=1e-4, Dp=1e-10, r=0.001)

        assert Bi > 0
        assert np.isfinite(Bi)

    def test_biot_number_film_controlled(self):
        """Test Bi < 1 indicates film diffusion control."""
        # Low kf relative to Dp/r
        Bi = calculate_biot_number(kf=1e-6, Dp=1e-8, r=0.001)
        assert Bi < 1

    def test_biot_number_pore_controlled(self):
        """Test Bi > 100 indicates pore diffusion control."""
        # High kf relative to Dp/r
        Bi = calculate_biot_number(kf=1e-2, Dp=1e-12, r=0.001)
        assert Bi > 100


class TestRateLimitingStep:
    """Tests for rate-limiting step identification."""

    def test_identify_rate_limiting(self, kinetic_data_extended):
        """Test rate-limiting step identification."""
        t, qt = kinetic_data_extended
        qe = qt[-1]

        result = identify_rate_limiting_step(t, qt, qe, particle_radius=0.001)

        # Function should return a dictionary with analysis results
        assert result is not None
        assert isinstance(result, dict)

    def test_identify_rate_limiting_with_radius(self):
        """Test with explicit particle radius."""
        t = np.array([0, 5, 10, 20, 30, 60, 120, 180])
        qt = np.array([0, 15, 28, 42, 50, 58, 62, 64])

        result = identify_rate_limiting_step(t, qt, qe=65, particle_radius=0.0005)

        assert result is not None

    def test_identify_rate_limiting_returns_dict(self):
        """Test returns dictionary structure."""
        t = np.array([0, 10, 20, 40, 60, 90, 120])
        qt = np.array([0, 20, 35, 50, 58, 62, 64])

        result = identify_rate_limiting_step(t, qt, qe=65)

        assert isinstance(result, dict)


class TestFitModelWithCI:
    """Tests for fitting with confidence intervals."""

    def test_fit_with_ci_returns_intervals(self, isotherm_data):
        """Test CI fitting returns confidence intervals."""
        Ce, qe = isotherm_data

        result = fit_model_with_ci(langmuir_model, Ce, qe, p0=[70, 0.05], bounds=([0, 0], [200, 1]))

        assert result is not None
        assert "params" in result
        assert "ci_95" in result  # Actual key name
        assert "r_squared" in result

    def test_fit_with_ci_reasonable_params(self, isotherm_data):
        """Test fitted parameters are reasonable."""
        Ce, qe = isotherm_data

        result = fit_model_with_ci(langmuir_model, Ce, qe, p0=[70, 0.05], bounds=([0, 0], [200, 1]))

        if result is not None:
            params = result["params"]
            # params is dict with 'p0', 'p1' keys
            assert params["p0"] > 0
            assert params["p0"] < 200
            assert params["p1"] > 0


# =============================================================================
# IPD MULTI-STAGE ANALYSIS
# =============================================================================


class TestIPDMultiStage:
    """Extended numerical stability tests."""

    def test_multicomponent_with_zeros(self):
        """Test multi-component models handle zero concentrations."""
        Ce_i = np.array([10, 20, 30])

        # With competitor at zero concentration
        qe = extended_langmuir_multicomponent(
            Ce_i, qm_i=100, KL_i=0.1, Ce_all=[Ce_i, np.array([0, 0, 0])], KL_all=[0.1, 0.05]
        )
        assert np.all(np.isfinite(qe))

    def test_biot_number_extreme_values(self):
        """Test Biot number with extreme inputs."""
        # Very small diffusivity
        Bi = calculate_biot_number(kf=0.01, Dp=1e-15, r=0.001)
        assert np.isfinite(Bi)
        assert Bi > 0

        # Very large diffusivity
        Bi = calculate_biot_number(kf=0.01, Dp=1e-6, r=0.001)
        assert np.isfinite(Bi)
        assert Bi > 0
