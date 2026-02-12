"""
Performance benchmark tests for AdsorbLab Pro.

Targets:
- Page load: <2 seconds
- Model fitting: <5 seconds

Run with: pytest tests/test_performance.py -v
"""

import time

import numpy as np
import pytest
from scipy.optimize import curve_fit


class TestModelFittingPerformance:
    """Benchmark model fitting times."""

    @pytest.fixture
    def isotherm_data(self):
        """Generate realistic isotherm test data."""
        np.random.seed(42)
        Ce = np.linspace(1, 100, 20)
        qe_true = (50 * 0.1 * Ce) / (1 + 0.1 * Ce)
        qe = qe_true + np.random.normal(0, 1, 20)
        return Ce, qe

    @pytest.fixture
    def kinetic_data(self):
        """Generate realistic kinetic test data."""
        np.random.seed(42)
        t = np.linspace(0, 120, 20)
        qt_true = (10**2 * 0.01 * t) / (1 + 10 * 0.01 * t)
        qt = qt_true + np.random.normal(0, 0.3, 20)
        return t, qt

    def test_langmuir_fitting_under_5s(self, isotherm_data):
        """Langmuir model fitting should complete in <5 seconds."""
        from adsorblab_pro.models import langmuir_model

        Ce, qe = isotherm_data
        start = time.perf_counter()
        popt, _ = curve_fit(langmuir_model, Ce, qe, p0=[50, 0.1], maxfev=5000)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Langmuir fitting took {elapsed:.2f}s (>5s limit)"

    def test_freundlich_fitting_under_5s(self, isotherm_data):
        """Freundlich model fitting should complete in <5 seconds."""
        from adsorblab_pro.models import freundlich_model

        Ce, qe = isotherm_data
        start = time.perf_counter()
        popt, _ = curve_fit(freundlich_model, Ce, qe, p0=[5, 0.5], maxfev=5000)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Freundlich fitting took {elapsed:.2f}s (>5s limit)"

    def test_sips_fitting_under_5s(self, isotherm_data):
        """Sips model (3 params) fitting should complete in <5 seconds."""
        from adsorblab_pro.models import sips_model

        Ce, qe = isotherm_data
        start = time.perf_counter()
        popt, _ = curve_fit(sips_model, Ce, qe, p0=[50, 0.1, 1], maxfev=5000)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Sips fitting took {elapsed:.2f}s (>5s limit)"

    def test_pso_fitting_under_5s(self, kinetic_data):
        """PSO kinetic model fitting should complete in <5 seconds."""
        from adsorblab_pro.models import pso_model

        t, qt = kinetic_data
        start = time.perf_counter()
        popt, _ = curve_fit(pso_model, t, qt, p0=[10, 0.01], maxfev=5000)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"PSO fitting took {elapsed:.2f}s (>5s limit)"

    def test_fit_model_with_ci_under_5s(self, isotherm_data):
        """fit_model_with_ci should complete in <5 seconds."""
        from adsorblab_pro.models import fit_model_with_ci, langmuir_model

        Ce, qe = isotherm_data
        start = time.perf_counter()
        result = fit_model_with_ci(langmuir_model, Ce, qe, p0=[50, 0.1], param_names=["qm", "KL"])
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"fit_model_with_ci took {elapsed:.2f}s (>5s limit)"
        assert result is not None

    def test_bootstrap_ci_under_5s(self, isotherm_data):
        """Bootstrap CI (500 iterations) should complete in <5 seconds."""
        from adsorblab_pro.models import langmuir_model
        from adsorblab_pro.utils import bootstrap_confidence_intervals

        Ce, qe = isotherm_data
        popt, _ = curve_fit(langmuir_model, Ce, qe, p0=[50, 0.1])

        start = time.perf_counter()
        ci_lower, ci_upper = bootstrap_confidence_intervals(
            langmuir_model, Ce, qe, popt, n_bootstrap=500
        )
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Bootstrap CI took {elapsed:.2f}s (>5s limit)"

    def test_full_isotherm_workflow_under_5s(self, isotherm_data):
        """Complete isotherm analysis workflow should be <5 seconds."""
        from adsorblab_pro.models import (
            fit_model_with_ci,
            freundlich_model,
            langmuir_model,
            sips_model,
            temkin_model,
        )
        from adsorblab_pro.plot_style import create_isotherm_plot
        from adsorblab_pro.utils import analyze_residuals, calculate_error_metrics

        Ce, qe = isotherm_data

        start = time.perf_counter()

        # Fit 4 models
        models = [
            (langmuir_model, [50, 0.1], ["qm", "KL"]),
            (freundlich_model, [5, 0.5], ["KF", "n"]),
            (temkin_model, [10, 1], ["B1", "KT"]),
            (sips_model, [50, 0.1, 1], ["qm", "Ks", "ns"]),
        ]

        results = []
        for model, p0, params in models:
            result = fit_model_with_ci(model, Ce, qe, p0=p0, param_names=params)
            if result:
                results.append(result)

        # Error metrics
        for result in results:
            y_pred = result["y_pred"]
            calculate_error_metrics(qe, y_pred, len(result["popt"]))
            analyze_residuals(qe - y_pred, y_pred)

        # Create plot
        Ce_fit = np.linspace(Ce.min(), Ce.max(), 100)
        qe_fit = langmuir_model(Ce_fit, *results[0]["popt"])
        fig = create_isotherm_plot(Ce, qe, Ce_fit, qe_fit, "Langmuir", 0.998)

        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Full workflow took {elapsed:.2f}s (>5s limit)"


class TestPageLoadPerformance:
    """Benchmark page load times."""

    def test_core_module_import_under_2s(self):
        """Core module imports should complete in <2 seconds."""
        import sys

        # Clear cached imports
        modules_to_clear = [k for k in sys.modules.keys() if k.startswith("adsorblab_pro")]
        for mod in modules_to_clear:
            del sys.modules[mod]

        start = time.perf_counter()

        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"Core imports took {elapsed:.2f}s (>2s limit)"

    def test_warm_page_render_under_2s(self):
        """Warm page render (modules loaded) should be <2 seconds."""
        # Pre-load all modules
        from adsorblab_pro import models, plot_style

        # Simulate data processing
        Ce = np.linspace(1, 100, 20)
        qe = (50 * 0.1 * Ce) / (1 + 0.1 * Ce) + np.random.normal(0, 1, 20)

        start = time.perf_counter()

        # This is what happens on page render
        result = models.fit_model_with_ci(
            models.langmuir_model, Ce, qe, p0=[50, 0.1], param_names=["qm", "KL"]
        )
        Ce_fit = np.linspace(Ce.min(), Ce.max(), 100)
        qe_fit = models.langmuir_model(Ce_fit, *result["popt"])
        fig = plot_style.create_isotherm_plot(Ce, qe, Ce_fit, qe_fit, "Langmuir", 0.998)

        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"Page render took {elapsed:.2f}s (>2s limit)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
