"""Cross-path regression checks for the combined scientific corrections."""

import numpy as np
import pandas as pd
import pytest

from adsorblab_pro.models import fit_model_with_ci, predict_revised_pso, revised_pso_model
from adsorblab_pro.statistical_criteria import information_criteria
from adsorblab_pro.tabs.report_tab import _gen_kinetic_model
from adsorblab_pro.utils import calculate_error_metrics


def test_fit_and_diagnostics_use_identical_information_criteria():
    x = np.arange(1.0, 9.0)
    y = 2 * x + 1 + np.array([0.1, -0.2, 0.3, -0.1, 0.1, -0.3, 0.2, -0.1])
    result = fit_model_with_ci(lambda t, a, b: a * t + b, x, y, [2, 1])
    assert result["converged"]
    diagnostic = calculate_error_metrics(y, result["y_pred"], n_params=2)
    for key in ("aic", "aicc", "bic"):
        assert diagnostic[key] == pytest.approx(result[key])


def test_exact_fit_is_finite_and_undefined_aicc_is_infinite():
    assert all(np.isfinite(information_criteria(0, 10, 2)))
    assert np.isinf(information_criteria(1, 4, 2)[1])


def _legacy_result():
    return {
        "converged": True,
        "r_squared": 0.99,
        "params": {"qe": 900.0, "k2": 0.001},
        "experimental_conditions": {"C0": 100.0, "m": 0.1, "V": 0.1},
    }


def test_legacy_prediction_and_report_use_the_fitted_equation():
    t = np.array([0.0, 1.0, 10.0, 100.0])
    result = _legacy_result()
    expected = revised_pso_model(t, 900, 0.001, 100, 0.1, 0.1)
    np.testing.assert_allclose(predict_revised_pso(t, result), expected)
    figure = _gen_kinetic_model(
        {
            "kinetic_models_fitted": {"rPSO": result},
            "kinetic_results_df": pd.DataFrame({"Time": t, "qt_mg_g": expected}),
        },
        "rPSO",
    )
    assert figure is not None
    for trace in figure.data:
        assert max(trace.y) <= 100


def test_legacy_prediction_refuses_missing_conditions():
    result = _legacy_result()
    del result["experimental_conditions"]
    with pytest.raises(ValueError, match="conditions"):
        predict_revised_pso(np.array([1.0]), result)


def test_wide_bounds_do_not_flag_an_interior_fit():
    x = np.arange(1.0, 9.0)
    result = fit_model_with_ci(lambda t, a: a * t, x, 2 * x, [1.0], bounds=([0], [1e12]))
    assert result["converged"]
    assert result["bounds_hit"] == []
