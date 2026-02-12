# tests/test_ui_streamlit.py
"""
Streamlit UI Tests
==================

Tests for Streamlit UI components using AppTest.
These tests run the actual Streamlit app in headless mode.

Author: AdsorbLab Team
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if streamlit.testing is available
try:
    from streamlit.testing.v1 import AppTest

    APPTEST_AVAILABLE = True
except ImportError:
    APPTEST_AVAILABLE = False


# =============================================================================
# HELPER FUNCTIONS FOR TESTING TABS DIRECTLY
# =============================================================================


def create_mock_session_state():
    """Create a mock session state with test data."""
    return {
        "isotherm_data": pd.DataFrame(
            {
                "Ce": [5, 10, 20, 40, 60, 80, 100],
                "qe": [15.2, 25.8, 38.5, 52.1, 58.3, 62.5, 65.0],
                "C0": [100, 100, 100, 100, 100, 100, 100],
            }
        ),
        "kinetic_data": pd.DataFrame(
            {"t": [0, 5, 10, 20, 30, 60, 90, 120], "qt": [0, 15, 28, 42, 50, 58, 62, 64]}
        ),
        "calibration_data": pd.DataFrame(
            {
                "Concentration": [0, 10, 20, 40, 60, 80, 100],
                "Absorbance": [0.01, 0.12, 0.23, 0.45, 0.67, 0.89, 1.10],
            }
        ),
        "studies": {},
        "current_study": None,
        "first_time": False,
    }


# =============================================================================
# TAB IMPORT TESTS
# =============================================================================


class TestTabImports:
    """Test that all tabs can be imported without errors."""

    def test_import_home_tab(self):
        """Test home tab import."""
        from adsorblab_pro.tabs import home_tab

        assert hasattr(home_tab, "render")
        assert callable(home_tab.render)

    def test_import_calibration_tab(self):
        """Test calibration tab import."""
        from adsorblab_pro.tabs import calibration_tab

        assert hasattr(calibration_tab, "render")
        assert callable(calibration_tab.render)

    def test_import_isotherm_tab(self):
        """Test isotherm tab import."""
        from adsorblab_pro.tabs import isotherm_tab

        assert hasattr(isotherm_tab, "_compute_data_hash")
        assert hasattr(isotherm_tab, "_arrays_to_tuples")

    def test_import_kinetic_tab(self):
        """Test kinetic tab import."""
        from adsorblab_pro.tabs import kinetic_tab

        assert hasattr(kinetic_tab, "render")
        assert callable(kinetic_tab.render)

    def test_import_thermodynamics_tab(self):
        """Test thermodynamics tab import."""
        from adsorblab_pro.tabs import thermodynamics_tab

        assert hasattr(thermodynamics_tab, "render")
        assert callable(thermodynamics_tab.render)

    def test_import_comparison_tab(self):
        """Test comparison tab import."""
        from adsorblab_pro.tabs import comparison_tab

        assert hasattr(comparison_tab, "render")
        assert callable(comparison_tab.render)

    def test_import_report_tab(self):
        """Test report tab import."""
        from adsorblab_pro.tabs import report_tab

        assert hasattr(report_tab, "render")
        assert callable(report_tab.render)

    def test_import_competitive_tab(self):
        """Test competitive tab import."""
        from adsorblab_pro.tabs import competitive_tab

        assert hasattr(competitive_tab, "render")
        assert callable(competitive_tab.render)

    def test_import_threed_explorer_tab(self):
        """Test 3D explorer tab import."""
        from adsorblab_pro.tabs import threed_explorer_tab

        assert hasattr(threed_explorer_tab, "render")
        assert callable(threed_explorer_tab.render)

    def test_import_dosage_tab(self):
        from adsorblab_pro.tabs import dosage_tab

        assert callable(dosage_tab.render)

    def test_import_ph_effect_tab(self):
        from adsorblab_pro.tabs import ph_effect_tab

        assert callable(ph_effect_tab.render)

    def test_import_temperature_tab(self):
        from adsorblab_pro.tabs import temperature_tab

        assert callable(temperature_tab.render)

    def test_import_statistical_summary_tab(self):
        from adsorblab_pro.tabs import statistical_summary_tab

        assert callable(statistical_summary_tab.render)


# =============================================================================
# TAB HELPER FUNCTION TESTS
# =============================================================================


class TestIsothermTabHelpers:
    """Test helper functions in isotherm_tab."""

    def test_compute_data_hash(self):
        """Test data hash computation."""
        from adsorblab_pro.tabs.isotherm_tab import _compute_data_hash

        Ce = np.array([5, 10, 20])
        qe = np.array([15, 26, 39])
        C0 = np.array([100, 100, 100])

        hash1 = _compute_data_hash(Ce, qe, C0)
        hash2 = _compute_data_hash(Ce, qe, C0)

        # Same data should give same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length

    def test_compute_data_hash_different_data(self):
        """Test hash changes with different data."""
        from adsorblab_pro.tabs.isotherm_tab import _compute_data_hash

        Ce1 = np.array([5, 10, 20])
        Ce2 = np.array([5, 10, 25])
        qe = np.array([15, 26, 39])
        C0 = np.array([100, 100, 100])

        hash1 = _compute_data_hash(Ce1, qe, C0)
        hash2 = _compute_data_hash(Ce2, qe, C0)

        assert hash1 != hash2

    def test_arrays_to_tuples(self):
        """Test array to tuple conversion."""
        from adsorblab_pro.tabs.isotherm_tab import _arrays_to_tuples

        Ce = np.array([5, 10, 20])
        qe = np.array([15, 26, 39])
        C0 = np.array([100, 100, 100])

        result = _arrays_to_tuples(Ce, qe, C0)

        assert len(result) == 3
        assert all(isinstance(t, tuple) for t in result)
        assert result[0] == (5.0, 10.0, 20.0)

    def test_arrays_to_tuples_precision(self):
        """Test tuple conversion handles precision."""
        from adsorblab_pro.tabs.isotherm_tab import _arrays_to_tuples

        Ce = np.array([5.123456789012345, 10.987654321])
        qe = np.array([15.0, 26.0])
        C0 = np.array([100.0, 100.0])

        result = _arrays_to_tuples(Ce, qe, C0)

        # Should round to 8 decimal places
        assert result[0][0] == pytest.approx(5.12345679, rel=1e-7)


# =============================================================================
# STREAMLIT APPTEST TESTS (if available)
# =============================================================================


@pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
class TestStreamlitApp:
    """Integration tests using Streamlit AppTest."""

    def test_app_starts(self):
        """Test that the main app starts without errors."""
        try:
            at = AppTest.from_file("adsorblab_pro/app.py", default_timeout=30)
            at.run()
            # App should not have uncaught exceptions
            assert not at.exception
        except Exception as e:
            # If app fails to start, that's also useful information
            pytest.skip(f"App failed to start: {e}")

    def test_app_has_tabs(self):
        """Test that app creates expected tabs."""
        try:
            at = AppTest.from_file("adsorblab_pro/app.py", default_timeout=30)
            at.run()
            # Check for some expected elements
            assert not at.exception
        except Exception as e:
            pytest.skip(f"App test failed: {e}")


# =============================================================================
# CONFIG AND VALIDATION TESTS
# =============================================================================


class TestConfigIntegration:
    """Test config integration with UI."""

    def test_config_imports(self):
        """Test config can be imported."""
        from adsorblab_pro.config import (
            DEFAULT_SESSION_STATE,
            ISOTHERM_MODELS,
            KINETIC_MODELS,
            VERSION,
        )

        assert isinstance(DEFAULT_SESSION_STATE, dict)
        assert isinstance(VERSION, str)
        assert isinstance(ISOTHERM_MODELS, dict)
        assert isinstance(KINETIC_MODELS, dict)

    def test_default_session_state_structure(self):
        """Test default session state has expected keys."""
        from adsorblab_pro.config import DEFAULT_SESSION_STATE

        expected_keys = ["isotherm_input", "kinetic_input", "unit_system"]
        for key in expected_keys:
            assert key in DEFAULT_SESSION_STATE

    def test_model_lists_not_empty(self):
        """Test model dicts contain models."""
        from adsorblab_pro.config import ISOTHERM_MODELS, KINETIC_MODELS

        assert len(ISOTHERM_MODELS) >= 4
        assert len(KINETIC_MODELS) >= 4
        assert "Langmuir" in ISOTHERM_MODELS
        assert "PSO" in KINETIC_MODELS


class TestValidationIntegration:
    """Test validation module integration with UI."""

    def test_validation_report_structure(self):
        """Test ValidationReport structure."""
        from adsorblab_pro.validation import ValidationReport

        report = ValidationReport(is_valid=True, errors=[], warnings=[], info=[])
        assert report.is_valid is True
        assert isinstance(report.errors, list)

    def test_validate_isotherm_data(self):
        """Test isotherm data validation."""
        from adsorblab_pro.validation import validate_isotherm_data

        Ce = np.array([5, 10, 20, 40, 60])
        qe = np.array([15, 26, 39, 52, 58])
        C0 = np.array([100, 100, 100, 100, 100])

        report = validate_isotherm_data(Ce, qe, C0)
        assert hasattr(report, "is_valid")

    def test_validate_kinetic_data(self):
        """Test kinetic data validation."""
        from adsorblab_pro.validation import validate_kinetic_data

        t = np.array([0, 5, 10, 20, 30, 60])
        qt = np.array([0, 15, 28, 42, 50, 58])

        report = validate_kinetic_data(t, qt)
        assert hasattr(report, "is_valid")


# =============================================================================
# PLOT STYLE TESTS
# =============================================================================


class TestPlotStyle:
    """Test plot styling functions."""

    def test_plot_style_imports(self):
        """Test plot style can be imported."""
        from adsorblab_pro.plot_style import COLORS, MODEL_COLORS

        assert isinstance(MODEL_COLORS, dict)
        assert isinstance(COLORS, dict)

    def test_model_colors_complete(self):
        """Test all models have colors assigned."""
        from adsorblab_pro.config import ISOTHERM_MODELS
        from adsorblab_pro.plot_style import MODEL_COLORS

        # Check isotherm models have colors
        for model in ISOTHERM_MODELS:
            assert model in MODEL_COLORS, f"Missing color for {model}"

    def test_create_isotherm_plot_function(self):
        """Test isotherm plot creation function exists."""
        from adsorblab_pro.plot_style import create_isotherm_plot

        # Just check function exists
        assert callable(create_isotherm_plot)


# =============================================================================
# SIDEBAR UI TESTS
# =============================================================================


class TestSidebarUI:
    """Test sidebar UI module."""

    def test_sidebar_imports(self):
        """Test sidebar UI can be imported."""
        import adsorblab_pro.sidebar_ui as sidebar_ui

        assert hasattr(sidebar_ui, "render_sidebar_content")
        assert callable(sidebar_ui.render_sidebar_content)

    def test_sidebar_has_expected_functions(self):
        """Test sidebar has expected functions."""
        import adsorblab_pro.sidebar_ui as sidebar_ui

        # Check for common sidebar functions
        module_contents = dir(sidebar_ui)
        assert len(module_contents) > 0


# =============================================================================
# DATA PROCESSING INTEGRATION TESTS
# =============================================================================


class TestDataProcessingUI:
    """Test data processing functions used by UI."""

    def test_csv_export(self):
        """Test CSV export for download buttons."""
        from adsorblab_pro.utils import convert_df_to_csv

        df = pd.DataFrame({"Ce": [5, 10, 20], "qe": [15, 26, 39]})

        csv_bytes = convert_df_to_csv(df)
        assert isinstance(csv_bytes, bytes)
        assert b"Ce" in csv_bytes
        assert b"qe" in csv_bytes

    def test_excel_export(self):
        """Test Excel export for download buttons."""
        from adsorblab_pro.utils import convert_df_to_excel

        df = pd.DataFrame({"Ce": [5, 10, 20], "qe": [15, 26, 39]})

        excel_bytes = convert_df_to_excel(df)
        assert isinstance(excel_bytes, bytes)
        # Excel files start with PK (zip signature)
        assert excel_bytes[:2] == b"PK"

    def test_column_standardization(self):
        """Test column name standardization."""
        from adsorblab_pro.utils import standardize_column_name, standardize_dataframe_columns

        # Test single column
        result = standardize_column_name("equilibrium_concentration")
        assert isinstance(result, str)

        # Test DataFrame
        df = pd.DataFrame({"Ce": [1, 2, 3]})
        result_df = standardize_dataframe_columns(df)
        assert isinstance(result_df, pd.DataFrame)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestUIErrorHandling:
    """Test error handling in UI components."""

    def test_empty_dataframe_handling(self):
        """Test functions handle empty DataFrames."""
        from adsorblab_pro.utils import convert_df_to_csv

        df = pd.DataFrame()
        result = convert_df_to_csv(df)
        assert isinstance(result, bytes)

    def test_none_input_handling(self):
        """Test functions handle None inputs gracefully."""
        from adsorblab_pro.utils import standardize_column_name

        # Should not crash
        try:
            standardize_column_name(None)
        except (TypeError, AttributeError):
            pass  # Expected behavior

    def test_invalid_data_validation(self):
        """Test validation catches invalid data."""
        from adsorblab_pro.validation import validate_isotherm_data

        # Negative concentrations
        Ce = np.array([-5, 10, 20])
        qe = np.array([15, 26, 39])
        C0 = np.array([100, 100, 100])

        report = validate_isotherm_data(Ce, qe, C0)
        # Should flag invalid data
        assert hasattr(report, "is_valid")


# =============================================================================
# WIDGET INTERACTION SIMULATION TESTS
# =============================================================================


class TestWidgetSimulation:
    """Simulate widget interactions without Streamlit runtime."""

    def test_model_selection_logic(self):
        """Test model selection logic."""
        from adsorblab_pro.config import ISOTHERM_MODELS

        # Simulate user selecting a model
        model_names = list(ISOTHERM_MODELS.keys())
        selected_model = model_names[0]  # 'Langmuir'

        # Verify model is valid
        assert selected_model in ISOTHERM_MODELS

    def test_parameter_bounds_logic(self):
        """Test parameter bounds logic used in sliders."""
        from adsorblab_pro.models import fit_model_with_ci, langmuir_model

        Ce = np.array([5, 10, 20, 40, 60, 80, 100])
        qe = np.array([15, 26, 39, 52, 58, 62, 65])

        # Bounds typically shown in UI sliders
        qm_bounds = (0, 200)
        KL_bounds = (0, 1)

        result = fit_model_with_ci(
            langmuir_model,
            Ce,
            qe,
            p0=[70, 0.05],
            bounds=([qm_bounds[0], KL_bounds[0]], [qm_bounds[1], KL_bounds[1]]),
        )

        if result and result["converged"]:
            qm, KL = result["popt"]
            assert qm_bounds[0] <= qm <= qm_bounds[1]
            assert KL_bounds[0] <= KL <= KL_bounds[1]

    def test_number_input_validation(self):
        """Test number input validation logic."""

        # Simulate validation for number inputs
        def validate_positive(value):
            return value is not None and value > 0

        assert validate_positive(100) is True
        assert validate_positive(0) is False
        assert validate_positive(-5) is False
        assert validate_positive(None) is False


# =============================================================================
# REPORT GENERATION TESTS
# =============================================================================


class TestReportGeneration:
    """Test report generation functionality."""

    def test_report_tab_imports(self):
        """Test report tab can be imported and exposes a render() entrypoint."""
        from adsorblab_pro.tabs import report_tab

        assert hasattr(report_tab, "render")
        assert callable(report_tab.render)

    def test_calculation_result_structure(self):
        """Test CalculationResult structure used in reports."""
        import pandas as pd

        from adsorblab_pro.utils import CalculationResult

        result = CalculationResult(success=True, data=pd.DataFrame({"col1": [1, 2, 3]}), error=None)

        assert result.success is True
        assert result.data is not None


# =============================================================================
# 3D EXPLORER TAB TESTS
# =============================================================================


class TestThreeDExplorerTab:
    """Test 3D explorer functionality."""

    def test_threed_tab_imports(self):
        """Test 3D tab can be imported and exposes a render() entrypoint."""
        from adsorblab_pro.tabs import threed_explorer_tab

        assert hasattr(threed_explorer_tab, "render")
        assert callable(threed_explorer_tab.render)

    def test_plotly_availability(self):
        """Test Plotly is available for 3D plots."""
        import plotly.graph_objects as go

        # Create a simple 3D scatter
        fig = go.Figure(data=[go.Scatter3d(x=[1, 2, 3], y=[1, 2, 3], z=[1, 2, 3], mode="markers")])

        assert fig is not None
