"""
Extended tests for adsorblab_pro/config.py - boosting coverage.
"""

import numpy as np
import pytest

from adsorblab_pro.config import (
    ALLOWED_FILE_TYPES,
    BOOTSTRAP_DEFAULT_ITERATIONS,
    BOOTSTRAP_MIN_SUCCESS,
    DEFAULT_GLOBAL_SESSION_STATE,
    DEFAULT_SESSION_STATE,
    EPSILON_DIV,
    EPSILON_LOG,
    EPSILON_ZERO,
    EXPORT_DPI,
    FONT_FAMILY,
    FUZZY_MATCH_CUTOFF,
    ISOTHERM_MODELS,
    KD_WARNING_THRESHOLD,
    KINETIC_MODELS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MAX_FIT_ITERATIONS,
    MECHANISM_CRITERIA,
    MIN_DATA_POINTS,
    MULTICOMPONENT_MODELS,
    PI_SQUARED,
    PLOT_TEMPLATE,
    QUALITY_GRADES,
    QUALITY_THRESHOLDS,
    R_GAS_CONSTANT,
    SESSION_INPUT_KEYS_TO_CLEAR,
    SESSION_WIDGET_PREFIXES_TO_CLEAR,
    STUDY_METRIC_DATA_KEYS,
    TEMP_MAX_KELVIN,
    TEMP_MIN_KELVIN,
    VERSION,
    get_calibration_grade,
    get_grade_from_r_squared,
    get_grade_from_score,
)


class TestConfigConstants:
    def test_version(self):
        assert isinstance(VERSION, str)
        assert len(VERSION) > 0

    def test_file_size_limits(self):
        assert MAX_FILE_SIZE_MB == 10
        assert MAX_FILE_SIZE_BYTES == 10 * 1024 * 1024

    def test_allowed_file_types(self):
        assert "csv" in ALLOWED_FILE_TYPES
        assert "xlsx" in ALLOWED_FILE_TYPES
        assert "xls" in ALLOWED_FILE_TYPES

    def test_epsilon_hierarchy(self):
        assert EPSILON_DIV > 0
        assert EPSILON_ZERO > 0
        assert EPSILON_LOG > 0
        assert EPSILON_DIV <= EPSILON_ZERO

    def test_r_gas_constant(self):
        assert R_GAS_CONSTANT == pytest.approx(8.314, rel=0.01)

    def test_pi_squared(self):
        assert PI_SQUARED == pytest.approx(np.pi**2)

    def test_temperature_limits(self):
        assert TEMP_MIN_KELVIN < TEMP_MAX_KELVIN
        assert TEMP_MIN_KELVIN > 0

    def test_kd_warning_threshold(self):
        assert KD_WARNING_THRESHOLD > 0

    def test_fuzzy_match_cutoff(self):
        assert 0 < FUZZY_MATCH_CUTOFF <= 1.0

    def test_min_data_points(self):
        assert MIN_DATA_POINTS >= 2

    def test_max_fit_iterations(self):
        assert MAX_FIT_ITERATIONS > 1000

    def test_bootstrap_constants(self):
        assert BOOTSTRAP_DEFAULT_ITERATIONS > 0
        assert BOOTSTRAP_MIN_SUCCESS > 0


class TestModelConfigurations:
    def test_isotherm_models(self):
        assert isinstance(ISOTHERM_MODELS, dict)
        assert "Langmuir" in ISOTHERM_MODELS
        assert "Freundlich" in ISOTHERM_MODELS

    def test_kinetic_models(self):
        assert isinstance(KINETIC_MODELS, dict)
        assert "PFO" in KINETIC_MODELS
        assert "PSO" in KINETIC_MODELS

    def test_multicomponent_models(self):
        assert isinstance(MULTICOMPONENT_MODELS, dict)

    def test_mechanism_criteria(self):
        assert isinstance(MECHANISM_CRITERIA, dict)


class TestQualityGrades:
    def test_quality_thresholds(self):
        assert isinstance(QUALITY_THRESHOLDS, dict)

    def test_quality_grades(self):
        assert isinstance(QUALITY_GRADES, dict)
        for grade_name, grade_info in QUALITY_GRADES.items():
            assert "min_score" in grade_info
            assert "status" in grade_info


class TestGradingFunctions:
    def test_grade_from_r_squared_excellent(self):
        grade = get_grade_from_r_squared(0.99)
        assert isinstance(grade, dict)
        assert "grade" in grade
        # Should be a high grade (A+, A, or A-)
        assert grade["grade"] in ("A+", "A", "A-")

    def test_grade_from_r_squared_good(self):
        grade = get_grade_from_r_squared(0.95)
        assert isinstance(grade, dict)
        assert "grade" in grade

    def test_grade_from_r_squared_poor(self):
        grade = get_grade_from_r_squared(0.5)
        assert isinstance(grade, dict)
        assert "grade" in grade

    def test_grade_from_score_high(self):
        grade = get_grade_from_score(95)
        assert isinstance(grade, dict)
        assert "grade" in grade

    def test_grade_from_score_low(self):
        grade = get_grade_from_score(30)
        assert isinstance(grade, dict)
        assert "grade" in grade

    def test_calibration_grade_excellent(self):
        grade = get_calibration_grade(0.999)
        assert isinstance(grade, dict)
        assert "grade" in grade

    def test_calibration_grade_poor(self):
        grade = get_calibration_grade(0.8)
        assert isinstance(grade, dict)
        assert "grade" in grade


class TestPlotSettings:
    def test_plot_template(self):
        assert isinstance(PLOT_TEMPLATE, str)

    def test_font_family(self):
        assert isinstance(FONT_FAMILY, str)

    def test_export_dpi(self):
        assert EXPORT_DPI > 0


class TestSessionStateDefaults:
    def test_default_session_state(self):
        assert isinstance(DEFAULT_SESSION_STATE, dict)

    def test_default_global_session_state(self):
        assert isinstance(DEFAULT_GLOBAL_SESSION_STATE, dict)

    def test_session_input_keys(self):
        assert isinstance(SESSION_INPUT_KEYS_TO_CLEAR, list | tuple)

    def test_session_widget_prefixes(self):
        assert isinstance(SESSION_WIDGET_PREFIXES_TO_CLEAR, list | tuple)

    def test_study_metric_data_keys(self):
        assert isinstance(STUDY_METRIC_DATA_KEYS, list | tuple)
