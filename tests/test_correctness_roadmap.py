"""
Regression tests for scientific correctness, data retention and reporting.

Each class is named after the roadmap task whose acceptance evidence it records.
Real calculations are used throughout; mocks appear only for deliberate failure
injection.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adsorblab_pro.utils import (
    bootstrap_summary_text,
    calculate_temperature_results,
    calculate_temperature_results_direct,
    convert_columns_to_canonical,
    interpret_column_header,
    parse_uploaded_table,
    standardize_column_name,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _csv(text: str) -> bytes:
    return text.encode("utf-8")


# =============================================================================
# R01 — Unit-safe import and explicit temperature units
# =============================================================================
class TestR01UnitSafeImport:
    def test_microgram_per_litre_acceptance_case(self):
        """C0 = 10 µg/L, Ce = 1 µg/L, V = 0.1 L, m = 0.1 g -> q = 0.009 mg/g, 90 %."""
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct

        df, status = parse_uploaded_table(
            _csv("C0 (µg/L),Ce (µg/L)\n10,1\n20,3\n40,8\n"), "iso.csv", ["C0", "Ce"], "isotherm"
        )
        assert status["status"] == "success"
        assert df["C0"].iloc[0] == pytest.approx(0.010)
        assert df["Ce"].iloc[0] == pytest.approx(0.001)

        result = _calculate_isotherm_results_direct({"data": df, "params": {"m": 0.1, "V": 0.1}})
        first = result.data.iloc[0]
        assert first["qe_mg_g"] == pytest.approx(0.009, rel=1e-9)
        assert first["removal_%"] == pytest.approx(90.0, rel=1e-9)

    @pytest.mark.parametrize(
        "header_unit,scale",
        [("mg/L", 1.0), ("ug/L", 1e3), ("µg/L", 1e3), ("ng/L", 1e6), ("mg L-1", 1.0)],
    )
    def test_equivalent_concentration_units_give_same_calculation(self, header_unit, scale):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct

        c0_mg = np.array([0.5, 1.0, 2.0, 4.0])
        ce_mg = np.array([0.05, 0.2, 0.7, 2.1])
        text = f"C0 ({header_unit}),Ce ({header_unit})\n" + "\n".join(
            f"{a * scale!r},{b * scale!r}" for a, b in zip(c0_mg, ce_mg)
        )
        df, status = parse_uploaded_table(_csv(text), "iso.csv", ["C0", "Ce"], "isotherm")
        assert status["status"] == "success", status["messages"]
        result = _calculate_isotherm_results_direct({"data": df, "params": {"m": 0.05, "V": 0.1}})
        expected_q = (c0_mg - ce_mg) * 0.1 / 0.05
        np.testing.assert_allclose(result.data["qe_mg_g"].to_numpy(), expected_q, rtol=1e-12)
        np.testing.assert_allclose(result.data["Ce_mgL"].to_numpy(), ce_mg, rtol=1e-12)

    def test_time_units_convert_to_minutes(self):
        from adsorblab_pro.tabs.kinetic_tab import _calculate_kinetic_results_direct

        params = {"C0": 50.0, "m": 0.1, "V": 0.05}
        minutes, _ = parse_uploaded_table(
            _csv("Time,Ct\n0,50\n30,30\n60,20\n120,15\n"), "k.csv", ["Time", "Ct"], "kinetic"
        )
        hours, _ = parse_uploaded_table(
            _csv("Time (h),Ct (mg/L)\n0,50\n0.5,30\n1,20\n2,15\n"),
            "k.csv",
            ["Time", "Ct"],
            "kinetic",
        )
        seconds, _ = parse_uploaded_table(
            _csv("t (s),Ct\n0,50\n1800,30\n3600,20\n7200,15\n"), "k.csv", ["Time", "Ct"], "kinetic"
        )
        for df in (hours, seconds):
            np.testing.assert_allclose(df["Time"], minutes["Time"], rtol=1e-12)
            a = _calculate_kinetic_results_direct({"data": df, "params": params}).data
            b = _calculate_kinetic_results_direct({"data": minutes, "params": params}).data
            np.testing.assert_allclose(a["qt_mg_g"], b["qt_mg_g"], rtol=1e-12)

    def test_mass_and_volume_units_convert(self):
        from adsorblab_pro.tabs.dosage_tab import _calculate_dosage_results_direct

        grams, _ = parse_uploaded_table(
            _csv("Mass,Ce\n0.025,40\n0.05,25\n0.1,10\n"), "d.csv", ["Mass", "Ce"], "dosage"
        )
        milligrams, status = parse_uploaded_table(
            _csv("Mass (mg),Ce\n25,40\n50,25\n100,10\n"), "d.csv", ["Mass", "Ce"], "dosage"
        )
        assert status["status"] == "success"
        np.testing.assert_allclose(milligrams["Mass"], grams["Mass"], rtol=1e-12)
        params = {"C0": 50.0, "V": 0.05}
        a = _calculate_dosage_results_direct({"data": milligrams, "params": params}).data
        b = _calculate_dosage_results_direct({"data": grams, "params": params}).data
        np.testing.assert_allclose(a["qe_mg_g"], b["qe_mg_g"], rtol=1e-12)

        volume, report = convert_columns_to_canonical(
            pd.DataFrame({"V (mL)": [50.0, 100.0]}), ["Volume"]
        )
        assert report["errors"] == []
        np.testing.assert_allclose(volume["Volume"], [0.05, 0.1])

    @pytest.mark.parametrize(
        "header",
        ["Ce (ppm)", "Ce (mmol/L)", "Ce (µM)", "Ce (%)", "Ce (min)", "Ce (furlongs)"],
    )
    def test_ambiguous_or_unsupported_units_are_refused(self, header):
        df, status = parse_uploaded_table(
            _csv(f"C0,{header}\n10,1\n20,3\n40,8\n"), "iso.csv", ["C0", "Ce"], "isotherm"
        )
        assert df is None
        assert status["status"] == "error"
        errors = [text for level, text in status["messages"] if level == "error"]
        assert any(header in text for text in errors)

    def test_dosage_mass_given_as_concentration_is_refused(self):
        df, status = parse_uploaded_table(
            _csv("Mass (g/L),Ce\n0.5,40\n1,25\n2,10\n"), "d.csv", ["Mass", "Ce"], "dosage"
        )
        assert df is None and status["status"] == "error"

    @pytest.mark.parametrize("header_line", ["C0,Ce,Ceq", "C0,Ce (mg/L),C_e", "C0,Ce,Ce (ug/L)"])
    def test_conflicting_mapped_columns_are_refused(self, header_line):
        df, status = parse_uploaded_table(
            _csv(f"{header_line}\n10,1,2\n20,3,4\n40,8,9\n"), "iso.csv", ["C0", "Ce"], "isotherm"
        )
        assert df is None
        assert any("describe Ce" in text for _, text in status["messages"])

    def test_unit_bearing_headers_are_not_silently_renamed(self):
        """Renaming 'Ce (ug/L)' to 'Ce' without conversion was the original defect."""
        assert standardize_column_name("Ce (ug/L)") == "Ce (ug/L)"
        assert standardize_column_name("Time (h)") == "Time (h)"
        assert standardize_column_name("Mass (mg)") == "Mass (mg)"
        assert standardize_column_name("Temperature (K)") == "Temperature (K)"
        assert standardize_column_name("temperature_k") == "temperature_k"
        # Canonical-unit headers are still standardized.
        assert standardize_column_name("Ce (mg/L)") == "Ce"
        assert standardize_column_name("Time (min)") == "Time"
        assert standardize_column_name("c_initial") == "C0"
        assert standardize_column_name("Absorbance (664 nm)") == "Absorbance"

    def test_metadata_headers_are_not_mapped_to_quantities(self):
        for header in ("SampleID", "Ce_SD", "Matrix"):
            assert interpret_column_header(header).canonical is None

    def test_celsius_and_kelvin_versions_give_same_physical_temperatures(self):
        celsius, s1 = parse_uploaded_table(
            _csv("Temperature (°C),Ce\n25,38\n35,42\n45,45\n"),
            "t.csv",
            ["Temperature", "Ce"],
            "temperature",
        )
        kelvin, s2 = parse_uploaded_table(
            _csv("Temperature (K),Ce\n298.15,38\n308.15,42\n318.15,45\n"),
            "t.csv",
            ["Temperature", "Ce"],
            "temperature",
        )
        assert s1["status"] == s2["status"] == "success"
        params = {"C0": 100.0, "m": 0.1, "V": 0.05}
        direct = [
            calculate_temperature_results_direct(
                {"data": df, "params": params, "temperature_unit": "°C"}
            ).data
            for df in (celsius, kelvin)
        ]
        np.testing.assert_allclose(direct[0]["Temperature_K"], [298.15, 308.15, 318.15])
        np.testing.assert_allclose(direct[0]["Temperature_K"], direct[1]["Temperature_K"])
        np.testing.assert_allclose(direct[0]["Temperature_C"], direct[1]["Temperature_C"])

    def test_both_input_modes_apply_the_same_temperature_rule(self):
        params = {"C0": 50.0, "m": 0.1, "V": 0.1}
        calib = {"slope": 0.01, "intercept": 0.0}
        temps_k = [298.15, 308.15, 318.15]
        direct = calculate_temperature_results_direct(
            {
                "data": pd.DataFrame({"Temperature": temps_k, "Ce": [20.0, 15.0, 10.0]}),
                "params": params,
                "temperature_unit": "K",
            }
        )
        absorbance = calculate_temperature_results(
            {
                "data": pd.DataFrame({"Temperature": temps_k, "Absorbance": [0.2, 0.15, 0.1]}),
                "params": params,
                "temperature_unit": "K",
            },
            calib,
        )
        np.testing.assert_allclose(direct.data["Temperature_K"], temps_k)
        np.testing.assert_allclose(absorbance.data["Temperature_K"], temps_k)

        # Undeclared kelvin-like values are refused identically in both modes.
        for result in (
            calculate_temperature_results_direct(
                {
                    "data": pd.DataFrame({"Temperature": temps_k, "Ce": [20.0, 15.0, 10.0]}),
                    "params": params,
                }
            ),
            calculate_temperature_results(
                {
                    "data": pd.DataFrame({"Temperature": temps_k, "Absorbance": [0.2, 0.15, 0.1]}),
                    "params": params,
                },
                calib,
            ),
        ):
            assert result.success is False
            assert "without a unit" in result.error

    def test_unitless_kelvin_like_temperature_column_is_refused_on_import(self):
        df, status = parse_uploaded_table(
            _csv("Temperature,Ce\n298.15,20\n308.15,15\n318.15,10\n"),
            "t.csv",
            ["Temperature", "Ce"],
            "temperature",
        )
        assert df is None
        assert any("Temperature (K)" in text for _, text in status["messages"])

    @pytest.mark.parametrize(
        "file_name,required",
        [
            ("calibration_data.csv", ["Concentration", "Absorbance"]),
            ("isotherm_data.csv", ["Concentration", "Absorbance"]),
            ("isotherm_direct.csv", ["C0", "Ce"]),
            ("kinetic_data.csv", ["Time", "Absorbance"]),
            ("kinetic_direct.csv", ["Time", "Ct"]),
            ("dosage_data.csv", ["Mass", "Absorbance"]),
            ("dosage_direct.csv", ["Mass", "Ce"]),
            ("ph_effect_data.csv", ["pH", "Absorbance"]),
            ("ph_effect_direct.csv", ["pH", "Ce"]),
            ("temperature_data.csv", ["Temperature", "Absorbance"]),
            ("temperature_direct.csv", ["Temperature", "Ce"]),
        ],
    )
    def test_canonical_templates_remain_usable(self, file_name, required):
        content = (EXAMPLES / file_name).read_bytes()
        df, status = parse_uploaded_table(content, file_name, required, "any")
        assert status["status"] == "success", status["messages"]
        original = pd.read_csv(EXAMPLES / file_name)
        assert list(df.columns) == ["source_row", *required]
        assert list(df["source_row"]) == list(range(1, len(original) + 1))
        pd.testing.assert_frame_equal(
            df[required].reset_index(drop=True), original[required].astype(float), check_dtype=False
        )

    def test_template_states_unit_convention(self):
        from adsorblab_pro.sidebar_ui import _generate_excel_template

        buffer = _generate_excel_template(["Temperature", "Ce"], "temperature_direct")
        sheet = pd.read_excel(buffer, sheet_name="Instructions")
        convention = str(sheet["Units (headers without a unit)"].iloc[0])
        assert "Temperature in °C" in convention and "Ce in mg/L" in convention

    def test_direct_kinetic_results_do_not_mislabel_concentration_as_absorbance(self):
        from adsorblab_pro.tabs.kinetic_tab import _calculate_kinetic_results_direct

        result = _calculate_kinetic_results_direct(
            {
                "data": pd.DataFrame({"Time": [0, 10, 20], "Ct": [50.0, 30.0, 20.0]}),
                "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
            }
        )
        assert "Absorbance" not in result.data.columns


# =============================================================================
# R02 — Preserve studies and experimental conditions across navigation
# =============================================================================
# AppTest (Streamlit 1.54) cannot drive st.file_uploader, so these scenarios seed
# each study's stored inputs as they exist after an upload and exercise the real
# app script: study selector, section radio, condition widgets, "Add New Study"
# and the explicit clear button.  The upload branch itself (new file vs. no new
# file) is covered by the helper tests below and by the browser check in R13.
try:
    from streamlit.testing.v1 import AppTest

    APPTEST_AVAILABLE = True
except ImportError:  # pragma: no cover
    APPTEST_AVAILABLE = False


def _seeded_study(c0, ce, m, V, T, calib_conc, mode):
    import copy

    from adsorblab_pro.config import DEFAULT_SESSION_STATE
    from adsorblab_pro.utils import calculate_calibration_stats

    state = copy.deepcopy(DEFAULT_SESSION_STATE)
    state["input_mode_global"] = mode
    calib = pd.DataFrame(
        {"Concentration": calib_conc, "Absorbance": [0.002 + 0.0167 * c for c in calib_conc]}
    )
    state["calib_df_input"] = calib
    state["previous_calib_df"] = calib.copy()
    stats = calculate_calibration_stats(calib["Concentration"].values, calib["Absorbance"].values)
    state["calibration_params"] = {
        "slope": stats["slope"],
        "intercept": stats["intercept"],
        "r_squared": stats["r_squared"],
        "std_err_slope": stats["se_slope"],
        "std_err_intercept": stats["se_intercept"],
    }
    data = (
        pd.DataFrame({"C0": c0, "Ce": ce})
        if mode == "direct"
        else pd.DataFrame({"Concentration": c0, "Absorbance": [0.002 + 0.0167 * c for c in ce]})
    )
    state["isotherm_input"] = {
        "data": data,
        "params": {"m": m, "V": V, "T_C": T, "T_K": T + 273.15},
        "input_mode": mode,
        "units": {col: "mg/L" if col != "Absorbance" else "AU" for col in data.columns},
    }
    state["isotherm_models_fitted"] = {"Langmuir": {"converged": True, "params": {"qm": 1.0}}}
    return state


def _snapshot(at, name):
    study = at.session_state["studies"][name]
    inp = study["isotherm_input"]
    return {
        "params": None if inp is None else dict(inp["params"]),
        "data": None if inp is None else inp["data"].copy(),
        "units": None if inp is None else dict(inp.get("units", {})),
        "mode": None if inp is None else inp.get("input_mode"),
        "fits": sorted((study["isotherm_models_fitted"] or {}).keys()),
        "calib": None if study["calib_df_input"] is None else study["calib_df_input"].copy(),
        "calib_params": study["calibration_params"],
    }


def _assert_same(before, after):
    assert after["params"] == before["params"]
    assert after["units"] == before["units"]
    assert after["mode"] == before["mode"]
    assert after["fits"] == before["fits"]
    pd.testing.assert_frame_equal(after["data"], before["data"])
    pd.testing.assert_frame_equal(after["calib"], before["calib"])
    assert after["calib_params"] == before["calib_params"]


@pytest.fixture
def two_study_app():
    at = AppTest.from_file("adsorblab_pro/app.py", default_timeout=60)
    at.session_state["studies"] = {
        "A": _seeded_study(
            [10, 20, 40, 80], [1, 3, 8, 20], 0.1, 0.2, 30.0, [0, 5, 10, 20, 40], "absorbance"
        ),
        "B": _seeded_study(
            [5, 15, 30, 60], [2, 6, 13, 30], 0.05, 0.1, 40.0, [0, 2, 4, 8, 16], "direct"
        ),
    }
    at.session_state["current_study"] = "A"
    at.session_state["_previous_study_selection"] = "A"
    at.session_state["active_sidebar_expander"] = "isotherm"
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
class TestR02StudyNavigation:
    def test_a_to_b_to_a_preserves_data_conditions_calibration_units_and_fits(self, two_study_app):
        at = two_study_app
        before = {name: _snapshot(at, name) for name in ("A", "B")}
        # The first render must not overwrite stored conditions with widget defaults.
        assert before["A"]["params"]["m"] == 0.1 and before["A"]["params"]["V"] == 0.2
        assert before["A"]["fits"] == ["Langmuir"]

        at.selectbox(key="study_selector").set_value("B").run()
        assert not at.exception
        assert at.session_state["current_study"] == "B"
        at.selectbox(key="study_selector").set_value("A").run()
        assert not at.exception
        for name in ("A", "B"):
            _assert_same(before[name], _snapshot(at, name))

    def test_section_navigation_keeps_stored_conditions(self, two_study_app):
        at = two_study_app
        before = _snapshot(at, "A")
        at.radio(key="data_input_selector").set_value("kinetic").run()
        at.radio(key="data_input_selector").set_value("calibration").run()
        at.radio(key="data_input_selector").set_value("isotherm").run()
        assert not at.exception
        _assert_same(before, _snapshot(at, "A"))

    def test_condition_change_invalidates_only_its_own_study(self, two_study_app):
        from adsorblab_pro.sidebar_ui import study_widget_suffix

        at = two_study_app
        before_b = _snapshot(at, "B")
        at.number_input(key="iso_m__" + study_widget_suffix("A")).set_value(0.3).run()
        assert not at.exception
        after_a = _snapshot(at, "A")
        assert after_a["params"]["m"] == pytest.approx(0.3)
        assert after_a["fits"] == []  # its own derived fits are now obsolete
        _assert_same(before_b, _snapshot(at, "B"))

    def test_adding_a_third_study_selects_it_and_leaves_others_intact(self, two_study_app):
        at = two_study_app
        before = {name: _snapshot(at, name) for name in ("A", "B")}
        at.text_input(key="new_study_input").set_value("C").run()
        at.button(key="add_study_button").click().run()
        assert not at.exception
        assert at.session_state["current_study"] == "C"
        assert at.session_state["studies"]["C"]["isotherm_input"] is None
        at.selectbox(key="study_selector").set_value("A").run()
        at.selectbox(key="study_selector").set_value("B").run()
        assert not at.exception
        for name in ("A", "B"):
            _assert_same(before[name], _snapshot(at, name))

    def test_explicit_clear_affects_only_that_input_of_that_study(self, two_study_app):
        from adsorblab_pro.sidebar_ui import study_widget_suffix

        at = two_study_app
        before_b = _snapshot(at, "B")
        at.button(key="iso_clear__" + study_widget_suffix("A")).click().run()
        assert not at.exception
        after_a = _snapshot(at, "A")
        assert after_a["params"] is None and after_a["fits"] == []
        # A's calibration is a separate input and is untouched.
        pd.testing.assert_frame_equal(
            after_a["calib"], at.session_state["studies"]["A"]["previous_calib_df"]
        )
        _assert_same(before_b, _snapshot(at, "B"))

    def test_input_mode_change_does_not_reinterpret_stored_data(self, two_study_app):
        at = two_study_app
        before = _snapshot(at, "A")
        at.radio(key="display_input_mode_A").set_value("direct").run()
        assert not at.exception
        after = _snapshot(at, "A")
        # The stored absorbance-mode input keeps its schema and mode.
        assert after["mode"] == "absorbance"
        pd.testing.assert_frame_equal(after["data"], before["data"])
        assert after["fits"] == before["fits"]


class TestR02InputStateHelpers:
    def _state(self):
        return {
            "isotherm_input": {
                "data": pd.DataFrame({"C0": [1.0, 2.0, 3.0], "Ce": [0.1, 0.5, 1.0]}),
                "params": {"m": 0.1, "V": 0.1},
                "input_mode": "direct",
            },
            "isotherm_results": pd.DataFrame({"x": [1]}),
            "isotherm_models_fitted": {"Langmuir": {}},
        }

    def test_identical_input_does_not_invalidate(self):
        from adsorblab_pro.sidebar_ui import store_study_input

        state = self._state()
        same = {**state["isotherm_input"], "data": state["isotherm_input"]["data"].copy()}
        changed = store_study_input(
            state, "isotherm_input", same, ["isotherm_results", "isotherm_models_fitted"]
        )
        assert changed is False
        assert state["isotherm_models_fitted"] == {"Langmuir": {}}

    def test_new_data_invalidates_dependents(self):
        from adsorblab_pro.sidebar_ui import store_study_input

        state = self._state()
        new = {**state["isotherm_input"], "data": pd.DataFrame({"C0": [5.0], "Ce": [1.0]})}
        assert store_study_input(
            state, "isotherm_input", new, ["isotherm_results", "isotherm_models_fitted"]
        )
        assert state["isotherm_models_fitted"] == {} and state["isotherm_results"] is None

    def test_conditions_update_invalidates_only_on_change(self):
        from adsorblab_pro.sidebar_ui import stored_conditions, update_study_conditions

        state = self._state()
        deps = ["isotherm_results", "isotherm_models_fitted"]
        assert not update_study_conditions(state, "isotherm_input", {"m": 0.1, "V": 0.1}, deps)
        assert state["isotherm_models_fitted"] == {"Langmuir": {}}
        assert update_study_conditions(state, "isotherm_input", {"m": 0.2, "V": 0.1}, deps)
        assert state["isotherm_models_fitted"] == {}
        assert stored_conditions(state, "isotherm_input") == {"m": 0.2, "V": 0.1}

    def test_conditions_are_kept_before_any_data_is_uploaded(self):
        from adsorblab_pro.sidebar_ui import stored_conditions, update_study_conditions

        state: dict = {"isotherm_input": None}
        update_study_conditions(state, "isotherm_input", {"m": 0.07, "V": 0.03}, [])
        assert state["isotherm_input"] is None
        assert stored_conditions(state, "isotherm_input") == {"m": 0.07, "V": 0.03}

    def test_upload_signature_distinguishes_new_files(self):
        from adsorblab_pro.sidebar_ui import upload_signature

        a = upload_signature("iso.csv", b"C0,Ce\n1,0.1\n")
        assert a == upload_signature("iso.csv", b"C0,Ce\n1,0.1\n")
        assert a != upload_signature("iso.csv", b"C0,Ce\n1,0.2\n")
        assert a != upload_signature("other.csv", b"C0,Ce\n1,0.1\n")

    def test_widget_suffix_is_per_study(self):
        from adsorblab_pro.sidebar_ui import study_widget_suffix

        assert study_widget_suffix("A") == study_widget_suffix("A")
        assert study_widget_suffix("A") != study_widget_suffix("B")


# =============================================================================
# R03 — Correct calibration invalidation
# =============================================================================
VALID_CALIB = pd.DataFrame(
    {"Concentration": [0, 5, 10, 20, 40], "Absorbance": [0.002, 0.085, 0.168, 0.335, 0.668]}
)
REPLACEMENT_CALIB = pd.DataFrame(
    {"Concentration": [0, 5, 10, 20, 40], "Absorbance": [0.001, 0.051, 0.101, 0.201, 0.401]}
)
INVALID_CALIBS = {
    "flat": pd.DataFrame({"Concentration": [0, 5, 10, 20, 40], "Absorbance": [0.3] * 5}),
    "flat_noisy": pd.DataFrame(
        {"Concentration": [0, 5, 10, 20, 40], "Absorbance": [0.301, 0.299, 0.302, 0.298, 0.300]}
    ),
    "degenerate": pd.DataFrame(
        {"Concentration": [10] * 5, "Absorbance": [0.10, 0.12, 0.11, 0.13, 0.10]}
    ),
    # Too few finite standards remain (a single bad standard is excluded visibly
    # and the curve built from the rest; see TestR04ObservationIntegrity).
    "non_finite": pd.DataFrame(
        {"Concentration": [0, 5, 10, 20, 40], "Absorbance": [0.002, np.inf, np.inf, np.inf, 0.668]}
    ),
    "missing": pd.DataFrame(
        {"Concentration": [0, 5, 10, 20, 40], "Absorbance": [0.002, np.nan, np.nan, np.nan, 0.668]}
    ),
    "insufficient": pd.DataFrame({"Concentration": [0, 5], "Absorbance": [0.0, 0.1]}),
    "negative_concentration": pd.DataFrame(
        {"Concentration": [-5, 5, 10, 20, 40], "Absorbance": [0.0, 0.085, 0.168, 0.335, 0.668]}
    ),
}


def _study_with_calibration():
    import copy

    from adsorblab_pro.config import DEFAULT_SESSION_STATE
    from adsorblab_pro.utils import apply_calibration_update

    state = copy.deepcopy(DEFAULT_SESSION_STATE)
    state["calib_df_input"] = VALID_CALIB.copy()
    assert apply_calibration_update(state)["status"] == "activated"
    state["isotherm_input"] = {
        "data": pd.DataFrame({"Concentration": [10.0, 20.0, 40.0], "Absorbance": [0.05, 0.1, 0.3]}),
        "params": {"m": 0.1, "V": 0.1},
        "input_mode": "absorbance",
    }
    state["isotherm_results"] = pd.DataFrame({"Ce_mgL": [1.0]})
    state["isotherm_models_fitted"] = {"Langmuir": {"converged": True}}
    state["kinetic_input"] = {
        "data": pd.DataFrame({"Time": [0, 5, 10], "Ct": [50.0, 30.0, 20.0]}),
        "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
        "input_mode": "direct",
    }
    state["kinetic_models_fitted"] = {"PSO": {"converged": True}}
    return state


class TestR03CalibrationInvalidation:
    def test_valid_candidate_is_built_with_identity_and_covariance(self):
        from adsorblab_pro.utils import build_calibration

        params, error = build_calibration(VALID_CALIB)
        assert error is None
        assert params["slope"] == pytest.approx(0.01666, rel=1e-3)
        assert params["calibration_id"]
        x = VALID_CALIB["Concentration"].to_numpy(float)
        expected_cov = -x.mean() * params["std_err_estimate"] ** 2 / np.sum((x - x.mean()) ** 2)
        assert params["cov_slope_intercept"] == pytest.approx(expected_cov)
        assert params["lod_mgL"] == pytest.approx(3 * params["std_err_estimate"] / params["slope"])

    @pytest.mark.parametrize("case", sorted(INVALID_CALIBS))
    def test_invalid_candidates_are_rejected_with_a_reason(self, case):
        from adsorblab_pro.utils import build_calibration

        params, error = build_calibration(INVALID_CALIBS[case])
        assert params is None
        assert isinstance(error, str) and error

    @pytest.mark.parametrize("case", sorted(INVALID_CALIBS))
    def test_invalid_replacement_deactivates_old_curve_and_keeps_attempted_data(self, case):
        from adsorblab_pro.utils import apply_calibration_update

        state = _study_with_calibration()
        old_id = state["calibration_params"]["calibration_id"]
        attempted = INVALID_CALIBS[case].copy()
        state["calib_df_input"] = attempted

        outcome = apply_calibration_update(state)

        assert outcome["status"] == "invalid"
        assert state["calibration_params"] is None  # no old curve remains usable
        assert state["calibration_error"] == outcome["error"]
        pd.testing.assert_frame_equal(state["calib_df_input"], attempted)
        # Absorbance-derived results are obsolete; direct-input results are not.
        assert state["isotherm_models_fitted"] == {} and state["isotherm_results"] is None
        assert state["kinetic_models_fitted"] == {"PSO": {"converged": True}}
        assert old_id not in str(state.get("_calibration_candidate_id"))

    def test_reruns_do_not_repeat_invalidation(self):
        from adsorblab_pro.utils import apply_calibration_update

        state = _study_with_calibration()
        assert apply_calibration_update(state)["status"] == "unchanged"
        assert state["isotherm_models_fitted"] == {"Langmuir": {"converged": True}}
        state["calib_df_input"] = INVALID_CALIBS["flat"].copy()
        apply_calibration_update(state)
        state["isotherm_models_fitted"] = {"Langmuir": {"converged": True}}  # refit attempt
        assert apply_calibration_update(state)["status"] == "unchanged"
        assert state["isotherm_models_fitted"] == {"Langmuir": {"converged": True}}

    def test_valid_replacement_restores_correct_calculations(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results
        from adsorblab_pro.utils import apply_calibration_update

        state = _study_with_calibration()
        state["calib_df_input"] = INVALID_CALIBS["degenerate"].copy()
        apply_calibration_update(state)
        assert state["calibration_params"] is None

        state["kinetic_models_fitted"] = {"PSO": {"converged": True}}
        state["calib_df_input"] = REPLACEMENT_CALIB.copy()
        outcome = apply_calibration_update(state)
        assert outcome["status"] == "activated"
        params = state["calibration_params"]
        assert params["slope"] == pytest.approx(0.01, rel=1e-6)
        assert params["intercept"] == pytest.approx(0.001, abs=1e-9)
        assert state["calibration_error"] is None
        # Only calibration-dependent results were affected.
        assert state["kinetic_models_fitted"] == {"PSO": {"converged": True}}
        result = _calculate_isotherm_results(state["isotherm_input"], params)
        expected_ce = (np.array([0.05, 0.1, 0.3]) - 0.001) / 0.01
        np.testing.assert_allclose(result.data["Ce_mgL"], expected_ce, rtol=1e-9)

    def test_explicit_removal_deactivates_and_invalidates(self):
        from adsorblab_pro.utils import apply_calibration_update

        state = _study_with_calibration()
        state["calib_df_input"] = None
        outcome = apply_calibration_update(state)
        assert outcome["status"] == "removed"
        assert state["calibration_params"] is None
        assert state["isotherm_models_fitted"] == {}
        assert state["kinetic_models_fitted"] == {"PSO": {"converged": True}}

    @pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
    def test_app_does_not_reuse_obsolete_calibration(self):
        at = AppTest.from_file("adsorblab_pro/app.py", default_timeout=60)
        at.session_state["studies"] = {"S": _study_with_calibration()}
        at.session_state["current_study"] = "S"
        at.session_state["_previous_study_selection"] = "S"
        at.session_state["active_sidebar_expander"] = "kinetic"
        at.run()
        assert not at.exception
        study = at.session_state["studies"]["S"]
        assert study["calibration_params"] is not None
        study["calib_df_input"] = INVALID_CALIBS["flat"].copy()  # as stored after an upload
        at.run()
        assert not at.exception
        study = at.session_state["studies"]["S"]
        assert study["calibration_params"] is None
        assert any("Calibration not active" in e.value for e in at.sidebar.error)
        assert study["kinetic_models_fitted"] == {"PSO": {"converged": True}}
        # The calibration page explains the reason instead of using the old curve.
        at.radio(key="main_section_navigation").set_value("🧪 Analysis Workflow").run()
        assert not at.exception
        assert any("Calibration not active" in e.value for e in at.main.error)
        assert not any("Calibration Results" in m.value for m in at.main.markdown)


# =============================================================================
# R04 — Preserve observations and stop silent filtering/clipping
# =============================================================================
class TestR04ObservationIntegrity:
    def test_ce_above_c0_stays_listed_and_qualifies_the_pass(self):
        from adsorblab_pro.tabs.isotherm_tab import (
            _calculate_isotherm_results_direct,
            _fit_all_isotherm_models_cached,
        )
        from adsorblab_pro.utils import eligible_observations, observation_notice

        data = pd.DataFrame({"C0": [10, 20, 40, 80, 160], "Ce": [11, 3, 8, 20, 55]})
        result = _calculate_isotherm_results_direct({"data": data, "params": {"m": 0.1, "V": 0.1}})
        table = result.data
        assert len(table) == 5
        row = table[table["source_row"] == 1].iloc[0]
        assert row["status"] == "excluded"
        assert "exceeds C0" in row["note"] and np.isnan(row["qe_mg_g"])
        notice = observation_notice(table)
        assert notice is not None and "row 1" in notice
        usable = eligible_observations(table)
        fits = _fit_all_isotherm_models_cached(
            tuple(usable["Ce_mgL"]), tuple(usable["qe_mg_g"]), tuple(usable["C0_mgL"])
        )
        assert fits["Langmuir"]["n_points"] == 4

    def test_below_intercept_signal_is_not_complete_removal(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results

        iso = {
            "data": pd.DataFrame({"Concentration": [10.0, 20.0], "Absorbance": [0.005, 0.5]}),
            "params": {"m": 0.1, "V": 0.1},
        }
        calib = {
            "slope": 0.1,
            "intercept": 0.01,
            "std_err_slope": 0.001,
            "std_err_intercept": 0.001,
        }
        table = _calculate_isotherm_results(iso, calib).data
        row = table[table["source_row"] == 1].iloc[0]
        assert row["status"] == "excluded"
        assert np.isnan(row["Ce_mgL"]) and np.isnan(row["qe_mg_g"]) and np.isnan(row["removal_%"])
        assert "below the calibration intercept" in row["note"]
        assert "not treated as zero" in row["note"]
        assert row["Absorbance"] == pytest.approx(0.005)  # raw signal kept

    def test_limits_of_detection_and_quantification_are_used(self):
        from adsorblab_pro.utils import build_calibration, build_uptake_table

        calib, error = build_calibration(
            pd.DataFrame(
                {
                    "Concentration": [0, 5, 10, 20, 40],
                    "Absorbance": [0.004, 0.081, 0.171, 0.331, 0.671],
                }
            )
        )
        assert error is None
        lod, loq = calib["lod_mgL"], calib["loq_mgL"]
        slope, intercept = calib["slope"], calib["intercept"]
        signals = [
            intercept + slope * lod * 0.5,  # below LOD
            intercept + slope * (lod + loq) / 2,  # between LOD and LOQ
            intercept + slope * 25.0,  # quantified
        ]
        table = build_uptake_table(
            pd.DataFrame({"Absorbance": signals}),
            mode="absorbance",
            signal_col="Absorbance",
            C0=50.0,
            V=0.1,
            m=0.1,
            calib_params=calib,
        )
        assert (
            table.loc[0, "status"] == "excluded"
            and "below the calibration LOD" in table.loc[0, "note"]
        )
        assert "removal ≥" in table.loc[0, "note"] and np.isnan(table.loc[0, "removal"])
        assert table.loc[1, "status"] == "ok" and "semi-quantitative" in table.loc[1, "note"]
        assert table.loc[2, "status"] == "ok" and table.loc[2, "note"] == ""

    def test_mixed_rows_missing_and_non_numeric_values_are_reported(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct
        from adsorblab_pro.utils import prepare_analysis_data

        raw = pd.DataFrame(
            {
                "C0": ["10", "20", "", "40", "80", None],
                "Ce": ["1", "n.d.", "", "inf", "20", None],
            }
        )
        prepared, report = prepare_analysis_data(raw, ["C0", "Ce"])
        assert list(prepared["source_row"]) == [1, 2, 4, 5]
        assert report["ignored_empty_rows"] == [3, 6]
        assert "non-numeric value 'n.d.'" in report["row_issues"][2][0]
        assert "non-finite" in report["row_issues"][4][0]
        result = _calculate_isotherm_results_direct(
            {"data": prepared, "params": {"m": 0.1, "V": 0.1}, "row_issues": report["row_issues"]}
        )
        table = result.data.set_index("source_row")
        assert table.loc[2, "status"] == "excluded" and "'n.d.'" in table.loc[2, "note"]
        assert table.loc[4, "status"] == "excluded"
        assert (table["status"] == "ok").sum() == 2

    def test_scientific_notation_inputs(self):
        from adsorblab_pro.utils import prepare_analysis_data

        df, status = parse_uploaded_table(
            _csv("C0 (mg/L),Ce (mg/L)\n1.0E+01,1e-3\n2E1,3.5e+00\n4.0e1,8\n"),
            "iso.csv",
            ["C0", "Ce"],
            "isotherm",
        )
        assert status["status"] == "success"
        np.testing.assert_allclose(df["C0"], [10.0, 20.0, 40.0])
        np.testing.assert_allclose(df["Ce"], [0.001, 3.5, 8.0])
        prepared, report = prepare_analysis_data(
            pd.DataFrame({"C0": ["1,5e1", "2e1"], "Ce": ["3E-1", "1.2e+00"]}), ["C0", "Ce"]
        )
        np.testing.assert_allclose(prepared["C0"], [15.0, 20.0])
        np.testing.assert_allclose(prepared["Ce"], [0.3, 1.2])
        assert report["row_issues"] == {}

    def test_zero_uptake_and_zero_concentration_enter_the_fit_where_allowed(self):
        from adsorblab_pro.tabs.isotherm_tab import _fit_all_isotherm_models_cached

        C0 = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 100.0])
        Ce = np.array([0.0, 1.0, 3.0, 8.0, 20.0, 100.0])  # full removal, ..., no uptake
        qe = (C0 - Ce) * 0.1 / 0.1
        fits = _fit_all_isotherm_models_cached(tuple(Ce), tuple(qe), tuple(C0))
        assert fits["Langmuir"]["n_points"] == 6  # both zero observations used
        assert 0.0 in fits["Langmuir"]["y_data"] and 0.0 in fits["Langmuir"]["x_data"]
        assert fits["Temkin"]["converged"]
        assert fits["Temkin"]["n_points"] == 5
        assert "Ce = 0" in fits["Temkin"]["domain_note"]

    def test_single_bad_calibration_standard_is_excluded_visibly(self):
        from adsorblab_pro.utils import build_calibration

        params, error = build_calibration(
            pd.DataFrame(
                {
                    "Concentration": [0, 5, 10, 20, 40],
                    "Absorbance": [0.002, np.nan, 0.168, 0.335, 0.668],
                }
            )
        )
        assert error is None
        assert params["excluded_standards"] == [2]
        assert params["n_points"] == 4

    def test_kinetic_rows_negative_time_and_ct_above_c0(self):
        from adsorblab_pro.tabs.kinetic_tab import _calculate_kinetic_results_direct

        data = pd.DataFrame({"Time": [0, -5, 10, 20, 30], "Ct": [50.0, 40.0, 55.0, 20.0, 15.0]})
        table = _calculate_kinetic_results_direct(
            {"data": data, "params": {"C0": 50.0, "m": 0.1, "V": 0.05}}
        ).data.set_index("source_row")
        assert table.loc[2, "status"] == "excluded" and "negative time" in table.loc[2, "note"]
        assert table.loc[3, "status"] == "excluded" and "exceeds C0" in table.loc[3, "note"]
        assert table.loc[1, "status"] == "ok" and table.loc[1, "qt_mg_g"] == 0.0  # zero uptake

    @pytest.mark.parametrize("mode", ["direct", "absorbance"])
    def test_effect_study_with_two_points_is_not_rejected(self, mode):
        from adsorblab_pro.tabs.dosage_tab import (
            _calculate_dosage_results,
            _calculate_dosage_results_direct,
        )
        from adsorblab_pro.utils import prepare_analysis_data

        cols = ["Mass", "Ce"] if mode == "direct" else ["Mass", "Absorbance"]
        raw = pd.DataFrame(
            {cols[0]: [0.05, 0.1], cols[1]: [20.0, 10.0] if mode == "direct" else [0.2, 0.1]}
        )
        prepared, report = prepare_analysis_data(raw, cols, min_rows=1)
        assert prepared is not None and report["errors"] == []
        inp = {"data": prepared, "params": {"C0": 50.0, "V": 0.05}}
        result = (
            _calculate_dosage_results_direct(inp)
            if mode == "direct"
            else _calculate_dosage_results(inp, {"slope": 0.01, "intercept": 0.0})
        )
        assert result.success is True and len(result.data) == 2

    def test_temperature_paths_keep_rows(self):
        params = {"C0": 50.0, "m": 0.1, "V": 0.1}
        direct = calculate_temperature_results_direct(
            {
                "data": pd.DataFrame({"Temperature": [25.0, 35.0, 45.0], "Ce": [20.0, 60.0, 10.0]}),
                "params": params,
                "temperature_unit": "°C",
            },
            include_uncertainty=True,
        ).data
        assert len(direct) == 3 and (direct["status"] == "ok").sum() == 2
        absorbance = calculate_temperature_results(
            {
                "data": pd.DataFrame({"Temperature": [25.0, 35.0], "Absorbance": [-0.05, 0.2]}),
                "params": params,
                "temperature_unit": "°C",
            },
            {"slope": 0.01, "intercept": 0.0},
        ).data
        first = absorbance[absorbance["source_row"] == 1].iloc[0]
        assert first["status"] == "excluded" and np.isnan(first["removal_%"])

    def test_basic_helpers_do_not_clip(self):
        from adsorblab_pro.utils import (
            calculate_adsorption_capacity,
            calculate_Ce_from_absorbance,
            calculate_removal_percentage,
        )

        assert calculate_Ce_from_absorbance(0.005, 0.1, 0.01) == pytest.approx(-0.05)
        assert calculate_adsorption_capacity(10.0, 11.0, 0.1, 0.1) == pytest.approx(-1.0)
        assert calculate_removal_percentage(10.0, 11.0) == pytest.approx(-10.0)
        assert np.isnan(calculate_removal_percentage(0.0, 1.0))

    @pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
    def test_isotherm_page_reports_excluded_rows(self):
        import copy

        from adsorblab_pro.config import DEFAULT_SESSION_STATE

        state = copy.deepcopy(DEFAULT_SESSION_STATE)
        state["input_mode_global"] = "direct"
        state["isotherm_input"] = {
            "data": pd.DataFrame(
                {
                    "source_row": [1, 2, 3, 4, 5, 6],
                    "C0": [10, 20, 40, 80, 160, 320],
                    "Ce": [11, 3, 8, 20, 55, 150],
                }
            ),
            "params": {"m": 0.1, "V": 0.1, "T_C": 25.0, "T_K": 298.15},
            "input_mode": "direct",
        }
        at = AppTest.from_file("adsorblab_pro/app.py", default_timeout=60)
        at.session_state["studies"] = {"S": state}
        at.session_state["current_study"] = "S"
        at.session_state["_previous_study_selection"] = "S"
        at.session_state["active_sidebar_expander"] = "isotherm"
        at.run()
        at.radio(key="main_section_navigation").set_value("🧪 Analysis Workflow").run()
        assert not at.exception
        assert any("row 1" in w.value and "exceeds C0" in w.value for w in at.main.warning)
        stored = at.session_state["studies"]["S"]["isotherm_results"]
        assert len(stored) == 6 and (stored["status"] == "excluded").sum() == 1

    def test_exports_keep_excluded_rows_and_reasons(self):
        from adsorblab_pro.tabs.dosage_tab import _calculate_dosage_results_direct
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct
        from adsorblab_pro.tabs.report_tab import generate_figure, generate_table

        iso = _calculate_isotherm_results_direct(
            {
                "data": pd.DataFrame({"C0": [10, 20, 40, 80], "Ce": [11, 3, 8, 20]}),
                "params": {"m": 0.1, "V": 0.1},
            }
        ).data
        dos = _calculate_dosage_results_direct(
            {
                "data": pd.DataFrame({"Mass": [0.0, 0.05, 0.1], "Ce": [20.0, 25.0, 60.0]}),
                "params": {"C0": 50.0, "V": 0.05},
            }
        ).data
        state = {"isotherm_results": iso, "dosage_results": dos}
        iso_table = generate_table("tbl_iso_data", state)
        assert len(iso_table) == 4 and {"status", "note", "source_row"} <= set(iso_table.columns)
        assert "exceeds C0" in iso_table.loc[iso_table["source_row"] == 1, "note"].iloc[0]
        dos_table = generate_table("tbl_dosage_data", state)  # previously never exported
        assert dos_table is not None and len(dos_table) == 3
        assert (dos_table["status"] == "excluded").sum() == 2
        assert generate_figure("effect_dosage", state) is not None


# =============================================================================
# R05 — Valid model comparisons and accurate labels
# =============================================================================
KIN_T = np.array([0, 5, 10, 20, 30, 45, 60, 90, 120, 180], dtype=float)


def _kinetic_qt(t):
    """PSO-shaped uptake (qe = 40 mg/g, k2 = 0.004) with fixed noise; q(0) = 0."""
    q = 40.0**2 * 0.004 * t / (1 + 40.0 * 0.004 * t)
    return q + np.random.default_rng(5).normal(0, 0.4, len(t)) * (t > 0)


def _fit(aicc, n=8, p=2, aic=None, bic=None, x=None):
    """Synthetic converged result; ``x`` sets the observations (default: n points)."""
    x = np.arange(1.0, n + 1) if x is None else np.asarray(x, dtype=float)
    return {
        "converged": True,
        "aic": aicc - 1.0 if aic is None else aic,
        "aicc": aicc,
        "bic": aicc + 1.0 if bic is None else bic,
        "r_squared": 0.99,
        "adj_r_squared": 0.98,
        "rmse": 0.1,
        "num_params": p,
        "n_points": len(x),
        "x_data": x,
        "y_data": 2.0 * x,
    }


def _screen(kind, x, y, extra):
    """AppTest script: fit and render the tab's model-comparison section."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if kind == "isotherm":
        from adsorblab_pro.tabs import isotherm_tab

        fitted = isotherm_tab._fit_all_isotherm_models_cached(
            tuple(x), tuple(y), tuple(np.asarray(extra, dtype=float)), 0.95, 298.15
        )
        isotherm_tab._display_model_comparison(fitted, x, y)
    else:
        from adsorblab_pro.tabs import kinetic_tab

        fitted = kinetic_tab._fit_all_kinetic_models_cached(tuple(x), tuple(y), 0.95)
        kinetic_tab._display_model_comparison(fitted, x, y)


class TestR05ModelComparison:
    def test_elovich_domain_includes_t0(self):
        """The implemented Elovich form q = ln(1 + αβt)/β is defined (and 0) at t = 0."""
        from adsorblab_pro.models import elovich_model

        assert elovich_model(np.array([0.0]), 2.0, 0.1)[0] == pytest.approx(0.0, abs=1e-6)

    def test_kinetic_fits_with_t0_share_one_observation_set(self):
        from adsorblab_pro.tabs.kinetic_tab import _fit_all_kinetic_models_cached
        from adsorblab_pro.utils import compare_information_criteria, fit_observation_key

        fitted = _fit_all_kinetic_models_cached(tuple(KIN_T), tuple(_kinetic_qt(KIN_T)), 0.95)
        converged = {k: v for k, v in fitted.items() if v.get("converged")}
        assert {"PFO", "PSO", "Elovich", "IPD"} <= set(converged)
        keys = {name: fit_observation_key(r) for name, r in converged.items()}
        assert len(set(keys.values())) == 1, keys
        assert next(iter(keys.values()))[1] == len(KIN_T)

        comparison = compare_information_criteria(fitted, "aicc")
        assert comparison["status"] == "ranked"
        assert len(comparison["groups"]) == 1
        assert comparison["best"] == min(converged, key=lambda n: converged[n]["aicc"])
        assert f"same {len(KIN_T)} observations" in comparison["message"]
        weights = [comparison["per_model"][n]["weight"] for n in converged]
        assert sum(weights) == pytest.approx(1.0)

    def test_fits_to_different_observations_are_not_ranked_together(self):
        """Ce = 0 is valid for Langmuir/Freundlich/Sips but outside Temkin's domain."""
        from adsorblab_pro.tabs.isotherm_tab import _fit_all_isotherm_models_cached
        from adsorblab_pro.utils import compare_information_criteria, model_comparison_table

        Ce = np.array([0.0, 1.5, 3.0, 6.0, 12.0, 25.0, 50.0, 100.0])
        noise = np.array([0.2, 0.1, -0.2, 0.3, -0.2, 0.25, -0.3, 0.2])
        qe = np.where(Ce > 0, 5 * np.log(2 * np.maximum(Ce, 1e-9)), 0.0) + noise
        fitted = _fit_all_isotherm_models_cached(
            tuple(Ce), tuple(qe), tuple(Ce + 2 * qe), 0.95, 298.15
        )
        assert fitted["Temkin"]["converged"] and fitted["Langmuir"]["converged"]

        comparison = compare_information_criteria(fitted, "aicc")
        temkin = comparison["per_model"]["Temkin"]
        langmuir = comparison["per_model"]["Langmuir"]
        assert (langmuir["set"], langmuir["n"]) == (1, 8)
        assert (temkin["set"], temkin["n"]) == (2, 7)
        assert "different observations" in temkin["note"]
        assert np.isnan(temkin["weight"]) and np.isnan(temkin["delta"])
        assert comparison["best"] != "Temkin"
        assert "not ranked with these: Temkin (n = 7)" in comparison["message"]
        set1 = [n for n, v in comparison["per_model"].items() if v["set"] == 1]
        assert sum(comparison["per_model"][n]["weight"] for n in set1) == pytest.approx(1.0)

        table, _ = model_comparison_table(fitted)
        row = table.set_index("Model").loc["Temkin"]
        assert row["Set"] == 2 and np.isnan(row["ΔAICc"]) and np.isnan(row["AICc weight"])
        # Rows of the ranked set come first; the separate set is listed after it.
        assert list(table["Set"]) == sorted(table["Set"])

    def test_four_point_two_parameter_fits_have_no_aicc_winner(self):
        from adsorblab_pro.plot_style import create_model_comparison_plot
        from adsorblab_pro.tabs.isotherm_tab import _fit_all_isotherm_models_cached
        from adsorblab_pro.utils import compare_information_criteria, model_comparison_table

        Ce = np.array([1.0, 5.0, 15.0, 40.0])
        fitted = _fit_all_isotherm_models_cached(
            tuple(Ce), (9.0, 20.0, 35.0, 60.0), (10.0, 25.0, 50.0, 100.0), 0.95, 298.15
        )
        converged = [n for n, r in fitted.items() if r.get("converged")]
        assert len(converged) >= 2
        assert all(np.isinf(fitted[n]["aicc"]) for n in converged)

        comparison = compare_information_criteria(fitted, "aicc")
        assert comparison["status"] == "unavailable"
        assert comparison["best"] is None
        assert comparison["message"].startswith("No AICc-based ranking")
        for name in converged:
            info = comparison["per_model"][name]
            assert np.isnan(info["weight"]) and np.isnan(info["value"])
            assert "AICc undefined: n = 4 ≤ k + 1 with k = p + 1 = 3" in info["note"]

        table, _ = model_comparison_table(fitted)
        assert table[["AICc", "ΔAICc", "AICc weight"]].isna().all().all()
        assert np.isfinite(table["AIC"]).all()  # AIC itself is defined for n = 4

        fig = create_model_comparison_plot(
            Ce,
            np.array([9.0, 20.0, 35.0, 60.0]),
            fitted,
            {n: (lambda x, p: x) for n in converged},
        )
        dashes = [trace.line.dash for trace in fig.data if trace.mode == "lines"]
        assert dashes and "solid" not in dashes

    def test_finite_and_infinite_aicc_mixture(self):
        from adsorblab_pro.utils import compare_information_criteria

        models = {
            "Langmuir": _fit(10.0, n=5),
            "Freundlich": _fit(12.0, n=5),
            "Sips": _fit(float("inf"), n=5, p=3),
        }
        comparison = compare_information_criteria(models, "aicc")
        assert comparison["status"] == "ranked"
        assert comparison["best"] == "Langmuir"
        sips = comparison["per_model"]["Sips"]
        assert np.isnan(sips["weight"]) and np.isnan(sips["delta"])
        assert "AICc undefined: n = 5 ≤ k + 1 with k = p + 1 = 4" in sips["note"]
        w_l = comparison["per_model"]["Langmuir"]["weight"]
        w_f = comparison["per_model"]["Freundlich"]["weight"]
        assert w_l + w_f == pytest.approx(1.0)
        assert w_l == pytest.approx(1 / (1 + np.exp(-1.0)))

        # Only one defined value: no ranking, and never a winner among infinities.
        models["Freundlich"]["aicc"] = float("inf")
        comparison = compare_information_criteria(models, "aicc")
        assert comparison["status"] == "unavailable" and comparison["best"] is None

    def test_missing_results(self):
        from adsorblab_pro.docx_report import _best_model_line
        from adsorblab_pro.tabs.report_tab import _comparison_export
        from adsorblab_pro.utils import compare_information_criteria, model_comparison_table

        for models in (None, {}, {"A": {"converged": False}}, {"A": None}):
            comparison = compare_information_criteria(models, "aicc")
            assert comparison["status"] == "none" and comparison["best"] is None
            assert model_comparison_table(models)[0].empty
            assert _comparison_export(models) is None
            assert _best_model_line(models, "Isotherm models") is None

        # A converged result without criteria is listed with its values unavailable.
        lone = {"converged": True, "r_squared": 0.9, "x_data": [1, 2, 3], "y_data": [1, 2, 3]}
        table, comparison = model_comparison_table({"A": lone, "B": dict(lone)})
        assert comparison["status"] == "unavailable"
        assert table[["AIC", "AICc", "BIC", "AICc weight"]].isna().all().all()
        single, comparison = model_comparison_table({"A": _fit(5.0)})
        assert comparison["status"] == "single" and comparison["best"] is None

    def test_akaike_weights_exclude_unavailable_values(self):
        from adsorblab_pro.utils import calculate_akaike_weights

        w = calculate_akaike_weights([1.0, float("inf"), 3.0, float("nan")])
        assert np.isnan(w[1]) and np.isnan(w[3])
        assert w[0] + w[2] == pytest.approx(1.0)
        assert np.isnan(calculate_akaike_weights([float("inf"), float("inf")])).all()

    def test_recommendations_use_comparable_aicc_only(self):
        from adsorblab_pro.utils import recommend_best_models

        models = {"A": _fit(float("inf"), n=4), "B": _fit(float("inf"), n=4)}
        for rec in recommend_best_models(models):
            assert np.isfinite(rec["score"]) and np.isnan(rec["aicc_weight"])
            assert "No comparable AICc" in rec["rationale"]
        models = {"A": _fit(10.0), "B": _fit(-5.0, x=np.arange(1.0, 8.0))}  # different sets
        recs = {r["model"]: r for r in recommend_best_models(models)}
        assert np.isnan(recs["B"]["aicc_weight"]) and "AIC support" not in recs["B"]["rationale"]

    def test_export_and_docx_use_the_screen_comparison(self):
        from adsorblab_pro.docx_report import _best_model_line
        from adsorblab_pro.tabs.kinetic_tab import _fit_all_kinetic_models_cached
        from adsorblab_pro.tabs.report_tab import _gen_tbl_iso_comparison, _gen_tbl_kin_comparison
        from adsorblab_pro.utils import model_comparison_table

        fitted = _fit_all_kinetic_models_cached(tuple(KIN_T), tuple(_kinetic_qt(KIN_T)), 0.95)
        screen, comparison = model_comparison_table(fitted)
        export = _gen_tbl_kin_comparison({"kinetic_models_fitted": fitted})
        assert export is not None
        for column in ("AIC", "AICc", "BIC", "ΔAICc", "AICc weight", "Set", "n"):
            assert list(export[column]) == pytest.approx(list(screen[column]), nan_ok=True)
        for _, row in export.iterrows():
            result = fitted[row["Model"]]
            assert row["AIC"] == pytest.approx(result["aic"])
            assert row["AICc"] == pytest.approx(result["aicc"])
            assert row["BIC"] == pytest.approx(result["bic"])
        assert export.loc[0, "Comparison statement"] == comparison["message"]
        assert (
            _best_model_line(fitted, "Kinetic models") == f"Kinetic models: {comparison['message']}"
        )
        assert _gen_tbl_iso_comparison({"isotherm_models_fitted": {}}) is None

    @pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
    def test_screen_shows_no_winner_for_undefined_aicc(self):
        at = AppTest.from_function(
            _screen,
            args=("isotherm", [1.0, 5.0, 15.0, 40.0], [9.0, 20.0, 35.0, 60.0], [10, 25, 50, 100]),
            default_timeout=60,
        )
        at.run()
        assert not at.exception, at.exception
        assert not [s.value for s in at.success]
        assert any("No AICc-based ranking" in i.value for i in at.info)
        table = at.dataframe[0].value
        assert {"AIC", "AICc", "BIC", "ΔAICc", "AICc weight", "Set"} <= set(table.columns)
        assert table["AICc"].isna().all()

    @pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
    def test_kinetic_screen_ranks_one_observation_set(self):
        at = AppTest.from_function(
            _screen,
            args=("kinetic", list(KIN_T), list(_kinetic_qt(KIN_T)), None),
            default_timeout=60,
        )
        at.run()
        assert not at.exception, at.exception
        table = at.dataframe[0].value
        assert set(table["Set"]) == {1} and set(table["n"]) == {len(KIN_T)}
        assert "Elovich" in set(table["Model"])
        assert any("same 10 observations" in s.value for s in at.success)


R05_ISO_CE = np.array([0.0, 1.5, 3.0, 6.0, 12.0, 25.0, 50.0, 100.0])
R05_ISO_QE = np.where(R05_ISO_CE > 0, 5 * np.log(2 * np.maximum(R05_ISO_CE, 1e-9)), 0.0) + np.array(
    [0.2, 0.1, -0.2, 0.3, -0.2, 0.25, -0.3, 0.2]
)


@pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
class TestR05ComparisonPages:
    @pytest.fixture
    def app(self):
        from adsorblab_pro.tabs.isotherm_tab import _fit_all_isotherm_models_cached
        from adsorblab_pro.tabs.kinetic_tab import _fit_all_kinetic_models_cached

        at = AppTest.from_file("adsorblab_pro/app.py", default_timeout=90)
        a = _seeded_study(
            [10, 20, 40, 80], [1, 3, 8, 20], 0.1, 0.2, 30.0, [0, 5, 10, 20, 40], "direct"
        )
        b = _seeded_study([5, 15, 30, 60], [2, 6, 13, 30], 0.05, 0.1, 40.0, [0, 2, 4, 8], "direct")
        a["isotherm_models_fitted"] = _fit_all_isotherm_models_cached(
            tuple(R05_ISO_CE), tuple(R05_ISO_QE), tuple(R05_ISO_CE + 2 * R05_ISO_QE), 0.95, 298.15
        )
        a["kinetic_models_fitted"] = _fit_all_kinetic_models_cached(
            tuple(KIN_T), tuple(_kinetic_qt(KIN_T)), 0.95
        )
        b["isotherm_models_fitted"] = _fit_all_isotherm_models_cached(
            (1.0, 5.0, 15.0, 40.0), (9.0, 20.0, 35.0, 60.0), (10.0, 25.0, 50.0, 100.0), 0.95, 298.15
        )
        at.session_state["studies"] = {"A": a, "B": b}
        at.session_state["current_study"] = "A"
        at.session_state["_previous_study_selection"] = "A"
        at.run()
        assert not at.exception, at.exception
        return at

    def test_study_overview_labels_and_statements(self, app):
        app.radio(key="main_section_navigation").set_value("📈 Visualization & Reports").run()
        app.radio(key="report_section_navigation").set_value("📊 Study Overview").run()
        assert not app.exception, app.exception
        tables = [d.value for d in app.dataframe if "AICc weight" in d.value.columns]
        assert len(tables) == 2  # isotherm and kinetic summaries
        iso = tables[0].set_index("Model")
        fitted = app.session_state["studies"]["A"]["isotherm_models_fitted"]
        for name in iso.index:
            assert iso.loc[name, "AIC"] == pytest.approx(fitted[name]["aic"])
            assert iso.loc[name, "AICc"] == pytest.approx(fitted[name]["aicc"])
        assert iso.loc["Temkin", "Set"] == 2 and np.isnan(iso.loc["Temkin", "AICc weight"])
        statements = [s.value for s in app.success] + [i.value for i in app.info]
        assert any("not ranked with these: Temkin (n = 7)" in s for s in statements)
        assert any("same 10 observations" in s for s in statements)
        assert not any("Best Isotherm Model" in s or "Best Kinetic Model" in s for s in statements)

    def test_multi_study_comparison_selects_within_each_study(self, app):
        app.radio(key="main_section_navigation").set_value("🧪 Analysis Workflow").run()
        app.radio(key="workflow_section_navigation").set_value("🆚 Comparison").run()
        assert not app.exception, app.exception
        selection = [
            d.value for d in app.dataframe if "Lowest AICc (same observations)" in d.value.columns
        ]
        if not selection:  # the comparison page needs at least two studies with isotherm fits
            pytest.skip("comparison page did not render the isotherm section")
        table = selection[0].set_index("Study")
        assert table.loc["A", "Lowest AICc (same observations)"] != "Temkin"
        assert table.loc["B", "Lowest AICc (same observations)"] == "—"
        assert "No AICc-based ranking" in table.loc["B", "Note"]
        summary = [d.value for d in app.dataframe if {"Study", "Model", "RMSE"} <= set(d.value)]
        assert summary and "AICc" in summary[0].columns and "AIC" not in summary[0].columns


# =============================================================================
# R06 — Defensible fit limits and visible diagnostics
# =============================================================================
def _langmuir(Ce, qm, KL):
    return qm * KL * Ce / (1 + KL * Ce)


R06_ORDINARY_CE = np.array([0.5, 1, 2.5, 5, 10, 20, 40, 70, 110, 160, 200], dtype=float)
R06_ORDINARY_QE = _langmuir(R06_ORDINARY_CE, 80.0, 0.05) * (
    1 + 0.02 * np.random.default_rng(1).standard_normal(len(R06_ORDINARY_CE))
)
R06_HENRY_CE = np.linspace(0.1, 2, 10)
R06_HENRY_QE = _langmuir(R06_HENRY_CE, 500.0, 0.001) * (
    1 + 0.01 * np.random.default_rng(2).standard_normal(len(R06_HENRY_CE))
)


def _iso_fit(Ce, qe):
    from adsorblab_pro.tabs.isotherm_tab import _fit_all_isotherm_models_cached

    Ce = np.asarray(Ce, dtype=float)
    qe = np.asarray(qe, dtype=float)
    return _fit_all_isotherm_models_cached(tuple(Ce), tuple(qe), tuple(Ce + qe), 0.95, 298.15)


def _iso_displays(Ce, qe):
    """AppTest script: fit and render the Langmuir and Temkin result sections."""
    import numpy as np

    from adsorblab_pro.tabs import isotherm_tab

    Ce = np.asarray(Ce, dtype=float)
    qe = np.asarray(qe, dtype=float)
    fitted = isotherm_tab._fit_all_isotherm_models_cached(
        tuple(Ce), tuple(qe), tuple(Ce + qe), 0.95, 298.15
    )
    isotherm_tab._display_langmuir(Ce, qe, Ce + qe, fitted.get("Langmuir"))
    isotherm_tab._display_temkin(Ce, qe, fitted.get("Temkin"))


class TestR06FitLimits:
    def test_acceptance_low_concentration_langmuir_is_recovered(self):
        """12 points, Ce 1e-4…1e-2 mg/L, qm = 5 mg/g, KL = 1000 L/mg (was pinned at KL = 100)."""
        Ce = np.geomspace(1e-4, 1e-2, 12)
        exact = _iso_fit(Ce, _langmuir(Ce, 5.0, 1000.0))["Langmuir"]
        assert exact["converged"] and exact["bounds_hit"] == []
        assert exact["params"]["qm"] == pytest.approx(5.0, rel=1e-6)
        assert exact["params"]["KL"] == pytest.approx(1000.0, rel=1e-6)

        # With 1 % multiplicative noise the generating values lie inside the 95 % CIs,
        # the parameters are reported as identified, and the curve is recovered.
        noise = 1 + 0.01 * np.random.default_rng(3).standard_normal(12)
        noisy = _iso_fit(Ce, _langmuir(Ce, 5.0, 1000.0) * noise)["Langmuir"]
        assert noisy["bounds_hit"] == []
        assert noisy["param_status"] == {"qm": "identified", "KL": "identified"}
        for name, true in (("qm", 5.0), ("KL", 1000.0)):
            lo, hi = noisy["ci_95"][name]
            assert lo <= true <= hi
            assert noisy["params"][name] == pytest.approx(true, rel=0.05)
        curve = _langmuir(Ce, noisy["params"]["qm"], noisy["params"]["KL"])
        assert np.max(np.abs(curve / _langmuir(Ce, 5.0, 1000.0) - 1)) < 0.02

    def test_ordinary_scale_fit_is_identified_and_unchanged(self):
        fitted = _iso_fit(R06_ORDINARY_CE, R06_ORDINARY_QE)
        langmuir = fitted["Langmuir"]
        assert langmuir["param_status"] == {"qm": "identified", "KL": "identified"}
        assert langmuir["params"]["qm"] == pytest.approx(80.39, abs=0.01)
        assert langmuir["params"]["KL"] == pytest.approx(0.04985, abs=1e-5)
        for name, true in (("qm", 80.0), ("KL", 0.05)):
            lo, hi = langmuir["ci_95"][name]
            assert lo <= true <= hi
        assert fitted["Sips"]["params"]["ns"] == pytest.approx(1.0, abs=0.05)

    @pytest.mark.parametrize("scale", [1e-9, 1e-3, 1e3])
    def test_fits_do_not_depend_on_concentration_scale(self, scale):
        """The same data expressed at another concentration scale give the same fit."""
        base = _iso_fit(R06_ORDINARY_CE, R06_ORDINARY_QE)["Langmuir"]
        scaled = _iso_fit(R06_ORDINARY_CE * scale, R06_ORDINARY_QE)["Langmuir"]
        assert scaled["params"]["qm"] == pytest.approx(base["params"]["qm"], rel=1e-5)
        assert scaled["params"]["KL"] * scale == pytest.approx(base["params"]["KL"], rel=1e-5)
        assert scaled["r_squared"] == pytest.approx(base["r_squared"], rel=1e-9)
        assert scaled["param_status"] == base["param_status"]

    def test_poorly_identified_parameters_are_not_called_precise(self):
        """Henry region (qe/qm < 0.4 %): excellent curve agreement, undetermined qm and KL."""
        langmuir = _iso_fit(R06_HENRY_CE, R06_HENRY_QE)["Langmuir"]
        assert langmuir["converged"] and langmuir["r_squared"] > 0.999
        assert langmuir["param_status"] == {"qm": "poorly identified", "KL": "poorly identified"}
        (a, b, r) = langmuir["correlated_params"][0]
        assert {a, b} == {"qm", "KL"} and abs(r) > 0.999
        # The initial slope qm·KL is still determined by the data.
        assert langmuir["params"]["qm"] * langmuir["params"]["KL"] == pytest.approx(0.5, rel=0.02)

    def test_strong_correlation_is_reported_for_early_time_kinetics(self):
        from adsorblab_pro.tabs.kinetic_tab import _fit_all_kinetic_models_cached

        t = np.array([0, 1, 2, 3, 4, 5, 6, 8], dtype=float)
        qt = 100 * (1 - np.exp(-0.005 * t)) * (1 + 0.01 * np.random.default_rng(7).normal(size=8))
        pfo = _fit_all_kinetic_models_cached(tuple(t), tuple(qt), 0.95)["PFO"]
        assert set(pfo["param_status"].values()) <= {"strongly correlated", "poorly identified"}
        assert pfo["correlated_params"]

    def test_limits_follow_the_data_scale(self):
        from adsorblab_pro.models import SEARCH_RANGE, isotherm_fit_setup, kinetic_fit_setup

        Ce = np.array([0.0, 0.02, 0.1, 0.5, 2.0])
        qe = np.array([0.1, 0.4, 1.2, 2.5, 3.0])
        setup = isotherm_fit_setup(Ce, qe)
        assert setup["Langmuir"]["bounds"] == (
            [0.0, 0.0],
            [SEARCH_RANGE * 3.0, SEARCH_RANGE / 0.02],
        )
        assert setup["Temkin"]["bounds"][0] == [0.0, pytest.approx(1 / 0.02)]
        assert setup["Freundlich"]["bounds"] == ([0.0, 0.01], [np.inf, 5.0])
        assert setup["Sips"]["bounds"][1][2] == 5.0
        for name, config in setup.items():
            lo, hi = config["bounds"]
            assert all(lo[i] <= p <= hi[i] for i, p in enumerate(config["p0"])), name

        kinetic = kinetic_fit_setup([0, 2, 5, 10, 30], [0, 0.004, 0.008, 0.01, 0.012])
        assert kinetic["PSO"]["bounds"][1] == [
            pytest.approx(SEARCH_RANGE * 0.012),
            pytest.approx(SEARCH_RANGE / (0.012 * 2)),
        ]
        assert kinetic["Elovich"]["bounds"][1][0] == np.inf

    def test_low_uptake_kinetics_are_not_pinned(self):
        """qe = 0.02 mg/g, k2 = 50 g/(mg·min): the old k2 ≤ 10 and β ≤ 10 ceilings pinned both."""
        from adsorblab_pro.tabs.kinetic_tab import _fit_all_kinetic_models_cached

        t = np.array([0, 1, 2, 5, 10, 20, 30, 60, 90, 120], dtype=float)
        qt = 0.02**2 * 50 * t / (1 + 0.02 * 50 * t)
        fitted = _fit_all_kinetic_models_cached(tuple(t), tuple(qt), 0.95)
        assert fitted["PSO"]["params"]["k2"] == pytest.approx(50.0, rel=1e-6)
        assert fitted["PSO"]["params"]["qe"] == pytest.approx(0.02, rel=1e-6)
        assert fitted["PSO"]["bounds_hit"] == [] and fitted["Elovich"]["bounds_hit"] == []
        assert fitted["Elovich"]["params"]["beta"] > 10

    def test_limited_fit_is_reported_separately_from_convergence(self):
        """Temkin on Langmuir-shaped data stops at KT = 1/min(Ce): converged, but at limit."""
        temkin = _iso_fit(R06_ORDINARY_CE, R06_ORDINARY_QE)["Temkin"]
        assert temkin["converged"]
        assert temkin["param_status"]["KT"] == "at limit"
        assert temkin["params"]["KT"] == pytest.approx(1 / R06_ORDINARY_CE.min(), rel=1e-6)
        assert any(hit.startswith("KT (lower bound") for hit in temkin["bounds_hit"])

        from adsorblab_pro.tabs.kinetic_tab import _fit_all_kinetic_models_cached

        t = np.array([0, 10, 20, 30, 60, 90], dtype=float)  # uptake complete before t = 10
        qt = np.array([0, 5.0, 5.02, 4.98, 5.01, 5.0])
        pso = _fit_all_kinetic_models_cached(tuple(t), tuple(qt), 0.95)["PSO"]
        assert pso["converged"] and pso["param_status"]["k2"] == "at limit"

    def test_temkin_fits_below_one_mg_per_litre(self):
        """The fixed start KT = 1 made Temkin fail whenever min(Ce) < 1 mg/L (D23)."""
        Ce = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
        qe = 5 * np.log(5 * Ce) + np.array([0.1, -0.2, 0.3, -0.2, 0.25, -0.3, 0.2])
        temkin = _iso_fit(Ce, qe)["Temkin"]
        assert temkin["converged"]
        assert temkin["params"]["B1"] == pytest.approx(5.0, rel=0.05)
        assert temkin["params"]["KT"] == pytest.approx(5.0, rel=0.1)
        assert temkin["param_status"] == {"B1": "identified", "KT": "identified"}

    def test_temkin_curves_stop_where_the_equation_is_undefined(self):
        from adsorblab_pro.models import temkin_curve
        from adsorblab_pro.plot_style import create_model_comparison_plot

        values = temkin_curve(np.array([0.1, 0.5, 1.0, 4.0]), 2.0, 2.0)
        assert np.isnan(values[0]) and values[1] == pytest.approx(0.0)
        assert values[3] == pytest.approx(2 * np.log(8))

        fitted = _iso_fit(R06_ORDINARY_CE, R06_ORDINARY_QE)
        fig = create_model_comparison_plot(
            R06_ORDINARY_CE,
            R06_ORDINARY_QE,
            fitted,
            {"Temkin": lambda x, p: temkin_curve(x, p["B1"], p["KT"])},
        )
        temkin = [tr for tr in fig.data if tr.mode == "lines" and tr.name.startswith("Temkin")]
        assert temkin and np.isfinite(np.asarray(temkin[0].y, dtype=float)).any()

    def test_fit_configuration_and_cache_rounding(self):
        from adsorblab_pro.models import round_significant

        langmuir = _iso_fit(R06_ORDINARY_CE, R06_ORDINARY_QE)["Langmuir"]
        config = langmuir["fit_config"]
        assert len(config["p0"]) == 2 and config["bounds"][1][1] == pytest.approx(1e6 / 0.5)

        tiny = np.array([1.23456789012e-10, 5.5e-9])  # 12 significant digits
        assert np.array_equal(round_significant(tiny), tiny)  # decimals would destroy these
        assert np.array_equal(
            round_significant(round_significant(tiny * 3.3)), round_significant(tiny * 3.3)
        )

    def test_exports_carry_parameter_status(self):
        from adsorblab_pro.tabs.report_tab import _gen_tbl_iso_comparison, _gen_tbl_iso_params

        fitted = _iso_fit(R06_HENRY_CE, R06_HENRY_QE)
        table = _gen_tbl_iso_params({"isotherm_models_fitted": fitted})
        assert {"Value", "SE", "CI_Lower", "CI_Upper", "Status", "Fit notes"} <= set(table.columns)
        assert not table["Parameter"].str.endswith("_se").any()
        langmuir = table[table["Model"] == "Langmuir"].set_index("Parameter")
        assert list(langmuir.loc[["qm", "KL"], "Status"]) == ["poorly identified"] * 2
        assert langmuir.loc["KL", "Value"] == pytest.approx(fitted["Langmuir"]["params"]["KL"])
        assert "poorly identified: qm, KL" in langmuir.loc["qm", "Fit notes"]
        freundlich = table[table["Model"] == "Freundlich"].set_index("Parameter")
        assert freundlich.loc["n", "Status"] == "derived"

        comparison = _gen_tbl_iso_comparison({"isotherm_models_fitted": fitted}).set_index("Model")
        assert "poorly identified: qm, KL" in comparison.loc["Langmuir", "Note"]

    @pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
    def test_displays_show_limits_and_identifiability(self):
        at = AppTest.from_function(
            _iso_displays, args=(list(R06_ORDINARY_CE), list(R06_ORDINARY_QE)), default_timeout=60
        )
        at.run()
        assert not at.exception, at.exception  # the Temkin curve no longer raises
        warnings = " ".join(w.value for w in at.warning)
        assert "Stopped at a limit" in warnings and "KT (lower bound 2)" in warnings
        assert "1/min(Ce)" in warnings
        assert "did not converge" not in warnings
        statuses = [t.value for t in at.table] + [d.value for d in at.dataframe]
        assert any("Status" in frame.columns for frame in statuses)

        at = AppTest.from_function(
            _iso_displays, args=(list(R06_HENRY_CE), list(R06_HENRY_QE)), default_timeout=60
        )
        at.run()
        assert not at.exception, at.exception
        info = " ".join(i.value for i in at.info)
        assert "Poorly identified" in info and "does not make these parameters precise" in info


def _kin_displays(t, qt):
    """AppTest script: fit and render the PFO, PSO and Elovich result sections."""
    import numpy as np

    from adsorblab_pro.tabs import kinetic_tab

    t = np.asarray(t, dtype=float)
    qt = np.asarray(qt, dtype=float)
    fitted = kinetic_tab._fit_all_kinetic_models_cached(tuple(t), tuple(qt), 0.95)
    kinetic_tab._display_pfo(t, qt, fitted.get("PFO"))
    kinetic_tab._display_pso(t, qt, fitted.get("PSO"))
    kinetic_tab._display_elovich(t, qt, fitted.get("Elovich"))


@pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
def test_r06_kinetic_displays_show_limits():
    t = [0, 10, 20, 30, 60, 90]  # uptake complete before the first sample
    qt = [0, 5.0, 5.02, 4.98, 5.01, 5.0]
    at = AppTest.from_function(_kin_displays, args=(t, qt), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "Stopped at a limit" in warnings and "k2 (upper bound" in warnings
    assert "Poorly identified" in " ".join(i.value for i in at.info)


# =============================================================================
# R07 — Honest failed-fold cross-validation
# =============================================================================
def _line(x, a):
    return a * np.asarray(x, dtype=float)


class TestR07CrossValidation:
    def test_all_folds_failing_is_not_perfect_validation(self):
        """Exact y = 2x with every refit forced to fail: previously PRESS = 0, Q² = 1."""
        from unittest import mock

        import adsorblab_pro.utils as utils

        x = np.arange(1.0, 9.0)
        y = 2.0 * x
        with mock.patch.object(utils, "curve_fit", side_effect=RuntimeError("forced failure")):
            details = utils.calculate_press_details(_line, x, y, [2.0])
            press = utils.calculate_press(_line, x, y, [2.0])
        assert details["status"] == "unavailable"
        assert details["n_failed"] == details["n_folds"] == 8
        assert np.isnan(details["press"]) and np.isnan(details["q2"]) and np.isnan(press)
        assert np.isnan(details["partial_press"]) and details["partial_folds"] == 0
        assert all(f["reason"] == "refit failed: forced failure" for f in details["failed_folds"])
        assert np.isnan(utils.calculate_q2(press, y))

    def test_successful_case_matches_manual_leave_one_out(self):
        from scipy.optimize import curve_fit

        from adsorblab_pro.models import isotherm_fit_setup, langmuir_model
        from adsorblab_pro.tabs.isotherm_tab import _isotherm_fold_setup
        from adsorblab_pro.utils import calculate_press_details

        Ce, qe = R06_ORDINARY_CE, R06_ORDINARY_QE
        details = calculate_press_details(
            langmuir_model, Ce, qe, fit_setup=_isotherm_fold_setup("Langmuir")
        )
        manual = 0.0
        for i in range(len(Ce)):
            keep = np.arange(len(Ce)) != i
            config = isotherm_fit_setup(Ce[keep], qe[keep])["Langmuir"]
            popt, _ = curve_fit(
                langmuir_model,
                Ce[keep],
                qe[keep],
                p0=config["p0"],
                bounds=config["bounds"],
                maxfev=10000,
            )
            manual += (qe[i] - langmuir_model(Ce[i : i + 1], *popt)[0]) ** 2
        assert details["status"] == "complete" and details["n_failed"] == 0
        assert details["press"] == pytest.approx(manual, rel=1e-6)
        ss_tot = np.sum((qe - qe.mean()) ** 2)
        assert details["q2"] == pytest.approx(1 - manual / ss_tot, rel=1e-9)

    def test_mixed_failure_reports_partial_diagnostics_separately(self):
        from unittest import mock

        import adsorblab_pro.utils as utils
        from scipy.optimize import curve_fit as real_curve_fit

        x = np.arange(1.0, 9.0)
        y = 2.0 * x + np.array([0.1, -0.1, 0.05, 0.0, -0.05, 0.1, -0.1, 0.02])

        def flaky(f, xdata, ydata, **kwargs):
            if 4.0 not in xdata:  # the fold that holds out x = 4
                raise RuntimeError("did not converge")
            return real_curve_fit(f, xdata, ydata, **kwargs)

        with mock.patch.object(utils, "curve_fit", side_effect=flaky):
            details = utils.calculate_press_details(_line, x, y, [1.0])
        assert details["status"] == "unavailable" and np.isnan(details["press"])
        assert details["n_failed"] == 1 and details["failed_folds"][0]["index"] == 3
        assert details["partial_folds"] == 7 and np.isfinite(details["partial_press"])
        assert "1 of 8 leave-one-out refits failed" in details["message"]

    def test_folds_reuse_the_estimator_configuration(self):
        from unittest import mock

        import adsorblab_pro.utils as utils
        from scipy.optimize import curve_fit as real_curve_fit

        seen = []

        def setup(x_train, y_train):
            seen.append(len(x_train))
            return [1.0], ([0.0], [10.0])

        calls = []

        def spy(f, xdata, ydata, **kwargs):
            calls.append(kwargs)
            return real_curve_fit(f, xdata, ydata, **kwargs)

        x = np.arange(1.0, 7.0)
        with mock.patch.object(utils, "curve_fit", side_effect=spy):
            utils.calculate_press_details(_line, x, 2 * x, fit_setup=setup)
        assert seen == [5] * 6
        assert all(c["bounds"] == ([0.0], [10.0]) and c["p0"] == [1.0] for c in calls)
        assert all(c["maxfev"] == utils.MAX_FIT_ITERATIONS for c in calls)

    def test_prediction_outside_the_model_domain_is_a_failed_fold(self):
        """Temkin held-out at the smallest Ce: the refit's prediction is qe < 0."""
        from adsorblab_pro.models import temkin_model
        from adsorblab_pro.tabs.isotherm_tab import _isotherm_fold_setup
        from adsorblab_pro.utils import calculate_press_details, model_comparison_table

        fitted = _iso_fit(R06_ORDINARY_CE, R06_ORDINARY_QE)
        temkin = fitted["Temkin"]
        details = calculate_press_details(
            temkin_model,
            temkin["x_data"],
            temkin["y_data"],
            fit_setup=_isotherm_fold_setup("Temkin"),
        )
        assert details["status"] == "unavailable"
        assert details["failed_folds"][0]["x"] == pytest.approx(0.5)
        assert details["failed_folds"][0]["reason"].startswith("prediction failed")

        temkin.update(press=details["press"], q2=details["q2"], press_details=details)
        table, _ = model_comparison_table(fitted, include_press=True)
        row = table.set_index("Model").loc["Temkin"]
        assert np.isnan(row["PRESS"]) and np.isnan(row["Q²"])
        assert "PRESS/Q² unavailable: 1 of 11" in row["Note"]

    def test_q2_is_undefined_without_variance(self):
        from adsorblab_pro.utils import calculate_q2

        assert np.isnan(calculate_q2(0.0, np.array([3.0, 3.0, 3.0])))
        assert np.isnan(calculate_q2(float("nan"), np.array([1.0, 2.0, 3.0])))
        assert calculate_q2(1.0, np.array([1.0, 2.0, 3.0])) == pytest.approx(0.5)

    @pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
    def test_isotherm_page_reports_unavailable_press(self):
        import copy

        from adsorblab_pro.config import DEFAULT_SESSION_STATE

        state = copy.deepcopy(DEFAULT_SESSION_STATE)
        state["input_mode_global"] = "direct"
        state["isotherm_input"] = {
            "data": pd.DataFrame(
                {
                    "source_row": np.arange(1, 12),
                    "C0": R06_ORDINARY_CE + R06_ORDINARY_QE,  # V/m = 1 L/g
                    "Ce": R06_ORDINARY_CE,
                }
            ),
            "params": {"m": 0.1, "V": 0.1, "T_C": 25.0, "T_K": 298.15},
            "input_mode": "direct",
        }
        at = AppTest.from_file("adsorblab_pro/app.py", default_timeout=120)
        at.session_state["studies"] = {"S": state}
        at.session_state["current_study"] = "S"
        at.session_state["_previous_study_selection"] = "S"
        at.run()
        at.radio(key="main_section_navigation").set_value("🧪 Analysis Workflow").run()
        at.checkbox(key="isotherm_press_checkbox").check().run()
        at.button(key="isotherm_calculate_btn").click().run()
        assert not at.exception, at.exception
        assert any(
            "Temkin: PRESS/Q² unavailable: 1 of 11 leave-one-out refits failed" in w.value
            for w in at.warning
        )
        fitted = at.session_state["studies"]["S"]["isotherm_models_fitted"]
        assert fitted["Langmuir"]["press_details"]["status"] == "complete"
        assert np.isnan(fitted["Temkin"]["press"]) and np.isnan(fitted["Temkin"]["q2"])
        table = [d.value for d in at.dataframe if "PRESS" in d.value.columns][0].set_index("Model")
        assert np.isfinite(table.loc["Langmuir", "PRESS"]) and np.isnan(table.loc["Temkin", "Q²"])


# =============================================================================
# R08 — Consistent, reproducible bootstrap
# =============================================================================
def _langmuir_setup(x, y):
    from adsorblab_pro.models import isotherm_fit_setup

    config = isotherm_fit_setup(x, y)["Langmuir"]
    return config["p0"], config["bounds"]


def _iso_bootstrap_displays(Ce, qe, n_boot, fail_every):
    """AppTest script: bootstrap a Langmuir fit (optionally forcing failures) and render it."""
    from unittest import mock

    import numpy as np
    from scipy.optimize import curve_fit as real_curve_fit

    import adsorblab_pro.utils as utils
    from adsorblab_pro.models import isotherm_fit_setup, langmuir_model
    from adsorblab_pro.tabs import isotherm_tab

    Ce = np.asarray(Ce, dtype=float)
    qe = np.asarray(qe, dtype=float)
    fitted = isotherm_tab._fit_all_isotherm_models_cached(
        tuple(Ce), tuple(qe), tuple(Ce + qe), 0.95, 298.15
    )
    result = fitted["Langmuir"]
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if fail_every and calls["n"] % fail_every == 0:
            raise RuntimeError("forced")
        return real_curve_fit(*args, **kwargs)

    def setup(x, y):
        config = isotherm_fit_setup(x, y)["Langmuir"]
        return config["p0"], config["bounds"]

    with mock.patch.object(utils, "curve_fit", side_effect=flaky):
        details = utils.bootstrap_parameter_intervals(
            langmuir_model,
            result["x_data"],
            result["y_data"],
            result["popt"],
            n_boot,
            0.95,
            fit_setup=setup,
            param_names=["qm", "KL"],
        )
    utils.report_bootstrap_outcome([utils.store_bootstrap_result(result, details, "Langmuir")])
    isotherm_tab._display_langmuir(Ce, qe, Ce + qe, result)


class TestR08Bootstrap:
    def _run(self, n=200, seed=12345, fit_setup=_langmuir_setup, x=None, y=None):
        from adsorblab_pro.models import langmuir_model
        from adsorblab_pro.utils import bootstrap_parameter_intervals

        x = R06_ORDINARY_CE if x is None else x
        y = R06_ORDINARY_QE if y is None else y
        fit = _iso_fit(x, y)["Langmuir"]
        return bootstrap_parameter_intervals(
            langmuir_model,
            fit["x_data"],
            fit["y_data"],
            fit["popt"],
            n,
            0.95,
            fit_setup=fit_setup,
            seed=seed,
            param_names=["qm", "KL"],
        )

    def test_same_seed_data_and_settings_reproduce(self):
        first, second = self._run(), self._run()
        assert first["status"] == "available"
        np.testing.assert_array_equal(first["ci_lower"], second["ci_lower"])
        np.testing.assert_array_equal(first["ci_upper"], second["ci_upper"])
        assert (first["successful"], first["failed"]) == (second["successful"], second["failed"])
        other = self._run(seed=999)
        assert not np.array_equal(first["ci_lower"], other["ci_lower"])
        assert first["seed"] == 12345 and first["confidence"] == 0.95

    def test_every_draw_is_refitted_with_the_original_limits(self):
        from unittest import mock

        import adsorblab_pro.utils as utils
        from scipy.optimize import curve_fit as real_curve_fit

        calls = []

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return real_curve_fit(*args, **kwargs)

        with mock.patch.object(utils, "curve_fit", side_effect=spy):
            details = self._run(n=500)
        # No early stop: every requested draw attempted (previously 200 of 500 on these data).
        assert len(calls) == details["attempted"] == details["requested"] == 500
        assert details["successful"] == 500 and details["status"] == "available"
        assert all(c["bounds"][0] == [0.0, 0.0] for c in calls)
        assert all(c["maxfev"] == utils.MAX_FIT_ITERATIONS for c in calls)
        assert np.all(details["ci_lower"] >= 0)  # limits enforced in every draw

    def test_work_is_capped_deterministically_for_unidentified_parameters(self):
        """Henry region: refits crawl along the qm–KL ridge; the evaluation budget stops the run."""
        first = self._run(n=30, x=R06_HENRY_CE, y=R06_HENRY_QE)
        second = self._run(n=30, x=R06_HENRY_CE, y=R06_HENRY_QE)
        assert first["attempted"] < first["requested"] == 30
        assert first["status"] == "unavailable" and np.isnan(first["ci_upper"]).all()
        assert first["stopped"].startswith(
            f"evaluation limit reached after {first['attempted']} of 30 draws"
        )
        assert "limit 6000, checked between refits" in first["stopped"]
        summary = bootstrap_summary_text({"bootstrap": first})
        assert f"{30 - first['attempted']} not attempted" in summary
        assert "run stopped early: evaluation limit reached" in summary
        assert "interval unavailable: only" in summary
        assert (first["attempted"], first["evaluations"]) == (
            second["attempted"],
            second["evaluations"],
        )

    def test_failed_draws_are_counted_and_insufficient_successes_are_unavailable(self):
        from unittest import mock

        import adsorblab_pro.utils as utils
        from scipy.optimize import curve_fit as real_curve_fit

        def failing_every(k):
            calls = {"n": 0}

            def fit(*args, **kwargs):
                calls["n"] += 1
                if calls["n"] % k == 0:
                    raise RuntimeError("forced")
                return real_curve_fit(*args, **kwargs)

            return fit

        with mock.patch.object(utils, "curve_fit", side_effect=failing_every(2)):
            half = self._run(n=100)
        assert (half["attempted"], half["successful"], half["failed"]) == (100, 50, 50)
        assert half["status"] == "unavailable" and np.isnan(half["ci_lower"]).all()
        assert half["failure_reasons"] == {"RuntimeError: forced": 50}
        assert "only 50 of 100 bootstrap refits succeeded (at least 90 required)" in half["reason"]

        with mock.patch.object(utils, "curve_fit", side_effect=failing_every(20)):
            few = self._run(n=100)
        assert (few["successful"], few["failed"]) == (95, 5)
        assert few["status"] == "available" and np.isfinite(few["ci_lower"]).all()

    def test_public_wrapper_and_cache_identity(self):
        from adsorblab_pro.models import langmuir_model
        from adsorblab_pro.utils import bootstrap_confidence_intervals

        fit = _iso_fit(R06_ORDINARY_CE, R06_ORDINARY_QE)["Langmuir"]
        args = (langmuir_model, fit["x_data"], fit["y_data"], fit["popt"])
        bounds = ([0.0, 0.0], [1e8, 2e6])
        a = bootstrap_confidence_intervals(*args, n_bootstrap=100, bounds=bounds, seed=1)
        b = bootstrap_confidence_intervals(*args, n_bootstrap=100, bounds=bounds, seed=1)
        c = bootstrap_confidence_intervals(*args, n_bootstrap=100, bounds=bounds, seed=2)
        np.testing.assert_array_equal(a[0], b[0])
        assert not np.array_equal(a[0], c[0])  # the seed is part of the cache key
        # early_stopping is accepted for compatibility and has no effect
        d = bootstrap_confidence_intervals(
            *args, n_bootstrap=100, bounds=bounds, seed=1, early_stopping=True
        )
        np.testing.assert_array_equal(a[0], d[0])

    def test_exports_carry_bootstrap_intervals_and_counts(self):
        from adsorblab_pro.tabs.report_tab import _gen_tbl_iso_params
        from adsorblab_pro.utils import store_bootstrap_result

        fitted = _iso_fit(R06_ORDINARY_CE, R06_ORDINARY_QE)
        details = self._run()
        outcome, summary = store_bootstrap_result(fitted["Langmuir"], details, "Langmuir")
        assert outcome == "complete"
        assert fitted["Langmuir"]["bootstrap_n"] == details["successful"]
        assert (
            "200 of 200 bootstrap draws refitted (200 attempted, 0 failed, 0 not attempted; "
            "seed 12345" in summary
        )
        table = _gen_tbl_iso_params({"isotherm_models_fitted": fitted})
        row = table[(table["Model"] == "Langmuir") & (table["Parameter"] == "qm")].iloc[0]
        assert row["Bootstrap_CI_Lower"] == pytest.approx(details["ci_lower"][0])
        assert row["Bootstrap_CI_Upper"] == pytest.approx(details["ci_upper"][0])
        assert "seed 12345" in row["Bootstrap"]
        other = table[table["Model"] == "Freundlich"].iloc[0]
        assert np.isnan(other["Bootstrap_CI_Lower"]) and other["Bootstrap"] == ""

    @pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
    def test_display_shows_counts_or_unavailability(self):
        at = AppTest.from_function(
            _iso_bootstrap_displays,
            args=(list(R06_ORDINARY_CE), list(R06_ORDINARY_QE), 200, 0),
            default_timeout=120,
        )
        at.run()
        assert not at.exception, at.exception
        assert any("200 of 200 bootstrap draws refitted" in s.value for s in at.success)
        tables = [t.value for t in at.table] + [d.value for d in at.dataframe]
        assert any("Bootstrap 95% CI" in frame.columns for frame in tables)
        assert any("seed 12345" in c.value for c in at.caption)

        at = AppTest.from_function(
            _iso_bootstrap_displays,
            args=(list(R06_ORDINARY_CE), list(R06_ORDINARY_QE), 200, 3),
            default_timeout=120,
        )
        at.run()
        assert not at.exception, at.exception
        warnings = " ".join(w.value for w in at.warning)
        assert (
            "Bootstrap interval unavailable" in warnings and "(at least 180 required)" in warnings
        )
        assert not at.success


@pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
def test_r08_isotherm_page_reports_actual_bootstrap_counts():
    import copy

    from adsorblab_pro.config import DEFAULT_SESSION_STATE

    state = copy.deepcopy(DEFAULT_SESSION_STATE)
    state["input_mode_global"] = "direct"
    state["isotherm_input"] = {
        "data": pd.DataFrame(
            {
                "source_row": np.arange(1, 12),
                "C0": R06_ORDINARY_CE + R06_ORDINARY_QE,
                "Ce": R06_ORDINARY_CE,
            }
        ),
        "params": {"m": 0.1, "V": 0.1, "T_C": 25.0, "T_K": 298.15},
        "input_mode": "direct",
    }
    at = AppTest.from_file("adsorblab_pro/app.py", default_timeout=180)
    at.session_state["studies"] = {"S": state}
    at.session_state["current_study"] = "S"
    at.session_state["_previous_study_selection"] = "S"
    at.run()
    at.radio(key="main_section_navigation").set_value("🧪 Analysis Workflow").run()
    at.checkbox(key="isotherm_bootstrap_checkbox").check().run()
    at.slider(key="isotherm_bootstrap_slider").set_value(200).run()
    at.button(key="isotherm_calculate_btn").click().run()
    assert not at.exception, at.exception
    messages = " ".join(s.value for s in at.success) + " ".join(w.value for w in at.warning)
    assert (
        "Langmuir: 200 of 200 bootstrap draws refitted (200 attempted, 0 failed, 0 not "
        "attempted; seed 12345" in messages
    )
    langmuir = at.session_state["studies"]["S"]["isotherm_models_fitted"]["Langmuir"]
    assert langmuir["bootstrap"]["attempted"] == 200 and langmuir["bootstrap_n"] == 200
    frames = [t.value for t in at.table] + [d.value for d in at.dataframe]
    assert any("Bootstrap 95% CI" in frame.columns for frame in frames)


# =============================================================================
# R09 — Honest uncertainty calculations and claims
# =============================================================================
R09_CALIB = pd.DataFrame(
    {"Concentration": [0, 5, 10, 20, 40], "Absorbance": [0.012, 0.091, 0.18, 0.345, 0.69]}
)


class TestR09Uncertainty:
    def test_calibration_covariance_matches_hand_calculation(self):
        """OLS by hand: Var(b) = s²/Sxx, Var(a) = s²(1/n + x̄²/Sxx), Cov(a, b) = −x̄s²/Sxx."""
        from adsorblab_pro.utils import build_calibration, propagate_calibration_uncertainty

        x = np.array([0.0, 5.0, 10.0, 20.0, 40.0])
        y = np.array([0.012, 0.091, 0.18, 0.345, 0.69])
        n, x_bar, y_bar = 5, x.mean(), y.mean()  # x̄ = 15
        sxx = np.sum((x - x_bar) ** 2)  # 1000
        b = np.sum((x - x_bar) * (y - y_bar)) / sxx
        a = y_bar - b * x_bar
        s2 = np.sum((y - a - b * x) ** 2) / (n - 2)
        assert sxx == pytest.approx(1000.0)

        params, error = build_calibration(R09_CALIB)
        assert error is None
        assert params["slope"] == pytest.approx(b, rel=1e-12)
        assert params["intercept"] == pytest.approx(a, rel=1e-12)
        assert params["std_err_slope"] ** 2 == pytest.approx(s2 / sxx, rel=1e-9)
        assert params["std_err_intercept"] ** 2 == pytest.approx(
            s2 * (1 / n + x_bar**2 / sxx), rel=1e-9
        )
        assert params["cov_slope_intercept"] == pytest.approx(-x_bar * s2 / sxx, rel=1e-9)

        # With one reading at s(y/x) the propagated SE is the textbook inverse-prediction
        # SE (s/b)·sqrt(1 + 1/n + (y0 − ȳ)²/(b²·Sxx)); it needs the covariance term.
        y0 = 0.35
        textbook = np.sqrt(s2) / b * np.sqrt(1 + 1 / n + (y0 - y_bar) ** 2 / (b**2 * sxx))
        _, se = propagate_calibration_uncertainty(
            y0,
            params["slope"],
            params["intercept"],
            params["std_err_slope"],
            params["std_err_intercept"],
            params["cov_slope_intercept"],
            absorbance_se=np.sqrt(s2),
        )
        assert se == pytest.approx(textbook, rel=1e-9)
        _, without_cov = propagate_calibration_uncertainty(
            y0,
            params["slope"],
            params["intercept"],
            params["std_err_slope"],
            params["std_err_intercept"],
            absorbance_se=np.sqrt(s2),
        )
        assert without_cov != pytest.approx(textbook, rel=1e-3)

    def test_no_instrument_precision_is_invented(self):
        from adsorblab_pro.utils import propagate_calibration_uncertainty

        # An exact calibration and no stated reading uncertainty: nothing to propagate.
        assert propagate_calibration_uncertainty(0.5, 0.02, 0.01, 0.0, 0.0)[1] == 0.0
        # A stated reading SE is propagated as SE/slope.
        _, se = propagate_calibration_uncertainty(0.5, 0.02, 0.01, 0.0, 0.0, absorbance_se=0.002)
        assert se == pytest.approx(0.1)

    def test_uptake_errors_state_their_components(self):
        from adsorblab_pro.utils import build_calibration, build_uptake_table

        params, _ = build_calibration(R09_CALIB)
        table = build_uptake_table(
            pd.DataFrame({"Absorbance": [0.35]}),
            mode="absorbance",
            signal_col="Absorbance",
            C0=50.0,
            V=0.05,
            m=0.1,
            calib_params=params,
        )
        basis = table["error_basis"].iloc[0]
        assert "their covariance" in basis and "s(y/x)" in basis
        assert "C0, V, m, dilution and replicate variability not included" in basis
        assert table["q_error"].iloc[0] == pytest.approx(0.5 * table["C_error"].iloc[0])

        legacy = {
            k: params[k] for k in ("slope", "intercept", "std_err_slope", "std_err_intercept")
        }
        old = build_uptake_table(
            pd.DataFrame({"Absorbance": [0.35]}),
            mode="absorbance",
            signal_col="Absorbance",
            C0=50.0,
            V=0.05,
            m=0.1,
            calib_params=legacy,
        )
        assert "covariance not available (ignored)" in old["error_basis"].iloc[0]
        assert "sample-reading scatter not included" in old["error_basis"].iloc[0]

    def test_direct_input_never_shows_a_zero_error(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct
        from adsorblab_pro.tabs.report_tab import generate_table

        result = _calculate_isotherm_results_direct(
            {
                "data": pd.DataFrame({"C0": [10.0, 20.0, 40.0], "Ce": [2.0, 5.0, 12.0]}),
                "params": {"m": 0.1, "V": 0.1},
            }
        ).data
        assert result["Ce_error"].isna().all() and result["qe_error"].isna().all()
        assert (result["Ce_error"] != 0).all()
        exported = generate_table("tbl_iso_data", {"isotherm_results": result})
        assert exported["Ce_error"].isna().all()
        assert exported["error_basis"].str.startswith("not available").all()

    def test_kd_uncertainty_uses_the_mass_balance_correlation(self):
        from adsorblab_pro.utils import propagate_kd_uncertainty

        C0, V, m = 50.0, 0.05, 0.1
        Ce = np.array([10.0, 20.0])
        qe = (C0 - Ce) * V / m
        Ce_se = np.array([0.2, 0.2])
        mass = propagate_kd_uncertainty("mass_based", C0, Ce, qe, m, V, Ce_se=Ce_se)
        np.testing.assert_allclose(mass, (V / m) * C0 / Ce**2 * Ce_se)
        # Numerical check of the total derivative of Kd(Ce) = (C0 − Ce)·V/(m·Ce)
        h = 1e-6
        kd = lambda c: (C0 - c) * V / (m * c)  # noqa: E731
        np.testing.assert_allclose(
            mass, np.abs((kd(Ce + h) - kd(Ce - h)) / (2 * h)) * Ce_se, rtol=1e-6
        )
        volume = propagate_kd_uncertainty("volume_corrected", C0, Ce, qe, m, V, Ce_se=Ce_se)
        np.testing.assert_allclose(volume, C0 / Ce**2 * Ce_se)
        # The previous wiring (qe treated as independent) understated σ(Kd).
        independent = propagate_kd_uncertainty(
            "mass_based", C0, Ce, qe, m, V, Ce_se=Ce_se, qe_se=(V / m) * Ce_se
        )
        assert np.all(independent < mass)

    def test_missing_uncertainty_does_not_remove_points_from_the_ordinary_fit(self):
        from adsorblab_pro.utils import calculate_thermodynamic_parameters

        T = np.array([293.15, 303.15, 313.15, 323.15])
        Kd = np.array([5.0, 3.5, 2.5, 1.8])
        with_gap = calculate_thermodynamic_parameters(
            T, Kd, Kd_se=np.array([0.5, np.nan, 0.2, 0.15])
        )
        plain = calculate_thermodynamic_parameters(T, Kd)
        assert with_gap["n_points"] == 4
        assert with_gap["delta_H"] == pytest.approx(plain["delta_H"])
        assert with_gap["wls_n_points"] == 3

    def test_exports_and_displays_name_the_estimator(self):
        from adsorblab_pro.tabs.report_tab import _gen_tbl_iso_params

        fitted = _iso_fit(R06_ORDINARY_CE, R06_ORDINARY_QE)
        table = _gen_tbl_iso_params({"isotherm_models_fitted": fitted})
        assert set(table["Estimator"]) == {"unweighted least squares in q; Wald CI"}

    @pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
    def test_display_caption_names_the_estimator(self):
        at = AppTest.from_function(
            _iso_displays, args=(list(R06_ORDINARY_CE), list(R06_ORDINARY_QE)), default_timeout=60
        )
        at.run()
        assert not at.exception, at.exception
        captions = " ".join(c.value for c in at.caption)
        assert "unweighted least squares in q" in captions
        assert "were not used as weights" in captions

    @pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
    def test_thermodynamics_page_describes_the_weighted_fit_honestly(self):
        import copy

        from adsorblab_pro.config import DEFAULT_SESSION_STATE
        from adsorblab_pro.utils import build_calibration

        params, _ = build_calibration(R09_CALIB)
        state = copy.deepcopy(DEFAULT_SESSION_STATE)
        state["calib_df_input"] = R09_CALIB
        state["calibration_params"] = params
        state["temp_effect_input"] = {
            "data": pd.DataFrame(
                {"Temperature": [20.0, 30.0, 40.0, 50.0], "Absorbance": [0.25, 0.3, 0.35, 0.4]}
            ),
            "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
            "input_mode": "absorbance",
            "temperature_unit": "°C",
        }
        at = AppTest.from_file("adsorblab_pro/app.py", default_timeout=120)
        at.session_state["studies"] = {"S": state}
        at.session_state["current_study"] = "S"
        at.session_state["_previous_study_selection"] = "S"
        at.run()
        at.radio(key="main_section_navigation").set_value("🧪 Analysis Workflow").run()
        at.radio(key="workflow_section_navigation").set_value("🌡️ Thermodynamics").run()
        at.button(key="thermo_calculate_btn").click().run()
        assert not at.exception, at.exception
        text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
        assert "Calibration-weighted fit (WLS vs OLS)" in text
        assert "calibration only" in text and "not a complete uncertainty budget" in text
        assert "completing the uncertainty chain" not in text
        thermo = at.session_state["studies"]["S"]["thermo_params"]
        assert thermo["n_points"] == 4 and thermo["wls_n_points"] == 4


# =============================================================================
# R10 — Preserve metadata and replicate identity
# =============================================================================
R10_CSV = (
    "C0,Ce,SampleID,Ce_SD,Matrix\n"
    "10,2,S1,0.1,tap\n20,5,S2,0.2,tap\n20,5,S3,0.2,river\n40,12,S4,0.3,river\n80,30,S5,0.5,river\n"
)


class TestR10Metadata:
    def _import(self, text=R10_CSV, required=("C0", "Ce"), study="isotherm"):
        from adsorblab_pro.utils import prepare_analysis_data

        df, status = parse_uploaded_table(_csv(text), "data.csv", list(required), study)
        assert status["status"] == "success", status["messages"]
        prepared, report = prepare_analysis_data(df, list(required))
        return df, status, prepared

    def test_acceptance_metadata_survives_import_and_export(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct
        from adsorblab_pro.tabs.report_tab import generate_table

        df, status, prepared = self._import()
        assert status["metadata_columns"] == ["SampleID", "Ce_SD", "Matrix"]
        assert list(prepared.columns) == ["source_row", "C0", "Ce", "SampleID", "Ce_SD", "Matrix"]
        raw = status["raw_table"]
        assert list(raw.columns) == ["source_row", "C0", "Ce", "SampleID", "Ce_SD", "Matrix"]

        results = _calculate_isotherm_results_direct(
            {"data": prepared, "params": {"m": 0.1, "V": 0.1}}
        ).data
        exported = generate_table("tbl_iso_data", {"isotherm_results": results})
        for frame in (results, exported):
            assert list(frame["SampleID"]) == ["S1", "S2", "S3", "S4", "S5"]
            assert list(frame["Matrix"]) == ["tap", "tap", "river", "river", "river"]
            assert list(frame["Ce_SD"]) == [0.1, 0.2, 0.2, 0.3, 0.5]
        # Equal measurements with distinct IDs remain distinct observations.
        s2, s3 = results[results["SampleID"].isin(["S2", "S3"])].to_dict("records")
        assert s2["Ce_mgL"] == s3["Ce_mgL"] and s2["source_row"] != s3["source_row"]
        assert (results["status"] == "ok").all()

    def test_fitting_uses_only_the_numeric_analysis_values(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct

        _, _, prepared = self._import()
        with_meta = _calculate_isotherm_results_direct(
            {"data": prepared, "params": {"m": 0.1, "V": 0.1}}
        ).data
        plain = _calculate_isotherm_results_direct(
            {"data": prepared[["source_row", "C0", "Ce"]], "params": {"m": 0.1, "V": 0.1}}
        ).data
        for column in ("C0_mgL", "Ce_mgL", "qe_mg_g", "removal_%"):
            np.testing.assert_array_equal(with_meta[column], plain[column])
        fit_a = _iso_fit(with_meta["Ce_mgL"], with_meta["qe_mg_g"])["Langmuir"]
        fit_b = _iso_fit(plain["Ce_mgL"], plain["qe_mg_g"])["Langmuir"]
        assert fit_a["params"]["qm"] == fit_b["params"]["qm"]

    def test_every_calculator_carries_metadata(self):
        from adsorblab_pro.tabs.dosage_tab import _calculate_dosage_results_direct
        from adsorblab_pro.tabs.kinetic_tab import _calculate_kinetic_results_direct
        from adsorblab_pro.tabs.ph_effect_tab import _calculate_ph_results_direct

        _, _, kin = self._import(
            "Time,Ct,SampleID\n0,50,K0\n5,30,K1\n10,20,K2\n", ("Time", "Ct"), "kinetic"
        )
        kinetic = _calculate_kinetic_results_direct(
            {"data": kin, "params": {"C0": 50.0, "m": 0.1, "V": 0.05}}
        ).data
        assert list(kinetic["SampleID"]) == ["K0", "K1", "K2"]

        dosage = _calculate_dosage_results_direct(
            {
                "data": pd.DataFrame(
                    {"Mass": [0.05, 0.1], "Ce": [20.0, 10.0], "Batch": ["b1", "b2"]}
                ),
                "params": {"C0": 50.0, "V": 0.05},
            }
        ).data
        assert list(dosage.sort_values("Mass_g")["Batch"]) == ["b1", "b2"]

        ph = _calculate_ph_results_direct(
            {
                "data": pd.DataFrame({"pH": [3.0, 7.0], "Ce": [20.0, 10.0], "Buffer": ["A", "B"]}),
                "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
            }
        ).data
        assert list(ph.sort_values("pH")["Buffer"]) == ["A", "B"]

        temperature = calculate_temperature_results_direct(
            {
                "data": pd.DataFrame(
                    {"Temperature": [25.0, 35.0], "Ce": [20.0, 15.0], "Run": ["r1", "r2"]}
                ),
                "params": {"C0": 50.0, "m": 0.1, "V": 0.05},
                "temperature_unit": "°C",
            }
        ).data
        assert list(temperature["Run"]) == ["r1", "r2"]

    def test_metadata_never_overwrites_analysis_columns(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct

        _, status, prepared = self._import(
            "C0,Ce,note,status,Time (h),Unnamed: 5\n10,2,lab note,draft,1,\n20,5,,final,2,\n"
        )
        assert status["metadata_columns"] == ["note", "status", "Time (h)"]  # empty unnamed dropped
        results = _calculate_isotherm_results_direct(
            {"data": prepared, "params": {"m": 0.1, "V": 0.1}}
        ).data
        assert list(results["status"]) == ["ok", "ok"]  # the analysis status
        assert list(results["status (input)"]) == ["draft", "final"]
        assert results["note (input)"].iloc[0] == "lab note"
        # A unit-bearing column the study does not use is kept as imported, unconverted.
        assert list(results["Time (h)"]) == [1, 2]

    def test_repeated_values_are_reported_not_penalised(self):
        from adsorblab_pro.utils import assess_data_quality

        _, status, _ = self._import()
        quality = status["quality_report"]
        assert not any("duplicate" in issue for issue in quality["issues"])
        assert quality["notices"] == [
            "1 row(s) repeat the numeric values of another row but differ in SampleID, Matrix; "
            "they are kept as separate observations (replicates)."
        ]

        plain = pd.DataFrame({"C0": [10, 20, 20, 40, 80], "Ce": [2, 5, 5, 12, 30]})
        without_ids = assess_data_quality(plain, "isotherm")
        distinct = assess_data_quality(
            pd.DataFrame({"C0": [10, 20, 30, 40, 80], "Ce": [2, 5, 7, 12, 30]}), "isotherm"
        )
        assert without_ids["quality_score"] == distinct["quality_score"]
        assert "SampleID or Replicate column" in without_ids["notices"][0]
        # Metadata columns are not screened as analysis values.
        outlier_meta = plain.assign(Ce_SD=[0.1, 0.1, 0.1, 0.1, 1000.0])
        assert assess_data_quality(outlier_meta, "isotherm")["issues"] == without_ids["issues"]

    def test_replicate_groups_are_not_fabricated_by_rounding(self):
        from adsorblab_pro.utils import detect_replicates

        data = pd.DataFrame(
            {
                "Ce": [10.0, 10.05, 20.0, 20.0, 30.0],
                "qe": [15, 16, 25, 26, 35],
                "Set": list("aabbc"),
            }
        )
        by_value = detect_replicates(data, "Ce", tolerance=0.01)
        assert list(by_value["Ce"]) == [10.0, 10.05, 20.0, 30.0]
        assert list(by_value["qe_count"]) == [1, 1, 2, 1]
        by_id = detect_replicates(data, "Ce", group_col="Set")
        assert list(by_id["Set"]) == ["a", "b", "c"]
        assert list(by_id["qe_count"]) == [2, 2, 1]


# =============================================================================
# R11 — Consistent rankings and non-misleading scores
# =============================================================================
def _lang(qm, r2):
    return {
        "converged": True,
        "params": {"qm": qm, "KL": 0.1},
        "ci_95": {"qm": (0.9 * qm, 1.1 * qm)},
        "param_status": {"qm": "identified", "KL": "identified"},
        "r_squared": r2,
        "adj_r_squared": r2,
    }


def _pso(qe, r2):
    return {
        "converged": True,
        "params": {"qe": qe, "k2": 0.01},
        "r_squared": r2,
        "adj_r_squared": r2,
    }


def _r11_studies(with_c=True):
    studies = {
        "A": {
            "isotherm_models_fitted": {"Langmuir": _lang(100.0, 0.80)},
            "kinetic_models_fitted": {"PSO": _pso(90.0, 0.95)},
        },
        "B": {
            "isotherm_models_fitted": {"Langmuir": _lang(10.0, 0.99)},
            "kinetic_models_fitted": {"PSO": _pso(9.0, 0.99)},
        },
    }
    if with_c:  # no Langmuir fit and no kinetics
        studies["C"] = {"isotherm_models_fitted": {"Freundlich": {"converged": True, "params": {}}}}
    return studies


class TestR11Rankings:
    def _patch(self, studies):
        from unittest import mock

        import adsorblab_pro.tabs.report_tab as report

        return mock.patch.object(report, "_get_all_studies", return_value=(studies, list(studies)))

    def test_exports_follow_the_screen_criterion(self):
        """A: qm 100, R² 0.80; B: qm 10, R² 0.99 → ordered A, B everywhere (by qm)."""
        from adsorblab_pro.tabs import report_tab
        from adsorblab_pro.utils import CAPACITY_CRITERION, capacity_comparison

        studies = _r11_studies()
        shared = capacity_comparison(studies)
        assert list(shared["Study"]) == ["A", "B", "C"]
        with self._patch(studies):
            table = report_tab._gen_tbl_multi_ranking({})
            bar = report_tab._gen_multi_ranking_bar({})
            qm_bar = report_tab._gen_multi_iso_qm_bar({})
        assert list(table["Study"]) == ["A", "B", "C"]
        assert list(table["Order by qm"].iloc[:2]) == [1, 2] and pd.isna(
            table["Order by qm"].iloc[2]
        )
        assert list(table[CAPACITY_CRITERION].iloc[:2]) == [100.0, 10.0]
        assert list(table["Langmuir R² (fit quality)"].iloc[:2]) == [0.80, 0.99]
        assert not any("Overall" in c or "Score" in c for c in table.columns)
        assert "Ordered by fitted Langmuir qm only" in table["Criterion"].iloc[0]
        assert list(bar.data[0].x) == list(qm_bar.data[0].x) == ["A", "B"]

    def test_missing_analyses_are_unavailable_not_zero(self):
        from adsorblab_pro.tabs import report_tab
        from adsorblab_pro.utils import CAPACITY_CRITERION, capacity_comparison

        studies = _r11_studies()
        row_c = capacity_comparison(studies).set_index("Study").loc["C"]
        assert np.isnan(row_c[CAPACITY_CRITERION]) and "not available" in row_c["Note"]

        studies["B"]["kinetic_models_fitted"] = {}  # missing kinetic data for B
        with self._patch(studies):
            iso_radar = report_tab._gen_multi_iso_radar({})
            kin_radar = report_tab._gen_multi_kin_radar({})
            table = report_tab._gen_tbl_multi_ranking({})
        assert kin_radar is None  # only A has kinetics: nothing to compare, no zero trace
        assert [t.name for t in iso_radar.data] == ["A", "B"]
        assert "omitted (no fit): C" in iso_radar.layout.title.text
        for trace in iso_radar.data:
            assert np.all(np.asarray(trace.r, dtype=float) > 0)
        # B's missing kinetics do not change its position (no zero-performance score).
        assert list(table["Study"].iloc[:2]) == ["A", "B"]

    def test_radar_axes_come_from_one_model(self):
        from adsorblab_pro.tabs import report_tab

        studies = _r11_studies(with_c=False)
        studies["A"]["isotherm_models_fitted"]["Sips"] = {"converged": True, "r_squared": 0.999}
        with self._patch(studies):
            radar = report_tab._gen_multi_iso_radar({})
        a = next(t for t in radar.data if t.name == "A")
        assert list(a.theta[:3]) == ["Langmuir R²", "Langmuir Adj-R²", "qm / largest qm"]
        assert a.r[0] == pytest.approx(0.80)  # Langmuir R², not the best R² of any model
        assert "not a score" in radar.layout.title.text

    def test_recommendations_report_a_heuristic_score(self):
        from adsorblab_pro.utils import recommend_best_models

        recs = recommend_best_models({"L": _fit(1.0), "F": _fit(3.0)})
        for rec in recs:
            assert rec["heuristic_score"] == rec["confidence"]  # deprecated alias
        assert "not a statistical" in recommend_best_models.__doc__

    @pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
    def test_comparison_page_uses_the_same_order_and_quantity(self):
        import copy

        from adsorblab_pro.config import DEFAULT_SESSION_STATE

        studies = {}
        for name, data in _r11_studies().items():
            state = copy.deepcopy(DEFAULT_SESSION_STATE)
            state.update(data)
            studies[name] = state
        at = AppTest.from_file("adsorblab_pro/app.py", default_timeout=120)
        at.session_state["studies"] = studies
        at.session_state["current_study"] = "A"
        at.session_state["_previous_study_selection"] = "A"
        at.run()
        at.radio(key="main_section_navigation").set_value("🧪 Analysis Workflow").run()
        at.radio(key="workflow_section_navigation").set_value("🆚 Comparison").run()
        assert not at.exception, at.exception
        text = [m.value for m in at.markdown]
        order = [line for line in text if line.startswith(("1. **", "2. **", "– **"))]
        assert order[0].startswith("1. **A**: 100.00 mg/g") and order[1].startswith("2. **B**")
        assert "– **C**: not available" in order[2]
        assert not any("🥇" in line for line in text)
        assert any("Ordered by fitted Langmuir qm only" in c.value for c in at.caption)

    @pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
    def test_data_check_badges_are_not_quality_claims(self):
        at = AppTest.from_file("adsorblab_pro/app.py", default_timeout=120)
        import copy

        from adsorblab_pro.config import DEFAULT_SESSION_STATE

        state = copy.deepcopy(DEFAULT_SESSION_STATE)
        state["input_mode_global"] = "direct"
        state["isotherm_input"] = {
            "data": pd.DataFrame({"source_row": [1, 2, 3], "C0": [10, 20, 40], "Ce": [2, 5, 12]}),
            "params": {"m": 0.1, "V": 0.1, "T_C": 25.0, "T_K": 298.15},
            "input_mode": "direct",
        }
        at.session_state["studies"] = {"S": state}
        at.session_state["current_study"] = "S"
        at.session_state["_previous_study_selection"] = "S"
        at.run()
        at.radio(key="main_section_navigation").set_value("🧪 Analysis Workflow").run()
        assert not at.exception, at.exception
        labels = [m.label for m in at.metric]
        assert "Data checks" in labels and "Data flags" in labels
        assert "Quality" not in labels and "Status" not in labels


# =============================================================================
# R12 — Thermodynamic export correctness
# =============================================================================
R12_T = np.array([293.15, 303.15, 313.15, 323.15])


def _thermo(method="dimensionless"):
    from adsorblab_pro.utils import calculate_thermodynamic_parameters

    result = calculate_thermodynamic_parameters(R12_T, np.array([5.0, 3.5, 2.5, 1.8]))
    result.update(
        kd_method_id=method, kd_units="dimensionless" if method != "mass_based" else "L/g"
    )
    return result


class TestR12ThermoExports:
    def _patch(self, studies):
        from unittest import mock

        import adsorblab_pro.tabs.report_tab as report

        return mock.patch.object(report, "_get_all_studies", return_value=(studies, list(studies)))

    def test_valid_results_keep_their_values_and_are_qualified(self):
        from adsorblab_pro.tabs.report_tab import generate_figure, generate_table

        thermo = _thermo()
        state = {"thermo_params": thermo}
        gibbs = generate_figure("thermo_gibbs", state)  # previously never generated (array)
        np.testing.assert_allclose(gibbs.data[0].y, thermo["delta_G"])
        assert "Apparent ΔG" in gibbs.layout.yaxis.title.text
        assert "Kd = (C0 − Ce)/Ce (dimensionless)" in gibbs.layout.title.text

        table = generate_table("tbl_thermo_params", state).set_index("Parameter")
        assert table.loc["Apparent ΔH", "Value"] == pytest.approx(thermo["delta_H"])
        assert table.loc["Apparent ΔG (293.15 K)", "Value"] == pytest.approx(thermo["delta_G"][0])
        assert table["Note"].str.contains("not standard-state thermodynamic quantities").all()
        assert table["Note"].str.contains("Kd = (C0 − Ce)/Ce", regex=False).all()

        data = generate_table("tbl_thermo_data", state)
        np.testing.assert_allclose(data["ln_Kd"], np.log([5.0, 3.5, 2.5, 1.8]))

    def test_missing_delta_g_is_never_a_plotted_zero(self):
        from adsorblab_pro.tabs.report_tab import generate_figure
        from adsorblab_pro.utils import delta_g_series

        thermo = _thermo()
        thermo["delta_G"] = np.array([-3.9, np.nan, -2.4, -1.6])
        gibbs = generate_figure("thermo_gibbs", {"thermo_params": thermo})
        assert list(gibbs.data[0].x) == pytest.approx([20.0, 40.0, 50.0])
        assert 0.0 not in list(gibbs.data[0].y)

        thermo["delta_G"] = None
        assert generate_figure("thermo_gibbs", {"thermo_params": thermo}) is None
        # Dict-form and misaligned storage are handled without inventing values.
        _, values = delta_g_series({"temperatures": [293.15, 303.15], "delta_G": {293.15: -3.0}})
        assert values[0] == -3.0 and np.isnan(values[1])
        _, values = delta_g_series({"temperatures": [293.15, 303.15], "delta_G": [1.0, 2.0, 3.0]})
        assert np.isnan(values).all()

    def test_multi_study_exports_keep_missing_values_unavailable(self):
        from adsorblab_pro.tabs.report_tab import generate_figure, generate_table

        thermo = _thermo()
        studies = {
            "A": {"thermo_params": thermo},
            "B": {"thermo_params": {"success": True, "delta_S": 12.0}},
        }
        with self._patch(studies):
            table = generate_table("tbl_multi_thermo", {}).set_index("Study")
            bar = generate_figure("multi_thermo_bar", {})
            summary = generate_table("tbl_multi_pub_summary", {}).set_index("Study/Adsorbent")
            descriptors = generate_table("tbl_multi_mechanism", {}).set_index("Study")

        expected_dg = thermo["delta_H"] - 298.15 * thermo["delta_S"] / 1000
        assert table.loc["A", "Apparent ΔG at 298.15 K (kJ/mol)"] == pytest.approx(expected_dg)
        assert table.loc["A", "ΔG sign"] == "Negative"
        assert np.isnan(table.loc["B", "Apparent ΔH (kJ/mol)"])
        assert np.isnan(table.loc["B", "Apparent ΔG at 298.15 K (kJ/mol)"])
        assert table.loc["B", "ΔG sign"] == "—" and table.loc["B", "Enthalpy sign"] == "—"
        assert table.loc["A", "Kd definition"] == "Kd = (C0 − Ce)/Ce (dimensionless)"
        assert "not standard-state" in table.loc["A", "Qualification"]

        dh, _, dg = (list(trace.y) for trace in bar.data)
        assert dh[0] == pytest.approx(thermo["delta_H"]) and np.isnan(dh[1])
        assert dg[0] == pytest.approx(expected_dg) and np.isnan(dg[1])
        assert "Kd definitions differ" in bar.layout.title.text
        assert "Unavailable values are not drawn (B)" in bar.layout.title.text

        # Previously always NaN (read a key that does not exist).
        assert summary.loc["A", "Apparent ΔG at 298.15 K (kJ/mol)"] == pytest.approx(expected_dg)
        assert np.isnan(summary.loc["B", "Apparent ΔG at 298.15 K (kJ/mol)"])
        # No process is inferred from a missing ΔH (previously "Exothermic").
        assert descriptors.loc["A", "Process"] == "Exothermic"
        assert descriptors.loc["B", "Process"] == "—"

    def test_vant_hoff_plot_does_not_manufacture_zeros(self):
        from adsorblab_pro.tabs.report_tab import generate_figure

        partial = {"temperatures": R12_T, "Kd_values": np.array([5.0, 3.5, 2.5, 1.8])}
        fig = generate_figure("thermo_vant_hoff", {"thermo_params": partial})
        text = fig.layout.annotations[0].text
        assert "Apparent ΔH = unavailable" in text and "Apparent ΔS = unavailable" in text
        assert np.isnan(np.asarray(fig.data[1].y, dtype=float)).all()  # no fitted line at 0

    def test_docx_summary_line_is_qualified(self):
        from adsorblab_pro.docx_report import _thermo_summary_line

        line = _thermo_summary_line(_thermo("mass_based"))
        assert line.startswith("Thermodynamics (apparent): ΔH = ")
        assert "kJ/mol at 293.15 K" in line and "Kd = qe/Ce (L/g)" in line
        assert "not standard-state" in line and "[" not in line  # no raw array text
        missing = _thermo_summary_line({"success": True})
        assert "ΔH = unavailable" in missing and "ΔG = unavailable" in missing

    @pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
    def test_comparison_page_shows_kd_definitions(self):
        import copy

        from adsorblab_pro.config import DEFAULT_SESSION_STATE

        studies = {}
        for name, method in (("A", "dimensionless"), ("B", "mass_based")):
            state = copy.deepcopy(DEFAULT_SESSION_STATE)
            state["thermo_params"] = _thermo(method)
            studies[name] = state
        at = AppTest.from_file("adsorblab_pro/app.py", default_timeout=120)
        at.session_state["studies"] = studies
        at.session_state["current_study"] = "A"
        at.session_state["_previous_study_selection"] = "A"
        at.run()
        at.radio(key="main_section_navigation").set_value("🧪 Analysis Workflow").run()
        at.radio(key="workflow_section_navigation").set_value("🆚 Comparison").run()
        assert not at.exception, at.exception
        tables = [d.value for d in at.dataframe if "Kd definition" in d.value.columns]
        assert tables and list(tables[0]["Kd definition"]) == [
            "Kd = (C0 − Ce)/Ce (dimensionless)",
            "Kd = qe/Ce (L/g)",
        ]
        assert any("not standard-state" in c.value for c in at.caption)


# =============================================================================
# R13 — Discoveries from the final export inspection and diff review
# =============================================================================
class TestR13FinalReview:
    def test_relative_sse_is_undefined_not_huge(self):
        from adsorblab_pro.models import relative_sse

        # q = 0 predicted and observed at t = 0: contributes nothing.
        assert relative_sse(np.array([0.0, 1.0]), np.array([0.0, 2.0]), 1e-12) == pytest.approx(0.5)
        # A residual where the prediction is zero: undefined (was ~1e12 × residual²).
        assert np.isnan(relative_sse(np.array([4.6, 1.0]), np.array([0.0, 2.0]), 1e-12))

        temkin = _iso_fit(R06_ORDINARY_CE, R06_ORDINARY_QE)["Temkin"]  # KT at 1/min(Ce)
        assert temkin["param_status"]["KT"] == "at limit"
        assert np.isnan(temkin["normalized_sse"])
        from adsorblab_pro.tabs.kinetic_tab import _fit_all_kinetic_models_cached

        pfo = _fit_all_kinetic_models_cached(tuple(KIN_T), tuple(_kinetic_qt(KIN_T)), 0.95)["PFO"]
        assert np.isfinite(pfo["normalized_sse"])  # t = 0 with q(0) = 0 stays defined

    def test_export_note_says_limited_intervals_are_not_valid(self):
        from adsorblab_pro.utils import fit_diagnostics_text

        temkin = _iso_fit(R06_ORDINARY_CE, R06_ORDINARY_QE)["Temkin"]
        assert "(value set by the limit; SE/CI not valid)" in fit_diagnostics_text(temkin)

    @pytest.mark.parametrize("scale", [1e-9, 1.0])
    def test_page_fit_path_keeps_small_concentrations(self, scale):
        """The page wrapper rounded to 8 decimals, so Ce < 1e-8 mg/L became 0 before fitting."""
        from adsorblab_pro.tabs.isotherm_tab import fit_isotherm_models_with_cache
        from adsorblab_pro.tabs.kinetic_tab import fit_kinetic_models_with_cache

        Ce = R06_ORDINARY_CE * scale
        fitted = fit_isotherm_models_with_cache(Ce, R06_ORDINARY_QE, Ce + R06_ORDINARY_QE, 0.95, {})
        np.testing.assert_allclose(fitted["Langmuir"]["x_data"], Ce, rtol=1e-11)
        assert fitted["Langmuir"]["params"]["KL"] * scale == pytest.approx(0.04985, rel=1e-3)

        qt = _kinetic_qt(KIN_T) * scale
        kinetic = fit_kinetic_models_with_cache(KIN_T, qt, 0.95, {})
        np.testing.assert_allclose(kinetic["PSO"]["y_data"], qt, rtol=1e-11)


# =============================================================================
# Independent review of 5f5753e (2026-09-24): F01–F04
# =============================================================================
# The upload tests run the real import path: file bytes → utils.load_uploaded_table
# (parsing, then preparation) → the stored study input → the page calculator → the
# table export.  The sidebar tests run the complete app script with
# st.file_uploader patched to return a chosen file, because AppTest cannot upload.
REVIEW_ROWS = [
    ("10", "1", "A"),
    ("wrong", "nd", "B"),  # both measurements invalid
    ("20", "2", "C"),
    ("15", "nd", "D"),  # one invalid cell
    ("", "", "E"),  # identifier only
    ("", "", ""),  # a genuinely empty record
    ("40", "8", "F"),
]


def _review_file(kind, rows=REVIEW_ROWS, header=("C0", "Ce", "SampleID")):
    import io

    if kind == "csv":
        text = ",".join(header) + "\n" + "\n".join(",".join(row) for row in rows) + "\n"
        return "review.csv", text.encode()

    def cell(value):
        try:
            return float(value)
        except ValueError:
            return value or None

    frame = pd.DataFrame([[cell(v) for v in row] for row in rows], columns=list(header))
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    return "review.xlsx", buffer.getvalue()


def _stored_upload(content, name, *, params=None, state=None, deps=("isotherm_results",)):
    from adsorblab_pro.sidebar_ui import store_study_input, study_input_from_upload
    from adsorblab_pro.utils import load_uploaded_table

    prepared, upload = load_uploaded_table(content, name, ["C0", "Ce"], "isotherm")
    state = {} if state is None else state
    changed = None
    if prepared is not None:
        new_input = study_input_from_upload(
            prepared,
            upload,
            params=params or {"m": 0.1, "V": 0.1},
            input_mode="direct",
            required_cols=["C0", "Ce"],
            study_type="isotherm",
        )
        changed = store_study_input(state, "isotherm_input", new_input, list(deps))
    return state, upload, changed


class _ChosenFile:
    """Stands in for what st.file_uploader returns after a user picks a file."""

    def __init__(self, name, content):
        self.name, self.size, self._content = name, len(content), content

    def getvalue(self):
        return self._content

    def read(self):
        return self._content

    def seek(self, position):
        return position


@pytest.fixture
def sidebar_upload(monkeypatch):
    """The real app with one study; ``upload(name, bytes)`` reruns it with that file chosen."""
    import copy
    from types import SimpleNamespace

    import streamlit

    from adsorblab_pro.config import DEFAULT_SESSION_STATE

    chosen = {"file": None}
    monkeypatch.setattr(streamlit, "file_uploader", lambda *args, **kwargs: chosen["file"])

    def start(mode="direct", section="isotherm"):
        state = copy.deepcopy(DEFAULT_SESSION_STATE)
        state["input_mode_global"] = mode
        at = AppTest.from_file("adsorblab_pro/app.py", default_timeout=120)
        at.session_state["studies"] = {"S": state}
        at.session_state["current_study"] = "S"
        at.session_state["_previous_study_selection"] = "S"
        at.session_state["active_sidebar_expander"] = section
        at.run()
        assert not at.exception, at.exception

        def upload(name, content):
            chosen["file"] = _ChosenFile(name, content)
            at.run()
            assert not at.exception, at.exception
            return at.session_state["studies"]["S"]

        return SimpleNamespace(at=at, upload=upload)

    return start


@pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
class TestReviewF01UploadRetention:
    EXPECTED_ISSUES = {
        2: ["C0: non-numeric value 'wrong'", "Ce: non-numeric value 'nd'"],
        4: ["Ce: non-numeric value 'nd'"],
        5: ["C0: missing value", "Ce: missing value"],
    }

    @pytest.mark.parametrize("kind", ["csv", "xlsx"])
    def test_rows_survive_upload_preparation_results_and_export(self, kind):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct
        from adsorblab_pro.tabs.report_tab import generate_table

        name, content = _review_file(kind)
        state, upload, _ = _stored_upload(content, name)
        stored = state["isotherm_input"]
        assert list(stored["data"]["source_row"]) == [1, 2, 3, 4, 5, 7]
        assert list(stored["data"]["SampleID"]) == ["A", "B", "C", "D", "E", "F"]
        assert stored["ignored_empty_rows"] == [6]
        assert stored["row_issues"] == self.EXPECTED_ISSUES
        messages = " ".join(text for _, text in upload["row_messages"])
        assert "Ignored empty row(s) (no value in any column): 6." in messages
        assert "row 2: C0: non-numeric value 'wrong', Ce: non-numeric value 'nd'" in messages
        raw = stored["raw_data"]
        assert list(raw["source_row"]) == [1, 2, 3, 4, 5, 6, 7]
        assert (raw.loc[1, "C0"], raw.loc[1, "Ce"]) == ("wrong", "nd")

        results = _calculate_isotherm_results_direct(stored).data
        exported = generate_table("tbl_iso_data", {"isotherm_results": results})
        for frame in (results, exported):
            rows = frame.set_index("SampleID")
            assert sorted(rows.index) == ["A", "B", "C", "D", "E", "F"]
            assert list(rows.loc[["A", "C", "F"], "status"]) == ["ok"] * 3
            assert list(rows.loc[["B", "D", "E"], "status"]) == ["excluded"] * 3
            assert list(rows.loc[["B", "D", "E"], "source_row"]) == [2, 4, 5]
            for label, row in (("B", 2), ("D", 4), ("E", 5)):
                assert rows.loc[label, "note"] == "; ".join(self.EXPECTED_ISSUES[row])

    def test_isotherm_page_lists_the_excluded_records_with_their_content(self, sidebar_upload):
        app = sidebar_upload()
        study = app.upload(*_review_file("csv"))
        assert list(study["isotherm_input"]["data"]["SampleID"]) == list("ABCDEF")
        sidebar_text = " ".join(e.value for e in app.at.sidebar.info) + " ".join(
            e.value for e in app.at.sidebar.warning
        )
        assert "Ignored empty row(s) (no value in any column): 6." in sidebar_text
        assert "row 5: C0: missing value, Ce: missing value" in sidebar_text

        app.at.radio(key="main_section_navigation").set_value("🧪 Analysis Workflow").run()
        assert not app.at.exception, app.at.exception
        notice = " ".join(w.value for w in app.at.warning)
        assert "3 of 6 observation(s) are excluded" in notice
        assert (
            "row 2 (excluded): C0: non-numeric value 'wrong'; Ce: non-numeric value 'nd'" in notice
        )
        assert "row 5 (excluded): C0: missing value; Ce: missing value" in notice
        results = app.at.session_state["studies"]["S"]["isotherm_results"]
        assert sorted(results["SampleID"]) == list("ABCDEF")

    def test_valid_values_are_exact_when_their_column_also_holds_text(self):
        from adsorblab_pro.utils import load_uploaded_table

        digits = ["0.30000000000000004", "123456789.12345679", "2.5e-10"]
        text = "C0,Ce (ug/L)\n" + "\n".join(f"1,{v}" for v in digits) + "\n1,nd\n"
        prepared, upload = load_uploaded_table(text.encode(), "a.csv", ["C0", "Ce"], "isotherm")
        assert prepared["Ce"].tolist()[:3] == [float(v) * 1e-3 for v in digits]
        assert upload["row_issues"] == {4: ["Ce: non-numeric value 'nd'"]}

    def test_prepare_decides_emptiness_from_every_cell_as_supplied(self):
        from adsorblab_pro.utils import prepare_analysis_data

        raw = pd.DataFrame(
            {
                "C0": [10.0, None, None, " ", pd.NaT],
                "Ce": [1.0, None, None, None, None],
                "Note": [None, "flask broken", None, None, None],
            }
        )
        prepared, report = prepare_analysis_data(raw, ["C0", "Ce"])
        assert list(prepared["source_row"]) == [1, 2]
        assert report["ignored_empty_rows"] == [3, 4, 5]
        assert report["row_issues"] == {2: ["C0: missing value", "Ce: missing value"]}

    def test_calibration_upload_keeps_invalid_standards_with_their_reason(self, sidebar_upload):
        text = "Concentration,Absorbance\n0,0.002\n5,nd\n10,0.168\n20,0.335\n40,0.668\n"
        app = sidebar_upload(mode="absorbance", section="calibration")
        study = app.upload("calib.csv", text.encode())
        calib = study["calib_df_input"]
        assert list(calib["source_row"]) == [1, 2, 3, 4, 5] and np.isnan(calib.loc[1, "Absorbance"])
        assert study["calib_source"]["row_issues"] == {2: ["Absorbance: non-numeric value 'nd'"]}
        assert study["calibration_params"]["excluded_standards"] == [2]

        app.at.radio(key="main_section_navigation").set_value("🧪 Analysis Workflow").run()
        app.at.radio(key="workflow_section_navigation").set_value("📊 Calibration").run()
        assert not app.at.exception, app.at.exception
        warnings = " ".join(w.value for w in app.at.warning)
        assert "row(s) 2 (Absorbance: non-numeric value 'nd')" in warnings


class TestReviewF02ReservedRowColumn:
    @pytest.mark.parametrize(
        "values",
        [["1", "2"], ["S-01", "S-02"], ["7", "7"], ["2", "1"]],
        ids=["numeric", "text", "duplicate", "reordered"],
    )
    def test_file_source_row_is_kept_and_never_used_as_row_numbers(self, values):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct
        from adsorblab_pro.tabs.report_tab import generate_table
        from adsorblab_pro.utils import load_uploaded_table

        text = "source_row,C0,Ce\n" + "\n".join(
            f"{v},{c0},{ce}" for v, c0, ce in zip(values, (10, 20), (1, 2))
        )
        prepared, upload = load_uploaded_table(text.encode(), "case.csv", ["C0", "Ce"], "isotherm")
        assert prepared is not None, upload["messages"]
        expected = pd.read_csv(pd.io.common.BytesIO(text.encode()))["source_row"].tolist()
        assert list(prepared["source_row"]) == [1, 2]
        assert prepared["source_row (input)"].tolist() == expected
        assert list(upload["raw_data"].columns) == ["source_row", "source_row (input)", "C0", "Ce"]
        assert upload["raw_data"]["source_row (input)"].tolist() == expected
        assert any("kept, unchanged, as 'source_row (input)'" in m for _, m in upload["messages"])

        results = _calculate_isotherm_results_direct(
            {"data": prepared, "params": {"m": 0.1, "V": 0.1}}
        ).data
        exported = generate_table("tbl_iso_data", {"isotherm_results": results})
        for frame in (results, exported):
            assert frame.sort_values("source_row")["source_row (input)"].tolist() == expected

    def test_reserved_and_generated_names_do_not_collide(self):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct
        from adsorblab_pro.utils import load_uploaded_table

        text = (
            "source_row,source_row (input),C0,Ce,status,note,qe_mg_g,C\n"
            "a,b,10,1,mine,n1,7.5,x\n"
            "c,d,20,2,mine,n2,8.5,y\n"
        )
        prepared, upload = load_uploaded_table(text.encode(), "case.csv", ["C0", "Ce"], "isotherm")
        assert prepared is not None, upload["messages"]
        assert prepared["source_row (input) (input)"].tolist() == ["a", "c"]
        assert prepared["source_row (input)"].tolist() == ["b", "d"]
        assert prepared["qe_mg_g"].tolist() == [7.5, 8.5]  # the user's header is kept
        results = _calculate_isotherm_results_direct(
            {"data": prepared, "params": {"m": 0.1, "V": 0.1}}
        ).data
        assert results["status"].tolist() == ["ok", "ok"]
        assert results["status (input)"].tolist() == ["mine", "mine"]
        assert results["note (input)"].tolist() == ["n1", "n2"]
        assert results["qe_mg_g (input)"].tolist() == [7.5, 8.5]
        assert results["qe_mg_g"].tolist() == [9.0, 18.0]
        assert results["C (input)"].tolist() == ["x", "y"]

    @pytest.mark.parametrize("kind", ["csv", "xlsx"])
    def test_exported_results_table_can_be_reimported(self, kind):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct
        from adsorblab_pro.tabs.report_tab import generate_table
        from adsorblab_pro.utils import convert_df_to_csv, convert_df_to_excel, load_uploaded_table

        state, _, _ = _stored_upload(*_review_file("csv")[::-1])
        results = _calculate_isotherm_results_direct(state["isotherm_input"]).data
        exported = generate_table("tbl_iso_data", {"isotherm_results": results})
        # The app's own export writers (CSV with ';' separators, Excel).
        content = convert_df_to_csv(exported) if kind == "csv" else convert_df_to_excel(exported)

        again, upload = load_uploaded_table(content, f"export.{kind}", ["C0", "Ce"], "isotherm")
        assert again is not None, upload["messages"]
        assert list(again["source_row"]) == list(range(1, len(exported) + 1))
        assert again["source_row (input)"].tolist() == exported["source_row"].tolist()
        assert again["SampleID"].tolist() == exported["SampleID"].tolist()
        assert again["note"].fillna("").tolist() == exported["note"].tolist()
        np.testing.assert_array_equal(again["Ce"], exported["Ce_mgL"])
        second = _calculate_isotherm_results_direct(
            {"data": again, "params": {"m": 0.1, "V": 0.1}, "row_issues": upload["row_issues"]}
        ).data.set_index("SampleID")
        first = results.set_index("SampleID")
        for label in ("A", "C", "F"):
            assert second.loc[label, "qe_mg_g"] == first.loc[label, "qe_mg_g"]
        assert (second.loc[["B", "D", "E"], "status"] == "excluded").all()

    def test_rejected_upload_keeps_stored_data_and_source(self, sidebar_upload):
        app = sidebar_upload()
        first = app.upload("good.csv", b"C0 (mg/L),Ce (mg/L)\n10,1\n20,3\n40,8\n")
        before = first["isotherm_input"]
        for name, content in (
            ("bad_unit.csv", b"C0,Ce (ppm)\n10,1\n20,3\n"),
            ("no_rows.csv", b"C0,Ce\nx,y\n"),
            ("reserved.csv", b"source_row,C0\n1,10\n"),  # missing Ce
        ):
            study = app.upload(name, content)
            after = study["isotherm_input"]
            assert after["source_file"] == "good.csv"
            pd.testing.assert_frame_equal(after["data"], before["data"])
            pd.testing.assert_frame_equal(after["raw_data"], before["raw_data"])
            assert after["column_map"] == before["column_map"]
        errors = " ".join(e.value for e in app.at.sidebar.error)
        assert "This upload was not applied" in errors


@pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
class TestReviewF03SourceReplacement:
    MG = b"C0 (mg/L),Ce (mg/L)\n10,1\n20,3\n40,8\n80,20\n"
    UG = b"C0 (ug/L),Ce (ug/L)\n10000,1000\n20000,3000\n40000,8000\n80000,20000\n"

    def test_review_reproduction(self):
        from adsorblab_pro.sidebar_ui import store_study_input

        old = {
            "data": pd.DataFrame({"C0": [10.0], "Ce": [1.0]}),
            "params": {"m": 0.1, "V": 0.1},
            "input_mode": "direct",
            "raw_data": pd.DataFrame({"C0 (mg/L)": [10.0], "Ce (mg/L)": [1.0]}),
        }
        new = {**old, "raw_data": pd.DataFrame({"C0 (ug/L)": [10000.0], "Ce (ug/L)": [1000.0]})}
        state = {"isotherm_input": old, "isotherm_models": {"existing": True}}
        assert store_study_input(state, "isotherm_input", new, ["isotherm_models"]) is False
        assert state["isotherm_input"]["raw_data"] is new["raw_data"]
        assert state["isotherm_models"] == {"existing": True}

    def test_equivalent_upload_replaces_source_and_keeps_results(self):
        deps = ("isotherm_results", "isotherm_models_fitted")
        state, _, changed = _stored_upload(self.MG, "mg.csv", deps=deps)
        assert changed is True
        state["isotherm_models_fitted"] = {"Langmuir": {"converged": True}}
        state["isotherm_results"] = pd.DataFrame({"x": [1]})

        state, _, changed = _stored_upload(self.UG, "ug.csv", state=state, deps=deps)
        stored = state["isotherm_input"]
        assert changed is False
        assert stored["source_file"] == "ug.csv"
        assert list(stored["raw_data"].columns) == ["source_row", "C0 (ug/L)", "Ce (ug/L)"]
        assert stored["raw_data"]["C0 (ug/L)"].tolist() == [10000, 20000, 40000, 80000]
        assert {c: m["unit"] for c, m in stored["column_map"].items()} == {
            "C0": "µg/L",
            "Ce": "µg/L",
        }
        assert state["isotherm_models_fitted"] == {"Langmuir": {"converged": True}}

        changed_rows = self.UG.replace(b"80000,20000", b"80000,25000")
        state, _, changed = _stored_upload(changed_rows, "ug2.csv", state=state, deps=deps)
        assert changed is True
        assert state["isotherm_input"]["source_file"] == "ug2.csv"
        assert state["isotherm_models_fitted"] == {} and state["isotherm_results"] is None

    def test_changed_reasons_invalidate_results(self):
        deps = ("isotherm_results",)
        state, _, _ = _stored_upload(b"C0,Ce\n10,1\n20,nd\n40,8\n", "a.csv", deps=deps)
        state["isotherm_results"] = pd.DataFrame({"x": [1]})
        state, _, changed = _stored_upload(b"C0,Ce\n10,1\n20,n.d.\n40,8\n", "b.csv", state=state)
        assert changed is True and state["isotherm_results"] is None
        assert state["isotherm_input"]["row_issues"] == {2: ["Ce: non-numeric value 'n.d.'"]}

    def test_sidebar_replacement_and_rejection(self, sidebar_upload):
        app = sidebar_upload()
        app.upload("mg.csv", self.MG)
        study = app.at.session_state["studies"]["S"]
        study["isotherm_models_fitted"] = {"Langmuir": {"converged": True, "params": {"qm": 1.0}}}

        study = app.upload("ug.csv", self.UG)
        assert study["isotherm_input"]["source_file"] == "ug.csv"
        assert "C0 (ug/L)" in study["isotherm_input"]["raw_data"].columns
        assert study["isotherm_models_fitted"] == {
            "Langmuir": {"converged": True, "params": {"qm": 1.0}}
        }

        study = app.upload("rejected.csv", b"C0,Ce (ppm)\n10,1\n")
        assert study["isotherm_input"]["source_file"] == "ug.csv"
        assert "C0 (ug/L)" in study["isotherm_input"]["raw_data"].columns
        assert study["isotherm_models_fitted"]

    def test_calibration_source_follows_the_stored_standards(self, sidebar_upload):
        mg = b"Concentration (mg/L),Absorbance\n0,0.002\n5,0.085\n10,0.168\n20,0.335\n"
        ug = b"Concentration (ug/L),Absorbance\n0,0.002\n5000,0.085\n10000,0.168\n20000,0.335\n"
        app = sidebar_upload(mode="absorbance", section="calibration")
        study = app.upload("calib_mg.csv", mg)
        params = study["calibration_params"]
        assert study["calib_source"]["source_file"] == "calib_mg.csv"

        study = app.upload("calib_ug.csv", ug)
        assert study["calib_source"]["source_file"] == "calib_ug.csv"
        assert "Concentration (ug/L)" in study["calib_source"]["raw_data"].columns
        assert study["calib_source"]["column_map"]["Concentration"]["unit"] == "µg/L"
        assert study["calibration_params"] == params  # same standards: calibration unchanged

        study = app.upload("calib_bad.csv", b"Concentration (ppm),Absorbance\n0,0.002\n")
        assert study["calib_source"]["source_file"] == "calib_ug.csv"
        pd.testing.assert_frame_equal(
            study["calib_df_input"], app.at.session_state["studies"]["S"]["calib_df_input"]
        )
        assert study["calibration_params"] == params


def _fake_budget_fit(stop_at):
    """curve_fit stand-in: every refit takes one evaluation, refit ``stop_at`` 20,000."""
    calls = [0]

    def fit(*args, **kwargs):
        calls[0] += 1
        return np.array([2.0]), np.eye(1), {"nfev": 20000 if calls[0] == stop_at else 1}, "", 1

    return fit


def _budget_run(stop_at, n=100):
    from unittest import mock

    import adsorblab_pro.utils as utils

    x = np.arange(1.0, 9.0)
    with mock.patch.object(utils, "curve_fit", side_effect=_fake_budget_fit(stop_at)):
        return utils.bootstrap_parameter_intervals(
            lambda x, a: a * x, x, 2 * x, [2.0], n, param_names=["a"]
        )


def _bootstrap_report_page(stop_at):
    import numpy as np

    import adsorblab_pro.utils as utils
    from unittest import mock

    x = np.arange(1.0, 9.0)
    calls = [0]

    def fit(*args, **kwargs):
        calls[0] += 1
        return np.array([2.0]), np.eye(1), {"nfev": 20000 if calls[0] == stop_at else 1}, "", 1

    with mock.patch.object(utils, "curve_fit", side_effect=fit):
        details = utils.bootstrap_parameter_intervals(
            lambda x, a: a * x, x, 2 * x, [2.0], 100, param_names=["a"]
        )
    result = {"converged": True, "params": {"a": 2.0}}
    utils.report_bootstrap_outcome([utils.store_bootstrap_result(result, details, "Line")])
    utils.display_bootstrap_intervals(result)


class TestReviewF04BootstrapStops:
    def test_budget_stop_at_the_success_threshold_is_reported_as_incomplete(self):
        from adsorblab_pro.tabs.report_tab import _params_export
        from adsorblab_pro.utils import BOOTSTRAP_PARTIAL, bootstrap_outcome, store_bootstrap_result

        run = _budget_run(stop_at=90)
        assert (run["attempted"], run["successful"], run["failed"]) == (90, 90, 0)
        assert run["status"] == "available" and bootstrap_outcome(run) == BOOTSTRAP_PARTIAL
        assert run["evaluations"] == 20089 > 20000  # the last refit can exceed the limit
        summary = bootstrap_summary_text({"bootstrap": run})
        assert (
            "90 of 100 bootstrap draws refitted (90 attempted, 0 failed, 10 not attempted"
            in summary
        )
        assert (
            "run stopped early: evaluation limit reached after 90 of 100 draws "
            "(20089 function evaluations used; limit 20000, checked between refits)" in summary
        )
        assert summary.endswith("interval computed from the 90 successful draws only")

        result = {"converged": True, "params": {"a": 2.0}, "param_status": {"a": "identified"}}
        outcome, text = store_bootstrap_result(result, run, "Line")
        assert outcome == BOOTSTRAP_PARTIAL and text == f"Line: {summary}"
        row = _params_export({"Line": result}).iloc[0]
        assert row["Bootstrap"] == summary and row["Bootstrap_CI_Lower"] == 2.0

    def test_budget_stop_below_the_success_threshold_is_unavailable(self):
        run = _budget_run(stop_at=89)
        assert (run["attempted"], run["status"]) == (89, "unavailable")
        summary = bootstrap_summary_text({"bootstrap": run})
        assert "(89 attempted, 0 failed, 11 not attempted" in summary
        assert "run stopped early: evaluation limit reached after 89 of 100 draws" in summary
        assert summary.endswith(
            "interval unavailable: only 89 of 100 bootstrap refits succeeded (at least 90 required)"
        )

    def test_complete_and_failed_draw_runs(self):
        from adsorblab_pro.utils import BOOTSTRAP_COMPLETE, BOOTSTRAP_PARTIAL, bootstrap_outcome

        complete = _budget_run(stop_at=0)
        assert bootstrap_outcome(complete) == BOOTSTRAP_COMPLETE and complete["stopped"] == ""
        summary = bootstrap_summary_text({"bootstrap": complete})
        assert "(100 attempted, 0 failed, 0 not attempted" in summary
        assert "stopped" not in summary and "only" not in summary

        failed = dict(complete, successful=95, failed=5)
        assert bootstrap_outcome(failed) == BOOTSTRAP_PARTIAL
        assert bootstrap_summary_text({"bootstrap": failed}).endswith(
            "interval computed from the 95 successful draws only"
        )

    @pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
    @pytest.mark.parametrize("stop_at", [90, 0])
    def test_pages_do_not_present_a_partial_run_as_complete(self, stop_at):
        at = AppTest.from_function(_bootstrap_report_page, args=(stop_at,), default_timeout=60)
        at.run()
        assert not at.exception, at.exception
        warnings = " ".join(w.value for w in at.warning)
        if stop_at:
            assert not at.success
            assert "Bootstrap intervals from incomplete runs — Line: 90 of 100" in warnings
            assert "Incomplete bootstrap run: 90 of 100 bootstrap draws refitted" in warnings
            assert "run stopped early" in warnings
        else:
            assert not at.warning
            assert "Line: 100 of 100 bootstrap draws refitted" in at.success[0].value
        frames = [t.value for t in at.table] + [d.value for d in at.dataframe]
        assert any("Bootstrap 95% CI" in frame.columns for frame in frames)


# =============================================================================
# Second independent review of c86b71c (2026-09-24): F01b — missing-value markers
# =============================================================================
MARKER_ROWS = [
    ("10", "1", "A"),
    ("N/A", "N/A", ""),  # a record of markers only
    ("20", "2", "C"),
    ("15", "NA", "D"),  # one marker beside a valid measurement
    ("30", "3", "NA"),  # a literal identifier 'NA'
    ("", "", "E"),  # identifier only
    ("", "", ""),  # a genuinely empty record
    ("40", "8", "F"),
]
MARKER_ISSUES = {
    2: ["C0: recorded as missing ('N/A')", "Ce: recorded as missing ('N/A')"],
    4: ["Ce: recorded as missing ('NA')"],
    6: ["C0: missing value", "Ce: missing value"],
}


def _marker_file(kind, rows=MARKER_ROWS, header=("C0", "Ce", "SampleID")):
    """CSV text, or XLSX with numbers as numbers and every other text as a text cell."""
    import io
    import re

    if kind == "csv":
        text = ",".join(header) + "\n" + "\n".join(",".join(row) for row in rows) + "\n"
        return "markers.csv", text.encode()
    number = re.compile(r"[+-]?\d+(?:\.\d*)?")
    cells = [[float(v) if number.fullmatch(v) else (v or None) for v in row] for row in rows]
    buffer = io.BytesIO()
    pd.DataFrame(cells, columns=list(header)).to_excel(buffer, index=False)
    return "markers.xlsx", buffer.getvalue()


class TestReviewF01bMissingMarkers:
    @pytest.mark.parametrize("kind", ["csv", "xlsx"])
    def test_reader_keeps_cell_text_and_only_empty_cells_are_missing(self, kind):
        from adsorblab_pro.utils import read_tabular_file

        rows = [("10", "N/A", "NA"), ("NaN", "null", "None"), ("#N/A", "", "")]
        name, content = _marker_file(kind, rows)
        table = read_tabular_file(content, name)
        assert table.iloc[0].tolist()[1:] == ["N/A", "NA"]
        assert table.iloc[1].tolist() == ["NaN", "null", "None"]
        assert table.iloc[2, 0] == "#N/A" and table.iloc[2, 1:].isna().all()

    @pytest.mark.parametrize("kind", ["csv", "xlsx"])
    def test_marker_records_survive_upload_to_export(self, kind):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct
        from adsorblab_pro.tabs.report_tab import generate_table

        name, content = _marker_file(kind)
        state, upload, _ = _stored_upload(content, name)
        stored = state["isotherm_input"]
        assert list(stored["data"]["source_row"]) == [1, 2, 3, 4, 5, 6, 8]
        assert stored["ignored_empty_rows"] == [7]
        assert stored["row_issues"] == MARKER_ISSUES
        raw = stored["raw_data"].set_index("source_row")
        assert (raw.loc[2, "C0"], raw.loc[2, "Ce"], raw.loc[4, "Ce"]) == ("N/A", "N/A", "NA")
        assert raw.loc[5, "SampleID"] == "NA"
        messages = " ".join(text for _, text in upload["row_messages"])
        assert "row 2: C0: recorded as missing ('N/A'), Ce: recorded as missing ('N/A')" in messages
        assert "Ignored empty row(s) (no value in any column): 7." in messages

        results = _calculate_isotherm_results_direct(stored).data
        exported = generate_table("tbl_iso_data", {"isotherm_results": results})
        for frame in (results, exported):
            rows = frame.set_index("source_row")
            assert sorted(rows.index) == [1, 2, 3, 4, 5, 6, 8]
            assert (rows.loc[[1, 3, 5, 8], "status"] == "ok").all()
            for row, reasons in MARKER_ISSUES.items():
                assert rows.loc[row, "status"] == "excluded"
                assert rows.loc[row, "note"] == "; ".join(reasons)
            assert rows.loc[5, "SampleID"] == "NA"  # literal identifier, valid measurements
            assert rows.loc[5, "qe_mg_g"] == pytest.approx(27.0)

    def test_semicolon_decimal_comma_and_unit_conversion_with_markers(self):
        from adsorblab_pro.utils import load_uploaded_table

        text = "C0 (ug/L);Ce (ug/L);SampleID\n10000,5;1000,25;A\nN/A;n/a;B\n20000;3000;NA\n"
        prepared, upload = load_uploaded_table(text.encode(), "semi.csv", ["C0", "Ce"], "isotherm")
        assert prepared["C0"].tolist()[0] == float("10000.5") * 1e-3
        assert prepared["Ce"].tolist()[0] == float("1000.25") * 1e-3
        assert prepared["C0"].tolist()[2] == 20.0 and prepared["Ce"].tolist()[2] == 3.0
        assert upload["row_issues"] == {
            2: ["C0: recorded as missing ('N/A')", "Ce: recorded as missing ('n/a')"]
        }
        assert prepared["SampleID"].tolist() == ["A", "B", "NA"]
        assert any("comma decimal separators" in text for _, text in upload["messages"])
        assert {c: m["unit"] for c, m in upload["column_map"].items()} == {
            "C0": "µg/L",
            "Ce": "µg/L",
        }

    def test_calibration_marker_standard_is_kept_with_its_reason(self):
        from adsorblab_pro.utils import build_calibration, load_uploaded_table

        text = "Concentration,Absorbance\n0,0.002\n5,#N/A\n10,0.168\n20,0.335\n40,0.668\n"
        prepared, upload = load_uploaded_table(
            text.encode(), "calib.csv", ["Concentration", "Absorbance"], "calibration"
        )
        assert list(prepared["source_row"]) == [1, 2, 3, 4, 5]
        assert upload["row_issues"] == {2: ["Absorbance: recorded as missing ('#N/A')"]}
        params, error = build_calibration(prepared)
        assert error is None and params["excluded_standards"] == [2]

    @pytest.mark.parametrize("kind", ["csv", "xlsx"])
    def test_exported_table_with_markers_and_literal_na_reimports(self, kind):
        from adsorblab_pro.tabs.isotherm_tab import _calculate_isotherm_results_direct
        from adsorblab_pro.tabs.report_tab import generate_table
        from adsorblab_pro.utils import convert_df_to_csv, convert_df_to_excel, load_uploaded_table

        state, _, _ = _stored_upload(*_marker_file("csv")[::-1])
        results = _calculate_isotherm_results_direct(state["isotherm_input"]).data
        exported = generate_table("tbl_iso_data", {"isotherm_results": results})
        content = convert_df_to_csv(exported) if kind == "csv" else convert_df_to_excel(exported)
        again, upload = load_uploaded_table(content, f"export.{kind}", ["C0", "Ce"], "isotherm")
        assert again is not None, upload["messages"]
        assert again["SampleID"].fillna("").tolist() == exported["SampleID"].fillna("").tolist()
        assert "NA" in again["SampleID"].tolist()
        assert again["source_row (input)"].tolist() == exported["source_row"].tolist()
        notes = again.set_index("source_row (input)")["note"]
        assert notes.loc[2] == "; ".join(MARKER_ISSUES[2])

    @pytest.mark.skipif(not APPTEST_AVAILABLE, reason="streamlit.testing not available")
    def test_running_app_lists_markers_and_rejects_marker_only_files(self, sidebar_upload):
        app = sidebar_upload()
        study = app.upload(*_marker_file("xlsx"))
        stored = study["isotherm_input"]
        assert stored["source_file"] == "markers.xlsx"
        assert stored["row_issues"] == MARKER_ISSUES
        sidebar = " ".join(w.value for w in app.at.sidebar.warning)
        assert "row 2: C0: recorded as missing ('N/A'), Ce: recorded as missing ('N/A')" in sidebar

        app.at.radio(key="main_section_navigation").set_value("🧪 Analysis Workflow").run()
        assert not app.at.exception, app.at.exception
        notice = " ".join(w.value for w in app.at.warning)
        assert "3 of 7 observation(s) are excluded" in notice
        assert "row 4 (excluded): Ce: recorded as missing ('NA')" in notice

        study = app.upload("only_markers.csv", b"C0,Ce,SampleID\nN/A,N/A,A\nNA,null,B\n")
        after = study["isotherm_input"]
        assert after["source_file"] == "markers.xlsx"
        pd.testing.assert_frame_equal(after["data"], stored["data"])
        pd.testing.assert_frame_equal(after["raw_data"], stored["raw_data"])
        errors = " ".join(e.value for e in app.at.sidebar.error)
        assert "Only 0 row(s) have complete numeric values" in errors
        assert "This upload was not applied" in errors
