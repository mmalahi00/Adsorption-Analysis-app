# sidebar_ui.py
"""
AdsorbLab Pro - Sidebar User Interface (v2.0.0)
============================================================

Features:
- File upload (Excel/CSV) for all data inputs
- Excel template generation
- Data quality validation with statistical standards
- European decimal format support (comma as decimal separator)
"""

import hashlib
import io
import logging
from typing import Any

import pandas as pd

from adsorblab_pro.streamlit_compat import st

from .utils import (
    CANONICAL_UNITS,
    assess_data_quality,
    load_uploaded_table,
    parse_uploaded_table,
)
from .validation import format_validation_errors, validate_uploaded_file

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
@st.cache_data
def _parse_uploaded_file(file_content: bytes, file_name: str, required_cols: list, study_type: str):
    """
    Parse uploaded file content into a DataFrame.

    This is a pure data processing function (no Streamlit UI calls) to enable caching.
    Column headers are interpreted as quantity + unit and the required columns are
    converted to the canonical units (see ``utils.CANONICAL_UNITS``).

    Returns:
        tuple: (DataFrame or None, status_dict with messages and status)
    """
    return parse_uploaded_table(file_content, file_name, list(required_cols), study_type)


def _show_messages(messages: list[tuple[str, str]]) -> None:
    for msg_type, msg in messages:
        if msg_type == "error":
            st.error(msg)
        elif msg_type == "warning":
            st.warning(msg)
        elif msg_type == "info":
            st.info(msg)
        elif msg_type == "success":
            st.success(msg)


def _read_uploaded_file(uploaded_file, content: bytes, required_cols, study_type):
    """
    Validate, parse and prepare an uploaded file (see ``utils.load_uploaded_table``)
    and show its messages.  Returns ``(analysis view or None, upload record)``.
    """
    # Validate file size and type FIRST
    file_validation = validate_uploaded_file(
        file_size=uploaded_file.size, file_name=uploaded_file.name
    )
    if not file_validation.is_valid:
        st.error(format_validation_errors(file_validation))
        return None, None

    prepared, upload = load_uploaded_table(
        content, uploaded_file.name, list(required_cols), study_type, parse=_parse_uploaded_file
    )

    # Display messages (UI calls must be outside cached function)
    _show_messages(upload["messages"])

    # Display quality report
    quality_report = upload.get("quality_report")
    if quality_report:
        if quality_report["status"] == "success":
            st.success(f"✅ Loaded: {quality_report['quality_score']}/100 data checks (heuristic)")
        elif quality_report["status"] == "warning":
            st.warning(f"⚠️ Loaded: {quality_report['quality_score']}/100 data checks (heuristic)")
        else:
            st.error(f"❌ Loaded: {quality_report['quality_score']}/100 data checks (heuristic)")
        for notice in quality_report.get("notices", []):
            st.info(f"ℹ️ {notice}")

    if prepared is not None and upload["metadata_columns"]:
        st.caption(
            "Kept with each row (not used in calculations): "
            + ", ".join(map(str, upload["metadata_columns"]))
        )
    _show_messages(upload["row_messages"])
    return prepared, upload


# =============================================================================
# PER-STUDY INPUT STATE
# =============================================================================
# The stored inputs of each study (st.session_state.studies[name]) are the single
# authoritative copy of its data and experimental conditions.  Widgets are keyed
# per study and hydrated from that state, and an upload widget that is empty
# ("no new upload") never changes stored data; only a new upload or the explicit
# clear action does.  These helpers take the study dict explicitly so they can be
# tested without a Streamlit runtime.


def study_widget_suffix(study_name: str) -> str:
    """Stable, key-safe suffix that keeps widget state separate for each study."""
    return hashlib.sha1(str(study_name).encode("utf-8")).hexdigest()[:10]


def upload_signature(file_name: str, content: bytes) -> str:
    """Identity of an uploaded file, used to detect a genuinely new upload."""
    return f"{file_name}:{len(content)}:{hashlib.md5(content).hexdigest()}"


def _invalidate_dependents(study_state: dict[str, Any], dependent_keys: list[str]) -> None:
    for key in dependent_keys:
        if key in study_state:
            value = study_state[key]
            if isinstance(value, dict):
                study_state[key] = {}
            elif isinstance(value, list):
                study_state[key] = []
            else:
                study_state[key] = None


def _same_analysis_input(new_input: dict[str, Any], current: dict[str, Any]) -> bool:
    """True when two stored inputs give the same results (data, conditions, mode, reasons)."""
    try:
        return bool(
            new_input["data"].equals(current.get("data"))
            and new_input["params"] == current.get("params", {})
            and new_input.get("input_mode") == current.get("input_mode")
            and (new_input.get("row_issues") or {}) == (current.get("row_issues") or {})
            and new_input.get("temperature_unit") == current.get("temperature_unit")
        )
    except (AttributeError, KeyError, TypeError):
        return False


def store_study_input(
    study_state: dict[str, Any],
    state_key: str,
    new_input: dict[str, Any] | None,
    dependent_keys: list[str],
) -> bool:
    """
    Store (or explicitly clear, with ``None``) one input of one study.

    The new input always replaces the stored one as a whole, so its data and its
    source record (raw table, unit mapping, file name, row issues) always come
    from the same, latest accepted upload.  Dependent results are invalidated
    only when the analysis input changes: data, conditions, input mode, row
    reasons or temperature unit.  An equivalent upload (e.g. the same values in
    µg/L instead of mg/L) keeps the results valid.  Returns True when the
    analysis input changed (dependents were invalidated).
    """
    current = study_state.get(state_key)
    if new_input is None:
        changed = current is not None
    elif current is None:
        changed = True
    else:
        changed = not _same_analysis_input(new_input, current)

    study_state[state_key] = new_input
    if changed:
        _invalidate_dependents(study_state, dependent_keys)
        reports = study_state.setdefault("data_quality_reports", {})
        if new_input and "data" in new_input:
            study_type = state_key.replace("_input", "").replace("_", " ")
            reports[state_key] = assess_data_quality(new_input["data"], study_type)
        else:
            reports.pop(state_key, None)
    return changed


def upload_source(upload: dict[str, Any]) -> dict[str, Any]:
    """Source record of an accepted upload, stored with the data it produced."""
    return {
        "source_file": upload["source_file"],
        # The uploaded table as read (all columns, source rows).
        "raw_data": upload["raw_data"],
        # Source header, unit and conversion of each required column.
        "column_map": upload["column_map"],
        "metadata_columns": upload["metadata_columns"],
        # Import problems per source row, shown as exclusion reasons.
        "row_issues": upload["row_issues"],
        "ignored_empty_rows": upload["ignored_empty_rows"],
    }


def study_input_from_upload(
    prepared,
    upload: dict[str, Any],
    *,
    params: dict[str, Any],
    input_mode: str,
    required_cols: list[str],
    study_type: str,
) -> dict[str, Any]:
    """The stored input of one analysis built from an accepted upload."""
    new_input = {
        "data": prepared,
        "params": dict(params),
        "input_mode": input_mode,
        # Uploaded values are converted to these units on import.
        "units": {col: CANONICAL_UNITS.get(col, "") for col in required_cols},
        **upload_source(upload),
    }
    if study_type == "temperature":
        new_input["temperature_unit"] = CANONICAL_UNITS["Temperature"]
    return new_input


def store_calibration_upload(study_state: dict[str, Any], prepared, upload: dict[str, Any]) -> None:
    """
    Store accepted calibration data together with their source record.

    Both are replaced on every accepted upload (never one without the other); the
    calibration step then decides whether the calibration itself changed.
    """
    study_state["calib_df_input"] = prepared
    study_state["calib_source"] = upload_source(upload)


def stored_conditions(study_state: dict[str, Any], state_key: str) -> dict[str, Any]:
    """Experimental conditions stored for one input of one study (may be empty)."""
    current = study_state.get(state_key)
    if isinstance(current, dict) and isinstance(current.get("params"), dict):
        return current["params"]
    return study_state.get("input_conditions", {}).get(state_key, {})


def update_study_conditions(
    study_state: dict[str, Any],
    state_key: str,
    params: dict[str, Any],
    dependent_keys: list[str],
) -> bool:
    """
    Record the conditions shown in the widgets for one study.

    When stored data exist and the conditions differ, the stored input is updated
    and its dependents are invalidated.  Returns True when stored data changed.
    """
    study_state.setdefault("input_conditions", {})[state_key] = dict(params)
    current = study_state.get(state_key)
    if isinstance(current, dict) and current.get("params") != params:
        study_state[state_key] = {**current, "params": dict(params)}
        _invalidate_dependents(study_state, dependent_keys)
        return True
    return False


def _clear_study_input(study_name: str, state_key: str, dependent_keys: list[str]) -> None:
    """Callback for the explicit clear action: affects one input of one study only."""
    study_state = st.session_state.get("studies", {}).get(study_name)
    if study_state is None:
        return
    if state_key == "calib_df_input":
        study_state["calib_df_input"] = None
        study_state["calib_source"] = None
    else:
        store_study_input(study_state, state_key, None, dependent_keys)
    study_state.setdefault("upload_signatures", {}).pop(state_key, None)
    nonces = study_state.setdefault("uploader_nonce", {})
    nonces[state_key] = nonces.get(state_key, 0) + 1  # resets the upload widget


def _handle_input_change(
    state_key: str, new_input_dict: dict[str, Any] | None, dependent_keys: list[str]
) -> None:
    """Store an input for the active study (see :func:`store_study_input`)."""
    active_study_name = st.session_state.get("current_study")
    if not active_study_name:
        st.sidebar.error("Please add or select a study first.")
        return
    store_study_input(
        st.session_state.studies[active_study_name], state_key, new_input_dict, dependent_keys
    )


def _get_global_input_mode() -> str:
    """Read the per-study input mode selected in 📊 Display Units."""
    active_study_name = st.session_state.get("current_study")
    studies = st.session_state.get("studies", {})

    if active_study_name and active_study_name in studies:
        return studies[active_study_name].get("input_mode_global", "absorbance")

    return "absorbance"


def _generate_excel_template(columns: list[str], study_type: str) -> io.BytesIO:
    """
    Generate Excel template with example data loaded from the examples folder.

    Args:
        columns: List of required column names for this study
        study_type: Type of study (e.g., 'calibration', 'isotherm', 'kinetic', etc.)

    Returns:
        io.BytesIO: Excel file buffer ready for download
    """
    from pathlib import Path

    # Mapping of study types to their corresponding CSV files in examples folder
    study_type_to_file = {
        "calibration": "calibration_data.csv",
        "isotherm": "isotherm_data.csv",
        "isotherm_direct": "isotherm_direct.csv",
        "kinetic": "kinetic_data.csv",
        "kinetic_direct": "kinetic_direct.csv",
        "dosage": "dosage_data.csv",
        "dosage_direct": "dosage_direct.csv",
        "ph_effect": "ph_effect_data.csv",
        "ph_effect_direct": "ph_effect_direct.csv",
        "temperature": "temperature_data.csv",
        "temperature_direct": "temperature_direct.csv",
    }

    # Get the examples folder path (relative to this module)
    examples_folder = Path(__file__).parent.parent / "examples"

    # Try to load from examples folder, fallback to generated data if file doesn't exist
    csv_file = study_type_to_file.get(study_type)
    df = None
    source_info = ""

    if csv_file and examples_folder.exists():
        csv_path = examples_folder / csv_file
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                # Select only required columns that exist in the file
                available_cols = [col for col in columns if col in df.columns]
                if available_cols:
                    df = df[available_cols].copy()
                    source_info = f"Loaded from examples/{csv_file}"
            except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError) as e:
                logger.warning(f"Could not load {csv_file}: {e}")
                df = None

    # Fallback: Generate minimal example data if file not found
    if df is None or df.empty:
        df = pd.DataFrame({col: [] for col in columns})
        # Add a few empty rows for user guidance
        df = pd.concat([df, pd.DataFrame([{col: None for col in columns}] * 5)], ignore_index=True)
        source_info = "Generated template (examples folder not found)"

    # Create Excel workbook
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # Data sheet
        df.to_excel(writer, index=False, sheet_name="Data")

        # Instructions sheet
        unit_convention = "; ".join(
            f"{col} in {CANONICAL_UNITS[col]}" if CANONICAL_UNITS.get(col) else f"{col} (no unit)"
            for col in columns
        )
        instructions_df = pd.DataFrame(
            [
                {
                    "Study Type": study_type.replace("_", " ").title(),
                    "Required Columns": ", ".join(columns),
                    "Units (headers without a unit)": unit_convention,
                    "Other units": (
                        "State the unit in the header to have it converted, e.g. "
                        "'Ce (µg/L)', 'Time (h)', 'Mass (mg)', 'Temperature (K)'. "
                        "Ambiguous units (ppm, molar units) are refused."
                    ),
                    "Source": source_info,
                    "Instructions": "Replace example data with your experimental values",
                    "Note": "Include replicates for error estimation and quality validation",
                }
            ]
        )
        instructions_df.to_excel(writer, index=False, sheet_name="Instructions")

    buffer.seek(0)
    return buffer


def _render_enhanced_study_input(config):
    """Render study input with file upload only."""
    study_type = config.get("study_type", "")

    # Studies that support direct concentration input
    direct_input_studies = ["isotherm", "kinetic", "dosage", "ph_effect", "temperature"]

    # Global mode is chosen in 📊 Display Units (adsorption_app.py)
    input_mode = _get_global_input_mode() if study_type in direct_input_studies else "absorbance"

    # Update required columns based on input mode
    if input_mode == "direct":
        if study_type == "isotherm":
            config["required_cols"] = ["C0", "Ce"]
        elif study_type == "kinetic":
            config["required_cols"] = ["Time", "Ct"]
        elif study_type == "dosage":
            config["required_cols"] = ["Mass", "Ce"]
        elif study_type == "ph_effect":
            config["required_cols"] = ["pH", "Ce"]
        elif study_type == "temperature":
            config["required_cols"] = ["Temperature", "Ce"]

    study_name = st.session_state.get("current_study")
    study_state = st.session_state.studies[study_name]
    suffix = study_widget_suffix(study_name)
    state_key = config["state_key"]
    dependent_keys = config["dependent_keys"]
    conditions = stored_conditions(study_state, state_key)

    with st.sidebar.container(border=True):
        st.markdown(f"#### {config['expander_title']}")
        st.markdown(f"*{config['description']}*")

        # --- Fixed Experimental Conditions (hydrated from this study's state) ---
        st.markdown(f"**{config['intro_text']}**")
        fixed_params = {}
        cols = st.columns(len(config["fixed_params"]))
        for i, (param_key, param_config) in enumerate(config["fixed_params"].items()):
            stored_value = conditions.get(param_key)
            initial = float(stored_value) if stored_value is not None else param_config["value"]
            with cols[i % len(cols)]:
                fixed_params[param_key] = st.number_input(
                    param_config["label"],
                    value=initial,
                    min_value=param_config["min_value"],
                    step=param_config["step"],
                    help=param_config["help"],
                    key=f"{config['key_prefix']}{param_key}__{suffix}",
                    format="%.3f",
                )
        # Convert temperature to Kelvin if provided
        if "T_C" in fixed_params and fixed_params["T_C"] is not None:
            fixed_params["T_K"] = float(fixed_params["T_C"]) + 273.15
        update_study_conditions(study_state, state_key, fixed_params, dependent_keys)

        st.markdown("---")

        # --- Data Input Section (File Upload Only) ---
        stored_input = study_state.get(state_key)
        stored_mode = stored_input.get("input_mode", "absorbance") if stored_input else None
        if stored_input is not None and stored_mode != input_mode:
            st.info(
                f"Stored {config['study_name'].lower()} data were entered in {stored_mode} mode "
                f"and are still analysed as {stored_mode} data. Upload {input_mode}-mode data to "
                "replace them, or clear them."
            )

        template_study_type = f"{study_type}_direct" if input_mode == "direct" else study_type
        nonce = study_state.get("uploader_nonce", {}).get(state_key, 0)
        uploaded_file = st.file_uploader(
            f"📁 Upload {config['study_name']} Data",
            type=["xlsx", "xls", "csv"],
            key=f"{config['key_prefix']}file_{input_mode}__{suffix}_{nonce}",
            help="Upload Excel or CSV file. Replaces this study's stored data only.",
        )

        # Only a *new* upload changes stored data; an empty widget means "no new
        # upload", not "clear the study".
        if uploaded_file is not None:
            content = uploaded_file.getvalue()
            signature = upload_signature(uploaded_file.name, content)
            if signature != study_state.get("upload_signatures", {}).get(state_key):
                df_for_analysis, upload = _read_uploaded_file(
                    uploaded_file, content, config["required_cols"], study_type
                )
                if df_for_analysis is not None:
                    new_input = study_input_from_upload(
                        df_for_analysis,
                        upload,
                        params=fixed_params,
                        input_mode=input_mode,
                        required_cols=config["required_cols"],
                        study_type=study_type,
                    )
                    store_study_input(study_state, state_key, new_input, dependent_keys)
                    study_state.setdefault("upload_signatures", {})[state_key] = signature
                    excluded = len(upload["row_issues"])
                    st.success(
                        f"✅ {len(df_for_analysis)} rows loaded into '{study_name}'"
                        + (
                            f"; {excluded} of them kept as excluded (reasons above)."
                            if excluded
                            else "."
                        )
                    )
                else:
                    st.error(
                        "This upload was not applied. "
                        + (
                            "The previously stored data for this study are unchanged."
                            if stored_input is not None
                            else "No data are stored for this analysis."
                        )
                    )

        stored_input = study_state.get(state_key)
        if stored_input is not None:
            source = stored_input.get("source_file")
            st.caption(
                f"Stored for '{study_name}': {len(stored_input['data'])} rows"
                + (f" from {source}" if source else "")
                + f" ({stored_input.get('input_mode', 'absorbance')} mode)."
            )
            st.button(
                "🗑️ Clear stored data",
                key=f"{config['key_prefix']}clear__{suffix}",
                help="Removes this analysis' data and derived results for this study only.",
                on_click=_clear_study_input,
                args=(study_name, state_key, dependent_keys),
            )

        template_buffer = _generate_excel_template(config["required_cols"], template_study_type)
        st.download_button(
            "📥 Download Template",
            template_buffer,
            f"{config['study_type']}_template.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{config['key_prefix']}template_{input_mode}__{suffix}",
        )


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================


def _get_active_expander():
    """Get the currently active expander, defaulting to calibration."""
    if "active_sidebar_expander" not in st.session_state:
        st.session_state.active_sidebar_expander = "calibration"
    return st.session_state.active_sidebar_expander


def _set_active_expander(section: str):
    """Set the active expander section."""
    st.session_state.active_sidebar_expander = section


def render_sidebar_content():
    """Render all sidebar input sections with accordion behavior."""

    active_study_name = st.session_state.get("current_study")
    studies = st.session_state.get("studies", {})

    # If no studies exist yet
    if not studies:
        st.sidebar.info("Add a new study to enable 📥 Data Input.")
        return

    # If studies exist but none selected (or invalid selection)
    if not active_study_name or active_study_name not in studies:
        st.sidebar.info("Select a study to enable 📥 Data Input.")
        return

    global_mode = _get_global_input_mode()

    # Section labels with emojis
    section_options = {
        "calibration": "📊 Calibration Curve",
        "isotherm": "📈 Isotherm Study",
        "kinetic": "⏱️ Kinetic Study",
        "dosage": "⚖️ Dosage Effect",
        "ph_effect": "🧪 pH Effect",
        "temperature": "🌡️ Temperature Effect",
    }

    # Hide calibration completely when the user selects Direct input
    if global_mode == "direct":
        section_options.pop("calibration", None)

    # Select analysis to input data for
    st.sidebar.markdown("Select analysis to input data for:")

    allowed_sections = list(section_options.keys())

    # Get current active expander (and fix it if it points to a hidden section)
    current_selection = _get_active_expander()
    if current_selection not in allowed_sections:
        current_selection = allowed_sections[0]
        _set_active_expander(current_selection)

    # Radio button selector
    selected_section = st.sidebar.radio(
        label="Select analysis",
        options=allowed_sections,
        format_func=lambda x: section_options[x],
        index=allowed_sections.index(current_selection),
        key="data_input_selector",
        label_visibility="collapsed",
    )

    # Update active expander if changed
    if selected_section != current_selection:
        _set_active_expander(selected_section)

    active_expander = selected_section

    st.sidebar.markdown("---")

    # 1. CALIBRATION
    if global_mode != "direct" and active_expander == "calibration":
        study_state = studies[active_study_name]
        suffix = study_widget_suffix(active_study_name)
        with st.sidebar.container(border=True):
            st.markdown("#### 📊 Calibration Curve")
            st.markdown("*Establish Absorbance-Concentration relationship*")

            nonce = study_state.get("uploader_nonce", {}).get("calib_df_input", 0)
            uploaded_file = st.file_uploader(
                "📁 Upload Calibration Data",
                type=["xlsx", "xls", "csv"],
                key=f"calib_file__{suffix}_{nonce}",
                help="Replaces this study's calibration data only.",
            )
            # Only a new upload replaces the stored calibration data.
            if uploaded_file is not None:
                content = uploaded_file.getvalue()
                signature = upload_signature(uploaded_file.name, content)
                if signature != study_state.get("upload_signatures", {}).get("calib_df_input"):
                    # Stored even if too small or flawed: the calibration step then
                    # rejects it with the reason instead of reusing an older curve.
                    validated_calib, upload = _read_uploaded_file(
                        uploaded_file, content, ["Concentration", "Absorbance"], "calibration"
                    )
                    if validated_calib is not None:
                        store_calibration_upload(study_state, validated_calib, upload)
                        study_state.setdefault("upload_signatures", {})["calib_df_input"] = (
                            signature
                        )
                        excluded = len(upload["row_issues"])
                        st.success(
                            f"✅ {len(validated_calib)} standards loaded into '{active_study_name}'"
                            + (
                                f"; {excluded} of them kept as excluded (reasons above)."
                                if excluded
                                else "."
                            )
                        )
                    else:
                        st.error(
                            "This calibration upload was not applied; the stored calibration "
                            "data (if any) are unchanged."
                        )

            stored_calib = study_state.get("calib_df_input")
            if stored_calib is not None:
                st.caption(f"Stored for '{active_study_name}': {len(stored_calib)} standards.")
                st.button(
                    "🗑️ Clear stored calibration",
                    key=f"calib_clear__{suffix}",
                    help="Removes this study's calibration data and the calibration built from it.",
                    on_click=_clear_study_input,
                    args=(active_study_name, "calib_df_input", []),
                )

            template_buffer = _generate_excel_template(
                ["Concentration", "Absorbance"], "calibration"
            )
            st.download_button(
                "📥 Download Template",
                template_buffer,
                "calibration_template.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"calib_template__{suffix}",
            )

    # 2. ISOTHERM
    if active_expander == "isotherm":
        isotherm_config = {
            "study_name": "Isotherm",
            "study_type": "isotherm",
            "expander_title": "📈 Isotherm Study",
            "description": "Equilibrium adsorption at different concentrations",
            "intro_text": "Fixed experimental conditions:",
            "key_prefix": "iso_",
            "state_key": "isotherm_input",
            "required_cols": ["Concentration", "Absorbance"],
            "fixed_params": {
                "m": {
                    "label": "Mass (g)",
                    "value": 0.025,
                    "min_value": 1e-9,
                    "step": 0.001,
                    "help": "Adsorbent mass",
                },
                "V": {
                    "label": "Volume (L)",
                    "value": 0.050,
                    "min_value": 1e-6,
                    "step": 0.01,
                    "help": "Solution volume",
                },
                "T_C": {
                    "label": "Temperature (°C)",
                    "value": 25.0,
                    "min_value": -50.0,
                    "step": 1.0,
                    "help": "Experimental temperature (°C).",
                },
            },
            "dependent_keys": ["isotherm_results", "isotherm_models_fitted"],
        }
        _render_enhanced_study_input(isotherm_config)

    # 3. KINETIC
    if active_expander == "kinetic":
        kinetic_config = {
            "study_name": "Kinetics",
            "study_type": "kinetic",
            "expander_title": "⏱️ Kinetic Study",
            "description": "Adsorption capacity vs time",
            "intro_text": "Fixed experimental conditions:",
            "key_prefix": "kin_",
            "state_key": "kinetic_input",
            "required_cols": ["Time", "Absorbance"],
            "fixed_params": {
                "C0": {
                    "label": "C₀ (mg/L)",
                    "value": 50.0,
                    "min_value": 0.0,
                    "step": 1.0,
                    "help": "Initial concentration",
                },
                "m": {
                    "label": "Mass (g)",
                    "value": 0.025,
                    "min_value": 1e-9,
                    "step": 0.001,
                    "help": "Adsorbent mass",
                },
                "V": {
                    "label": "Volume (L)",
                    "value": 0.050,
                    "min_value": 1e-6,
                    "step": 0.01,
                    "help": "Solution volume",
                },
            },
            "dependent_keys": ["kinetic_results_df", "kinetic_models_fitted"],
        }
        _render_enhanced_study_input(kinetic_config)

    # 4. DOSAGE
    if active_expander == "dosage":
        dosage_config = {
            "study_name": "Dosage Effect",
            "study_type": "dosage",
            "expander_title": "⚖️ Dosage Effect",
            "description": "Effect of adsorbent mass on removal",
            "intro_text": "Fixed experimental conditions:",
            "key_prefix": "dos_",
            "state_key": "dosage_input",
            "required_cols": ["Mass", "Absorbance"],
            "fixed_params": {
                "C0": {
                    "label": "C₀ (mg/L)",
                    "value": 50.0,
                    "min_value": 0.0,
                    "step": 1.0,
                    "help": "Initial concentration",
                },
                "V": {
                    "label": "Volume (L)",
                    "value": 0.050,
                    "min_value": 1e-6,
                    "step": 0.01,
                    "help": "Solution volume",
                },
            },
            "dependent_keys": ["dosage_results"],
        }
        _render_enhanced_study_input(dosage_config)

    # 5. pH EFFECT
    if active_expander == "ph_effect":
        ph_config = {
            "study_name": "pH Effect",
            "study_type": "ph_effect",
            "expander_title": "🧪 pH Effect",
            "description": "Effect of pH on adsorption",
            "intro_text": "Fixed experimental conditions:",
            "key_prefix": "ph_",
            "state_key": "ph_effect_input",
            "required_cols": ["pH", "Absorbance"],
            "fixed_params": {
                "C0": {
                    "label": "C₀ (mg/L)",
                    "value": 50.0,
                    "min_value": 0.0,
                    "step": 1.0,
                    "help": "Initial concentration",
                },
                "m": {
                    "label": "Mass (g)",
                    "value": 0.025,
                    "min_value": 1e-9,
                    "step": 0.001,
                    "help": "Adsorbent mass",
                },
                "V": {
                    "label": "Volume (L)",
                    "value": 0.050,
                    "min_value": 1e-6,
                    "step": 0.01,
                    "help": "Solution volume",
                },
            },
            "dependent_keys": ["ph_effect_results"],
        }
        _render_enhanced_study_input(ph_config)

    # 6. TEMPERATURE
    if active_expander == "temperature":
        temp_config = {
            "study_name": "Temperature Effect",
            "study_type": "temperature",
            "expander_title": "🌡️ Temperature Effect",
            "description": "Effect of temperature (for thermodynamics)",
            "intro_text": "Fixed experimental conditions:",
            "key_prefix": "temp_",
            "state_key": "temp_effect_input",
            "required_cols": ["Temperature", "Absorbance"],
            "fixed_params": {
                "C0": {
                    "label": "C₀ (mg/L)",
                    "value": 50.0,
                    "min_value": 0.0,
                    "step": 1.0,
                    "help": "Initial concentration",
                },
                "m": {
                    "label": "Mass (g)",
                    "value": 0.025,
                    "min_value": 1e-9,
                    "step": 0.001,
                    "help": "Adsorbent mass",
                },
                "V": {
                    "label": "Volume (L)",
                    "value": 0.050,
                    "min_value": 1e-6,
                    "step": 0.01,
                    "help": "Solution volume",
                },
            },
            "dependent_keys": ["temp_effect_results", "thermo_params"],
        }
        _render_enhanced_study_input(temp_config)
