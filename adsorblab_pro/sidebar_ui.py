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

import io
import logging
from typing import Any

import pandas as pd

from adsorblab_pro.streamlit_compat import st

from .utils import (
    assess_data_quality,
    standardize_column_name,
    validate_data_editor,
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

    Returns:
        tuple: (DataFrame or None, status_dict with messages and status)
    """
    import io as io_module

    status = {"messages": [], "status": "success", "quality_report": None}

    try:
        # Read file based on extension
        if file_name.endswith(".csv"):
            try:
                # Try reading with common European separator first
                df = pd.read_csv(io_module.BytesIO(file_content), sep=";", decimal=",")
                if len(df.columns) == 1:  # If that fails, might be comma-separated
                    df = pd.read_csv(io_module.BytesIO(file_content), sep=",", decimal=".")
            except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
                df = pd.read_csv(io_module.BytesIO(file_content), sep=",", decimal=".")
        else:  # For Excel files
            df = pd.read_excel(io_module.BytesIO(file_content))

        # Standardize column names
        col_map = {col: standardize_column_name(col) for col in df.columns}
        df.rename(columns=col_map, inplace=True)

        # Check for required columns
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            status["messages"].append(("error", f"Missing columns: {', '.join(missing)}"))
            status["messages"].append(("info", f"Found columns: {', '.join(df.columns.tolist())}"))
            status["status"] = "error"
            return None, status

        # Handle comma decimals
        comma_was_replaced = False
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                str_col = df[col].astype(str)
                has_commas = str_col.str.contains(",", regex=False).any()

                if has_commas:
                    df[col] = str_col.str.replace(",", ".", regex=False)
                    comma_was_replaced = True

                df[col] = pd.to_numeric(df[col], errors="coerce")

        if comma_was_replaced:
            status["messages"].append(("info", "💡 Detected European decimal format. Replaced commas (,) with periods (.) as decimal separators."))

        # Quality assessment
        quality_report = assess_data_quality(df, study_type)
        status["quality_report"] = quality_report

        return df[required_cols].reset_index(drop=True), status

    except Exception as e:
        status["messages"].append(("error", f"Error reading file: {e}"))
        status["status"] = "error"
        return None, status


def _read_uploaded_file(uploaded_file, required_cols, study_type):
    """Enhanced file reader with validation and comma decimal handling."""
    if uploaded_file is None:
        return None

    # Validate file size and type FIRST
    file_validation = validate_uploaded_file(
        file_size=uploaded_file.size, file_name=uploaded_file.name
    )
    if not file_validation.is_valid:
        st.error(format_validation_errors(file_validation))
        return None

    # Read file content for caching (file objects aren't hashable)
    file_content = uploaded_file.read()
    uploaded_file.seek(0)  # Reset for potential re-reading

    # Call cached parsing function
    df, status = _parse_uploaded_file(file_content, uploaded_file.name, required_cols, study_type)

    # Display messages (UI calls must be outside cached function)
    for msg_type, msg in status.get("messages", []):
        if msg_type == "error":
            st.error(msg)
        elif msg_type == "warning":
            st.warning(msg)
        elif msg_type == "info":
            st.info(msg)
        elif msg_type == "success":
            st.success(msg)

    # Display quality report
    quality_report = status.get("quality_report")
    if quality_report:
        if quality_report["status"] == "success":
            st.success(f"✅ Loaded: {quality_report['quality_score']}/100 quality")
        elif quality_report["status"] == "warning":
            st.warning(f"⚠️ Loaded: {quality_report['quality_score']}/100 quality")
        else:
            st.error(f"❌ Loaded: {quality_report['quality_score']}/100 quality")

    return df


def _handle_input_change(
    state_key: str, new_input_dict: dict[str, Any] | None, dependent_keys: list[str]
) -> None:
    """State management with quality tracking for the active study."""
    active_study_name = st.session_state.get("current_study")
    if not active_study_name:
        st.sidebar.error("Please add or select a study first.")
        return

    current_study_state = st.session_state.studies[active_study_name]

    current_input = current_study_state.get(state_key)
    needs_update = False

    if new_input_dict is None:
        if current_input is not None:
            needs_update = True
    elif current_input is None:
        needs_update = True
    else:
        try:
            data_equal = new_input_dict["data"].equals(current_input.get("data"))
            params_equal = new_input_dict["params"] == current_input.get("params", {})
            if not data_equal or not params_equal:
                needs_update = True
        except (AttributeError, KeyError, TypeError):
            needs_update = True

    if needs_update:
        current_study_state[state_key] = new_input_dict
        for key in dependent_keys:
            if key in current_study_state:
                if isinstance(current_study_state[key], dict):
                    current_study_state[key] = {}
                elif isinstance(current_study_state[key], list):
                    current_study_state[key] = []
                else:
                    current_study_state[key] = None

        if new_input_dict and "data" in new_input_dict:
            study_type = state_key.replace("_input", "").replace("_", " ")
            quality_report = assess_data_quality(new_input_dict["data"], study_type)
            current_study_state.setdefault("data_quality_reports", {})[state_key] = quality_report

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
            except Exception as e:
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
        instructions_df = pd.DataFrame(
            [
                {
                    "Study Type": study_type.replace("_", " ").title(),
                    "Required Columns": ", ".join(columns),
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

    with st.sidebar.container(border=True):
        st.markdown(f"#### {config['expander_title']}")
        st.markdown(f"*{config['description']}*")

        # --- Fixed Experimental Conditions ---
        st.markdown(f"**{config['intro_text']}**")
        fixed_params = {}
        cols = st.columns(len(config["fixed_params"]))
        for i, (param_key, param_config) in enumerate(config["fixed_params"].items()):
            with cols[i % len(cols)]:
                fixed_params[param_key] = st.number_input(
                    param_config["label"],
                    value=param_config["value"],
                    min_value=param_config["min_value"],
                    step=param_config["step"],
                    help=param_config["help"],
                    key=f"{config['key_prefix']}{param_key}",
                    format="%.3f",
                )
        # Convert temperature to Kelvin if provided
        if "T_C" in fixed_params and fixed_params["T_C"] is not None:
            fixed_params["T_K"] = float(fixed_params["T_C"]) + 273.15

        st.markdown("---")

        # --- Data Input Section (File Upload Only) ---
        uploaded_data_key = config.get("uploaded_data_key")

        template_study_type = f"{study_type}_direct" if input_mode == "direct" else study_type

        uploaded_file = st.file_uploader(
            f"📁 Upload {config['study_name']} Data",
            type=["xlsx", "xls", "csv"],
            key=f"{config['key_prefix']}file_{input_mode}",
            help="Upload Excel or CSV file",
        )

        if uploaded_file:
            uploaded_df = _read_uploaded_file(uploaded_file, config["required_cols"], study_type)
            if uploaded_df is not None and not uploaded_df.empty:
                st.session_state[uploaded_data_key] = uploaded_df.copy()
                st.success(f"✅ {len(uploaded_df)} data points loaded.")

        template_buffer = _generate_excel_template(config["required_cols"], template_study_type)
        st.download_button(
            "📥 Download Template",
            template_buffer,
            f"{config['study_type']}_template.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{config['key_prefix']}template_{input_mode}",
        )

        df_for_analysis = validate_data_editor(
            st.session_state.get(uploaded_data_key), config["required_cols"]
        )

        if df_for_analysis is not None and not df_for_analysis.empty:
            new_input = {"data": df_for_analysis, "params": fixed_params, "input_mode": input_mode}
            _handle_input_change(config["state_key"], new_input, config["dependent_keys"])
        else:
            _handle_input_change(config["state_key"], None, config["dependent_keys"])


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

        with st.sidebar.container(border=True):
            st.markdown("#### 📊 Calibration Curve")
            st.markdown("*Establish Absorbance-Concentration relationship*")

            uploaded_file = st.file_uploader(
                "📁 Upload Calibration Data", type=["xlsx", "xls", "csv"], key="calib_file"
            )
            if uploaded_file:
                uploaded_calib = _read_uploaded_file(
                    uploaded_file, ["Concentration", "Absorbance"], "calibration"
                )
                if uploaded_calib is not None and not uploaded_calib.empty:
                    st.session_state["uploaded_calib_data"] = uploaded_calib.copy()
                    st.success(f"✅ {len(uploaded_calib)} points loaded.")

            template_buffer = _generate_excel_template(
                ["Concentration", "Absorbance"], "calibration"
            )
            st.download_button(
                "📥 Download Template",
                template_buffer,
                "calibration_template.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="calib_template",
            )

            # --- Final validation and state update ---
            validated_calib = validate_data_editor(
                st.session_state.get("uploaded_calib_data"), ["Concentration", "Absorbance"]
            )

            active_study_name = st.session_state.get("current_study")
            if active_study_name:
                if validated_calib is not None and not validated_calib.empty:
                    st.session_state.studies[active_study_name]["calib_df_input"] = validated_calib
                else:
                    st.session_state.studies[active_study_name]["calib_df_input"] = None
            elif validated_calib is not None:
                st.error("Please add a study first before entering data.")

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
            "uploaded_data_key": "uploaded_iso_data",
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
            "uploaded_data_key": "uploaded_kin_data",
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
            "uploaded_data_key": "uploaded_dos_data",
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
            "uploaded_data_key": "uploaded_ph_data",
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
            "uploaded_data_key": "uploaded_temp_data",
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
