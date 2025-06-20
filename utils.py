# utils.py
import streamlit as st
import pandas as pd
from translations import _t 

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False, sep=';').encode('utf-8')

# --- Fonction de Validation Générique pour Data Editor ---
def validate_data_editor(edited_df, required_cols, success_message_key="validate_n_valid_points_success", error_message_key="validate_error_message"):
    """
    Validates data from Streamlit's data_editor.
    Converts required columns to numeric, handles NaNs, and checks for positive mass/volume if present.
    """
    validated_df = None
    if edited_df is not None and not edited_df.empty:
        try:
            temp_df = edited_df.copy()
            
            # Check for presence of all required columns
            all_cols_present = all(col in temp_df.columns for col in required_cols)
            if not all_cols_present:
                 missing_cols = [col for col in required_cols if col not in temp_df.columns]
                 # Using _t directly as it's imported
                 st.sidebar.warning(_t("validate_missing_cols_warning", missing_cols=', '.join(missing_cols)), icon="⚠️")
                 return None

            # Convert required columns to numeric
            for col in required_cols:
                 if col in temp_df.columns: 
                    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
                 else: 
                     # This case should be caught by all_cols_present check, but as a safeguard:
                     st.sidebar.error(_t("validate_col_not_found_error", col=col))
                     return None
            
            # Drop rows with NaNs in any of the required columns (after numeric conversion)
            temp_df.dropna(subset=required_cols, inplace=True)

            # Specific checks for mass and volume if columns exist
            if 'Masse_Adsorbant_g' in temp_df.columns:
                 if (temp_df['Masse_Adsorbant_g'] <= 0).any():
                     st.sidebar.warning(_t("validate_mass_non_positive_warning"), icon="⚠️")
                     temp_df = temp_df[temp_df['Masse_Adsorbant_g'] > 0]
            
            # After all filtering, check if any valid data remains
            if not temp_df.empty:
                validated_df = temp_df
                st.sidebar.success(_t(success_message_key, count=len(validated_df)))
            else:
                st.sidebar.warning(_t("validate_no_valid_points_warning"))
        
        except Exception as e:
            st.sidebar.error(_t(error_message_key, error=e))
    else:
        # No data entered or DataFrame is empty initially
        st.sidebar.info(_t("validate_enter_data_info"))
        
    return validated_df