# utils.py
import streamlit as st
import pandas as pd

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False, sep=';').encode('utf-8')

# --- Generic Validation Function for Data Editor ---
def validate_data_editor(edited_df, required_cols):
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
                 st.sidebar.warning(f"Missing required columns: {', '.join(missing_cols)}. Please add them.", icon="⚠️")
                 return None

            for col in required_cols:
                 if col in temp_df.columns: 
                    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
                 else: 
                     st.sidebar.error(f"Required column '{col}' not found during validation.")
                     return None
            
            temp_df.dropna(subset=required_cols, inplace=True)

            if 'Masse_Adsorbant_g' in temp_df.columns:
                 if (temp_df['Masse_Adsorbant_g'] <= 0).any():
                     st.sidebar.warning("Column 'Masse_Adsorbant_g' contains non-positive values. These rows will be ignored.", icon="⚠️")
                     temp_df = temp_df[temp_df['Masse_Adsorbant_g'] > 0]
            if 'Volume_L' in temp_df.columns:
                if (temp_df['Volume_L'] <= 0).any():
                    st.sidebar.warning("Column 'Volume_L' contains non-positive values. These rows will be ignored.", icon="⚠️")
                    temp_df = temp_df[temp_df['Volume_L'] > 0]
            
            if not temp_df.empty:
                validated_df = temp_df
                st.sidebar.success(f"{len(validated_df)} valid points.")
            else:
                st.sidebar.warning("No valid points after numerical conversion and filtering. Check your inputs.")
        
        except Exception as e:
            st.sidebar.error(f"Validation error: {e}")
    else:
        st.sidebar.info("Enter at least one data point.")
        
    return validated_df