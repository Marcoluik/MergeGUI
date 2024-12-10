import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
from typing import Tuple, Set
import json
from pathlib import Path
from io import BytesIO

def load_column_mappings() -> dict:
    """Load saved column mappings from file"""
    try:
        with open('column_mappings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_column_mappings(mappings: dict):
    """Save column mappings to file"""
    with open('column_mappings.json', 'w') as f:
        json.dump(mappings, f)

def analyze_dataframe(df: pd.DataFrame, filename: str) -> dict:
    """Analyze dataframe and return insights"""
    analysis = {
        'filename': filename,
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # In MB
        'column_types': df.dtypes.value_counts().to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
    }
    return analysis

def suggest_match_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[str, str]:
    """Suggest matching columns based on content similarity"""
    best_match = (None, None, 0)
    
    for col1 in df1.columns:
        for col2 in df2.columns:
            # Check if column names are similar
            name_similarity = len(set(col1.lower().split()) & set(col2.lower().split())) > 0
            
            # Check if data types match
            type_match = df1[col1].dtype == df2[col2].dtype
            
            # Check for unique values ratio similarity
            unique_ratio1 = len(df1[col1].unique()) / len(df1)
            unique_ratio2 = len(df2[col2].unique()) / len(df2)
            ratio_similarity = abs(unique_ratio1 - unique_ratio2) < 0.1
            
            score = sum([name_similarity, type_match, ratio_similarity])
            
            if score > best_match[2]:
                best_match = (col1, col2, score)
    
    return best_match[0], best_match[1]

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, key_column: str) -> Tuple[Set, Set, Set]:
    """
    Compare two dataframes and identify added, removed, and common rows based on a key column.
    
    Args:
        df1: First dataframe (considered as the "old" or "reference" dataframe)
        df2: Second dataframe (considered as the "new" dataframe)
        key_column: Column name to use for comparison
        
    Returns:
        Tuple of (added_keys, removed_keys, common_keys)
    """
    # Create sets of keys from both dataframes
    keys1 = set(df1[key_column])
    keys2 = set(df2[key_column])
    
    # Find added (in df2 but not in df1)
    added_keys = keys2 - keys1
    
    # Find removed (in df1 but not in df2)
    removed_keys = keys1 - keys2
    
    # Find common keys (in both)
    common_keys = keys1 & keys2
    
    return added_keys, removed_keys, common_keys

def main():
    st.set_page_config(page_title="Smart Prissammenligningsværktøj", layout="wide")
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stMetric { box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .stAlert { border-radius: 10px; }
        .stDataFrame { box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🔄 Smart Prissammenligningsværktøj")
    
    # Initialize session states
    if 'actions_log' not in st.session_state:
        st.session_state.actions_log = []
    if 'column_mappings' not in st.session_state:
        st.session_state.column_mappings = load_column_mappings()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Indstillinger")
        debug_mode = st.checkbox("Debug tilstand")
        show_visualizations = st.checkbox("Vis visualiseringer", value=True)
        merge_type = st.selectbox(
            "Fletningstype",
            options=['outer', 'inner', 'left', 'right'],
            help="Vælg hvordan filerne skal flettes sammen"
        )
        
        st.markdown("---")
        st.header("🎨 Visuelle indstillinger")
        color_added = st.color_picker("Farve for tilføjede rækker", "#90EE90")
        color_removed = st.color_picker("Farve for fjernede rækker", "#FFB6C6")

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.header("📁 Første Fil")
        file1 = st.file_uploader("Upload første Excel-fil", type=['xlsx', 'xls', 'csv'])
        
        if file1 is not None:
            # Load and analyze file
            if file1.name.endswith('csv'):
                df1 = pd.read_csv(file1)
            else:
                df1 = pd.read_excel(file1)
            
            analysis1 = analyze_dataframe(df1, file1.name)
            
            # Show file analysis
            with st.expander("📊 Filanalyse", expanded=True):
                st.metric("Antal rækker", analysis1['rows'])
                st.metric("Hukommelsesforbrug", f"{analysis1['memory_usage']:.2f} MB")
                
                if analysis1['missing_values'] > 0:
                    st.warning(f"⚠️ Filen indeholder {analysis1['missing_values']} manglende værdier")
                if analysis1['duplicates'] > 0:
                    st.warning(f"⚠️ Filen indeholder {analysis1['duplicates']} duplikerede rækker")

            # Column selection
            saved_mapping = st.session_state.column_mappings.get(file1.name, {})
            default_match = saved_mapping.get('match_column', df1.columns[0])
            
            match_column1 = st.selectbox(
                "Vælg matchende kolonne",
                options=df1.columns,
                index=df1.columns.get_loc(default_match) if default_match in df1.columns else 0,
                key="match1"
            )
            
            select_all_1 = st.checkbox("Vælg alle kolonner", key="select_all_1")
            if select_all_1:
                columns_to_keep1 = df1.columns.tolist()
                st.success("✅ Alle kolonner valgt")
            else:
                default_cols = saved_mapping.get('columns', [match_column1])
                columns_to_keep1 = st.multiselect(
                    "Vælg kolonner der skal beholdes",
                    options=df1.columns,
                    default=[col for col in default_cols if col in df1.columns],
                    key="cols1"
                )

    # Similar structure for second file...
    # (Code for col2 would be very similar to col1)
    with col2:
        st.header("📁 Anden Fil")
        file2 = st.file_uploader("Upload Anden Excel-fil", type=['xlsx', 'xls', 'csv'])
        
        if file2 is not None:
            # Load and analyze file
            if file2.name.endswith('csv'):
                df2 = pd.read_csv(file2)
            else:
                df2 = pd.read_excel(file2)
            
            analysis2 = analyze_dataframe(df2, file2.name)
            
            # Show file analysis
            with st.expander("📊 Filanalyse", expanded=True):
                st.metric("Antal rækker", analysis2['rows'])
                st.metric("Hukommelsesforbrug", f"{analysis2['memory_usage']:.2f} MB")
                
                if analysis2['missing_values'] > 0:
                    st.warning(f"⚠️ Filen indeholder {analysis2['missing_values']} manglende værdier")
                if analysis2['duplicates'] > 0:
                    st.warning(f"⚠️ Filen indeholder {analysis2['duplicates']} duplikerede rækker")

            # Column selection
            saved_mapping = st.session_state.column_mappings.get(file2.name, {})
            default_match = saved_mapping.get('match_column', df2.columns[0])
            
            match_column2 = st.selectbox(
                "Vælg matchende kolonne",
                options=df2.columns,
                index=df2.columns.get_loc(default_match) if default_match in df2.columns else 0,
                key="match2"
            )
            
            select_all_2 = st.checkbox("Vælg alle kolonner", key="select_all_2")
            if select_all_2:
                columns_to_keep2 = df2.columns.tolist()
                st.success("✅ Alle kolonner valgt")
            else:
                default_cols = saved_mapping.get('columns', [match_column2])
                columns_to_keep2 = st.multiselect(
                    "Vælg kolonner der skal beholdes",
                    options=df2.columns,
                    default=[col for col in default_cols if col in df2.columns],
                    key="cols2"
                )

    # Merge section
    if st.button("🔄 Flet Filer", use_container_width=True):
        if file1 is None or file2 is None:
            st.error("⚠️ Upload venligst begge filer")
            return

        try:
            # Save column mappings
            st.session_state.column_mappings[file1.name] = {
                'match_column': match_column1,
                'columns': columns_to_keep1
            }
            st.session_state.column_mappings[file2.name] = {
                'match_column': match_column2,
                'columns': columns_to_keep2
            }
            save_column_mappings(st.session_state.column_mappings)

            # Keep only selected columns
            df1 = df1[columns_to_keep1]
            df2 = df2[columns_to_keep2]

            # Get status of rows before merge
            added_keys, removed_keys, common_keys = compare_dataframes(
                df1, df2, match_column1
            )

            # Rename matching columns to prepare for merge
            df1 = df1.rename(columns={match_column1: 'merge_key'})
            df2 = df2.rename(columns={match_column2: 'merge_key'})

            # Merge dataframes
            merged_df = pd.merge(
                df1,
                df2,
                on='merge_key',
                how=merge_type,
                suffixes=('_fil1', '_fil2')
            )

            # Add status column with corrected logic
            merged_df['status'] = 'uændret'
            # If key is in removed_keys (missing from df2), it was removed
            merged_df.loc[merged_df['merge_key'].isin(removed_keys), 'status'] = 'fjernet'
            # If key is in added_keys (only in df2), it was added
            merged_df.loc[merged_df['merge_key'].isin(added_keys), 'status'] = 'tilføjet'

            # Create a clean version of merged_df without the status column for export
            export_df = merged_df.drop(columns=['status'])

            # Rename merge_key back to original name
            merged_df = merged_df.rename(columns={'merge_key': match_column1})
            export_df = export_df.rename(columns={'merge_key': match_column1})

            # Calculate status counts for visualization
            status_counts = merged_df['status'].value_counts()

            # Display results
            st.success(f"✅ Fletning gennemført! {len(merged_df)} rækker i alt")
            
            # Show summary metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Tilføjede rækker", status_counts.get('tilføjet', 0))
            with metric_col2:
                st.metric("Fjernede rækker", status_counts.get('fjernet', 0))
            with metric_col3:
                st.metric("Uændrede rækker", status_counts.get('uændret', 0))

            # Display merged dataframe with colored status
            st.dataframe(
                merged_df.style.apply(lambda x: [
                    f'background-color: {color_added}' if v == 'tilføjet'
                    else f'background-color: {color_removed}' if v == 'fjernet'
                    else '' for v in x
                ], subset=['status'])
            )

            # Add visualizations
            if show_visualizations:
                st.markdown("### 📊 Visualiseringer")
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Pie chart of merge status
                    fig1 = px.pie(
                        values=status_counts.values,
                        names=status_counts.index,
                        title="Fordeling af ændringer",
                        color_discrete_map={
                            'tilføjet': color_added,
                            'fjernet': color_removed,
                            'uændret': '#FFFFFF'
                        }
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                with viz_col2:
                    # Bar chart of column completeness
                    completeness = (1 - merged_df.isnull().mean()) * 100
                    fig2 = px.bar(
                        x=completeness.index,
                        y=completeness.values,
                        title="Kolonne komplethed (%)",
                        labels={'x': 'Kolonne', 'y': 'Komplethed (%)'}
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            # Export options
            st.markdown("### 💾 Eksport muligheder")
            export_format = st.selectbox(
                "Vælg eksport format",
                options=['Excel (.xlsx)', 'CSV (.csv)', 'JSON (.json)']
            )
            
            # Fix the file extension extraction
            format_extensions = {
                'Excel (.xlsx)': 'xlsx',
                'CSV (.csv)': 'csv',
                'JSON (.json)': 'json'
            }
            
            if st.download_button(
                "⬇️ Download flettet fil",
                data=convert_to_format(export_df, export_format),
                file_name=f"merged_file.{format_extensions[export_format]}",
                mime=get_mime_type(export_format)
            ):
                st.success("✅ Fil downloadet!")

        except Exception as e:
            st.error(f"❌ Der opstod en fejl: {str(e)}")
            if debug_mode:
                st.sidebar.error("🐛 Debug information:")
                st.sidebar.exception(e)

def convert_to_format(df: pd.DataFrame, format_str: str) -> bytes:
    """Convert dataframe to specified format"""
    if format_str == 'Excel (.xlsx)':
        output = BytesIO()
        df.to_excel(output, index=False)
        return output.getvalue()
    elif format_str == 'CSV (.csv)':
        return df.to_csv(index=False).encode('utf-8')
    else:  # JSON
        return df.to_json(orient='records').encode('utf-8')

def get_mime_type(format_str: str) -> str:
    """Get MIME type for file format"""
    mime_types = {
        'Excel (.xlsx)': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'CSV (.csv)': 'text/csv',
        'JSON (.json)': 'application/json'
    }
    return mime_types[format_str]

if __name__ == "__main__":
    main()

