import streamlit as st
import pandas as pd
from utils.visualizations import MatplotlibVisualizer
import numpy as np
import os
from glob import glob

# Page configuration
st.set_page_config(
    page_title="Two-Child Policy Analysis Dashboard",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file):
    """Load and cache the survey data"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def calculate_proscore(df, statement_cols):
    """Calculate ProScore (mean of support statements)"""
    proscore_cols = []
    for col in statement_cols:
        if col in df.columns:
            # Convert Likert scale to numeric (1-5)
            likert_map = {
                'Strongly Disagree': 1, 'Disagree': 2, 'Neutral': 3,
                'Agree': 4, 'Strongly Agree': 5
            }
            df[f'{col}_numeric'] = df[col].map(likert_map)
            proscore_cols.append(f'{col}_numeric')
    
    if proscore_cols:
        df['ProScore'] = df[proscore_cols].mean(axis=1)
    return df

def calculate_netscore(df, col):
    """Calculate Net Score (% Positive - % Negative)"""
    likert_map = {
        'Strongly Disagree': -2, 'Disagree': -1, 'Neutral': 0,
        'Agree': 1, 'Strongly Agree': 2
    }
    df[f'{col}_numeric'] = df[col].map(likert_map)
    return df

def normalize_likert_values(df, columns):
    """Normalize Likert text to standard labels and support numeric/abbr inputs."""
    def standardize(val):
        if pd.isna(val):
            return None
        s = str(val).strip().lower()
        # numeric codes 1-5
        if s in {'1','1.0'}:
            return 'Strongly Disagree'
        if s in {'2','2.0'}:
            return 'Disagree'
        if s in {'3','3.0'}:
            return 'Neutral'
        if s in {'4','4.0'}:
            return 'Agree'
        if s in {'5','5.0'}:
            return 'Strongly Agree'
        # abbreviations
        if s in {'sd','stronglydisagree','strongly disagree'}:
            return 'Strongly Disagree'
        if s in {'d','disagree'}:
            return 'Disagree'
        if s in {'n','neutral'}:
            return 'Neutral'
        if s in {'a','agree'}:
            return 'Agree'
        if s in {'sa','stronglyagree','strongly agree'}:
            return 'Strongly Agree'
        # fallback: title-case first letter mapping
        return str(val).strip()
    for c in columns:
        if c in df.columns:
            df[c] = df[c].apply(standardize)
    return df

def infer_column_mapping(df):
    """Infer common column names from arbitrary survey headers.
    Returns mapping for: Age, Education, Income, CurrentChildren, A1, B1, C1, C4
    """
    lower_to_original = {c.lower(): c for c in df.columns}

    def find_col(*keywords):
        for lower_name, original in lower_to_original.items():
            if all(k in lower_name for k in keywords):
                return original
        return None

    mapping = {}
    mapping['Age'] = find_col('age')
    mapping['Education'] = find_col('education') or find_col('qualification') or find_col('degree')
    mapping['Income'] = find_col('income') or find_col('salary') or find_col('monthly', 'income')
    mapping['CurrentChildren'] = find_col('current', 'children') or find_col('number', 'children') or find_col('no', 'children') or find_col('children')

    # Likert statements (best-effort keyword heuristics)
    mapping['A1'] = find_col('right', 'violation') or find_col('human', 'right') or find_col('violation')
    mapping['B1'] = find_col('population', 'control') or find_col('effective') or find_col('policy', 'support')
    mapping['C1'] = find_col('aging') or find_col('ageing') or find_col('old', 'population')
    mapping['C4'] = find_col('coercion') or find_col('force') or find_col('fear')
    return mapping

def create_diverging_bar(df, column, title):
    """Backwards-compatible wrapper to use Matplotlib visual for diverging bar."""
    visualizer = MatplotlibVisualizer()
    return visualizer.create_diverging_stacked_bar(df, column, title)

def _show_fig(fig, caption: str = None):
    """Safely render a Matplotlib figure without breaking the app."""
    try:
        if caption:
            st.caption(caption)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to render chart: {e}")

def main():
    # Header
    st.markdown('<div class="main-header">👶 Two-Child Policy Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/family.png", width=100)
        st.title("📊 Dashboard Controls")
        
        uploaded_file = st.file_uploader(
            "Upload Survey Data",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your survey data in CSV or Excel format"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This dashboard analyzes public opinion on two-child policy 
        across different demographics and concerns.
        
        **Key Metrics:**
        - ProScore: Average support level
        - Net Score: Positive - Negative responses
        """)
    
    if uploaded_file is None:
        # Attempt to auto-load a default dataset from data/ if present
        default_files = []
        try:
            base_dir = os.path.dirname(__file__)
            default_files = (
                glob(os.path.join(base_dir, 'data', '*.csv')) +
                glob(os.path.join(base_dir, 'data', '*.xlsx')) +
                glob(os.path.join(base_dir, 'data', '*.xls'))
            )
        except Exception:
            default_files = []

        if default_files:
            default_path = sorted(default_files)[0]
            try:
                if default_path.lower().endswith('.csv'):
                    df = pd.read_csv(default_path)
                else:
                    df = pd.read_excel(default_path)
                uploaded_filename = os.path.basename(default_path)
                st.info(f"Loaded default dataset: {uploaded_filename}")
            except Exception as e:
                st.error(f"Failed to load default dataset: {e}")
                return
        else:
            st.warning("⬆️ Please upload your survey data file to begin analysis")
            st.markdown("""
            ### Expected Data Format
            Your data should include columns such as:
            - **Demographics**: Age, Education, Income, CurrentChildren
            - **Support Statements**: A1, A2, B1, B2, C1, C2, C3, C4
            - **Likert Responses**: Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree
            """)
            return
    else:
        # Load uploaded file
        df = load_data(uploaded_file)
        if df is None:
            return
        uploaded_filename = uploaded_file.name

    st.success(f"✅ Data loaded successfully! {len(df)} responses found from {uploaded_filename}.")

    # Choose Matplotlib visualizer for clearer trends
    visualizer = MatplotlibVisualizer()

    # Column mapping detection and sidebar overrides
    inferred = infer_column_mapping(df)
    with st.sidebar:
        st.markdown("### Column Mapping")
        cols_list = [None] + list(df.columns)
        def idx(col_name):
            return (cols_list.index(inferred[col_name]) if inferred.get(col_name) in cols_list else 0)
        age_col = st.selectbox("Age column", options=cols_list, index=idx('Age'))
        edu_col = st.selectbox("Education column", options=cols_list, index=idx('Education'))
        inc_col = st.selectbox("Income column", options=cols_list, index=idx('Income'))
        kids_col = st.selectbox("Current Children column", options=cols_list, index=idx('CurrentChildren'))
        a1_col = st.selectbox("Rights Violation Concern (A1)", options=cols_list, index=idx('A1'))
        b1_col = st.selectbox("Population Control Effectiveness (B1)", options=cols_list, index=idx('B1'))
        c1_col = st.selectbox("Aging Population Risk (C1)", options=cols_list, index=idx('C1'))
        c4_col = st.selectbox("Coercion Implementation Fear (C4)", options=cols_list, index=idx('C4'))
        st.markdown("---")
        st.markdown("### Chart Options")
        chart_style = st.selectbox("Demographic chart type", options=["Bar", "Line"], index=0)
        show_b1_pie = st.checkbox("Show B1 overview as Pie", value=False)
        bin_age = st.checkbox("Group Age into bins", value=True)

    # Build working dataframe with standardized names
    working_df = df.copy()
    rename_map = {}
    if age_col: rename_map[age_col] = 'Age'
    if edu_col: rename_map[edu_col] = 'Education'
    if inc_col: rename_map[inc_col] = 'Income'
    if kids_col: rename_map[kids_col] = 'CurrentChildren'
    if a1_col: rename_map[a1_col] = 'A1'
    if b1_col: rename_map[b1_col] = 'B1'
    if c1_col: rename_map[c1_col] = 'C1'
    if c4_col: rename_map[c4_col] = 'C4'
    if rename_map:
        working_df = working_df.rename(columns=rename_map)
    
    # Display basic stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Responses", len(df))
    with col2:
        if 'Age' in working_df.columns:
            if bin_age:
                # Show dominant age group when binning is enabled
                age_counts = working_df['Age'].value_counts()
                dominant_age = age_counts.index[0] if len(age_counts) > 0 else "N/A"
                st.metric("Dominant Age Group", dominant_age)
            else:
                st.metric("Avg Age", f"{pd.to_numeric(working_df['Age'], errors='coerce').mean():.1f}")
    with col3:
        if 'Education' in working_df.columns:
            st.metric("Education Levels", working_df['Education'].nunique())
    with col4:
        if 'CurrentChildren' in working_df.columns:
            st.metric("Avg Children", f"{pd.to_numeric(working_df['CurrentChildren'], errors='coerce').mean():.2f}")
    
    st.markdown("---")
    
    # Normalize Likert responses before scoring
    statement_cols = [col for col in working_df.columns if col in ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'C3', 'C4']]
    working_df = normalize_likert_values(working_df, statement_cols)
    working_df = calculate_proscore(working_df, statement_cols)

    # Optional: bin ages for clearer trends
    if 'Age' in working_df.columns and bin_age:
        age_numeric = pd.to_numeric(working_df['Age'], errors='coerce')
        # Define bins
        bins = [0, 18, 25, 35, 45, 60, np.inf]
        labels = ['<=18', '19-25', '26-35', '36-45', '46-60', '60+']
        working_df['Age'] = pd.cut(
            age_numeric,
            bins=bins,
            labels=labels,
            include_lowest=True,
            ordered=True
        )

    # Additional key metrics row
    if 'ProScore' in working_df.columns:
        colp1, colp2, colp3 = st.columns(3)
        with colp1:
            st.metric("Overall ProScore (1-5)", f"{working_df['ProScore'].mean():.2f}")
        with colp2:
            if 'B1' in working_df.columns:
                likert_map_tmp = {'Strongly Disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly Agree': 5}
                agree_series = working_df['B1'].map(likert_map_tmp)
                agree_pct = (agree_series.isin([4, 5]).mean() * 100) if agree_series.notna().any() else 0.0
                st.metric("% Agree/Strongly Agree (B1)", f"{agree_pct:.1f}%")
        with colp3:
            st.metric("Columns Detected", len(df.columns))
    
    # Tabs for different analysis sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Demographic Analysis", 
        "📊 Opinion Distributions", 
        "🔍 Cross Analysis",
        "📋 Raw Data"
    ])
    
    with tab1:
        st.markdown('<div class="sub-header">Demographic Analysis</div>', unsafe_allow_html=True)
        
        # Age group table removed per user request
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 2. Education Level vs. Policy Support
            if 'Education' in working_df.columns and 'ProScore' in working_df.columns:
                st.subheader("2️⃣ Education Level vs. Support")
                if chart_style == "Bar":
                    fig2 = visualizer.create_demographic_bar_chart(
                        working_df, 'Education', 'ProScore',
                        title="Policy Support by Education Level",
                        y_label='Average Support Score'
                    )
                else:
                    fig2 = visualizer.create_line_chart(
                        working_df, 'Education', 'ProScore',
                        title="Policy Support by Education Level",
                        y_label='Average Support Score',
                        add_markers=True,
                        add_trendline=False
                    )
                _show_fig(fig2)
        
        with col2:
            # 3. Income Level vs. Policy Acceptance
            if 'Income' in working_df.columns and 'ProScore' in working_df.columns:
                st.subheader("3️⃣ Income Level vs. Acceptance")
                if chart_style == "Bar":
                    fig3 = visualizer.create_demographic_bar_chart(
                        working_df, 'Income', 'ProScore',
                        title="Policy Acceptance by Income Level",
                        y_label='Average Support Score'
                    )
                else:
                    fig3 = visualizer.create_line_chart(
                        working_df, 'Income', 'ProScore',
                        title="Policy Acceptance by Income Level",
                        y_label='Average Support Score',
                        add_markers=True,
                        add_trendline=True
                    )
                _show_fig(fig3)
        
        # 4. Family Size vs. Net Opinion
        if 'CurrentChildren' in working_df.columns:
            st.subheader("4️⃣ Family Size vs. Net Opinion")
            # Calculate net score for a key statement
            if 'B1' in working_df.columns:
                # Show stacked share of Negative/Neutral/Positive to highlight red parts
                fig4 = visualizer.create_pos_neg_stacked_share(
                    working_df, group_col='CurrentChildren', likert_col='B1',
                    title="Net Opinion by Current Number of Children (Share)",
                    figsize=(8, 4)
                )
                _show_fig(fig4)
    
    with tab2:
        st.markdown('<div class="sub-header">Opinion Distributions</div>', unsafe_allow_html=True)
        st.markdown("*Likert scale responses across all respondents*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 5. Rights Violation Concern (A1)
            if 'A1' in working_df.columns:
                st.subheader("5️⃣ Rights Violation Concern (A1)")
                fig5 = visualizer.create_diverging_stacked_bar(working_df, 'A1', "Distribution: Rights Violation Concern")
                _show_fig(fig5)
                if show_b1_pie:
                    st.caption("Overall distribution of B1 (Pie)")
                    if 'B1' in working_df.columns:
                        fig5b = visualizer.create_pie_chart(working_df, 'B1', 'B1 Distribution (Pie)')
                        _show_fig(fig5b)
            
            # 6. Population Control Effectiveness (B1)
            if 'B1' in working_df.columns:
                st.subheader("6️⃣ Population Control Effectiveness (B1)")
                fig6 = visualizer.create_diverging_stacked_bar(working_df, 'B1', "Distribution: Population Control Effectiveness")
                _show_fig(fig6)
        
        with col2:
            # 7. Aging Population Risk (C1)
            if 'C1' in working_df.columns:
                st.subheader("7️⃣ Aging Population Risk (C1)")
                fig7 = visualizer.create_diverging_stacked_bar(working_df, 'C1', "Distribution: Aging Population Risk")
                _show_fig(fig7)
            
            # 8. Coercion Implementation Fear (C4)
            if 'C4' in working_df.columns:
                st.subheader("8️⃣ Coercion Implementation Fear (C4)")
                fig8 = visualizer.create_diverging_stacked_bar(working_df, 'C4', "Distribution: Coercion Implementation Fear")
                _show_fig(fig8)
    
    with tab3:
        st.markdown('<div class="sub-header">Cross-Dimensional Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 9. Age vs. Rights Violation Concern
            if 'Age' in working_df.columns and 'A1' in working_df.columns:
                st.subheader("9️⃣ Age vs. Rights Violation Concern")
                likert_map = {
                    'Strongly Disagree': 1, 'Disagree': 2, 'Neutral': 3,
                    'Agree': 4, 'Strongly Agree': 5
                }
                working_df['A1_numeric'] = working_df['A1'].map(likert_map)
                age_rights = working_df.groupby('Age')['A1_numeric'].mean().reset_index()
                
                fig9 = visualizer.create_demographic_bar_chart(
                    working_df, 'Age', 'A1_numeric',
                    title="Rights Violation Concern by Age",
                    y_label='Mean Score (A1)'
                )
                _show_fig(fig9)
        
        with col2:
            # 10. Current Children vs. Coercion Fear
            if 'CurrentChildren' in working_df.columns and 'C4' in working_df.columns:
                st.subheader("🔟 Children vs. Coercion Fear")
                likert_map = {
                    'Strongly Disagree': 1, 'Disagree': 2, 'Neutral': 3,
                    'Agree': 4, 'Strongly Agree': 5
                }
                working_df['C4_numeric'] = working_df['C4'].map(likert_map)
                children_coercion = working_df.groupby('CurrentChildren')['C4_numeric'].mean().reset_index()
                
                fig10 = visualizer.create_demographic_bar_chart(
                    working_df, 'CurrentChildren', 'C4_numeric',
                    title="Coercion Fear by Number of Children",
                    x_label='Current Children',
                    y_label='Mean Score (C4)'
                )
                _show_fig(fig10)
    
    with tab4:
        st.markdown('<div class="sub-header">Raw Data View</div>', unsafe_allow_html=True)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'Age' in working_df.columns:
                age_options = [x for x in working_df['Age'].dropna().unique().tolist() if pd.notna(x)]
                # Preserve categorical order if present
                if pd.api.types.is_categorical_dtype(working_df['Age']):
                    cats = [c for c in working_df['Age'].cat.categories if c in age_options]
                    age_options = cats
                age_filter = st.multiselect("Filter by Age", options=age_options)
        with col2:
            if 'Education' in working_df.columns:
                edu_filter = st.multiselect("Filter by Education", options=working_df['Education'].unique())
        with col3:
            if 'Income' in working_df.columns:
                income_filter = st.multiselect("Filter by Income", options=working_df['Income'].unique())
        
        # Apply filters
        filtered_df = working_df.copy()
        if age_filter:
            filtered_df = filtered_df[filtered_df['Age'].isin(age_filter)]
        if edu_filter:
            filtered_df = filtered_df[filtered_df['Education'].isin(edu_filter)]
        if income_filter:
            filtered_df = filtered_df[filtered_df['Income'].isin(income_filter)]
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="filtered_survey_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()