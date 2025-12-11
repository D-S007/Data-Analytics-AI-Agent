import streamlit as st
import pandas as pd
import os
import tempfile
from modules.data_ingestion import DataIngestion
from modules.data_cleaning import DataCleaning
from modules.database_manager import DatabaseManager
from modules.ai_services import AIServices
from modules.visualization import Visualization
from modules.profiling import DataProfiler
from utils.helpers import format_bytes, validate_file_size

# Page configuration
st.set_page_config(
    page_title="CleanFlow : Version Controlled Big-Data Cleaner",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()
if 'ai_services' not in st.session_state:
    st.session_state.ai_services = AIServices()
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

def main():
    st.title("🤖 Data Analytics AI Agent")
    st.markdown("Upload your dataset, clean it automatically, and get AI-powered insights!")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("📋 Workflow Steps")
        steps = [
            "1️⃣ Data Ingestion",
            "2️⃣ Data Cleaning",
            "3️⃣ Data Storage",
            "4️⃣ Query & Analysis",
            "5️⃣ Visualization",
            "6️⃣ Profiling Report"
        ]
        
        for i, step in enumerate(steps, 1):
            if st.session_state.current_step >= i:
                st.success(step)
            else:
                st.write(step)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📥 Data Ingestion", 
        "🧹 Data Cleaning", 
        "💾 Storage & Queries", 
        "📊 Visualization", 
        "📈 Profiling Report",
        "⚙️ Settings"
    ])
    
    with tab1:
        handle_data_ingestion()
    
    with tab2:
        handle_data_cleaning()
    
    with tab3:
        handle_storage_and_queries()
    
    with tab4:
        handle_visualization()
    
    with tab5:
        handle_profiling()
    
    with tab6:
        handle_settings()

def handle_data_ingestion():
    st.header("📥 Data Ingestion")
    
    ingestion_method = st.selectbox(
        "Choose your data source:",
        ["Upload File", "API Connection", "Database Connection"]
    )
    
    data_ingestion = DataIngestion()
    
    if ingestion_method == "Upload File":
        st.subheader("Upload CSV or XLSX File")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx'],
            help="Maximum file size: 100MB"
        )
        
        if uploaded_file is not None:
            if validate_file_size(uploaded_file, max_size_mb=100):
                with st.spinner("Loading data..."):
                    try:
                        st.session_state.data = data_ingestion.load_file(uploaded_file)
                        st.session_state.current_step = max(st.session_state.current_step, 2)
                        st.success(f"✅ Data loaded successfully! Shape: {st.session_state.data.shape}")
                        
                        # Show preview
                        st.subheader("Data Preview")
                        st.dataframe(st.session_state.data.head(), use_container_width=True)
                        
                        # Show basic info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", st.session_state.data.shape[0])
                        with col2:
                            st.metric("Columns", st.session_state.data.shape[1])
                        with col3:
                            st.metric("Memory Usage", format_bytes(st.session_state.data.memory_usage(deep=True).sum()))
                            
                    except Exception as e:
                        st.error(f"❌ Error loading file: {str(e)}")
            else:
                st.error("❌ File size exceeds 100MB limit")
    
    elif ingestion_method == "API Connection":
        st.subheader("API Data Source")
        api_url = st.text_input("API Endpoint URL")
        api_key = st.text_input("API Key (if required)", type="password")
        headers_input = st.text_area("Custom Headers (JSON format)", placeholder='{"Authorization": "Bearer token"}')
        
        if st.button("Connect to API"):
            if api_url:
                with st.spinner("Connecting to API..."):
                    try:
                        st.session_state.data = data_ingestion.load_from_api(api_url, api_key, headers_input)
                        st.session_state.current_step = max(st.session_state.current_step, 2)
                        st.success("✅ Data loaded from API successfully!")
                        st.dataframe(st.session_state.data.head())
                    except Exception as e:
                        st.error(f"❌ API connection failed: {str(e)}")
            else:
                st.error("Please provide API endpoint URL")
    
    elif ingestion_method == "Database Connection":
        st.subheader("Database Connection")
        db_type = st.selectbox("Database Type", ["SQLite", "MySQL", "PostgreSQL"])
        
        if db_type == "SQLite":
            db_file = st.file_uploader("Upload SQLite file", type=['db', 'sqlite'])
            table_name = st.text_input("Table Name")
            
            if db_file and table_name:
                if st.button("Connect to Database"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
                        tmp_file.write(db_file.read())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        st.session_state.data = data_ingestion.load_from_database(
                            f"sqlite:///{tmp_file_path}", table_name
                        )
                        st.session_state.current_step = max(st.session_state.current_step, 2)
                        st.success("✅ Data loaded from database successfully!")
                        st.dataframe(st.session_state.data.head())
                    except Exception as e:
                        st.error(f"❌ Database connection failed: {str(e)}")
                    finally:
                        os.unlink(tmp_file_path)
        
        else:
            st.info("MySQL and PostgreSQL connections require connection strings")
            connection_string = st.text_input("Connection String", type="password")
            table_name = st.text_input("Table Name")
            
            if connection_string and table_name:
                if st.button("Connect to Database"):
                    try:
                        st.session_state.data = data_ingestion.load_from_database(connection_string, table_name)
                        st.session_state.current_step = max(st.session_state.current_step, 2)
                        st.success("✅ Data loaded from database successfully!")
                        st.dataframe(st.session_state.data.head())
                    except Exception as e:
                        st.error(f"❌ Database connection failed: {str(e)}")

def handle_data_cleaning():
    st.header("🧹 Data Cleaning")
    
    if st.session_state.data is None:
        st.warning("Please load data first in the Data Ingestion tab")
        return
    
    data_cleaner = DataCleaning()
    
    st.subheader("Data Quality Assessment")
    quality_info = data_cleaner.assess_data_quality(st.session_state.data)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Missing Values", quality_info['missing_count'])
    with col2:
        st.metric("Duplicate Rows", quality_info['duplicate_count'])
    with col3:
        st.metric("Data Types Issues", quality_info['type_issues'])
    
    # Cleaning options
    st.subheader("Cleaning Options")
    
    col1, col2 = st.columns(2)
    with col1:
        handle_missing = st.checkbox("Handle Missing Values", value=True)
        remove_duplicates = st.checkbox("Remove Duplicates", value=True)
        standardize_dates = st.checkbox("Standardize Date Formats", value=True)
    
    with col2:
        clean_whitespace = st.checkbox("Clean Whitespace", value=True)
        fix_data_types = st.checkbox("Fix Data Types", value=True)
        remove_outliers = st.checkbox("Remove Outliers", value=False)
    
    # Missing value strategy
    if handle_missing:
        missing_strategy = st.selectbox(
            "Missing Value Strategy",
            ["drop", "mean", "median", "mode", "forward_fill", "backward_fill"]
        )
    
    if st.button("🧹 Clean Data"):
        with st.spinner("Cleaning data..."):
            try:
                cleaning_options = {
                    'handle_missing': handle_missing,
                    'remove_duplicates': remove_duplicates,
                    'standardize_dates': standardize_dates,
                    'clean_whitespace': clean_whitespace,
                    'fix_data_types': fix_data_types,
                    'remove_outliers': remove_outliers,
                    'missing_strategy': missing_strategy if handle_missing else 'drop'
                }
                
                cleaned_data, cleaning_report = data_cleaner.clean_data(st.session_state.data, cleaning_options)
                st.session_state.cleaned_data = cleaned_data
                st.session_state.current_step = max(st.session_state.current_step, 3)
                
                st.success("✅ Data cleaned successfully!")
                
                # Show cleaning report
                st.subheader("Cleaning Report")
                for step, result in cleaning_report.items():
                    st.write(f"**{step}**: {result}")
                
                # Show before/after comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Before Cleaning")
                    st.write(f"Shape: {st.session_state.data.shape}")
                    st.dataframe(st.session_state.data.head())
                
                with col2:
                    st.subheader("After Cleaning")
                    st.write(f"Shape: {cleaned_data.shape}")
                    st.dataframe(cleaned_data.head())
                
            except Exception as e:
                st.error(f"❌ Cleaning failed: {str(e)}")
    
    # Download cleaned data
    if st.session_state.cleaned_data is not None:
        csv = st.session_state.cleaned_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Cleaned Data",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

def handle_storage_and_queries():
    st.header("💾 Storage & Queries")
    
    if st.session_state.cleaned_data is None:
        st.warning("Please clean your data first")
        return
    
    # Store data in database
    if st.button("💾 Store Data in Database"):
        with st.spinner("Storing data..."):
            try:
                table_name = st.session_state.db_manager.store_data(st.session_state.cleaned_data)
                st.session_state.current_step = max(st.session_state.current_step, 4)
                st.success(f"✅ Data stored in table: {table_name}")
            except Exception as e:
                st.error(f"❌ Storage failed: {str(e)}")
    
    # Natural language queries
    st.subheader("🗣️ Natural Language Queries")
    
    # Show AI status warning
    if not st.session_state.ai_services.use_ai:
        st.info("ℹ️ Using rule-based SQL generation. Add HF_API_KEY in Settings for AI-powered queries.")
    
    user_question = st.text_input("Ask a question about your data:")
    
    if user_question and st.button("🔍 Analyze"):
        ai_services = st.session_state.ai_services
        
        # Show error if token validation failed during initialization
        if ai_services.last_error:
            st.warning(f"⚠️ {ai_services.last_error} - Using rule-based fallback. Check Settings to validate your HF token.")
        
        with st.spinner("Converting question to SQL and analyzing..."):
            try:
                # Convert natural language to SQL
                sql_query = ai_services.natural_language_to_sql(
                    user_question, 
                    st.session_state.cleaned_data.columns.tolist(),
                    st.session_state.cleaned_data.dtypes.to_dict()
                )
                
                st.subheader("Generated SQL Query")
                st.code(sql_query, language="sql")
                
                if not ai_services.use_ai:
                    st.caption("⚠️ Generated using keyword matching (no AI). Configure HF_API_KEY for better results.")
                
                # Execute query
                result = st.session_state.db_manager.execute_query(sql_query)
                
                st.subheader("Query Results")
                st.dataframe(result, use_container_width=True)
                
                # Generate insights
                insights = ai_services.generate_insights(user_question, result)
                st.subheader("AI Insights")
                st.write(insights)
                
                if not ai_services.use_ai:
                    st.caption("⚠️ Basic statistical insights. Configure HF_API_KEY for AI-powered analysis.")
                
                # Show error if it was set during execution
                if ai_services.last_error:
                    st.warning(f"⚠️ {ai_services.last_error} - Results generated using rule-based fallback.")
                
            except Exception as e:
                st.error(f"❌ Query failed: {str(e)}")
    
    # Custom SQL queries
    st.subheader("📝 Custom SQL Queries")
    custom_sql = st.text_area("Enter your SQL query:")
    
    if custom_sql and st.button("▶️ Execute SQL"):
        try:
            result = st.session_state.db_manager.execute_query(custom_sql)
            st.dataframe(result, use_container_width=True)
        except Exception as e:
            st.error(f"❌ SQL execution failed: {str(e)}")

def handle_visualization():
    st.header("📊 Visualization")
    
    if st.session_state.cleaned_data is None:
        st.warning("Please clean your data first")
        return
    
    visualizer = Visualization()
    ai_services = st.session_state.ai_services
    
    # Show AI status for visualization suggestions
    if not ai_services.use_ai:
        st.info("ℹ️ Using rule-based visualization suggestions (works great without AI!)")
    
    # Visualization request
    viz_question = st.text_input("What would you like to visualize?")
    
    if viz_question and st.button("📈 Generate Visualization"):
        with st.spinner("Creating visualization..."):
            try:
                # Generate appropriate visualization
                chart_config = ai_services.suggest_visualization(
                    viz_question, 
                    st.session_state.cleaned_data.columns.tolist(),
                    st.session_state.cleaned_data.head().to_dict()
                )
                
                # Create the chart
                fig = visualizer.create_chart(st.session_state.cleaned_data, chart_config)
                st.plotly_chart(fig, use_container_width=True)
                
                # Generate explanation
                explanation = ai_services.explain_visualization(viz_question, chart_config)
                st.subheader("📝 Visualization Explanation")
                st.write(explanation)
                
            except Exception as e:
                st.error(f"❌ Visualization failed: {str(e)}")
    
    # Quick visualizations
    st.subheader("Quick Visualizations")
    
    numeric_cols = st.session_state.cleaned_data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = st.session_state.cleaned_data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if len(numeric_cols) >= 2:
            st.subheader("Correlation Heatmap")
            corr_fig = visualizer.create_correlation_heatmap(st.session_state.cleaned_data[numeric_cols])
            st.plotly_chart(corr_fig, use_container_width=True)
    
    with col2:
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column for distribution:", numeric_cols)
            if selected_col:
                dist_fig = visualizer.create_distribution_plot(st.session_state.cleaned_data, selected_col)
                st.plotly_chart(dist_fig, use_container_width=True)

def handle_profiling():
    st.header("📈 Profiling Report")
    
    if st.session_state.cleaned_data is None:
        st.warning("Please clean your data first")
        return
    
    profiler = DataProfiler()
    
    if st.button("📊 Generate Profiling Report"):
        with st.spinner("Generating comprehensive data profile..."):
            try:
                report_path = profiler.generate_profile_report(st.session_state.cleaned_data)
                st.session_state.current_step = max(st.session_state.current_step, 6)
                st.success("✅ Profiling report generated!")
                
                # Read and display HTML report
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_html = f.read()
                
                # Create download button
                st.download_button(
                    label="📥 Download HTML Report",
                    data=report_html,
                    file_name="data_profile_report.html",
                    mime="text/html"
                )
                
                # Display report in iframe
                st.subheader("Report Preview")
                st.html(report_html)

            except Exception as e:
                st.error(f"❌ Report generation failed: {str(e)}")

def handle_settings():
    st.header("⚙️ Settings")
    
    st.subheader("AI Model Configuration")
    st.info("🤗 Using Hugging Face free models (Mistral-7B-Instruct)")
    
    ai_services = st.session_state.ai_services
    
    # Validate token button
    if st.button("🔍 Test HF API Token"):
        with st.spinner("Validating token..."):
            validation_result = ai_services.validate_token()
            
            if validation_result["valid"]:
                st.success(f"✅ {validation_result['message']}")
                st.info(validation_result['details'])
                # Update status
                ai_services.use_ai = True
                ai_services.last_error = None
            else:
                st.error(f"❌ {validation_result['message']}")
                st.warning(validation_result['details'])
                # Update status
                ai_services.use_ai = False
                ai_services.last_error = validation_result['message']
    
    # Show actual current status (not just env var)
    if ai_services.use_ai and not ai_services.last_error:
        st.success("✅ AI features are currently ENABLED")
        st.write("✅ Natural language to SQL (AI-powered)")
        st.write("✅ AI-generated insights")
        st.write("ℹ️ Token validated and working")
    elif os.getenv("HF_API_KEY") and ai_services.last_error:
        st.error(f"❌ AI features DISABLED - {ai_services.last_error}")
        st.write("⚠️ HF_API_KEY is set but invalid/expired")
        st.write("⚠️ Using rule-based fallbacks")
        st.markdown("""
        **To fix:**
        1. Check your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
        2. Update your `HF_API_KEY` environment variable
        3. Restart the application
        4. Click 'Test HF API Token' to verify
        """)
    else:
        st.warning("HF_API_KEY not configured - Using rule-based fallback")
        st.write("⚠️ SQL generation uses keyword matching")
        st.write("⚠️ Insights use statistical summaries")
        st.write("✅ Visualization suggestions work great without AI!")
        st.markdown("""
        **To enable AI features (completely free):**
        1. Create a free account at [huggingface.co](https://huggingface.co/join)
        2. Get your API token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
        3. Add `HF_API_KEY` to your environment variables
        4. Restart the application
        5. Click 'Test HF API Token' to verify
        
        **Note:** Hugging Face tokens are completely free! Just create a free account.
        """)
    
    
    st.subheader("Database Information")
    if st.session_state.db_manager:
        db_info = st.session_state.db_manager.get_database_info()
        st.json(db_info)
    
    st.subheader("Data Cleaning Modules")
    cleaning_modules = [
        "Missing Value Handler",
        "Duplicate Remover", 
        "Date Standardizer",
        "Whitespace Cleaner",
        "Data Type Fixer",
        "Outlier Detector"
    ]
    
    for module in cleaning_modules:
        st.write(f"✅ {module}")
    
    if st.button("🗑️ Clear All Data"):
        st.session_state.data = None
        st.session_state.cleaned_data = None
        st.session_state.current_step = 1
        st.success("✅ All data cleared")
        st.rerun()

if __name__ == "__main__":
    main()
