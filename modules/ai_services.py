import os
import json
import pandas as pd
import requests
import time
from typing import Dict, Any, List, Optional

class AIServices:
    """AI-powered services for data analysis and insights using Hugging Face models"""
    
    def __init__(self):
        # Using Hugging Face's free Inference API
        # Free API token required - get one at https://huggingface.co/settings/tokens
        self.hf_api_key = os.getenv("HF_API_KEY", "")
        
        # Free models from Hugging Face
        self.sql_model = "mistralai/Mistral-7B-Instruct-v0.2"
        self.text_model = "mistralai/Mistral-7B-Instruct-v0.2"
        
        self.base_url = "https://api-inference.huggingface.co/models/"
        self.headers = {}
        if self.hf_api_key:
            self.headers["Authorization"] = f"Bearer {self.hf_api_key}"
        
        # Track if we're in fallback mode
        self.use_ai = bool(self.hf_api_key)
        self.last_error = None
    
    def validate_token(self) -> Dict[str, Any]:
        """Validate Hugging Face API token"""
        if not self.hf_api_key:
            return {
                "valid": False,
                "message": "No HF_API_KEY environment variable set",
                "details": "Create a free account at huggingface.co and get your token"
            }
        
        try:
            # Test with a simple prompt
            test_url = f"{self.base_url}{self.text_model}"
            test_payload = {
                "inputs": "Hello",
                "parameters": {"max_new_tokens": 10}
            }
            
            response = requests.post(test_url, headers=self.headers, json=test_payload, timeout=10)
            
            if response.status_code == 401 or response.status_code == 403:
                return {
                    "valid": False,
                    "message": "Invalid or expired HF_API_KEY",
                    "details": "Please check your token at huggingface.co/settings/tokens"
                }
            elif response.status_code == 503:
                return {
                    "valid": True,
                    "message": "Token is valid (model loading)",
                    "details": "Hugging Face models may take a moment to start"
                }
            elif response.status_code == 200:
                return {
                    "valid": True,
                    "message": "Token is valid and working",
                    "details": "AI features are fully enabled"
                }
            else:
                return {
                    "valid": False,
                    "message": f"Unexpected response: HTTP {response.status_code}",
                    "details": "Please try again or check Hugging Face status"
                }
                
        except Exception as e:
            return {
                "valid": False,
                "message": "Token validation failed",
                "details": str(e)
            }
    
    def _query_hf_model(self, model_name: str, prompt: str, max_tokens: int = 512, 
                       temperature: float = 0.3, max_retries: int = 3) -> str:
        """Query Hugging Face Inference API"""
        
        if not self.hf_api_key:
            raise Exception("HF_API_KEY not configured. Using rule-based fallback.")
        
        url = f"{self.base_url}{model_name}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=self.headers, json=payload, timeout=30)
                
                if response.status_code == 401 or response.status_code == 403:
                    self.use_ai = False
                    self.last_error = "Invalid or expired HF_API_KEY"
                    raise Exception("Hugging Face API token is missing or invalid. Using rule-based fallback.")
                
                if response.status_code == 503:
                    # Model is loading, wait and retry
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    else:
                        raise Exception("Model is currently loading. Using rule-based fallback.")
                
                response.raise_for_status()
                result = response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '').strip()
                elif isinstance(result, dict) and 'generated_text' in result:
                    return result['generated_text'].strip()
                else:
                    raise Exception(f"Unexpected response format. Using rule-based fallback.")
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise Exception(f"API request failed: {str(e)}. Using rule-based fallback.")
        
        raise Exception("Failed to get response from model after retries. Using rule-based fallback.")
    
    def natural_language_to_sql(self, question: str, columns: List[str], dtypes: Dict[str, Any]) -> str:
        """Convert natural language question to SQL query using Hugging Face models"""
        
        # Prepare column information
        column_info = []
        for col, dtype in dtypes.items():
            column_info.append(f"{col} ({dtype})")
        
        prompt = f"""[INST] You are a SQL expert. Convert the natural language question into a SQL query.

Available columns and their types:
{', '.join(column_info)}

Table name: data_table
Question: {question}

Return ONLY the SQL query without any explanation or markdown formatting. [/INST]

SQL Query:"""
        
        try:
            sql_query = self._query_hf_model(self.sql_model, prompt, max_tokens=256, temperature=0.1)
            
            # Clean up the response
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            # Remove any explanatory text after the query
            if '\n\n' in sql_query:
                sql_query = sql_query.split('\n\n')[0]
            
            # Ensure it starts with SELECT
            if not sql_query.upper().startswith('SELECT'):
                # Try to extract SELECT statement
                lines = sql_query.split('\n')
                for line in lines:
                    if line.strip().upper().startswith('SELECT'):
                        sql_query = line.strip()
                        break
            
            return sql_query
            
        except Exception as e:
            # Fallback to simple query generation based on keywords
            return self._generate_fallback_sql(question, columns, dtypes)
    
    def _generate_fallback_sql(self, question: str, columns: List[str], dtypes: Dict[str, Any]) -> str:
        """Generate a simple SQL query as fallback"""
        question_lower = question.lower()
        
        # Simple keyword-based SQL generation
        if 'average' in question_lower or 'avg' in question_lower or 'mean' in question_lower:
            numeric_cols = [col for col, dtype in dtypes.items() if 'int' in str(dtype) or 'float' in str(dtype)]
            if numeric_cols:
                return f"SELECT AVG({numeric_cols[0]}) as average FROM data_table"
        
        if 'count' in question_lower or 'how many' in question_lower:
            return "SELECT COUNT(*) as count FROM data_table"
        
        if 'max' in question_lower or 'maximum' in question_lower or 'highest' in question_lower:
            numeric_cols = [col for col, dtype in dtypes.items() if 'int' in str(dtype) or 'float' in str(dtype)]
            if numeric_cols:
                return f"SELECT MAX({numeric_cols[0]}) as maximum FROM data_table"
        
        if 'min' in question_lower or 'minimum' in question_lower or 'lowest' in question_lower:
            numeric_cols = [col for col, dtype in dtypes.items() if 'int' in str(dtype) or 'float' in str(dtype)]
            if numeric_cols:
                return f"SELECT MIN({numeric_cols[0]}) as minimum FROM data_table"
        
        if 'group by' in question_lower or 'by' in question_lower:
            categorical_cols = [col for col, dtype in dtypes.items() if 'object' in str(dtype)]
            numeric_cols = [col for col, dtype in dtypes.items() if 'int' in str(dtype) or 'float' in str(dtype)]
            if categorical_cols and numeric_cols:
                return f"SELECT {categorical_cols[0]}, COUNT(*) as count FROM data_table GROUP BY {categorical_cols[0]} ORDER BY count DESC"
        
        # Default: show all data with limit
        return "SELECT * FROM data_table LIMIT 100"
    
    def suggest_visualization(self, question: str, columns: List[str], sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest appropriate visualization based on question and data"""
        
        # Use rule-based logic for visualization suggestions (more reliable than LLM for this task)
        question_lower = question.lower()
        
        # Identify column types
        numeric_cols = []
        categorical_cols = []
        date_cols = []
        
        for col, value in sample_data.items():
            if isinstance(value, (int, float)):
                numeric_cols.append(col)
            elif isinstance(value, str):
                categorical_cols.append(col)
        
        # Time series / trend analysis
        if any(word in question_lower for word in ['trend', 'over time', 'time series', 'change', 'evolution']):
            if len(numeric_cols) >= 1:
                x_col = categorical_cols[0] if categorical_cols else columns[0]
                return {
                    "chart_type": "line",
                    "x_column": x_col,
                    "y_column": numeric_cols[0],
                    "color_column": None,
                    "title": f"{numeric_cols[0]} Trend",
                    "explanation": "Line chart shows trends and changes over time or categories"
                }
        
        # Distribution analysis
        if any(word in question_lower for word in ['distribution', 'spread', 'histogram', 'frequency']):
            if numeric_cols:
                return {
                    "chart_type": "histogram",
                    "x_column": numeric_cols[0],
                    "y_column": None,
                    "color_column": None,
                    "title": f"Distribution of {numeric_cols[0]}",
                    "explanation": "Histogram shows the distribution and frequency of values"
                }
        
        # Comparison / categorical analysis
        if any(word in question_lower for word in ['compare', 'comparison', 'versus', 'vs', 'by category', 'by']):
            if categorical_cols and numeric_cols:
                return {
                    "chart_type": "bar",
                    "x_column": categorical_cols[0],
                    "y_column": numeric_cols[0],
                    "color_column": None,
                    "title": f"{numeric_cols[0]} by {categorical_cols[0]}",
                    "explanation": "Bar chart effectively compares values across categories"
                }
        
        # Correlation / relationship
        if any(word in question_lower for word in ['correlation', 'relationship', 'scatter', 'relation']):
            if len(numeric_cols) >= 2:
                return {
                    "chart_type": "scatter",
                    "x_column": numeric_cols[0],
                    "y_column": numeric_cols[1],
                    "color_column": categorical_cols[0] if categorical_cols else None,
                    "title": f"{numeric_cols[1]} vs {numeric_cols[0]}",
                    "explanation": "Scatter plot reveals relationships and correlations between variables"
                }
        
        # Proportion / composition
        if any(word in question_lower for word in ['proportion', 'composition', 'pie', 'percentage', 'share']):
            if categorical_cols:
                return {
                    "chart_type": "pie",
                    "x_column": categorical_cols[0],
                    "y_column": numeric_cols[0] if numeric_cols else None,
                    "color_column": None,
                    "title": f"Composition by {categorical_cols[0]}",
                    "explanation": "Pie chart shows proportions and relative sizes of categories"
                }
        
        # Default intelligent selection based on data structure
        if len(numeric_cols) >= 2:
            return {
                "chart_type": "scatter",
                "x_column": numeric_cols[0],
                "y_column": numeric_cols[1],
                "color_column": None,
                "title": "Data Relationship Analysis",
                "explanation": "Scatter plot to explore relationships between numeric variables"
            }
        elif categorical_cols and numeric_cols:
            return {
                "chart_type": "bar",
                "x_column": categorical_cols[0],
                "y_column": numeric_cols[0],
                "color_column": None,
                "title": "Category Analysis",
                "explanation": "Bar chart for categorical comparison"
            }
        elif numeric_cols:
            return {
                "chart_type": "histogram",
                "x_column": numeric_cols[0],
                "y_column": None,
                "color_column": None,
                "title": "Data Distribution",
                "explanation": "Histogram showing data distribution"
            }
        else:
            return {
                "chart_type": "bar",
                "x_column": columns[0] if columns else "index",
                "y_column": "count",
                "color_column": None,
                "title": "Data Overview",
                "explanation": "Bar chart overview of data"
            }
    
    def generate_insights(self, question: str, data: pd.DataFrame) -> str:
        """Generate insights from query results using Hugging Face models"""
        
        # Prepare data summary
        data_summary = {
            "shape": data.shape,
            "columns": data.columns.tolist(),
        }
        
        # Get basic statistics for numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        stats_text = ""
        if len(numeric_cols) > 0 and not data.empty:
            stats = data[numeric_cols].describe()
            stats_text = f"\nNumeric Statistics:\n{stats.to_string()}"
        
        # Limit sample data to avoid token limits
        sample_text = ""
        if not data.empty:
            sample_text = f"\nSample Data (first 3 rows):\n{data.head(3).to_string()}"
        
        prompt = f"""[INST] Analyze the following data and provide insights.

Question: {question}

Data Shape: {data_summary['shape'][0]} rows, {data_summary['shape'][1]} columns
Columns: {', '.join(data_summary['columns'])}
{stats_text[:500]}
{sample_text[:500]}

Provide a concise analysis covering:
1. Key Findings
2. Notable Patterns
3. Recommendations

Keep the response brief and actionable. [/INST]

Analysis:"""
        
        try:
            insights = self._query_hf_model(self.text_model, prompt, max_tokens=512, temperature=0.5)
            return insights
            
        except Exception as e:
            # Fallback to rule-based insights
            return self._generate_fallback_insights(question, data)
    
    def _generate_fallback_insights(self, question: str, data: pd.DataFrame) -> str:
        """Generate basic insights without AI model"""
        insights = []
        
        insights.append(f"**Key Findings:**")
        insights.append(f"- Dataset contains {data.shape[0]} rows and {data.shape[1]} columns")
        
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:  # Top 3 numeric columns
                mean_val = data[col].mean()
                insights.append(f"- Average {col}: {mean_val:.2f}")
        
        insights.append(f"\n**Data Quality:**")
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            insights.append(f"- Found {missing_count} missing values")
        else:
            insights.append(f"- No missing values detected")
        
        insights.append(f"\n**Recommendations:**")
        insights.append(f"- Explore visualizations to identify patterns")
        insights.append(f"- Consider analyzing correlations between numeric variables")
        
        return '\n'.join(insights)
    
    def explain_visualization(self, question: str, chart_config: Dict[str, Any]) -> str:
        """Explain why a specific visualization was chosen"""
        
        chart_type = chart_config.get('chart_type', 'chart')
        x_col = chart_config.get('x_column', 'variables')
        y_col = chart_config.get('y_column', 'values')
        
        # Rule-based explanations
        explanations = {
            'line': f"This line chart visualizes {y_col} across {x_col}, making it easy to spot trends, patterns, and changes over time or categories.",
            'bar': f"This bar chart compares {y_col} values across different {x_col} categories, allowing for quick visual comparison of magnitudes.",
            'scatter': f"This scatter plot reveals the relationship between {x_col} and {y_col}, helping identify correlations, clusters, and outliers.",
            'histogram': f"This histogram shows the distribution of {x_col} values, revealing the frequency and spread of data points.",
            'pie': f"This pie chart illustrates the proportional composition of {x_col}, showing how the whole is divided into parts.",
            'heatmap': f"This heatmap displays correlations between variables, with colors indicating the strength of relationships.",
            'box': f"This box plot shows the distribution of {y_col} across {x_col} categories, highlighting median, quartiles, and outliers."
        }
        
        explanation = explanations.get(chart_type, chart_config.get('explanation', ''))
        
        return f"{explanation}\n\n**How to interpret:** Look for patterns, outliers, and trends that answer your question: '{question}'"
    
    def suggest_data_cleaning(self, data_quality_info: Dict[str, Any]) -> List[str]:
        """Suggest data cleaning steps based on quality assessment"""
        
        recommendations = []
        
        missing_count = data_quality_info.get('missing_count', 0)
        missing_pct = data_quality_info.get('missing_percentage', 0)
        
        if missing_count > 0:
            if missing_pct > 50:
                recommendations.append(f"⚠️ High missing data ({missing_pct:.1f}%) - Consider if this dataset is suitable or collect more data")
            elif missing_pct > 10:
                recommendations.append(f"Handle {missing_count} missing values using appropriate imputation method (mean/median/mode)")
            else:
                recommendations.append(f"Remove rows with missing values ({missing_count} cells affected)")
        
        duplicate_count = data_quality_info.get('duplicate_count', 0)
        if duplicate_count > 0:
            recommendations.append(f"Remove {duplicate_count} duplicate rows to ensure data quality")
        
        type_issues = data_quality_info.get('type_issues', 0)
        if type_issues > 0:
            recommendations.append(f"Fix data type inconsistencies in {type_issues} columns for accurate analysis")
        
        columns_with_missing = data_quality_info.get('columns_with_missing', [])
        if len(columns_with_missing) > 5:
            recommendations.append(f"Focus on cleaning key columns: {', '.join(columns_with_missing[:5])}")
        
        if not recommendations:
            recommendations.append("✅ Data quality is excellent! No major cleaning needed.")
        
        return recommendations
    
    def analyze_data_trends(self, data: pd.DataFrame, column: str) -> str:
        """Analyze trends in a specific column"""
        
        if column not in data.columns:
            return f"Column '{column}' not found in data"
        
        col_data = data[column].dropna()
        
        if col_data.empty:
            return f"No data available for column '{column}'"
        
        # Generate basic analysis
        analysis = [f"**Analysis of '{column}':**\n"]
        
        total_values = len(col_data)
        unique_values = col_data.nunique()
        
        analysis.append(f"- Total values: {total_values}")
        analysis.append(f"- Unique values: {unique_values}")
        analysis.append(f"- Uniqueness: {(unique_values/total_values*100):.1f}%")
        
        if pd.api.types.is_numeric_dtype(col_data):
            analysis.append(f"\n**Numeric Statistics:**")
            analysis.append(f"- Mean: {col_data.mean():.2f}")
            analysis.append(f"- Median: {col_data.median():.2f}")
            analysis.append(f"- Std Dev: {col_data.std():.2f}")
            analysis.append(f"- Range: {col_data.min():.2f} to {col_data.max():.2f}")
            
            # Identify trends
            if col_data.std() / col_data.mean() > 0.5:
                analysis.append(f"\n**Trend:** High variability detected - data is spread out")
            else:
                analysis.append(f"\n**Trend:** Low variability - data is relatively consistent")
                
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            analysis.append(f"\n**Date Range:**")
            analysis.append(f"- From: {col_data.min()}")
            analysis.append(f"- To: {col_data.max()}")
            analysis.append(f"- Span: {(col_data.max() - col_data.min()).days} days")
            
        else:
            analysis.append(f"\n**Top Values:**")
            top_values = col_data.value_counts().head(5)
            for val, count in top_values.items():
                analysis.append(f"- {val}: {count} ({count/total_values*100:.1f}%)")
        
        return '\n'.join(analysis)
