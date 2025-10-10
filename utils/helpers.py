import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
import re
from datetime import datetime
import os

def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def validate_file_size(uploaded_file, max_size_mb: int = 100) -> bool:
    """Validate if uploaded file size is within limits"""
    try:
        # Get file size
        file_size = uploaded_file.size if hasattr(uploaded_file, 'size') else len(uploaded_file.read())
        
        # Reset file pointer if we read the file
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
        
        max_size_bytes = max_size_mb * 1024 * 1024
        return file_size <= max_size_bytes
    except:
        return False

def detect_file_encoding(file_content: bytes) -> str:
    """Detect file encoding for CSV files"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            file_content.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    
    return 'utf-8'  # Default fallback

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names to be SQL-friendly"""
    df_copy = df.copy()
    
    # Clean column names
    df_copy.columns = [
        re.sub(r'[^a-zA-Z0-9_]', '_', str(col)).lower()
        for col in df_copy.columns
    ]
    
    # Ensure columns don't start with numbers
    df_copy.columns = [
        f"col_{col}" if col[0].isdigit() else col
        for col in df_copy.columns
    ]
    
    # Handle duplicate column names
    cols = pd.Series(df_copy.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols == dup] = [f"{dup}_{i}" for i in range(sum(cols == dup))]
    df_copy.columns = cols
    
    return df_copy

def infer_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Intelligently infer and convert data types"""
    df_copy = df.copy()
    
    for col in df_copy.columns:
        # Skip if already numeric or datetime
        if df_copy[col].dtype in ['int64', 'float64', 'datetime64[ns]']:
            continue
        
        # Try to convert to numeric
        if df_copy[col].dtype == 'object':
            # Check if it looks like a number
            try:
                # Remove common non-numeric characters
                cleaned = df_copy[col].astype(str).str.replace(r'[\$,\%]', '', regex=True)
                numeric_vals = pd.to_numeric(cleaned, errors='coerce')
                
                # If most values can be converted, use numeric
                if numeric_vals.notna().sum() / len(df_copy) > 0.8:
                    df_copy[col] = numeric_vals
                    continue
            except:
                pass
            
            # Try to convert to datetime
            try:
                datetime_vals = pd.to_datetime(df_copy[col], errors='coerce')
                if datetime_vals.notna().sum() / len(df_copy) > 0.8:
                    df_copy[col] = datetime_vals
                    continue
            except:
                pass
            
            # Convert to category if low cardinality
            unique_ratio = df_copy[col].nunique() / len(df_copy)
            if unique_ratio < 0.1 and df_copy[col].nunique() > 1:
                df_copy[col] = df_copy[col].astype('category')
    
    return df_copy

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive data summary"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime']).columns
    
    summary = {
        'basic_info': {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'dtypes': df.dtypes.value_counts().to_dict()
        },
        'missing_data': {
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'columns_with_missing': df.columns[df.isnull().any()].tolist()
        },
        'duplicates': {
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        },
        'column_types': {
            'numeric': len(numeric_cols),
            'categorical': len(categorical_cols),
            'datetime': len(datetime_cols)
        }
    }
    
    # Add numeric column statistics
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    # Add categorical column info
    if len(categorical_cols) > 0:
        categorical_info = {}
        for col in categorical_cols:
            categorical_info[col] = {
                'unique_count': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'frequency': df[col].value_counts().iloc[0] if len(df) > 0 else 0
            }
        summary['categorical_summary'] = categorical_info
    
    return summary

def generate_sample_queries(columns: List[str], dtypes: Dict[str, Any]) -> List[str]:
    """Generate sample SQL queries based on data structure"""
    queries = [
        "SELECT * FROM data_table LIMIT 10",
        f"SELECT COUNT(*) FROM data_table",
    ]
    
    numeric_cols = [col for col, dtype in dtypes.items() if 'int' in str(dtype) or 'float' in str(dtype)]
    categorical_cols = [col for col, dtype in dtypes.items() if 'object' in str(dtype) or 'category' in str(dtype)]
    
    if numeric_cols:
        queries.extend([
            f"SELECT AVG({numeric_cols[0]}) FROM data_table",
            f"SELECT MIN({numeric_cols[0]}), MAX({numeric_cols[0]}) FROM data_table"
        ])
    
    if categorical_cols:
        queries.extend([
            f"SELECT {categorical_cols[0]}, COUNT(*) FROM data_table GROUP BY {categorical_cols[0]} ORDER BY COUNT(*) DESC",
            f"SELECT DISTINCT {categorical_cols[0]} FROM data_table"
        ])
    
    if len(numeric_cols) >= 2:
        queries.append(f"SELECT {numeric_cols[0]}, {numeric_cols[1]} FROM data_table WHERE {numeric_cols[0]} > (SELECT AVG({numeric_cols[0]}) FROM data_table)")
    
    return queries

def validate_sql_query(query: str, allowed_tables: List[str] = None) -> bool:
    """Basic SQL query validation for security"""
    if allowed_tables is None:
        allowed_tables = ['data_table']
    
    # Convert to lowercase for checking
    query_lower = query.lower().strip()
    
    # Block potentially dangerous operations
    dangerous_keywords = [
        'drop', 'delete', 'insert', 'update', 'alter', 'create', 
        'truncate', 'replace', 'grant', 'revoke', '--', ';'
    ]
    
    for keyword in dangerous_keywords:
        if keyword in query_lower:
            return False
    
    # Ensure query starts with SELECT
    if not query_lower.startswith('select'):
        return False
    
    # Basic table name validation
    for table in allowed_tables:
        if table.lower() in query_lower:
            return True
    
    return False

def export_data_to_formats(df: pd.DataFrame, formats: List[str] = None) -> Dict[str, bytes]:
    """Export DataFrame to multiple formats"""
    if formats is None:
        formats = ['csv', 'json', 'xlsx']
    
    exports = {}
    
    if 'csv' in formats:
        exports['csv'] = df.to_csv(index=False).encode('utf-8')
    
    if 'json' in formats:
        exports['json'] = df.to_json(orient='records', date_format='iso').encode('utf-8')
    
    if 'xlsx' in formats:
        # Use BytesIO for Excel export
        from io import BytesIO
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
        exports['xlsx'] = buffer.getvalue()
    
    return exports

def create_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a data dictionary describing the dataset"""
    
    dictionary_data = []
    
    for col in df.columns:
        col_info = {
            'Column Name': col,
            'Data Type': str(df[col].dtype),
            'Non-Null Count': df[col].count(),
            'Null Count': df[col].isnull().sum(),
            'Null Percentage': f"{(df[col].isnull().sum() / len(df)) * 100:.2f}%",
            'Unique Values': df[col].nunique(),
            'Unique Percentage': f"{(df[col].nunique() / len(df)) * 100:.2f}%"
        }
        
        # Add type-specific information
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                'Min Value': df[col].min(),
                'Max Value': df[col].max(),
                'Mean': f"{df[col].mean():.2f}" if not pd.isna(df[col].mean()) else 'N/A',
                'Sample Values': str(df[col].dropna().head(3).tolist())
            })
        elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
            col_info.update({
                'Max Length': df[col].astype(str).str.len().max() if len(df) > 0 else 0,
                'Min Length': df[col].astype(str).str.len().min() if len(df) > 0 else 0,
                'Sample Values': str(df[col].dropna().head(3).tolist())
            })
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_info.update({
                'Min Date': df[col].min(),
                'Max Date': df[col].max(),
                'Date Range (days)': (df[col].max() - df[col].min()).days if df[col].notna().any() else 0,
                'Sample Values': str(df[col].dropna().head(3).tolist())
            })
        
        dictionary_data.append(col_info)
    
    return pd.DataFrame(dictionary_data)

def log_activity(activity: str, details: Dict[str, Any] = None, log_file: str = "activity.log"):
    """Log user activities for debugging and analysis"""
    try:
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'activity': activity,
            'details': details or {}
        }
        
        # Simple file logging (in production, consider using proper logging framework)
        log_line = f"{timestamp} - {activity}"
        if details:
            log_line += f" - {details}"
        log_line += "\n"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_line)
            
    except Exception as e:
        # Don't let logging errors break the application
        pass

def generate_data_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate an overall data quality score"""
    
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    # Calculate component scores (0-100)
    completeness_score = ((total_cells - missing_cells) / total_cells) * 100
    uniqueness_score = ((len(df) - duplicate_rows) / len(df)) * 100 if len(df) > 0 else 100
    
    # Data type consistency score
    type_consistency_score = 100  # Start with perfect score
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if numeric data is stored as strings
            try:
                pd.to_numeric(df[col].dropna(), errors='raise')
                # If successful, penalize for poor type consistency
                type_consistency_score -= 10
            except:
                pass
    
    type_consistency_score = max(0, type_consistency_score)
    
    # Overall quality score (weighted average)
    overall_score = (
        completeness_score * 0.4 +  # 40% weight
        uniqueness_score * 0.3 +    # 30% weight  
        type_consistency_score * 0.3 # 30% weight
    )
    
    return {
        'overall_score': round(overall_score, 2),
        'completeness_score': round(completeness_score, 2),
        'uniqueness_score': round(uniqueness_score, 2),
        'type_consistency_score': round(type_consistency_score, 2),
        'grade': get_quality_grade(overall_score),
        'recommendations': get_quality_recommendations(completeness_score, uniqueness_score, type_consistency_score)
    }

def get_quality_grade(score: float) -> str:
    """Convert numeric quality score to letter grade"""
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

def get_quality_recommendations(completeness: float, uniqueness: float, consistency: float) -> List[str]:
    """Generate recommendations based on quality scores"""
    recommendations = []
    
    if completeness < 90:
        recommendations.append("Handle missing values to improve data completeness")
    
    if uniqueness < 95:
        recommendations.append("Remove duplicate rows to improve data uniqueness")
    
    if consistency < 90:
        recommendations.append("Fix data type inconsistencies for better analysis")
    
    if not recommendations:
        recommendations.append("Data quality is excellent! No major issues detected.")
    
    return recommendations
