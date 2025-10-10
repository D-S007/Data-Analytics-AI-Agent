import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.preprocessing import StandardScaler
import re
from datetime import datetime

class DataCleaning:
    """Comprehensive data cleaning and preprocessing module"""
    
    def __init__(self):
        self.cleaning_methods = {
            'handle_missing': self._handle_missing_values,
            'remove_duplicates': self._remove_duplicates,
            'standardize_dates': self._standardize_dates,
            'clean_whitespace': self._clean_whitespace,
            'fix_data_types': self._fix_data_types,
            'remove_outliers': self._remove_outliers
        }
    
    def assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality and return quality metrics"""
        quality_info = {
            'missing_count': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'duplicate_count': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / df.shape[0]) * 100,
            'type_issues': 0,
            'columns_with_missing': df.columns[df.isnull().any()].tolist(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Check for potential data type issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric data is stored as string
                try:
                    pd.to_numeric(df[col].dropna().head(100))
                    quality_info['type_issues'] += 1
                except (ValueError, TypeError):
                    pass
        
        return quality_info
    
    def clean_data(self, df: pd.DataFrame, options: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Clean data based on specified options"""
        cleaned_df = df.copy()
        cleaning_report = {}
        
        for method_name, should_apply in options.items():
            if should_apply and method_name in self.cleaning_methods:
                try:
                    if method_name == 'handle_missing':
                        cleaned_df, report = self.cleaning_methods[method_name](
                            cleaned_df, options.get('missing_strategy', 'drop')
                        )
                    else:
                        cleaned_df, report = self.cleaning_methods[method_name](cleaned_df)
                    
                    cleaning_report[method_name] = report
                    
                except Exception as e:
                    cleaning_report[method_name] = f"Failed: {str(e)}"
        
        return cleaned_df, cleaning_report
    
    def _handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop') -> Tuple[pd.DataFrame, str]:
        """Handle missing values based on strategy"""
        initial_shape = df.shape
        
        if strategy == 'drop':
            df_cleaned = df.dropna()
            report = f"Dropped {initial_shape[0] - df_cleaned.shape[0]} rows with missing values"
            
        elif strategy == 'mean':
            df_cleaned = df.copy()
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
            report = f"Filled missing values with mean for {len(numeric_cols)} numeric columns"
            
        elif strategy == 'median':
            df_cleaned = df.copy()
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
            report = f"Filled missing values with median for {len(numeric_cols)} numeric columns"
            
        elif strategy == 'mode':
            df_cleaned = df.copy()
            for col in df_cleaned.columns:
                mode_value = df_cleaned[col].mode()
                if len(mode_value) > 0:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_value[0])
            report = "Filled missing values with mode for all columns"
            
        elif strategy == 'forward_fill':
            df_cleaned = df.fillna(method='ffill')
            report = "Applied forward fill for missing values"
            
        elif strategy == 'backward_fill':
            df_cleaned = df.fillna(method='bfill')
            report = "Applied backward fill for missing values"
            
        else:
            df_cleaned = df.copy()
            report = f"Unknown strategy '{strategy}', no action taken"
        
        return df_cleaned, report
    
    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Remove duplicate rows"""
        initial_count = len(df)
        df_cleaned = df.drop_duplicates()
        duplicates_removed = initial_count - len(df_cleaned)
        
        report = f"Removed {duplicates_removed} duplicate rows"
        return df_cleaned, report
    
    def _standardize_dates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Standardize date formats"""
        df_cleaned = df.copy()
        date_columns_fixed = 0
        
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                # Try to identify and convert date columns
                sample_values = df_cleaned[col].dropna().head(100)
                
                # Common date patterns
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                    r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                    r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                    r'\d{1,2}/\d{1,2}/\d{4}',  # M/D/YYYY
                ]
                
                is_date_column = False
                for pattern in date_patterns:
                    if sample_values.astype(str).str.match(pattern).sum() > len(sample_values) * 0.8:
                        is_date_column = True
                        break
                
                if is_date_column:
                    try:
                        df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                        date_columns_fixed += 1
                    except:
                        pass
        
        report = f"Standardized {date_columns_fixed} date columns"
        return df_cleaned, report
    
    def _clean_whitespace(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Clean whitespace from string columns"""
        df_cleaned = df.copy()
        columns_cleaned = 0
        
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                # Remove leading/trailing whitespace and multiple spaces
                df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
                df_cleaned[col] = df_cleaned[col].str.replace(r'\s+', ' ', regex=True)
                columns_cleaned += 1
        
        report = f"Cleaned whitespace in {columns_cleaned} text columns"
        return df_cleaned, report
    
    def _fix_data_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Attempt to fix data type issues"""
        df_cleaned = df.copy()
        types_fixed = 0
        
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    # Check if it looks like numbers
                    sample = df_cleaned[col].dropna().head(100)
                    numeric_sample = pd.to_numeric(sample, errors='coerce')
                    
                    if numeric_sample.notna().sum() > len(sample) * 0.8:
                        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                        types_fixed += 1
                        continue
                except:
                    pass
                
                # Try to convert to category if low cardinality
                unique_ratio = df_cleaned[col].nunique() / len(df_cleaned[col])
                if unique_ratio < 0.1 and df_cleaned[col].nunique() > 1:
                    df_cleaned[col] = df_cleaned[col].astype('category')
                    types_fixed += 1
        
        report = f"Fixed data types for {types_fixed} columns"
        return df_cleaned, report
    
    def _remove_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Remove outliers using IQR method"""
        df_cleaned = df.copy()
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        outliers_removed = 0
        
        for col in numeric_cols:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers before removal
            outliers_before = len(df_cleaned)
            
            # Remove outliers
            df_cleaned = df_cleaned[
                (df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)
            ]
            
            outliers_removed += outliers_before - len(df_cleaned)
        
        report = f"Removed {outliers_removed} outlier rows from {len(numeric_cols)} numeric columns"
        return df_cleaned, report
    
    def add_cleaning_method(self, name: str, method_function):
        """Add a custom cleaning method"""
        self.cleaning_methods[name] = method_function
    
    def get_cleaning_suggestions(self, df: pd.DataFrame) -> List[str]:
        """Get suggestions for cleaning operations based on data assessment"""
        suggestions = []
        quality_info = self.assess_data_quality(df)
        
        if quality_info['missing_count'] > 0:
            suggestions.append(f"Handle {quality_info['missing_count']} missing values")
        
        if quality_info['duplicate_count'] > 0:
            suggestions.append(f"Remove {quality_info['duplicate_count']} duplicate rows")
        
        if quality_info['type_issues'] > 0:
            suggestions.append(f"Fix data types for {quality_info['type_issues']} columns")
        
        # Check for whitespace issues
        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            suggestions.append(f"Clean whitespace in {len(text_columns)} text columns")
        
        return suggestions
