import pandas as pd
import requests
import json
import sqlalchemy as sa
from typing import Optional, Dict, Any
import io

class DataIngestion:
    """Handles data ingestion from various sources"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx']
        self.max_file_size_mb = 100
    
    def load_file(self, uploaded_file) -> pd.DataFrame:
        """Load data from uploaded CSV or XLSX file"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try different encodings for CSV files
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
                    
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            if df.empty:
                raise ValueError("The uploaded file is empty")
                
            return df
            
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    def load_from_api(self, api_url: str, api_key: Optional[str] = None, 
                     headers_json: Optional[str] = None) -> pd.DataFrame:
        """Load data from API endpoint"""
        try:
            headers = {}
            
            # Add API key if provided
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
            
            # Parse custom headers if provided
            if headers_json:
                try:
                    custom_headers = json.loads(headers_json)
                    headers.update(custom_headers)
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON format in custom headers")
            
            # Make API request
            response = requests.get(api_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Try to parse response as JSON first
            try:
                data = response.json()
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Handle nested JSON structures
                    if 'data' in data:
                        df = pd.DataFrame(data['data'])
                    elif 'results' in data:
                        df = pd.DataFrame(data['results'])
                    else:
                        df = pd.json_normalize(data)
                else:
                    raise ValueError("Unsupported JSON structure")
                    
            except json.JSONDecodeError:
                # Try to parse as CSV
                df = pd.read_csv(io.StringIO(response.text))
            
            if df.empty:
                raise ValueError("No data received from API")
                
            return df
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing API response: {str(e)}")
    
    def load_from_database(self, connection_string: str, table_name: str, 
                          query: Optional[str] = None) -> pd.DataFrame:
        """Load data from database"""
        try:
            engine = sa.create_engine(connection_string)
            
            if query:
                df = pd.read_sql_query(query, engine)
            else:
                df = pd.read_sql_table(table_name, engine)
            
            if df.empty:
                raise ValueError(f"No data found in table: {table_name}")
                
            return df
            
        except sa.exc.SQLAlchemyError as e:
            raise Exception(f"Database connection failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading from database: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate loaded data and return basic info"""
        validation_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        return validation_info
    
    def get_sample_data(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Get sample of the data for preview"""
        return df.head(n)
