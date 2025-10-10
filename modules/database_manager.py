import sqlite3
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Text
from typing import Dict, Any, Optional, List
import tempfile
import os
from datetime import datetime

class DatabaseManager:
    """Manages database operations for the AI agent"""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Create temporary database
            self.db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
            self.db_path = self.db_file.name
            self.db_file.close()
        else:
            self.db_path = db_path
        
        self.connection_string = f"sqlite:///{self.db_path}"
        self.engine = create_engine(self.connection_string)
        self.metadata = MetaData()
        self.tables = {}
    
    def store_data(self, df: pd.DataFrame, table_name: Optional[str] = None) -> str:
        """Store DataFrame in database with automatic schema detection"""
        if table_name is None:
            table_name = f"data_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Infer SQLAlchemy data types from pandas dtypes
            dtype_mapping = self._infer_sql_types(df)
            
            # Store data in database
            df.to_sql(
                table_name, 
                self.engine, 
                if_exists='replace', 
                index=False,
                dtype=dtype_mapping
            )
            
            self.tables[table_name] = {
                'created_at': datetime.now(),
                'shape': df.shape,
                'columns': df.columns.tolist()
            }
            
            return table_name
            
        except Exception as e:
            raise Exception(f"Failed to store data: {str(e)}")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            result_df = pd.read_sql_query(query, self.engine)
            return result_df
        except Exception as e:
            raise Exception(f"Query execution failed: {str(e)}")
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table"""
        try:
            # Get table schema
            inspector = sa.inspect(self.engine)
            columns = inspector.get_columns(table_name)
            
            # Get row count
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            count_result = pd.read_sql_query(count_query, self.engine)
            row_count = count_result.iloc[0]['count']
            
            table_info = {
                'table_name': table_name,
                'columns': [
                    {
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col['nullable']
                    } for col in columns
                ],
                'row_count': row_count,
                'creation_info': self.tables.get(table_name, {})
            }
            
            return table_info
            
        except Exception as e:
            raise Exception(f"Failed to get table info: {str(e)}")
    
    def list_tables(self) -> List[str]:
        """List all tables in the database"""
        try:
            inspector = sa.inspect(self.engine)
            return inspector.get_table_names()
        except Exception as e:
            raise Exception(f"Failed to list tables: {str(e)}")
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database information"""
        try:
            tables = self.list_tables()
            db_info = {
                'database_path': self.db_path,
                'connection_string': self.connection_string.replace(self.db_path, '[DB_PATH]'),
                'total_tables': len(tables),
                'tables': []
            }
            
            for table in tables:
                try:
                    table_info = self.get_table_info(table)
                    db_info['tables'].append(table_info)
                except:
                    db_info['tables'].append({'table_name': table, 'error': 'Failed to get info'})
            
            return db_info
            
        except Exception as e:
            return {'error': str(e)}
    
    def _infer_sql_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Infer appropriate SQL types from pandas DataFrame"""
        dtype_mapping = {}
        
        for column in df.columns:
            pandas_dtype = df[column].dtype
            
            if pandas_dtype == 'object':
                # Check if it's actually text or could be parsed as something else
                max_length = df[column].astype(str).str.len().max()
                if pd.isna(max_length) or max_length < 255:
                    dtype_mapping[column] = String(255)
                else:
                    dtype_mapping[column] = Text
                    
            elif pandas_dtype in ['int64', 'int32', 'int16', 'int8']:
                dtype_mapping[column] = Integer
                
            elif pandas_dtype in ['float64', 'float32']:
                dtype_mapping[column] = Float
                
            elif pandas_dtype in ['datetime64[ns]', 'datetime64[ns, UTC]']:
                dtype_mapping[column] = DateTime
                
            elif pandas_dtype == 'bool':
                dtype_mapping[column] = Integer  # Store as 0/1
                
            elif pandas_dtype == 'category':
                dtype_mapping[column] = String(255)
                
            else:
                # Default to Text for unknown types
                dtype_mapping[column] = Text
        
        return dtype_mapping
    
    def create_index(self, table_name: str, column_name: str, index_name: Optional[str] = None):
        """Create an index on a column"""
        if index_name is None:
            index_name = f"idx_{table_name}_{column_name}"
        
        try:
            with self.engine.connect() as conn:
                conn.execute(sa.text(f"CREATE INDEX {index_name} ON {table_name}({column_name})"))
                conn.commit()
        except Exception as e:
            raise Exception(f"Failed to create index: {str(e)}")
    
    def get_column_stats(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Get statistics for a specific column"""
        try:
            # Basic stats query
            stats_query = f"""
            SELECT 
                COUNT(*) as total_count,
                COUNT({column_name}) as non_null_count,
                COUNT(*) - COUNT({column_name}) as null_count
            FROM {table_name}
            """
            
            stats_df = pd.read_sql_query(stats_query, self.engine)
            stats = stats_df.iloc[0].to_dict()
            
            # Try to get numeric stats if the column is numeric
            try:
                numeric_stats_query = f"""
                SELECT 
                    MIN({column_name}) as min_value,
                    MAX({column_name}) as max_value,
                    AVG({column_name}) as avg_value
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
                """
                
                numeric_df = pd.read_sql_query(numeric_stats_query, self.engine)
                stats.update(numeric_df.iloc[0].to_dict())
                
            except:
                # Column is not numeric
                pass
            
            # Get unique values count
            unique_query = f"SELECT COUNT(DISTINCT {column_name}) as unique_count FROM {table_name}"
            unique_df = pd.read_sql_query(unique_query, self.engine)
            stats['unique_count'] = unique_df.iloc[0]['unique_count']
            
            return stats
            
        except Exception as e:
            raise Exception(f"Failed to get column stats: {str(e)}")
    
    def backup_table(self, table_name: str, backup_path: str):
        """Backup a table to a file"""
        try:
            df = pd.read_sql_table(table_name, self.engine)
            df.to_csv(backup_path, index=False)
        except Exception as e:
            raise Exception(f"Backup failed: {str(e)}")
    
    def close(self):
        """Clean up database connection"""
        try:
            self.engine.dispose()
            if hasattr(self, 'db_file'):
                os.unlink(self.db_path)
        except:
            pass
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close()
