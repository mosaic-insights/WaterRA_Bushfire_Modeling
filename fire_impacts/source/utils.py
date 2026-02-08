"""
Utility functions for Source model configuration and data processing.
"""

import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def validate_csv_files(*csv_paths: str) -> None:
    """
    Validate that CSV files exist and are readable.
    
    Args:
        *csv_paths: Variable number of CSV file paths to validate
        
    Raises:
        FileNotFoundError: If any CSV file doesn't exist
        pd.errors.EmptyDataError: If any CSV file is empty
    """
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path, nrows=1)  # Read just first row to validate
            if df.empty:
                raise pd.errors.EmptyDataError(f"CSV file is empty: {csv_path}")
            logger.debug(f"Validated CSV file: {csv_path}")
        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty: {csv_path}")
            raise


def check_dataframe_alignment(df1: pd.DataFrame, df2: pd.DataFrame, 
                            name1: str = "df1", name2: str = "df2") -> bool:
    """
    Check if two dataframes have compatible indices and columns.
    
    Args:
        df1: First dataframe
        df2: Second dataframe  
        name1: Name for first dataframe in log messages
        name2: Name for second dataframe in log messages
        
    Returns:
        True if dataframes are compatible
    """
    # Check index alignment
    if not df1.index.equals(df2.index):
        logger.warning(f"Index mismatch between {name1} and {name2}")
        logger.debug(f"{name1} index range: {df1.index.min()} to {df1.index.max()}")
        logger.debug(f"{name2} index range: {df2.index.min()} to {df2.index.max()}")
        return False
    
    # Check for overlapping columns
    common_cols = set(df1.columns) & set(df2.columns)
    if common_cols:
        logger.info(f"Common columns between {name1} and {name2}: {common_cols}")
    
    logger.debug(f"Dataframe alignment check passed for {name1} and {name2}")
    return True


def summarize_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> Dict:
    """
    Generate summary statistics for a dataframe.
    
    Args:
        df: Dataframe to summarize
        name: Name for the dataframe in summary
        
    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        'name': name,
        'shape': df.shape,
        'columns': list(df.columns),
        'index_range': (df.index.min(), df.index.max()),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'null_counts': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict()
    }
    
    # Add numeric summaries for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    logger.debug(f"Generated summary for {name}: {df.shape}")
    return summary


def verify_veneer_connection(v) -> Dict:
    """
    Verify and get basic information about a Veneer connection.
    
    Args:
        v: Veneer connection object
        
    Returns:
        Dictionary with connection information
    """
    try:
        info = {
            'scenario_info': v.scenario_info(),
            'constituents': v.model.get_constituents(),
            'input_sets': len(v.input_sets()) if hasattr(v, 'input_sets') else 'Unknown'
        }
        logger.info("Veneer connection verified successfully")
        return info
    except Exception as e:
        logger.error(f"Failed to verify Veneer connection: {e}")
        raise


def get_model_configuration_summary(v) -> Dict:
    """
    Get a summary of the current Source model configuration.
    
    Args:
        v: Veneer connection object
        
    Returns:
        Dictionary with model configuration details
    """
    try:
        config = {}
        
        # Get generation models
        try:
            gen_models = v.model.catchment.generation.model_table(constituents='TSS')
            config['generation_models'] = {
                'unique_models': gen_models.model.unique().tolist() if hasattr(gen_models, 'model') else [],
                'count': len(gen_models) if gen_models is not None else 0
            }
        except Exception as e:
            config['generation_models'] = {'error': str(e)}
        
        # Get runoff models
        try:
            rr_names = list(v.model.catchment.runoff.enumerate_names())
            config['runoff_models'] = {
                'functional_units': list({nm[1] for nm in rr_names if nm[1] != 'Water'}),
                'catchment_count': len([nm for nm in rr_names if nm[1] != 'Water'])
            }
        except Exception as e:
            config['runoff_models'] = {'error': str(e)}
        
        # Get current input set
        try:
            config['current_input_set'] = v.model.simulation.current_input_set()
        except Exception as e:
            config['current_input_set'] = {'error': str(e)}
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to get model configuration: {e}")
        return {'error': str(e)}


def format_time_series_data(df: pd.DataFrame, 
                          time_column: Optional[str] = None,
                          value_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Format time series data for use with Veneer.
    
    Args:
        df: Input dataframe
        time_column: Name of time column (if not index)
        value_columns: List of value columns to keep (if None, keep all)
        
    Returns:
        Formatted dataframe ready for Veneer
    """
    formatted_df = df.copy()
    
    # Set time column as index if specified
    if time_column and time_column in formatted_df.columns:
        formatted_df[time_column] = pd.to_datetime(formatted_df[time_column])
        formatted_df.set_index(time_column, inplace=True)
    
    # Ensure index is datetime
    if not isinstance(formatted_df.index, pd.DatetimeIndex):
        formatted_df.index = pd.to_datetime(formatted_df.index)
    
    # Filter to specified columns if provided
    if value_columns:
        missing_cols = set(value_columns) - set(formatted_df.columns)
        if missing_cols:
            logger.warning(f"Missing columns in dataframe: {missing_cols}")
        available_cols = [col for col in value_columns if col in formatted_df.columns]
        formatted_df = formatted_df[available_cols]
    
    # Sort by index
    formatted_df.sort_index(inplace=True)
    
    logger.debug(f"Formatted time series data: {formatted_df.shape}")
    return formatted_df