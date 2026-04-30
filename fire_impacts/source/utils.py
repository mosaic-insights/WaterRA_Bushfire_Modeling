"""
Utility functions for Source model configuration and data processing.
"""

import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV and DataFrame validation helpers
# ---------------------------------------------------------------------------

def validate_csv_files(*csv_paths: str) -> None:
    """
    Validate that one or more CSV files exist and are readable.

    Parameters:
    - *csv_paths: Variable number of file-path strings to validate.

    Returns:
    - None
    ------------------------------------------------------------------------
    Notes:
    - Raises FileNotFoundError if any path does not exist on disk.
    - Raises pd.errors.EmptyDataError if any file contains no data rows.
    ------------------------------------------------------------------------
    """
    for csv_path in csv_paths:
        try:
            # Read only the first row to avoid loading large files
            df = pd.read_csv(csv_path, nrows=1)
            if df.empty:
                raise pd.errors.EmptyDataError(
                    f"CSV file is empty: {csv_path}"
                )
            logger.debug(f"Validated CSV file: {csv_path}")
        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty: {csv_path}")
            raise


def check_dataframe_alignment(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name1: str = "df1",
    name2: str = "df2",
) -> bool:
    """
    Check whether two DataFrames share compatible datetime indices.

    Parameters:
    - df1: First DataFrame.
    - df2: Second DataFrame.
    - name1: Label for the first DataFrame used in log messages.
    - name2: Label for the second DataFrame used in log messages.

    Returns:
    - True if the indices match exactly; False otherwise.
    """
    # Mismatched indices cause silent NaN injection in downstream
    # merges — log the ranges to make diagnosis easier.
    if not df1.index.equals(df2.index):
        logger.warning(
            f"Index mismatch between {name1} and {name2}"
        )
        logger.debug(
            f"{name1} index range: "
            f"{df1.index.min()} to {df1.index.max()}"
        )
        logger.debug(
            f"{name2} index range: "
            f"{df2.index.min()} to {df2.index.max()}"
        )
        return False

    # Log any columns that appear in both DataFrames — overlap is not
    # necessarily a problem but is useful to know during debugging.
    common_cols = set(df1.columns) & set(df2.columns)
    if common_cols:
        logger.info(
            f"Common columns between {name1} and {name2}: "
            f"{common_cols}"
        )

    logger.debug(
        f"Alignment check passed for {name1} and {name2}"
    )
    return True


# ---------------------------------------------------------------------------
# DataFrame inspection utilities
# ---------------------------------------------------------------------------

def summarize_dataframe(
    df: pd.DataFrame, name: str = "DataFrame"
) -> Dict:
    """
    Generate shape, memory, null-count, and dtype summary for a DataFrame.

    Parameters:
    - df: DataFrame to summarise.
    - name: Label for the DataFrame used in the returned summary dict.

    Returns:
    - Dict with keys: name, shape, columns, index_range,
      memory_usage_mb, null_counts, dtypes, and (when numeric
      columns are present) numeric_summary.
    """
    summary = {
        'name': name,
        'shape': df.shape,
        'columns': list(df.columns),
        'index_range': (df.index.min(), df.index.max()),
        'memory_usage_mb': (
            df.memory_usage(deep=True).sum() / 1024**2
        ),
        'null_counts': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict(),
    }

    # Append descriptive statistics for numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = (
            df[numeric_cols].describe().to_dict()
        )

    logger.debug(f"Generated summary for {name}: {df.shape}")
    return summary


# ---------------------------------------------------------------------------
# Veneer connection utilities
# ---------------------------------------------------------------------------

def verify_veneer_connection(v) -> Dict:
    """
    Verify a Veneer connection is alive and return basic scenario info.

    Parameters:
    - v: Active Veneer connection object.

    Returns:
    - Dict with keys: scenario_info, constituents, input_sets.
    ------------------------------------------------------------------------
    Notes:
    - Re-raises the underlying exception on failure so the caller can
      decide how to handle a broken connection.
    ------------------------------------------------------------------------
    """
    try:
        info = {
            'scenario_info': v.scenario_info(),
            'constituents': v.model.get_constituents(),
            'input_sets': (
                len(v.input_sets())
                if hasattr(v, 'input_sets')
                else 'Unknown'
            ),
        }
        logger.info("Veneer connection verified successfully")
        return info
    except Exception as e:
        logger.error(f"Failed to verify Veneer connection: {e}")
        raise


def get_model_configuration_summary(v) -> Dict:
    """
    Retrieve a summary of the generation, runoff, and input-set
    configuration from the running Source model.

    Parameters:
    - v: Active Veneer connection object.

    Returns:
    - Dict with keys: generation_models, runoff_models,
      current_input_set.  Each value is either a results sub-dict
      or a dict containing an 'error' key if that query failed.
    ------------------------------------------------------------------------
    Notes:
    - Each sub-query is wrapped independently so a partial failure does
      not suppress the rest of the summary.
    ------------------------------------------------------------------------
    """
    try:
        config = {}

        # Generation models — what constituent-production model is
        # assigned to each subcatchment/functional-unit combination.
        try:
            gen_models = v.model.catchment.generation.model_table(
                constituents='TSS'
            )
            config['generation_models'] = {
                'unique_models': (
                    gen_models.model.unique().tolist()
                    if hasattr(gen_models, 'model')
                    else []
                ),
                'count': (
                    len(gen_models) if gen_models is not None else 0
                ),
            }
        except Exception as e:
            config['generation_models'] = {'error': str(e)}

        # Runoff models — rainfall-runoff model per catchment and
        # functional unit, excluding the built-in Water FU.
        try:
            rr_names = list(
                v.model.catchment.runoff.enumerate_names()
            )
            config['runoff_models'] = {
                'functional_units': list(
                    {nm[1] for nm in rr_names if nm[1] != 'Water'}
                ),
                'catchment_count': len(
                    [nm for nm in rr_names if nm[1] != 'Water']
                ),
            }
        except Exception as e:
            config['runoff_models'] = {'error': str(e)}

        # Active input set (the scenario or climate variant currently
        # selected in Source).
        try:
            config['current_input_set'] = (
                v.model.simulation.current_input_set()
            )
        except Exception as e:
            config['current_input_set'] = {'error': str(e)}

        return config

    except Exception as e:
        logger.error(f"Failed to get model configuration: {e}")
        return {'error': str(e)}


# ---------------------------------------------------------------------------
# Time-series formatting helper
# ---------------------------------------------------------------------------

def format_time_series_data(
    df: pd.DataFrame,
    time_column: Optional[str] = None,
    value_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Normalise a time-series DataFrame into the format expected by Veneer.

    Ensures the index is a DatetimeIndex, optionally promotes a named
    column to the index, and filters to a specified column subset.

    Parameters:
    - df: Input DataFrame.
    - time_column: Name of a column to convert and use as the index.
      Pass None to use the existing index unchanged.
    - value_columns: List of column names to retain.  Pass None to
      keep all columns.

    Returns:
    - A copy of df with a sorted DatetimeIndex and only the requested
      columns present.
    """
    formatted_df = df.copy()

    # Promote the named time column to the index if provided
    if time_column and time_column in formatted_df.columns:
        formatted_df[time_column] = pd.to_datetime(
            formatted_df[time_column]
        )
        formatted_df.set_index(time_column, inplace=True)

    # Coerce whatever index is present to a proper DatetimeIndex
    if not isinstance(formatted_df.index, pd.DatetimeIndex):
        formatted_df.index = pd.to_datetime(formatted_df.index)

    # Filter to requested columns, warning about any that are absent
    if value_columns:
        missing_cols = (
            set(value_columns) - set(formatted_df.columns)
        )
        if missing_cols:
            logger.warning(
                f"Missing columns in dataframe: {missing_cols}"
            )
        available_cols = [
            col for col in value_columns
            if col in formatted_df.columns
        ]
        formatted_df = formatted_df[available_cols]

    formatted_df.sort_index(inplace=True)
    logger.debug(
        f"Formatted time series data: {formatted_df.shape}"
    )
    return formatted_df
