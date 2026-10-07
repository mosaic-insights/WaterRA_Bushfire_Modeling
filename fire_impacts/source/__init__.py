"""
eWater Source model configuration module for fire impacts modeling.
"""

from .veneer_config import (
    connect_to_veneer,
    check_load_distributor_plugin,
    check_plugins_loaded,
    configure_load_distributor_model,
    detect_constituent,
    detect_functional_unit,
    load_fire_impact_data,
    create_veneer_data_sources,
    assign_fire_sediment_timeseries,
    assign_rainfall_timeseries,
    run_model_simulation,
    run_succeeded,
    save_model,
    configure_source_model_with_fire_data,
    LIKELY_CONSTITUENTS,
    LIKELY_FUNCTIONAL_UNITS,
    LOAD_DISTRIBUTOR_DLL,
)

from .launch import (
    SourceSession,
    open_source,
    veneer_command_line,
)

from .utils import (
    validate_csv_files,
    check_dataframe_alignment,
    summarize_dataframe,
    verify_veneer_connection,
    get_model_configuration_summary,
    format_time_series_data
)

__all__ = [
    # Configuration constants
    'LIKELY_CONSTITUENTS',
    'LIKELY_FUNCTIONAL_UNITS',
    'LOAD_DISTRIBUTOR_DLL',

    # Main configuration functions
    'connect_to_veneer',
    'check_load_distributor_plugin',
    'check_plugins_loaded',
    'configure_load_distributor_model',
    'detect_constituent',
    'detect_functional_unit',
    'load_fire_impact_data',
    'create_veneer_data_sources',
    'assign_fire_sediment_timeseries',
    'assign_rainfall_timeseries',
    'run_model_simulation',
    'run_succeeded',
    'save_model',
    'configure_source_model_with_fire_data',

    # Opening the Source model a study names
    'SourceSession',
    'open_source',
    'veneer_command_line',
    
    # Utility functions
    'validate_csv_files',
    'check_dataframe_alignment',
    'summarize_dataframe',
    'verify_veneer_connection',
    'get_model_configuration_summary',

    'format_time_series_data'
]