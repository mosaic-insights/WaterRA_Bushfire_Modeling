"""
eWater Source model configuration using Veneer API.

This module provides functions to configure Source models with rainfall and 
sediment data from fire impacts modeling.
"""

import pandas as pd
import veneer
from typing import Optional, Dict, List, Union, Set
import logging

logger = logging.getLogger(__name__)

# Default candidates for constituent and functional unit selection
LIKELY_CONSTITUENTS = ['TSS', 'Sediment', 'Contaminant', 'Pollutant']
LIKELY_FUNCTIONAL_UNITS = ['Forested', 'Forest', 'Urban', 'Bushland', 'Burned', 'Native Vegetation']


def connect_to_veneer(port: int = 9877) -> veneer.Veneer:
    """
    Connect to a Veneer instance.
    
    Args:
        port: Port number for Veneer connection
        
    Returns:
        Veneer connection object
    """
    try:
        v = veneer.Veneer(port)
        logger.info(f"Connected to Veneer on port {port}")
        return v
    except Exception as e:
        logger.error(f"Failed to connect to Veneer on port {port}: {e}")
        raise


def _detect_parameter(
    getter_func,
    candidate_list: List[str],
    param_name: str,
    filter_func=None
) -> str:
    """
    Generic utility function to detect and select a parameter from available options.
    
    Matches candidates using substring matching (case-insensitive): a candidate is
    considered a match if it appears as a substring in any available option.
    
    Args:
        getter_func: Callable that returns list of available options from model
        candidate_list: List of candidate substrings to check in order
        param_name: Name of parameter being detected (for logging)
        filter_func: Optional function to filter available options
        
    Returns:
        Selected parameter name (from available_options, not candidate_list)
        
    Raises:
        ValueError: If no suitable parameter is found
    """
    try:
        available_options = getter_func()
        logger.debug(f"Available {param_name} in model: {available_options}")
    except Exception as e:
        logger.error(f"Failed to get {param_name} from model: {e}")
        raise
    
    # Apply filter if provided
    if filter_func:
        available_options = filter_func(available_options)
        logger.debug(f"Filtered {param_name}: {available_options}")
    
    if not available_options:
        raise ValueError(f"No {param_name} found in model")
    
    # Find first match from candidate list using substring matching
    for candidate in candidate_list:
        for option in available_options:
            if candidate.lower() in option.lower():
                logger.info(f"Selected {param_name}: '{option}' (matched candidate '{candidate}') "
                           f"from available: {available_options}")
                return option
    
    # If no match found, use the first available
    selected = available_options[0]
    logger.warning(f"No matching {param_name} found in candidates {candidate_list}. "
                  f"Using first available: '{selected}'")
    return selected


def detect_constituent(v: veneer.Veneer, candidate_list: Optional[List[str]] = None) -> str:
    """
    Detect and select a likely constituent from the Source model.
    
    Args:
        v: Veneer connection object
        candidate_list: List of constituent names to check in order (default: LIKELY_CONSTITUENTS)
        
    Returns:
        Selected constituent name
        
    Raises:
        ValueError: If no suitable constituent is found
    """
    if candidate_list is None:
        candidate_list = LIKELY_CONSTITUENTS
    
    return _detect_parameter(
        getter_func=v.model.get_constituents,
        candidate_list=candidate_list,
        param_name="constituents"
    )


def detect_functional_unit(v: veneer.Veneer, candidate_list: Optional[List[str]] = None) -> str:
    """
    Detect and select a likely functional unit from the Source model.
    
    Args:
        v: Veneer connection object
        candidate_list: List of functional unit names to check in order (default: LIKELY_FUNCTIONAL_UNITS)
        
    Returns:
        Selected functional unit name
        
    Raises:
        ValueError: If no suitable functional unit is found
    """
    if candidate_list is None:
        candidate_list = LIKELY_FUNCTIONAL_UNITS
    
    # Filter out 'Water' as it's typically not a target for sediment/fire data
    def filter_water(fus):
        return [fu for fu in fus if fu != 'Water']
    
    return _detect_parameter(
        getter_func=v.model.catchment.functional_units,
        candidate_list=candidate_list,
        param_name="functional units",
        filter_func=filter_water
    )


def configure_load_distributor_model(
    v: veneer.Veneer,
    constituent: str = 'TSS',
    load_attenuation: float = 10.0,
    maximum_concentration: float = 1000.0
) -> None:
    """
    Configure the Source model to use Load Distributor model for constituent generation.
    
    Args:
        v: Veneer connection object
        constituent: Constituent to configure (default: 'TSS')
        load_attenuation: Load attenuation parameter value
        maximum_concentration: Maximum concentration parameter value
    """
    logger.info(f"Configuring Load Distributor model for {constituent}")
    
    # Get current generation models
    generation_models = v.model.catchment.generation.model_table(constituents=constituent)
    logger.debug(f"Current models: {generation_models.model.unique()}")
    
    # Get current EMC/DWC parameters to identify functional units
    emc_dwc_params = v.model.catchment.generation.tabulate_parameters(
        'RiverSystem.Catchments.Models.ContaminantGenerationModels.EmcDwcCGModel',
        constituents=constituent
    )
    functional_units = list(emc_dwc_params['Functional Unit'].unique())
    
    # Set models to Load Distributor
    v.model.catchment.generation.set_models(
        "FlowMatters.Source.LoadDistributor.DistributeLoadModel",
        constituents=constituent,
        fus=functional_units
    )
    
    # Set parameter values
    v.model.catchment.generation.set_param_values('LoadAttenuation', load_attenuation)
    v.model.catchment.generation.set_param_values('MaximumConcentration', maximum_concentration)
    
    logger.info(f"Load Distributor model configured with LoadAttenuation={load_attenuation}, "
                f"MaximumConcentration={maximum_concentration}")


def load_fire_impact_data(
    rusle_csv_path: str,
    debris_flow_csv_path: str,
    rainfall_csv_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and combine fire impact TSS data with rainfall data.
    
    Args:
        rusle_csv_path: Path to RUSLE hourly load CSV file
        debris_flow_csv_path: Path to debris flow hourly load CSV file  
        rainfall_csv_path: Path to hourly rainfall CSV file
        
    Returns:
        Tuple of (combined_tss_dataframe, rainfall_dataframe)
    """
    logger.info("Loading fire impact data")
    
    # Load TSS data
    hourly_tss_rusle = pd.read_csv(rusle_csv_path, index_col=0, parse_dates=True)
    hourly_tss_debris_flow = pd.read_csv(debris_flow_csv_path, index_col=0, parse_dates=True)
    
    # Align indices (debris flow has one extra row)
    hourly_tss_rusle.index = hourly_tss_debris_flow.index[:-1]
    
    # Load rainfall data
    hourly_rainfall = pd.read_csv(rainfall_csv_path, index_col=0, parse_dates=True)
    
    # Combine TSS data
    total_tss = (hourly_tss_debris_flow + hourly_tss_rusle).dropna()
    
    # Rename columns to match Source naming convention
    def rename_col(c):
        return c.replace('AVSC', 'SC #')
    
    total_tss = total_tss.rename(columns=rename_col)
    hourly_rainfall = hourly_rainfall.rename(columns=rename_col)
    
    logger.info(f"Loaded TSS data shape: {total_tss.shape}, rainfall shape: {hourly_rainfall.shape}")
    
    return total_tss, hourly_rainfall


def create_veneer_data_sources(
    v: veneer.Veneer,
    tss_data: pd.DataFrame,
    rainfall_data: pd.DataFrame,
    tss_source_name: str = 'fire_tss',
    rainfall_source_name: str = 'stochastic_rain'
) -> None:
    """
    Create data sources in Veneer for TSS and rainfall data.
    
    Args:
        v: Veneer connection object
        tss_data: TSS data DataFrame
        rainfall_data: Rainfall data DataFrame
        tss_source_name: Name for TSS data source in Veneer
        rainfall_source_name: Name for rainfall data source in Veneer
    """
    logger.info(f"Creating data sources: {tss_source_name}, {rainfall_source_name}")
    
    # Create data sources
    v.create_data_source(tss_source_name, tss_data, units='kg/h')
    v.create_data_source(rainfall_source_name, rainfall_data, units='mm/h')
    
    logger.info("Data sources created successfully")


def assign_fire_sediment_timeseries(
    v: veneer.Veneer,
    tss_source_name: str = 'fire_tss',
    constituent: str = 'TSS',
    functional_unit: str = 'Forested'
) -> None:
    """
    Assign fire-related TSS time series to the Source model.
    
    Args:
        v: Veneer connection object
        tss_source_name: Name of TSS data source in Veneer
        constituent: Constituent name (default: 'TSS')
        functional_unit: Functional unit to assign data to (default: 'Forested')
    """
    logger.info(f"Assigning TSS time series for {functional_unit} functional unit")
    
    # Get functional unit names for the constituent
    full_names = v.model.catchment.generation.enumerate_names(
        constituents=constituent,
        sources='Default'
    )
    tss_sc_order = [nm[0] for nm in full_names if nm[1] == functional_unit]
    
    # Assign time series
    v.model.catchment.generation.assign_time_series(
        'Load',
        tss_sc_order,
        tss_source_name,
        fromList=True,
        constituents=constituent,
        fus=functional_unit
    )
    
    logger.info(f"Assigned TSS time series to {len(tss_sc_order)} catchments")


def assign_rainfall_timeseries(
    v: veneer.Veneer,
    rainfall_source_name: str = 'stochastic_rain'
) -> None:
    """
    Assign rainfall time series to all functional units in the Source model.
    
    Args:
        v: Veneer connection object
        rainfall_source_name: Name of rainfall data source in Veneer
    """
    logger.info("Assigning rainfall time series")
    
    # Clear existing rainfall and PET data
    v.model.catchment.runoff.clear_time_series('rainfall')
    v.model.catchment.runoff.clear_time_series('Pet')
    
    # Get functional units for rainfall-runoff
    rr_names = list(v.model.catchment.runoff.enumerate_names())
    fus = {nm[1] for nm in rr_names if nm[1] != 'Water'}
    
    # Assign rainfall to all functional units
    v.model.catchment.runoff.assign_time_series(
        'rainfall', 'rainfall', rainfall_source_name, fus=fus
    )
    
    logger.info(f"Assigned rainfall to functional units: {fus}")


def run_model_simulation(
    v: veneer.Veneer,
    start_date: str = '01/01/1900',
    end_date: str = '31/12/1901',
    input_set: str = 'Default'
) -> Dict:
    """
    Run a Source model simulation and return results.
    
    Args:
        v: Veneer connection object
        start_date: Simulation start date (format: DD/MM/YYYY)
        end_date: Simulation end date (format: DD/MM/YYYY) 
        input_set: Input set to use for simulation
        
    Returns:
        Dictionary containing run results and status
    """
    logger.info(f"Running simulation from {start_date} to {end_date}")
    
    # Select input set
    v.model.simulation.select_input_set(input_set)
    
    # Run model
    v.run_model(start=start_date, end=end_date)
    
    # Retrieve results
    results = v.retrieve_run()
    
    logger.info(f"Simulation completed with status: {results['Status']}")
    
    if results['Status'] != 'Finished':
        logger.warning("Simulation did not complete successfully")
        if 'RunLog' in results:
            logger.debug("Run log:\n" + '\n'.join(results['RunLog']))
    
    return results


def save_model(v: veneer.Veneer, filename: str) -> None:
    """
    Save the Source model to a file.
    
    Args:
        v: Veneer connection object
        filename: Filename to save the model as
    """
    logger.info(f"Saving model as {filename}")
    v.model.save(filename)


def configure_source_model_with_fire_data(
    port: int,
    rusle_csv_path: str,
    debris_flow_csv_path: str,
    rainfall_csv_path: str,
    output_model_name: Optional[str] = None,
    constituent: Optional[str] = None,
    functional_unit: Optional[str] = None,
    load_attenuation: float = 10.0,
    maximum_concentration: float = 1000.0,
    run_simulation: bool = False
) -> Dict:
    """
    Complete workflow to configure a Source model with fire impact data.
    
    Args:
        port: Veneer port number
        rusle_csv_path: Path to RUSLE TSS data CSV
        debris_flow_csv_path: Path to debris flow TSS data CSV
        rainfall_csv_path: Path to rainfall data CSV
        output_model_name: Name for saved model file (optional)
        constituent: Constituent name (e.g., 'TSS'). If None, auto-detects from model
        functional_unit: Functional unit name (e.g., 'Forested'). If None, auto-detects from model
        load_attenuation: Load attenuation parameter value
        maximum_concentration: Maximum concentration parameter value
        run_simulation: Whether to run a test simulation
        
    Returns:
        Dictionary with operation results, detected parameters, and model run results if simulation was run
    """
    logger.info("Starting complete Source model configuration workflow")
    
    # Connect to Veneer
    v = connect_to_veneer(port)
    
    # Auto-detect constituent if not provided
    if constituent is None:
        constituent = detect_constituent(v)
    else:
        logger.info(f"Using provided constituent: '{constituent}'")
    
    # Auto-detect functional unit if not provided
    if functional_unit is None:
        functional_unit = detect_functional_unit(v)
    else:
        logger.info(f"Using provided functional unit: '{functional_unit}'")
    
    # Configure load distributor model
    configure_load_distributor_model(v, constituent, load_attenuation, maximum_concentration)
    
    # Load fire impact data
    tss_data, rainfall_data = load_fire_impact_data(
        rusle_csv_path, debris_flow_csv_path, rainfall_csv_path
    )
    
    # Create data sources in Veneer
    create_veneer_data_sources(v, tss_data, rainfall_data)
    
    # Assign time series data
    assign_fire_sediment_timeseries(v, constituent=constituent, functional_unit=functional_unit)
    assign_rainfall_timeseries(v)
    
    results = {
        'status': 'configured',
        'constituent': constituent,
        'functional_unit': functional_unit
    }
    
    # Run simulation if requested
    if run_simulation:
        sim_results = run_model_simulation(v)
        results['simulation'] = sim_results
    
    # Save model if output name provided
    if output_model_name:
        save_model(v, output_model_name)
        results['saved_model'] = output_model_name
    
    logger.info("Source model configuration workflow completed")
    
    return results