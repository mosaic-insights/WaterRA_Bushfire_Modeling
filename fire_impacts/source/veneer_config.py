"""
eWater Source model configuration using the Veneer API.

Functions here connect to a running Source instance, configure the
Load Distributor constituent-generation model, push fire-impact
time-series as Veneer data sources, and trigger simulation runs.
"""

import pandas as pd
import veneer
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants — candidate names for auto-detection
# ---------------------------------------------------------------------------

# Checked in order; the first substring match wins.
LIKELY_CONSTITUENTS = ['TSS', 'Sediment', 'Contaminant', 'Pollutant']
LIKELY_FUNCTIONAL_UNITS = [
    'Forested', 'Forest', 'Urban',
    'Bushland', 'Burned', 'Native Vegetation',
]


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def connect_to_veneer(port: int = 9877) -> veneer.Veneer:
    """
    Open a connection to a Veneer REST server on the given port.

    Parameters:
    - port: TCP port on which Veneer is listening (default 9877).

    Returns:
    - veneer.Veneer connection object ready for API calls.
    ------------------------------------------------------------------------
    Notes:
    - Re-raises the underlying exception if the connection attempt fails,
      so callers can distinguish a bad port from a network error.
    ------------------------------------------------------------------------
    """
    try:
        v = veneer.Veneer(port)
        logger.info(f"Connected to Veneer on port {port}")
        return v
    except Exception as e:
        logger.error(
            f"Failed to connect to Veneer on port {port}: {e}"
        )
        raise


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------

def _detect_parameter(
    getter_func,
    candidate_list: List[str],
    param_name: str,
    filter_func=None,
) -> str:
    """
    Select a Source model parameter by matching against a priority list.

    Queries the model for available options, applies an optional filter,
    then returns the first option whose name contains one of the candidate
    substrings (case-insensitive).  Falls back to the first available
    option if no candidate matches.

    Parameters:
    - getter_func: Callable that returns the list of available options
      from the Source model.
    - candidate_list: Substrings to check, in preference order.
    - param_name: Human-readable name for the parameter, used in logs.
    - filter_func: Optional callable that narrows available_options
      before matching.  Receives the full option set; returns a
      filtered collection.

    Returns:
    - The matching option name exactly as Source reports it.
    ------------------------------------------------------------------------
    Notes:
    - Raises ValueError if getter_func returns no options (after
      filtering).
    ------------------------------------------------------------------------
    """
    try:
        available_options = set(getter_func())
        logger.debug(
            f"Available {param_name} in model: {available_options}"
        )
    except Exception as e:
        logger.error(
            f"Failed to get {param_name} from model: {e}"
        )
        raise

    # Narrow options using the caller-supplied filter if provided
    if filter_func:
        available_options = filter_func(available_options)
        logger.debug(f"Filtered {param_name}: {available_options}")

    if not available_options:
        raise ValueError(f"No {param_name} found in model")

    # Return the first option that contains one of the candidate
    # substrings — earlier candidates have higher priority.
    for candidate in candidate_list:
        for option in available_options:
            if candidate.lower() in option.lower():
                logger.info(
                    f"Selected {param_name}: '{option}' "
                    f"(matched '{candidate}') "
                    f"from available: {available_options}"
                )
                return option

    # No candidate matched — fall back to the first available option
    selected = next(iter(available_options))
    logger.warning(
        f"No matching {param_name} found in candidates "
        f"{candidate_list}. Using first available: '{selected}'"
    )
    return selected


def detect_constituent(
    v: veneer.Veneer,
    candidate_list: Optional[List[str]] = None,
) -> str:
    """
    Detect and select the most likely constituent from the Source model.

    Parameters:
    - v: Active Veneer connection object.
    - candidate_list: Constituent substrings to check in order.
      Defaults to LIKELY_CONSTITUENTS if not supplied.

    Returns:
    - The constituent name exactly as it appears in Source.
    ------------------------------------------------------------------------
    Notes:
    - Raises ValueError if no constituent can be found in the model.
    ------------------------------------------------------------------------
    """
    if candidate_list is None:
        candidate_list = LIKELY_CONSTITUENTS

    return _detect_parameter(
        getter_func=v.model.get_constituents,
        candidate_list=candidate_list,
        param_name="constituents",
    )


def detect_functional_unit(
    v: veneer.Veneer,
    candidate_list: Optional[List[str]] = None,
) -> str:
    """
    Detect and select the most likely functional unit from the Source model.

    Parameters:
    - v: Active Veneer connection object.
    - candidate_list: Functional-unit substrings to check in order.
      Defaults to LIKELY_FUNCTIONAL_UNITS if not supplied.

    Returns:
    - The functional unit name exactly as it appears in Source.
    ------------------------------------------------------------------------
    Notes:
    - The built-in 'Water' functional unit is always excluded before
      matching, as it is not a valid target for sediment or fire data.
    - Raises ValueError if no functional unit remains after filtering.
    ------------------------------------------------------------------------
    """
    if candidate_list is None:
        candidate_list = LIKELY_FUNCTIONAL_UNITS

    # Exclude the built-in Water FU — it cannot receive load inputs
    def filter_water(fus):
        return [fu for fu in fus if fu != 'Water']

    return _detect_parameter(
        getter_func=v.model.catchment.get_functional_unit_types,
        candidate_list=candidate_list,
        param_name="functional units",
        filter_func=filter_water,
    )


# ---------------------------------------------------------------------------
# Load Distributor model configuration
# ---------------------------------------------------------------------------

def configure_load_distributor_model(
    v: veneer.Veneer,
    constituent: str = 'TSS',
    load_attenuation: float = 10.0,
    maximum_concentration: float = 1000.0,
) -> None:
    """
    Switch the constituent-generation model to Load Distributor and set
    its attenuation and concentration-cap parameters.

    Reads the current EMC/DWC model table to discover which functional
    units are configured, then replaces those models with
    DistributeLoadModel across the board.

    Parameters:
    - v: Active Veneer connection object.
    - constituent: Constituent to configure (default: 'TSS').
    - load_attenuation: LoadAttenuation parameter value.
    - maximum_concentration: MaximumConcentration parameter value.

    Returns:
    - None
    """
    logger.info(
        f"Configuring Load Distributor model for {constituent}"
    )

    # Inspect what generation models are currently set
    generation_models = v.model.catchment.generation.model_table(
        constituents=constituent
    )
    logger.debug(
        f"Current models: {generation_models.model.unique()}"
    )

    # Discover which functional units exist by reading the EMC/DWC
    # parameter table — these are the FUs we need to replace.
    emc_dwc_params = (
        v.model.catchment.generation.tabulate_parameters(
            'RiverSystem.Catchments.Models.'
            'ContaminantGenerationModels.EmcDwcCGModel',
            constituents=constituent,
        )
    )
    functional_units = list(
        emc_dwc_params['Functional Unit'].unique()
    )

    # Replace all matching FUs with the Load Distributor model
    v.model.catchment.generation.set_models(
        "FlowMatters.Source.LoadDistributor.DistributeLoadModel",
        constituents=constituent,
        fus=functional_units,
    )

    # Apply the attenuation and concentration cap to all catchments
    v.model.catchment.generation.set_param_values(
        'LoadAttenuation', load_attenuation
    )
    v.model.catchment.generation.set_param_values(
        'MaximumConcentration', maximum_concentration
    )

    logger.info(
        f"Load Distributor configured: "
        f"LoadAttenuation={load_attenuation}, "
        f"MaximumConcentration={maximum_concentration}"
    )


# ---------------------------------------------------------------------------
# Fire impact data loading
# ---------------------------------------------------------------------------

def load_fire_impact_data(
    rusle_csv_path: str,
    debris_flow_csv_path: str,
    rainfall_csv_path: str,
) -> tuple:
    """
    Load and combine RUSLE and debris-flow TSS data with rainfall data.

    Reads three CSV files produced by the fire-impacts ensemble,
    aligns their indices, sums the two TSS components, and renames
    columns to Source subcatchment naming convention.

    Parameters:
    - rusle_csv_path: Path to the RUSLE hourly-load CSV file.
    - debris_flow_csv_path: Path to the debris-flow hourly-load CSV.
    - rainfall_csv_path: Path to the hourly rainfall CSV file.

    Returns:
    - Tuple of (combined_tss_df, rainfall_df) where combined_tss_df
      is RUSLE + debris-flow loads (kg/hr) and rainfall_df is
      rainfall depth (mm/hr), both indexed by datetime.
    ------------------------------------------------------------------------
    Notes:
    - The debris-flow CSV has one extra trailing row; the RUSLE index
      is re-aligned to match before summing.
    - Column names are converted from 'AVSC...' to 'SC #...' to match
      the subcatchment naming convention used in Source.
    ------------------------------------------------------------------------
    """
    logger.info("Loading fire impact data")

    # Load per-hour TSS loads for the two erosion components
    hourly_tss_rusle = pd.read_csv(
        rusle_csv_path, index_col=0, parse_dates=True
    )
    hourly_tss_debris_flow = pd.read_csv(
        debris_flow_csv_path, index_col=0, parse_dates=True
    )

    # Re-align RUSLE index to the debris-flow index (debris has one
    # extra trailing row that we drop by slicing to [:-1])
    hourly_tss_rusle.index = hourly_tss_debris_flow.index[:-1]

    # Load rainfall data
    hourly_rainfall = pd.read_csv(
        rainfall_csv_path, index_col=0, parse_dates=True
    )

    # Sum the two TSS components, dropping any rows with NaN
    total_tss = (hourly_tss_debris_flow + hourly_tss_rusle).dropna()

    # Rename columns to match Source subcatchment naming convention
    def rename_col(c):
        return c.replace('AVSC', 'SC #')

    total_tss = total_tss.rename(columns=rename_col)
    hourly_rainfall = hourly_rainfall.rename(columns=rename_col)

    logger.info(
        f"Loaded TSS shape: {total_tss.shape}, "
        f"rainfall shape: {hourly_rainfall.shape}"
    )
    return total_tss, hourly_rainfall


# ---------------------------------------------------------------------------
# Veneer data source management
# ---------------------------------------------------------------------------

def create_veneer_data_sources(
    v: veneer.Veneer,
    tss_data: pd.DataFrame,
    rainfall_data: pd.DataFrame,
    tss_source_name: str = 'fire_tss',
    rainfall_source_name: str = 'stochastic_rain',
    timestep: str = 'day',
) -> None:
    """
    Register TSS and rainfall DataFrames as named data sources in Veneer.

    Parameters:
    - v: Active Veneer connection object.
    - tss_data: DataFrame of TSS loads indexed by datetime.
    - rainfall_data: DataFrame of rainfall depths indexed by datetime.
    - tss_source_name: Name to register the TSS source under in Veneer.
    - rainfall_source_name: Name to register the rainfall source under.
    - timestep: Time-unit string appended to the units label
      (e.g. 'day' produces 'kg/day').

    Returns:
    - None
    """
    logger.info(
        f"Creating data sources: "
        f"'{tss_source_name}', '{rainfall_source_name}'"
    )
    v.create_data_source(
        tss_source_name, tss_data, units=f'kg/{timestep}'
    )
    v.create_data_source(
        rainfall_source_name, rainfall_data, units=f'mm/{timestep}'
    )
    logger.info("Data sources created successfully")


# ---------------------------------------------------------------------------
# Time-series assignment helpers
# ---------------------------------------------------------------------------

def assign_fire_sediment_timeseries(
    v: veneer.Veneer,
    tss_source_name: str = 'fire_tss',
    constituent: str = 'TSS',
    functional_unit: str = 'Forested',
) -> None:
    """
    Wire a Veneer TSS data source to the Load Distributor load inputs.

    Enumerates catchment names for the given constituent and functional
    unit, then assigns each catchment's column from the data source to
    the 'Load' parameter of its Load Distributor model.

    Parameters:
    - v: Active Veneer connection object.
    - tss_source_name: Name of the TSS data source registered in Veneer.
    - constituent: Constituent name in Source (default: 'TSS').
    - functional_unit: Functional unit to assign loads to.

    Returns:
    - None
    """
    logger.info(
        f"Assigning TSS time series for "
        f"'{functional_unit}' functional unit"
    )

    # Get the ordered list of catchment names for this FU — the order
    # must match the column order in the data source.
    full_names = v.model.catchment.generation.enumerate_names(
        constituents=constituent,
        sources='Default',
    )
    tss_sc_order = [
        nm[0] for nm in full_names if nm[1] == functional_unit
    ]

    # Assign the time series, mapping each catchment to its column
    v.model.catchment.generation.assign_time_series(
        'Load',
        tss_sc_order,
        tss_source_name,
        fromList=True,
        constituents=constituent,
        fus=functional_unit,
    )
    logger.info(
        f"Assigned TSS time series to {len(tss_sc_order)} catchments"
    )


def assign_rainfall_timeseries(
    v: veneer.Veneer,
    rainfall_source_name: str = 'stochastic_rain',
) -> None:
    """
    Assign a stochastic rainfall data source to all rainfall-runoff models.

    Clears any existing rainfall and PET assignments, then re-assigns
    the named data source to every functional unit (excluding 'Water').

    Parameters:
    - v: Active Veneer connection object.
    - rainfall_source_name: Name of the rainfall data source in Veneer.

    Returns:
    - None
    """
    logger.info("Assigning rainfall time series")

    # Clear any existing rainfall and PET assignments so we start fresh
    v.model.catchment.runoff.clear_time_series('rainfall')
    v.model.catchment.runoff.clear_time_series('Pet')

    # Collect the set of functional units to assign rainfall to,
    # excluding the built-in Water FU.
    rr_names = list(v.model.catchment.runoff.enumerate_names())
    fus = {nm[1] for nm in rr_names if nm[1] != 'Water'}

    v.model.catchment.runoff.assign_time_series(
        'rainfall', 'rainfall', rainfall_source_name, fus=fus
    )
    logger.info(f"Assigned rainfall to functional units: {fus}")


# ---------------------------------------------------------------------------
# Simulation runner and model saver
# ---------------------------------------------------------------------------

def run_model_simulation(
    v: veneer.Veneer,
    start_date: str = '01/01/1900',
    end_date: str = '31/12/1901',
    input_set: str = 'Default',
) -> Dict:
    """
    Run a Source simulation and return the run-result dictionary.

    Parameters:
    - v: Active Veneer connection object.
    - start_date: Simulation start date in DD/MM/YYYY format.
    - end_date: Simulation end date in DD/MM/YYYY format.
    - input_set: Name of the Source input set to activate before running.

    Returns:
    - Dict returned by Veneer's retrieve_run(), including at minimum
      a 'Status' key ('Finished' on success).
    """
    logger.info(
        f"Running simulation from {start_date} to {end_date}"
    )

    # Select the requested input set then trigger the run
    v.model.simulation.select_input_set(input_set)
    v.run_model(start=start_date, end=end_date)
    results = v.retrieve_run()

    logger.info(
        f"Simulation completed with status: {results['Status']}"
    )

    # Log the run log at debug level if the simulation did not finish
    if results['Status'] != 'Finished':
        logger.warning("Simulation did not complete successfully")
        if 'RunLog' in results:
            logger.debug(
                "Run log:\n" + '\n'.join(results['RunLog'])
            )

    return results


def save_model(v: veneer.Veneer, filename: str) -> None:
    """
    Save the current Source model to a project file.

    Parameters:
    - v: Active Veneer connection object.
    - filename: Target filename (typically *.rsproj).

    Returns:
    - None
    """
    logger.info(f"Saving model as '{filename}'")
    v.model.save(filename)


# ---------------------------------------------------------------------------
# High-level end-to-end workflow
# ---------------------------------------------------------------------------

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
    run_simulation: bool = False,
) -> Dict:
    """
    Run the complete workflow to configure a Source model with fire data.

    Connects to Veneer, auto-detects or uses the provided constituent
    and functional unit, configures the Load Distributor model, loads
    and pushes fire-impact time series, and optionally runs a test
    simulation and saves the result.

    Parameters:
    - port: TCP port on which Veneer is listening.
    - rusle_csv_path: Path to the RUSLE TSS data CSV file.
    - debris_flow_csv_path: Path to the debris-flow TSS data CSV file.
    - rainfall_csv_path: Path to the rainfall data CSV file.
    - output_model_name: If provided, save the configured model under
      this filename.
    - constituent: Constituent name (e.g. 'TSS').  Auto-detected from
      the model if not supplied.
    - functional_unit: Functional unit name (e.g. 'Forested').
      Auto-detected from the model if not supplied.
    - load_attenuation: LoadAttenuation parameter for Load Distributor.
    - maximum_concentration: MaximumConcentration parameter for Load
      Distributor.
    - run_simulation: If True, run a test simulation after configuration.

    Returns:
    - Dict with keys: status, constituent, functional_unit, and
      (if run_simulation is True) simulation, and (if
      output_model_name is provided) saved_model.
    """
    logger.info(
        "Starting complete Source model configuration workflow"
    )

    # Step 1 — establish connection
    v = connect_to_veneer(port)

    # Step 2 — resolve constituent and functional unit, auto-detecting
    # from the model if the caller did not specify them.
    if constituent is None:
        constituent = detect_constituent(v)
    else:
        logger.info(f"Using provided constituent: '{constituent}'")

    if functional_unit is None:
        functional_unit = detect_functional_unit(v)
    else:
        logger.info(
            f"Using provided functional unit: '{functional_unit}'"
        )

    # Step 3 — switch generation models to Load Distributor
    configure_load_distributor_model(
        v, constituent, load_attenuation, maximum_concentration
    )

    # Step 4 — load CSVs and push them to Veneer as data sources
    tss_data, rainfall_data = load_fire_impact_data(
        rusle_csv_path, debris_flow_csv_path, rainfall_csv_path
    )
    create_veneer_data_sources(v, tss_data, rainfall_data)

    # Step 5 — wire the data sources to the correct model inputs
    assign_fire_sediment_timeseries(
        v,
        constituent=constituent,
        functional_unit=functional_unit,
    )
    assign_rainfall_timeseries(v)

    results = {
        'status': 'configured',
        'constituent': constituent,
        'functional_unit': functional_unit,
    }

    # Optional: run a test simulation to verify end-to-end wiring
    if run_simulation:
        sim_results = run_model_simulation(v)
        results['simulation'] = sim_results

    # Optional: save the configured model to disk
    if output_model_name:
        save_model(v, output_model_name)
        results['saved_model'] = output_model_name

    logger.info(
        "Source model configuration workflow completed"
    )
    return results
