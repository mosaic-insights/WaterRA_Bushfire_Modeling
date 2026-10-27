"""
eWater Source model configuration using the Veneer API.

Functions here connect to a running Source instance, configure the
Load Distributor constituent-generation model, push fire-impact
time-series as Veneer data sources, and trigger simulation runs.
"""

import json
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
# Created by ensure_constituent when the model has none of the above.
DEFAULT_CONSTITUENT = 'TSS'
LIKELY_FUNCTIONAL_UNITS = [
    'Forested', 'Forest', 'Urban',
    'Bushland', 'Burned', 'Native Vegetation',
]

# Filename of the Load Distributor plugin, as listed in 'PluginsLoaded'.
LOAD_DISTRIBUTOR_DLL = 'FlowMatters.Source.LoadDistributor.dll'


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


def _plugin_filename(path) -> str:
    """Return a plugin path's filename, whichever separator it uses."""
    return str(path).replace('\\', '/').rsplit('/', 1)[-1]


def check_plugins_loaded(v: veneer.Veneer, plugins) -> None:
    """
    Verify that each of a list of plugins is loaded in Source.

    Parameters:
    - v: Veneer connection object.
    - plugins: Plugin DLLs, as filenames or full paths.

    Returns:
    - None. Raises if any plugin is known to be absent.
    ------------------------------------------------------------------------
    Notes:
    - Matches on filename only, so the plugin's install directory does not
      matter.
    - Reads 'PluginsLoaded' from Veneer's root endpoint (v.status()).
    - Older Veneer releases omit 'PluginsLoaded'. That case is logged as a
      warning and allowed through, rather than failing a run that would
      otherwise work.
    ------------------------------------------------------------------------
    """
    wanted = [_plugin_filename(p) for p in plugins]
    if not wanted:
        return

    status = v.status()
    loaded = status.get('PluginsLoaded') if isinstance(status, dict) else None

    if loaded is None:
        logger.warning(
            "Veneer did not report 'PluginsLoaded'; cannot verify that %s "
            "are loaded. This is expected on older Veneer releases.",
            ', '.join(wanted),
        )
        return

    loaded_names = {_plugin_filename(entry).lower() for entry in loaded}
    missing = [name for name in wanted if name.lower() not in loaded_names]
    if missing:
        raise RuntimeError(
            f"Plugin(s) not loaded in the connected Source instance: "
            f"{', '.join(missing)}. "
            f"Plugins reported: {loaded}"
        )
    logger.info("Plugins found: %s", ', '.join(wanted))


def check_load_distributor_plugin(v: veneer.Veneer) -> None:
    """
    Verify that the Load Distributor plugin is loaded in Source.

    Parameters:
    - v: Veneer connection object.

    Returns:
    - None. Raises if the plugin is known to be absent.
    ------------------------------------------------------------------------
    Notes:
    - See check_plugins_loaded, which this delegates to.
    ------------------------------------------------------------------------
    """
    check_plugins_loaded(v, [LOAD_DISTRIBUTOR_DLL])


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------

def _first_match(options, candidate_list, param_name) -> Optional[str]:
    """
    Return the first option containing one of the candidate substrings
    (case-insensitive), earlier candidates taking priority; or None.
    """
    for candidate in candidate_list:
        for option in options:
            if candidate.lower() in option.lower():
                logger.info(
                    f"Selected {param_name}: '{option}' "
                    f"(matched '{candidate}') "
                    f"from available: {options}"
                )
                return option
    return None


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
        # Deduplicate but keep the order Source reported the options in:
        # both the substring scan and the fallback below are documented
        # as returning the *first* match, and a set would make that
        # depend on string hashing (ie vary between runs).
        available_options = list(dict.fromkeys(getter_func()))
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

    option = _first_match(available_options, candidate_list, param_name)
    if option is not None:
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


def ensure_constituent(
    v: veneer.Veneer,
    constituent: Optional[str] = None,
    candidate_list: Optional[List[str]] = None,
    default: str = DEFAULT_CONSTITUENT,
) -> str:
    """
    Return the constituent to load fire sediment into, creating it in the
    Source model if the model does not have it.

    Parameters:
    - v: Active Veneer connection object.
    - constituent: Constituent to use (eg source.constituent). Created if
      the model lacks it. Left as None, one is picked from the model.
    - candidate_list: Constituent substrings to pick from, in order, when
      constituent is None. Defaults to LIKELY_CONSTITUENTS.
    - default: Constituent to create when constituent is None and the
      model has nothing matching candidate_list (default 'TSS').

    Returns:
    - The constituent name exactly as it appears in Source.
    ------------------------------------------------------------------------
    Notes:
    - A named constituent matches one in the model regardless of case,
      rather than creating a second that differs only in case.
    - Unlike detect_constituent, this never falls back to an unrelated
      constituent (Salt, say): configuring the Load Distributor replaces
      that constituent's generation models. It creates `default` instead,
      and says which constituents it passed over.
    - Creating a constituent uses veneer-py's add_constituent, which
      requires Veneer's "Allow Scripts".
    ------------------------------------------------------------------------
    """
    available = list(dict.fromkeys(v.model.get_constituents()))

    if constituent is not None:
        if constituent in available:
            logger.info(f"Using provided constituent: '{constituent}'")
            return constituent
        same_but_case = [c for c in available
                         if c.lower() == constituent.lower()]
        if len(same_but_case) == 1:
            logger.warning(
                f"Constituent '{constituent}' is called "
                f"'{same_but_case[0]}' in the Source model; using "
                f"'{same_but_case[0]}'.")
            return same_but_case[0]
        return _add_constituent(v, constituent, available)

    if candidate_list is None:
        candidate_list = LIKELY_CONSTITUENTS
    match = _first_match(available, candidate_list, 'constituent')
    if match is not None:
        return match

    if available:
        logger.warning(
            f"None of the model's constituents {available} looks like "
            f"sediment (checked for {candidate_list}), so creating "
            f"'{default}'. If one of them is the one to use, name it in "
            f"source.constituent.")
    return _add_constituent(v, default, available)


def _add_constituent(v: veneer.Veneer, constituent: str,
                     available: List[str]) -> str:
    """Create a constituent in Source, and check that it now exists."""
    v.model.add_constituent(constituent)
    if constituent not in v.model.get_constituents():
        raise RuntimeError(
            f"Asked Source to create constituent '{constituent}', but the "
            f"model still does not have it. Create it in Source and "
            f"re-run.")
    logger.info(
        f"Created constituent '{constituent}' in the Source model "
        f"(it had {available or 'none'}).")
    return constituent


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

def restricted_data_source_name(name: str) -> str:
    """
    Return the name the Source interface gives a data source called `name`.

    Parameters:
    - name: A data source name, as given to Veneer.

    Returns:
    - The name with '.', '@', '>', ',' and backslash each replaced by an
      underscore, cut to 200 characters, and stripped of surrounding
      spaces.
    ------------------------------------------------------------------------
    Notes:
    - This mirrors Source's DataManager.RestrictName. Source applies it
      when its interface shows a data source, and the new name sticks. So
      with the interface open, a data source Veneer creates from a file
      path ends up named after the path with its backslashes and full stop
      turned into underscores. A Veneer command line keeps the name as
      given.
    ------------------------------------------------------------------------
    """
    for reserved in '.@>,\\':
        name = name.replace(reserved, '_')
    return name[:200].strip(' ')


def _check_data_source_name(name: str) -> None:
    """Refuse a data source name the Source interface would change."""
    restricted = restricted_data_source_name(name)
    if restricted != name:
        raise ValueError(
            f"Source would rename data source '{name}' to '{restricted}' "
            f"when its interface is open, and then not find it under the "
            f"name asked for. Use a name without '.', '@', '>', ',' or "
            f"backslashes, such as '{restricted}'.")


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
    ------------------------------------------------------------------------
    Notes:
    - Raises ValueError for a name the Source interface would change; see
      restricted_data_source_name.
    ------------------------------------------------------------------------
    """
    _check_data_source_name(tss_source_name)
    _check_data_source_name(rainfall_source_name)
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


def create_file_data_source(
    v: veneer.Veneer,
    name: str,
    path,
    units: Optional[str] = None,
    reload_on_run: bool = True,
) -> str:
    """
    Create a data source that Source loads from a CSV file, under a name.

    Parameters:
    - v: Active Veneer connection object.
    - name: Name to give the data source - the one to assign from.
    - path: CSV file Source loads (and, with reload_on_run, re-reads at
      the start of every run).
    - units: Units for the series, as for v.create_data_source.
    - reload_on_run: Re-read the file before each run (default True).

    Returns:
    - name, for passing on to the assignment functions.
    ------------------------------------------------------------------------
    Notes:
    - Veneer names a file-based data source after the path it was given,
      not the file's stem, and with its interface open Source then turns
      that path's backslashes and full stop into underscores (see
      restricted_data_source_name). Assigning from ``name`` would find no
      such data source, and Source fails with a NullReferenceException -
      the same error a column/catchment name mismatch gives. So the data
      source is renamed here, whichever of the two names it has; the file
      it loads from is stored separately and is unaffected.
    - Units are applied after the rename, under ``name``. veneer-py would
      otherwise look the data source up under the path, and fail when the
      interface has renamed it.
    - Any other data source already called ``name`` is removed first - an
      in-memory one of the same name, say, or the empty one a failed
      assignment leaves behind - as is any left from an earlier attempt
      under either version of the path. Requires Veneer's "Allow Scripts".
    - Raises ValueError for a ``name`` the Source interface would itself
      change.
    ------------------------------------------------------------------------
    """
    _check_data_source_name(name)
    path = str(path)
    logger.info(f"Creating data source '{name}' from {path}")
    v.create_data_source(path, reload_on_run=reload_on_run)

    # String literals IronPython reads back unchanged. Pasted in raw, a
    # Windows path's '\2001...' and '\f...' would become escape sequences;
    # and non-ASCII characters go in as themselves, because IronPython 2
    # leaves a \u escape in an ordinary string as literal characters.
    def literal(s):
        return json.dumps(s, ensure_ascii=False)

    # Source's own renaming rule where it has one, so a version whose rule
    # differs is still matched; the copy here otherwise.
    script = (
        f'raw = {literal(path)}\n'
        'try:\n'
        '    import clr\n'
        "    clr.AddReference('RiverSystem')\n"
        '    from RiverSystem.DataManagement.DataManager import DataManager\n'
        '    restricted = DataManager.RestrictName(raw)\n'
        'except Exception:\n'
        f'    restricted = {literal(restricted_data_source_name(path))}\n'
        f'name = {literal(name)}\n'
        'dm = scenario.Network.DataManager\n'
        'from_file = [g for g in dm.DataGroups\n'
        '             if g.Name == raw or g.Name == restricted]\n'
        'if not from_file:\n'
        '    raise Exception("No data source was created from " + raw)\n'
        '# The one just created: Veneer adds it after any earlier ones.\n'
        'grp = from_file[-1]\n'
        'for other in [g for g in dm.DataGroups\n'
        '              if g.Name in (name, raw, restricted)\n'
        '              and not g is grp]:\n'
        '    dm.RemoveGroup(other)\n'
        'grp.Name = name\n'
        'result = grp.Name\n'
    )
    response = v.model.run_script(script, init=True)
    if response.get('Exception') is not None:
        raise Exception(response['Exception'])

    if units is not None:
        v.model.set_data_source_units(name, units)
    return name


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

# The run status Source reports for a run that completed: the name of its
# RunResultType, as Veneer passes it through. Anything else - RunIncomplete,
# say - means the run did not.
RUN_SUCCESS = 'RunSuccess'


def run_succeeded(results: Dict) -> bool:
    """
    Return whether a Source run completed successfully.

    Parameters:
    - results: Dict returned by Veneer's retrieve_run(), or by
      run_model_simulation().

    Returns:
    - True if its 'Status' is RUN_SUCCESS.
    """
    return results.get('Status') == RUN_SUCCESS


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
      a 'Status' key - see run_succeeded().
    """
    logger.info(
        f"Running simulation from {start_date} to {end_date}"
    )

    # Select the requested input set then trigger the run
    v.model.simulation.select_input_set(input_set)
    v.run_model(start=start_date, end=end_date)
    results = v.retrieve_run()

    logger.info(
        f"Simulation completed with status: {results.get('Status')}"
    )

    # Log the run log at debug level if the simulation did not succeed
    if not run_succeeded(results):
        logger.warning(
            "Simulation did not complete successfully (status: %s)",
            results.get('Status'))
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
    (creating it if need be) and functional unit, configures the Load
    Distributor model, loads
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
      the model if not supplied, and created in the model if it lacks
      it - see ensure_constituent.
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
    # from the model if the caller did not specify them, and creating the
    # constituent if the model lacks it.
    constituent = ensure_constituent(v, constituent)

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
