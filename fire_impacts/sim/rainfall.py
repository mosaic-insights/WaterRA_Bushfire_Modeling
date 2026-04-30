"""
Rainfall data handling for fire-impacts simulations.

Covers three main concerns:
- Converting and reshaping pyraingen stochastic rainfall output into
  formats the simulation can consume.
- Importing and resampling observed rainfall from Excel files.
- Low-level helpers for converting between depth and intensity.
"""

from datetime import datetime
import os
import datetime as dt
import xarray as xr
import pandas as pd
import numpy as np
from fire_impacts import const as c


# ---------------------------------------------------------------------------
# Julian date conversion utilities
# ---------------------------------------------------------------------------

def convert_from_julian(dates):
    """Convert an iterable of Julian dates to Python datetime objects."""
    from pyraingen.jdtodatevec import jdToDateVec
    return [
        datetime(*[int(c) for c in jdToDateVec(jd)]) for jd in dates
    ]


def convert_to_julian(dates_array):
    """
    Convert an array of date strings to Julian Date numbers.

    Uses the algorithm from Jean Meeus' Astronomical Algorithms,
    adapted with the assistance of ChatGPT.

    Parameters:
    - dates_array: Array-like of date strings or numpy datetime64
      values, at day precision.

    Returns:
    - 1-D numpy float64 array of Julian Date values corresponding
      to each input date.
    """
    # Convert to numpy datetime objects at day precision
    d_arr = np.asarray(dates_array, dtype='datetime64[D]')

    # Number of whole months since the Unix epoch (Jan 1970)
    months_since_epoch = d_arr.astype('datetime64[M]').astype(int)
    # Derive the Gregorian year and 1-based month-of-year
    years_since_zero = (months_since_epoch // 12) + 1970
    months_leftover = (months_since_epoch % 12) + 1

    # Day of month (1-based) via numpy timedelta
    month_day_delta = d_arr - d_arr.astype('datetime64[M]')
    day_of_month = month_day_delta.astype(int) + 1

    # Shift Jan/Feb back by one year to simplify leap-year calculation
    months_le2 = months_leftover <= 2
    shifted_years = years_since_zero - months_le2.astype(int)
    shifted_months = months_leftover + 12 * months_le2.astype(int)

    # Gregorian calendar correction term
    century = shifted_years // 100
    correction = 2 - century + (century // 4)

    # Constants from the Meeus formula
    days_per_year = 365.25
    years_bc = 4716
    days_per_month = 30.6001
    epoch_anchor = 1524.5

    # Assemble the Julian Date
    whole_year_days = np.floor(
        days_per_year * (shifted_years + years_bc)
    )
    whole_month_days = np.floor(
        days_per_month * (shifted_months + 1)
    )
    jd = (
        whole_year_days
        + whole_month_days
        + day_of_month
        + correction
        - epoch_anchor
    ).astype(np.float64)

    return jd


# ---------------------------------------------------------------------------
# Pyraingen rainfall reshaping
# ---------------------------------------------------------------------------

def flatten_pyraingen_rainfall(
    source, rain_data_start=None, rain_data_end=None
):
    """
    Flatten sub-daily stochastic rainfall from pyraingen format to a
    continuous time series per simulation.

    Pyraingen stores data as (day × subday × simulation); this function
    stacks day and subday into a single 'time' dimension.  If the input
    is already in (simulation × time) format it is returned unchanged.

    Parameters:
    - source: Path string to a NetCDF file, or an xarray.Dataset
      already loaded in memory.
    - rain_data_start: Optional start date for slicing (inclusive).
    - rain_data_end: Optional end date for slicing (inclusive).

    Returns:
    - xarray.Dataset with a 'time' dimension indexed by datetime,
      one variable per simulation.
    """
    if isinstance(source, str):
        ds = (
            xr.open_dataset(source)
            .rio.write_grid_mapping(inplace=True)
            .rio.write_crs("EPSG:4326", inplace=True)
        )
    else:
        ds = source

    # If the dataset already has a time dimension it's already flat
    if 'time' in ds.dims:
        return ds

    # Convert Julian day values to Python datetimes, then optionally
    # slice to the requested date window
    ds['day'] = convert_from_julian(ds['day'].values)
    if rain_data_start is not None or rain_data_end is not None:
        ds = ds.sel(day=slice(rain_data_start, rain_data_end))

    # Stack day and subday into a single time coordinate
    subdaily = ds.stack({'time': ['day', 'subday']})
    subday_seconds = (subdaily.subday.values * 86400).astype(int)
    new_index = (
        subdaily['day'].data
        + pd.to_timedelta(subday_seconds, unit='s')
    )
    subdaily = (
        subdaily
        .drop_vars(['time', 'day', 'subday'])
        .assign_coords(time=('time', new_index))
    )
    return subdaily


def convert_rainfall_depth_to_intensity(
    rainfall_depth_mm: xr.Dataset,
):
    """
    Convert rainfall depth (mm) to intensity (mm/h) by dividing by
    the timestep duration in hours.

    Parameters:
    - rainfall_depth_mm: xarray.Dataset with a 'rainfall' variable
      and a 'time' dimension at uniform sub-hourly resolution.

    Returns:
    - xarray.Dataset with 'rainfall' values in mm/h and the 'units'
      attribute updated accordingly.
    """
    # Sample the first 10 intervals to infer the timestep robustly
    TIMESTEP_SAMPLE_COUNT = 10
    timestamps = rainfall_depth_mm['time'].values
    timestep_hours = (
        (timestamps[TIMESTEP_SAMPLE_COUNT] - timestamps[0])
        / (np.timedelta64(1, 'h') * TIMESTEP_SAMPLE_COUNT)
    )
    timestep_hours = float(timestep_hours)

    result = rainfall_depth_mm / timestep_hours
    result['rainfall'].attrs['units'] = 'mm/h'
    return result


def aggregate_rainfall_data(
    source,
    rain_data_start=None,
    rain_data_end=None,
    time_res='30min',
):
    """
    Aggregate sub-daily stochastic rainfall to a coarser time resolution.

    Input data may be a NetCDF path or an xarray.Dataset in either
    pyraingen (day × subday × simulation) or flat (simulation × time)
    format.  The dataset is flattened if needed before resampling.

    Rainfall in mm is aggregated by summing; rainfall in mm/h is
    aggregated by averaging to preserve the correct intensity units.

    Parameters:
    - source: Path string to a NetCDF file, or an xarray.Dataset.
    - rain_data_start: Optional start date for slicing before
      aggregation.
    - rain_data_end: Optional end date for slicing before aggregation.
    - time_res: Resampling rule string accepted by xarray/pandas
      (e.g. '30min', '1h', 'D').  Default is '30min'.

    Returns:
    - xarray.Dataset or DataFrame of rainfall aggregated to the
      requested time resolution, one variable/column per simulation.
    """
    r_flat = flatten_pyraingen_rainfall(
        source, rain_data_start, rain_data_end
    )

    # resample() is called slightly differently for DataFrames vs
    # xarray Datasets
    if isinstance(r_flat, pd.DataFrame):
        r_agg = r_flat.resample(time_res)
    elif isinstance(r_flat, xr.Dataset):
        r_agg = r_flat.resample(time=time_res)
    else:
        print(
            f'r_flat is of type {type(r_flat)}. Assuming it is an '
            'xarray dataset and expects the time argument...'
        )
        r_agg = r_flat.resample({'time': time_res})

    units = r_flat['rainfall'].attrs['units']
    if units == 'mm':
        r_agg = r_agg.sum()
    elif units == 'mm/h':
        r_agg = r_agg.mean()
    else:
        raise ValueError(f"Unrecognized rainfall units: {units}")

    return r_agg


def _coerce_replicate_id(label):
    """
    Normalise a replicate coordinate label to an integer ID.

    Accepts plain ints, numeric strings ('0', '01'), and labelled
    strings like 'Simulation_0' or 'replicate_3'.

    Parameters:
    - label: Coordinate label to normalise.

    Returns:
    - Integer replicate ID extracted from the label.
    ------------------------------------------------------------------------
    Notes:
    - Raises ValueError if an integer cannot be extracted.
    ------------------------------------------------------------------------
    """
    if isinstance(label, (int, np.integer)):
        return int(label)
    s = str(label)
    trailing = s.rsplit('_', 1)[-1]
    try:
        return int(trailing)
    except ValueError as e:
        raise ValueError(
            f"Cannot extract integer replicate id from label "
            f"{label!r}"
        ) from e


def convert_rainfall_to_dataframe(source):
    """
    Pivot a rainfall Dataset to a time × replicate DataFrame.

    Columns are integer replicate IDs — matching the keys returned by
    load_ensemble_combined — regardless of whether the underlying
    xarray dimension is named 'replicate' or 'simulation' and whether
    its coordinate labels are integers or strings.

    Parameters:
    - source: xarray.Dataset with a 'rainfall' variable containing
      either a 'replicate' or 'simulation' dimension.

    Returns:
    - pandas DataFrame indexed by time, columns are integer replicate
      IDs sorted in ascending order.
    """
    replicate_dim = next(
        (
            d for d in ('replicate', 'simulation')
            if d in source['rainfall'].dims
        ),
        None,
    )
    if replicate_dim is None:
        raise ValueError(
            "Rainfall dataset has no 'replicate' or 'simulation' "
            f"dimension; got dims {source['rainfall'].dims}"
        )
    df = (
        source['rainfall']
        .to_dataframe()
        .reset_index()
        .pivot(
            index='time',
            columns=replicate_dim,
            values='rainfall',
        )
    )
    df.columns = [_coerce_replicate_id(col) for col in df.columns]
    df = df.reindex(columns=sorted(df.columns))
    df.attrs['units'] = source['rainfall'].attrs.get('units', 'unknown')
    return df


# ---------------------------------------------------------------------------
# Observed rainfall import
# ---------------------------------------------------------------------------

def import_measured_rainfall(
    excel_path: str,
    rain_col: str,
    out_time_res: str,
    datetime_col: str = 'Datetime',
    in_measure: str = 'intensity',
    in_units: str = 'mm/h',
    out_measure: str = 'depth',
    out_units: str = 'mm',
    mult_factor=1,
    save_daily_timeseries: bool = True,
    daily_ts_loc: str | None = None,
) -> pd.DataFrame:
    """
    Import observed rainfall from an Excel file and resample it.

    Reads a single-station Excel file, validates the requested unit
    conversions, resamples to the requested output resolution, and
    optionally saves a daily depth timeseries to disk.

    Parameters:
    - excel_path: Path to the Excel file containing timestamps and
      observed rainfall values.
    - rain_col: Name of the column holding rainfall values.
    - out_time_res: Output resampling rule (e.g. '30min', '12min'),
      as accepted by pd.DataFrame.resample().
    - datetime_col: Name of the datetime stamp column.
    - in_measure: Measurement type in the input file — 'depth' or
      'intensity'.
    - in_units: Units of the input measurement.  Must be 'mm' for
      depth or 'mm/h' for intensity.
    - out_measure: Whether the output values should be 'depth' or
      'intensity'.
    - out_units: Units for the output.  Must be 'mm' or 'mm/h'.
    - mult_factor: Factor to multiply rainfall values by before
      resampling.  Default is 1 (no scaling).
    - save_daily_timeseries: If True, save a daily depth timeseries
      CSV alongside the output.
    - daily_ts_loc: Directory path for the daily timeseries CSV.
      Required when save_daily_timeseries is True.

    Returns:
    - DataFrame with a DatetimeIndex and a single rainfall column at
      the requested measure, units, and time resolution.
    ------------------------------------------------------------------------
    Notes:
    - Currently coded for single-station Excel files only.
    ------------------------------------------------------------------------
    """
    input_meas = in_measure.lower().strip()
    input_unit = in_units.lower().strip()
    output_meas = out_measure.lower().strip()
    output_unit = out_units.lower().strip()

    allowed_measures = ['intensity', 'depth']
    allowed_units = ['mm', 'mm/h']

    # Validate that the requested measure/unit combinations are supported
    if input_meas not in allowed_measures:
        raise ValueError(
            'Imported rainfall values must be intensity or depth '
            f'measurements, but {in_measure} was requested.'
        )
    if input_unit not in allowed_units:
        raise ValueError(
            'Imported rainfall units must be either "mm" for depth '
            f'or "mm/h" for intensity. {in_units} were requested '
            f'for {in_measure}'
        )
    if output_meas not in allowed_measures:
        raise ValueError(
            f'Output was requested in {out_measure} values but must '
            f'be one of {allowed_measures}.'
        )
    if output_unit not in allowed_units:
        raise ValueError(
            f'Output was requested for {out_measure} in {out_units}, '
            'but must be either "depth" in "mm" or "intensity" in '
            '"mm/h"'
        )

    # Read the Excel file and verify that the required columns exist
    df = pd.read_excel(excel_path)
    if rain_col not in df.columns:
        raise ValueError(
            f'Could not find {rain_col} in columns of imported Excel '
            f'rainfall file. Actual columns: {df.columns}'
        )
    if datetime_col not in df.columns:
        raise ValueError(
            f'Could not find {datetime_col} in columns of imported '
            f'Excel rainfall file. Actual columns: {df.columns}'
        )

    # Set the datetime column as the index and apply the scale factor
    df2 = df.set_index(
        pd.to_datetime(df[datetime_col]), drop=True
    )[[rain_col]]
    df2[rain_col] *= mult_factor

    # Resample to the requested output frequency and measure
    df_out, new_col_name = resample_rainfall_timeseries(
        rainfall_data=df2,
        data_col_name=rain_col,
        out_time_res=out_time_res,
        out_measure=output_meas,
        input_measure=input_meas,
    )

    # Optionally save a daily depth timeseries for reference
    if save_daily_timeseries:
        df_daily, inter_col_name = resample_rainfall_timeseries(
            df_out,
            data_col_name=new_col_name,
            out_time_res='d',
            out_measure='depth',
            input_measure=input_meas,
        )
        df_daily = df_daily.rename(
            columns={inter_col_name: 'rain_depth_mm'}
        )
        name_ext = os.path.join(
            daily_ts_loc,
            c.RAIN_DAILY_DEPTH_TIMESERIES_NAME + '.csv',
        )
        df_daily.to_csv(name_ext)

    return df_out[[new_col_name]]


# ---------------------------------------------------------------------------
# Low-level depth / intensity conversion helpers
# ---------------------------------------------------------------------------

def get_stamps_per_hour(ts: pd.DataFrame) -> float:
    """
    Infer the number of timestamps per hour from a DataFrame's index.

    Parameters:
    - ts: DataFrame with a DatetimeIndex at a uniform sub-hourly
      resolution.

    Returns:
    - Float number of timestamps per hour (e.g. 2.0 for 30-minute
      data, 5.0 for 12-minute data).
    ------------------------------------------------------------------------
    Notes:
    - Uses the median inter-timestamp delta to be robust against
      isolated duplicates or gaps.
    - Raises ValueError if all timestamp deltas are zero.
    ------------------------------------------------------------------------
    """
    # Sort the index and compute inter-timestamp differences
    ts2 = ts.sort_index()
    deltas = ts2.index.to_series().diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]

    if deltas.empty:
        raise ValueError(
            'All timestamp deltas are 0 i.e. all timestamps are '
            'duplicates'
        )

    median_seconds = deltas.median().total_seconds()
    stamps_per_hour = 3600 / median_seconds
    return stamps_per_hour


def depth_to_intensity(depth: pd.DataFrame, depth_col: str) -> pd.Series:
    """
    Convert rainfall depth values to intensity in mm/h.

    Parameters:
    - depth: DataFrame with a DatetimeIndex at uniform sub-hourly
      resolution and a depth column in mm.
    - depth_col: Name of the column containing depth values.

    Returns:
    - Series of intensity values in mm/h.
    """
    hourly_steps = get_stamps_per_hour(depth)
    intensity = depth[depth_col] * hourly_steps
    return intensity


def intensity_to_depth(
    intensity: pd.DataFrame, int_col: str
) -> pd.Series:
    """
    Convert rainfall intensity values (mm/h) to per-timestep depth (mm).

    Parameters:
    - intensity: DataFrame with a DatetimeIndex at uniform sub-hourly
      resolution and an intensity column in mm/h.
    - int_col: Name of the column containing intensity values.

    Returns:
    - Series of rainfall depth values in mm per timestep.
    """
    hourly_steps = get_stamps_per_hour(intensity)
    depth = intensity[int_col] / hourly_steps
    return depth


def resample_rainfall_timeseries(
    rainfall_data: pd.DataFrame,
    data_col_name: str,
    out_time_res: str,
    out_measure: str,
    input_measure: str,
) -> tuple[pd.DataFrame, str]:
    """
    Resample rainfall to a requested frequency, converting measure if needed.

    Converts to depth first (if input is intensity), resamples by
    summing, then converts back to intensity if the output measure
    requires it.

    Parameters:
    - rainfall_data: DataFrame with a DatetimeIndex and a rainfall
      column.
    - data_col_name: Name of the rainfall column in rainfall_data.
    - out_time_res: Output resampling rule (e.g. '30min', 'D').
    - out_measure: Target measure — 'depth' or 'intensity'.
    - input_measure: Input measure — 'depth' or 'intensity'.

    Returns:
    - Tuple of (resampled_df, output_column_name) where
      output_column_name is either 'depth_mm' or 'intensity_mm_hr'.
    ------------------------------------------------------------------------
    Notes:
    - All intensity values are assumed to be in mm/h.
    - All depth values are assumed to be per-timestamp (not cumulative).
    ------------------------------------------------------------------------
    """
    depth_col_name = 'depth_mm'
    int_col_name = 'intensity_mm_hr'
    rain_data = rainfall_data.copy()

    # Convert intensity to depth before resampling so we can always
    # sum (summing intensity directly would give wrong units)
    if input_measure == 'intensity':
        rain_data[depth_col_name] = intensity_to_depth(
            rain_data, data_col_name
        )
    elif input_measure == 'depth':
        rain_data[depth_col_name] = rainfall_data[data_col_name]
    else:
        raise ValueError(
            'Input measure must be either "depth" or "intensity"; '
            f'received {input_measure}.'
        )

    # Aggregate to the requested frequency by summing depth
    rain_inter = (
        rain_data[[depth_col_name]].resample(out_time_res).sum()
    )

    if out_measure == 'depth':
        return rain_inter, depth_col_name
    elif out_measure == 'intensity':
        # Convert the resampled depth back to intensity at the new
        # (coarser) timestep
        rain_inter[int_col_name] = depth_to_intensity(
            rain_inter, depth_col_name
        )
        return rain_inter, int_col_name
    else:
        raise ValueError(
            'Output measure must be either "depth" or "intensity"; '
            f'{out_measure} was requested.'
        )
