from datetime import datetime
import os
import datetime as dt
import xarray as xr
import pandas as pd
import numpy as np
from fire_impacts import const as c

def convert_from_julian(dates):
    from pyraingen.jdtodatevec import jdToDateVec
    return [datetime(*[int(c) for c in jdToDateVec(jd)]) for jd in dates]

def convert_to_julian(dates_array):
    """
    Convert an array of date strings to an array of Julian Dates
    --------------------------------------------------------------------
    Adapted from Jean Meeus' Astronomical Algorithms book with the 
    assistance of ChatGPT-5
    --------------------------------------------------------------------
    """
    # Convert to array of numpy datetime objects with day precision:
    d_arr = np.asarray(dates_array, dtype='datetime64[D]')

    # Number of whole months since Jan 1970:
    months_since_epoch = d_arr.astype('datetime64[M]').astype(int)
    # Number of whole years in gregorian terms:
    years_since_zero = (months_since_epoch // 12) + 1970

    # Get the actual 1-based month-of-year number
    months_leftover = (months_since_epoch % 12) + 1

    # Get the day of month using numpy timedelta objects:
    month_day_delta = d_arr - d_arr.astype('datetime64[M]')
    day_of_month = month_day_delta.astype(int) + 1
    
    # Apply a shift to Jan/Feb to help wiht leap year calcs:
    months_le2 = months_leftover <= 2
    shifted_years = years_since_zero - months_le2.astype(int)
    shifted_months = months_leftover + 12 * months_le2.astype(int)

    
    # Gregorian calendar correction:
    century = shifted_years // 100
    correction = 2 - century + (century // 4)

    # Constants from formula:
    days_per_year = 365.25
    years_bc = 4716
    days_per_month = 30.6001
    epoch_anchor = 1524.5

    # Actual equation:
    whole_year_days = np.floor(days_per_year * (shifted_years + years_bc))
    whole_month_days = np.floor(days_per_month * (shifted_months + 1))
    jd = (whole_year_days
          + whole_month_days
          + day_of_month
          + correction
          - epoch_anchor 
          ).astype(np.float64)
    
    return jd

def flatten_pyraingen_rainfall(source, rain_data_start=None, rain_data_end=None):
    '''
    Flattens subdaily stochastic rainfall data from the pyraingen conventions (day x subday x simulation) to a single
    long time series per simulation.

    Returned dataframe is simulation (columns) x time (rows), with rainfall values at subdaily resolution.
 
    Parameters:
    - source (str or xarray.Dataset): Path to the netCDF file containing rainfall data or an xarray Dataset.
    - rain_data_start (datetime): Start date for the rainfall data.
    - rain_data_end (datetime): End date for the rainfall data.

    Returns:
    - r_flat (DataFrame): DataFrame with flattened rainfall by simulation.
    '''
    if isinstance(source, str):
        ds = xr.open_dataset(source)\
          .rio.write_grid_mapping(inplace=True)\
          .rio.write_crs("EPSG:4326", inplace=True)
    else:
        ds = source

    if 'time' in ds.dims:
        return ds

    ds['day'] = convert_from_julian(ds['day'].values)
    if rain_data_start is not None or rain_data_end is not None:
        ds = ds.sel(day=slice(rain_data_start,rain_data_end))
    subdaily = ds.stack({'time':['day','subday']})
    subday_seconds = (subdaily.subday.values*86400).astype(int)
    new_index = subdaily['day'].data+pd.to_timedelta(subday_seconds,unit='s')
    subdaily = subdaily.drop_vars(['time','day','subday']).assign_coords(time=('time',new_index))
    return subdaily

def convert_rainfall_depth_to_intensity(rainfall_depth_mm:xr.Dataset):
    '''
    Converts rainfall depth (in mm) to intensity (in mm/h) by dividing by the duration (in hours).
    '''
    TIMESTEP_SAMPLE_COUNT=10
    timestamps = rainfall_depth_mm['time'].values
    timestep_hours = (timestamps[TIMESTEP_SAMPLE_COUNT] - timestamps[0]) / (np.timedelta64(1, 'h') * TIMESTEP_SAMPLE_COUNT)
    timestep_hours = float(timestep_hours)

    result = rainfall_depth_mm / timestep_hours
    result['rainfall'].attrs['units'] = 'mm/h'
    return result

def aggregate_rainfall_data(source, rain_data_start=None, rain_data_end=None, time_res='30min'):
    '''
    Aggregates subdaily stochastic rainfall data from a netCDF file to a specified time resolution.

    Input data is expected to be rainfall replicates (eg from pyraingen), with data stored by simulation x day x subday,
     or simulation x time. If the input data is in day x subday format, it is first flattened to a long time series per simulation.

    Returned dataframe is simulation (columns) x time (rows), with rainfall values aggregated to the specified time resolution.

    If the rainfall units are in mm, the aggregation is done by summing the rainfall over the time period.
    If the rainfall units are in mm/h, the aggregation is done by averaging the rainfall over the time period.

    Parameters:
    - source (str or xarray.Dataset): Path to the netCDF file containing rainfall data or an xarray Dataset.
    - rain_data_start (datetime): Start date for the rainfall data.
    - rain_data_end (datetime): End date for the rainfall data.
    - time_res (str): Time resolution for the aggregated data. Default is '30min'.

    Returns:
    - r_agg (DataFrame): DataFrame with aggregated rainfall by simulation.
    '''
    r_flat = flatten_pyraingen_rainfall(source, rain_data_start, rain_data_end)

    # Handle call to resample() slightly differently whether we've got 
    #a dataframe or xarray:
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

def convert_rainfall_to_dataframe(source):
    df = source['rainfall'].to_dataframe().reset_index().pivot(index='time',columns='simulation',values='rainfall')
    df.attrs['units'] = source['rainfall'].attrs.get('units','unknown')
    return df

def import_measured_rainfall(
    excel_path:str,
    rain_col:str,
    out_time_res:str,
    datetime_col:str='Datetime',
    in_measure:str='intensity',
    in_units:str='mm/h',
    out_measure:str='depth',
    out_units:str='mm',
    mult_factor=1,
    save_daily_timeseries:bool=True,
    daily_ts_loc:str|None=None
    ) -> pd.DataFrame:
    """
    Import an excel file of rainfall observations

    Parameters:
    - excel_path (str): path to an excel file with timestamps and 
    observed rainfall values
    - rain_col (str): name of the column in the excel file holding the
    rainfall values
    - out_time_res (str): Time resolution desired for the resulting 
    dataframe. Should match the formats expected by the rule argument 
    for pd.DataFrame.resample() e.g. '30min' '12min' 
    - datetime_col (str): Name of the date-time stamp column in the 
    excel file
    - in_measure (str): Measurement used in the input data. Should be 
    either 'depth' or 'intensity'
    - in_units (str): Units of the input measurement. For depth it 
    should always be mm, and intensity should always be mm/h
    - out_measure (str): whether the output values should be depth or 
    intensity
    - out_units (str): units of output values; should be mm for depth 
    or mm/h for intensity
    - mult_factor (float): For testing purposes, factor by which the 
    rainfall values should be multiplied. Mainly used to force debris 
    flow events for testing, should be left at 1 in most cases
    - save_daily_timeseries (bool): whether the timeseries should be saved to 
    the Results folder

    Returns:
    - rain_data (DataFrame): Dataframe with a DateTime index and 
    rainfall values for the requested measure/units and resampled to 
    the requested frequency.
    --------------------------------------------------------------------
    Notes:
    - Currently only coding this for a single-station Excel file
    --------------------------------------------------------------------
    """
    input_meas = in_measure.lower().strip()
    input_unit = in_units.lower().strip()
    output_meas = out_measure.lower().strip()
    output_unit = out_units.lower().strip()

    allowed_measures = ['intensity', 'depth']
    allowed_units = ['mm', 'mm/h']

    # Checks to make sure request has been coded for:
    if input_meas not in allowed_measures:
        raise ValueError(
            'Imported rainfall values must be intensity or depth '
            f'measurements, but {in_measure} was requested.'
            )
    if input_unit not in allowed_units:
        raise ValueError(
            'imported rainfall units must be either "mm" for depth or '
            f'"mm/h" for intensity. {in_units} were requested for '
            f'{in_measure}'
            )
    if output_meas not in allowed_measures:
        raise ValueError(
            f'Output was requested in {out_measure} values but must be '
            f'one of {allowed_measures}.'
            )
    if output_unit not in allowed_units:
        raise ValueError(
            f'Output was requested for {out_measure} in {out_units}, '
            'but must be either "depth" in "mm" or "intensity" in '
            '"mm/h"'
            )
    
    # Read in excel and make sure it has the relevant columns:
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
    
    # Convert the datetime column to datetime objects and make it the 
    #index.
    df2 = df.set_index(pd.to_datetime(df[datetime_col]), drop=True)[[rain_col]]
    # Multiply the rainfall values by the requested factor:
    df2[rain_col] *= mult_factor
    
    # Resample to the requested frequency:
    df_out, new_col_name = resample_rainfall_timeseries(
        rainfall_data=df2,
        data_col_name=rain_col,
        out_time_res=out_time_res,
        out_measure=output_meas,
        input_measure=input_meas
        )
    
    if save_daily_timeseries:
        df_daily, inter_col_name = resample_rainfall_timeseries(
            df_out,
            data_col_name=new_col_name,
            out_time_res='d',
            out_measure='depth',
            input_measure=input_meas
            )
        df_daily = df_daily.rename(columns={inter_col_name: 'rain_depth_mm'})
        name_ext = os.path.join(
            daily_ts_loc,
            c.RAIN_DAILY_DEPTH_TIMESERIES_NAME + '.csv'
            )
        df_daily.to_csv(name_ext)
    
    return df_out[[new_col_name]]

###############################################################################
def get_stamps_per_hour(ts:pd.DataFrame) -> float:
    """
    Look at a Datetime index and extract the number of timestamps per 
    hour
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Ensure sorted index:
    ts2 = ts.sort_index()
    
    # Get the time differences ignoring na values:
    deltas = ts2.index.to_series().diff().dropna()
    # Exclude any 0-delta (duplicate) timesteps:
    deltas = deltas[deltas > pd.Timedelta(0)]
    # Throw an error if there are no deltas > 0:
    if deltas.empty:
        raise ValueError(
            'All timestamp deltas are 0 i.e. all timestamps are '
            'duplicates'
            )

    # Get the number of seconds:
    median_seconds = deltas.median().total_seconds()

    # Convert from seconds to hours:
    stamps_per_hour = 3600 / median_seconds

    return stamps_per_hour

###############################################################################
def depth_to_intensity(depth:pd.DataFrame, depth_col:str) -> pd.Series:
    """
    --------------------------------------------------------------------
    Notes:
    - Returns intensity values in mm/hr
    --------------------------------------------------------------------
    """
    hourly_steps = get_stamps_per_hour(depth)
    intensity = depth[depth_col] * hourly_steps
    return intensity

###############################################################################
def intensity_to_depth(intensity:pd.DataFrame, int_col:str) -> pd.Series:
    """
    --------------------------------------------------------------------
    Notes:
    - Assumes intensity values are in mm/hr.
    --------------------------------------------------------------------
    """
    hourly_steps = get_stamps_per_hour(intensity)
    depth = intensity[int_col] / hourly_steps
    return depth

###############################################################################
def resample_rainfall_timeseries(
    rainfall_data:pd.DataFrame,
    data_col_name:str,
    out_time_res:str,
    out_measure:str,
    input_measure:str
    ) -> tuple[pd.DataFrame, str]:
    """
    Resample rainfall values to the requested frequency and return a 
    timeseries dataframe
    --------------------------------------------------------------------
    Assumptions:
    - Intensity values are all in mm/hr
    - Depth values are all per-timestamp and not cumulative
    --------------------------------------------------------------------
    """
    depth_col_name = 'depth_mm'
    int_col_name = 'intensity_mm_hr'
    rain_data = rainfall_data.copy()
    if input_measure == 'intensity':
        # Convert to depth first before resampling:
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

    # Get just depth values as the requested output frequency:
    rain_inter = rain_data[[depth_col_name]].resample(out_time_res).sum()

    # If the target measure is depth we're already there:
    if out_measure == 'depth':
        return rain_inter, depth_col_name
    # Otherwise we need to convert back to intensity:
    elif out_measure == 'intensity':
        # Multiply the depth measured for each timestamp
        rain_inter[int_col_name] = depth_to_intensity(
            rain_inter,
            depth_col_name
            )
        return rain_inter, int_col_name
    else:
        raise ValueError(
            'Output measure must be either "depth" or "intensity"; '
            f'{out_measure} was requested.'
            )
    





    
