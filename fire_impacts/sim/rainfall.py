from datetime import datetime
import datetime as dt
import xarray as xr
import pandas as pd
import numpy as np

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


def aggregate_rainfall_data(netcdf_path, rain_data_start, rain_data_end, time_res='30min'):
    '''
    Aggregates subdaily stochastic rainfall data from a netCDF file to a specified time resolution.

    Input data is expected to be from pyraingen, with data stored by simulation x day x subday.

    Returned dataframe is simulation (columns) x time (rows), with rainfall values aggregated to the specified time resolution.

    Parameters:
    - netcdf_path (str): Path to the netCDF file containing rainfall data.
    - rain_data_start (datetime): Start date for the rainfall data.
    - rain_data_end (datetime): End date for the rainfall data.
    - time_res (str): Time resolution for the aggregated data. Default is '30min'.

    Returns:
    - r_agg (DataFrame): DataFrame with aggregated rainfall by simulation.
    '''

    ds = xr.open_dataset(netcdf_path)\
      .rio.write_grid_mapping(inplace=True)\
      .rio.write_crs("EPSG:4326", inplace=True)
    ds['day'] = convert_from_julian(ds['day'].values)
    ds = ds.sel(day=slice(rain_data_start,rain_data_end))

    subdaily = ds.stack({'time':['day','subday']})

    subday_seconds = (subdaily.subday.values*86400).astype(int)
    new_index = subdaily['day'].data+pd.to_timedelta(subday_seconds,unit='s')
    subdaily = subdaily.drop_vars(['time','day','subday']).assign_coords(time=('time',new_index))
    df = subdaily['rainfall'].to_dataframe()
    rainfall_by_simulation = df.reset_index().pivot(index='time',columns='simulation',values='rainfall')

    r_agg = rainfall_by_simulation.resample(time_res).sum()

    return r_agg

def import_measured_rainfall(
    excel_path:str,
    rain_col:str,
    datetime_col:str='Datetime',
    attributes:dict=None
    ):
    """
    Import an excel file of rainfall observations into an xarray in the 
    format the module is expecting.

    Parameters:
    - excel_path (str): path to an excel file with timestamps and 
    observed rainfall values (mm)
    - rain_col (str): name of the column in the excel file holding the
    rainfall values
    - datetime_col (str): Name of the date-time stamp column in the 
    excel file
    - attributes (dict): Dictionary of metadata attributes describing
    the dataset

    Returns:
    - rain_array (xarray Dataset): an xarray which mimics the pyraingen
    output that aggregate_rainfall_data is expecting.
    --------------------------------------------------------------------
    Notes:
    - Currently only coding this for a single-station Excel file
    --------------------------------------------------------------------
    """
    df = pd.read_excel(excel_path)
    df['day'] = df[datetime_col].dt.strftime('%Y-%m-%d')
    df['subday'] = df[datetime_col].dt.strftime('%H:%M')
    df['simulation'] = 0

    # Convert day to numpy, then to Julian date:
    day_arr = df['day'].to_numpy(dtype=str)
    julian_day_arr = convert_to_julian(day_arr)
    df['day'] = julian_day_arr

    # Convert subday to numpy for efficient vectorised operations:
    subd_arr = df['subday'].to_numpy(dtype=str)

    # Extract fraction of day for subday:
    h, _, m = np.strings.partition(subd_arr, ':')
    sec_in_day = 24 * 60 * 60
    hour_secs = h.astype(int) * 60 * 60
    minute_secs = m.astype(int) * 60
    tot_secs = hour_secs + minute_secs
    subday_dim_coord = (tot_secs / sec_in_day).astype(np.float64)
    df['subday'] = subday_dim_coord

    # Convert the dimensional columns to a MultiIndex in prep for
    #converting to xarray:
    df.index = pd.MultiIndex.from_frame(
        df[['day', 'subday', 'simulation']]
        )

    # Get rid of all columns but the actual rainfall one, because their
    #values are now stored in the MultiIndex:
    df2 = df.drop(columns=['day', 'subday', 'simulation', datetime_col])

    # Use groupby to get rid of any duplicates in the MultiIndex. There
    #shouldn't theoretically be any, but if there are two or more 
    #records for the same day/subday/simulation, this will convert to
    #just one record with the mean rainfall value of the inputs:
    df3 = df2.groupby(level=df2.index.names, sort=False).mean()
    
    if attributes is not None:
        attribute_dict = attributes
    else:
        hist = (
            'Initial receipt date unknown. '
            f'Converted to xarray {datetime.now()}'
        )
        unspec = 'Not provided'
        attribute_dict = {
            'Title': unspec,
            'History': hist,
            'Source': unspec,
            'Institution': unspec,
            'Conventions': unspec 
            }

    # Convert to an xarray Dataset with the MultiIndex levels as 
    #dimenstions:
    ds = df3.to_xarray()
    
    # Adjust so simulation is a non-coordinate dimension to match 
    #expected pyraingen output:
    ds = ds.drop_indexes('simulation')
    ds = ds.reset_coords('simulation', drop=True)


    ds.attrs = attribute_dict
    return ds