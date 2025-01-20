from datetime import datetime
import xarray as xr
import pandas as pd

def convert_from_julian(dates):
    from pyraingen.jdtodatevec import jdToDateVec
    return [datetime(*[int(c) for c in jdToDateVec(jd)]) for jd in dates]

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
    - r30 (DataFrame): DataFrame with aggregated rainfall
    '''

    ds = xr.open_dataset(netcdf_path)\
      .rio.write_grid_mapping(inplace=True)\
      .rio.write_crs("EPSG:4326", inplace=True)
    ds['day'] = convert_from_julian(ds['day'].values)
    ds = ds.sel(day=slice(rain_data_start,rain_data_end))

    subdaily = ds.stack({'time':['day','subday']})

    subday_seconds = (subdaily.subday.values*86400).astype(int)

    subdaily = subdaily.assign_coords(time=('time',subdaily['day'].data+pd.to_timedelta(subday_seconds,unit='s')))
    df = subdaily['rainfall'].to_dataframe()
    rainfall_by_simulation = df.reset_index().pivot(index='time',columns='simulation',values='rainfall')

    r_agg = rainfall_by_simulation.resample(time_res).sum()

    return r_agg
