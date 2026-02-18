import requests
import pandas as pd
import numpy as np
from fire_impacts.pre.project import FireImpactsProject
from fire_impacts.pre.util import read_raster
import xarray as xr
from ...pre.data_sources import STOCHASTIC_RAINFALL_API

def decode_rle(rle):
    values = []
    for entry in rle:
        if isinstance(entry, list) and len(entry) == 2:
            value, count = entry
        else:
            value, count = entry, 1
        values.extend([value] * count)
    return np.array(values)

def hg_to_data_frame(data):
    index = data['indexes'][0]
    dates = pd.date_range(index['start'], periods=index['length'], freq=f'{index["step"]}s')
    values = np.array([decode_rle(ts['values'])/ts.get('scale',1.0) for ts in data['timeseries']])
    result = pd.DataFrame(data=values.T, index=dates, columns=[f'Simulation_{i}' for i in range(len(values))])
    return result

def get_replicates(lat,lon,elev,annual_rain,mean_temp,num_years,num_sims,api_url=STOCHASTIC_RAINFALL_API):
    '''
    Get stochastic rainfall replicates from the API and return as a DataFrame.

    Parameters:
    - lat (float): Latitude of the location.
    - lon (float): Longitude of the location.
    - elev (float): Elevation in meters.
    - annual_rain (float): Mean annual rainfall in mm.
    - mean_temp (float): Mean temperature in °C.
    - num_years (int): Length of data in years.
    - num_sims (int): Number of samples/replicates.
    - api_url (str): URL of the stochastic rainfall API. Default is STOCHASTIC_RAINFALL_API.

    Returns:
    - Dataset: XArray dataset with datetime index and simulations as columns.
    '''
    api_response = requests.get(
        api_url,
        params=dict(
            latitude=lat,
            longitude=lon,
            elevation=elev,
            mean_annual_rainfall=annual_rain,
            mean_temperature=mean_temp,
            length=num_years,
            count=num_sims),
        timeout=600 # 10 minutes
    )
    assert api_response.status_code==200
    result = hg_to_data_frame(api_response.json())
    result_x = result.to_xarray().to_array()
    result_x.attrs['units']='mm'
    result_x = result_x.rename({'variable':'replicate','index':'time'})
    result_x = xr.Dataset({'rainfall':result_x})
    return result_x

###############################################################################
def get_rainfall_replicates(
    proj:FireImpactsProject,
    catchment,
    start,
    end,
    num_replicates,
    mean_annual_rainfall,
    average_temperature,
    num_years=2
    ):
    """
    Get stochastic rainfall replicates for one or more catchments in 
    the project.

    Parameters:
    - proj (FireImpactsProject): A dictionary of project folders 
    created for catchments.
    - catchment (str): OPTIONAL: Name of the catchment to process. If 
    None, process all catchments.
    - start (str): Start date for the rainfall data.
    - end (str): End date for the rainfall data.
    - num_replicates (int): Number of rainfall replicates to generate.
    - mean_annual_rainfall (float): Mean annual rainfall in mm for the 
    catchment.
    - average_temperature (float): Average temperature in °C for the 
    catchment.
    - num_years (int): Length of data in years. Default is 2.

    Returns:
    - Dataset: XArray dataset with datetime index and simulations as 
    columns.
    --------------------------------------------------------------------
    Notes:
    - Infer location and elevation from catchment boundary and DEM. 
    User supplied climate statistics are used to generate the 
    replicates.
    --------------------------------------------------------------------
    """
    if catchment is None:
        return proj.for_each_catchment(lambda c: get_rainfall_replicates(
            proj, c, start, end, num_replicates, mean_annual_rainfall, average_temperature, num_years))

    boundary = proj.catchment_boundary(catchment).to_crs(epsg=4326)
    centroid = boundary.geometry.centroid
    lat = centroid.y.values[0]
    lon = centroid.x.values[0]
    dem,_ = read_raster(
        proj.catchment_path(
            catchment,
            'Topography',
            'DEM.tif'
            )
        )
    elev = np.nanmean(dem)
    rep = get_replicates(
        lat, lon, elev, mean_annual_rainfall, average_temperature, num_years, num_replicates)
    return rep

