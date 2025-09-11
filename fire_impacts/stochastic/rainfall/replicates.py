import requests
import pandas as pd
import numpy as np
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
    - DataFrame: DataFrame with datetime index and simulations as columns.
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
    return hg_to_data_frame(api_response.json())
