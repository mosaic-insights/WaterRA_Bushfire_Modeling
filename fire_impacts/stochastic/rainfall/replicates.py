import logging
import requests
import pandas as pd
import numpy as np
from fire_impacts.pre.project import FireImpactsProject
from fire_impacts.pre.util import read_raster
import xarray as xr
from ...pre.data_sources import STOCHASTIC_RAINFALL_API
logger = logging.getLogger(__name__)

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
    logger.debug(f"Requesting {num_sims} replicates for location (lat: {lat}, lon: {lon}, elev: {elev}) with annual rainfall {annual_rain} mm and mean temperature {mean_temp} °C for {num_years} years.")
    params=dict(
        latitude=lat,
        longitude=lon,
        elevation=elev,
        length=num_years,
        count=num_sims
    )
    if annual_rain is not None:
        params['mean_annual_rainfall'] = annual_rain
    if mean_temp is not None:
        params['mean_temperature'] = mean_temp

    api_response = requests.get(
        api_url,
        params=params,
        timeout=600 # 10 minutes
    )
    logger.debug(f"API response status code: {api_response.status_code}")
    assert api_response.status_code==200, f"API request failed with status code {api_response.status_code}: {api_response.text}"
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
    start=None,
    end=None,
    num_replicates=10,
    mean_annual_rainfall=None,
    average_temperature=None,
    num_years=None,
    ):
    """
    Get stochastic rainfall replicates for one or more catchments in
    the project.

    Parameters
    ----------
    proj : FireImpactsProject
    catchment : str or None
        Catchment to process.  If *None*, process all catchments.
    start, end : str or pd.Timestamp, optional
        Calendar window for the returned series.  When both are given,
        *num_years* is inferred (one API year per calendar year spanned)
        and the result is sliced to ``[start, end]``.  When omitted,
        the API output is returned unshifted starting Jan 1 of an
        arbitrary year — pass *num_years* in that case.
    num_replicates : int
        Number of rainfall replicates to request.
    mean_annual_rainfall : float, optional
        Mean annual rainfall (mm) for the catchment.
    average_temperature : float, optional
        Average temperature (°C) for the catchment.
    num_years : int, optional
        Length of data in years to request from the API.  Required
        when *start* / *end* are not both supplied.  When supplied
        alongside *start* + *end*, overrides the inferred length (must
        still be large enough to cover the requested span).

    Returns
    -------
    xarray.Dataset
        Rainfall replicates.  Time axis is shifted by whole years
        (preserving seasonality) so that Jan 1 of the API output
        aligns with Jan 1 of *start*'s year, then sliced to
        ``[start, end]`` when both are given.

    Notes
    -----
    The stochastic rainfall API returns *num_years* of data starting
    Jan 1 of an internal epoch year.  Because seasonality is
    January-anchored, only whole-year shifts of the time axis are safe.
    Earlier versions shifted by ``start - api_first`` (any number of
    days), which silently moved e.g. peak summer rainfall into autumn
    when the user requested a non-Jan-1 start.
    """
    if catchment is None:
        return proj.for_each_catchment(lambda c: get_rainfall_replicates(
            proj, c, start, end, num_replicates, mean_annual_rainfall,
            average_temperature, num_years))

    if num_replicates is None:
        raise ValueError("num_replicates must be specified.")

    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None

    if start_ts is not None and end_ts is not None:
        # One API year per calendar year spanned (e.g. 2000-02-01 →
        # 2002-01-31 spans 2000, 2001, 2002 → 3 years).
        span_years = end_ts.year - start_ts.year + 1
        if num_years is None:
            num_years = span_years
        elif num_years < span_years:
            raise ValueError(
                f"num_years={num_years} is too short to cover "
                f"{start_ts.date()}..{end_ts.date()} "
                f"(needs at least {span_years})."
            )
    elif num_years is None:
        raise ValueError(
            "Provide either start+end (num_years inferred) or num_years."
        )

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
        lat, lon, elev, mean_annual_rainfall, average_temperature,
        num_years, num_replicates)

    # When the user supplied a start date, shift the time axis by a
    # whole number of years so Jan 1 of the API output aligns with
    # Jan 1 of the start year.  Whole-year shifts preserve seasonality
    # — partial-day shifts do not.
    if start_ts is not None:
        api_first = pd.Timestamp(rep.time.values[0])
        year_offset = start_ts.year - api_first.year
        if year_offset != 0:
            new_index = rep.time.to_index() + pd.DateOffset(years=year_offset)
            rep = rep.assign_coords(time=new_index)

    # Slice to the user's calendar window when both endpoints are given.
    if start_ts is not None and end_ts is not None:
        rep = rep.sel(time=slice(start_ts, end_ts))

    return rep

