"""
Functions for requesting stochastic rainfall replicates from the
pyraingen API and reshaping the response into xarray Datasets.
"""

import logging
import os
import getpass
import uuid
import requests
import pandas as pd
import numpy as np
from fire_impacts.pre.project import FireImpactsProject
from fire_impacts.pre.util import read_raster
import xarray as xr
from ...pre.data_sources import STOCHASTIC_RAINFALL_API

logger = logging.getLogger(__name__)

# Environment variable name that opts a session into sending request IDs
REQUEST_ID_ENV_VAR = 'FIRE_IMPACTS_REQUEST_ID'


# ---------------------------------------------------------------------------
# Request ID helpers
# ---------------------------------------------------------------------------

def _request_id_enabled():
    """Return True if the opt-in env var is set to a truthy value."""
    return (
        os.environ.get(REQUEST_ID_ENV_VAR, '').strip().lower()
        in ('1', 'true', 'yes', 'on')
    )


def build_request_id(catchment=None):
    """
    Build an X-Request-ID header value identifying the caller and catchment.

    Includes the OS username, an optional catchment name, and a short
    UUID so that repeated calls remain distinguishable in server logs.

    Parameters:
    - catchment: Optional catchment name to embed in the ID string.

    Returns:
    - A forward-slash-separated string of the form
      'fire-impacts/<user>[/<catchment>]/<8-char-uuid>', with spaces
      replaced by underscores.  Returns None when the opt-in env var
      (FIRE_IMPACTS_REQUEST_ID) is not set.
    """
    if not _request_id_enabled():
        return None
    try:
        user = getpass.getuser()
    except Exception:
        user = 'unknown'
    parts = ['fire-impacts', user]
    if catchment:
        parts.append(str(catchment))
    parts.append(uuid.uuid4().hex[:8])
    return '/'.join(parts).replace(' ', '_')


# ---------------------------------------------------------------------------
# API response parsing
# ---------------------------------------------------------------------------

def decode_rle(rle):
    """
    Decode a run-length-encoded list into a flat numpy array.

    Parameters:
    - rle: List where each item is either a scalar value (implicit
      count of 1) or a [value, count] pair.

    Returns:
    - 1-D numpy array of decoded values.
    """
    values = []
    for entry in rle:
        if isinstance(entry, list) and len(entry) == 2:
            value, count = entry
        else:
            value, count = entry, 1
        values.extend([value] * count)
    return np.array(values)


def hg_to_data_frame(data):
    """
    Convert a pyraingen API response dict to a pandas DataFrame.

    Parameters:
    - data: Dict from the API JSON response, containing 'indexes'
      (time metadata) and 'timeseries' (RLE-encoded rainfall per
      simulation).

    Returns:
    - DataFrame indexed by datetime, one column per simulation,
      columns named 'Simulation_0', 'Simulation_1', etc.
    """
    index = data['indexes'][0]
    dates = pd.date_range(
        index['start'],
        periods=index['length'],
        freq=f'{index["step"]}s',
    )
    values = np.array([
        decode_rle(ts['values']) / ts.get('scale', 1.0)
        for ts in data['timeseries']
    ])
    result = pd.DataFrame(
        data=values.T,
        index=dates,
        columns=[f'Simulation_{i}' for i in range(len(values))],
    )
    return result


# ---------------------------------------------------------------------------
# API request functions
# ---------------------------------------------------------------------------

def get_replicates(
    lat,
    lon,
    elev,
    annual_rain,
    mean_temp,
    num_years,
    num_sims,
    api_url=STOCHASTIC_RAINFALL_API,
    request_id=None,
):
    """
    Request stochastic rainfall replicates from the pyraingen API.

    Parameters:
    - lat: Latitude of the target location (decimal degrees).
    - lon: Longitude of the target location (decimal degrees).
    - elev: Elevation in metres.
    - annual_rain: Mean annual rainfall in mm.  Pass None to let the
      API use its own climatological estimate.
    - mean_temp: Mean annual temperature in °C.  Pass None to let the
      API use its own estimate.
    - num_years: Length of each replicate in years.
    - num_sims: Number of independent replicates to generate.
    - api_url: Base URL for the stochastic rainfall API.
    - request_id: Optional X-Request-ID header value for server-side
      log correlation.  Build one with build_request_id().

    Returns:
    - xarray.Dataset with a 'rainfall' variable dimensioned
      (replicate, time), units attribute set to 'mm'.
    """
    logger.debug(
        f"Requesting {num_sims} replicates for "
        f"(lat={lat}, lon={lon}, elev={elev}), "
        f"annual_rain={annual_rain} mm, "
        f"mean_temp={mean_temp} °C, "
        f"num_years={num_years}"
    )

    params = dict(
        latitude=lat,
        longitude=lon,
        elevation=elev,
        length=num_years,
        count=num_sims,
    )
    if annual_rain is not None:
        params['mean_annual_rainfall'] = annual_rain
    if mean_temp is not None:
        params['mean_temperature'] = mean_temp

    headers = {}
    if request_id:
        headers['X-Request-ID'] = request_id
        logger.debug(f"Using X-Request-ID: {request_id}")

    api_response = requests.get(
        api_url,
        params=params,
        headers=headers or None,
        timeout=600,  # 10-minute timeout for large requests
    )
    logger.debug(
        f"API response status code: {api_response.status_code}"
    )
    assert api_response.status_code == 200, (
        f"API request failed with status code "
        f"{api_response.status_code}: {api_response.text}"
    )

    # Convert the JSON response to a DataFrame then wrap as xarray
    result = hg_to_data_frame(api_response.json())
    result_x = result.to_xarray().to_array()
    result_x.attrs['units'] = 'mm'
    result_x = result_x.rename(
        {'variable': 'replicate', 'index': 'time'}
    )
    result_x = xr.Dataset({'rainfall': result_x})
    return result_x


def get_rainfall_replicates(
    proj: FireImpactsProject,
    catchment,
    start=None,
    end=None,
    num_replicates=10,
    mean_annual_rainfall=None,
    average_temperature=None,
    num_years=None,
):
    """
    Get stochastic rainfall replicates for one or more project catchments.

    Derives the catchment centroid and mean elevation from project
    data, calls get_replicates(), then shifts the time axis by a whole
    number of years so the API output aligns with the requested
    calendar window.

    Parameters:
    - proj: FireImpactsProject instance.
    - catchment: Catchment name to process.  Pass None to run over all
      catchments in the project.
    - start: Start of the requested calendar window (str or
      pd.Timestamp).  When supplied with end, num_years is inferred.
    - end: End of the requested calendar window.  The result is sliced
      to [start, end] when both are given.
    - num_replicates: Number of rainfall replicates to request.
    - mean_annual_rainfall: Mean annual rainfall in mm for the
      catchment.  Passed to the API; if None the API uses its own
      estimate.
    - average_temperature: Mean annual temperature in °C.  Passed to
      the API; if None the API uses its own estimate.
    - num_years: Length of data in years to request from the API.
      Required when start/end are not both supplied.  When supplied
      alongside start+end it overrides the inferred length (must still
      be large enough to cover the requested span).

    Returns:
    - xarray.Dataset of rainfall replicates with the time axis shifted
      by whole years to align with start's calendar year, then sliced
      to [start, end] when both endpoints are provided.
    ------------------------------------------------------------------------
    Notes:
    - The API returns num_years of data anchored to an internal epoch
      year (first timestamp ~Dec 31 of the prior year per pyraingen's
      labelling convention).  Only whole-year shifts of the time axis
      are safe — partial-day shifts would silently move peak summer
      rainfall into the wrong season.
    - A one-year buffer is added to the API request to ensure the
      front edge of the requested window is not accidentally truncated
      after the shift-and-slice step.
    ------------------------------------------------------------------------
    """
    # When no specific catchment is given, recurse over all catchments
    if catchment is None:
        return proj.for_each_catchment(
            lambda c: get_rainfall_replicates(
                proj, c, start, end, num_replicates,
                mean_annual_rainfall, average_temperature, num_years,
            )
        )

    if num_replicates is None:
        raise ValueError("num_replicates must be specified.")

    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None

    # Work out how many years of API data to request.  We add one year
    # of buffer because pyraingen labels its first interval at ~Dec 31
    # of the prior year, so without the buffer the front edge of the
    # requested range can be silently dropped after the shift+slice.
    if start_ts is not None and end_ts is not None:
        span_years = end_ts.year - start_ts.year + 1
        required = span_years + 1
        if num_years is None:
            num_years = required
        elif num_years < required:
            raise ValueError(
                f"num_years={num_years} is too short to cover "
                f"{start_ts.date()}..{end_ts.date()} "
                f"(needs at least {required} including 1 year of "
                "anchor buffer)."
            )
    elif num_years is None:
        raise ValueError(
            "Provide either start+end (num_years inferred) "
            "or num_years."
        )

    # Extract catchment centroid coordinates and mean DEM elevation
    boundary = proj.catchment_boundary(catchment).to_crs(epsg=4326)
    centroid = boundary.geometry.centroid
    lat = centroid.y.values[0]
    lon = centroid.x.values[0]
    dem, _ = read_raster(
        proj.catchment_path(catchment, 'Topography', 'DEM.tif')
    )
    elev = np.nanmean(dem)

    rep = get_replicates(
        lat, lon, elev, mean_annual_rainfall, average_temperature,
        num_years, num_replicates,
        request_id=build_request_id(catchment),
    )

    # Shift the time axis by a whole number of years so the API output
    # covers the requested calendar window.  Only whole-year shifts are
    # safe — partial-day shifts would corrupt seasonality.
    #
    # Pick the largest year_offset such that the shifted first
    # timestamp still lands at or before start_ts; otherwise the
    # downstream slice would silently drop the front of the range.
    if start_ts is not None:
        api_index = rep.time.to_index()
        api_first = pd.Timestamp(api_index[0])
        year_offset = start_ts.year - api_first.year
        shifted_first = api_first + pd.DateOffset(years=year_offset)
        if shifted_first > start_ts:
            year_offset -= 1
            shifted_first = (
                api_first + pd.DateOffset(years=year_offset)
            )
        if year_offset != 0:
            # Rebuild the index from the shifted anchor using the
            # original step frequency.  Applying DateOffset(years=N)
            # to the whole index would collapse Feb 29 → Feb 28 and
            # produce duplicate timestamps; rebuilding from a single
            # anchor avoids that.
            step = api_index[1] - api_index[0]
            new_index = pd.date_range(
                shifted_first,
                periods=len(api_index),
                freq=step,
            )
            rep = rep.assign_coords(time=new_index)

    # Slice to the user's calendar window when both endpoints are given
    if start_ts is not None and end_ts is not None:
        rep = rep.sel(time=slice(start_ts, end_ts))

    return rep
