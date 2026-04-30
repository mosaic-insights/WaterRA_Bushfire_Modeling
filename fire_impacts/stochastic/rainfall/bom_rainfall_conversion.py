"""
Prepare BOM daily and sub-daily rainfall data for use with PyRainGen.

Key functions:
- convert_daily: Convert daily rainfall data files to PyRainGen format.
- convert_sub_daily: Convert sub-daily rainfall data files to PyRainGen
  format.
"""

from pathlib import Path
import xarray as xr
import geopandas as gpd
import rioxarray
import pandas as pd
from shapely.geometry import mapping
import numpy as np
import dask
import os
from datetime import datetime
from glob import glob
import netCDF4 as nc
import datetime
import re
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory and path helpers
# ---------------------------------------------------------------------------

def ensure_dest(dest, clear=False):
    """
    Create a destination directory, optionally clearing it first.

    Parameters:
    - dest: Path string or Path object for the target directory.
    - clear: If True, delete and recreate the directory before use.

    Returns:
    - None
    """
    if clear and os.path.exists(dest):
        logger.info('Clearing directory: %s', dest)
        import shutil
        shutil.rmtree(dest)
    if not os.path.exists(dest):
        logger.info('Creating directory: %s', dest)
        os.makedirs(dest, exist_ok=True)


def get_top_level(src_data):
    """
    Return an iterable of top-level paths from a flexible input type.

    Parameters:
    - src_data: One of: a glob string (expanded to a list of Paths),
      a Path-like object with an iterdir() method, or any other
      iterable of Path objects (returned as-is).

    Returns:
    - Iterable of pathlib.Path objects.
    """
    if isinstance(src_data, str):
        return [Path(p) for p in glob(src_data)]
    elif hasattr(src_data, 'iterdir'):
        return src_data.iterdir()
    return src_data


# ---------------------------------------------------------------------------
# Daily rainfall conversion
# ---------------------------------------------------------------------------

def extract_and_format_daily_rainfall(file, dest):
    """
    Parse a BOM daily rainfall CSV and write it in PyRainGen format.

    Reads a single BOM station CSV, extracts the date and precipitation
    columns, and writes a fixed-width text file in the format expected
    by PyRainGen's daily-rainfall input.  The output filename is
    derived from the station number embedded in the source filename.

    Parameters:
    - file: pathlib.Path to the BOM daily rainfall CSV file.
    - dest: pathlib.Path to the output directory.

    Returns:
    - None.  If the file contains no valid data rows, returns early
      without writing an output file.
    """
    file_name = file.name
    # Extract station number from the BOM filename convention
    station = file_name.split("_")[2]
    output_file = dest / f"rev_dr{station}.txt"

    formatted_data = []

    with open(file, "r") as infile:
        next(infile)  # Skip header row
        for line in infile:
            parts = line.split(",")
            try:
                year = parts[2].strip()
                month = parts[3].strip()
                day = parts[4].strip()
                precip = parts[5].strip()
                formatted_data.append(
                    [int(year), int(month), int(day), float(precip)]
                )
            except (IndexError, ValueError):
                continue  # Skip any malformed lines

    if not formatted_data:
        return

    formatted_array = np.array(formatted_data)

    # Determine first year and total year count for the file header
    unique_years = np.unique(formatted_array[:, 0])
    first_year = int(unique_years[0])
    number_of_years = len(unique_years)

    np.savetxt(
        output_file,
        formatted_array,
        fmt="%7d %4d %4d %9.2f",
        header=(
            f" Daily rainfall at station dr{station}"
            f"        {first_year}      {number_of_years}"
        ),
        comments="",
    )


def convert_daily(src_data, dest):
    """
    Convert BOM daily rainfall CSV files to PyRainGen format in parallel.

    Walks src_data for .txt files (excluding Notes and StnDet files),
    schedules each one via dask.delayed, and processes them with the
    'processes' scheduler for parallelism.

    Parameters:
    - src_data: Source directory, glob string, or list of directories
      containing BOM daily rainfall CSV files.
    - dest: Destination directory path for the converted output files.

    Returns:
    - None
    """
    ensure_dest(dest)
    jobs = []
    extract_delayed = dask.delayed(extract_and_format_daily_rainfall)

    for folder in get_top_level(src_data):
        # Collect .txt files, skipping BOM metadata files
        if folder.is_dir():
            matches = folder.glob("*.txt")
        else:
            matches = [folder]
        matches = [
            m for m in matches
            if m.is_file()
            and 'Notes' not in m.name
            and 'StnDet' not in m.name
            and m.name.endswith(".txt")
        ]
        if not matches:
            logger.debug('No valid files found in %s', folder)
            continue

        logger.info('Found %d files in %s', len(matches), folder)
        for file in matches:
            try:
                jobs.append(extract_delayed(file, dest))
            except Exception:
                logger.error('Error processing %s', str(file))
                raise

    logger.info('Processing %d files', len(jobs))
    dask.compute(*jobs, scheduler='processes')
    logger.info('Done!')


# ---------------------------------------------------------------------------
# Sub-daily rainfall conversion
# ---------------------------------------------------------------------------

def extract_and_format_sub_daily_rainfall(file, dest):
    """
    Parse a BOM pluviograph text file and write it to a sub-daily NetCDF.

    Reads a BOM sub-daily rainfall text file, standardises the date
    format, decodes the per-6-minute-interval rainfall values, and
    writes a NetCDF4 file in the ARR Project 4 schema via
    produceSubDailyNetCDF.

    Parameters:
    - file: pathlib.Path to the BOM sub-daily rainfall text file.
      The filename must follow the pattern 'outputFor<station>_...'.
    - dest: pathlib.Path to the output directory for NetCDF files.

    Returns:
    - None.  Files with unexpected station-number lengths are skipped
      with a log message.
    """
    file_name = file.name

    # Derive the output filename from the station number, which is
    # embedded in the source filename between 'outputFor' and '_'.
    st = file_name.index("outputFor") + len("outputFor")
    en = file_name.index("_")
    station = file_name[st:en]

    if len(station) == 4:
        output_file = dest / f"plv00{station}.nc"
    elif len(station) == 5:
        output_file = dest / f"plv0{station}.nc"
    elif len(station) == 6:
        output_file = dest / f"plv{station}.nc"
    else:
        logger.info(
            f"Skipping file {file_name}: "
            f"Unexpected station number format"
        )
        return

    with open(file, 'r') as sub:
        lines = sub.readlines()

    lines = lines[1:]  # Skip the header row
    data = []
    for line in lines:
        line = line.strip()
        station = line[:6]
        date = line[8:20].strip()
        rain = line[20:].strip()
        data.append([station, date, rain])

    df = pd.DataFrame(data, columns=['Station', 'Date', 'Rain'])
    df = df.iloc[1:, :]

    # Standardise the date strings to YYYYMMDD[HH[MM]] format
    def standardize_date(date_str):
        date_str = date_str.strip()
        if " " in date_str:
            parts = date_str.split()
            if len(parts) == 2:
                return f"{parts[0]}{int(parts[1]):02d}"
            elif len(parts) == 3:
                return (
                    f"{parts[0]}"
                    f"{int(parts[1]):02d}"
                    f"{int(parts[2]):02d}"
                )
        return date_str

    df['StandardizedDate'] = df['Date'].apply(standardize_date)
    df['DateFormatted'] = pd.to_datetime(
        df['StandardizedDate'], format='%Y%m%d'
    )

    # Parse each row's rainfall string into a numpy array of per-
    # interval values, replacing negative BOM flag values with -9999.
    def parse_rainfall(row):
        row = re.sub(r'-\d+\.\d+', '-9999.0', row)
        cleaned_row = re.sub(
            r'(?<!\s)(-9999.0|-8888.0|-1544.4|-1872.0|-1132.0)',
            r' \1',
            row,
        )
        cleaned_row = " ".join(cleaned_row.split())
        rain_values = np.array(
            cleaned_row.split(), dtype=np.float32
        )
        return rain_values

    df['RainArray'] = df['Rain'].apply(parse_rainfall)
    df['DateFormatted'] = pd.to_datetime(df['DateFormatted'])

    # Build a continuous daily date range and map data onto it,
    # leaving gaps filled with the -9999 missing-value sentinel.
    full_date_range = pd.date_range(
        start=df['DateFormatted'].iloc[0],
        end=df['DateFormatted'].iloc[-1],
        freq='D',
    )
    dayVector = full_date_range.to_julian_date().values
    rainfall_dict = dict(zip(
        df['DateFormatted'].apply(lambda x: x.to_julian_date()),
        df['RainArray'],
    ))

    # 240 intervals of 6 minutes each covers one full day
    n_intervals_per_day = 240
    subDailyData = np.full(
        (1, len(dayVector), n_intervals_per_day),
        -9999.0,
        dtype=np.float32,
    )
    for i, julian_day in enumerate(dayVector):
        if julian_day in rainfall_dict:
            subDailyData[0, i, :] = rainfall_dict[julian_day]

    produceSubDailyNetCDF(
        fnameNC=output_file,
        subDailyData=subDailyData,
        dayVector=dayVector,
        title="Sub-Daily Rainfall Dataset",
        institution="Your Institution Name",
    )


def convert_sub_daily(src_data, dest, clear=False):
    """
    Convert BOM sub-daily rainfall files to NetCDF format in parallel.

    Walks src_data for files matching the 'outputFor*_*' pattern,
    schedules each one via dask.delayed, and processes them with the
    'processes' scheduler for parallelism.

    Parameters:
    - src_data: Source directory, glob string, or list of directories
      containing BOM sub-daily rainfall text files.
    - dest: Destination directory path for the converted NetCDF files.
    - clear: If True, clear the destination directory before processing.

    Returns:
    - None
    """
    ensure_dest(dest, clear)
    jobs = []
    extract_delayed = dask.delayed(extract_and_format_sub_daily_rainfall)

    for folder in get_top_level(src_data):
        matches = folder.glob("outputFor*")
        matches = [
            m for m in matches
            if 'outputFor' in m.name
            and m.is_file()
            and '_' in m.name
        ]
        logger.info('Found %d files in %s', len(matches), folder)
        for file in matches:
            try:
                jobs.append(extract_delayed(file, dest))
            except Exception as e:
                logger.error(
                    'Error processing %s: %s', str(file), str(e)
                )
                raise

    logger.info('Processing %d files', len(jobs))
    dask.compute(*jobs, scheduler='processes')
    logger.info('Done!')


# ---------------------------------------------------------------------------
# NetCDF writer (ARR Project 4 sub-daily schema)
# ---------------------------------------------------------------------------

def produceSubDailyNetCDF(fnameNC, subDailyData, dayVector, **kwargs):
    """
    Write sub-daily rainfall data to a new NetCDF4 file in ARR Project 4
    schema.

    Creates dimensions 'simulation', 'day', and 'subday', writes the
    Julian day vector and sub-day fraction vector as coordinate
    variables, and stores the rainfall array with compression and a
    -9999 fill value.  Ported from MATLAB by P. Brady (2016) and
    C. Dykman (2021).

    Parameters:
    - fnameNC: Path to the output NetCDF file.  A Warning is raised
      and the function returns False if the file already exists.
    - subDailyData: 3-D numpy array with shape
      (simulation, day, n_intervals_per_day).  Missing values should
      be represented as -9999.0.
    - dayVector: 1-D array of Julian dates, one per day in the record.
    - **kwargs: Optional keyword overrides:
        - title (str): Dataset title attribute.
          Default: 'Sub-Daily Rainfall'.
        - institution (str): Institution attribute.
          Default: 'UNSW Water Research Centre'.

    Returns:
    - True if the file was written successfully; False if the file
      already existed.
    """
    if os.path.exists(fnameNC):
        raise Warning(
            'PSDNC:Exists — the file {} already exists. '
            'Delete it and try again.'.format(fnameNC)
        )
        return False

    status = True
    dataSetTitle = 'Sub-Daily Rainfall'
    dataSetInstitution = 'UNSW Water Research Centre'
    nRecordsPerDay = np.size(subDailyData, axis=2)

    # Apply any caller-supplied attribute overrides
    for key, value in kwargs.items():
        if key.lower() == 'title':
            dataSetTitle = value
        if key.lower() == 'institution':
            dataSetInstitution = value

    # Build the sub-day fraction vector (equally spaced within [0, 1])
    deltaT = 1 / nRecordsPerDay
    subDayVec = np.arange(deltaT, 1 + deltaT, deltaT)

    # Create the NetCDF file and define its dimensions
    SubDailync = nc.Dataset(fnameNC, 'w', format='NETCDF4')
    SubDailync.createDimension('day', len(dayVector))
    SubDailync.createDimension('subday', len(subDayVec))
    SubDailync.createDimension(
        'simulation', np.size(subDailyData, axis=0)
    )

    # Create coordinate and data variables with zlib compression
    days = SubDailync.createVariable(
        'day', 'f8', ('day',), zlib=True, complevel=9, shuffle=True,
    )
    subdays = SubDailync.createVariable(
        'subday', 'f8', ('subday',),
        zlib=True, complevel=9, shuffle=True,
    )
    SCALE_FACTOR = 1.0
    rainfalls = SubDailync.createVariable(
        'rainfall', 'f4', ('simulation', 'day', 'subday'),
        zlib=True, complevel=9, shuffle=True,
        fill_value=-9999.0 * SCALE_FACTOR,
    )

    # Write coordinate and rainfall data
    days[:] = dayVector
    subdays[:] = subDayVec
    rainfalls[:, :, :] = subDailyData * SCALE_FACTOR

    # Set global attributes
    SubDailync.title = dataSetTitle
    SubDailync.history = 'Generated {} by {}@{}'.format(
        datetime.datetime.now(), 'username', 'hostname'
    )
    SubDailync.source = 'Python function: {}@{}'.format(
        'headURL', 'revision'
    )
    SubDailync.institution = dataSetInstitution
    SubDailync.conventions = 'ARR Project 4'

    # Set variable attributes
    days.units = (
        'Julian date, no timezone, '
        '1900-01-01 00:00:00 == 2415020.5'
    )
    days.long_name = 'Julian Date'
    subdays.units = (
        'Julian date, no timezone, '
        '1900-01-01 00:00:00 == 2415020.5'
    )
    subdays.long_name = 'Julian Date'
    subdays.description = (
        'fractional time of day that when added to day '
        'gives the time of the rainfall'
    )
    rainfalls.units = 'mm'
    rainfalls.long_name = 'Subdaily Rainfall'

    SubDailync.close()
    return status
