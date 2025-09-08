'''
Prepare BOM daily and sub-daily rainfall data for use with PyRainGen.

Key functions:
- convert_daily: Convert daily rainfall data files to PyRainGen format.
- convert_sub_daily: Convert sub-daily rainfall data files to PyRainGen format.
'''
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

def extract_and_format_daily_rainfall(file, dest):
    file_name = file.name
    station = file_name.split("_")[2]  # Extract station number from data file name
    output_file = dest / f"rev_dr{station}.txt"

    # Initialize a list to hold the formatted data
    formatted_data = []

    # Process the input file
    with open(file, "r") as infile:
        # Skip the first line (header)
        next(infile)

        for line in infile:
            # Extract relevant sections by splitting the line
            parts = line.split(",")
            try:
                # Parse year, month, day, and precipitation from parts
                year = parts[2].strip()  # Year
                month = parts[3].strip()  # Month
                day = parts[4].strip()  # Day
                precip = parts[5].strip()  # Precipitation value

                # Create a formatted line
                formatted_line = [int(year), int(month), int(day), float(precip.strip())]
                formatted_data.append(formatted_line)
            except (IndexError, ValueError):
                # Skip malformed lines
                continue

    # Skip processing if no valid data was extracted
    if not formatted_data:
        return

    # Convert formatted data to a numpy array
    formatted_array = np.array(formatted_data)

    # Determine the number of individual years and the first year in the data
    unique_years = np.unique(formatted_array[:, 0])  # Extract the unique years from the first column
    first_year = int(unique_years[0])  # Get the first year (assuming chronological order)
    number_of_years = len(unique_years)  # Count the number of unique years

    # Save the formatted data using numpy.savetxt
    np.savetxt(
        output_file,
        formatted_array,
        fmt="%7d %4d %4d %9.2f",
        header= f" Daily rainfall at station dr{station}        {first_year}      {number_of_years}",
        comments="",
    )

def ensure_dest(dest,clear=False):
    if clear and os.path.exists(dest):
        logger.info('Clearing directory: %s', dest)
        import shutil
        shutil.rmtree(dest)
    if not os.path.exists(dest):
        logger.info('Creating directory: %s', dest)
        os.makedirs(dest, exist_ok=True)

def get_top_level(src_data):
    if isinstance(src_data, str):
        return [Path(p) for p in glob(src_data)]
    elif hasattr(src_data,'iterdir'):
        return src_data.iterdir()

    return src_data

def convert_daily(src_data,dest):
    '''
    Convert daily rainfall data files to PyRainGen format.
    
    Parameters:
    - src_data: Source directory or list of directories containing daily rainfall data files.
    - dest: Destination directory to save the converted files.
    '''
    ensure_dest(dest)
    # call .iterdir() because BOM_DATA_DAILY is a Path object
    jobs = []
    extract_and_format_daily_rainfall_ = dask.delayed(extract_and_format_daily_rainfall)
    for folder in get_top_level(src_data):
        # Look for all .txt files directly under this folder
        if folder.is_dir():
            matches = folder.glob("*.txt")
        else:
            matches = [folder]
        matches = [m for m in matches if \
                   m.is_file() and \
                  'Notes' not in m.name and \
                  'StnDet' not in m.name and \
                  m.name.endswith(".txt")]
        if not len(matches):
            logger.debug('No valid files found in %s', folder)
            continue

        logger.info('Found %d files in %s', len(matches), folder)
        for file in matches:
            try:
                jobs.append(extract_and_format_daily_rainfall_(file, dest))
            except:
                logger.error('Error processing %s', str(file))
                raise

            # Exclude files with "Notes" or "StnDet" in their name
            # if "Notes" in file_name or "StnDet" in file_name:
            #     logger.info(f"Skipping file: {file_name}")
            #     continue
            #logger.info(f"Formatted data saved to {output_file}")

    logger.info('Processing %d files', len(jobs))
    dask.compute(*jobs, scheduler='processes')
    logger.info('Done!')

def extract_and_format_sub_daily_rainfall(file, dest):
    file_name = file.name
    # Extract station number
    st = file_name.index("outputFor") + len("outputFor")
    en = file_name.index("_")
    station = file_name[st:en]
    
    # Format output file based on station number length
    if len(station) == 4:
        output_file = dest / f"plv00{station}.nc"
    elif len(station) == 5:
        output_file = dest / f"plv0{station}.nc"
    elif len(station) == 6:
        output_file = dest / f"plv{station}.nc"
    else:
        logger.info(f"Skipping file {file_name}: Unexpected station number format")
        return
    
    # Read file content
    with open(file, 'r') as sub:
        lines = sub.readlines()
    
    # Skip the first row
    lines = lines[1:]
    data = []
    
    # Process each line
    for line in lines:
        line = line.strip()
        station = line[:6]
        date = line[8:20].strip()
        rain = line[20:].strip()
        data.append([station, date, rain])
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['Station', 'Date', 'Rain'])
    df = df.iloc[1:, :]
    
    # Standardize date format
    def standardize_date(date_str):
        date_str = date_str.strip()
        if " " in date_str:
            parts = date_str.split()
            if len(parts) == 2:
                return f"{parts[0]}{int(parts[1]):02d}"
            elif len(parts) == 3:
                return f"{parts[0]}{int(parts[1]):02d}{int(parts[2]):02d}"
        return date_str
    
    df['StandardizedDate'] = df['Date'].apply(standardize_date)
    df['DateFormatted'] = pd.to_datetime(df['StandardizedDate'], format='%Y%m%d')
    
    # Parse rainfall data
    def parse_rainfall(row):
        row = re.sub(r'-\d+\.\d+', '-9999.0', row)
        cleaned_row = re.sub(r'(?<!\s)(-9999.0|-8888.0|-1544.4|-1872.0|-1132.0)', r' \1', row)
        cleaned_row = " ".join(cleaned_row.split())
        rain_values = np.array(cleaned_row.split(), dtype=np.float32)
        return rain_values
    
    df['RainArray'] = df['Rain'].apply(parse_rainfall)
    df['DateFormatted'] = pd.to_datetime(df['DateFormatted'])
    
    # Generate a full date range from the first to last date
    full_date_range = pd.date_range(start=df['DateFormatted'].iloc[0], end=df['DateFormatted'].iloc[-1], freq='D')
    
    # Convert full date range to Julian format
    dayVector = full_date_range.to_julian_date().values
    
    # Create a dictionary to map existing data to full date range
    rainfall_dict = dict(zip(df['DateFormatted'].apply(lambda x: x.to_julian_date()), df['RainArray']))
    
    # Define sub-daily intervals (6-minute intervals per day)
    n_intervals_per_day = 240
    subDailyData = np.full((1, len(dayVector), n_intervals_per_day), -9999.0, dtype=np.float32) # fill missing days with -9999.0
    
    # Fill in available data
    for i, julian_day in enumerate(dayVector):
        if julian_day in rainfall_dict:
            subDailyData[0, i, :] = rainfall_dict[julian_day]
    
    # Call the function to create NetCDF
    status = produceSubDailyNetCDF(
        fnameNC=output_file,
        subDailyData=subDailyData,
        dayVector=dayVector,
        title="Sub-Daily Rainfall Dataset",
        institution="Your Institution Name"
    )

def convert_sub_daily(src_data,dest,clear=False):
    '''
    Convert sub-daily rainfall data files to PyRainGen format.

    Parameters:
    - src_data: Source directory or list of directories containing sub-daily rainfall data files.
    - dest: Destination directory to save the converted files.
    - clear: If True, clear the destination directory before processing.
    '''
    ensure_dest(dest,clear)
    # Loop through folders and process files
    jobs = []
    extract_and_format_sub_daily_rainfall_ = dask.delayed(extract_and_format_sub_daily_rainfall)
    for folder in get_top_level(src_data):
        # Look for all .txt files directly under this folder
        matches = folder.glob("outputFor*")
        matches = [m for m in matches if 'outputFor' in m.name and m.is_file() and '_' in m.name]
        logger.info('Found %d files in %s', len(matches), folder)
        for file in matches:
            try:
                job = extract_and_format_sub_daily_rainfall_(file, dest)
                jobs.append(job)
            except Exception as e:
                logger.error('Error processing %s: %s', str(file), str(e))
                raise
    logger.info('Processing %d files', len(jobs))
    dask.compute(*jobs, scheduler='processes')
    logger.info('Done!')

def produceSubDailyNetCDF(fnameNC,subDailyData,dayVector,**kwargs):
    #PRODUCESUBDAILYNETCDF Create from scratch a sub-daily netCDF as per schema
    #   This function is a single write point for sub-daily data for the ARR
    #   Project 4 code in MATLAB.  That way we have a consistent layout across
    #   all code and a standardised netCDF layout.
    #
    #   Inputs:
    #       -) fnameNC: the name of the netCDF file to dump data into.  If the
    #           file exists, a warning will be thrown and the fuction return an
    #           error.
    #       -) subDailyData: an array of the sub-daily with the following
    #           dimension lengths:
    #               1) number of records per day
    #               2) number of days
    #               3) number of simulations
    #           For BoM or other "real" data, as opposed to simulated data, the
    #           simulation dimension will == 1.  Data may be padded with BoM or
    #           similar negative numbers.
    #       -) dayVector: a vector of Julian dates for the day of the
    #           measurement recording.  Remember that Julian days start at
    #           midday.  No timezone information is stored.
    #
    #   Optional Inputs (in name/value pairs)
    #       -) title: data set title to be inserted into the global attribute
    #           "title".  Default: "Sub-Daily Rainfall".
    #       -) institution: a text field to identify the institution that
    #           created this data set in the global attribute "institution".
    #
    #   Output:
    #       -) status: true for OK, false for an error. 
    #
    #   Dr Peter Brady <peter.brady@wmawater.com.au>
    #   2016-08-26
    #
    #  Python Conversion
    #  Caleb Dykman
    #  2021-09-27

    ## Check of the file exists
    if os.path.exists(fnameNC):
        raise Warning('PSDNC:Exists ' + 'The file {}, exists please check the '
        'name and/or delete the file and try again.'.format(fnameNC))
        status = False
        return status
    
    ## Set some defaults
    status = True
    dataSetTitle = 'Sub-Daily Rainfall'
    dataSetInstitution = 'UNSW Water Research Centre'
    nRecordsPerDay = np.size(subDailyData, axis=2)

    # Unpack the kwargs
    if len(kwargs) > 0:
        for key in kwargs.keys():
            if key.lower() == 'title':
               dataSetTitle = kwargs[key] 

            if key.lower() == 'institution':
               dataSetInstitution = kwargs[key]  

    ## Create the Sub-Day Vector
    deltaT = 1 / nRecordsPerDay
    subDayVec = np.arange(deltaT, 1+deltaT, deltaT)

    ## Now Write
    # Create NetCDF
    SubDailync = nc.Dataset(fnameNC, 'w', format='NETCDF4')
   
    # Define Dimensions
    SubDailync.createDimension('day',len(dayVector))
    SubDailync.createDimension('subday', len(subDayVec))
    SubDailync.createDimension('simulation',
        np.size(subDailyData,axis=0)
        )
       
    # Create Variables
    days = SubDailync.createVariable('day','f8',('day',),zlib=True,
        complevel=9, shuffle=True
    )
    subdays = SubDailync.createVariable('subday','f8',('subday',),
        zlib=True, complevel=9, shuffle=True, 
    )
    rainfalls = SubDailync.createVariable('rainfall','f4',
        ('simulation','day','subday'), zlib=True, complevel=9, 
        shuffle=True, fill_value=-999.9
    )

    ## Writing Data
    days[:] = dayVector
    subdays[:] = subDayVec
    rainfalls[:,:,:] = subDailyData

    ## Add Attributes
    # Global
    SubDailync.title = dataSetTitle
    SubDailync.history = 'Generated {} by {}@{}'.format(datetime.datetime.now()
        ,'username','hostname'
    )
    SubDailync.source = 'Python function: {}@{}'.format(
        'headURL','revision'
    )
    SubDailync.institution = dataSetInstitution
    SubDailync.conventions = 'ARR Project 4'
    
    # Variable
    days.units = 'Julian date, no timezone, 1900-01-01 00:00:00 == 2415020.5'
    days.long_name = 'Julian Date'
    subdays.units = 'Julian date, no timezone, 1900-01-01 00:00:00 == 2415020.5'
    subdays.long_name = 'Julian Date'
    subdays.description = 'fractional time of day that when added to day gives the time of the rainfall'
    rainfalls.units = 'mm'
    rainfalls.long_name = 'Subdaily Rainfall'

    SubDailync.close()

    return status

