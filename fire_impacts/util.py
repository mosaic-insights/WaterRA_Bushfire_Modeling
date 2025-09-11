import os
import logging
logger = logging.getLogger(__name__)

def retry(fn,retries=5,initial_delay=8,delay_scale=3,specific_exceptions=None):
    import time

    try:
        return fn()
    except Exception as e:
        if retries<=0:
            raise e

        if specific_exceptions is not None:
            if e.__class__ not in specific_exceptions:
                raise e

        logger.warning('Failed with %s. Retrying after %d seconds'%(str(e),initial_delay))
        time.sleep(initial_delay)
        return retry(fn,retries-1,initial_delay*delay_scale,delay_scale,specific_exceptions)

def package_data_path(fn=None):
    dirname = os.path.join(os.path.dirname(__file__),'..','data')
    if fn is None:
        return dirname
    return os.path.join(dirname,fn)


def load_package_data(fn):
    fn = package_data_path(fn)
    if fn.endswith('.csv'):
        logger.info(f"Loading data from {fn}")
        import pandas as pd
        return pd.read_csv(fn)
    logger.error(f"Unsupported file type: {fn}")
    return None

def file_matching_all(path,*substrings):
    """Check if a file contains all substrings and return a list of matches"""
    files = os.listdir(path)
    return [fn for fn in files if all(p in fn for p in substrings)]

def unique_file_matching(path,*substrings):
    """Check if a single file contains all substrings and return the unique match"""
    matches = file_matching_all(path,*substrings)
    if len(matches) == 0:
        raise FileNotFoundError(f"No file found in {path} matching patterns: {substrings}")
    elif len(matches) > 1:
        raise FileExistsError(f"Multiple files found in {path} matching patterns: {substrings}")
    return matches[0]



