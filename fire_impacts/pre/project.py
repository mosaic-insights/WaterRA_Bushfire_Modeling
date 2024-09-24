import os
from glob import glob
import shutil
import logging
logger = logging.getLogger(__name__)

def initialise_project(directory:str,catchment_shapefiles=[],exist_ok=False,clear=False):
    '''
    Initialise a folder structure to contain working data for fire impacts studies.

    Parameters:
    - directory (str): Path to the directory where the project folder will be created.
    - catchment_shapefiles (list): List of paths to shapefiles representing catchments.
    - exist_ok (bool): OPTIONAL: If False, raise an error if the project folder already exists. Default is False.
    - clear (bool): OPTIONAL: If True, clear the project folder if it already exists. Default is False.
    '''
    folder_paths = {}
    # Define initial folder structure
    subfolders = ['Output', 'Output/Topography', 'Output/FireSeverity','Output/Topography/Catchments_DEM']
    # Create folders and store their paths
    for folder in subfolders:
        path = os.path.join(directory, folder)
        if os.path.exists(path) and clear:
            logger.info('Clearing folder: %s',path)
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=exist_ok)
        folder_name = folder.split('/')[-1]
        folder_paths[folder_name] = path
        # globals()[folder_name] = path

    folder_paths['Catchment_Shapefiles'] = catchment_shapefiles
    folder_paths['Catchment_Names'] = [os.path.splitext(os.path.basename(shapefile))[0] for shapefile in catchment_shapefiles]
    catchment_folders = create_catchment_folders(folder_paths)
    folder_paths.update(catchment_folders)
    return folder_paths

def create_catchment_folders(project):
    catchment_names = project['Catchment_Names']
    topography_path = project['Topography']
    folder_paths = {}
    for catch_name in catchment_names:
        main_path = os.path.join(topography_path, catch_name)
        os.makedirs(main_path, exist_ok=True)
        # Create subfolders inside each main folder
        subfolders = ['HW_SHPs', 'HW_Rasters', 'Catchment_Files']
        for subfolder in subfolders:
            subfolder_path = os.path.join(main_path, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)
        folder_name = catch_name.split('/')[-1]
        folder_paths[folder_name] = main_path
        # globals()[folder_name] = main_path
    return folder_paths

def find_all_shapefiles(base_directory):
    '''
    Find all shapefiles in a directory and its subdirectories.
    '''
    assert os.path.isdir(base_directory), f"Directory not found: {base_directory}"
    shapefiles = glob(os.path.join(base_directory, '**','*.shp'),recursive=True)
    return shapefiles
