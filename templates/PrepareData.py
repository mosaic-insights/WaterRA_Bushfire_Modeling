# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Preparing data for fire impact analysis
#
# This notebook demonstrates the process for processing input data for use with the fire impacts library.
#
# The process is largely automated and, in many cases, the data is downloaded automatically from the publisher.
#
# There are a few things that you will need, however:
#
# * A catchment boundary for your area of interest (eg Shapefile, GeoJSON)
# * A suitable, high resolution DEM for the area. The 1" SRTM derived DEM-H is a good option for most cases.
# * The date of a fire event within the catchment,
# * Several other datasets that you will need to manually download and store locally
#

# %% [markdown]
# ## Installation
#
# You will need a Python installation with the various libaries installed.
#
# As a starting point, a base scientific Python installation that includes `numpy`, `pandas`, `jupyter`, `matplotlib`. For Windows users, the easiest way to set up such an environment is to use Anaconda Python, or miniconda.
#
# This base environment should be extended with specific libraries that are used by the fire impacts library. These are listed in `requirements.txt` and can be installed using `pip` from a command prompt:
#
# ```
# cd <directory-with-library>
# pip install -r requirements.txt
# ```
#
# Finally, the fire impacts library itself should be installed. If you have cloned the git repository, you can install from your local copy. From the command prompt:
#
# ```
# cd <directory-with-library>
# pip install -e .
# ```
#
# When the installation has completed, the following import statements should run without error

# %%
from fire_impacts import FireImpactsProject
from fire_impacts.pre import project, topography, severity, soil, rusle

# %% [markdown]
# ## Logging
#
# We use logging statements to provide feedback on progress through various steps. This allows you to tailor what level of information you see by setting a log 'level':
#
# * `DEBUG`: Low level information about progress
# * `INFO`: General progress updates
# * `WARNING`: Problems or potential problems that the system can handle
# * `EROR`: Problems that prevent the system from running correctly
#
# When you set a log level, you will see those messages as well as the more serious ones. So if you choose `INFO`, you will also see `WARNING` and `ERROR` messages. If you choose `WARNING`, you will also see `ERROR`.

# %%
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# %% [markdown]
# ## Projects
#
# We organise all the data related to a study in a project directory. This should generally be a *new* directory that will be populated with data that has been computed with the library. The library will manage the subdirectories and data within the main project directory.
#
# In Python, we use a `FireImpactsProject` object to represent this directory and all the data stored within it
#
# Here, we create a new project in the current directory. **Note:** In this case we will delete (`clear`) any existing data in that directory.
#
# By default, *current directory* here is the directory this Notebook is saved in.
#
#
# **Try** running this next code not on OneDrive and see if we still get hte access error

# %%

proj = FireImpactsProject('.', exist_ok=True, clear=False)

# %% [markdown]
# ## Catchment areas
#
# A project can store data related to one or more catchments.
#
# Catchments are added to the project by providing a boundary coverage (eg a Shapefile or a GeoJSON file).
#
# Here, we add a small example catchment, from the `test_data` directory, but you can use your own.
#
# **Note:** The boundary coverage should include a coordinate reference system (CRS). This is the CRS that will be used for all other data stored in relation to this catchment in the project.

# %%
proj.add_catchment('..\\test_data\\example_small_catchment.json') # PUT THE PATH TO YOUR CATCHMENT BOUNDARY HERE

# %%
proj.catchments

# %% [markdown]
# ## Pro processing steps
#
# We will now work through each step of data pre-processing to support for the fire impacts modelling:
#
# * **Topography**: Determine various topographic properties from a DEM, including headwater catchments used to model debris flow triggers
# * **Fire severity**: Analyse satellite information to determine the intensity of a historical fire event
# * **Soils**: Extract relevant soil properties from the Soil and Landscape Grid of Australia
# * **RUSLE**: Compute RUSLE terms (K, L, S and C), including fire modified versions of K and C
#
# There is some flexibility to run these processes in different orders, except that the RUSLE processing should be done _after_ other processes.
#
# The following sections describe each process and the data required.

# %% [markdown]
# ## Topography
#
# The topographic processing uses a DEM to identify relevant topographic properties required for the modelling.
#
# This includes:
#
# * Headwater catchment delineation, used to model the areas likely to trigger debris flow, and
# * Slope and hillslope length, used in the parameterisation of erosion modelling.
#
# You will need a suitable DEM. We have provided a DEM for the example catchment, but if you are using your own catchment you need a DEM covering the entire catchment.
#
# If using the [national 1" DEM](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/72759), use the hydrologically enforced DEM (DEM-H).
#

# %%
DEM_FILENAME='..\\test_data\\example_dem.tif' # PUT THE PATH TO YOUR DEM FILE HERE

# %%
topography.extract_catchment_dems(proj,DEM_FILENAME)

# %%
headwaters = topography.extract_headwaters(proj)

# %%
headwaters['example_small_catchment']

# %% [markdown]
# ## Aside: Visualising data
#
# The `proj` object includes a convenience function for visualising the processed data layers:

# %%
proj.plot_catchment_raster('Topography','DEM.tif')

# %%

# %%
proj.plot_catchment_raster('Topography','Slope')

# %%
proj.plot_catchment_raster('Topography','Flow_accumulation')

# %%

# %%

# %% [markdown]
# ## Aside: Function documentation
#
# Most of the library functions include documentation describing the process, the function parameters and any return values
#
# You can access the help in Jupyter with the `?` operator, for example:
#
# ```
# topography.extract_headwaters?
# ```

# %%
# topography.extract_headwaters?

# %% [markdown]
# ## Fire Severity

# %%
fire_start_date = '2019-01-15'  # Set fire start date (the date that fire started)
fire_end_date = '2019-03-07'    # Set fire end date (the date that fire ended)

# %%
severity.calculate_fire_severity(
    project=proj,
    catchment=None,
    fire_start_date=fire_start_date,
    fire_end_date=fire_end_date,
)


# %% [markdown]
# ## Soils
#
#

# %%
ARIDITY=r'..\\test_data\\Aridity_PT.tif'

# %%
soil.download_soil_data(proj)

# Extract aridity data for each catchment
soil.extract_aridity_data(proj,aridity_raster=ARIDITY)

# %%

# %% [markdown]
# ## RUSLE
#
#

# %%
c_factor_path='..\\test_data\\c_factor_g94.tif'
k_factor_path='..\\test_data\\k_factor_g94.tif'


# %%
rusle.compute_adjusted_k_c(proj,catchment=None,c_factor_fn=c_factor_path,k_factor_fn=k_factor_path)

# %%

results = rusle.compute_lsi(proj)


# %%

# %% [markdown]
# ## Summary information for Fire Severity

# %%
summary = project.summary_stats(proj)

# %%

# %%
summary['example_small_catchment']

# %%
