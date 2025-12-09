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

# %%
import os


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
# > **Access Note:** running this Notebook and associated code from a OneDrive folder may result in 'Access Denied' errors when creating or updating a project. We recommend setting up on your C:\ (or other primary local) drive.

# %%
# proj = FireImpactsProject('.\\fire_impacts_example_project', clear=True)
proj = FireImpactsProject('\\zz_TempDump\\fire_impacts_example_project',clear=True)

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
example_catchment_name = 'example_small_catchment'
proj.add_catchment(f'..\\test_data\\{example_catchment_name}.json')

# %%
proj.catchments

# %% [markdown]
# ## Pre-processing steps
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
# ## Aside: Visualising data
#
# The `proj` object includes convenience functions for visualising the processed data layers as they are created. Here, we show the visualisation after each step using built-in default parameters.

# %% [markdown]
# ## Topography
#
# The topographic processing uses a DEM to identify relevant topographic properties required for the modelling.. 
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

# %% [markdown]
# ### Digital Elevation Model (DEM)

# %%
# Point python to the DEM. For any case other than this example catchment, update the path to point to your actual DEM:
DEM_FILENAME='..\\test_data\\example_dem.tif'
# Extract the catchment only from the input DEM:
topography.extract_catchment_dems(proj,DEM_FILENAME)
# Visualise the processed DEM:
proj.plot_catchment_raster('Topography','DEM.tif')

# %% [markdown]
# ### Headwaters
# We will see the headwaters visualised later on in this notebook.

# %%
# Just call the extract_headwaters() method from the topography module:
headwaters = topography.extract_headwaters(proj)

# %%
# See a snapshot of what the table of headwaters looks like:
headwaters['example_small_catchment'].head()

# %% [markdown]
# ### Slope

# %%
proj.plot_catchment_raster('Topography','Slope')

# %% [markdown]
# ### Flow Accumulation

# %%
proj.plot_catchment_raster('Topography','Flow_accumulation')

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
# Fire severity data requires measuring Normalised Burn Ration (NBR) before and after a fire, to produce a measure of change called delta-NBR (abbreviated as dNBR or ΔNBR).
#
# This package automatically finds and downloads relevant satellite-based NBR datasets and computes fire severity for your catchment. 
# > **Note**: Currently, our example small catchment does not have a known fire date, so dNBR values will not be high. This will be updated in a future release to use an actual or simulated fire for this area.

# %%
fire_start_date = '2019-01-15'  # Set fire start date (the date that fire started)
fire_end_date = '2019-03-07'    # Set fire end date (the date that fire ended)

# %%
# This method may take a few minutes to download and process data.
severity.calculate_fire_severity(
    project=proj,
    catchment=None,
    fire_start_date=fire_start_date,
    fire_end_date=fire_end_date,
)


# %%
proj.plot_catchment_raster('FireSeverity', 'dNBR')

# %% [markdown]
# ## Soils
# Soil data are required for RUSLE (erosion) and debris flow simulations.
#
# ### Required Inputs
# Many of these will be downloaded and processed automatically by the package, but you will need a **TERN API KEY** to access them. We suggest creating your API key then saving it as an Environment Variable under your user profile (for Windows). We assume this has already been set up for the purpose of this example notebook.
#
# You will also need an **Aridity raster** file, which is included for the example catchment but will need to be obtained for your catchment of interest before loading the subsequent datasets or running the simulations.

# %%
# Tell python where your API key is:
API_KEY = os.environ.get('TERN_API_KEY')
# Download the relevant data from TERN:
soil.download_soil_data_stac(proj, api_key=API_KEY)

# Existing aridity raster location:
ARIDITY=r'..\\test_data\\Aridity_PT.tif'
# Extract aridity data for each catchment:
soil.extract_aridity_data(proj, aridity_raster=ARIDITY)

# %%
# Visualise the processed aridity raster:
proj.plot_catchment_raster('Soils', 'Aridity')

# %% [markdown]
# ## RUSLE
# The **R**evised **U**niversal **S**oil **L**oss **E**quation is used to estimate general post-fire erosion and is calculated for the entire catchment.
#
# ### Required Inputs
# You will need to obtain RUSLE factor rasters for the erosion simulations. The package will prepare them for simulation in each catchment. 
# Both **C-factor** and **K-factor** rasters are required. They have been provided for this example catchment to demonstrate functionality.
#

# %% [markdown]
# ### C- and K-Factors

# %%
# Point to your provided rasters:
c_factor_path='..\\test_data\\c_factor_g94.tif'
k_factor_path='..\\test_data\\k_factor_g94.tif'

# Compute adjusted K- and C-factors ready for erosion simulation:
rusle.compute_adjusted_k_c(proj, catchment=example_catchment_name, c_factor_fn=c_factor_path, k_factor_fn=k_factor_path)

# %% [markdown]
# ## Summary information for Fire Severity
# We can now produce a summary table of all the key inputs to the simulation steps to follow:
#
# This may take a few minutes while aggregations are computed for each headwater.

# %%
summary = project.summary_stats(proj)

# %% [markdown]
# The *summary stats* table shows these inputs for each headwater. You can view them in table form easily...

# %%
summary['example_small_catchment'].head()

# %% [markdown]
# ...and also in a map, where we can see the shape of those headwaters now:

# %%
proj.plot_headwaters(example_catchment_name, colour_col='dNBR_mean', table=summary['example_small_catchment'])
