# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
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
# * The date of a fire event within the catchment,
# * Several other datasets that you will need to manually download and store locally
#

# %% [markdown]
# ## Installation
#
# You will need Python with a set of scientific libraries installed, plus the `fire_impacts` package itself. There are two ways to set up the environment, described below. **Method 1 is recommended** — it creates a fully self-contained environment and is less likely to run into conflicts.
#
# #### A note on environments
#
# A Python *environment* is an isolated copy of Python with its own set of installed libraries. Using an environment means the packages needed for this project won't interfere with anything else on your computer. The instructions below use [Miniforge](https://github.com/conda-forge/miniforge), a lightweight tool for managing Python environments in scientific work. If you don't already have it installed, download and install Miniforge first. Miniforge includes `mamba`, a faster drop-in replacement for the `conda` command that is recommended for resolving complex environments like this one.
#
#
# ### Method 1 — Create a new environment from `environment.yml` (recommended)
#
# The file `environment.yml` in this repository describes a complete Python environment: the Python version, all required libraries, and their versions. Mamba will create this environment for you in one step.
#
# Open a **Miniforge Prompt** (Windows) or terminal (Mac/Linux), navigate to the folder containing this repository, and run:
#
# ```
# mamba env create -f environment.yml
# ```
#
# This will take a few minutes. When it finishes, activate the new environment:
#
# ```
# conda activate bushfire-py313
# ```
#
# You will need to activate the environment each time you open a new terminal or Miniforge Prompt before working with this project.
#
# > **Note:** `environment.yml` includes a patched version of the `pysheds` library (installed directly from GitHub) to fix a compatibility issue with NumPy 2.0. It also installs `dea-tools`, which requires the PostgreSQL development headers (`pg_config`) to be available on your system. On Windows this is usually satisfied automatically via conda-forge; if the install fails at that step, see the [dea-tools documentation](https://github.com/GeoscienceAustralia/dea-notebooks).
#
#
# ### Method 2 — Install into an existing Python environment using `requirements.txt`
#
# Use this method if you already have a working scientific Python installation (with `numpy`, `pandas`, `matplotlib`, and `jupyter`) and want to add the project's extra dependencies to it.
#
# A `requirements.txt` file is a plain text list of Python packages. The `pip` tool reads this file and installs each one.
#
# Open a Miniforge Prompt or terminal, activate your existing environment if you have one, navigate to the folder containing this repository, and run:
#
# ```
# pip install -r requirements.txt
# ```
#
# > **Note:** `requirements.txt` installs the standard `pysheds` release from PyPI. If you encounter errors related to `pysheds` and NumPy, install the patched version manually:
# > ```
# > pip install https://github.com/joelrahman/pysheds/archive/refs/heads/master.zip
# > ```
#
#
# ### Installing the `fire_impacts` package itself
#
# Whichever method you used above, the final step is the same: install the `fire_impacts` library from your local copy of the repository. The `-e` flag installs it in *editable* mode, meaning any changes to the source code take effect immediately without reinstalling.
#
# From the repository folder, run:
#
# ```
# pip install -e .
# ```
#
#
# #### Checking the installation
#
# When the installation is complete, all the following imports should run without error.

# %%
from fire_impacts import FireImpactsProject
from fire_impacts import const
from fire_impacts.context import RunContext
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
proj = FireImpactsProject('.',clear=True)

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
example_catchment_name = 'EgSmallCatchment_7899'
proj.add_catchment(f'..\\test_data\\{example_catchment_name}.shp')

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

# %% [markdown]
# ### Digital Elevation Model (DEM)
# By default, the package will download the Geoscience Australia hydrologically-enforced DEM ([national 1" DEM](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/72759)), which has a horizontal spatial resolution of 1 arc second (approx. 30 metres).
#
# > **Option**: If you have a specific DEM you wish to use, you can provide the path\filename.ext as the second argument of `topography.extract_catchment_dems()`. Make sure the DEM covers the entire catchment area.
#

# %%
# If you have a specific DEM you want to use, point python to it.
#We have a DEM for the example catchment included in the test date for this package.
optional_DEM_filename = '..\\test_data\\example_dem.tif'

# Static catchment-level preprocessing (DEM, soil, headwaters) doesn't
# depend on a fire event — use a catchment-only RunContext.
prep_ctx = RunContext.solo_catchment(proj)

# Get a DEM. To use your own DEM, replace None with the filename path:
topography.extract_catchment_dems(prep_ctx, None)
# Visualise the processed DEM:
proj.plot_catchment_raster('Topography','DEM.tif')

# %% [markdown]
# ### Headwaters
# This package will automatically derived headwaters from the DEM.

# %%
# Just call the extract_headwaters() method from the topography module:
headwaters = topography.extract_headwaters(prep_ctx)

# %% [markdown]
# Headwaters can be visualised straightaway as a map using `FireImpactsProject.plot_headwaters()`. You can also view a quick summary of the headwaters in table form using the built-in `.head()` method.

# %%
# Plot the raw headwaters to see what they look like:
proj.plot_headwaters(example_catchment_name)

# See a snapshot of what they look like in tabular form:
headwaters.head()

# %% [markdown]
# ### Slope
# A slope layer is calculated when the headwaters are defined. You can view it easily:

# %%
# View the slope raster which was derived from the DEM:
proj.plot_catchment_raster('Topography','Slope')

# %% [markdown]
# ### Flow Accumulation
# Similarly, hydrology rasters such as flow accumulation are also saved and can be viewed in the same way:

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
# > **Note**: The fire start and end dates here correspond to an actual fire that took place in this area.

# %%
fire_start_date = '2019-01-15'  # Set fire start date (the date that fire started)
fire_end_date = '2019-03-07'    # Set fire end date (the date that fire ended)
# Each fire is identified by an event name. Every fire-dependent
# operation binds to a RunContext combining the catchment and event;
# RunContext.solo_event resolves the catchment when there is exactly
# one in the project.
ctx = RunContext.solo_event(proj, event='2019_fire')

# %%
# This method may take a few minutes to download and process data.
severity.calculate_fire_severity(
    ctx,
    fire_start_date=fire_start_date,
    fire_end_date=fire_end_date,
)


# %%
# Visualise the dNBR raster. Fire-severity rasters live under the event
# (Events/<event>/FireSeverity/), so pass the event context ctx so the
# plotter reads from the right place.
proj.plot_catchment_raster('FireSeverity', 'dNBR', ctx=ctx)

# %%
# Visualise the masked dNBR raster, which is the version of dNBR that is used in the modelling (it is masked to the headwater areas):
proj.plot_catchment_raster('FireSeverity', 'masked_dNBR', ctx=ctx)

# %% [markdown]
# ## Soils
# Soil data are required for RUSLE (erosion) and debris flow simulations.
#
# ### Required Inputs
# Many of these will be downloaded and processed automatically by the package, but you will need a **TERN API KEY** to access them. The following instructions show you how to obtain an API and save it on Windows in such a way that python will be able to access it:
# 1. Create a TERN account - [Instructions for creating a TERN account (TERN Youtube channel)](https://youtu.be/HTlc0xk4zf8?si=z_vEqToQDwnK4-KI)
# 2. Generate an API key - [Generating an API key](https://youtu.be/HTlc0xk4zf8?si=z_vEqToQDwnK4-KI)
# 3. Save the full API key as a user-level environment variable:
#    1. *System Properties > Advanced > Environment Variables > User variables > New*
#    2. *Variable name*: `'TERN_API_KEY'` or similar
#    3. *Variable value*: Paste the full API key here
#    4. *OK*
# 5. Use `os.environ.get()` with your variable name so python can see what it is, without having to store the key itself in this notebook.
#
# You will also need an **Aridity raster** file, which is included for the example catchment but will need to be obtained for your catchment of interest before loading the subsequent datasets or running the simulations.

# %%
# Tell python where your API key is:
API_KEY = os.environ.get('TERN_API_KEY')
# Download the relevant data from TERN:
soil.download_soil_data_stac(prep_ctx, api_key=API_KEY)

# Existing aridity raster location:
ARIDITY=r'..\\test_data\\AridityPT_EgSmallCatchment_7899.tif'
# Extract aridity data for each catchment:
soil.extract_aridity_data(prep_ctx, aridity_raster=ARIDITY)

# %%
# Visualise the processed aridity raster:
proj.plot_catchment_raster('Soils', 'Aridity')

# %% [markdown]
# ## RUSLE
# The **R**evised **U**niversal **S**oil **L**oss **E**quation is used to estimate general post-fire erosion and is calculated for the entire catchment.
#
# ### Required Inputs
# The package will automatically download the national **C-factor** and **K-factor** rasters and prepare them for simulation in each catchment. No manual inputs are required for this step.

# %% [markdown]
# ## Substituting an input
#
# Sometimes you want to drive the model with something other than the real
# data — a scenario fire, a supplied raster, or a uniform value. That is an
# *input binding*, kept separate from calibration parameters: a parameter
# says what coefficient the model uses, a binding says where an input comes
# from. `dnbr` and `c_factor` are bindable.

# %%
# ctx.set_event_binding_overrides({
#     'dnbr': {'source': 'synthetic', 'severity': 'high'},
# })
# ...then materialise it before the cells that consume it:
# from fire_impacts.pre.materialise import materialise_dnbr
# materialise_dnbr(ctx)

# %% [markdown]
# ## Calibration parameters
#
# Values like the peak post-fire cover factor (`c_peak`), the SDR ceiling
# (`max_sdr`) or the debris-flow erosion coefficients are **calibration
# parameters** — the literature reports ranges for them and they may need
# tuning for your region. They are separate from unit conversions and from the
# fixed coefficients of published equations, which are not user-editable.
#
# > **Status:** every group is live — changing a value changes the layers
# > the next cells build, and the values used are recorded in a
# > `provenance.json` beside the outputs.
# >
# > `severity` is live too, so set it before running the fire-severity
# > cell — re-running that invalidates every layer downstream of the dNBR.
#
# > **Set these BEFORE running the cells below.** Changing a parameter does
# > not rebuild anything by itself — if you change one after building the
# > layers, re-run `compute_adjusted_k_c`.
#
# Overrides resolve through five layers, most specific winning:
#
# ```
# package defaults
#   └─ <project>/parameters.json                    "this study uses these values"
#       └─ Catchments/<c>/parameters.json           "this catchment differs"
#           └─ Events/<e>/event.json ("parameters") "this fire differs"
#               └─ ctx.parameters(...)              one call
# ```
#
# Every file is *sparse* — write only what you are changing.

# %%
# Inspect what is currently resolved. This returns the values AND where each
# one came from, which is the part that matters when you come back to a
# project months later.
record = ctx.parameters()
record.parameters.delivery.max_sdr

# %%
# 'default' means nobody chose it — it is just the package value.
record.sources['delivery.max_sdr']

# %% [markdown]
# ### Setting overrides
#
# Write them from Python (validated before writing) or hand-edit the JSON.
# A typo raises with a suggestion rather than being silently ignored — a
# dropped override would let you believe you had calibrated the model when
# you had not.

# %%
# Project-wide: applies to every catchment in this project.
# proj.set_parameter_overrides({'delivery': {'max_sdr': 0.75}})

# This catchment only:
# proj.set_catchment_parameter_overrides(
#     example_catchment_name, {'topography': {'max_slope_length_m': 200.0}})

# This fire only:
# ctx.set_event_parameter_overrides({'fire_adjustment': {'c_peak': 0.40}})

# One call only (not persisted, recorded as 'call'):
# ctx.parameters(delivery__max_sdr=0.9)

# %% [markdown]
# ### Not every parameter can be set at every level
#
# A parameter may only be set at a level at least as broad as the output it
# controls. `topography` and `delivery` write layers built **once per
# catchment** and shared by every fire, so they cannot be set per event — an
# event-level value would either be ignored, or would overwrite a file the
# other events depend on. Writing one to the wrong file raises, and the error
# names the file to use instead:
#
# | Group | Settable at |
# |---|---|
# | `topography`, `delivery` | project, catchment |
# | `fire_adjustment` (except `default_c_factor`), `severity` | project, catchment, event |
# | `erosion`, `debris` | any level |

# %%
# This raises: max_sdr controls SDR_baseline.tif, which is per-catchment.
# ctx.set_event_parameter_overrides({'delivery': {'max_sdr': 0.75}})

# %% [markdown]
# ### What was actually used
#
# The resolved record is written to a `provenance.json` alongside the outputs
# it produced — at catchment, event or run scope. Note the two file names are
# **not** interchangeable:
#
# | File | Who writes it | What it holds |
# |---|---|---|
# | `parameters.json` | you | a *sparse* set of overrides |
# | `provenance.json` | the library | the *full* resolved set used, plus each value's origin |

# %%
# Everything nobody chose — i.e. still on package defaults:
len(record.sources_for('default'))

# %%
# A digest identifying this exact parameter set, used to detect that derived
# layers were built with different values than a later run resolves.
record.digest

# %% [markdown]
# ### C- and K-Factors

# %%
# Compute recovery-specific K-, C- and SDR layers ready for erosion
# simulation.
#
# Recovery is specified as a single array of BREAKPOINTS in years after the
# fire end date: n+1 breakpoints define n contiguous recovery windows, and
# window i is modelled at recovery time b_i (the window start). The
# breakpoints are stored in the event's event.json, so the Simulation
# notebook reads them back automatically — you don't re-specify them there.
#
# If omitted, the package default is used. Shown here for reference:
print("Default recovery breakpoints:", const.DEFAULT_RECOVERY_BREAKPOINTS)

# To override, pass recovery_breakpoints (examples):
#   [0, 1, 2, 3]             -> yearly windows
#   [0, 0.25, 0.5, 0.75, 1]  -> quarterly windows
rusle.compute_adjusted_k_c(
    ctx,
    # recovery_breakpoints=[0, 1, 2, 3],
)

# %% [markdown]
# ## Summary information for Fire Severity
# We can now produce a summary table of all the key inputs to the simulation steps to follow:
#
# This may take a few minutes while aggregations are computed for each headwater.

# %%
summary = project.summary_stats(ctx)

# %% [markdown]
# The *summary stats* table shows these inputs for each headwater. You can view them in table form easily...

# %%
summary.head()

# %% [markdown]
# ...and also in a map, where we can see the same headwaters now coloured differently based on the severity of the fire in that area:

# %%
proj.plot_headwaters(example_catchment_name, colour_col='dNBR_mean', table=summary)
