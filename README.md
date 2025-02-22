# Bushfire impacts on water quality

This Python package contains functionality for modelling the impacts of bushfire on the water quality of catchment runoff. The functionality is packaged as a library, intended to be used from a Python data science environment, such as Jupyter or Spyder, or incorporated into other scripts.

The library includes functionality for simulating erosion processes and debris flow and includes data pre-processing functionality intended to work with commonly available datasets (DEM-H, Soil and Landscape Grid of Australia, Sentinel-2 derived dNBR).

The package is designed for, and is being tested on, Australian conditions.


## Installation

The library can be installed using `pip` and assumes that you have a functioning 'scientific Python' installation, such as you might get by installing Anaconda Python.

### Install core library

1. Download the provided zip file and unzip to a convenient location.
2. Open a command prompt with your Python data science environment activated (eg open 'Anaconda Command Prompt')
3. Switch to the installation directory and run

 ```
 pip install .
 ```

 Alternatively, to keep the downloaded copy of the code editable after installation, use:

 ```
 pip install -e .
 ```


### Dependencies

We have assumed the user has an existing Python distribution including the most common libraries for scientific and numerical computing (eg `numpy`, `scipy`, `pandas`). Additiona dependencies are listed in `requirements.txt` and can be installed using conda from the command prompt:

```
conda install --yes --file requirements.txt
```

In environments without conda, install with pip:

```
pip install -r requirements.txt
```

## Usage

The library supports two different modes:

1. A **high level interface**, where the library manages key datasets in a standard directory structure, and
2. A **low level interface**, where the user is responsible for data management.

We anticipated that the high level interface will suit most people, most of the time.

### Data requirements

The following table lists the key data requirements for the library. The user must provide a catchment boundary and have local access to the DEM-H for their area. The hihg level interface automatically retrieves the other data sources from published web services.

| Data | Source |
|------|--------|
| Catchment / study area boundary | User provided |
| DEM | DEM-H (local file) |
| Soils | Soil and Landscape Grid of Australia (CSIRO web service) |
| dNBR | Sentinel 2 (DEA Web service) |

### High level interface

The high level interface is implemented through the `FireImpactsProject` class, which creates and manages a folder structure containing the relevant data inputs to the water quality analyses, included processed input data and simulation results.

```python
from fire_impacts import FireImpactsProject
project = FireImpactsProject('./my-project')
```

A single `FireImpactsProject` can manage data associated with one or catchment areas, provided initially as catchment boundaries:

```python
project.add_catchment('big-river-catchment-boundary-boundary.shp',name='Big-River')
```

As you proceed through the data pre-processing and simulation steps, the `FireImpactsProject` will build a directory stucture containing the relevant, processed data:

```
my-project
└── Catchments
    └── Big-River
        ├── Delivery
        ├── Erodibility
        ├── FireSeverity
        ├── Soils
        │   ├── BULK_DENSITY
        │   ├── CLAY
        │   ├── SAND
        │   └── SILT
        └── Topography
```

All pre-processing and simulation functions in the high level interface accept a `FireImpactsProject` as the first parameter. For example:

``` python
def extract_headwaters(project:FireImpactsProject,
                       name:str=None,
                       threshold_m2:float=DEFAULT_HW_THRESHOLD,
                       crs_unit_to_metres:float=None):
```

The high level interface automatically harmonises data wherever possible, bringing different imported datasets into a common coordinate reference system and resolution and clipping datasets to the relevant catchment boundaries.

Internally, the high level interface calls the underlying functionality from the low level interface.

### Low level interface

The low level interface provides access to the core functionality of the library while leaving the user/caller to manage data storage, I/O and harmonisation.

**Note:** The function calls in the low level interface are not yet consistent with each other and, as a result, are very likely to change as we refine the library.


### Worked example

A worked example, showing usage of the high level interface, is provided in the [examples/PrepareData.ipynb](examples/PrepareData.ipynb).


### Status

The following table summaries the status of each component of the library

| Stage | Functionality | Initial import | High level interface | Low level interface | Case study 1 | Case study 1 validated | Case study 2 | Case study 2 validated |
|-------------|-------|-------------|--------------------|---------------------|------------|--------------------|------------|------------|
| **Pre-processing** | Topographic | :heavy_check_mark: | :heavy_check_mark: | :construction: | :heavy_check_mark: | :heavy_check_mark:| | |
| | Fire severity | :heavy_check_mark: | :heavy_check_mark: | :construction: | :heavy_check_mark: | | | |
| | Soils | :heavy_check_mark: | :heavy_check_mark: | :construction: | :heavy_check_mark: | | | |
| | Erodibility | :heavy_check_mark: | :heavy_check_mark: | :construction: | :heavy_check_mark: | | | |
| | Stochastic Rainfall | :heavy_check_mark: | | :construction: | | | | |
| **Simulation** | Erosion | :heavy_check_mark: | :heavy_check_mark: | :construction: | :heavy_check_mark: | | | |
| | Debris | :heavy_check_mark: | | :construction: | :heavy_check_mark: | | | |

**Note:** The low level interface is very likely to change.

## Organisation

The core library code is stored in `fire_impacts` directory. The code repository also includes key parameter files, examples and test data.

| Directory | Contents |
|-----------|----|
| `<top-level>` | |
| `├── data` | Common parameter files (eg concentrations of pollutants in ash and debris) |
| `├── examples` | Worked example files |
| `├── test_data` | Small spatial datasets to support examples and unit tests |
| `└── fire_impacts` | Library code |
| `    ├── pre` | |
| `    │   └── tests` | |
| `    └── sim` | |


## Funding, development and support

This project is funded by a consortium of Australian water utilities through Water Research Australia and by the National Emergency Management Agency.

The project was undertaken by Alluvium Consulting with support from Flow Matters.

For questions relating to the use of the library, please contact Joel Rahman (joel@flowmatters.com.au).

