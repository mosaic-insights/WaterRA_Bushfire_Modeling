# Installing WaterRA Bushfire Modeling on Windows

This guide provides step-by-step instructions for installing the WaterRA Bushfire Modeling package on Windows using Miniforge and the `env-base313.yml` environment file.

## Prerequisites

- Windows 10 or later

## Step 1: Install Miniforge

### Download Miniforge3

1. Visit [https://github.com/conda-forge/miniforge/releases](https://github.com/conda-forge/miniforge/releases)
2. Download `Miniforge3-Windows-x86_64.exe` for Windows

### Install Miniforge3

1. Run the downloaded installer (no administrator privileges required)
2. When prompted, choose **"Install for the current user only"** (this is the recommended option)
3. Choose "Add Miniforge3 to PATH" during installation
4. Complete the installation with default settings

### Verify Installation

Open Command Prompt or PowerShell and run:

```powershell
mamba --version
```

This should display mamba version information.

## Step 2: Create Clean Environment from env-base313.yml

### Navigate to Project Directory

```powershell
cd "c:\src\projects\WaterRA_Bushfire_Modeling"
```

### Create Environment

```powershell
mamba env create -f env-base313.yml
```

This creates a new environment named `bushfire-py313` with:

- Python 3.13
- Scientific computing stack (numpy 2.4.2, pandas 2.3.3, scipy 1.17.0)
- Geospatial libraries (rasterio 1.5.0, geopandas 1.1.2)
- Jupyter Lab with widgets support
- Additional dependencies including dea-tools and custom pysheds

### Activate Environment

```powershell
mamba activate bushfire-py313
```

## Step 3: Install the Fire Impacts Package

### Install in Development Mode

```powershell
pip install -e .
```

This installs the `fire_impacts` package in editable mode, allowing for code modifications without reinstallation.

### Verify Installation

```powershell
python -c "import fire_impacts; print('Package installed successfully')"
```

### Test CLI Tool

```powershell
fire-impacts --help
```

## Step 4: Verify Environment Setup

### Launch Jupyter Lab

```powershell
jupyter lab
```

### Test Key Libraries

Open a new notebook and run:

```python
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import fire_impacts

print("All libraries imported successfully!")
```

## Important Notes

- **Environment Management:** The `env-base313.yml` uses Python 3.13 with carefully pinned versions to avoid compatibility issues
- **Special Dependencies:** 
  - `dea-tools` requires `pg_config` during installation (handled automatically via pip)
  - `pysheds` is installed from a custom GitHub repository
- **Development vs Production:** Use `-e` flag with pip install for development work, omit for production deployments
- **Windows-Specific:** All paths use Windows conventions, and the environment has been tested on Windows systems

## Troubleshooting

If you encounter package compatibility issues, refer to [ENVIRONMENT_FIX_GUIDE.md](ENVIRONMENT_FIX_GUIDE.md) which provides solutions for common numpy/pandas version conflicts.

## Testing Your Installation

The environment includes comprehensive testing capabilities with pytest and example notebooks in the `examples/` directory to validate your installation.

### Run Example Notebooks

Navigate to the examples directory and try running the provided notebooks:

- `PrepareData.ipynb` - Data preprocessing workflows
- `Severity.ipynb` - Fire severity analysis
- `Simulation.ipynb` - Bushfire impact simulation
- `Topography.ipynb` - Topographic analysis

### Run Tests

```powershell
pytest
```

This will run the test suite to verify all components are working correctly.

## Next Steps

Once installation is complete, you can:

1. Explore the example notebooks in the `examples/` directory
2. Review the package documentation for API reference
3. Begin working with your own data using the fire impacts modeling tools

For questions or issues, please refer to the project documentation or contact the maintainers.
