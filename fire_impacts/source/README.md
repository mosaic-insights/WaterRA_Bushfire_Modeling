# Fire Impacts Source Model Configuration

This module provides functions to configure eWater Source models with fire impact data using the Veneer API. The functionality has been extracted from the original `assign_model.ipynb` and `import_timeseries.ipynb` notebooks for better reusability and maintainability.

## Overview

The module provides both low-level functions for specific operations and a high-level workflow function for complete model configuration. The main components are:

- **veneer_config.py**: Core functions for Source model configuration
- **utils.py**: Utility functions for data validation and processing  
- **example_usage.py**: Example scripts showing how to use the functions

## Key Features

### Auto-Detection of Model Parameters

The module can automatically detect suitable constituents and functional units from your Source model:

```python
from fire_impacts.source import configure_source_model_with_fire_data

# Auto-detection example - constituent and functional_unit are determined automatically
results = configure_source_model_with_fire_data(
    port=9880,
    rusle_csv_path='path/to/rusle_data.csv',
    debris_flow_csv_path='path/to/debris_flow_data.csv', 
    rainfall_csv_path='path/to/rainfall_data.csv',
    output_model_name='configured_model.rsproj',
    run_simulation=True
    # constituent and functional_unit default to None (auto-detect)
)

print(f"Used constituent: {results['constituent']}")
print(f"Used functional_unit: {results['functional_unit']}")
```

### Explicit Parameter Specification

You can also explicitly specify the constituent and functional unit:

```python
results = configure_source_model_with_fire_data(
    port=9880,
    rusle_csv_path='path/to/rusle_data.csv',
    debris_flow_csv_path='path/to/debris_flow_data.csv', 
    rainfall_csv_path='path/to/rainfall_data.csv',
    constituent='TSS',  # Explicitly specified
    functional_unit='Forested',  # Explicitly specified
    output_model_name='configured_model.rsproj',
    run_simulation=True
)
```

## Auto-Detection Behavior

When `constituent` or `functional_unit` parameters are `None`, the module uses the following logic:

### Constituent Detection
1. Queries the model for available constituents using `v.model.get_constituents()`
2. Searches for matches against the `LIKELY_CONSTITUENTS` list (in order): `['TSS', 'Sediment', 'Contaminant', 'Pollutant']`
3. Uses the first match found
4. If no match found, uses the first available constituent from the model
5. Logs the selection process for debugging

### Functional Unit Detection
1. Queries the model for available functional units using `v.model.catchment.functional_units()`
2. Filters out 'Water' as it's typically not a target for sediment/fire data
3. Searches for matches against the `LIKELY_FUNCTIONAL_UNITS` list (in order): `['Forested', 'Forest', 'Urban', 'Bushland', 'Burned', 'Native Vegetation']`
4. Uses the first match found
5. If no match found, uses the first available functional unit
6. Logs the selection process for debugging

## Customizing Auto-Detection Candidates

You can modify the default candidate lists if needed:

```python
from fire_impacts.source import (
    detect_constituent,
    detect_functional_unit,
    LIKELY_CONSTITUENTS,
    LIKELY_FUNCTIONAL_UNITS
)

# View current candidates
print(LIKELY_CONSTITUENTS)
print(LIKELY_FUNCTIONAL_UNITS)

# Use custom candidates for detection
v = connect_to_veneer(9880)
custom_constituent = detect_constituent(v, candidate_list=['MyConstituent', 'TSS', 'Sediment'])
custom_fu = detect_functional_unit(v, candidate_list=['MyFU', 'Forested', 'Urban'])
```

## Key Functions

### High-Level Workflow

```python
from fire_impacts.source import configure_source_model_with_fire_data

# Complete workflow in one function call (with auto-detection)
results = configure_source_model_with_fire_data(
    port=9880,
    rusle_csv_path='path/to/rusle_data.csv',
    debris_flow_csv_path='path/to/debris_flow_data.csv', 
    rainfall_csv_path='path/to/rainfall_data.csv',
    output_model_name='configured_model.rsproj',
    run_simulation=True
)
```

### Detection Functions

```python
from fire_impacts.source import detect_constituent, detect_functional_unit

v = connect_to_veneer(9880)
constituent = detect_constituent(v)
functional_unit = detect_functional_unit(v)
```

### Step-by-Step Configuration

```python
from fire_impacts.source import (
    connect_to_veneer,
    configure_load_distributor_model,
    load_fire_impact_data,
    create_veneer_data_sources,
    assign_fire_sediment_timeseries,
    assign_rainfall_timeseries,
    run_model_simulation,
    save_model
)

# Step 1: Connect to Veneer
v = connect_to_veneer(port=9880)

# Step 2: Configure load distributor model  
configure_load_distributor_model(v, load_attenuation=10.0, maximum_concentration=1000.0)

# Step 3: Load fire impact data
tss_data, rainfall_data = load_fire_impact_data(
    'rusle_data.csv', 'debris_flow_data.csv', 'rainfall_data.csv'
)

# Step 4: Create data sources in Veneer
create_veneer_data_sources(v, tss_data, rainfall_data)

# Step 5: Assign time series data
assign_fire_sediment_timeseries(v, functional_unit='Forested')
assign_rainfall_timeseries(v)

# Step 6: Run simulation (optional)
results = run_model_simulation(v, start_date='01/01/1900', end_date='31/12/1901')

# Step 7: Save model
save_model(v, 'configured_model.rsproj')
```

## Data Requirements

The functions expect CSV files with specific formats:

### RUSLE and Debris Flow Data
- Index: DateTime (hourly data)
- Columns: Catchment names (e.g., 'AVSC001', 'AVSC002', etc.)
- Units: kg/h (sediment load)

### Rainfall Data  
- Index: DateTime (hourly data)
- Columns: Catchment names matching the sediment data
- Units: mm/h

## Utility Functions

The module also provides utility functions for data validation and model inspection:

```python
from fire_impacts.source import (
    validate_csv_files,
    verify_veneer_connection,
    get_model_configuration_summary,
    summarize_dataframe
)

# Validate input files before processing
validate_csv_files('data1.csv', 'data2.csv', 'data3.csv')

# Get model configuration summary
v = connect_to_veneer(9880)
config = get_model_configuration_summary(v)
```

## Error Handling

All functions include comprehensive error handling and logging. Set up logging to see detailed progress information:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Original Notebook Workflows

### assign_model.ipynb Equivalent
The `configure_load_distributor_model()` function replaces the notebook workflow that:
1. Connected to Veneer
2. Examined existing TSS generation models  
3. Set all functional units to use Load Distributor model
4. Configured LoadAttenuation and MaximumConcentration parameters
5. Saved the model

### import_timeseries.ipynb Equivalent  
The data loading and assignment functions replace the notebook workflow that:
1. Connected to Veneer
2. Loaded RUSLE, debris flow, and rainfall CSV data
3. Combined and processed TSS data
4. Created Veneer data sources
5. Assigned TSS data to Forested functional units
6. Assigned rainfall to all functional units
7. Ran a test simulation
8. Saved the configured model

## Dependencies

- veneer: eWater Veneer Python API
- pandas: Data manipulation
- logging: Progress and error reporting

## Example Usage

See `example_usage.py` for complete working examples that replicate the original notebook workflows using the extracted functions.