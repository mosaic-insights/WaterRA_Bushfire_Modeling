"""
Example usage of fire_impacts.source module to configure Source models.

This script demonstrates how to replicate the functionality from the 
assign_model.ipynb and import_timeseries.ipynb notebooks using the
extracted functions.
"""

import logging
from fire_impacts.source import (
    connect_to_veneer,
    detect_constituent,
    detect_functional_unit,
    configure_load_distributor_model,
    load_fire_impact_data,
    create_veneer_data_sources,
    assign_fire_sediment_timeseries,
    assign_rainfall_timeseries,
    run_model_simulation,
    save_model,
    configure_source_model_with_fire_data
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_assign_model_workflow():
    """
    Replicate the assign_model.ipynb notebook workflow.
    """
    logger.info("=== Assign Model Workflow ===")
    
    # Connect to Veneer
    v = connect_to_veneer(port=9877)
    
    # Get scenario info (for verification)
    scenario_info = v.scenario_info()
    logger.info(f"Connected to scenario: {scenario_info}")
    
    # Auto-detect constituent and functional unit
    constituent = detect_constituent(v)
    functional_unit = detect_functional_unit(v)
    logger.info(f"Using constituent: {constituent}, functional_unit: {functional_unit}")
    
    # Configure load distributor model
    configure_load_distributor_model(
        v, 
        constituent=constituent,
        load_attenuation=10.0,
        maximum_concentration=1000.0
    )
    
    # Save the configured model
    save_model(v, 'Avon_0.36_with_load_distributor.rsproj')
    
    logger.info("Model assignment workflow completed")


def example_import_timeseries_workflow():
    """
    Replicate the import_timeseries.ipynb notebook workflow.
    """
    logger.info("=== Import Timeseries Workflow ===")
    
    # Connect to Veneer
    v = connect_to_veneer(port=9880)
    
    # Auto-detect constituent and functional unit
    constituent = detect_constituent(v)
    functional_unit = detect_functional_unit(v)
    logger.info(f"Using constituent: {constituent}, functional_unit: {functional_unit}")
    
    # Load fire impact data
    tss_data, rainfall_data = load_fire_impact_data(
        rusle_csv_path='../rusle_hourly_load_per_catchment.csv',
        debris_flow_csv_path='../debris_flow_hourly_load_per_catchment.csv',
        rainfall_csv_path='../hourly_rainfall.csv'
    )
    
    # Create data sources in Veneer
    create_veneer_data_sources(v, tss_data, rainfall_data)
    
    # Assign TSS time series
    assign_fire_sediment_timeseries(v, constituent=constituent, functional_unit=functional_unit)
    
    # Assign rainfall time series
    assign_rainfall_timeseries(v)
    
    # Run a test simulation
    results = run_model_simulation(
        v,
        start_date='01/01/1900',
        end_date='31/12/1901'
    )
    
    # Check results
    if results['Status'] == 'Finished':
        logger.info("Simulation completed successfully")
        if 'Results' in results:
            df = results['Results'].as_dataframe()
            logger.info(f"Results shape: {df.shape}")
    else:
        logger.warning(f"Simulation failed with status: {results['Status']}")
    
    # Save the model with sediment inputs
    save_model(v, 'Avon_0.36_with_sediment_inputs.rsproj')
    
    logger.info("Timeseries import workflow completed")


def example_complete_workflow_with_auto_detection():
    """
    Demonstrate the complete workflow using the high-level function with auto-detection.
    """
    logger.info("=== Complete Workflow with Auto-Detection ===")
    
    results = configure_source_model_with_fire_data(
        port=9880,
        rusle_csv_path='../rusle_hourly_load_per_catchment.csv',
        debris_flow_csv_path='../debris_flow_hourly_load_per_catchment.csv',
        rainfall_csv_path='../hourly_rainfall.csv',
        output_model_name='Avon_complete_fire_model.rsproj',
        # constituent and functional_unit left as None to auto-detect
        load_attenuation=10.0,
        maximum_concentration=1000.0,
        run_simulation=True
    )
    
    logger.info(f"Complete workflow results: {results}")
    logger.info(f"Detected constituent: {results.get('constituent')}")
    logger.info(f"Detected functional_unit: {results.get('functional_unit')}")


def example_complete_workflow_with_explicit_params():
    """
    Demonstrate the complete workflow with explicit constituent and functional unit.
    """
    logger.info("=== Complete Workflow with Explicit Parameters ===")
    
    results = configure_source_model_with_fire_data(
        port=9880,
        rusle_csv_path='../rusle_hourly_load_per_catchment.csv',
        debris_flow_csv_path='../debris_flow_hourly_load_per_catchment.csv',
        rainfall_csv_path='../hourly_rainfall.csv',
        output_model_name='Avon_complete_fire_model_explicit.rsproj',
        constituent='TSS',  # Explicitly specify
        functional_unit='Forested',  # Explicitly specify
        load_attenuation=10.0,
        maximum_concentration=1000.0,
        run_simulation=True
    )
    
    logger.info(f"Complete workflow results: {results}")


if __name__ == "__main__":
    # Uncomment the workflow you want to run
    
    # Run the model assignment workflow (equivalent to assign_model.ipynb)
    # example_assign_model_workflow()
    
    # Run the timeseries import workflow (equivalent to import_timeseries.ipynb)  
    # example_import_timeseries_workflow()
    
    # Run the complete workflow with auto-detection
    # example_complete_workflow_with_auto_detection()
    
    # Run the complete workflow with explicit parameters
    # example_complete_workflow_with_explicit_params()
    
    print("Example script ready. Uncomment the desired workflow function to run.")