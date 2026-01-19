
M_TO_KM=1e-3

M2_TO_HA=1e-4

PERCENT_TO_FRACTION=1e-2

MILLIGRAMS_TO_KILOGRAMS=1e-6
APPROX_KM_PER_DEGREE = 111

STATS=['mean', 'max', 'min', 'median', 'std']

DEFAULT_HW_THRESHOLD=20000
APPROX_DEGREES_TO_METRES=111000
D8_FLOW_DIRECTIONS = (
    64, #north
    128, #northeast
    1, #east
    2, #southeast
    4, #south
    8, #southwest
    16, #west
    32 #northwest
    )
FLOW_ROUTING_TYPE = 'd8'
CRS_METRE_UNITS={'m','meter','meters','metre','metres'}


#------ Dtype standards: -----------------------------------------------

# Convert numpy one-character dtype.kind attributes into more general 
#descriptors that will map in default_dtypes_raster:
numpy_kind_to_desc = {
    'i': 'int',
    'u': 'int',
    'f': 'float'
    }
NODATA_VAL_INT = -9999

#------ Standardised file names: ---------------------------------------
DEM_FN = 'DEM'
FLOW_ACCUMULATION_FN = 'Flow_accumulation'
FLOW_DIRECTION_FN = 'Flow_direction'
HEADWATERS_FN = 'Headwaters'
SLOPE_FN = 'Slope'
SLOPE_HYDRO_FN = 'Slope_hydro_enforced'

#------- Directory folder names: ---------------------------------------
PER_CATCHMENT_FOLDERS = [
    'Topography',
    'FireSeverity',
    'Soils',
    'Erodibility',
    'Delivery',
    'Subcatchments'
]