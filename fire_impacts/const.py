
#------- Conversions: --------------------------------------------------
# Basic units:
M_TO_KM = 1e-3
M2_TO_HA = 1e-4
PERCENT_TO_FRACTION = 1e-2
MILLIGRAMS_TO_KILOGRAMS = 1e-6
APPROX_KM_PER_DEGREE = 111
APPROX_DEGREES_TO_METRES = 111000
KG_TO_TONNES = 1e-3

#------ General lookups: -----------------------------------------------
CRS_METRE_UNITS={'m','meter','meters','metre','metres'}
STATS=['mean', 'max', 'min', 'median', 'std']

#------- Hydrology constants: ------------------------------------------
DEFAULT_HW_THRESHOLD=20000
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

#------ Debris flow Constants: -----------------------------------------
HILLSLOPE_AREA = 1.3e4
CHANNELISED_FLOW_THRESHOLD = 1.4e7
SEDIMENT_BULK_DENSITY=1270 # kg/m3
ROCK_BULK_DENSITY=2220 # kg/m3
HILLSLOPE_ROCK_FRACTION=0.12
CHANNEL_ROCK_FRACTION=0.45
HILLSLOPE_PARAMETERS=dict(
    ae=4.5e-4,
    be=0.36,
    ad=0.3*4.5e-4,
    bd=0.36,
    rock=HILLSLOPE_ROCK_FRACTION
    )
CHANNEL_PARAMETERS=dict(
    ae=4.1e-4,
    be=0.52,
    ad=3.7e-7,
    bd=1.06,
    rock=CHANNEL_ROCK_FRACTION
    )
NUM_SIM_YEARS = 2

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
# Key hydrology files:
DEM_FN = 'DEM'
FLOW_ACCUMULATION_FN = 'Flow_accumulation'
FLOW_DIRECTION_FN = 'Flow_direction'
HEADWATERS_FN = 'Headwaters'
SLOPE_FN = 'Slope'
SLOPE_HYDRO_FN = 'Slope_hydro_enforced'
# Intermediate erosion rasters:
ERO_CUM_M_ALL_FN = 'Erosion_cum_mass_all'
ERO_CUM_M_CLY_FN = 'Erosion_cum_mass_clay'
ERO_CUM_M_SED_FN = 'Erosion_cum_mass_sediment'

#------ Standardised field names: --------------------------------------
# Debris flow:
CLY_M_ACC_KG = 'Clay mass accumulation (kg)'
TOT_EM_ACC_KG = 'Total erosion mass accumulation (kg)'
TOT_EM_ACC_KG_HA = 'Total erosion mass accumulation (kg/ha)'
SED_M_ACC_KG = 'Sediment mass accumulation (kg)'
# Headwaters:
HW_ID = 'hw_ID'
HW_ENDP_X = 'X_EndP'
HW_ENDP_Y = 'Y_EndP'
# Debris flow constituents lookup:
PCLE_CTUENT_NAME = 'Particulate constituent'
AVG_CTUENT_MGPKG = 'Average amount (mgkg-1)'
#HFlookup:
HF_ARID_IDX_THRESH = 'AI'
HF_DNBR_THRESH = 'dNBR'
HF_YEARS_THRESH = 'years'
HF_GRADIENT_THRESH = 'slope'
HF_I12_CRIT = 'I12_crit_mean'
# Summary stats (fire_impact_data) table:
adjusted_suffix = '_adjusted'
year_suffix = '_Year_'
ARID_MEAN = 'Aridity_mean'
ARID_MEAN_ADJ = ARID_MEAN + adjusted_suffix
DNBR_MEAN = 'dNBR_mean'
DNBR_MEAN_ADJ = DNBR_MEAN + adjusted_suffix
SLOPE_DEG_MEAN = 'Slope_mean'
SLOPE_DEG_MEAN_ADJ = SLOPE_DEG_MEAN + adjusted_suffix
I12_CRIT_Y = HF_I12_CRIT + year_suffix



#------- Directory folder names: ---------------------------------------
TOPOGRAPHY_FOLDER_NAME = 'Topography'
FIRE_SEVERITY_FOLDER_NAME = 'FireSeverity'
SOILS_FOLDER_NAME = 'Soils'
ERODIBILITY_FOLDER_NAME = 'Erodibility'
DELIVERY_FOLDER_NAME = 'Delivery'
SUBCATCHMENTS_FOLDER_NAME = 'Subcatchments'
PER_CATCHMENT_FOLDERS = [
    TOPOGRAPHY_FOLDER_NAME,
    FIRE_SEVERITY_FOLDER_NAME,
    SOILS_FOLDER_NAME,
    ERODIBILITY_FOLDER_NAME,
    DELIVERY_FOLDER_NAME,
    SUBCATCHMENTS_FOLDER_NAME
    ]