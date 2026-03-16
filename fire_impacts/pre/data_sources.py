

DEA_STAC="https://explorer.dea.ga.gov.au/stac"
DEA_STATUS_URL='https://status.dea.ga.gov.au/'
TERN_SLGA_STAC='https://data.tern.org.au/model-derived/slga/NationalMaps/SoilAndLandscapeGrid/catalog.json'
ASRIS_WCS='https://www.asris.csiro.au/arcgis/services/TERN/${LAYER}_ACLEP_AU_NAT_C/MapServer/WCSServer?SERVICE=WCS&REQUEST=GetCapabilities'
STOCHASTIC_RAINFALL_API='https://stochastic-rain.apps.hydrograph.au/rainfall'

USLE_GRIDS='https://doi.org/10.4225/08/582cef2dd5966' # No services

# DEMH_WCS='https://elevation.fsdf.org.au/geoserver/wcs?SERVICE=WCS&REQUEST=GetCapabilities'
DEMH='https://dea-public-data.s3-ap-southeast-2.amazonaws.com/projects/elevation/ga_srtm_dem1sv1_0/demh1sv1_0.tif'

CSIRO_C_FACTOR_GRID='https://bushfire.blob.core.windows.net/bushfire/c_factor_g94.tif'
CSIRO_K_FACTOR_GRID='https://bushfire.blob.core.windows.net/bushfire/k_factor_g94.tif'
ARIDITY_GRID_COARSE='https://bushfire.blob.core.windows.net/bushfire/Aridity_PT_rs.tif'

SENTINEL_2_COLLECTIONS=('ga_s2am_ard_3','ga_s2bm_ard_3')
LANDSAT_COLLECTIONS=('ga_ls5t_ard_3', 'ga_ls7e_ard_3', 'ga_ls8c_ard_3', 'ga_ls9c_ard_3')
# DEA ARD Landsat collections; you can tweak this if you want fewer sensors
DEA_LANDCOVER = "https://thredds.nci.org.au/thredds/fileServer/jw04/ga_ls_landcover_class_cyear_3/2-0-0/continental_mosaics"

# Pre-clipped reference dNBR rasters for synthetic fire generation.
# These are real fires with NaN outside the burned area, hosted via HTTPS
# so rasterio/GDAL can read them directly without downloading.
SYNTHETIC_FIRE_MEDIUM_DNBR = 'https://bushfire.blob.core.windows.net/bushfire/reference_fires/Medium_dnbr_clipped.tif'
SYNTHETIC_FIRE_HIGH_DNBR = 'https://bushfire.blob.core.windows.net/bushfire/reference_fires/High_dnbr_clipped.tif'