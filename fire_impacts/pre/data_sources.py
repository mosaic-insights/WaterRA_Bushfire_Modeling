"""
External data source URLs and collection identifiers used in pre-processing.

All values are string constants pointing to public web services or
cloud-hosted datasets.  Import this module and reference these names
rather than hard-coding URLs elsewhere in the package.
"""

# ---------------------------------------------------------------------------
# Digital Earth Australia (DEA)
# ---------------------------------------------------------------------------

DEA_STAC = "https://explorer.dea.ga.gov.au/stac"
DEA_STATUS_URL = "https://status.dea.ga.gov.au/"

# Sentinel-2 and Landsat ARD collection identifiers on DEA STAC
SENTINEL_2_COLLECTIONS = ("ga_s2am_ard_3", "ga_s2bm_ard_3")
LANDSAT_COLLECTIONS = (
    "ga_ls5t_ard_3",
    "ga_ls7e_ard_3",
    "ga_ls8c_ard_3",
    "ga_ls9c_ard_3",
)

DEA_LANDCOVER = (
    "https://thredds.nci.org.au/thredds/fileServer/jw04"
    "/ga_ls_landcover_class_cyear_3/2-0-0/continental_mosaics"
)

# ---------------------------------------------------------------------------
# Soil and landscape data
# ---------------------------------------------------------------------------

# TERN Soil and Landscape Grid of Australia (SLGA) STAC catalogue
TERN_SLGA_STAC = (
    "https://data.tern.org.au/model-derived/slga"
    "/NationalMaps/SoilAndLandscapeGrid/catalog.json"
)

# ASRIS WCS — substitute ${LAYER} with the desired layer name
ASRIS_WCS = (
    "https://www.asris.csiro.au/arcgis/services/TERN"
    "/${LAYER}_ACLEP_AU_NAT_C/MapServer/WCSServer"
    "?SERVICE=WCS&REQUEST=GetCapabilities"
)

# ---------------------------------------------------------------------------
# RUSLE factor grids
# ---------------------------------------------------------------------------

# USLE raster grids (DOI landing page — no direct web service available)
USLE_GRIDS = "https://doi.org/10.4225/08/582cef2dd5966"

# CSIRO-hosted C-factor and K-factor grids aligned to the g94 grid
CSIRO_C_FACTOR_GRID = (
    "https://bushfire.blob.core.windows.net/bushfire/c_factor_g94.tif"
)
CSIRO_K_FACTOR_GRID = (
    "https://bushfire.blob.core.windows.net/bushfire/k_factor_g94.tif"
)

# Coarse-resolution aridity grid used for rainfall-erosivity adjustment
ARIDITY_GRID_COARSE = (
    "https://bushfire.blob.core.windows.net/bushfire/Aridity_PT_rs.tif"
)

# ---------------------------------------------------------------------------
# Elevation
# ---------------------------------------------------------------------------

# SRTM DEM-H 1-second mosaic (cloud-optimised GeoTIFF on AWS)
DEMH = (
    "https://dea-public-data.s3-ap-southeast-2.amazonaws.com"
    "/projects/elevation/ga_srtm_dem1sv1_0/demh1sv1_0.tif"
)

# ---------------------------------------------------------------------------
# Stochastic rainfall API
# ---------------------------------------------------------------------------

STOCHASTIC_RAINFALL_API = (
    "https://stochastic-rain.apps.hydrograph.au/rainfall"
)

# ---------------------------------------------------------------------------
# Synthetic fire reference dNBR rasters
# ---------------------------------------------------------------------------

# Pre-clipped reference dNBR rasters for synthetic fire generation.
# These are real fires with NaN outside the burned area, hosted via
# HTTPS so rasterio/GDAL can read them directly without downloading.
SYNTHETIC_FIRE_MEDIUM_DNBR = (
    "https://bushfire.blob.core.windows.net/bushfire"
    "/reference_fires/Medium_dnbr_clipped.tif"
)
SYNTHETIC_FIRE_HIGH_DNBR = (
    "https://bushfire.blob.core.windows.net/bushfire"
    "/reference_fires/High_dnbr_clipped.tif"
)
