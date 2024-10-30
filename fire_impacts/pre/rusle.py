
from fire_impacts.pre.util import clip_and_reproject_raster, read_raster, read_aligned
import rasterio as rio
from .project import FireImpactsProject
import numpy as np
import logging
logger = logging.getLogger(__name__)

def compute_adjusted_rusle_terms(proj: FireImpactsProject,catchment:str, c_factor_fn: str, k_factor_fn: str):

  if catchment is None:
    proj.for_each_catchment(lambda c: compute_adjusted_rusle_terms(proj, c, c_factor_fn, k_factor_fn))
    return

  # bounds = proj.catchment_bounds(catchment)
  shp = proj.boundary_files[catchment]
  clip_and_reproject_raster(c_factor_fn,shp,proj.catchment_path(catchment,'Erodibility','C_factor.tif'))
  clip_and_reproject_raster(k_factor_fn,shp,proj.catchment_path(catchment,'Erodibility','K_factor.tif'))

  dem_fn = proj.catchment_path(catchment,'Topography','DEM.tif')
  dem, dem_transform, dem_crs = read_raster(dem_fn)
  dNBR = read_aligned(proj.catchment_path(catchment,'FireSeverity','dNBR.tif'), dem_transform, dem_crs, dem.shape)
  Cbase = read_aligned(proj.catchment_path(catchment,'Erodibility','C_factor.tif'), dem_transform, dem_crs, dem.shape)
  Kbase = read_aligned(proj.catchment_path(catchment,'Erodibility','K_factor.tif'), dem_transform, dem_crs, dem.shape)
  AI = read_aligned(proj.catchment_path(catchment,'Soils','Aridity.tif'), dem_transform, dem_crs, dem.shape)

  t = 1
  x_c = 0.4
  x_k = 1
  Kfire = 0.081

  CdNBR = dNBR * 1000
  CdNBR[CdNBR < 0] = 0
  CdNBR[CdNBR > 400] = 0.081
  dNBRmask = (CdNBR > 0) & (CdNBR <= 400)
  CdNBR[dNBRmask] = Cbase[dNBRmask] + ((0.081 - Cbase[dNBRmask]) * (CdNBR[dNBRmask] / 400))

  C = (CdNBR - Cbase) * np.exp(-t / (x_c * AI)) + Cbase
  K = (Kfire - Kbase) * np.exp(-t / (x_k * AI)) + Kbase

  out_meta = {
    'driver': 'GTiff',
    'height': dem.shape[0],
    'width': dem.shape[1],
    'count': 1,
    'dtype': 'float32',
    'crs': dem_crs,
    'transform': dem_transform,
    'compress': 'lzw',
    'nodata': np.nan
  }
  with rio.open(proj.catchment_path(catchment,'Erodibility','C_factor_adjusted.tif'), 'w', **out_meta) as dest:
    dest.write(C,1)

  with rio.open(proj.catchment_path(catchment,'Erodibility','K_factor_adjusted.tif'), 'w', **out_meta) as dest:
    dest.write(K,1)
