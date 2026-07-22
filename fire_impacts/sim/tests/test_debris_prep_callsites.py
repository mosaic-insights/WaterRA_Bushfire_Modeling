"""Regression tests for debris.py call sites left behind by the RunContext
migration.

get_flow_layers()' compute fallback and prep_debris_flow_simulation()'s
dem_to_slope call were still passing (project, catchment) into functions
whose signatures had moved to ctx-first, so both raised as soon as they
were exercised (the fallback path only fires when no saved flow rasters
exist, which the test suite never hit).
"""
from pathlib import Path

import numpy as np

from fire_impacts.pre.tests.util import *  # noqa: F401,F403 - fixtures
from fire_impacts.const import D8_FLOW_DIRECTIONS
from fire_impacts.context import RunContext
from fire_impacts.pre import topography
from fire_impacts.pre.util import read_raster
from fire_impacts.sim import debris


def test_get_flow_layers_compute_fallback(get_project, get_file):
    """With no saved Flow_direction/Flow_accumulation rasters,
    get_flow_layers must compute them from the DEM (and save them)."""
    proj = get_project()
    ctx = RunContext.solo_catchment(proj)
    topography.extract_catchment_dems(ctx, get_file(DEM_FILE))

    dem_path = ctx.catchment_path('Topography', 'DEM.tif')
    _, dem_meta = read_raster(dem_path)
    hydro_dem, grid = topography.hydro_force_dem(dem_path)

    fdir, fdir_meta, facc, facc_meta = debris.get_flow_layers(
        hydro_dem, dem_meta, grid, D8_FLOW_DIRECTIONS, ctx,
    )

    assert np.asarray(fdir).shape == np.asarray(facc).shape
    assert np.asarray(facc).max() > 1  # accumulation actually happened
    # the fallback saves what it computed for next time
    assert Path(
        ctx.catchment_path('Topography', 'Flow_direction.tif')).exists()
    assert Path(
        ctx.catchment_path('Topography', 'Flow_accumulation.tif')).exists()


def test_prep_slope_call_signature(get_project, get_file):
    """prep_debris_flow_simulation computes slope via
    dem_to_slope(ctx, (data, meta), gradient=True, hydro=True, save=False).
    Exercise that exact call shape."""
    proj = get_project()
    ctx = RunContext.solo_catchment(proj)
    topography.extract_catchment_dems(ctx, get_file(DEM_FILE))

    dem_path = ctx.catchment_path('Topography', 'DEM.tif')
    dem_data, dem_meta = read_raster(dem_path)

    slope, meta = topography.dem_to_slope(
        ctx, (dem_data, dem_meta), gradient=True, hydro=True, save=False,
    )
    assert slope.shape == dem_data.shape
    assert np.nanmax(slope) > 0
    # save=False must not write either slope output
    assert not Path(
        ctx.catchment_path('Topography', 'Slope.tif')).exists()
