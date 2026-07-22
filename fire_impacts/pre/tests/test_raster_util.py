"""Unit tests for the consolidated raster I/O helpers in pre/util.py."""
import os

import numpy as np
import rasterio as rio
from rasterio.transform import from_origin

from .util import *  # noqa: F401,F403 - fixtures (get_file) + data names

from fire_impacts.pre.util import (
    RasterGrid,
    read_raster,
    read_raster_masked,
    read_aligned,
    read_aligned_like,
    write_raster,
    slope_from_dem,
    clip_raster,
    condition_dem,
    breach_pits,
    upslope_weighted_mean,
    _dem_with_real_nodata,
    PYSHEDS_PIT_CODE,
)

CRS = 'EPSG:7855'


def _write_tif(path, data, nodata, dtype='float32',
               transform=from_origin(500000, 6000000, 10, 10)):
    meta = {
        'driver': 'GTiff', 'height': data.shape[0], 'width': data.shape[1],
        'count': 1, 'dtype': dtype, 'crs': CRS, 'transform': transform,
        'nodata': nodata,
    }
    with rio.open(path, 'w', **meta) as dst:
        dst.write(data.astype(dtype), 1)
    return path


# --- read_raster_masked ----------------------------------------------------

def test_read_raster_masked_numeric_nodata(tmp_path):
    data = np.arange(12, dtype='float32').reshape(3, 4)
    data[0, 0] = -9999
    fn = _write_tif(str(tmp_path / 'a.tif'), data, nodata=-9999)

    grid = read_raster_masked(fn)
    assert np.isnan(grid.data[0, 0])
    assert grid.nodata_mask[0, 0]
    assert grid.nodata_mask.sum() == 1
    assert np.array_equal(grid.data[grid.data > 0], data[data > 0])


def test_read_raster_masked_nan_nodata_is_detected(tmp_path):
    """With NaN nodata (the standard project output), the nodata cells are
    detected via isnan rather than the `data == nodata` test (which never
    matches NaN and would report an empty mask). The data still carries NaN
    at those cells and valid cells are unchanged."""
    data = np.ones((3, 4), dtype='float32')
    data[1, 1] = np.nan
    fn = _write_tif(str(tmp_path / 'b.tif'), data, nodata=np.nan)

    grid = read_raster_masked(fn)
    assert grid.nodata_mask[1, 1]
    assert grid.nodata_mask.sum() == 1
    assert np.isnan(grid.data[1, 1])
    assert grid.data[0, 0] == 1.0


def test_read_raster_masked_no_nodata(tmp_path):
    data = np.ones((2, 2), dtype='float32')
    fn = _write_tif(str(tmp_path / 'c.tif'), data, nodata=None)

    grid = read_raster_masked(fn)
    assert not grid.nodata_mask.any()
    assert np.array_equal(grid.data, data)


def test_raster_grid_properties(tmp_path):
    data = np.ones((3, 4), dtype='float32')
    fn = _write_tif(str(tmp_path / 'd.tif'), data, nodata=np.nan)
    grid = read_raster_masked(fn)

    assert grid.shape == (3, 4)
    assert grid.xres == 10
    assert grid.yres == 10
    assert grid.pixel_area == 100

    meta = grid.meta()
    assert meta['dtype'] == 'float32'
    assert np.isnan(meta['nodata'])
    assert meta['height'] == 3 and meta['width'] == 4
    assert meta['transform'] == grid.transform
    # overrides pass through
    assert grid.meta(nodata=0.0)['nodata'] == 0.0


# --- write_raster ----------------------------------------------------------

def test_write_raster_roundtrip(tmp_path):
    data = np.arange(6, dtype='float64').reshape(2, 3)
    template = np.zeros((2, 3), dtype='float32')
    fn = _write_tif(str(tmp_path / 'tpl.tif'), template, nodata=np.nan)
    _, meta = read_raster(fn)

    out = str(tmp_path / 'sub' / 'out.tif')  # parent dir doesn't exist yet
    write_raster(out, data, meta)

    with rio.open(out) as src:
        assert src.dtypes[0] == 'float32'
        assert np.isnan(src.nodata)
        assert src.profile['compress'] == 'lzw'
        assert np.array_equal(src.read(1), data.astype('float32'))


def test_write_raster_dtype_and_nodata_overrides(tmp_path):
    data = np.full((2, 2), 7, dtype='int64')
    template = np.zeros((2, 2), dtype='float32')
    fn = _write_tif(str(tmp_path / 'tpl.tif'), template, nodata=np.nan)
    _, meta = read_raster(fn)

    out = str(tmp_path / 'out_int.tif')
    write_raster(out, data, meta, dtype='int32', nodata=-9999)
    with rio.open(out) as src:
        assert src.dtypes[0] == 'int32'
        assert src.nodata == -9999
        assert np.array_equal(src.read(1), data.astype('int32'))


def test_write_raster_fills_masked_arrays_with_nodata(tmp_path):
    """Regression: a masked array (as returned by read_aligned) must be
    filled with the output nodata value. Dropping the mask exposes the
    undefined values underneath — this showed up as finite K factors
    outside the catchment boundary."""
    template = np.zeros((2, 2), dtype='float32')
    fn = _write_tif(str(tmp_path / 'tpl.tif'), template, nodata=np.nan)
    _, meta = read_raster(fn)

    garbage_under_mask = np.ma.masked_array(
        data=np.array([[1.0, 123.0], [3.0, 4.0]]),
        mask=[[False, True], [False, False]],
    )
    out = str(tmp_path / 'masked.tif')
    write_raster(out, garbage_under_mask, meta)

    with rio.open(out) as src:
        result = src.read(1)
    assert np.isnan(result[0, 1])
    assert result[0, 0] == 1.0


# --- read_aligned_like -----------------------------------------------------

def test_read_aligned_like_matches_read_aligned(tmp_path):
    target = np.zeros((4, 4), dtype='float32')
    target_fn = _write_tif(str(tmp_path / 'target.tif'), target,
                           nodata=np.nan)
    src_data = np.arange(4, dtype='float32').reshape(2, 2)
    src_fn = _write_tif(
        str(tmp_path / 'src.tif'), src_data, nodata=np.nan,
        transform=from_origin(500000, 6000000, 20, 20))

    like = read_raster_masked(target_fn)
    a = read_aligned(src_fn, like.transform, like.crs, like.shape)
    b = read_aligned_like(src_fn, like)
    assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


# --- slope_from_dem --------------------------------------------------------

def test_slope_from_dem_on_a_uniform_plane():
    # Elevation increases 1 m per cell along axis 0 with 10 m cells:
    # rise/run = 0.1 everywhere, no cross-slope component.
    dem = np.arange(5, dtype='float64')[:, None] * np.ones(4)
    slope_ratio, dz_dx, dz_dy = slope_from_dem(dem, 10.0, 10.0)
    assert np.allclose(slope_ratio, 0.1)
    assert np.allclose(dz_dx, 0.1)
    assert np.allclose(dz_dy, 0.0)


def test_slope_from_dem_propagates_nan():
    dem = np.ones((4, 4))
    dem[2, 2] = np.nan
    slope_ratio, _, _ = slope_from_dem(dem, 10.0, 10.0)
    # np.gradient spreads the NaN to the neighbours whose central
    # difference straddles it (the NaN cell itself differences its two
    # finite neighbours, so it comes back finite)
    assert np.isnan(slope_ratio[1, 2])
    assert np.isnan(slope_ratio[2, 1])
    assert np.isfinite(slope_ratio[0, 0])


# --- DEM conditioning: nodata sentinel + breach ----------------------------

def _pysheds_raster(arr, nodata):
    """Build a pysheds Raster from a numpy array with a given nodata."""
    from pysheds.view import Raster, ViewFinder
    vf = ViewFinder(
        affine=from_origin(500000, 6000000, 10, 10),
        shape=arr.shape, nodata=np.dtype(arr.dtype).type(nodata), crs=CRS)
    return Raster(arr, viewfinder=vf)


def test_dem_with_real_nodata_replaces_nan_sentinel():
    """A NaN nodata (which pysheds' `dem == nodata` never matches) is
    swapped for a finite sentinel below the DEM range, with the nodata
    cells filled to that sentinel."""
    arr = np.array([[10.0, 11.0], [np.nan, 12.0]], dtype='float32')
    out = _dem_with_real_nodata(_pysheds_raster(arr, np.nan))
    nod = float(out.viewfinder.nodata)
    assert np.isfinite(nod)
    assert nod < 10.0                      # below the valid range
    data = np.asarray(out)
    assert data[1, 0] == np.dtype('float32').type(nod)   # NaN cell filled
    assert data[0, 0] == 10.0              # valid cells untouched
    assert not np.isnan(data).any()        # no NaN survives for pysheds


def test_dem_with_real_nodata_passes_through_finite_nodata():
    """A DEM that already has a finite, matchable nodata is returned
    unchanged."""
    arr = np.array([[10.0, 11.0], [12.0, 13.0]], dtype='float32')
    src = _pysheds_raster(arr, -9999.0)
    assert _dem_with_real_nodata(src) is src


def test_breach_pits_drains_an_enclosed_pit():
    """A cell lower than all its neighbours is a pit (fdir == -2); breaching
    carves a descending channel to an outlet so flow direction resolves."""
    from pysheds.grid import Grid
    # A walled interior valley (row 3) draining off the right edge, dammed
    # at column 3 so column 2 is an enclosed pit. The high walls make the
    # dam the cheapest escape, so breaching must carve *through* it rather
    # than draining off a nearby edge.
    arr = np.full((7, 7), 10.0, dtype='float32')
    arr[3] = [10, 4, 3, 6, 2, 1, 0]
    dem = _pysheds_raster(arr, -9999.0)
    grid = Grid(viewfinder=dem.viewfinder)

    fdir_before = np.asarray(grid.flowdir(dem))
    assert fdir_before[3, 2] == PYSHEDS_PIT_CODE      # the dammed cell

    breached = breach_pits(grid, dem)
    fdir_after = np.asarray(grid.flowdir(breached))
    assert fdir_after[3, 2] != PYSHEDS_PIT_CODE        # pit now drains
    # Breaching only lowers cells, never raises them.
    assert np.all(np.asarray(breached) <= np.asarray(dem) + 1e-6)
    # The dam (column 3) is the cell that gets carved through.
    assert np.asarray(breached)[3, 3] < 6.0


def test_condition_dem_nan_boundary_has_no_spurious_pits(tmp_path):
    """A DEM clipped to an irregular shape (NaN boundary) draining to one
    edge must condition without a rash of boundary pits. Regression for
    pysheds treating the NaN boundary as valid terrain."""
    # Plane sloping down to the west; NaN border on all four sides mimics
    # the clipped-catchment nodata frame.
    arr = (np.arange(10, dtype='float32')[None, :] * np.ones((8, 1))).copy()
    arr[0, :] = np.nan
    arr[-1, :] = np.nan
    arr[:, -1] = np.nan
    fn = _write_tif(str(tmp_path / 'dem.tif'), arr, nodata=np.nan)

    inflated, grid = condition_dem(fn)
    fdir = np.asarray(grid.flowdir(inflated))
    finite = np.isfinite(arr)
    assert not (fdir[finite] == PYSHEDS_PIT_CODE).any()


def test_condition_dem_does_not_route_through_nodata(tmp_path):
    """The nodata region must be masked out of routing. Regression: the
    real sentinel that fixes boundary pits is itself a routable (low)
    value, so without a viewfinder mask pysheds drains the sentinel-filled
    nodata area and sprays spurious streams/headwaters outside the
    catchment."""
    arr = (np.arange(10, dtype='float32')[None, :] * np.ones((8, 1))).copy()
    arr[0, :] = np.nan
    arr[-1, :] = np.nan
    arr[:, -1] = np.nan
    fn = _write_tif(str(tmp_path / 'dem.tif'), arr, nodata=np.nan)

    inflated, grid = condition_dem(fn)
    acc = np.asarray(grid.accumulation(grid.flowdir(inflated)))
    nodata = ~np.isfinite(arr)
    # No contributing area should accumulate onto the missing-data cells.
    assert not (acc[nodata] > 1).any()


# --- upslope_weighted_mean -------------------------------------------------

def test_upslope_weighted_mean_ignores_nan_weights():
    """A NaN weight upstream must not null the downstream average (the SDR
    stream-nulling bug): it is dropped from both the sum and the count, so
    the mean is over the valid contributors only."""
    from pysheds.grid import Grid
    from pysheds.view import Raster, ViewFinder
    # Single row draining east to the edge: every cell flows into the next.
    dem = np.array([[5, 4, 3, 2, 1]], dtype='float32')
    vf = ViewFinder(affine=from_origin(500000, 6000000, 10, 10),
                    shape=dem.shape, nodata=np.float32(-9999), crs=CRS)
    grid = Grid(viewfinder=vf)
    fdir = grid.flowdir(Raster(dem, viewfinder=vf))

    # Weight 2.0 everywhere except a NaN at the head of the chain.
    w = np.full(dem.shape, 2.0, dtype='float64')
    w[0, 0] = np.nan
    mean = upslope_weighted_mean(grid, fdir, Raster(w, viewfinder=vf))

    # The most-downstream cell drains all five cells; four have weight 2,
    # one is NaN → mean is exactly 2.0, not NaN.
    assert np.isclose(mean[0, -1], 2.0)
    assert not np.isnan(mean[0, -1])
    # The NaN cell has no valid contributor, so it is NaN.
    assert np.isnan(mean[0, 0])


# --- clip_raster temp handling ---------------------------------------------

def test_clip_raster_temp_file_not_in_cwd(tmp_path, monkeypatch, get_file):
    """Regression: clip_raster used a fixed 'clipped_temp.tif' in the
    working directory, which pollutes the cwd and collides between
    concurrent runs."""
    monkeypatch.chdir(tmp_path)
    temp_file, shp_crs = clip_raster(
        str(get_file(DEM_FILE)), str(get_file(CATCHMENT_FILE)))
    try:
        assert os.path.exists(temp_file)
        assert os.path.abspath(os.path.dirname(temp_file)) != str(tmp_path)
        assert not (tmp_path / 'clipped_temp.tif').exists()
    finally:
        os.remove(temp_file)
