"""
read_aligned: pull a raster onto someone else's grid.

Every RUSLE factor is combined this way - C and K are stored at their
coarse native resolution and resampled onto the DEM grid before being
multiplied - so a misalignment here shifts erosion across the catchment
without failing anything.
"""

import numpy as np
import pytest
import rasterio as rio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling

from fire_impacts.pre.util import read_aligned, read_raster


CRS_7899 = CRS.from_epsg(7899)


def write_raster(path, data, transform, crs=CRS_7899, nodata=None,
                 dtype='float32'):
    data = np.asarray(data, dtype=dtype)
    profile = {
        'driver': 'GTiff',
        'height': data.shape[0],
        'width': data.shape[1],
        'count': 1,
        'dtype': dtype,
        'crs': crs,
        'transform': transform,
    }
    if nodata is not None:
        profile['nodata'] = nodata
    with rio.open(path, 'w', **profile) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture()
def source(tmp_path):
    """A 5x5 raster of 10 m cells with a known value pattern."""
    data = np.arange(25, dtype='float32').reshape(5, 5)
    transform = Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)
    return write_raster(tmp_path / 'source.tif', data, transform), transform


class TestSameGrid:

    def test_identical_grid_returns_the_original_values(self, source):
        path, transform = source
        original, _ = read_raster(path)

        aligned = read_aligned(path, transform, CRS_7899, (5, 5))

        assert aligned.shape == (5, 5)
        assert np.allclose(aligned, original)


class TestResampling:

    def test_finer_target_grid_upsamples(self, source):
        path, _ = source
        # Same extent, 5 m cells instead of 10 m.
        finer = Affine(5.0, 0.0, 1000.0, 0.0, -5.0, 2000.0)

        aligned = read_aligned(path, finer, CRS_7899, (10, 10))

        assert aligned.shape == (10, 10)
        # Nearest-neighbour: each source cell becomes a 2x2 block.
        assert aligned[0, 0] == pytest.approx(0.0)
        assert aligned[0, 1] == pytest.approx(0.0)
        assert aligned[1, 0] == pytest.approx(0.0)
        assert aligned[0, 2] == pytest.approx(1.0)

    def test_coarser_target_grid_downsamples(self, source):
        path, _ = source
        coarser = Affine(20.0, 0.0, 1000.0, 0.0, -20.0, 2000.0)

        aligned = read_aligned(path, coarser, CRS_7899, (2, 2))

        assert aligned.shape == (2, 2)
        assert np.all(np.isfinite(aligned))

    def test_resampling_method_is_honoured(self, tmp_path):
        # Alternating columns, so each 20 m target cell covers a mix of
        # 0 and 100: averaging lands between them, nearest never does.
        data = np.array([[0.0, 100.0, 0.0, 100.0]] * 4)
        transform = Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)
        path = write_raster(tmp_path / 'step.tif', data, transform)
        coarser = Affine(20.0, 0.0, 1000.0, 0.0, -20.0, 2000.0)

        nearest = read_aligned(
            path, coarser, CRS_7899, (2, 2), resampling=Resampling.nearest)
        averaged = read_aligned(
            path, coarser, CRS_7899, (2, 2), resampling=Resampling.average)

        assert set(np.unique(np.asarray(nearest))) <= {0.0, 100.0}
        assert np.allclose(averaged, 50.0)


class TestAlignment:

    def test_offset_target_window_reads_the_right_cells(self, source):
        path, _ = source
        # Shift the origin one cell east and one cell south, and ask for
        # a 3x3 window: it should land on the source's [1:4, 1:4] block.
        offset = Affine(10.0, 0.0, 1010.0, 0.0, -10.0, 1990.0)

        aligned = read_aligned(path, offset, CRS_7899, (3, 3))

        expected = np.arange(25).reshape(5, 5)[1:4, 1:4]
        assert np.allclose(aligned, expected)

    def test_uncovered_area_is_nan_even_without_a_source_nodata_tag(
            self, source):
        # REGRESSION. read_aligned used to inherit the source's nodata,
        # so a source carrying no tag got reproject's default 0 fill and
        # an empty read-back mask - uncovered area came back as real
        # zeros. Silent, and dangerous: this is how the C and K factors
        # reach the DEM grid, and a 0 factor is a valid value meaning
        # "no erosion here". The destination now always declares NaN.
        path, _ = source
        elsewhere = Affine(10.0, 0.0, 9000.0, 0.0, -10.0, 2000.0)

        aligned = read_aligned(path, elsewhere, CRS_7899, (5, 5))

        assert np.all(np.isnan(aligned))

    def test_uncovered_area_is_nan_when_the_source_declares_nodata(
            self, tmp_path):
        # The already-well-formed case, unchanged.
        data = np.arange(25, dtype='float32').reshape(5, 5)
        transform = Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)
        path = write_raster(
            tmp_path / 'tagged.tif', data, transform, nodata=-9999.0)
        elsewhere = Affine(10.0, 0.0, 9000.0, 0.0, -10.0, 2000.0)

        aligned = read_aligned(path, elsewhere, CRS_7899, (5, 5))

        assert np.all(np.isnan(aligned))

    def test_partial_overlap_fills_the_gap_with_nan(self, source):
        # Straddle the eastern edge: left two columns overlap the source,
        # the rest falls outside and must be NaN, not 0.
        path, _ = source
        straddle = Affine(10.0, 0.0, 1030.0, 0.0, -10.0, 2000.0)

        aligned = np.asarray(read_aligned(path, straddle, CRS_7899, (5, 5)))

        expected_first_col = np.arange(25).reshape(5, 5)[:, 3]
        assert np.allclose(aligned[:, 0], expected_first_col)
        assert np.all(np.isnan(aligned[:, 2:]))

    def test_a_real_zero_is_not_confused_with_a_gap(self, tmp_path):
        # The distinction the change exists to preserve: a genuine 0 in
        # the source stays 0, while uncovered area becomes NaN.
        data = np.zeros((5, 5), dtype='float32')
        transform = Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)
        path = write_raster(tmp_path / 'zeros.tif', data, transform)
        straddle = Affine(10.0, 0.0, 1030.0, 0.0, -10.0, 2000.0)

        aligned = np.asarray(read_aligned(path, straddle, CRS_7899, (5, 5)))

        assert np.all(aligned[:, :2] == 0.0)
        assert np.all(np.isnan(aligned[:, 2:]))


class TestDtype:
    """
    NaN has to be representable in the output, so integer sources are
    promoted. Float widths are left alone.
    """

    def test_integer_source_is_promoted_to_float(self, tmp_path):
        data = np.arange(25, dtype='int16').reshape(5, 5)
        transform = Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)
        path = write_raster(
            tmp_path / 'categorical.tif', data, transform, dtype='int16')

        aligned = read_aligned(path, transform, CRS_7899, (5, 5))

        assert np.issubdtype(aligned.dtype, np.floating)
        assert np.allclose(np.asarray(aligned), data)

    def test_integer_source_gaps_are_nan(self, tmp_path):
        data = np.arange(25, dtype='int16').reshape(5, 5)
        transform = Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)
        path = write_raster(
            tmp_path / 'categorical.tif', data, transform, dtype='int16')
        elsewhere = Affine(10.0, 0.0, 9000.0, 0.0, -10.0, 2000.0)

        aligned = read_aligned(path, elsewhere, CRS_7899, (5, 5))

        assert np.all(np.isnan(aligned))

    def test_float32_stays_float32(self, source):
        path, transform = source
        aligned = read_aligned(path, transform, CRS_7899, (5, 5))

        assert aligned.dtype == np.float32

    def test_float64_precision_is_not_downcast(self, tmp_path):
        data = np.full((3, 3), 1.0 + 1e-12, dtype='float64')
        transform = Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)
        path = write_raster(
            tmp_path / 'precise.tif', data, transform, dtype='float64')

        aligned = read_aligned(path, transform, CRS_7899, (3, 3))

        assert aligned.dtype == np.float64
        assert np.asarray(aligned)[0, 0] != 1.0


class TestNoData:

    def test_nodata_becomes_nan(self, tmp_path):
        data = np.array([[1.0, -9999.0], [3.0, 4.0]])
        transform = Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)
        path = write_raster(
            tmp_path / 'nodata.tif', data, transform, nodata=-9999.0)

        aligned = read_aligned(path, transform, CRS_7899, (2, 2))

        assert np.isnan(aligned[0, 1])
        assert aligned[0, 0] == pytest.approx(1.0)
        assert aligned[1, 1] == pytest.approx(4.0)

    def test_nodata_is_not_left_as_a_sentinel(self, tmp_path):
        # -9999 leaking into a RUSLE factor multiplication would be far
        # worse than a NaN, which propagates visibly.
        data = np.full((3, 3), -9999.0)
        data[1, 1] = 5.0
        transform = Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)
        path = write_raster(
            tmp_path / 'mostly_nodata.tif', data, transform, nodata=-9999.0)

        aligned = read_aligned(path, transform, CRS_7899, (3, 3))

        assert not np.any(aligned == -9999.0)
        assert np.isnan(aligned).sum() == 8


class TestReprojection:

    def test_reprojects_to_a_different_crs(self, source):
        path, _ = source
        # The same extent expressed in GDA2020 geographic coordinates.
        target_crs = CRS.from_epsg(7844)
        with rio.open(path) as src:
            from rasterio.warp import calculate_default_transform
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds)

        aligned = read_aligned(path, transform, target_crs, (height, width))

        assert aligned.shape == (height, width)
        assert np.any(np.isfinite(aligned))
