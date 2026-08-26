"""
Resolve input bindings by writing the raster they describe.

This is the single point where a binding is applied. Everything
downstream reads ``masked_dNBR.tif`` and needs no knowledge of whether it
came from satellite imagery, a synthetic draw, a supplied file or a
scalar — see :mod:`fire_impacts.bindings` for why that is the rule.

Each resolution writes a record beside the raster it produced, carrying
the binding, the *effective* random seed, and the content hash of what
was written. Hashing the output rather than the inputs is what makes the
record verifiable: a path can change under you, and a seed of None
describes no particular draw.
"""

import hashlib
import json
import logging
import os

import numpy as np
import rasterio

from .. import const as c
from ..bindings import Constant, Derived, FromFile, SyntheticFire
from .util import read_raster, read_raster_masked, write_raster

logger = logging.getLogger(__name__)

#: Written beside the raster a binding produced.
BINDING_RECORD_NAME = 'dnbr_binding.json'


def _raster_digest(path):
    """Content hash of a raster's first band, NaN-normalised."""
    with rasterio.open(path) as src:
        data = src.read(1)
    canonical = np.nan_to_num(
        np.ascontiguousarray(data, dtype=np.float64), nan=-9e99)
    return 'sha256:' + hashlib.sha256(canonical.tobytes()).hexdigest()


def _domain_mask(ctx, binding, dem_grid):
    """
    Return a boolean array of the cells a Constant should fill.

    Raises for a mask domain whose layer has not been produced yet,
    rather than falling back to the whole catchment — that fallback is
    exactly the "lakes and rock burned too" error the domain exists to
    avoid.
    """
    domain = binding.domain
    if domain == 'dem_valid':
        return ~dem_grid.nodata_mask

    if domain.startswith('mask:'):
        rel = domain.split(':', 1)[1]
        parts = [p for p in rel.replace('\\', '/').split('/') if p]
        for scope in (ctx.event_path, ctx.catchment_path):
            try:
                candidate = scope(*parts)
            except ValueError:
                continue
            if os.path.exists(candidate):
                borrowed, _ = read_raster(candidate)
                if borrowed.shape != dem_grid.shape:
                    raise ValueError(
                        f'Mask layer {candidate} has shape '
                        f'{borrowed.shape}, but the DEM grid is '
                        f'{dem_grid.shape}. A borrowed mask must be on '
                        f'the DEM grid.'
                    )
                return np.isfinite(borrowed)
        raise FileNotFoundError(
            f'Binding domain {domain!r} borrows the valid cells of '
            f'{rel!r}, which does not exist yet. Produce that layer '
            f'first, or use domain "catchment" or "dem_valid".'
        )

    # 'catchment': rasterise the boundary onto the DEM grid.
    import rasterio.features
    geometry = ctx.project.catchment_boundary(ctx.catchment).geometry.values
    painted = rasterio.features.rasterize(
        geometry,
        transform=dem_grid.transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
        out_shape=dem_grid.shape,
    )
    return painted == 1


def _warn_if_degenerate(values, binding):
    """
    Warn when a uniform dNBR makes downstream splits all-or-nothing.

    Not an error — a uniform severity scenario is a legitimate thing to
    model — but the results look like bugs to whoever reads them next:
    one of the two severity outputs is identically zero, the per-headwater
    dNBR statistics have no spread, and the debris-flow dNBR filter
    selects every headwater or none.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0 or finite.min() != finite.max():
        return
    conventional = float(finite.max()) * c.DNBR_SCALE
    logger.warning(
        'dNBR is uniform at %.1f (conventional scale) across the whole '
        'domain. Every downstream threshold therefore falls entirely on '
        'one side: the high- and low-severity erosion split will report '
        'one of them as zero, per-headwater dNBR statistics will have no '
        'spread, and the debris-flow dNBR cutoff (%.0f) will include '
        'every headwater or none. This is expected for a %s binding — '
        'noted so the outputs are not mistaken for a fault.',
        conventional, c.DEFAULT_DEBRIS_DNBR_THRESHOLD, binding.SOURCE,
    )


def materialise_dnbr(ctx, bindings=None, *, seed_sequence=None):
    """
    Write ``masked_dNBR.tif`` according to this context's dNBR binding.

    Parameters:
    - ctx: event-level RunContext.
    - bindings: an InputBindings to apply. When None the project /
      catchment / event layers are resolved from the context.
    - seed_sequence: optional integer used when a SyntheticFire binding
      leaves random_seed as None, so callers that need reproducibility
      without pinning a seed in configuration can supply one. When None a
      seed is drawn and recorded.

    Returns:
    - The binding record dict that was written, or None when the binding
      is Derived (in which case nothing is written and the normal
      severity pipeline is responsible for the raster).
    """
    ctx.validate(require_event_dir=False)
    if bindings is None:
        bindings = ctx.bindings()
    binding = bindings.dnbr

    if isinstance(binding, Derived):
        logger.info(
            'dNBR uses the derived pipeline; nothing to materialise.')
        return None

    out_path = ctx.event_path(c.FIRE_SEVERITY_FOLDER_NAME, 'masked_dNBR.tif')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    dem_grid = read_raster_masked(ctx.catchment_path('Topography', 'DEM.tif'))
    effective_seed = None

    if isinstance(binding, Constant):
        values = np.full(dem_grid.shape, np.nan, dtype=np.float32)
        inside = _domain_mask(ctx, binding, dem_grid)
        values[inside] = binding.to_stored_scale()
        logger.info(
            'Painting a constant dNBR of %g (%s) over %d cells (%s).',
            binding.value, binding.units, int(inside.sum()), binding.domain,
        )

    elif isinstance(binding, FromFile):
        from .util import read_aligned_like
        values = read_aligned_like(binding.path, dem_grid).astype(np.float32)
        values = values * binding.scale_to_stored()
        logger.info(
            'Using supplied dNBR raster %s (%s), aligned to the DEM grid.',
            binding.path, binding.units,
        )

    elif isinstance(binding, SyntheticFire):
        from .synthetic_fire import generate_synthetic_fire
        # Draw a seed when none was pinned, so the record describes the
        # draw that actually happened. Recording None would leave the
        # record unable to reproduce the raster sitting next to it.
        effective_seed = binding.random_seed
        if effective_seed is None:
            effective_seed = (
                seed_sequence if seed_sequence is not None
                else int(np.random.SeedSequence().entropy % (2 ** 31))
            )
            logger.info(
                'SyntheticFire binding pinned no seed; drew %d and '
                'recording it so the draw is reproducible.', effective_seed,
            )
        values = generate_synthetic_fire(
            ctx, severity=binding.severity, random_seed=effective_seed,
            reference_url=binding.reference_url,
        )

    else:
        raise ValueError(f'Unsupported dNBR binding: {binding!r}')

    if not isinstance(binding, SyntheticFire):
        # generate_synthetic_fire writes the raster itself.
        write_raster(out_path, values, dem_grid.meta())

    _warn_if_degenerate(
        np.asarray(values, dtype=np.float64), binding)

    record = {
        'input': 'dnbr',
        'binding': binding.to_dict(),
        'effective_seed': effective_seed,
        'written': os.path.relpath(out_path, ctx.project.project_path),
        'digest': _raster_digest(out_path),
        'stored_units': 'dnbr',
    }
    record_path = ctx.event_path(
        c.FIRE_SEVERITY_FOLDER_NAME, BINDING_RECORD_NAME)
    with open(record_path, 'w') as f:
        json.dump(record, f, indent=2)
    logger.info('Wrote dNBR binding record to %s', record_path)
    return record


def read_binding_record(ctx):
    """Return the dNBR binding record for this event, or None."""
    path = ctx.event_path(c.FIRE_SEVERITY_FOLDER_NAME, BINDING_RECORD_NAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
