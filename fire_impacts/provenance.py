"""
Per-layer provenance: stamp derived rasters, and detect stale ones.

A ``provenance.json`` records what a *step* resolved. This module records
what a *file* was built from, as GeoTIFF tags on the file itself. The tag
is the better authority of the two, because it is per-file:

- **Partial rebuilds.** Re-running one producer rewrites the shared JSON
  record, which then claims provenance for layers an earlier run built
  under different values. The tag on each raster stays correct.
- **Changed breakpoints.** Layers are written per recovery window.
  Shorten the breakpoint list and the orphaned ``SDR_t3.tif`` remains on
  disk, described by a record that no longer mentions it. Its tag still
  describes it.
- **Files that travel.** A raster copied out of the project keeps its
  tags and loses the JSON entirely.

What this does *not* cover: the digest is over parameters only. Re-extract
the DEM, re-derive dNBR, or hand-edit a C factor, and every digest still
matches. Staleness here is one edge of a dependency graph, not the whole
of it.
"""

import json
import logging

import rasterio

logger = logging.getLogger(__name__)

#: GeoTIFF tag holding the parameter values a layer was built from.
PARAMS_TAG = 'FIRE_IMPACTS_PARAMS'
#: GeoTIFF tag holding the digest of those values.
DIGEST_TAG = 'FIRE_IMPACTS_DIGEST'
#: GeoTIFF tag holding the package version that wrote the layer.
VERSION_TAG = 'FIRE_IMPACTS_VERSION'


def parameter_tags(record, *paths):
    """
    Build the GeoTIFF tags stamping a layer with its parameters.

    Parameters:
    - record: the ParameterRecord the producing step resolved.
    - paths: the parameter groups or leaves that producer actually
      consumed, e.g. ``'delivery'`` or ``'topography.max_slope_length_m'``.
      Naming only what was consumed is what keeps an unrelated change
      from flagging this layer stale.

    Returns:
    - Dict of tag name to string value, ready for ``write_raster(tags=)``.
    """
    values = record.parameters.subset(*paths)
    return {
        PARAMS_TAG: json.dumps(values, sort_keys=True,
                               separators=(',', ':')),
        DIGEST_TAG: record.parameters.group_digest(*paths),
        VERSION_TAG: record.version,
    }


def read_parameter_tags(path):
    """
    Read a layer's parameter tags back.

    Returns:
    - Dict with 'values', 'digest' and 'version', or None when the layer
      carries no parameter tags (written before tagging, or by something
      else).
    """
    with rasterio.open(path) as src:
        tags = src.tags()
    if DIGEST_TAG not in tags:
        return None
    raw = tags.get(PARAMS_TAG)
    return {
        'values': json.loads(raw) if raw else {},
        'digest': tags[DIGEST_TAG],
        'version': tags.get(VERSION_TAG, 'unknown'),
    }


def _differing_leaves(built, resolved, prefix=''):
    """Yield (path, built_value, resolved_value) for every differing leaf."""
    keys = set(built) | set(resolved)
    for key in sorted(keys):
        here = f'{prefix}{key}'
        a, b = built.get(key), resolved.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            yield from _differing_leaves(a, b, prefix=f'{here}.')
        elif a != b:
            yield here, a, b


def check_layer_freshness(path, record, *paths, strict=True):
    """
    Check a derived layer against the parameters now in force.

    Parameters:
    - path: the raster to check.
    - record: the ParameterRecord currently resolved.
    - paths: the parameter groups or leaves that layer depends on.
    - strict: raise on a mismatch (the default). Pass False to log a
      warning and continue.

    Returns:
    - True when the layer matches, False when it does not and strict is
      False. Raises ValueError on a mismatch when strict.

    Notes:
    - An untagged layer is reported as unknown rather than stale: it was
      most likely written before tagging existed, and treating that as a
      mismatch would make every existing project fail on upgrade.
    - The message names the differing values rather than quoting two
      digests, which tell a reader nothing. Digests are only used for the
      fast "are these identical" test.
    """
    tagged = read_parameter_tags(path)
    if tagged is None:
        logger.info(
            'No parameter tags on %s, so it cannot be checked against '
            'the current values. It predates layer tagging; rebuild it '
            'to make this checkable.', path,
        )
        return True

    if tagged['digest'] == record.parameters.group_digest(*paths):
        return True

    differences = list(_differing_leaves(
        tagged['values'], record.parameters.subset(*paths)))
    detail = '; '.join(
        f'{leaf}: built with {built!r}, now {now!r}'
        for leaf, built, now in differences
    ) or 'the recorded values differ but no leaf comparison was possible'

    message = (
        f'{path} was built with different parameters than this run '
        f'resolves — {detail}. Re-run the step that produces it, or pass '
        f'the parameters it was built with.'
    )
    if strict:
        raise ValueError(message)
    logger.warning('%s', message)
    return False


def check_layers_fresh(layers, record, strict=True):
    """
    Check several layers at once.

    Parameters:
    - layers: iterable of (path, paths_tuple) pairs.
    - record: the ParameterRecord currently resolved.
    - strict: as :func:`check_layer_freshness`.

    Returns:
    - List of paths that failed the check (empty when all are fresh, or
      when strict and nothing raised).
    """
    import os

    stale = []
    for path, paths in layers:
        if not os.path.exists(path):
            continue
        if not check_layer_freshness(path, record, *paths, strict=strict):
            stale.append(path)
    return stale
