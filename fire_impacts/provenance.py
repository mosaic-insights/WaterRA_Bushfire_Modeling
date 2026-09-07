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
from dataclasses import dataclass

import rasterio

from .params import ParameterRecord

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


# ---------------------------------------------------------------------------
# Reading a run's provenance back
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunProvenance:
    """Everything recorded about how one set of results was produced.

    Assembled from three files that are written separately — the run's
    identity, the parameters the run resolved, and the digests of the
    input layers it read. Kept together here because "what produced
    these results?" is one question, and answering it should not mean
    knowing which of three accessors holds which third of the answer.
    """

    #: event / ensemble / label for the run these results belong to.
    run: dict
    #: The full resolved parameter set, with the origin of each value.
    parameters: ParameterRecord
    #: Input layer name -> the digest of the parameters that built it.
    inputs: dict
    #: Which results section this describes.
    section: str

    def to_frame(self):
        """The parameters as a DataFrame — see ParameterRecord.to_frame."""
        return self.parameters.to_frame()

    def chosen(self):
        """Just the parameters somebody set, as a DataFrame."""
        return self.parameters.chosen()

    def summary(self) -> str:
        """A short human-readable account, for printing in a notebook."""
        chosen = self.chosen()
        if len(chosen):
            settings = ', '.join(
                f'{row.parameter}={row.value} ({row.source})'
                for row in chosen.itertuples()
            )
        else:
            settings = 'every parameter on its package default'
        return (
            f"{self.section} for event {self.run.get('event')!r}, ensemble "
            f"{self.run.get('ensemble')!r} (run {self.run.get('label')!r})\n"
            f"  resolved at {self.parameters.resolved_at}\n"
            f"  fire_impacts {self.parameters.version}\n"
            f"  {settings}\n"
            f"  built from {len(self.inputs)} input layer(s)"
        )


def read_run_provenance(ctx, section=None):
    """
    Read back how one set of results was produced.

    Parameters:
    - ctx: run-level RunContext.
    - section: results sub-folder. Defaults to the standard Results
      folder.

    Returns:
    - A :class:`RunProvenance`, or None when that section has no record
      — it was produced before provenance was written, or the run has
      not been executed.
    """
    import json
    import os

    from . import const as c

    if section is None:
        section = c.RESULTS_FOLDER_NAME
    path = ctx.run_path(section, c.PROVENANCE_FILE_NAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        stored = json.load(f)

    signature = stored.get('run_signature') or {}
    return RunProvenance(
        run=ctx.run_identity() or {
            'event': ctx.event, 'ensemble': ctx.ensemble,
            'label': ctx.run_label,
        },
        parameters=ParameterRecord.from_dict(stored),
        inputs=dict(signature.get('inputs', {})),
        section=section,
    )


# ---------------------------------------------------------------------------
# Run outputs
# ---------------------------------------------------------------------------

def layer_digests(layers):
    """
    Return {path: digest} for the tagged layers a run read.

    A run's outputs depend on two things: the parameters the run itself
    consumed, and the layers it was given. Recording only the former
    would let a run overwrite outputs produced from *rebuilt* input
    layers, since its own parameters would be unchanged.
    """
    import os

    out = {}
    for path, _paths in layers:
        if not os.path.exists(path):
            continue
        tagged = read_parameter_tags(path)
        out[os.path.basename(path)] = (
            tagged['digest'] if tagged else 'untagged')
    return out


def run_signature(record, layers, *paths):
    """
    Describe what a run's outputs depend on.

    Parameters:
    - record: the ParameterRecord the run resolved.
    - layers: (path, consumed_paths) pairs the run read.
    - paths: the parameter groups the run itself consumed.

    Returns:
    - Dict with 'parameters' and 'inputs', JSON-serialisable.
    """
    return {
        'parameters': record.parameters.subset(*paths),
        'inputs': layer_digests(layers),
    }


def describe_signature_change(previous, current):
    """Return a human-readable account of how two run signatures differ."""
    reasons = []
    reasons += [
        f'{leaf}: was {was!r}, now {now!r}'
        for leaf, was, now in _differing_leaves(
            previous.get('parameters', {}), current.get('parameters', {}))
    ]
    old_inputs = previous.get('inputs', {})
    new_inputs = current.get('inputs', {})
    for name in sorted(set(old_inputs) | set(new_inputs)):
        if old_inputs.get(name) != new_inputs.get(name):
            reasons.append(
                f'input layer {name} was rebuilt since those results were '
                f'written'
            )
    return '; '.join(reasons) or 'the recorded inputs differ'


def check_run_not_overwritten(path, signature, *, strict=True):
    """
    Refuse to replace results produced under a different configuration.

    Parameters:
    - path: the run's provenance record for this results section.
    - signature: the current :func:`run_signature`.
    - strict: raise on a mismatch (the default); False logs and continues.

    Returns:
    - True when the section is absent or matches, False on a mismatch
      when not strict.

    Notes:
    - Results directories are keyed by (event, ensemble) only, so a
      second run of the same pair replaces the first. Without this the
      replacement is silent, and a parameter sweep quietly destroys every
      result but the last.
    """
    import json
    import os

    if not os.path.exists(path):
        return True
    with open(path) as f:
        stored = json.load(f)
    previous = stored.get('run_signature')
    if previous is None or previous == signature:
        return True

    message = (
        f'{os.path.dirname(path)} already holds results produced under a '
        f'different configuration — {describe_signature_change(previous, signature)}. '
        f'Re-running would replace them. Pass overwrite=True to do that '
        f'deliberately, or results_section= to write alongside them.'
    )
    if strict:
        raise ValueError(message)
    logger.warning('%s', message)
    return False
