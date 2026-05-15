"""
Persist and reload ensemble simulation outputs.

The broader catchment-transport model is driven replicate-by-replicate
and needs access to the per-replicate subcatchment sediment loads and
the matching rainfall.  This module standardises where those artefacts
live on disk, so the downstream notebook can pick them up without
ad-hoc file paths.

Layout (per catchment)::

    Catchments/<catchment>/
        Ensembles/<ensemble>/
            rainfall.nc                     # climate-only, shared
        Runs/<event>/<ensemble>/
            manifest.json
            replicates/
                00/
                    rusle/
                        grids/<key>.tif     # opt-in
                        subcatchment_timeseries.parquet
                    debris_flow/
                        summary.parquet
                        mass_ts.parquet
                    combined/
                        subcatchment_<freq>.parquet
                01/ ...

Events and ensembles are siblings of each other so the same rainfall
realisation can drive multiple fires without duplication; the
(event, ensemble) cartesian product is stored under Runs/.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


MANIFEST_NAME = 'manifest.json'
RAINFALL_NAME = 'rainfall.nc'
REPLICATES_DIR = 'replicates'


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _run_root(ctx):
    """Return the root Path for the context's (event, ensemble) run."""
    return Path(ctx.run_path())


def _ensemble_root(ctx):
    """Return the root Path for the context's ensemble folder."""
    return Path(ctx.ensemble_path())


def _replicate_dir(root: Path, replicate_idx: int) -> Path:
    """Return the per-replicate subdirectory path for a given index."""
    return root / REPLICATES_DIR / f'{replicate_idx:02d}'


def _safe_key(key: str) -> str:
    """Sanitise an arbitrary dict key so it makes a safe filename."""
    return ''.join(
        c if c.isalnum() or c in ('-', '_') else '_' for c in str(key)
    )


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_ensemble_run(
    ctx,
    *,
    rainfall_ds: xr.Dataset | None = None,
    rusle_results: dict | None = None,
    debris_results: dict | None = None,
    combined_by_freq: dict | None = None,
    include_rusle_grids: bool = False,
    include_raw_debris: bool = True,
    subcatchment_label_field: str | None = None,
    extra_manifest: dict | None = None,
) -> Path:
    """
    Persist a simulation run to the library-managed directory.

    Everything is optional — pass whichever artefacts have been computed.
    The primary artefacts for driving a downstream catchment model are
    combined_by_freq and rainfall_ds.

    Parameters:
    - ctx: run-level RunContext. Rainfall is written under the
      catchment's Ensembles/<ensemble>/ folder (climate-only, shareable
      across events); run outputs land under Runs/<event>/<ensemble>/.
    - rainfall_ds: Multi-replicate rainfall dataset (the input that
      drove the simulations). Saved as a single NetCDF at the ensemble
      root.
    - rusle_results: Dict of the form {replicate: {catchment: {key:
      ...}}} as produced by run_rusle_all_replicates(). Gridded recorder
      outputs are saved as GeoTIFFs only when include_rusle_grids is
      True.
    - debris_results: Dict of the form {replicate: {catchment:
      (summary_df, mass_ts)}} as produced by
      run_debris_flow_all_replicates(). Stored only when
      include_raw_debris is True.
    - combined_by_freq: Dict of the form {freq: {replicate: DataFrame}},
      output of combine_rusle_and_debris_subcatchment() for each
      resolution to persist. freq is used as the filename suffix
      (e.g. 'h', 'D', 'YS', 'total').
    - include_rusle_grids: If True, save per-replicate RUSLE grid
      recorders as GeoTIFFs. Off by default — these dominate disk usage
      and are reproducible from the rainfall input.
    - include_raw_debris: If True, save per-replicate debris headwater
      summary and mass timeseries.
    - subcatchment_label_field: Recorded in the manifest for downstream
      consumers. If None, the project's per-catchment setting is used.
    - extra_manifest: Additional key-value pairs merged into the
      manifest (e.g. rainfall provenance, climate scenario, seed).

    Returns:
    - pathlib.Path to the run root directory.
    """
    catchment = ctx.catchment
    root = _run_root(ctx)
    root.mkdir(parents=True, exist_ok=True)
    logger.info('Saving run to %s', root)

    if subcatchment_label_field is None:
        subcatchment_label_field = (
            ctx.project.subcatchment_label_field(catchment)
            if hasattr(ctx.project, 'subcatchment_label_field') else None
        )

    replicate_ids: set[int] = set()

    if rainfall_ds is not None:
        # Rainfall is climate-only — write under the ensemble root so a
        # single rainfall realisation can be shared by multiple events.
        ens_root = _ensemble_root(ctx)
        ens_root.mkdir(parents=True, exist_ok=True)
        _save_rainfall(rainfall_ds, ens_root)
        if 'replicate' in rainfall_ds.dims:
            replicate_ids.update(range(rainfall_ds.sizes['replicate']))

    if rusle_results is not None:
        replicate_ids.update(rusle_results.keys())
        for rep, per_catchment in rusle_results.items():
            rep_dir = _replicate_dir(root, rep)
            _save_rusle_replicate(
                per_catchment[catchment], rep_dir,
                include_grids=include_rusle_grids,
            )

    if debris_results is not None and include_raw_debris:
        replicate_ids.update(debris_results.keys())
        for rep, per_catchment in debris_results.items():
            rep_dir = _replicate_dir(root, rep)
            _save_debris_replicate(per_catchment[catchment], rep_dir)

    if combined_by_freq is not None:
        for freq, per_rep in combined_by_freq.items():
            replicate_ids.update(per_rep.keys())
            for rep, df in per_rep.items():
                rep_dir = _replicate_dir(root, rep)
                _save_combined_replicate(df, rep_dir, freq=freq)

    manifest = _build_manifest(
        catchment=catchment,
        event=ctx.event,
        ensemble=ctx.ensemble,
        replicate_ids=sorted(replicate_ids),
        rainfall_ds=rainfall_ds,
        combined_by_freq=combined_by_freq,
        include_rusle_grids=include_rusle_grids,
        include_raw_debris=include_raw_debris,
        has_rusle=rusle_results is not None,
        has_debris=debris_results is not None,
        subcatchment_label_field=subcatchment_label_field,
        extra=extra_manifest,
    )
    with open(root / MANIFEST_NAME, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info('Wrote manifest: %s', root / MANIFEST_NAME)

    return root


def _save_rainfall(rainfall_ds: xr.Dataset, root: Path) -> None:
    """Write the rainfall dataset to a NetCDF file at the ensemble root."""
    path = root / RAINFALL_NAME
    rainfall_ds.to_netcdf(path)
    logger.info('Saved rainfall ensemble to %s', path)


def _save_rusle_replicate(catchment_results, rep_dir, *, include_grids):
    """Persist one replicate's RUSLE recorder outputs."""
    rusle_dir = rep_dir / 'rusle'
    rusle_dir.mkdir(parents=True, exist_ok=True)

    for key, value in catchment_results.items():
        if key == 'the_transform':
            continue  # transform is redundant with the GeoTIFFs
        if isinstance(value, pd.DataFrame):
            path = rusle_dir / f'{_safe_key(key)}.parquet'
            value.to_parquet(path)
        elif isinstance(value, xr.DataArray) and include_grids:
            grids_dir = rusle_dir / 'grids'
            grids_dir.mkdir(exist_ok=True)
            path = grids_dir / f'{_safe_key(key)}.tif'
            _write_dataarray_as_geotiff(value, path)


def _write_dataarray_as_geotiff(da: xr.DataArray, path: Path) -> None:
    """
    Write an xarray.DataArray to a GeoTIFF file via rioxarray.

    Falls back to a NetCDF file if georeferencing metadata is missing
    or rioxarray raises an error.

    Parameters:
    - da: DataArray to write. May be 2-D or 3-D (with a time dimension).
    - path: Destination file path for the GeoTIFF output.

    Returns:
    - None
    """
    try:
        import rioxarray  # noqa: F401 - registers .rio accessor
        da.rio.to_raster(path)
    except Exception as exc:
        nc_path = path.with_suffix('.nc')
        logger.warning(
            'Could not write %s as GeoTIFF (%s); saving NetCDF at %s',
            path, exc, nc_path,
        )
        da.to_netcdf(nc_path)


def _save_debris_replicate(catchment_tuple, rep_dir):
    """Persist one replicate's debris-flow output tuple."""
    summary_df, mass_ts = catchment_tuple
    debris_dir = rep_dir / 'debris_flow'
    debris_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(debris_dir / 'summary.parquet')
    mass_ts.to_parquet(debris_dir / 'mass_ts.parquet')


def _save_combined_replicate(df: pd.DataFrame, rep_dir, *, freq: str):
    """Write one replicate's combined subcatchment DataFrame as parquet."""
    combined_dir = rep_dir / 'combined'
    combined_dir.mkdir(parents=True, exist_ok=True)
    # Parquet columns must be strings; enforce that here so numeric
    # sc_ID columns (if the caller skipped SiteID labelling) roundtrip.
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    path = combined_dir / f'subcatchment_{_safe_key(freq)}.parquet'
    out.to_parquet(path)


def _build_manifest(
    *, catchment, event, ensemble, replicate_ids,
    rainfall_ds, combined_by_freq, include_rusle_grids,
    include_raw_debris, has_rusle, has_debris,
    subcatchment_label_field, extra,
):
    """Build and return the manifest dict for an ensemble run."""
    rainfall_meta: dict = {}
    if rainfall_ds is not None and 'time' in rainfall_ds.coords:
        tcoord = rainfall_ds['time']
        rainfall_meta = {
            'start': pd.Timestamp(tcoord.values[0]).isoformat(),
            'end': pd.Timestamp(tcoord.values[-1]).isoformat(),
            'n_timesteps': int(tcoord.size),
        }
    manifest = {
        'catchment': catchment,
        'event': event,
        'ensemble': ensemble,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'replicates': list(replicate_ids),
        'n_replicates': len(replicate_ids),
        'rainfall': rainfall_meta,
        'combined_frequencies': (
            sorted(combined_by_freq.keys()) if combined_by_freq else []
        ),
        'artefacts': {
            'rainfall': rainfall_ds is not None,
            'rusle_timeseries': has_rusle,
            'rusle_grids': has_rusle and include_rusle_grids,
            'debris_flow_raw': has_debris and include_raw_debris,
            'combined': bool(combined_by_freq),
        },
        'subcatchment_label_field': subcatchment_label_field,
    }
    if extra:
        manifest.update(extra)
    return manifest


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_ensemble_manifest(ctx) -> dict:
    """
    Read and return the run manifest JSON for the context.

    Parameters:
    - ctx: run-level RunContext.

    Returns:
    - Dict parsed from the manifest.json file.
    """
    root = _run_root(ctx)
    path = root / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f'No run manifest at {path} — has save_ensemble_run '
            f'been called for this catchment/event/ensemble?'
        )
    with open(path) as f:
        return json.load(f)


def list_ensembles(project, catchment) -> list[str]:
    """
    Return the ensemble names available under a catchment.

    Parameters:
    - project: FireImpactsProject managing the directory structure.
    - catchment: Name of the catchment to query.

    Returns:
    - Sorted list of ensemble name strings found on disk.
    """
    base = Path(project.catchment_path(catchment)) / 'Ensembles'
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def list_events(project, catchment) -> list[str]:
    """
    Return the event names available under a catchment.

    Parameters:
    - project: FireImpactsProject managing the directory structure.
    - catchment: Name of the catchment to query.

    Returns:
    - Sorted list of event name strings found on disk.
    """
    base = Path(project.catchment_path(catchment)) / 'Events'
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def list_runs(project, catchment, *, event: str | None = None) -> list[tuple[str, str]]:
    """
    Return the (event, ensemble) runs available under a catchment.

    Parameters:
    - project: FireImpactsProject managing the directory structure.
    - catchment: Name of the catchment to query.
    - event: Optional event filter; if given, only runs under
      Runs/<event>/ are returned.

    Returns:
    - Sorted list of (event_name, ensemble_name) tuples found on disk.
    """
    base = Path(project.catchment_path(catchment)) / 'Runs'
    if not base.exists():
        return []
    out: list[tuple[str, str]] = []
    event_dirs = (
        [base / event] if event is not None
        else [p for p in base.iterdir() if p.is_dir()]
    )
    for ev_dir in event_dirs:
        if not ev_dir.exists():
            continue
        for ens_dir in ev_dir.iterdir():
            if ens_dir.is_dir():
                out.append((ev_dir.name, ens_dir.name))
    return sorted(out)


def load_ensemble_rainfall(ctx) -> xr.Dataset:
    """
    Reload the rainfall ensemble NetCDF for the context's ensemble.

    Rainfall lives at the ensemble level (independent of fire event),
    so this loader only uses ctx.ensemble (ctx.event is ignored).

    Parameters:
    - ctx: RunContext with a non-None ensemble.

    Returns:
    - xarray.Dataset of rainfall replicates loaded from disk.
    """
    root = _ensemble_root(ctx)
    path = root / RAINFALL_NAME
    return xr.open_dataset(path)


def load_ensemble_combined(
    ctx,
    *,
    freq='D',
) -> dict[int, pd.DataFrame]:
    """
    Reload combined RUSLE+debris subcatchment loads at a given frequency.

    Column labels match whatever was saved (typically SiteID strings —
    see the manifest's subcatchment_label_field).

    Parameters:
    - ctx: run-level RunContext.
    - freq: Temporal resolution key matching a freq used in
      save_ensemble_run (e.g. 'D', 'h', 'YS', 'total').

    Returns:
    - Dict mapping replicate index (int) to subcatchment load DataFrame.
    """
    root = _run_root(ctx)
    suffix = _safe_key(freq)
    replicates_dir = root / REPLICATES_DIR
    if not replicates_dir.exists():
        raise FileNotFoundError(
            f'No replicates folder at {replicates_dir}'
        )

    out: dict[int, pd.DataFrame] = {}
    for rep_dir in sorted(replicates_dir.iterdir()):
        if not rep_dir.is_dir():
            continue
        path = rep_dir / 'combined' / f'subcatchment_{suffix}.parquet'
        if not path.exists():
            continue
        out[int(rep_dir.name)] = pd.read_parquet(path)
    if not out:
        raise FileNotFoundError(
            f'No combined parquet files found for freq={freq!r} under '
            f'{replicates_dir}. Available frequencies: see manifest.'
        )
    return out


def load_ensemble_rusle_timeseries(
    ctx,
    *,
    key='erosion_daily_time_series',
) -> dict[int, pd.DataFrame]:
    """
    Reload a per-replicate RUSLE recorder timeseries by key.

    Parameters:
    - ctx: run-level RunContext.
    - key: Recorder key used when saving (e.g.
      'erosion_daily_time_series'). Default is the standard RUSLE
      timeseries key.

    Returns:
    - Dict mapping replicate index (int) to timeseries DataFrame.
    """
    root = _run_root(ctx)
    replicates_dir = root / REPLICATES_DIR
    out: dict[int, pd.DataFrame] = {}
    for rep_dir in sorted(replicates_dir.iterdir()):
        if not rep_dir.is_dir():
            continue
        path = rep_dir / 'rusle' / f'{_safe_key(key)}.parquet'
        if path.exists():
            out[int(rep_dir.name)] = pd.read_parquet(path)
    return out


def load_ensemble_debris_raw(
    ctx,
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Reload per-replicate debris-flow raw outputs from disk.

    Parameters:
    - ctx: run-level RunContext.

    Returns:
    - Dict mapping replicate index (int) to a (summary_df, mass_ts)
      tuple of DataFrames.
    """
    root = _run_root(ctx)
    replicates_dir = root / REPLICATES_DIR
    out = {}
    for rep_dir in sorted(replicates_dir.iterdir()):
        if not rep_dir.is_dir():
            continue
        summary_path = rep_dir / 'debris_flow' / 'summary.parquet'
        mass_path = rep_dir / 'debris_flow' / 'mass_ts.parquet'
        if summary_path.exists() and mass_path.exists():
            out[int(rep_dir.name)] = (
                pd.read_parquet(summary_path),
                pd.read_parquet(mass_path),
            )
    return out
