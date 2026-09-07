# Study configuration for the template notebooks

**Status:** design agreed, not yet implemented
**Date:** 2026-09-07

## The problem

The four template notebooks shipped in `fire_impacts/templates/` are generally
runnable, except for a handful of values each user has to supply. Those values
are scattered through the body of each notebook, and several of them have to be
entered into more than one notebook.

Two consequences follow. A new user cannot see at a glance what they are
required to customise: the values sit among prose, plots and teaching material.
And because the customisations live *inside* the notebooks, updating a project
to a newer template means reconciling the user's edits with the new version —
`fire-impacts update` backs the old one up, but the user still has to move their
values across by hand.

What is scattered today:

| Value | PrepareData | Simulation | SimulationEnsemble | SourceIntegration |
|---|---|---|---|---|
| project directory | yes (`clear=True`) | yes | yes | yes |
| catchment name | literal | `proj.catchments[0]` | `CATCHMENT` | `CATCHMENT` |
| boundary coverage | yes | | | |
| event name | `'2019_fire'` | `'2019_fire'` | `'2019_fire'` | `'2019_fire'` |
| ensemble name | | `'stochastic'` | `'stochastic'` | `'stochastic'` |
| fire start/end dates | yes | read back from ctx | read back | |
| DEM path | set, then **not used** | | | |
| aridity raster | yes | | | |
| TERN API key | env var lookup | | | |
| subcatchments | | commented out | | |
| replicate count | | literal `10` | `N_REPLICATES` | |
| replicate to inspect | | literal `[:, 9]` twice | | `REPLICATE = 0` (a different choice) |
| Veneer port / constituent / FU | | | | `PORT`, `CONSTITUENT`, `FUNCTIONAL_UNIT` |
| subcatchment label field | | | `'SiteID'` | (must match) |
| DEM cell area | | | `CELL_AREA_HA = 30*30/10_000` | |
| reporting thresholds | | | `THRESHOLD_T_HA`, `500` x3 | |
| parallel workers | | | `min(N_REPLICATES, 10)` x3 | |
| Load Distributor calibration | | | | `load_attenuation`, `maximum_concentration` |
| Source data-source names | | | | `'fire_tss'`, `'stochastic_rain'`, `'rainfall'` |
| Source timestep / date format | | | | `freq='D'`, `'%d/%m/%Y'` |

Genuinely duplicated across notebooks: project directory, catchment, event,
ensemble.

The lower half of that table was missed on a first reading and found by a
fact-check of this document against the code. Two of those values are not
merely scattered but wrong:

- `SourceIntegration.py` reads rainfall from a data source named
  `'stochastic_rain'` at line 236 and from one named `'rainfall'` at line
  327 — the same notebook disagreeing with itself about the name of the
  thing it just created.
- `PrepareData.py:310` writes the aridity path as `r'..\\test_data\\...'`
  — a raw string containing *doubled* backslashes, so the literal path
  contains `\\`. It resolves on Windows by luck rather than intent.

Both stop being possible once the value is a setting named once.

### What deliberately stays in the notebooks

Not everything with a literal in it is a user setting. Library-defined
names — plot layer names such as `'FireSeverity'`/`'dNBR'`, recorder result
keys such as `'RUSLE_sum_total'`, the debris column names — are not the
user's to choose. Neither is the notebooks' teaching material: the recorder
configuration is presented with a comment listing the available cadences,
the parameter-override cells are commented-out demonstrations, and the
presentation choices (colour maps, unit labels, axis scaling) are part of
what the notebook is showing you how to do. Moving those into a config file
would make both artefacts worse.

## The shape of the solution

A single `study.toml` in the project directory holds the values the user
supplies. Each notebook opens with one **Settings for this study** block that
loads the file and unpacks named constants; the body of the notebook refers to
those names.

### Decisions taken

1. **The templates are working templates, with the bundled example as their
   default configuration.** Notebook prose becomes catchment-agnostic; the
   scaffolded `study.toml` points at the example data, so a fresh project runs
   end to end before anything is edited. Switching to a real catchment means
   editing one file.

2. **The config file is authoritative for *inputs*; the project stays
   authoritative for what has been built.** `study.toml` declares identity
   (project, catchment, event, ensemble), input paths and run knobs.
   `PrepareData` consumes them and writes into the project. Downstream notebooks
   read identity from the config and everything else back from the project,
   exactly as they do now — for instance `ctx.simulation_period()` continues to
   derive the simulation window from the event definition rather than from the
   config. This is what prevents a config edited after a build from silently
   disagreeing with layers already on disk.

3. **TOML.** Comments, real types, lists and sections; read by the standard
   library's `tomllib` on the project's Python floor of 3.11, so no new
   dependency. TOML literal strings (single quotes) carry Windows paths without
   escaping.

4. **Scaffolded on `new`, never overwritten on `update`, drift reported by
   `status`.** Every setting has a package default, so an older `study.toml`
   keeps working against newer templates.

5. **Named constants unpacked at the top of each notebook**, rather than
   attribute access at each use site or threading the config into the library.
   Cell bodies stay readable as ordinary Python, and a user can override one
   value for an experiment by editing the block.

### Deliberate non-goal: this is not `parameters.json`

`study.toml` and `parameters.json` stay separate files answering different
questions — *what am I studying* versus *how is the model calibrated*.

Folding study settings into `params.py` was considered and rejected.
`parameters.json` is a sparse set of calibration overrides resolved through five
layers with a per-group scoping rule (`topography` and `delivery` cannot be set
per event, because they write layers shared by every event). A shapefile path
has neither a scope nor a calibration range. Worse, the parameter set feeds a
digest used to detect that derived layers were built with different values than
a run resolves — so putting a file path in it would mean renaming a directory
invalidated every built layer.

`study.describe()` prints a pointer to `parameters.json` so the relationship is
discoverable from the notebook.

## The file

Lives in the project directory, beside the notebooks. One convention carries
most of the discoverability:

> Uncommented means you must supply it. Commented out means optional, and the
> comment states the default.

```toml
# study.toml - everything this study needs you to supply.
#
# Uncommented settings are required. Commented-out settings are
# optional; the comment shows what happens if you leave them out.
# Values shown here drive the bundled example catchment, so a new
# project runs end to end before you change anything.

[project]
directory = "."        # project data root, relative to this file
# clear = false        # Leave this out and PrepareData starts from
#                      # scratch, wiping existing project data. Set it
#                      # false once the project holds work worth keeping.

[catchment]
name     = "EgSmallCatchment_7899"
boundary = '..\test_data\EgSmallCatchment_7899.shp'
aridity  = '..\test_data\AridityPT_EgSmallCatchment_7899.tif'
# dem           = ''   # default: download the GA 1" national DEM
# subcatchments = ''   # default: no subcatchment reporting
# subcatchment_id_field = 'SiteID'
#                      # attribute naming each subcatchment. Must match the
#                      # subcatchment names in your Source model.
# cell_size_m = 30     # DEM cell size in metres, used to convert per-cell
#                      # results to t/ha. Change it if your DEM is not 30 m.

[event]
name       = "2019_fire"
fire_start = "2019-01-15"
fire_end   = "2019-03-07"
# recovery_breakpoints = [0, 1, 2, 3]   # default: const.DEFAULT_RECOVERY_BREAKPOINTS

[ensemble]
name = "stochastic"
# num_replicates       = 10   # replicates drawn from pyraingen
# inspect_replicate    = 9    # which one the single-run Simulation notebook plots
# mean_annual_rainfall = 600  # mm; default: estimated from catchment lat/lon
# average_temperature  = 20   # degrees C; default: estimated from catchment lat/lon
# n_workers            = 10   # replicates run in parallel; cap for your machine

[reporting]
# Thresholds used by the ensemble notebook's exceedance maps. These are
# reporting choices, not model calibration - they change what the maps
# show, never what the model computes.
# erosion_threshold_t_ha    = 0.5
# delivered_threshold_kg_ha = 500

[secrets]
# Your TERN API key, needed to download soil data.
# Free - see the "Soils" section of PrepareData for how to get one.
# If left blank, the TERN_API_KEY environment variable is used instead.
tern_api_key = ""

[source]
# port            = 9876        # Veneer port for the running Source instance
# constituent     = 'TSS'       # default: auto-detected
# functional_unit = 'Forested'  # default: auto-detected
# replicate       = 0           # which ensemble replicate to push into Source
# timestep        = 'D'         # must match your Source model: 'D' or 'h'
# date_format     = '%d/%m/%Y'  # how your Source install writes run-period dates
# output_dir      = 'source_inputs'   # where the generated CSVs are written
# Load Distributor calibration:
# load_attenuation      = 10.0
# maximum_concentration = 1000.0      # mg/L
# Names of the Source data sources this notebook creates and then reads
# back. One name each - the notebook currently uses two different names
# for the rainfall source and reads the wrong one in one place.
# tss_data_source      = 'fire_tss'
# rainfall_data_source = 'stochastic_rain'
```

### Notes on specific settings

- **`project.clear` is tri-state, and the destructive default belongs to the
  notebook rather than the schema.** The setting defaults to *unset* (`None`)
  rather than to a boolean. Any library consumer treats unset as "do not
  clear" — the safe reading. `PrepareData` is the one notebook that builds a
  project from scratch, so its settings block resolves unset to `True`
  explicitly:

  ```python
  # PrepareData builds a project from scratch, so it clears by default.
  # Set clear = false in study.toml to keep data already in the project.
  CLEAR = True if study.project.clear is None else study.project.clear
  ```

  This keeps today's behaviour for a fresh example project, but the
  most destructive action in the system is now stated in the block the
  user reads first, and turning it off is a config edit rather than a
  notebook edit. It also means an explicit `clear = false` and a silent
  omission are distinguishable, which a plain boolean default could not
  express.

- **`catchment.dem` fixes a live defect.** `PrepareData` currently sets
  `optional_DEM_filename` and then passes `None` to `extract_catchment_dems()`.
  The variable is dead: the notebook downloads the national DEM regardless of
  what the user puts there. Routing the value through the config makes it reach
  the call.

- **`secrets.tern_api_key` holds the key itself.** The key is read-only and for
  a free service, so the convenience of keeping it with everything else
  outweighs the exposure. The `TERN_API_KEY` environment variable remains a
  fallback when the setting is blank, so anyone who followed the current
  notebook instructions keeps working unchanged. It is *not* required at load
  time — only the soil download needs it, and a blank key must not stop someone
  running the Simulation notebook — so `study.secret()` raises at the point of
  use.

- **`describe()` reports `tern_api_key: set`, never the value.** Not to protect
  the TOML file, but the notebook: `.ipynb` files carry their output cells and
  are the artefact most likely to be emailed or committed.

## The module: `fire_impacts/study.py`

Frozen dataclasses with `field(metadata=...)` carrying `help`, `required`,
`path` and `example`, matching the house pattern in `params.py`:

```python
@dataclass(frozen=True)
class CatchmentSettings:
    name: str = field(default=None, metadata={
        'required': True,
        'help': 'Name for this catchment within the project.'})
    boundary: str = field(default=None, metadata={
        'required': True, 'path': True,
        'help': 'Boundary coverage (.shp/.geojson). Must carry a CRS - '
                'it becomes the CRS for everything else in the catchment.'})
    dem: str = field(default=None, metadata={
        'path': True,
        'help': 'default: download the GA 1" national DEM'})
    ...
```

That metadata is the single source of truth for three consumers: the loader's
validation, the scaffold generator's comments, and the drift report. Because the
comments in a scaffolded file are generated from the same definitions the loader
validates against, they cannot drift from the code.

Groups: `ProjectSettings`, `CatchmentSettings`, `EventSettings`,
`EnsembleSettings`, `ReportingSettings`, `SecretSettings`,
`SourceIntegrationSettings`, composed into `StudySettings`.

**Two replicate settings, not one.** `ensemble.inspect_replicate` chooses
which replicate the single-run Simulation notebook plots; `source.replicate`
chooses which one is written into the Source model. They were conflated on a
first reading of the templates — `Simulation.py` uses a bare `9` twice while
`SourceIntegration.py` uses `REPLICATE = 0` and a separate hard-coded
`write_replicate_csvs(0)`. They serve different purposes and stay separate,
and `inspect_replicate` must be less than `num_replicates`.

### Public API

| Call | Behaviour |
|---|---|
| `load_study(path='.')` | Read, validate and return a `StudySettings` |
| `study.describe()` | Print each setting, its value, and whether it came from `study.toml` or a default |
| `study.secret('tern_api_key')` | Config value if set, else the `TERN_API_KEY` environment variable; raises a clear message if neither |
| `study.scaffold_text()` | Generate the commented TOML from the schema |
| `write_scaffold(path, seed=None)` | Write it, optionally seeded from an existing project |
| `check_study(path)` | Drift report for `fire-impacts status` |

### Behaviours

- **Paths resolve relative to `study.toml`, not the working directory.**
  Notebooks are run from varied places; `'..\test_data\x.shp'` must mean the
  same thing regardless. Every setting marked `path` — `project.directory`
  included — is returned as a resolved absolute path, so
  `FireImpactsProject(PROJECT_DIR)` does not depend on where the kernel was
  started.
- **Required paths are existence-checked at load.** A missing shapefile becomes
  `catchment.boundary: no such file <abs path> (set in study.toml)` rather than
  a geopandas traceback many cells later.
- **An unknown key raises, with a did-you-mean**, reusing the `difflib`
  behaviour in `params.py`. A silently ignored typo would let someone believe
  they had configured a study they had not.
- **A missing key falls back to its default, silently.** This is what lets an
  older `study.toml` work against newer templates.
- **A missing file raises a directed error** naming the directory searched and
  suggesting `fire-impacts update .`, not a bare `FileNotFoundError`.

### Shared schema helpers

`params.py` already contains `_did_you_mean`, `_coerce`, `_from_dict`,
`_deep_merge` and `_hints` — roughly the machinery `study.py` needs. These move
to a new internal `fire_impacts/_schema.py`, imported by both. The move is pure:
the old private names are re-exported from `params.py` so nothing referencing
them breaks, and the existing `params` tests passing unchanged is the evidence
that behaviour was preserved.

## The templates

Every template gains the same structure: imports, logging, a **Settings for this
study** block, then the existing prose and cells referring to the unpacked
names.

`PrepareData.py` carries the fullest block:

```python
from fire_impacts.study import load_study

study = load_study('.')
study.describe()

PROJECT_DIR = study.project.directory
CATCHMENT   = study.catchment.name
BOUNDARY    = study.catchment.boundary
DEM         = study.catchment.dem          # None -> download national DEM
ARIDITY     = study.catchment.aridity
EVENT       = study.event.name
FIRE_START  = study.event.fire_start
FIRE_END    = study.event.fire_end
BREAKPOINTS = study.event.recovery_breakpoints

# PrepareData builds a project from scratch, so it clears by default.
# Set clear = false in study.toml to keep data already in the project.
CLEAR = True if study.project.clear is None else study.project.clear
```

Body cells then read `FireImpactsProject(PROJECT_DIR, clear=CLEAR)`,
`proj.add_catchment(BOUNDARY)`, `topography.extract_catchment_dems(prep_ctx, DEM)`,
`severity.calculate_fire_severity(ctx, fire_start_date=FIRE_START, fire_end_date=FIRE_END)`
and `rusle.compute_adjusted_k_c(ctx, recovery_breakpoints=BREAKPOINTS)`.

Prose is rewritten to be catchment-agnostic — "the catchment named in
`study.toml`" rather than "a small example catchment" — so the sentence is true
on a first run and stays true after the file is edited.

The other three templates take a shorter block covering identity plus their own
knobs, so that

```python
ctx = RunContext.solo_run(
    proj, event=EVENT, ensemble=ENSEMBLE, catchment=CATCHMENT)
```

is identical in all three. `rainfall.rainfall[:, REPLICATE]` replaces the two
hard-coded `[:, 9]` in `Simulation`. `SourceIntegration` picks up `PORT`, and
`CONSTITUENT` / `FUNCTIONAL_UNIT` become "config value if set, else auto-detect"
rather than a commented-out override.

### What stays in the body

Teaching material does not move into the settings block: the commented-out
`set_event_parameter_overrides` demonstrations, the input-binding examples, and
the provenance walkthrough all stay where they are. Nor does `Simulation`
re-declare the fire dates — per the decision above, it continues to read them
back from the project.

## CLI and lifecycle

| Command | Behaviour |
|---|---|
| `fire-impacts new <path>` | Writes `study.toml` from the schema, pre-filled with the example values; reported alongside the notebooks |
| `fire-impacts update <path>` | Never overwrites `study.toml`; writes one only when absent. Afterwards reports settings the new templates understand that the file does not set — all optional and defaulted, so information rather than a demand |
| `fire-impacts status <path>` | Gains a config section: unknown keys with a suggestion, required-but-missing keys, and available-but-unset settings |

### Migrating a project that predates the config file

Such a project has notebooks but no `study.toml`, and after an update its
notebooks will call `load_study()`. `update` therefore writes one — seeded from
the project's own state rather than from the example. `proj.catchments`,
`proj.events(...)` and `proj.ensembles(...)` are readable from disk, so name,
event and ensemble come out correct. What cannot be recovered — boundary path,
aridity path, API key — is written blank with
`# could not be recovered from the project; please fill this in`. The command
says clearly that it did this and that the file needs attention.

`study.toml` is never fingerprinted, backed up or replaced, and is not tracked
in `.fire_impacts_notebooks.json`. The manifest concerns only files the tool
overwrites.

## Testing

- **`'study': StudySettings` added to `RECEIVER_TYPES` in
  `test_template_currency.py`.** Every `study.catchment.name` in every template
  is then checked against the real schema, so a template naming a setting that
  does not exist fails in CI. This is the link that keeps templates and schema
  honest, and it needs no new machinery.
- **One extension to that checker:** it currently resolves only `name.attr`,
  because it requires `node.value` to be an `ast.Name`. Every line of a settings
  block is a two-hop chain (`study.catchment.name`), so it must walk chains,
  resolving hop by hop against the probe instance. This also closes an existing
  gap — `record.parameters.delivery.max_sdr` in `PrepareData` is only
  half-checked today.
- **Scaffold round-trip:** `load_study()` on a freshly written scaffold
  succeeds, and every schema field appears in `scaffold_text()`. This is what
  stops a setting being added without being documented.
- **Loader:** defaults applied; unknown key raises with a suggestion; missing
  required raises naming the file; paths resolve relative to `study.toml` rather
  than the working directory; a missing required path is caught at load;
  `secret()` falls back to the environment variable; `describe()` does not print
  the key; a missing file gives the directed error.
- **Helper lift:** `test_params.py` and `test_params_persistence.py` pass
  unchanged.
- **CLI:** `new` creates the file; `update` on a legacy project seeds it from
  project state and leaves an existing one untouched; `status` reports a planted
  typo.

Nothing executes the notebooks end to end, consistent with the existing suite:
they need a real catchment, remote imagery and a rainfall service.
