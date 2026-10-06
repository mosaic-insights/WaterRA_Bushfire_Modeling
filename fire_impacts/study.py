"""
The settings a user supplies for one study.

The template notebooks are runnable except for a handful of values that
depend on the catchment being studied: where the boundary coverage is,
what the fire was called, when it burned. Those values used to sit in
the body of each notebook, which made them hard to find and - because
`fire-impacts update` replaces notebooks - hard to carry forward to a
newer template. They live in a single ``study.toml`` per project now,
and the notebooks read them.

The schema below is the single source of truth for three separate
consumers: this module's validation, the commented scaffold written by
`fire-impacts new`, and the drift report shown by `fire-impacts status`.
Keeping them on one definition is what stops a scaffold's explanatory
comments from describing settings the loader no longer accepts.

This is *not* ``parameters.json``. That file holds sparse calibration
overrides resolved through five scopes, and feeds a digest used to
detect that derived layers were built with different values than a run
resolves. A shapefile path has no calibration range and no scope, and
putting one in that digest would mean renaming a directory invalidated
every built layer. The two files answer different questions and stay
separate.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import tomllib
from dataclasses import dataclass, field, fields, replace

from ._schema import did_you_mean, from_dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_NAME = 'study.toml'

# The environment variable consulted when secrets.tern_api_key is blank,
# so anyone who followed the previous notebook instructions keeps working.
TERN_ENV_VAR = 'TERN_API_KEY'


###############################################################################
class StudyConfigError(Exception):
    """A study.toml is missing, unreadable, or does not match the schema."""


# ---------------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectSettings:
    """Where the project's data lives, and whether to start it afresh."""

    directory: str = field(default='.', metadata={
        'path': True,
        'help': 'project data root, relative to this file'})
    clear: bool | None = field(default=None, metadata={
        'example': 'false',
        'help': 'Leave this out and PrepareData starts from scratch, '
                'wiping existing project data. Set it false once the '
                'project holds work worth keeping.'})


@dataclass(frozen=True)
class CatchmentSettings:
    """The study area and the input rasters describing it."""

    name: str | None = field(default=None, metadata={
        'required': True,
        'help': 'name for this catchment within the project'})
    boundary: str | None = field(default=None, metadata={
        'required': True, 'path': True, 'must_exist': True,
        'help': 'boundary coverage (.shp/.geojson). Must carry a CRS - '
                'it becomes the CRS for everything else in the catchment.'})
    aridity: str | None = field(default=None, metadata={
        'required': True, 'path': True, 'must_exist': True,
        'help': 'aridity raster covering the catchment'})
    # These two read correctly whether the generated line is live or
    # commented out. A help string phrased as "default: <what happens
    # without it>" makes sense above a commented line and nonsense above
    # a seeded one - and the example seeds subcatchments.
    dem: str | None = field(default=None, metadata={
        'path': True, 'must_exist': True, 'example': "''",
        'help': 'DEM covering the catchment. Leave it out to download '
                'the GA 1 arc-second national DEM.'})
    subcatchments: str | None = field(default=None, metadata={
        'path': True, 'must_exist': True, 'example': "''",
        'help': 'subcatchment coverage (.shp/.geojson). Leave it out '
                'for no subcatchment reporting.'})
    subcatchment_id_field: str = field(default='SiteID', metadata={
        'example': "'SiteID'",
        'help': 'attribute naming each subcatchment. Must match the '
                'subcatchment names in your Source model.'})
    cell_size_m: float = field(default=30.0, metadata={
        'example': '30',
        'help': 'DEM cell size in metres, used to convert per-cell results '
                'to t/ha. Change it if your DEM is not 30 m.'})


@dataclass(frozen=True)
class EventSettings:
    """The fire being modelled."""

    name: str | None = field(default=None, metadata={
        'required': True,
        'help': 'name for this fire; becomes the Events/<name> directory'})
    fire_start: str | None = field(default=None, metadata={
        'required': True,
        'help': 'date the fire started, YYYY-MM-DD'})
    fire_end: str | None = field(default=None, metadata={
        'required': True,
        'help': 'date the fire ended, YYYY-MM-DD'})
    recovery_breakpoints: list[float] | None = field(default=None, metadata={
        'example': '[0, 1, 2, 3]',
        'help': 'default: const.DEFAULT_RECOVERY_BREAKPOINTS'})


@dataclass(frozen=True)
class EnsembleSettings:
    """The stochastic rainfall realisation driving the simulations."""

    name: str = field(default='stochastic', metadata={
        'help': 'names this climate realisation'})
    num_replicates: int = field(default=10, metadata={
        'example': '10',
        'help': 'replicates drawn from pyraingen'})
    inspect_replicate: int = field(default=9, metadata={
        'example': '9',
        'help': 'which one the single-run Simulation notebook plots. '
                'Numbered from zero, so it must be less than '
                'num_replicates - lower num_replicates and you must '
                'lower this too.'})
    mean_annual_rainfall: float | None = field(default=None, metadata={
        'example': '600',
        'help': 'mm; default: estimated from catchment lat/lon'})
    average_temperature: float | None = field(default=None, metadata={
        'example': '20',
        'help': 'degrees C; default: estimated from catchment lat/lon'})
    n_workers: int = field(default=10, metadata={
        'example': '10',
        'help': 'replicates run in parallel; cap this for your machine'})


@dataclass(frozen=True)
class ReportingSettings:
    """Thresholds the ensemble notebook's exceedance maps are drawn at.

    Reporting choices, not model calibration: they change what a map
    shows, never what the model computes. That is why they are here and
    not in parameters.json.
    """

    erosion_threshold_t_ha: float = field(default=0.5, metadata={
        'example': '0.5',
        'help': 'catchment erosion exceedance threshold, t/ha'})
    delivered_threshold_kg_ha: float = field(default=500.0, metadata={
        'example': '500',
        'help': 'delivered-load exceedance threshold, kg/ha'})


@dataclass(frozen=True)
class SecretSettings:
    """Credentials for the data services the preprocessing downloads from."""

    tern_api_key: str = field(default='', metadata={
        'secret': True, 'prompt': True,
        'help': 'Your TERN API key, needed to download soil data.\n'
                'Free - see the "Soils" section of PrepareData for how to '
                'get one.\n'
                f'If left blank, the {TERN_ENV_VAR} environment variable '
                'is used instead.'})


# The timesteps a Source model can be fed at, keyed by the pandas frequency
# source.timestep names. Two notebooks act on that one setting and have to
# agree: SimulationEnsemble records RUSLE at 'rusle_timestep' and saves the
# combined loads at the key, and SourceIntegration reads them back and
# labels the data sources it creates '<kg|mm>/<units>'. Source reads that
# label, so a frequency with no entry here is refused rather than labelled
# with a guess - a wrong label is a silent scaling error.
SOURCE_TIMESTEPS = {
    'D': {'units': 'day', 'rusle_timestep': '24h'},
    'h': {'units': 'hour', 'rusle_timestep': '1h'},
}


@dataclass(frozen=True)
class SourceIntegrationSettings:
    """Connecting to a running eWater Source instance via Veneer."""

    port: int = field(default=9876, metadata={
        'example': '9876',
        'help': 'Veneer port for the running Source instance. 9876 is the '
                'conventional one; note that connect_to_veneer() in the '
                'library defaults to 9877, which the notebooks never use '
                'because they always pass this setting.'})
    constituent: str | None = field(default=None, metadata={
        'example': "'TSS'",
        'help': 'default: auto-detected'})
    functional_unit: str | None = field(default=None, metadata={
        'example': "'Forested'",
        'help': 'default: auto-detected'})
    replicate: int = field(default=0, metadata={
        'example': '0',
        'help': 'which ensemble replicate to push into Source. Distinct '
                'from ensemble.inspect_replicate, which only chooses what '
                'the Simulation notebook plots.'})
    timestep: str = field(default='D', metadata={
        'example': "'D'",
        'help': "must match your Source model: 'D' for daily, 'h' hourly. "
                'SimulationEnsemble saves loads at this timestep, so set it '
                'before running that notebook.'})
    date_format: str = field(default='%d/%m/%Y', metadata={
        'example': "'%d/%m/%Y'",
        'help': 'how your Source install writes run-period dates'})
    output_dir: str = field(default='source_inputs', metadata={
        'example': "'source_inputs'",
        'help': 'where the generated CSVs are written, inside the project'})
    load_attenuation: float = field(default=10.0, metadata={
        'example': '10.0',
        'help': 'Load Distributor attenuation'})
    maximum_concentration: float = field(default=1000.0, metadata={
        'example': '1000.0',
        'help': 'Load Distributor concentration cap, mg/L'})
    tss_data_source: str = field(default='fire_tss', metadata={
        'example': "'fire_tss'",
        'help': 'name of the Source data source holding sediment load. '
                'Also names the CSV written for it, so keep it to '
                'characters a filename may contain.'})
    rainfall_data_source: str = field(default='stochastic_rain', metadata={
        'example': "'stochastic_rain'",
        'help': 'name of the Source data source holding rainfall. Also '
                'names the CSV written for it, so keep it to characters '
                'a filename may contain.'})

    def __post_init__(self):
        # Refused at load rather than in SourceIntegration: SimulationEnsemble
        # acts on this first, and finding out afterwards would cost a full
        # ensemble run.
        if self.timestep not in SOURCE_TIMESTEPS:
            raise ValueError(
                f'timestep = {self.timestep!r} is not a timestep this '
                f"library can feed Source; use 'D' for a daily model or "
                f"'h' for an hourly one.")

    ###########################################################################
    @property
    def timestep_units(self):
        """The unit word Source labels a per-timestep series with."""
        return SOURCE_TIMESTEPS[self.timestep]['units']

    ###########################################################################
    @property
    def rusle_timeseries_timestep(self):
        """
        The interval RUSLE has to record subcatchment loads at.

        Notes:
        - Anything coarser than the Source timestep cannot be split back
          out: resampling a daily series to hourly puts the whole day's
          load in its first hour.
        """
        return SOURCE_TIMESTEPS[self.timestep]['rusle_timestep']


@dataclass(frozen=True)
class StudySettings:
    """Everything one study needs a user to supply."""

    project: ProjectSettings = ProjectSettings()
    catchment: CatchmentSettings = CatchmentSettings()
    event: EventSettings = EventSettings()
    ensemble: EnsembleSettings = EnsembleSettings()
    reporting: ReportingSettings = ReportingSettings()
    secrets: SecretSettings = SecretSettings()
    source: SourceIntegrationSettings = SourceIntegrationSettings()

    ###########################################################################
    @property
    def root(self):
        """
        Return the directory the study.toml was read from.

        Returns:
        - Absolute path, as a string.
        --------------------------------------------------------------------
        Notes:
        - Raises rather than guessing (``os.getcwd()``) when this instance
          was never attached to a file - e.g. built directly, or produced
          by ``dataclasses.replace()``, which rebuilds through ``__init__``
          and so drops the attribute. A guessed path would be a plausible
          but wrong answer; failing loudly here is what stops that from
          surfacing three steps later as a mysteriously wrong directory.
        --------------------------------------------------------------------
        """
        if '_root' not in self.__dict__:
            raise StudyConfigError(
                'this StudySettings was not loaded from a file')
        return self._root

    ###########################################################################
    @property
    def provided(self):
        """
        Return the dotted names of settings the file actually set.

        Returns:
        - frozenset of names such as 'catchment.name'.
        --------------------------------------------------------------------
        Notes:
        - Distinguishing a value somebody chose from one that defaulted is
          what makes describe() worth reading.
        - Raises under the same circumstances as ``root`` - see its note.
        --------------------------------------------------------------------
        """
        if '_provided' not in self.__dict__:
            raise StudyConfigError(
                'this StudySettings was not loaded from a file')
        return self._provided

    ###########################################################################
    def secret(self, name):
        """
        Return a credential, from the config file or the environment.

        Parameters:
        - name: Setting name within [secrets], e.g. 'tern_api_key'.

        Returns:
        - The credential, as a string.
        --------------------------------------------------------------------
        Notes:
        - Raises here rather than at load time: only the soil download
          needs the TERN key, and a blank one must not stop someone
          running a notebook that never touches TERN.
        --------------------------------------------------------------------
        """
        known = {f.name for f in fields(SecretSettings)}
        if name not in known:
            raise StudyConfigError(
                f'Unknown secret {name!r}.{did_you_mean(name, known)}')

        value = (getattr(self.secrets, name).strip()
                 or os.environ.get(TERN_ENV_VAR, '').strip())
        if not value:
            raise StudyConfigError(
                f'No {name} available. Set it in the [secrets] section of '
                f'{CONFIG_NAME}, or set the {TERN_ENV_VAR} environment '
                f'variable. See the "Soils" section of the PrepareData '
                f'notebook for how to obtain one.'
            )
        return value

    ###########################################################################
    def describe_text(self):
        """
        Return a readable summary of the settings and their origins.

        Returns:
        - Multi-line string.
        --------------------------------------------------------------------
        Notes:
        - Secrets are reported as 'set'/'not set', never printed. Notebook
          output cells are saved into the .ipynb and get shared.
        --------------------------------------------------------------------
        """
        lines = [f'Study settings from {os.path.join(self.root, CONFIG_NAME)}',
                 '']
        for group in fields(StudySettings):
            value = getattr(self, group.name)
            for entry in fields(value):
                dotted = f'{group.name}.{entry.name}'
                is_provided = dotted in self.provided
                if entry.metadata.get('secret'):
                    own_value = getattr(value, entry.name).strip()
                    from_env = os.environ.get(TERN_ENV_VAR, '').strip()
                    shown = 'set' if (own_value or from_env) else 'not set'
                    origin = ('study.toml' if is_provided
                              else TERN_ENV_VAR if from_env else 'default')
                else:
                    shown = getattr(value, entry.name)
                    origin = 'study.toml' if is_provided else 'default'
                lines.append(f'  {dotted:<34} {str(shown):<44} ({origin})')

        lines += [
            '',
            'Calibration parameters are separate, and live in '
            'parameters.json;',
            'see the "Calibration parameters" section of the PrepareData '
            'notebook.',
        ]
        return '\n'.join(lines)

    ###########################################################################
    def describe(self):
        """
        Print a readable summary of the settings and their origins.
        --------------------------------------------------------------------
        """
        print(self.describe_text())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


###############################################################################
def config_path(path):
    """
    Return the path a project's study.toml would live at.

    Parameters:
    - path: Project directory.

    Returns:
    - Full path to the config file, which need not exist.
    --------------------------------------------------------------------
    """
    return os.path.join(path, CONFIG_NAME)


###############################################################################
def _flatten_provided(data, prefix=''):
    """
    Return the dotted names of every leaf present in a parsed file.

    Parameters:
    - data: Parsed TOML, as nested dicts.
    - prefix: Dotted prefix for recursion.

    Returns:
    - Set of dotted names.
    --------------------------------------------------------------------
    """
    names = set()
    for key, value in data.items():
        name = f'{prefix}{key}'
        if isinstance(value, dict):
            names |= _flatten_provided(value, f'{name}.')
        else:
            names.add(name)
    return names


###############################################################################
def _attach(settings, root, provided):
    """
    Attach a loaded StudySettings' origin, in one place.

    Parameters:
    - settings: The StudySettings just built.
    - root: Absolute path to the project directory it was read from.
    - provided: Dotted names of settings the file actually set.

    Returns:
    - settings, for convenience at the call site.
    --------------------------------------------------------------------
    Notes:
    - ``root`` and ``provided`` are not dataclass fields (a field would be
      a settable key in the TOML file), so they are attached to the
      instance's ``__dict__`` with ``object.__setattr__``. Routing every
      attachment through this one function - rather than two call sites -
      is what stops a later caller (Task 3 uses ``dataclasses.replace()``
      on a StudySettings) from having to remember to redo both by hand.
    --------------------------------------------------------------------
    """
    object.__setattr__(settings, '_root', root)
    object.__setattr__(settings, '_provided', frozenset(provided))
    return settings


###############################################################################
def _check_required(settings, path):
    """
    Raise if a setting marked required was not supplied.

    Parameters:
    - settings: The StudySettings just built.
    - path: Path to the config file, for the error message.
    --------------------------------------------------------------------
    """
    missing = []
    for group in fields(StudySettings):
        value = getattr(settings, group.name)
        for entry in fields(value):
            if not entry.metadata.get('required'):
                continue
            if getattr(value, entry.name) in (None, ''):
                missing.append(f'{group.name}.{entry.name}')

    if missing:
        raise StudyConfigError(
            f'{path} is missing {len(missing)} required setting(s): '
            f'{", ".join(missing)}. Every setting that is not commented '
            f'out in the file is one you need to supply.'
        )


###############################################################################
def _resolve_paths(settings, root):
    """
    Rewrite every path setting as an absolute path.

    Parameters:
    - settings: The StudySettings just built.
    - root: Directory the config file was read from.

    Returns:
    - A new StudySettings with paths resolved.
    --------------------------------------------------------------------
    Notes:
    - Relative to the config file, not the working directory: a notebook
      may be run from anywhere, and '..\\test_data\\x.shp' has to mean the
      same thing regardless.
    --------------------------------------------------------------------
    """
    groups = {}
    for group in fields(StudySettings):
        value = getattr(settings, group.name)
        changes = {}
        for entry in fields(value):
            if not entry.metadata.get('path'):
                continue
            raw = getattr(value, entry.name)
            if raw in (None, ''):
                continue
            changes[entry.name] = os.path.abspath(os.path.join(root, raw))
        groups[group.name] = replace(value, **changes) if changes else value

    return replace(settings, **groups)


###############################################################################
def _check_paths_exist(settings, path):
    """
    Raise if an input file named in the config is not there.

    Parameters:
    - settings: The StudySettings, with paths already resolved.
    - path: Path to the config file, for the error message.
    --------------------------------------------------------------------
    Notes:
    - Caught here rather than left to fail deep inside a read, so the
      message names the setting to fix instead of arriving as a geopandas
      traceback many cells into a notebook run.
    --------------------------------------------------------------------
    """
    missing = []
    for group in fields(StudySettings):
        value = getattr(settings, group.name)
        for entry in fields(value):
            if not entry.metadata.get('must_exist'):
                continue
            resolved = getattr(value, entry.name)
            if resolved and not os.path.exists(resolved):
                missing.append(
                    f'{group.name}.{entry.name}: no such file {resolved}')

    if missing:
        raise StudyConfigError(
            f'Input files named in {path} are missing:\n  '
            + '\n  '.join(missing)
        )


###############################################################################
def check_study(path):
    """
    Report how a project's study.toml compares with the current schema.

    Parameters:
    - path: Project directory.

    Returns:
    - Dict with keys 'missing_file', 'unknown', 'required_missing' and
      'unset'.
    --------------------------------------------------------------------
    Notes:
    - Reports rather than raises: `fire-impacts status` exists to tell a
      user what is wrong, so it must survive a file that will not load.
    - A setting in 'unset' is information, not a problem. It is how a
      user finds out that newer templates understand something their
      file does not mention.
    --------------------------------------------------------------------
    """
    report = {
        'missing_file': False,
        'unknown': [],
        'required_missing': [],
        'unset': [],
        }

    fn = config_path(path)
    if not os.path.exists(fn):
        report['missing_file'] = True
        return report

    try:
        with open(fn, 'rb') as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as exc:
        report['unknown'].append(f'{CONFIG_NAME} could not be read: {exc}')
        return report

    provided = _flatten_provided(data)
    blank = StudySettings()

    known_sections = {f.name for f in fields(StudySettings)}
    for section, contents in data.items():
        if not isinstance(contents, dict):
            # Mirrors load_study's "not inside a [section]" case: a plain
            # value where a table belongs, e.g. `catchment = "oops"`.
            # Iterating it below as if it were a section's settings would
            # walk its characters one at a time - report it here instead.
            report['unknown'].append(
                f'{section!r} is not inside a [section]; found a plain '
                f'value instead of a table.')
        elif section not in known_sections:
            report['unknown'].append(
                f'[{section}] is not a known section.'
                f'{did_you_mean(section, known_sections)}')

    for group in fields(StudySettings):
        value = getattr(blank, group.name)
        names = {f.name for f in fields(value)}

        section_data = data.get(group.name, {})
        if not isinstance(section_data, dict):
            section_data = {}

        for key in section_data:
            if key not in names:
                report['unknown'].append(
                    f'{group.name}.{key} is not a known setting.'
                    f'{did_you_mean(key, names)}')

        for entry in fields(value):
            dotted = f'{group.name}.{entry.name}'
            if isinstance(section_data.get(entry.name), dict):
                # A table where a scalar belongs, e.g. `boundary = {x = 1}`.
                # _flatten_provided turns this into 'catchment.boundary.x',
                # so the loop below would otherwise report the setting as
                # simply missing - true, but silent about what is actually
                # sitting there.
                report['unknown'].append(
                    f'{dotted} should be a plain value; found a table '
                    f'instead.')
            if dotted in provided:
                continue
            if entry.metadata.get('required'):
                report['required_missing'].append(dotted)
            else:
                report['unset'].append(dotted)

    return report


###############################################################################
def load_study(path='.'):
    """
    Read and validate a project's study.toml.

    Parameters:
    - path: Project directory holding the config file.

    Returns:
    - A StudySettings.
    --------------------------------------------------------------------
    Notes:
    - A setting left out of the file falls back to its default, silently.
      That is what lets a study.toml written against an older set of
      templates keep working after an update.
    - A setting that is *misspelled*, by contrast, raises. Ignoring it
      would let someone believe they had configured a study they had not.
    --------------------------------------------------------------------
    """
    fn = config_path(path)
    if not os.path.exists(fn):
        raise StudyConfigError(
            f'No {CONFIG_NAME} in {os.path.abspath(path)}. Run '
            f'"fire-impacts update {path}" to create one, or copy it '
            f'from another project.'
        )

    try:
        with open(fn, 'rb') as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as exc:
        raise StudyConfigError(f'{fn} could not be read: {exc}') from None

    known = {f.name for f in fields(StudySettings)}
    for key, value in data.items():
        if not isinstance(value, dict):
            # The most likely mistake a non-programmer makes: a setting
            # typed above the [section] it belongs in, rather than inside
            # one. "Unknown section [name]" would blame the wrong thing.
            raise StudyConfigError(
                f'{key!r} in {fn} is not inside a [section]. '
                f'Move it under one of: {sorted(known)}.'
            )
        if key not in known:
            raise StudyConfigError(
                f'Unknown section [{key}] in {fn}.'
                f'{did_you_mean(key, known)} '
                f'Valid sections: {sorted(known)}.'
            )

    try:
        settings = from_dict(
            StudySettings, data, path='', noun='setting')
    except ValueError as exc:
        raise StudyConfigError(f'{fn}: {exc}') from None

    root = os.path.abspath(path)
    settings = _resolve_paths(settings, root)
    _attach(settings, root, _flatten_provided(data))

    _check_required(settings, fn)
    _check_paths_exist(settings, fn)

    return settings
