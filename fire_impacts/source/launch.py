"""
Getting a Veneer connection to the Source model a study names.

Two ways, chosen by the [source] section of study.toml:

* No ``project_file``: connect to a Source the user already has open, with
  Veneer serving on ``port``. Nothing is started or stopped here.
* ``project_file`` set: start a Veneer command line - Source without its
  interface - on that project. The command line is either one the user
  already has (``veneer_command_line``), or one built here from a Source
  install and a matching Veneer release (``source_dir`` + ``veneer_dir``)
  using veneer-py's ``create_command_line``.

Either way the caller gets a SourceSession, so a notebook reads the same
whichever applies and closes it the same way at the end.

The settings are checked here, when the notebook connects, rather than
when study.toml loads: they only matter to the notebook that talks to
Source, and the ensemble may well be run on a machine that has no Source
installed at all.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import json
import logging
import os

from veneer import Veneer
from veneer.manage import create_command_line, kill_all_now, start

from ..study import StudyConfigError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VENEER_EXE_NAME = 'FlowMatters.Source.VeneerCmd.exe'

# Written beside a command line built here, recording what it was built
# from. create_command_line's own reuse check is only "the exe exists",
# which cannot tell that study.toml now names a different Veneer release.
BUILD_STAMP = 'fire_impacts_command_line.json'

# The settings that only mean something when a command line is started.
_COMMAND_LINE_SETTINGS = (
    'veneer_command_line', 'source_dir', 'veneer_dir', 'command_line_dir',
    'plugins',
)


###############################################################################
class SourceSession:
    """
    A Veneer connection, and the command line behind it if one was started.

    Attributes:
    - v: Veneer client.
    - port: Port v talks to. For a started command line, the port it
      actually bound, which may not be the one requested.
    - launched: True if this session started a command line, which close()
      will stop.
    - project_file: Project the command line loaded, or None.
    """

    def __init__(self, v, port, processes=(), project_file=None):
        self.v = v
        self.port = port
        self.project_file = project_file
        self._processes = list(processes)

    ###########################################################################
    @property
    def launched(self):
        """True if this session started the Source it talks to."""
        return bool(self._processes)

    ###########################################################################
    def close(self):
        """
        Stop the command line this session started, if any.
        --------------------------------------------------------------------
        Notes:
        - A Source the user had open is left running: it was not ours.
        - Safe to call more than once. veneer-py also stops command lines
          when Python exits, so a notebook that never gets this far does
          not leave one behind.
        --------------------------------------------------------------------
        """
        if self._processes:
            processes, self._processes = self._processes, []
            kill_all_now(processes)
            logger.info('Stopped the Veneer command line on port %s',
                        self.port)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


###############################################################################
def _is_set(value):
    return value not in (None, '', [])


###############################################################################
def _require_file(path, setting):
    if not os.path.isfile(path):
        raise StudyConfigError(f'source.{setting}: no such file {path}')


###############################################################################
def _require_dir(path, setting):
    if not os.path.isdir(path):
        raise StudyConfigError(f'source.{setting}: no such directory {path}')


###############################################################################
def default_command_line_dir(source_dir):
    """
    Return where a command line for a Source install is built by default.

    Parameters:
    - source_dir: Source install directory.

    Returns:
    - Directory path, which need not exist yet.
    --------------------------------------------------------------------
    Notes:
    - Per user and named after the install directory - normally
      'Source <version>' - so two Source versions get two builds, and each
      is built once rather than per project.
    --------------------------------------------------------------------
    """
    root = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
    name = os.path.basename(os.path.normpath(source_dir))
    return os.path.join(root, 'fire_impacts', 'veneer_cmd', name)


###############################################################################
def _build_command_line(source_dir, veneer_dir, dest):
    """
    Build a command line from Source and Veneer, or reuse an earlier build.

    Parameters:
    - source_dir: Source install directory.
    - veneer_dir: Veneer release directory, for the same Source version.
    - dest: Directory to build into.

    Returns:
    - Path to the command line exe.
    """
    exe = os.path.join(dest, VENEER_EXE_NAME)
    stamp_path = os.path.join(dest, BUILD_STAMP)
    stamp = {'source_dir': source_dir, 'veneer_dir': veneer_dir}

    if os.path.isfile(exe) and os.path.isfile(stamp_path):
        try:
            with open(stamp_path) as f:
                if json.load(f) == stamp:
                    logger.info('Reusing the Veneer command line in %s', dest)
                    return exe
        except (ValueError, OSError):
            pass    # unreadable stamp: rebuild

    logger.info('Building a Veneer command line in %s from %s and %s. This '
                'copies all of Source, so it takes a while - once.',
                dest, source_dir, veneer_dir)
    exe = str(create_command_line(
        veneer_dir, source_version=None, source_path=source_dir,
        dest=dest, force=True))
    with open(stamp_path, 'w') as f:
        json.dump(stamp, f, indent=2)
    return exe


###############################################################################
def veneer_command_line(settings):
    """
    Return the Veneer command line to start, building one if need be.

    Parameters:
    - settings: The [source] settings - ``study.source``.

    Returns:
    - Path to FlowMatters.Source.VeneerCmd.exe.
    --------------------------------------------------------------------
    Notes:
    - Exactly one of ``veneer_command_line``, or ``source_dir`` with
      ``veneer_dir``, has to be set. Both at once is refused rather than
      one quietly winning.
    --------------------------------------------------------------------
    """
    prebuilt = settings.veneer_command_line
    source_dir = settings.source_dir
    veneer_dir = settings.veneer_dir

    if _is_set(prebuilt):
        if _is_set(source_dir) or _is_set(veneer_dir):
            raise StudyConfigError(
                'source.veneer_command_line is set, and so is '
                'source.source_dir or source.veneer_dir. Set one or the '
                'other: a command line you already have, or the Source and '
                'Veneer to build one from.')
        _require_file(prebuilt, 'veneer_command_line')
        return prebuilt

    if not _is_set(source_dir) and not _is_set(veneer_dir):
        raise StudyConfigError(
            'source.project_file is set, so the notebook starts a Veneer '
            'command line - but nothing says which. Set '
            'source.veneer_command_line to one you already have, or '
            'source.source_dir and source.veneer_dir to build one.')
    if not _is_set(veneer_dir):
        raise StudyConfigError(
            'source.source_dir is set but source.veneer_dir is not. Building '
            'a command line needs both: set source.veneer_dir to the Veneer '
            'release for that Source version.')
    if not _is_set(source_dir):
        raise StudyConfigError(
            'source.veneer_dir is set but source.source_dir is not. Building '
            'a command line needs both: set source.source_dir to the Source '
            'install directory.')

    _require_dir(source_dir, 'source_dir')
    _require_dir(veneer_dir, 'veneer_dir')
    dest = settings.command_line_dir or default_command_line_dir(source_dir)
    return _build_command_line(source_dir, veneer_dir, dest)


###############################################################################
def open_source(settings, *, debug=False):
    """
    Connect to the Source model study.toml names, starting it if need be.

    Parameters:
    - settings: The [source] settings - ``study.source``.
    - debug: Echo the command line's startup output, for when it will not
      start.

    Returns:
    - A SourceSession. Call close() on it when finished, or use it in a
      with-block.
    --------------------------------------------------------------------
    Notes:
    - A started command line loads ``project_file`` but never writes to
      it; changes reach disk only through v.model.save() to a filename
      the caller chooses.
    - IronPython scripting is always enabled on a started command line:
      v.model.* - saving the project included - depends on it.
    --------------------------------------------------------------------
    """
    if not _is_set(settings.project_file):
        stray = [name for name in _COMMAND_LINE_SETTINGS
                 if _is_set(getattr(settings, name))]
        if stray:
            raise StudyConfigError(
                f'{", ".join("source." + s for s in stray)} only apply when '
                f'the notebook starts a Veneer command line, which it does '
                f'when source.project_file is set. Set source.project_file, '
                f'or remove these to use the Source you have open.')
        v = Veneer(settings.port)
        logger.info('Connecting to the Source already open on port %s',
                    settings.port)
        return SourceSession(v, settings.port)

    _require_file(settings.project_file, 'project_file')
    for plugin in settings.plugins:
        if not os.path.isfile(plugin):
            raise StudyConfigError(f'source.plugins: no such file {plugin}')
    exe = veneer_command_line(settings)

    logger.info('Starting a Veneer command line on %s', settings.project_file)
    processes, ports, _ = start(
        project_fn=settings.project_file,
        n_instances=1,
        ports=settings.port,
        veneer_exe=exe,
        script=True,
        additional_plugins=list(settings.plugins),
        detached=settings.detached,
        debug=debug,
        return_log_paths=True,
    )
    port = ports[0]
    logger.info('Veneer command line serving on port %s', port)
    return SourceSession(Veneer(port), port, processes,
                         project_file=settings.project_file)
