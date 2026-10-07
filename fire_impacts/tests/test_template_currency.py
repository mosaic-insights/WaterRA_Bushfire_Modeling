"""
The shipped template notebooks stay current with the API they demonstrate.

Templates are package data now, so a stale one reaches every user who runs
`fire-impacts new`. Nothing else in the suite executes them: they need a
real catchment, remote imagery and a rainfall service, so running them in
CI is not on. This checks what can be checked statically — that every
attribute they name still exists, and that a method is not referenced
without being called.

Both classes of drift have already happened. `record.digest` became a
method during the provenance work and the templates kept using it as a
property, which fails at runtime with no error until that cell is run;
and `ctx.set_parameter_overrides` was renamed to
`set_event_parameter_overrides`, which would have failed the same way.
"""

import ast
import pathlib

import pytest

from fire_impacts.context import RunContext
from fire_impacts.params import ParameterRecord, resolve_parameters
from fire_impacts.pre.project import FireImpactsProject
from fire_impacts.provenance import RunProvenance
from fire_impacts.source.launch import SourceSession
from fire_impacts.study import StudyConfigError, StudySettings

TEMPLATE_DIR = pathlib.Path(__file__).resolve().parents[1] / 'templates'

# Receiver names the templates use for objects, and the type each holds.
# A name absent here is skipped rather than guessed at — better a gap than
# a false failure that trains people to ignore this test.
RECEIVER_TYPES = {
    'ctx': RunContext,
    'prep_ctx': RunContext,
    'ev': RunContext,
    'run': RunContext,
    'proj': FireImpactsProject,
    'record': ParameterRecord,
    'prov': RunProvenance,
    # The import form matters for this one, and only for this one: 'study'
    # is the first receiver name that is also a fire_impacts submodule, and
    # _module_receivers wins the tie. `from fire_impacts.study import
    # load_study` binds the function, leaves the name free, and is what the
    # templates do; `from fire_impacts import study` would bind the module
    # here and report every settings line as drift.
    'study': StudySettings,
    'session': SourceSession,
}


def templates():
    return sorted(TEMPLATE_DIR.glob('*.py'))


def test_the_templates_are_where_the_test_expects(self=None):
    assert TEMPLATE_DIR.is_dir(), TEMPLATE_DIR
    assert templates(), f'no templates found in {TEMPLATE_DIR}'


def _module_receivers(tree):
    """Map local names to fire_impacts modules, from the template's own
    imports — so `rusle` resolves to pre.rusle or sim.rusle according to
    what that template actually imported."""
    import importlib

    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if not (node.module or '').startswith('fire_impacts'):
                continue
            for alias in node.names:
                try:
                    module = importlib.import_module(
                        f'{node.module}.{alias.name}')
                except ImportError:
                    continue
                found[alias.asname or alias.name] = module
    return found


# Outcomes of looking an attribute up on a probe, distinguished because a
# name that is there but cannot be evaluated is not the same thing as a
# name that is gone.
_ABSENT = object()
_UNEVALUATED = object()


def _probe(owner, attr):
    """Look an attribute up on a resolved receiver, without ever raising.

    Parameters:
    - owner: The object the attribute should exist on.
    - attr: Attribute name.

    Returns:
    - The attribute's value, `_ABSENT` if the name is not there at all, or
      `_UNEVALUATED` if it is there but evaluating it raised.
    --------------------------------------------------------------------
    Notes:
    - `StudySettings.root` and `.provided` raise `StudyConfigError` on any
      instance that did not come from `load_study`, and the probe is a
      bare `StudySettings()`. Neither `hasattr` nor `getattr(..., None)`
      helps: both swallow only AttributeError, so the study error escapes
      and takes the whole test with it. They are reported as present but
      unevaluable here — a property that raises on a probe is a real part
      of the API, not drift, and a checker that called it drift would be
      a false failure of exactly the kind this file exists to avoid.
    --------------------------------------------------------------------
    """
    try:
        return getattr(owner, attr)
    except AttributeError:
        return _ABSENT
    except Exception:
        return _UNEVALUATED


def _next_owner(owner, attr):
    """Return the object the next hop of a chain resolves against, or None.

    Parameters:
    - owner: The object the current hop was looked up on.
    - attr: The current hop's attribute name.

    Returns:
    - The value the attribute holds, or None when the next hop cannot be
      checked and the walk must stop.
    --------------------------------------------------------------------
    Notes:
    - A missing name, one that raised, and a method all stop the walk. So
      does an unset optional setting, by falling through and returning the
      None it holds: `catchment.dem` is None on a bare probe but a path
      string in a real study, and resolving a further hop against None
      would invent drift that is not there. Stopping leaves a gap;
      guessing trains people to ignore this test, which is worse.
    --------------------------------------------------------------------
    """
    value = _probe(owner, attr)
    if value is _ABSENT or value is _UNEVALUATED:
        return None
    if callable(value) and not isinstance(value, type):
        return None
    return value


# Probe instances, module-level so the chain walk can reach them without
# being handed them. Populated by the `probes` fixture, which is autouse
# so the walk is never asked to run against an empty map.
_PROBES = {}


def _attribute_uses(path):
    """Yield every attribute access in a template whose receiver resolves.

    Parameters:
    - path: Path to a template script.

    Returns:
    - An iterator of (lineno, receiver_path, attribute, owner, was_called),
      where `receiver_path` is the dotted source text left of the
      attribute — 'study.reporting' for the third hop of
      `study.reporting.erosion_threshold_t_ha` — and `owner` is the object
      the attribute has to exist on: a probe instance, a module, or
      whatever the previous hop resolved to.
    --------------------------------------------------------------------
    Notes:
    - Chains are followed hop by hop: in `record.parameters.delivery.max_sdr`
      `record.parameters` resolves first, then `delivery` is looked up on
      what that returned, and so on. Requiring the receiver to be a bare
      `ast.Name` — as this did originally — checked the first hop and let
      every later one through unverified, which is the shape a settings
      block and much of the provenance walkthrough are made of.
    - `receiver_path` accumulates rather than repeating the root name, so
      a failure message quotes text a reader can find in the template. The
      root alone would render a deep miss as `study.no_such_threshold`,
      and `name` lives on three of the settings groups, so the root would
      not even say which one drifted.
    - `owner` arrives already substituted for its probe, so a caller reads
      the same instance the walk did rather than re-deriving it.
    - Following a chain means `getattr` now fires on *intermediate*
      objects, not only on the root receiver. A chain passing through an
      expensive or side-effecting property would evaluate it, which the
      bare-name version never did. Nothing the current templates reach has
      one, but a template that walked into such a property would pay for
      it here.
    --------------------------------------------------------------------
    """
    tree = ast.parse(path.read_text())
    called = {id(node.func) for node in ast.walk(tree)
              if isinstance(node, ast.Call)}
    modules = _module_receivers(tree)

    def receiver(node):
        """Resolve the bare name a chain is rooted in, or None.

        A module is used as it stands. A RECEIVER_TYPES entry resolves to
        its probe *instance* or to nothing at all: falling back to the
        bare class would report every instance attribute — `record.
        parameters`, `record.sources` — as drift, a false failure of
        exactly the kind this file exists to avoid."""
        module = modules.get(node.id)
        if module is not None:
            return module
        owner = RECEIVER_TYPES.get(node.id)
        return None if owner is None else _PROBES.get(owner)

    def chain(node):
        """Split an attribute node into the name it is rooted in and the
        hops leading outwards from it, or (None, None) when the root is
        anything but a bare name — a call result, a subscript, a literal."""
        hops = []
        current = node
        while isinstance(current, ast.Attribute):
            hops.append(current)
            current = current.value
        if not isinstance(current, ast.Name):
            return None, None
        return current, list(reversed(hops))

    # ast.walk reaches the outermost node of a chain first and the whole
    # chain is handled from there, so the inner nodes it reaches later
    # would otherwise be re-walked and re-reported.
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or id(node) in seen:
            continue

        name_node, hops = chain(node)
        if name_node is None:
            continue
        seen.update(id(hop) for hop in hops)

        owner = receiver(name_node)
        receiver_path = name_node.id
        for hop in hops:
            if owner is None:
                break
            yield (hop.lineno, receiver_path, hop.attr, owner,
                   id(hop) in called)
            owner = _next_owner(owner, hop.attr)
            receiver_path = f'{receiver_path}.{hop.attr}'


@pytest.fixture(scope='module', autouse=True)
def probes(tmp_path_factory):
    """Real instances to resolve attributes against.

    Instances rather than classes, because instance attributes set in
    __init__ (project.catchments) and dataclass fields (record.sources)
    are invisible on the class and would look like drift.

    autouse because `_attribute_uses` reads `_PROBES` off the module
    rather than being handed it: a test that walked a template without
    requesting this fixture would check nothing at all.
    """
    root = tmp_path_factory.mktemp('currency')
    project = FireImpactsProject(str(root / 'proj'), exist_ok=False)
    project.catchments.append('probe')
    _PROBES.update({
        FireImpactsProject: project,
        RunContext: RunContext(
            project=project, catchment='probe', event='e', ensemble='n'),
        ParameterRecord: resolve_parameters([]),
        RunProvenance: RunProvenance(
            run={}, parameters=resolve_parameters([]), inputs={},
            section='Results'),
        StudySettings: StudySettings(),
        # No Veneer behind it: `session.v` resolves to None and the walk
        # stops there, which is all a template's use of it needs.
        SourceSession: SourceSession(v=None, port=9876),
    })
    return _PROBES


@pytest.mark.parametrize(
    'template', templates(), ids=lambda p: p.name)
def test_every_attribute_a_template_names_still_exists(template):
    missing = []
    for lineno, name, attr, owner, _called in _attribute_uses(template):
        if _probe(owner, attr) is _ABSENT:
            missing.append(f'{template.name}:{lineno} {name}.{attr}')
    assert not missing, (
        'templates reference attributes that no longer exist: '
        + '; '.join(missing)
    )


@pytest.mark.parametrize(
    'template', templates(), ids=lambda p: p.name)
def test_no_template_references_a_method_without_calling_it(template):
    """The failure that shipped: record.digest became a method and the
    template kept using it as a property. Evaluating it produces a bound
    method rather than a value, silently, until someone reads the cell
    output and wonders."""
    uncalled = []
    for lineno, name, attr, owner, called in _attribute_uses(template):
        if called:
            continue
        value = _probe(owner, attr)
        if value is _ABSENT or value is _UNEVALUATED:
            continue
        if callable(value) and not isinstance(value, type):
            uncalled.append(f'{template.name}:{lineno} {name}.{attr}')
    assert not uncalled, (
        'templates reference methods without calling them (add "()"): '
        + '; '.join(uncalled)
    )


@pytest.mark.parametrize(
    'template', templates(), ids=lambda p: p.name)
def test_every_template_parses(template):
    """A template that does not parse cannot be converted to a notebook,
    so `fire-impacts new` would fail on a fresh project."""
    ast.parse(template.read_text())


def test_prepare_data_opens_a_project_that_already_exists(tmp_path):
    """PrepareData's own `FireImpactsProject(...)` call, executed with
    `clear = false` in study.toml.

    The static checks above see that the name exists; they cannot see that
    the arguments make sense. `fire-impacts new` leaves a settings.json and
    a Catchments/ in the project folder, so even the FIRST PrepareData run
    in a fresh project meets a folder that exists — and a construction
    without `exist_ok` raises FileExistsError there, naming neither the
    setting nor study.toml. `clear = false` is exactly what the generated
    study.toml tells a user to set once the project holds work worth
    keeping, so that failure is on the recommended path.
    """
    source = (TEMPLATE_DIR / 'PrepareData.py').read_text(encoding='utf-8')
    calls = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'FireImpactsProject'
    ]
    assert len(calls) == 1, (
        f'expected one FireImpactsProject(...) call in PrepareData.py, '
        f'found {len(calls)}')

    # Evaluate the template's call verbatim, so the arguments under test
    # are the ones a user actually runs rather than a copy of them here.
    # eval is safe and is the point: the only expression compiled is the
    # single Call node located above, out of a template shipped inside
    # this package - not input from anywhere a user can reach.
    code = compile(
        ast.Expression(body=calls[0]), '<PrepareData.py>', 'eval')
    namespace = {
        'FireImpactsProject': FireImpactsProject,
        'PROJECT_DIR': str(tmp_path / 'proj'),
        'CLEAR': False,
    }

    first = eval(code, namespace)
    first.catchments.append('Kept')
    first._write()

    second = eval(code, namespace)  # must not raise FileExistsError
    assert second.catchments == ['Kept'], (
        'the second run re-initialised the project instead of reopening it')


def test_prepare_data_still_clears_when_asked_to(tmp_path):
    """The same call with `CLEAR = True` — the default, and what every
    existing user gets — must go on wiping the project."""
    source = (TEMPLATE_DIR / 'PrepareData.py').read_text(encoding='utf-8')
    call = next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'FireImpactsProject'
    )
    code = compile(ast.Expression(body=call), '<PrepareData.py>', 'eval')
    namespace = {
        'FireImpactsProject': FireImpactsProject,
        'PROJECT_DIR': str(tmp_path / 'proj'),
        'CLEAR': True,
    }

    first = eval(code, namespace)
    first.catchments.append('Gone')
    first._write()

    second = eval(code, namespace)
    assert second.catchments == [], (
        'clear = true must not carry the old catchments forward')


def test_the_checker_follows_a_chain_of_attributes(tmp_path):
    """`study.catchment.name` must have BOTH hops checked. Before this,
    _attribute_uses required node.value to be an ast.Name, so only the
    first hop of any chain was verified and the rest went unchecked —
    which is exactly the shape every settings block uses."""
    script = tmp_path / 'Chained.py'
    script.write_text(
        'study = None\n'
        'x = study.catchment.no_such_setting\n',
        encoding='utf-8')

    found = [(name, attr) for _, name, attr, _, _
             in _attribute_uses(script)]
    assert ('study', 'catchment') in found
    assert any(attr == 'no_such_setting' for _, attr in found)


def test_a_later_hop_naming_a_setting_the_schema_lacks_is_reported(tmp_path):
    """What following chains is *for*. A settings block that names a
    setting study.toml would reject has to reach the missing list, while
    a real setting sitting next to it — `catchment.name`, which is None
    until somebody fills the file in — must not."""
    script = tmp_path / 'Settings.py'
    script.write_text(
        'study = None\n'
        'name = study.catchment.name\n'
        'oops = study.catchment.no_such_setting\n',
        encoding='utf-8')

    missing = [attr for _, _, attr, owner, _ in _attribute_uses(script)
               if _probe(owner, attr) is _ABSENT]
    assert missing == ['no_such_setting']


def test_a_deep_miss_reports_a_dotted_path_a_reader_can_search_for(tmp_path):
    """The failure message has to quote text that is really in the
    template. Naming the root receiver alone renders a third-hop miss as
    `study.no_such_threshold`, which appears nowhere in the source — and
    `name` sits on catchment, event and ensemble alike, so the root would
    not even say which group had drifted. Tasks 9-11 make every settings
    line a chain of this shape, so every future failure of this test
    depends on getting it right."""
    script = tmp_path / 'Deep.py'
    script.write_text(
        'study = None\n'
        'x = study.reporting.no_such_threshold\n',
        encoding='utf-8')

    reported = [f'{path}.{attr}' for _, path, attr, owner, _
                in _attribute_uses(script)
                if _probe(owner, attr) is _ABSENT]
    assert reported == ['study.reporting.no_such_threshold']


def test_a_receiver_with_no_probe_is_skipped_rather_than_guessed_at(tmp_path):
    """`_attribute_uses` reads `_PROBES` off the module. Falling back to
    the bare receiver class when it is empty would report
    `record.parameters` — a dataclass field, invisible on the class — as
    drift. Skipping is the bargain this file states: better a gap than a
    false failure."""
    script = tmp_path / 'NoProbe.py'
    script.write_text(
        'record = None\n'
        'x = record.parameters.delivery.max_sdr\n',
        encoding='utf-8')

    assert [attr for _, _, attr, _, _ in _attribute_uses(script)] == [
        'parameters', 'delivery', 'max_sdr']

    saved = dict(_PROBES)
    _PROBES.clear()
    try:
        without_probes = list(_attribute_uses(script))
    finally:
        _PROBES.update(saved)
    assert without_probes == []


def test_the_walk_stops_at_a_hop_it_cannot_resolve(tmp_path, probes):
    """Two hops nothing sensible can be said past. `record.digest` is a
    method, and `study.catchment.dem` is None on a bare probe but a path
    string in a real study. Resolving a further hop against either would
    invent drift; the walk stops instead. A false failure trains people
    to ignore this test, which is worse than the gap."""
    script = tmp_path / 'Stops.py'
    script.write_text(
        'record = study = None\n'
        'x = record.digest.no_such_attribute\n'
        'y = study.catchment.dem.no_such_attribute\n',
        encoding='utf-8')

    attrs = [attr for _, _, attr, _, _ in _attribute_uses(script)]
    assert attrs == ['digest', 'catchment', 'dem']
    assert _next_owner(probes[ParameterRecord], 'digest') is None
    assert _next_owner(probes[StudySettings].catchment, 'dem') is None


def test_a_property_that_raises_on_the_probe_is_not_called_drift(probes):
    """`StudySettings.root` raises unless the instance came from
    `load_study`, and the probe is a bare `StudySettings()`. It is part of
    the API all the same, so it must not be reported missing — and the
    lookup must not let the error escape and take the run with it."""
    settings = probes[StudySettings]
    with pytest.raises(StudyConfigError):
        settings.root
    with pytest.raises(StudyConfigError):
        # Neither hasattr nor getattr's default swallows this: both catch
        # AttributeError only. Hence _probe.
        hasattr(settings, 'root')

    assert _probe(settings, 'root') is _UNEVALUATED
    assert _probe(settings, 'provided') is _UNEVALUATED
    assert _probe(settings, 'no_such_attribute') is _ABSENT
    assert _probe(settings, 'catchment') is settings.catchment
    assert _next_owner(settings, 'root') is None


def test_the_checker_would_catch_the_drift_it_exists_for(probes):
    """Guards the guard: if the receiver map or the call detection broke,
    the tests above would pass vacuously on a stale template."""
    source = 'record.digest\nrecord.sources_for("default")\n'
    tree = ast.parse(source)
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    uses = [
        (n.attr, id(n) in called) for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
    ]
    assert ('digest', False) in uses
    assert ('sources_for', True) in uses
    record = probes[ParameterRecord]
    assert callable(getattr(record, 'digest'))
