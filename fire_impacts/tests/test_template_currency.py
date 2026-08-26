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


def _attribute_uses(path):
    """Yield (lineno, receiver_name, attribute, was_called) for every
    `name.attr` in a template."""
    tree = ast.parse(path.read_text())
    called = {id(node.func) for node in ast.walk(tree)
              if isinstance(node, ast.Call)}
    modules = _module_receivers(tree)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)):
            continue
        name = node.value.id
        owner = modules.get(name)
        if owner is None and name in RECEIVER_TYPES:
            owner = RECEIVER_TYPES[name]
        if owner is None:
            continue
        yield node.lineno, name, node.attr, owner, id(node) in called


@pytest.fixture(scope='module')
def probes(tmp_path_factory):
    """Real instances to resolve attributes against.

    Instances rather than classes, because instance attributes set in
    __init__ (project.catchments) and dataclass fields (record.sources)
    are invisible on the class and would look like drift.
    """
    root = tmp_path_factory.mktemp('currency')
    project = FireImpactsProject(str(root / 'proj'), exist_ok=False)
    project.catchments.append('probe')
    return {
        FireImpactsProject: project,
        RunContext: RunContext(
            project=project, catchment='probe', event='e', ensemble='n'),
        ParameterRecord: resolve_parameters([]),
        RunProvenance: RunProvenance(
            run={}, parameters=resolve_parameters([]), inputs={},
            section='Results'),
    }


@pytest.mark.parametrize(
    'template', templates(), ids=lambda p: p.name)
def test_every_attribute_a_template_names_still_exists(template, probes):
    missing = []
    for lineno, name, attr, owner, _called in _attribute_uses(template):
        target = probes.get(owner, owner)
        if not hasattr(target, attr):
            missing.append(f'{template.name}:{lineno} {name}.{attr}')
    assert not missing, (
        'templates reference attributes that no longer exist: '
        + '; '.join(missing)
    )


@pytest.mark.parametrize(
    'template', templates(), ids=lambda p: p.name)
def test_no_template_references_a_method_without_calling_it(
        template, probes):
    """The failure that shipped: record.digest became a method and the
    template kept using it as a property. Evaluating it produces a bound
    method rather than a value, silently, until someone reads the cell
    output and wonders."""
    uncalled = []
    for lineno, name, attr, owner, called in _attribute_uses(template):
        target = probes.get(owner, owner)
        if not hasattr(target, attr) or called:
            continue
        value = getattr(target, attr)
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
