"""
The shared schema helpers behave the same after being lifted out of
params.py, and handle the list annotation that study settings need.
"""

from dataclasses import dataclass, field

import pytest

from fire_impacts import _schema


@dataclass(frozen=True)
class Inner:
    a: int = 1
    b: float = 2.0


@dataclass(frozen=True)
class Outer:
    inner: Inner = Inner()
    name: str = 'x'
    items: list | None = None
    plain_items: list = field(default_factory=list)


def test_unknown_key_raises_with_a_suggestion():
    with pytest.raises(ValueError, match="Did you mean 'name'"):
        _schema.from_dict(Outer, {'nme': 'y'}, path='')


def test_the_noun_in_the_error_is_configurable():
    with pytest.raises(ValueError, match='Unknown setting'):
        _schema.from_dict(Outer, {'nope': 1}, path='', noun='setting')


def test_the_default_noun_is_parameter():
    with pytest.raises(ValueError, match='Unknown parameter'):
        _schema.from_dict(Outer, {'nope': 1}, path='')


def test_a_partial_nested_override_keeps_the_other_defaults():
    result = _schema.from_dict(Outer, {'inner': {'a': 5}}, path='')
    assert result.inner.a == 5
    assert result.inner.b == 2.0


def test_an_int_for_a_float_field_normalises():
    result = _schema.from_dict(Outer, {'inner': {'b': 3}}, path='')
    assert isinstance(result.inner.b, float)


def test_a_string_where_a_number_belongs_is_rejected():
    with pytest.raises(ValueError, match='must be a whole number'):
        _schema.from_dict(Outer, {'inner': {'a': 'five'}}, path='')


def test_a_list_field_accepts_a_list():
    result = _schema.from_dict(Outer, {'items': [0, 1, 2]}, path='')
    assert result.items == [0, 1, 2]


def test_an_optional_list_field_rejects_a_scalar():
    """`list | None` goes through coerce's union branch, which reports the
    alternatives rather than the list specifically. Asserted separately from
    the plain-list case so a change to either message is visible."""
    with pytest.raises(ValueError, match='expected one of'):
        _schema.from_dict(Outer, {'items': 3}, path='')


def test_a_plain_list_field_rejects_a_scalar():
    with pytest.raises(ValueError, match='must be a list'):
        _schema.from_dict(Outer, {'plain_items': 3}, path='')


def test_params_still_exposes_the_old_private_names():
    """params.py re-exports them, so anything importing them keeps working."""
    from fire_impacts import params
    assert params._did_you_mean('nme', ['name']) == " Did you mean 'name'?"
    assert params._deep_merge({'a': {'b': 1}}, {'a': {'c': 2}}) == {
        'a': {'b': 1, 'c': 2}}
