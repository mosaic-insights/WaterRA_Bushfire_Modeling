"""
Schema machinery shared by the calibration parameters and the study
settings.

Both subsystems do the same three things to a dict that came out of a
file a user hand-edited: reject keys that are not in the schema (with a
suggestion, because a silently ignored typo lets someone believe they
configured something they did not), coerce values to the annotated type,
and merge a sparse override over a set of defaults.

These started life private in ``params.py``. They live here so that
``study.py`` can reuse them rather than grow a second, subtly different
copy of the same validation - two implementations of "did you spell this
right" would eventually disagree, and the disagreement would show up as
one file being stricter than the other for no reason a user could see.
"""

from __future__ import annotations

import difflib
import numbers
import types
from dataclasses import fields, is_dataclass
from typing import Any, Union, get_args, get_origin, get_type_hints


###############################################################################
def to_dict(obj: Any) -> Any:
    """Recursively convert nested frozen dataclasses to plain dicts."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj


###############################################################################
def did_you_mean(name: str, options) -> str:
    """Return a ' Did you mean X?' suffix, or '' if nothing is close."""
    close = difflib.get_close_matches(name, list(options), n=1, cutoff=0.6)
    return f' Did you mean {close[0]!r}?' if close else ''


###############################################################################
def from_dict(cls, data: dict, *, path: str, noun: str = 'parameter'):
    """
    Build a (possibly nested) dataclass from a dict, rejecting unknown keys.

    Parameters:
    - cls: Dataclass to build.
    - data: Dict of values, which may be sparse.
    - path: Dotted prefix used in error messages.
    - noun: What to call an entry in error messages - 'parameter' for
      calibration parameters, 'setting' for study settings.

    Returns:
    - An instance of cls.
    --------------------------------------------------------------------
    """
    if not isinstance(data, dict):
        raise ValueError(
            f'{path or noun + "s"}: expected an object, got '
            f'{type(data).__name__}.'
        )
    known = {f.name: f for f in fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in known:
            where = f'{path}{key}' if path else key
            raise ValueError(
                f'Unknown {noun} {where!r}.'
                f'{did_you_mean(key, known)} '
                f'Valid names here: {sorted(known)}.'
            )
        default = known[key].default
        if is_dataclass(default) and not isinstance(default, type):
            # Merge onto the default instance so a partial override of a
            # nested group works even when that group's own fields have
            # no defaults (DebrisDepthParams requires all five). The
            # study settings groups all have defaults, but they are read
            # the same way so the merge has to hold for both.
            if not isinstance(value, dict):
                raise ValueError(
                    f'{path}{key}: expected an object, got '
                    f'{type(value).__name__}.'
                )
            kwargs[key] = from_dict(
                type(default),
                deep_merge(to_dict(default), value),
                path=f'{path}{key}.',
                noun=noun,
            )
        else:
            kwargs[key] = coerce(
                value, hints(cls).get(key), f'{path}{key}',
            )
    try:
        return cls(**kwargs)
    except TypeError as exc:
        # A nested group with no default (DebrisDepthParams) needs every
        # field; surface that as a parameter - or setting - error rather
        # than a TypeError from a constructor the user never called.
        raise ValueError(f'{path or noun + "s"}: {exc}') from exc
    except ValueError as exc:
        # __post_init__ range errors name the field but not the group,
        # and a user editing one of three parameters.json layers, or one
        # section of a study.toml, needs the full path to know what to
        # change. Both arms re-prefix for the same reason.
        message = str(exc)
        prefix = path or ''
        if prefix and not message.startswith(prefix):
            message = f'{prefix}{message}'
        raise ValueError(message) from None


_TYPE_HINTS: dict = {}


###############################################################################
def hints(cls) -> dict:
    """Return (and cache) a dataclass's resolved type hints.

    ``from __future__ import annotations`` makes ``field.type`` a string,
    so the annotations have to be resolved before they can be used to
    coerce.
    """
    if cls not in _TYPE_HINTS:
        _TYPE_HINTS[cls] = get_type_hints(cls)
    return _TYPE_HINTS[cls]


###############################################################################
def coerce(value, annotation, path: str):
    """
    Coerce a JSON value to a field's annotated type, or raise.

    JSON has no int/float distinction, so a hand-edited ``1`` for a float
    field must normalise to ``1.0`` — otherwise it is a different value to
    the digest and would spuriously flag every derived layer stale. A
    string where a number belongs is rejected here rather than leaking a
    comparison TypeError out of __post_init__.
    """
    if annotation is None:
        return value

    # get_origin() returns types.UnionType for `X | None` on Python
    # 3.10-3.13 and typing.Union for typing.Optional[X]; 3.14 unified them.
    # Match both, or this branch is dead on the pinned runtime.
    if get_origin(annotation) in (Union, types.UnionType):
        args = get_args(annotation)
        if value is None and type(None) in args:
            return None
        for member in (a for a in args if a is not type(None)):
            try:
                return coerce(value, member, path)
            except ValueError:
                continue
        raise ValueError(
            f'{path}: expected one of '
            f'{[getattr(a, "__name__", a) for a in args]}, got '
            f'{type(value).__name__} ({value!r}).'
        )

    if annotation is bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f'{path} must be true or false, got {value!r}.')

    if annotation is int:
        # bool is a subclass of int; a JSON true is not a count. numbers.*
        # rather than int/float so numpy and pandas scalars (np.int64 out of
        # a DataFrame lookup) are accepted rather than rejected with a
        # confusing "must be a whole number".
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f'{path} must be a whole number, got {value!r}.')
        try:
            as_float = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f'{path}: {value!r} is out of range.') from exc
        if not as_float.is_integer():
            raise ValueError(f'{path} must be a whole number, got {value!r}.')
        return int(as_float)

    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f'{path} must be a number, got {value!r}.')
        try:
            return float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f'{path}: {value!r} is out of range.') from exc

    if annotation is str:
        if isinstance(value, str):
            return value
        raise ValueError(f'{path} must be a string, got {value!r}.')

    if annotation is list or get_origin(annotation) is list:
        if not isinstance(value, list):
            raise ValueError(f'{path} must be a list, got {value!r}.')
        member = (get_args(annotation) or (None,))[0]
        return [coerce(v, member, f'{path}[{i}]')
                for i, v in enumerate(value)]

    return value


###############################################################################
def deep_merge(base: dict, overlay: dict) -> dict:
    """Return a new dict with overlay merged recursively over base."""
    merged = dict(base)
    for key, value in overlay.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
