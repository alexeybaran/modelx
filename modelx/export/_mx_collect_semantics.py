"""Collection domain and per-policy result semantics shared by all runtimes."""

from collections import OrderedDict
from array import array
from collections.abc import Iterable, Sized
from copy import deepcopy
from itertools import chain


def normalize_domain(value):
    if (
        not isinstance(value, Sized)
        or not isinstance(value, Iterable)
        or isinstance(value, (str, bytes, set, frozenset))
        or iter(value) is value
    ):
        raise TypeError("collect_over domain must be finite, ordered and re-iterable")
    domain = tuple(value)
    if not domain:
        raise ValueError("collect_over domain must not be empty")
    try:
        unique = set(domain)
    except TypeError as exc:
        raise TypeError("collect_over coordinates must be hashable") from exc
    if len(unique) != len(domain):
        raise ValueError("collect_over domain contains duplicate coordinates")
    return domain


_IMMUTABLE_SCALARS = (float, int, bool, str, bytes, complex, type(None))


class CollectedSeries(dict):
    """A detached named output indexed by its original key or tuple of keys."""

    __slots__ = ()

    def __call__(self, key, /, *rest):
        return self[(key, *rest) if rest else key]


def numeric_keys(domain, layouts):
    """Inspect only bounded, immutable built-in domains, after the policy factory."""
    identity = id(domain)
    known = layouts.get(identity)
    if known is not None and known[0] is domain:
        return known[1]
    if type(domain) not in (tuple, range):
        return None
    try:
        size = len(domain)
    except OverflowError:
        return None
    if size > 4096:
        return None
    keys = tuple(domain)
    if (any(type(k) is not int or not -(1 << 63) <= k < (1 << 63) for k in keys)
            or len(set(keys)) != size):
        return None
    if len(layouts) < 16:
        layouts[identity] = domain, keys
    return keys


class PackedFloatSeries:
    """Internal sum leaf containing detached native doubles in source key order."""

    __slots__ = ('keys', 'values', 'add_values')

    def __init__(self, keys, add_values):
        self.keys = keys
        self.values = array('d', [0.0]) * len(keys)
        self.add_values = add_values

    def as_dict(self):
        return CollectedSeries(zip(self.keys, self.values))


def callable_fields(fields):
    # Native sum evaluators can supply a plain dict without visiting prepare_output.
    for name, value in fields.items():
        if type(value) is PackedFloatSeries:
            fields[name] = value.as_dict()
        elif type(value) is dict:
            fields[name] = CollectedSeries(value)
    return fields


def _all_scalars(items):
    for item in items:
        if type(item) not in _IMMUTABLE_SCALARS:
            return False
    return True


def _detach(value, live_types, dict_type=dict):
    kind = type(value)
    if kind in _IMMUTABLE_SCALARS:
        return value
    # A flat container of immutable scalars (exact types) needs no walk: its shallow copy equals the deep copy,
    # and no live object can be among the items. Anything else (nested, mutable or other items) as below.
    if kind in (dict, CollectedSeries):
        if _all_scalars(value) and _all_scalars(value.values()):
            return dict_type(value)
    elif kind is list:
        if _all_scalars(value):
            return list(value)
    elif kind is tuple:
        if _all_scalars(value):
            return value
    remaining = [value]
    seen = set()
    while remaining:
        item = remaining.pop()
        if isinstance(item, live_types):
            raise TypeError("collect_outputs must return data, not live modelx objects")
        if id(item) not in seen:
            seen.add(id(item))
            if type(item) in (dict, CollectedSeries):
                remaining.extend(chain(item, item.values()))
            elif type(item) in (list, tuple, set, frozenset):
                remaining.extend(item)
    return deepcopy(value)


def prepare_output(value, live_types):
    if type(value) is not dict:
        raise TypeError("collect_outputs must return a plain dict with 'individual' and/or 'sum' sections")
    unexpected = value.keys() - {"individual", "sum"}
    if unexpected:
        raise ValueError("collect_outputs contains unsupported top-level section(s): "
                         + ", ".join(sorted(map(str, unexpected))))
    if not value:
        raise ValueError("collect_outputs must declare at least one of 'individual' or 'sum'")
    prepared = {}
    for mode, outputs in value.items():
        if type(outputs) is not dict or any(not isinstance(name, str) for name in outputs):
            raise TypeError(f"collect_outputs[{mode!r}] must be a plain dict with string output names")
        if not outputs:
            raise ValueError(f"collect_outputs[{mode!r}] must not be empty")
        prepared[mode] = {name: _detach(item, live_types, CollectedSeries) for name, item in outputs.items()}
    return prepared


def detached_numeric_sum(outputs, live_types):
    # A non-native domain was already evaluated in source order on the root.
    # Detach its ordinary dictionary at the same point as prepare_output.
    for name, value in outputs.items():
        if type(value) is dict:
            outputs[name] = _detach(value, live_types, CollectedSeries)
    return {'sum': outputs}


def _shape(value):
    if type(value) in (dict, CollectedSeries):
        return {name: _shape(item) for name, item in value.items()}
    return None


def _check_shape(expected, value, path):
    if expected is None and type(value) not in (dict, CollectedSeries):
        return
    if expected is None or type(value) not in (dict, CollectedSeries) or tuple(value) != tuple(expected):
        raise ValueError(f"collect_outputs structure changed between loop iterations at {path}")
    for name, item in value.items():
        _check_shape(expected[name], item, path + (name,))


def _add(old, new, path):
    if type(old) is PackedFloatSeries:
        if type(new) is PackedFloatSeries and old.keys == new.keys:
            old.add_values(old.values, new.values)
            return old
        old = old.as_dict()
    if type(new) is PackedFloatSeries:
        new = new.as_dict()
    if type(old) in (dict, CollectedSeries) or type(new) in (dict, CollectedSeries):
        if type(old) not in (dict, CollectedSeries) or type(new) not in (dict, CollectedSeries) or tuple(old) != tuple(new):
            raise ValueError(f"collect_outputs structure changed between loop iterations at {path}")
        for name, item in new.items():
            old[name] = _add(old[name], item, path + (name,))
        return old
    try:
        return old + new
    except Exception as exc:
        raise TypeError(f"collect_outputs 'sum' leaf does not support addition at {path}") from exc


class Accumulator:
    """Fold detached per-policy outputs in the declared policy order."""

    def __init__(self, result_type):
        self._result_type = result_type
        self._values = None
        self._individual_shape = None

    def add(self, coordinate, outputs):
        if self._values is None:
            self._values = {mode: OrderedDict() if mode == "individual" else None for mode in outputs}
        elif tuple(self._values) != tuple(outputs):
            raise ValueError("collect_outputs modes changed between loop iterations")
        for mode, values in outputs.items():
            if mode == "individual":
                if self._individual_shape is None:
                    self._individual_shape = _shape(values)
                else:
                    _check_shape(self._individual_shape, values, (mode,))
                self._values[mode][coordinate] = self._result_type(values)
            else:
                current = self._values[mode]
                self._values[mode] = values if current is None else _add(current, values, (mode,))

    def result(self):
        return self._result_type({
            mode: values if mode == "individual" else self._result_type(values)
            for mode, values in self._values.items()
        })
