"""Collection domain and per-policy result semantics shared by all runtimes."""

from collections import OrderedDict
from array import array
from collections.abc import ItemsView, Iterable, MutableMapping, Sized, ValuesView
from threading import Lock
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

    @classmethod
    def from_buffer(cls, keys, values, add_values):
        """Take ownership of the evaluator's completed, detached buffer."""
        if len(keys) != len(values):
            raise ValueError("incompatible numeric collect widths")
        if len(set(keys)) != len(keys):
            return CollectedSeries(zip(keys, values))
        packed = cls.__new__(cls)
        packed.keys = keys
        packed.values = values
        packed.add_values = add_values
        return packed


class NativeCollectedSeries(MutableMapping):
    """A callable month mapping over a detached contiguous numeric buffer."""

    __slots__ = ('_start', '_storage', '_lock')

    def __init__(self, start, values):
        self._start = start
        self._storage = values
        self._lock = Lock()

    def _dictionary(self):
        with self._lock:
            storage = self._storage
            if type(storage) is not CollectedSeries:
                keys = range(self._start, self._start + len(storage))
                storage = self._storage = CollectedSeries(zip(keys, storage))
            return storage

    def __len__(self):
        return len(self._storage)

    def __iter__(self):
        storage = self._storage
        if type(storage) is CollectedSeries:
            return iter(storage)
        return iter(range(self._start, self._start + len(storage)))

    def __reversed__(self):
        storage = self._storage
        if type(storage) is CollectedSeries:
            return reversed(storage)
        return reversed(range(self._start, self._start + len(storage)))

    def __getitem__(self, key):
        storage = self._storage
        if type(storage) is CollectedSeries:
            return storage[key]
        if type(key) is int:
            offset = key - self._start
            if 0 <= offset < len(storage):
                return storage[offset]
            raise KeyError(key)
        return self._dictionary()[key]

    def __setitem__(self, key, value):
        self._dictionary()[key] = value

    def __delitem__(self, key):
        del self._dictionary()[key]

    def __call__(self, key, /, *rest):
        return self[(key, *rest) if rest else key]

    def popitem(self):
        return self._dictionary().popitem()

    def values(self):
        return _NativeValuesView(self)

    def items(self):
        return _NativeItemsView(self)

    def copy(self):
        storage = self._storage
        if type(storage) is CollectedSeries:
            return storage.copy()
        return dict(zip(range(self._start, self._start + len(storage)), storage))

    def __copy__(self):
        storage = self._storage
        return NativeCollectedSeries(self._start, CollectedSeries(storage) if type(storage) is CollectedSeries else storage[:])

    def __deepcopy__(self, memo):
        result = NativeCollectedSeries(self._start, array('d'))
        memo[id(self)] = result
        result._storage = deepcopy(self._storage, memo)
        return result

    def __reduce__(self):
        return NativeCollectedSeries, (self._start, self._storage)


class _NativeValuesView(ValuesView):
    def __iter__(self):
        storage = self._mapping._storage
        return iter(storage.values()) if type(storage) is CollectedSeries else iter(storage)


class _NativeItemsView(ItemsView):
    def __iter__(self):
        series = self._mapping
        storage = series._storage
        if type(storage) is CollectedSeries:
            return iter(storage.items())
        return zip(range(series._start, series._start + len(storage)), storage)


def public_numeric_series(keys, values):
    start = keys[0] if keys else 0
    if all(type(key) is int and key == start + offset for offset, key in enumerate(keys)):
        return NativeCollectedSeries(start, values)
    return CollectedSeries(zip(keys, values))

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
    if kind in (dict, CollectedSeries, NativeCollectedSeries):
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
            if type(item) in (dict, CollectedSeries, NativeCollectedSeries):
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
    if type(value) in (dict, CollectedSeries, NativeCollectedSeries):
        return {name: _shape(item) for name, item in value.items()}
    return None


def _check_shape(expected, value, path):
    if expected is None and type(value) not in (dict, CollectedSeries, NativeCollectedSeries):
        return
    if expected is None or type(value) not in (dict, CollectedSeries, NativeCollectedSeries) or tuple(value) != tuple(expected):
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
    if type(old) in (dict, CollectedSeries, NativeCollectedSeries) or type(new) in (dict, CollectedSeries, NativeCollectedSeries):
        if type(old) not in (dict, CollectedSeries, NativeCollectedSeries) or type(new) not in (dict, CollectedSeries, NativeCollectedSeries) or tuple(old) != tuple(new):
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
