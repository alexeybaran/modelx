"""The rules of ``collect()`` that live modelx and exported models share.

The exporter copies this module into a package that declares collect,
so it must not import modelx.
"""

from copy import deepcopy


def normalize_domain(value):
    """Return the collect domain as a tuple of its coordinates."""
    if (not hasattr(value, '__len__') or not hasattr(value, '__iter__')
            or isinstance(value, (str, bytes, set, frozenset, dict))
            or iter(value) is value):
        raise TypeError(
            "the collect domain must be a finite, ordered sequence, not %s"
            % type(value).__name__)
    domain = tuple(value)
    if not domain:
        raise ValueError("the collect domain is empty")
    try:
        unique = set(domain)
    except TypeError as exc:
        raise TypeError("the coordinates of a collect domain must be hashable") from exc
    if len(unique) != len(domain):
        raise ValueError("the collect domain has duplicate coordinates")
    return domain


class CollectedSeries(dict):
    """A dict output that can also be called: ``s(t)`` is ``s[t]``."""

    __slots__ = ()

    def __call__(self, key, *rest):
        return self[(key,) + rest if rest else key]


class CollectResult:
    """The value of ``collect()``.

    ``result.individual`` is a dict from each coordinate
    to a CollectResult of that item's ``individual`` outputs;
    ``result.sum`` is a CollectResult of the ``sum`` outputs added over
    the domain. A named output that is a dict is a :class:`CollectedSeries`.
    """

    __slots__ = ('_data', '_mx_native')

    def __init__(self, fields):
        self._data = {
            name: CollectedSeries(value) if type(value) is dict else value
            for name, value in fields.items()}
        self._mx_native = None

    @classmethod
    def _from_native(cls, native):
        result = cls.__new__(cls)
        result._data = {}
        result._mx_native = native
        return result

    @property
    def _fields(self):
        native = self._mx_native
        if native is not None:
            for name in native.fields:
                getattr(self, name)
        return self._data

    def __getattr__(self, name):
        # object.__getattribute__: a bare instance, as copy and pickle
        # create it, has no _fields yet and must not recurse here.
        try:
            data = object.__getattribute__(self, '_data')
            return data[name]
        except KeyError:
            native = object.__getattribute__(self, '_mx_native')
            if native is None or name not in native.fields:
                raise AttributeError(name) from None
            value = native.materialize(name)
            if type(value) is dict:
                value = CollectedSeries(value)
            return data.setdefault(name, value)

    def __reduce__(self):
        return type(self), (self._fields,)

    def __dir__(self):
        return sorted(set(object.__dir__(self)) | set(self._fields))

    def __repr__(self):
        return 'CollectResult(%s)' % ', '.join(self._fields)


_SCALARS = (float, int, bool, complex, str, bytes, type(None))
_MODES = ('individual', 'sum')


def _scalars(items):
    for item in items:
        if type(item) not in _SCALARS:
            return False
    return True


def _check_data(value, live_types, path):
    remaining = [value]
    seen = set()
    while remaining:
        item = remaining.pop()
        if isinstance(item, live_types):
            raise TypeError(
                "collect_outputs must return data, not modelx objects, at %s"
                % (path,))
        if id(item) not in seen and type(item) in (dict, list, tuple, set, frozenset):
            seen.add(id(item))
            remaining.extend(item)
            if type(item) is dict:
                remaining.extend(item.values())


def prepare_output(value, live_types):
    """Check one item's ``collect_outputs`` value and detach it from the item."""
    if type(value) is not dict or not value or value.keys() - set(_MODES):
        raise TypeError(
            "collect_outputs must return a dict with an 'individual' "
            "section, a 'sum' section or both")
    result = {}
    for mode in _MODES:
        if mode not in value:
            continue
        outputs = value[mode]
        if type(outputs) is not dict or not outputs or not all(
                isinstance(name, str) for name in outputs):
            raise TypeError(
                "collect_outputs[%r] must be a non-empty dict "
                "with str keys" % mode)
        detached = {}
        for name, item in outputs.items():
            if type(item) in _SCALARS:
                pass
            elif type(item) is dict and _scalars(item) and _scalars(item.values()):
                item = dict(item)       # equal to its deep copy
            else:
                _check_data(item, live_types, (mode, name))
                item = deepcopy(item)
            detached[name] = item
        result[mode] = detached
    return result


def _check_sum(value, path):
    if type(value) is dict:
        for name, item in value.items():
            _check_sum(item, path + (name,))
    elif isinstance(value, (str, bytes, list, tuple)):
        raise TypeError(
            "a collect_outputs 'sum' value must be a number or an array, "
            "not %s, at %s" % (type(value).__name__, path))


def _add(total, value, path):
    if type(total) is dict or type(value) is dict:
        if type(total) is not dict or type(value) is not dict or total.keys() != value.keys():
            raise ValueError(
                "collect_outputs 'sum' has a different structure at %s" % (path,))
        for name in total:
            total[name] = _add(total[name], value[name], path + (name,))
        return total
    return total + value


class Accumulator:
    """Folds the prepared outputs of the items in domain order.

    A ``sum`` value is ``total = total + value``, item after item, starting
    from the first item's value: what the same loop written by hand gives.
    """

    def __init__(self):
        self._modes = None
        self._individual = {}
        self._sum = None

    def add(self, coordinate, outputs):
        if self._modes is None:
            self._modes = tuple(outputs)
            if 'sum' in outputs:
                _check_sum(outputs['sum'], ('sum',))
        elif tuple(outputs) != self._modes:
            raise ValueError(
                "collect_outputs returned other sections for %r" % (coordinate,))
        if 'individual' in outputs:
            self._individual[coordinate] = CollectResult(outputs['individual'])
        if 'sum' in outputs:
            if self._sum is None:
                self._sum = outputs['sum']
            else:
                self._sum = _add(self._sum, outputs['sum'], ('sum',))

    def result(self):
        fields = {}
        if 'individual' in self._modes:
            fields['individual'] = self._individual
        if 'sum' in self._modes:
            fields['sum'] = CollectResult(self._sum)
        return CollectResult(fields)
