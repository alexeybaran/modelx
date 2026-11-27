"""The rules of ``collect()`` that live modelx and exported models share.

The exporter copies this module into a package that declares collect,
so it must not import modelx.
"""

from copy import deepcopy
from threading import Lock, RLock, local


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

    __slots__ = ('__storage',)

    def __init__(self, fields):
        self.__storage = {
            name: CollectedSeries(value) if type(value) is dict else value
            for name, value in fields.items()}

    @property
    def _fields(self):
        storage = self.__storage
        if type(storage) is _NativeFields:
            for name in storage.native.fields:
                storage[name]
            return storage.views
        return storage

    def __getattr__(self, name):
        # object.__getattribute__: a bare instance, as copy and pickle
        # create it, has no _fields yet and must not recurse here.
        try:
            return object.__getattribute__(self, '_CollectResult__storage')[name]
        except KeyError:
            raise AttributeError(name) from None

    def __reduce__(self):
        return type(self), (self._fields,)

    def __dir__(self):
        return sorted(set(object.__dir__(self)) | set(self._fields))

    def __repr__(self):
        return 'CollectResult(%s)' % ', '.join(self._fields)


class _NativeFields:
    __slots__ = ('native', 'views', 'rows')

    def __init__(self, native, individual):
        self.native = native
        self.views = {}
        self.rows = {} if individual else None

    def __getitem__(self, name):
        views = self.views
        if name in views:
            return views[name]
        if name not in self.native.fields:
            raise KeyError(name)
        value = self.native.materialize(name, self)
        if type(value) is dict:
            value = CollectedSeries(value)
        return views.setdefault(name, value)

def _native_result(native, individual=False):
    result = CollectResult.__new__(CollectResult)
    result._CollectResult__storage = _NativeFields(native, individual)
    return result


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


# An in-flight call is never stored in the result cache. The same cache
# may be read without taking this lock once a result is published.
_PENDING = {}
_PENDING_LOCK = Lock()
_ACTIVE = local()
_DEPENDENCIES = {}


def _supports_parallel_key(key):
    # The registry mutex must never invoke application __hash__ or __eq__.
    values = [key]
    while values:
        value = values.pop()
        kind = type(value)
        if kind is tuple or kind is frozenset:
            values.extend(value)
        elif not (kind is int or kind is str or kind is float or kind is bytes
                  or kind is bool or kind is complex or kind is type(None)):
            return False
    return True


def _depends_on(flight, other):
    pending = [flight]
    seen = set()
    while pending:
        dependency = pending.pop()
        if dependency is other:
            return True
        if dependency not in seen:
            seen.add(dependency)
            pending.extend(_DEPENDENCIES.get(dependency, ()))
    return False


def _collect_parallel(coords, evaluate, accumulator, workers):
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    from itertools import islice

    # Keep at most one submitted item per worker. Accumulate on this
    # thread, including when addition calls Python code.
    remaining = iter(coords)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(evaluate, c): c for c in islice(remaining, workers)}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for task in done:
                accumulator.add(pending.pop(task), task.result())
            for c in islice(remaining, len(done)):
                pending[pool.submit(evaluate, c)] = c


def _collect_once(space, coords, item, cache, key, workers):
    from concurrent.futures import Future

    if not _supports_parallel_key(key):
        return None

    model_lock = getattr(space._model, '_mx_lock', None)
    owns_lock = model_lock is not None and getattr(model_lock, '_is_owned', lambda: True)()
    caller = getattr(_ACTIVE, 'flight', None)
    pending_key = (id(cache), key)
    blocked = None
    with _PENDING_LOCK:
        flight = _PENDING.get(pending_key)
        owner = flight is None
        if owner:
            flight = _PENDING[pending_key] = Future()
        elif owns_lock:
            blocked = 'A collect() call cannot wait for an active result while holding the model lock'
        elif caller is not None and _depends_on(flight, caller):
            blocked = 'Circular dependency between collect() calls'
        if caller is not None and blocked is None:
            targets = _DEPENDENCIES.setdefault(caller, {})
            targets[flight] = targets.get(flight, 0) + 1

    if blocked is not None:
        result = cache.get(key)
        if result is not None:
            return result
        raise RuntimeError(blocked)

    try:
        if not owner:
            return flight.result()
        _ACTIVE.flight = flight
        parallel = caller is None and not owns_lock and len(coords) > 1
        try:
            # A result can be published between the initial cache miss and
            # claiming the key. Look it up again without the registry mutex:
            # the result cache can contain custom keys from the serial path.
            result = cache.get(key)
            if result is None:
                result = cache.setdefault(
                    key, _evaluate_collect(space, coords, item, workers, model_lock, parallel, flight))
        except BaseException as exc:
            flight.set_exception(exc)
            raise
        flight.set_result(result)
        return result
    finally:
        if owner:
            _ACTIVE.flight = caller
        with _PENDING_LOCK:
            if owner:
                del _PENDING[pending_key]
            if caller is not None:
                targets[flight] -= 1
                if not targets[flight]:
                    del targets[flight]
                if not targets:
                    del _DEPENDENCIES[caller]


def _evaluate_collect(space, coords, item, workers, model_lock, parallel, flight):
    from ._mx_sys import BaseMxObject, release_collect_item

    guard_factory = model_lock is not None and (
        getattr(space, '_mx_lock', None) is model_lock
        or hasattr(type(space), '_mx_apply_formula'))

    def outputs_at(coordinate):
        if guard_factory:
            with model_lock:
                root = item(coordinate)
        else:
            root = item(coordinate)
        try:
            if parallel and model_lock is not None:
                private_lock = None
                # References can still point to public Spaces. Only members
                # of the newly built tree get a different lock.
                for child in root._mx_walk():
                    if getattr(child, '_mx_lock', None) is model_lock:
                        if private_lock is None:
                            private_lock = RLock()
                        child._mx_lock = private_lock
            return prepare_output(root.collect_outputs(), (BaseMxObject,))
        finally:
            release_collect_item(root)
            del root

    def evaluate(coordinate):
        _ACTIVE.flight = flight
        try:
            return outputs_at(coordinate)
        finally:
            _ACTIVE.flight = None

    accumulator = Accumulator()
    if parallel:
        _collect_parallel(coords, evaluate, accumulator, min(workers, len(coords)))
    else:
        for coordinate in coords:
            accumulator.add(coordinate, outputs_at(coordinate))
    return accumulator.result()
