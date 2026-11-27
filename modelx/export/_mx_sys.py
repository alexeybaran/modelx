import sys
import collections
import pickle
from importlib import import_module
import pathlib
from types import BuiltinMethodType, MethodType
from typing import Dict
try:
    has_io = True
    from . import _mx_io
except ImportError:
    has_io = False


class BaseMxObject:

    __slots__ = ()


class BaseParent(BaseMxObject):

    # Annotations only: they create no class attributes and so never
    # conflict with the __slots__ of a subclass.
    __slots__ = ()

    _mx_spaces: Dict[str, 'BaseSpace']
    _parent: 'BaseParent'
    _model: 'BaseModel'
    _name: str

    def _mx_walk(self, skip_self: bool = False):
        """Generator yielding spaces in breadth-first order"""
        if skip_self:
            que = collections.deque(self._mx_spaces.values())
        else:
            que = collections.deque([self])
        while que:
            parent = que.popleft()
            yield parent
            for child in parent._mx_spaces.values():
                que.append(child)

    def _mx_assign_refs(self, io_data, pickle_data):
        raise NotImplementedError


class BaseModel(BaseParent):

    path = pathlib.Path(__file__).parent

    def _mx_load_io(self):

        ios = {}
        io_data = {}
        if has_io:
            for k, v in _mx_io.ios.items():
                cls = io_types[v['type']]
                load_from = self.path / k
                ios[k] = cls(k, load_from, v)

            for k, v in _mx_io.iospecs.items():
                cls = iospec_types[v['type']]
                io_data[k] = cls(ios[v['io']], v['kwargs']).load_value()

        p = self.path / '_mx_pickled'
        if p.exists():
            with open(p, mode='rb') as f:
                pickle_data = pickle.load(f)
        else:
            pickle_data = {}

        for m_or_s in self._mx_walk():
            m_or_s._mx_assign_refs(io_data, pickle_data)


class BaseSpace(BaseParent):

    # Subclasses generated with use_slots=False declare no __slots__ of
    # their own and so still get a __dict__, exactly as before.
    __slots__ = ()

    # Instance variables
    _mx_is_cells_set: bool
    _mx_cells: dict

    @property
    def _cells(self):
        if not self._mx_is_cells_set:
            for name in getattr(sys.modules[self.__class__.__module__],
                                "_v_cells_names_" + self._name):
                self._mx_cells[name] = getattr(self, name)
            self._mx_is_cells_set = True

        return self._mx_cells


    def _mx_is_in(self, parent: BaseParent):
        p = self
        while True:
            if p is parent:
                return True
            elif p is None:
                return False
            else:
                p = p._parent


    def _mx_get_object(self, keys):
        obj = self
        for name in keys:
            if name[0] == ".":
                for _ in name[1:]:
                    obj = obj._parent
            else:
                obj = getattr(obj, name)

        return obj


# The versions of the collect hook this runtime calls. See the page
# "collect hook" of the modelx reference: a compiled Space class takes
# over collect() by defining _mx_collect_hook and _mx_collect_hook_version.
COLLECT_HOOK_VERSIONS = (1,)

# The cache key, for the same arguments, of the domain tuple a result was last read with
_SAME_DOMAIN = object()


def release_collect_item(item):
    """Break the links that keep a private ItemSpace tree alive."""
    pending = [item]
    owned = set()
    candidates = []
    while pending:
        space: BaseSpace = pending.pop()
        owned.add(id(space))
        names = getattr(sys.modules[type(space).__module__], '_v_collect_refs_' + space._name, ())
        if names:
            candidates.append((space, names))
        pending.extend(space._mx_spaces.values())
        space._mx_spaces.clear()
        try:
            nested = space._mx_itemspaces
        except AttributeError:
            nested = None
        if nested is not None:
            pending.extend(nested.values())
            nested.clear()
        space._mx_cells.clear()
        space._mx_roots.clear()
        space._space = None
        space._parent = None

    for space, names in candidates:
        for name, is_cell in names:
            value = getattr(space, name, None)
            if is_cell:
                if not isinstance(value, (MethodType, BuiltinMethodType)):
                    continue
                value = value.__self__
            if id(value) in owned:
                setattr(space, name, None)


def collect_from_space(space, domain, item, args, cache):
    """Evaluate ``collect()`` of an exported Space.

    ``domain`` is the value of the domain, ``item(c)`` a private ItemSpace
    at coordinate ``c``, ``args`` the other Space parameters in signature
    order, and ``cache`` the Space's dict of results.

    A result is read again without normalizing the domain when the arguments
    are equal and the domain is the very tuple object it was last read with:
    a tuple cannot change, so its coordinates are those of the result.
    """
    from ._mx_collect import Accumulator, CollectResult, normalize_domain, prepare_output

    same = (_SAME_DOMAIN, args)
    if type(domain) is tuple:
        held = cache.get(same)
        if held is not None and held[0] is domain:
            return held[1]
    coords = normalize_domain(domain)
    key = (coords, args)
    result = cache.get(key)
    if result is not None:
        if type(domain) is tuple:
            cache[same] = (domain, result)
        return result

    cls = type(space)
    hook = getattr(cls, '_mx_collect_hook', None)
    if hook is not None:
        version = getattr(cls, '_mx_collect_hook_version', None)
        if type(version) is not int or version not in COLLECT_HOOK_VERSIONS:
            raise RuntimeError(
                "%s.%s defines _mx_collect_hook version %r; this exported "
                "model calls version %s" % (
                    cls.__module__, cls.__qualname__, version,
                    ', '.join(map(str, COLLECT_HOOK_VERSIONS))))
        result = hook(space, coords, args)
        if result is not None and type(result) is not CollectResult:
            raise TypeError(
                "_mx_collect_hook must return a CollectResult of _mx_collect "
                "or None, not %s" % type(result).__name__)

    if result is None and hook is None:
        workers = getattr(space._model, '_mx_collect_workers', 1)
        if type(workers) is not int or workers < 1:
            raise ValueError('_mx_collect_workers must be a positive int, not %r' % (workers,))
        if workers > 1 and not getattr(sys, '_is_gil_enabled', lambda: True)():
            from ._mx_collect import _collect_once
            result = _collect_once(space, coords, item, cache, key, workers)

    if result is None:
        accumulator = Accumulator()
        for coordinate in coords:
            root = item(coordinate)
            try:
                outputs = prepare_output(root.collect_outputs(), (BaseMxObject,))
            finally:
                release_collect_item(root)
                del root
            accumulator.add(coordinate, outputs)
        result = accumulator.result()
    result = cache.setdefault(key, result)
    if type(domain) is tuple:
        cache[same] = (domain, result)
    return result


class MiniBaseSharedIO:

    def __init__(self, path, load_from, kwargs):
        self.path = path
        self.load_from = load_from
        self.kwargs = kwargs

    def __getattr__(self, item):
        return self.kwargs[item]

    def load_io(self, root):
        raise NotImplementedError


class MiniBaseIOSpec:

    def __init__(self, io, kwargs):
        self._io = io
        self.kwargs = kwargs

    def load_value(self):
        raise NotImplementedError


class MiniPandasIO(MiniBaseSharedIO):

    def load_io(self, root):
        pass


class MiniPandasData(MiniBaseIOSpec):

    def load_value(self):
        self._on_unserialize(self.kwargs)
        return self._value

    def _on_unserialize(self, state):
        if self._io.file_type is None:
            self._io.file_type = state["filetype"]
        self._read_args = state["read_args"]
        if "squeeze" in state:
            self._squeeze = state["squeeze"]
        elif "squeeze" in self._read_args:
            self._squeeze = self._read_args.pop("squeeze")
        else:
            self._squeeze = False
        self.name = state["name"]
        self._sheet = state["sheet"] if "sheet" in state else None
        self._read_pandas()

    def _read_pandas(self):
        import pandas as pd

        if self._io.file_type == "excel":
            self._value = pd.read_excel(
                self._io.load_from, **self._read_args)
        elif self._io.file_type == "csv":
            self._value = pd.read_csv(
                self._io.load_from, **self._read_args)
        else:
            raise ValueError

        if self._squeeze:
            self._value = self._value.squeeze("columns")

        if isinstance(self._value, pd.Series):
            self._value.name = self.name

        if hasattr(self, "_is_hidden") and self._is_hidden:
            self._value._mx_dataclient = self


io_types = {
    'PandasIO': MiniPandasIO
}


iospec_types = {
    'PandasData': MiniPandasData
}

