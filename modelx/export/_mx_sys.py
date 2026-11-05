import sys
import collections
import pickle
from importlib import import_module
import pathlib
from copy import deepcopy
from collections import OrderedDict
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


class CollectResult:
    """Detached values returned by an exported collect-enabled Space."""

    __slots__ = ('_fields',)

    def __init__(self, fields):
        self._fields = fields

    def __getattr__(self, name):
        try:
            return self._fields[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._fields))

    def __repr__(self):
        return 'CollectResult(%s)' % ', '.join(self._fields)


def _mx_collect_detach(value):
    if isinstance(value, BaseSpace):
        raise TypeError('collect output cannot retain a live Space')
    if type(value) is dict:
        return {k: _mx_collect_detach(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_mx_collect_detach(v) for v in value)
    return deepcopy(value)


def _mx_collect_sum(old, new, path):
    if type(new) is dict:
        if old is None:
            old = {name: None for name in new}
        elif type(old) is not dict or tuple(old) != tuple(new):
            raise ValueError('collect output structure changed at ' + repr(path))
        for name, value in new.items():
            old[name] = _mx_collect_sum(old[name], value, path + (name,))
        return old
    if type(old) is dict:
        raise ValueError('collect output structure changed at ' + repr(path))
    if old is None:
        return new
    try:
        return old + new
    except Exception as exc:
        raise TypeError('collect sum value cannot be added at ' + repr(path)) from exc


def _mx_collect_shape(value):
    if type(value) is dict:
        return {name: _mx_collect_shape(item) for name, item in value.items()}
    return None


def _mx_collect_check_shape(expected, value, path):
    if expected is None:
        if type(value) is dict:
            raise ValueError('collect output structure changed at ' + repr(path))
        return
    if type(value) is not dict or tuple(value) != tuple(expected):
        raise ValueError('collect output structure changed at ' + repr(path))
    for name, item in value.items():
        _mx_collect_check_shape(expected[name], item, path + (name,))


def collect_from_space(base, domain, loop_name, names, args):
    """Evaluate an exported collection without retaining policy ItemSpaces."""

    if len(names) != len(args) or loop_name in names:
        raise ValueError('invalid collect arguments')
    if isinstance(domain, (str, bytes, set, frozenset)) or iter(domain) is domain:
        raise TypeError('collect domain must be finite, ordered and re-iterable')
    coords = tuple(domain)
    if not coords:
        raise ValueError('collect domain must not be empty')
    if len(set(coords)) != len(coords):
        raise ValueError('collect domain contains duplicate coordinates')

    result = None
    modes = None
    individual_shape = None
    scratch = {}
    for coordinate in coords:
        values = dict(zip(names, args))
        values[loop_name] = coordinate
        root = base.__class__(base)
        releases = []
        complete = False
        try:
            for space, template in zip(root._mx_walk(), base._mx_walk()):
                space._mx_copy_refs(template, base)
                for previous_root in base._mx_roots:
                    previous_root._mx_copy_params(space)
                base._mx_assign_params(space, **values)
                space._mx_roots.extend(base._mx_roots)
                space._mx_roots.append(root)
                release = getattr(space, '_mx_collect_release', None)
                if release is not None:
                    releases.append((release, template))
                attach = getattr(space, '_mx_collect_attach', None)
                if attach is not None:
                    attach(scratch, template)
            outputs = root.collect_outputs()
            if type(outputs) is not dict or not outputs or set(outputs) - {'individual', 'sum'}:
                raise ValueError('collect_outputs must declare individual and/or sum')
            if modes is None:
                modes = tuple(outputs)
                result = {mode: None for mode in modes}
            elif tuple(outputs) != modes:
                raise ValueError('collect_outputs modes changed between policies')
            for mode, items in outputs.items():
                if type(items) is not dict or not items or any(type(name) is not str for name in items):
                    raise TypeError('collect_outputs sections need named values')
                items = _mx_collect_detach(items)
                if mode == 'individual':
                    shape = {name: _mx_collect_shape(item)
                             for name, item in items.items()}
                    if individual_shape is None:
                        individual_shape = shape
                        result[mode] = OrderedDict()
                    else:
                        _mx_collect_check_shape(individual_shape, items, (mode,))
                    result[mode][coordinate] = CollectResult(items)
                else:
                    result[mode] = _mx_collect_sum(result[mode], items, (mode,))
            complete = True
        finally:
            cleanup_error = None
            for release, template in releases:
                try:
                    release(scratch, template)
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
            if not complete or cleanup_error is not None:
                scratch.clear()
            if complete and cleanup_error is not None:
                raise cleanup_error
        del root

    return CollectResult({
        mode: value if mode == 'individual' else CollectResult(value)
        for mode, value in result.items()})


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

