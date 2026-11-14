"""Owned immutable records and keyed tables for model inputs.

Construction is explicit: existing dictionaries and pandas objects are never
silently frozen. Keys and authored enum objects retain their Python identity.
The explicit preserve_scalar_types option owns structure while retaining raw
cell references, including mutable object cells; it is a shallow snapshot.
"""
from bisect import bisect_right
from collections.abc import Mapping
from types import MappingProxyType


_MISSING = object()
_RECTANGULAR_STORAGE = object()
_CONTIGUOUS_STORAGE = object()


class Record(Mapping):
    __modelx_cython_type__ = (1, 'modelx_cython.runtime._native_inputs',
                             'owned-table-1', 'modelx_helpers.immutable_inputs.Record')
    __slots__ = ('_items', '_index', '_native_groups', '_metadata', '_preserve_scalar_types')

    def __init__(self, values, *, preserve_scalar_types=False):
        items = (tuple(values.items()) if preserve_scalar_types else
                 tuple((key, freeze(value)) for key, value in values.items()))
        object.__setattr__(self, '_items', items)
        object.__setattr__(self, '_index', MappingProxyType(dict(items)))
        object.__setattr__(self, '_native_groups', {})
        object.__setattr__(self, '_metadata', None)
        object.__setattr__(self, '_preserve_scalar_types', bool(preserve_scalar_types))

    def __setattr__(self, name, value):
        raise AttributeError('immutable input')

    def __delattr__(self, name):
        raise AttributeError('immutable input')

    def __getitem__(self, key):
        return self._index[key]

    def lookup_enum(self, key, enum_type, *, default=_MISSING):
        """Resolve a stored label to its enum member with an optional default."""
        try:
            return enum_type[self[key]]
        except (KeyError, TypeError):
            if default is _MISSING:
                raise
            return default

    def slice(self, key, *keys):
        """Follow group keys, optionally ending at a prepared row."""
        value = self[key]
        return value.slice(*keys) if keys else value

    def __iter__(self):
        return iter(self._index)

    def __len__(self):
        return len(self._index)

    @property
    def index(self):
        """The field names as a pandas Index (a pandas row's ``.index``), built on each read."""
        import pandas as pd
        return pd.Index(tuple(self._index))

    def __reduce__(self):
        if getattr(self, '_preserve_scalar_types', False):
            return _restore_record, (type(self), dict(self._items), self._metadata, True)
        if self._metadata is None:
            return type(self), (dict(self._items),)
        return _restore_record, (type(self), dict(self._items), self._metadata)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class _TableIndex:
    """Read-only view of the table's complete row keys in input order."""

    __slots__ = ('_table',)

    def __init__(self, table):
        self._table = table

    def __contains__(self, key):
        return self._table.contains(key)

    def __iter__(self):
        return iter(self._table._keys)

    def __len__(self):
        return len(self._table._keys)

    def max_int(self) -> int:
        return int(max(self._table._keys))

    def max_float(self) -> float:
        return float(max(self._table._keys))


class Table(Mapping):
    __modelx_cython_type__ = (1, 'modelx_cython.runtime._native_inputs',
                             'owned-table-1', 'modelx_helpers.immutable_inputs.Table')
    __slots__ = ('_keys', '_values', '_columns', '_positions', '_ordered', '_native_cache', '_index_error', '_metadata', '_preserve_scalar_types', '_storage_form')

    def __init__(self, keys, values, columns, index_error=None, *, preserve_scalar_types=False):
        keys, columns = tuple(keys), tuple(columns)
        values = tuple(tuple(value if preserve_scalar_types else freeze(value)
                             for value in row) for row in values)
        Table._initialize(self, keys, values, columns, index_error, preserve_scalar_types)

    @classmethod
    def _from_float64(cls, keys, values, columns, index_error, *, exact_keys):
        """Own ndarray.tolist() rows, whose values are already immutable floats."""
        table = cls.__new__(cls)
        Table._initialize(table, tuple(keys), tuple(map(tuple, values)), tuple(columns),
                          index_error, False, exact_keys=exact_keys)
        return table

    def _initialize(self, keys, values, columns, index_error, preserve_scalar_types,
                    *, exact_keys=False):
        if not columns or len(set(columns)) != len(columns):
            raise ValueError('table columns must be nonempty and unique')
        if any(type(column) is not str for column in columns):
            raise TypeError('table columns must be strings')
        if len(keys) != len(values) or any(len(row) != len(columns) for row in values):
            raise ValueError('table dimensions do not match')
        position_keys = (keys if index_error is None or exact_keys
                         else tuple(_index_key(key, index_error) for key in keys))
        positions = dict(zip(position_keys, range(len(keys))))
        if len(positions) != len(keys):
            raise ValueError('duplicate table key')
        object.__setattr__(self, '_keys', keys)
        object.__setattr__(self, '_values', values)
        object.__setattr__(self, '_columns', columns)
        object.__setattr__(self, '_positions', MappingProxyType(positions))
        try:
            ordered = tuple(sorted(keys))
        except TypeError:
            ordered = None
        object.__setattr__(self, '_ordered', ordered)
        object.__setattr__(self, '_native_cache', {})
        object.__setattr__(self, '_index_error', freeze(index_error))
        object.__setattr__(self, '_metadata', None)
        object.__setattr__(self, '_preserve_scalar_types', bool(preserve_scalar_types))

    @classmethod
    def _from_rectangular(cls, axes, data, columns, index_error, metadata):
        table = cls.__new__(cls)
        positions = tuple(MappingProxyType(dict(zip(labels, range(len(labels))))) for labels in axes)
        object.__setattr__(table, '_columns', columns)
        object.__setattr__(table, '_native_cache', {_RECTANGULAR_STORAGE: (axes, data, positions)})
        object.__setattr__(table, '_index_error', index_error)
        object.__setattr__(table, '_metadata', metadata)
        object.__setattr__(table, '_preserve_scalar_types', False)
        return table

    @classmethod
    def _from_contiguous(cls, keys, data, columns, index_error, metadata, preserve_scalar_types):
        table = cls.__new__(cls)
        object.__setattr__(table, '_keys', keys)
        object.__setattr__(table, '_ordered', keys)
        object.__setattr__(table, '_columns', columns)
        object.__setattr__(table, '_native_cache', {_CONTIGUOUS_STORAGE: (keys[0], data)})
        object.__setattr__(table, '_index_error', index_error)
        object.__setattr__(table, '_metadata', metadata)
        object.__setattr__(table, '_preserve_scalar_types', preserve_scalar_types)
        return table

    def __getattr__(self, name):
        """The one attribute fallback of a Table.

        1. Rectangular and contiguous tables own their axes and values without
           allocating the expanded Python representation (``_keys``, ``_values``,
           ``_positions``, ``_ordered``) until legacy code asks for it.
        2. Any other private name raises AttributeError.
        3. A public name outside the table vocabulary (``iloc``, ``columns``,
           ``shape``...) is read from a pandas DataFrame materialised once from the
           table and handed out as a copy-on-write copy, so the table cannot change.
           Materialising expands compact storage for good; a ``hasattr`` probe of a
           public name does the same. Read the vocabulary instead.
        """
        if name not in ('_keys', '_values', '_positions', '_ordered'):
            if name.startswith('_'):
                raise AttributeError(name)
            return getattr(_pandas_frame(self).copy(deep=False), name)
        from itertools import product

        cache = object.__getattribute__(self, '_native_cache')
        storage = cache.get(_RECTANGULAR_STORAGE)
        if storage is None:
            contiguous = cache.get(_CONTIGUOUS_STORAGE)
            if contiguous is None:
                raise AttributeError(name)
            _, data = contiguous
            if name == '_values':
                result = (tuple(zip(*data.T)) if self._preserve_scalar_types else
                          tuple(map(tuple, data.tolist())))
            elif name == '_positions':
                result = MappingProxyType(dict(zip(self._keys, range(len(data)))))
            else:
                raise AttributeError(name)
            object.__setattr__(self, name, result)
            return result
        axes, data, _ = storage
        if name == '_keys':
            result = tuple(product(*axes))
        elif name == '_values':
            result = tuple((value,) for value in data.tolist())
        elif name == '_positions':
            result = MappingProxyType(dict(zip(self._keys, range(len(data)))))
        else:
            result = tuple(sorted(self._keys))
        object.__setattr__(self, name, result)
        return result

    def __setattr__(self, name, value):
        raise AttributeError('immutable input')

    def __delattr__(self, name):
        raise AttributeError('immutable input')

    def __getitem__(self, key):
        if key == 'keys':
            return self._keys
        if key == 'values':
            return self._values
        if key == 'columns':
            return self._columns
        if type(key) is str and key in self._columns:
            return _TableColumn(self, key)       # pandas spelling: frame["col"]
        raise KeyError(key)

    def __iter__(self):
        return iter(('keys', 'values', 'columns'))

    def __len__(self):
        return 3

    def lookup(self, key, column):
        if (isinstance(column, str) and self._index_error is not None
                and self._index_error._index.get('pandas_columns', False)):
            name = column
            try:
                column = self._columns.index(name)
            except ValueError:
                raise KeyError(name) from None
        rectangular = self._native_cache.get(_RECTANGULAR_STORAGE)
        contiguous = self._native_cache.get(_CONTIGUOUS_STORAGE)
        if rectangular is not None or contiguous is not None:
            try:
                position = self._compact_position(key, rectangular, contiguous)
            except KeyError:
                if self._index_error is None:
                    raise
                raise KeyError(key) from None
        elif self._index_error is None:
            position = self._positions[key]
        else:
            if not _valid_index_key(key, self._index_error):
                raise KeyError(key)
            try:
                position = self._positions[_index_key(key, self._index_error)]
            except KeyError:
                raise KeyError(key) from None
        column = self._columns.index(column) if isinstance(column, str) else column
        if rectangular is not None:
            return (float(rectangular[1][position]),)[column]
        if contiguous is not None:
            values = contiguous[1][position]
            return (tuple(values) if self._preserve_scalar_types else tuple(values.tolist()))[column]
        return self._values[position][column]

    def _compact_position(self, key, rectangular, contiguous):
        spec = self._index_error
        if spec is not None:
            if not _valid_index_key(key, spec):
                raise KeyError(key)
            key = _index_key(key, spec)
        if rectangular is not None:
            if type(key) is not tuple:
                return self._positions[key]
            if any(part is _INDEX_NAN for part in key):
                raise KeyError(key)
            if not all(type(part) in (str, int, float) or type(part).__module__ == 'numpy' for part in key):
                return self._positions[key]
            position = 0
            try:
                for label, positions in zip(key, rectangular[2]):
                    position = position * len(positions) + positions[label]
            except KeyError:
                raise KeyError(key) from None
            return position
        start, data = contiguous
        if type(key) is int or type(key) is bool:
            position = key - start
        elif type(key) is float and key.is_integer():
            position = int(key) - start
        elif type(key) in (str, bytes, float) or key is None:
            raise KeyError(key)
        else:
            if type(key).__module__ == 'numpy':
                from numpy import integer, floating
            else:
                return self._positions[key]
            if isinstance(key, (integer, floating)):
                try:
                    number = int(key)
                except (ValueError, OverflowError):
                    raise KeyError(key) from None
                if key != number or hash(key) != hash(number):
                    raise KeyError(key)
                position = number - start
            else:
                return self._positions[key]
        if 0 <= position < len(data):
            return position
        raise KeyError(key)

    def lookup_float(self, key, column, default=_MISSING) -> float:
        """Read a table value using the typed float lookup contract."""
        from .typed_inputs import lookup_float

        return lookup_float(self, key, column, default)

    def lookup_int(self, key, column, default=_MISSING) -> int:
        """Read a table value using the typed integer lookup contract."""
        from .typed_inputs import lookup_int

        return lookup_int(self, key, column, default)

    def interpolate_float(self, key, column) -> float:
        """Interpolate adjacent integer rows and hold the final row above its key."""
        from math import floor

        lower = int(floor(key))
        last = self.index.max_int()
        if lower >= last:
            return self.lookup_float(last, column)
        start = self.lookup_float(lower, column)
        end = self.lookup_float(lower + 1, column)
        return start + (end - start) * (key - lower)

    def slice(self, key) -> Record:
        """Select a row with the pandas row metadata captured at preparation."""
        from .table_operations import row

        return row(self, key)

    row = slice

    def previous(self, key, column):
        if self._ordered is None:
            raise TypeError('predecessor lookup requires ordered keys')
        index = bisect_right(self._ordered, key) - 1
        if index < 0:
            raise ValueError(f'No key at or before {key!r}')
        return self.lookup(self._ordered[index], column)

    def contains(self, key):
        rectangular = self._native_cache.get(_RECTANGULAR_STORAGE)
        contiguous = self._native_cache.get(_CONTIGUOUS_STORAGE)
        if rectangular is not None or contiguous is not None:
            try:
                self._compact_position(key, rectangular, contiguous)
            except KeyError:
                return False
            return True
        if self._index_error is None:
            return key in self._positions
        return (_valid_index_key(key, self._index_error)
                and _index_key(key, self._index_error) in self._positions)

    @property
    def index(self):
        """The table's complete row keys in their original order."""
        return _TableIndex(self)

    @property
    def loc(self):
        """Pandas-spelled label reads; see :class:`_TableLoc`."""
        return _TableLoc(self)

    @property
    def at(self):
        """Scalar label reads with pandas ``.at`` spelling: ``table.at[key, "col"]``."""
        return _TableAt(self)

    def iterrows(self):
        """``(label, row)`` pairs in row order, like ``DataFrame.iterrows()`` without pandas.

        Each row is the Record ``loc[label]`` returns: with ``rows=True`` the pandas-coerced
        row, otherwise the stored values (no dtype coercion across columns). The pairs are
        built once per table from its storage and shared; compact storage is read, not
        expanded into the table's ``_keys``/``_values``.
        """
        rows = self._native_cache.get('iterrows')
        if rows is None:
            metadata = self._metadata
            if metadata is not None and metadata[0] == 'frame' and metadata[4] is not None:
                rows = tuple((key, self.slice(key)) for key in _iter_keys(self))
            else:
                rows = tuple((key, _row_record(self, position))
                             for position, key in enumerate(_iter_keys(self)))
            rows = self._native_cache.setdefault('iterrows', rows)
        return iter(rows)

    def __reduce__(self):
        args = (self._keys, self._values, self._columns)
        if self._index_error is not None:
            args += (self._index_error,)
        if getattr(self, '_preserve_scalar_types', False):
            return _restore_table, (type(self), args, self._metadata, True)
        if self._metadata is None:
            return type(self), args
        return _restore_table, (type(self), args, self._metadata)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def _restore_record(cls, values, metadata, preserve_scalar_types=False):
    result = cls(values, preserve_scalar_types=preserve_scalar_types) if preserve_scalar_types else cls(values)
    object.__setattr__(result, '_metadata', metadata)
    return result


def _restore_table(cls, args, metadata, preserve_scalar_types=False):
    result = cls(*args, preserve_scalar_types=preserve_scalar_types) if preserve_scalar_types else cls(*args)
    object.__setattr__(result, '_metadata', metadata)
    return result


def freeze(value):
    kind = type(value)
    if kind is int or kind is float or kind is str or kind is bool or kind is bytes or value is None:
        return value
    if isinstance(value, (Record, Table)):
        return value
    if kind is dict:
        if value.keys() == {'keys', 'values', 'columns'}:
            return Table(value['keys'], value['values'], value['columns'])
        return Record(value)
    if kind is tuple or kind is list:
        return tuple(freeze(item) for item in value)
    if isinstance(value, (bool, int, float, str, bytes)):
        return value
    if kind.__module__.startswith('numpy'):
        import numpy as np
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value
    # These pandas missing scalars are immutable, pickleable singletons. Keep
    # their identity: replacing NA/NaT with NaN changes conversion exceptions.
    if kind.__module__.startswith('pandas.'):
        from pandas import NA, NaT
        if value is NA or value is NaT:
            return value
    raise TypeError(f'unsupported immutable input value: {kind.__name__}')


def _copy_container(value):
    """Copy a pandas object cell's containers while retaining its scalar values."""
    kind = type(value)
    if kind is dict:
        return {key: _copy_container(item) for key, item in value.items()}
    if kind is list:
        return [_copy_container(item) for item in value]
    if kind is tuple:
        return tuple(_copy_container(item) for item in value)
    return value


_INDEX_NAN = object()

def _missing_index_part(value):
    if value is None or isinstance(value, float) and value != value:
        return True
    if type(value).__module__.startswith('pandas.'):
        from pandas import NA, NaT
        return value is NA or value is NaT
    return False


def _index_key(key, spec):
    if spec is None:
        return key
    if spec['multi'] and type(key) is tuple:
        return tuple(_INDEX_NAN if _missing_index_part(part) else part for part in key)
    if type(key) is float and key != key:
        return _INDEX_NAN
    return key


def _valid_index_key(key, spec):
    parts = key if spec['multi'] and type(key) is tuple else (key,)
    if len(parts) != len(spec['boolean_modes']):
        return False
    # 0: mixed object labels; 1: boolean labels; 2: non-boolean labels.
    for part, mode in zip(parts, spec['boolean_modes']):
        if _missing_index_part(part):
            continue
        boolean = type(part) is bool
        if not boolean and type(part).__module__.startswith('numpy'):
            import numpy as np
            boolean = type(part) is np.bool_
        if mode == 1 and not boolean:
            return False
        if mode == 2 and boolean:
            return False
    return True


# ---------------------------------------------------------------------------
# Target-syntax read vocabulary (round 3, W1.H): loc, at, T["c"], iterrows,
# index.min/max/asof/searchsorted/[i], ``k in T.index``. Every read below goes
# through _storage_of/_find_row, which read contiguous and rectangular storage
# directly: they never set the lazy ``_keys``/``_values``/``_positions``/
# ``_ordered`` of a compact table. The legacy methods above are unchanged.

def _storage_of(table):
    """The table's storage form, fixed at construction and kept in the ``_storage_form`` slot:
    ``('contiguous', start, data, plain, levels)``,
    ``('rectangular', axes, data, positions, plain, levels)`` or ``('rows', plain, levels)``.
    ``plain``: an int or str key needs no pandas label-mode check (no bool-only level).
    ``_native_cache`` is left to the compiler (D51). The slot is filled on first use; threads
    reading one shared table may both compute it, but the form follows from immutable fields,
    so either equal value may be kept (a benign, idempotent write).
    """
    try:
        return object.__getattribute__(table, '_storage_form')
    except AttributeError:
        pass
    cache = table._native_cache
    contiguous = cache.get(_CONTIGUOUS_STORAGE)
    rectangular = cache.get(_RECTANGULAR_STORAGE)
    spec = table._index_error
    if spec is not None:
        levels = len(spec['boolean_modes']) if spec['multi'] else 1
        plain = not spec['multi'] and spec['boolean_modes'][0] != 1
    else:
        plain = True
        if contiguous is not None:
            levels = 1
        elif rectangular is not None:
            levels = len(rectangular[0])
        else:
            keys = table._keys
            levels = len(keys[0]) if keys and type(keys[0]) is tuple else 1
    if contiguous is not None:
        form = ('contiguous', contiguous[0], contiguous[1], plain, levels)
    elif rectangular is not None:
        form = ('rectangular', *rectangular, plain, levels)
    else:
        form = ('rows', plain, levels)
    object.__setattr__(table, '_storage_form', form)
    return form


def _row_count(table):
    storage = _storage_of(table)
    if storage[0] == 'rows':
        return len(table._keys)
    return len(storage[2])


def _levels(table):
    """Number of key levels: 1 for scalar keys."""
    return _storage_of(table)[-1]


def _iter_keys(table):
    storage = _storage_of(table)
    if storage[0] == 'rectangular':
        from itertools import product
        return product(*storage[1])
    return iter(table._keys)       # contiguous tables own their key tuple


class _ProductKeys:
    """The rectangular row keys as a read-only sequence, computed per position."""

    __slots__ = ('_axes', '_size')

    def __init__(self, axes):
        from math import prod
        self._axes, self._size = axes, prod(len(axis) for axis in axes)

    def __len__(self):
        return self._size

    def __getitem__(self, position):
        parts = []
        for axis in reversed(self._axes):
            position, digit = divmod(position, len(axis))
            parts.append(axis[digit])
        return tuple(reversed(parts))


def _key_sequence(table):
    storage = _storage_of(table)
    if storage[0] == 'rectangular':
        return _ProductKeys(storage[1])
    return table._keys


def _integral(key):
    """The int value of an integral key (int, IntEnum, integral float or numpy number), else None."""
    kind = type(key)
    if kind is int:
        return key
    if kind is bool:
        return None
    if kind is float:
        return int(key) if key.is_integer() else None
    if isinstance(key, int):           # IntEnum and other int subclasses
        return int(key)
    if kind.__module__ == 'numpy':
        from numpy import integer, floating
        if isinstance(key, (integer, floating)):
            try:
                number = int(key)
            except (ValueError, OverflowError):
                return None
            if key == number and hash(key) == hash(number):
                return number
    return None


def _find_row(table, key, storage=None):
    """Position of the complete key ``key``; KeyError(key) when it is not a row label.

    Pandas label rules: the index spec's boolean modes, a float equal to an int
    label matches, NaN matches a missing label of a general table.
    """
    if storage is None:
        storage = _storage_of(table)
    kind = storage[0]
    key_type = type(key)
    if not (storage[-2] and (key_type is int or key_type is str)):
        spec = table._index_error
        if spec is not None and not _valid_index_key(key, spec):
            raise KeyError(key)
    if kind == 'rows':
        spec = table._index_error
        try:
            if spec is None or key_type is int or key_type is str:
                return table._positions[key]
            return table._positions[_index_key(key, spec)]
        except (KeyError, TypeError):
            raise KeyError(key) from None
    if kind == 'contiguous':
        number = key if key_type is int else _integral(key)
        if number is None:
            if key_type in (str, bytes, float, bool, tuple) or key is None or key_type.__module__ == 'numpy':
                raise KeyError(key)
            try:
                return table._positions[key]       # exotic numbers: equality as a dict
            except (KeyError, TypeError):
                raise KeyError(key) from None
        position = number - storage[1]
        if 0 <= position < len(storage[2]):
            return position
        raise KeyError(key)
    axes, positions = storage[1], storage[3]
    if key_type is not tuple or len(key) != len(axes):
        raise KeyError(key)
    position = 0
    try:
        for part, labels in zip(key, positions):
            position = position * len(labels) + labels[part]
    except (KeyError, TypeError):
        raise KeyError(key) from None
    return position


def _row_values(table, position):
    storage = _storage_of(table)
    if storage[0] == 'contiguous':
        row = storage[2][position]
        return tuple(row) if table._preserve_scalar_types else tuple(row.tolist())
    if storage[0] == 'rectangular':
        return (float(storage[2][position]),)
    return table._values[position]


def _cell(table, position, column, storage=None):
    if storage is None:
        storage = _storage_of(table)
    kind = storage[0]
    if kind == 'rows':
        return table._values[position][column]
    if kind == 'contiguous':
        if table._preserve_scalar_types:
            return storage[2][position, column]
        return storage[2].item(position, column)
    return storage[2].item(position)


def _column_index(table, column):
    """The column's position; KeyError(column) as pandas when it is not a column."""
    try:
        return table._columns.index(column) if type(column) is str else _no_column(column)
    except ValueError:
        raise KeyError(column) from None


def _no_column(column):
    raise KeyError(column)


def _record(items):
    result = object.__new__(Record)
    object.__setattr__(result, '_items', items)
    object.__setattr__(result, '_index', MappingProxyType(dict(items)))
    object.__setattr__(result, '_native_groups', {})
    object.__setattr__(result, '_metadata', None)
    object.__setattr__(result, '_preserve_scalar_types', False)
    return result


def _row_record(table, position):
    """The Record of a stored row: the table's own (already frozen) values."""
    values = _row_values(table, position)
    if table._preserve_scalar_types:
        # Raw (possibly mutable) cells: Record construction freezes them.
        return Record(dict(zip(table._columns, values)))
    return _record(tuple(zip(table._columns, values)))


def _pandas_frame(table):
    """The DataFrame of the table, built once (expands compact storage); AttributeError if none."""
    frame = table._native_cache.get('pandas_frame')
    if frame is None:
        from .table_operations import to_pandas
        try:
            frame = to_pandas(table)
        except (TypeError, ValueError) as exc:   # hasattr()/getattr(..., default) expect AttributeError
            raise AttributeError('pandas passthrough unavailable') from exc
        frame = table._native_cache.setdefault('pandas_frame', frame)
    return frame


def _modes_admit(prefix, modes):
    """False when a part breaks its level's pandas boolean label mode (1: bool only, 2: no bool)."""
    for part, mode in zip(prefix, modes):
        if mode:
            boolean = type(part) is bool or type(part).__name__ == 'bool_'
            if boolean != (mode == 1) and not _missing_index_part(part):
                return False
    return True


def _spec_modes(table):
    spec = table._index_error
    return spec['boolean_modes'] if spec is not None else ()


def _child_spec(table, count, levels):
    spec = table._index_error
    if spec is None:
        return None
    modes = tuple(spec['boolean_modes'][count:])
    values = dict(spec._index)
    values.update(multi=levels - count > 1, boolean_modes=modes,
                  row_bool_labels=levels - count == 1 and modes == (1,))
    return Record(values)


def _child_metadata(table, positions, count):
    """Axes metadata of a child (a subset of the parent's pandas index), or None."""
    metadata = table._metadata
    if (metadata is None or metadata[0] not in ('axes', 'frame') or type(metadata[1]) is tuple
            or (metadata[0] == 'axes' and len(metadata) > 4)):
        return None
    return ('axes', ('subset', metadata[1], positions, tuple(range(count))), metadata[2], metadata[3])


def _prefix_groups(table, count):
    """{prefix: positions} of a row-stored table, built in one pass and memoised."""
    cache = table._native_cache
    groups = cache.get(('prefix_rows', count))
    if groups is None:
        grouping = {}
        spec = table._index_error
        for position, key in enumerate(table._keys):
            prefix = key[:count] if spec is None else _index_key(key, spec)[:count]
            grouping.setdefault(prefix, []).append(position)
        groups = cache.setdefault(('prefix_rows', count), grouping)
    return groups


def _prefix_block(table, prefix, storage):
    """(start, size) of a rectangular prefix block; KeyError when a part is not an axis label."""
    axes, positions = storage[1], storage[3]
    offset = 0
    for part, labels in zip(prefix, positions):
        offset = offset * len(labels) + labels[part]
    size = 1
    for axis in axes[len(prefix):]:
        size *= len(axis)
    return offset * size, size


def _has_prefix(table, prefix):
    if not _modes_admit(prefix, _spec_modes(table)):
        return False
    storage = _storage_of(table)
    try:
        if storage[0] == 'rectangular':
            _prefix_block(table, prefix, storage)
            return True
        spec = table._index_error
        key = prefix if spec is None else _index_key(prefix, spec) if spec['multi'] else prefix
        return key in _prefix_groups(table, len(prefix))
    except (KeyError, TypeError):
        return False


def _sub_table(table, item, prefix, levels):
    """``table.loc[prefix]`` on a multi-level table, as pandas: the matching rows with the
    selected levels dropped (scalar keys when one level is left); KeyError(item) when none.

    The child is a pure function of the immutable table and the prefix value, so it is
    built once and shared (per-prefix memo in ``_native_cache``, set with ``setdefault``;
    live and exported use only, never read by compiled code). A rectangular parent gives a
    contiguous child (one int level left, consecutive ascending labels) or a rectangular
    child (two or more levels left) over a slice of the same ndarray, otherwise a float-row
    child built from that slice; a row-stored parent gives a row-stored child.
    """
    count = len(prefix)
    modes = _spec_modes(table)
    if not _modes_admit(prefix, modes):
        raise KeyError(item)
    cache = table._native_cache
    entry = cache.get(('prefix_children', count))
    if entry is None:
        # (children by prefix value, the prefix levels' pandas boolean label modes)
        entry = cache.setdefault(('prefix_children', count), ({}, tuple(modes[:count])))
    children = entry[0]
    try:
        child = children.get(prefix)
    except TypeError:
        raise KeyError(item) from None
    if child is not None:
        return child
    storage = _storage_of(table)
    spec = _child_spec(table, count, levels)
    if storage[0] == 'rectangular':
        try:
            start, size = _prefix_block(table, prefix, storage)
        except (KeyError, TypeError):
            raise KeyError(item) from None
        rest, block = storage[1][count:], storage[2][start:start + size]
        metadata = _child_metadata(table, range(start, start + size), count)
        if len(rest) > 1:
            child = Table._from_rectangular(rest, block, table._columns, spec, metadata)
        else:
            labels = rest[0]
            if (all(type(label) is int for label in labels)
                    and labels == tuple(range(labels[0], labels[0] + len(labels)))):
                child = Table._from_contiguous(labels, block.reshape((-1, 1)), table._columns,
                                               spec, metadata, False)
            else:
                child = Table._from_float64(labels, ((value,) for value in block.tolist()),
                                            table._columns, spec, exact_keys=True)
                object.__setattr__(child, '_metadata', metadata)
    else:
        key = prefix
        if table._index_error is not None:
            key = _index_key(prefix, table._index_error) if table._index_error['multi'] else prefix
        try:
            positions = _prefix_groups(table, count).get(key)
        except TypeError:
            positions = None
        if positions is None:
            raise KeyError(item)
        keys, values = table._keys, table._values
        child = Table.__new__(Table)
        Table._initialize(child, tuple(keys[p][count:] if levels - count > 1 else keys[p][count]
                                       for p in positions),
                          tuple(values[p] for p in positions), table._columns, spec,
                          table._preserve_scalar_types)
        object.__setattr__(child, '_metadata', _child_metadata(table, tuple(positions), count))
    return children.setdefault(prefix, child)


class _TableLoc:
    """Label reads with pandas spelling.

    ``loc[key, "col"]`` the stored value; ``loc[key]`` the row as a Record (with
    ``rows=True`` the pandas-coerced row); ``loc[prefix]`` on a multi-level table the
    shared sub-table. A tuple key addresses a multi-level table. Errors follow pandas:
    a missing column raises KeyError(column) before the key is looked at, a missing key
    KeyError(key). One deliberate difference: a missing complete key in the row form
    ``loc[(a, b)]`` raises KeyError((a, b)); pandas reads the tuple as (row, column) and
    raises KeyError(b). Slices and lists raise TypeError.
    """

    __slots__ = ('_table',)

    def __init__(self, table):
        self._table = table

    def __getitem__(self, item):
        table = self._table
        if type(item) is tuple and len(item) == 2:
            key, column = item
            if type(column) is str and column in table._columns:
                storage = _storage_of(table)
                return _cell(table, _find_row(table, key, storage), table._columns.index(column), storage)
            levels = _levels(table)
            if levels == 1 or type(key) is tuple:
                if type(column) is slice or type(key) is slice:
                    raise TypeError('Table.loc supports loc[key, "column"], loc[key] and loc[prefix]')
                raise KeyError(column)
        prefix = item if type(item) is tuple else (item,)
        entry = table._native_cache.get(('prefix_children', len(prefix)))
        if entry is not None:
            try:
                child = entry[0].get(prefix)
            except TypeError:
                child = None
            if child is not None and _modes_admit(prefix, entry[1]):
                return child
        if type(item) is slice or (type(item) is tuple and any(type(part) is slice for part in item)):
            raise TypeError('Table.loc supports loc[key, "column"], loc[key] and loc[prefix], not slices')
        if type(item) is list:
            raise TypeError('Table.loc supports loc[key, "column"], loc[key] and loc[prefix], not lists')
        levels = _levels(table)
        if levels > 1:
            if len(prefix) < levels:
                return _sub_table(table, item, prefix, levels)
            if len(prefix) > levels:
                raise KeyError(item)
        metadata = table._metadata
        if metadata is not None and metadata[0] == 'frame' and metadata[4] is not None:
            return table.slice(item)
        return _row_record(table, _find_row(table, item))


class _TableAt:
    """``table.at[key, column]``: the stored value, as ``table.loc[key, column]`` (pandas ``.at``)."""

    __slots__ = ('_table',)

    def __init__(self, table):
        self._table = table

    def __getitem__(self, item):
        if not (type(item) is tuple and len(item) == 2):
            raise TypeError('Table.at supports at[key, "column"]')
        key, column = item
        table = self._table
        column = _column_index(table, column)       # pandas: the column error first
        return _cell(table, _find_row(table, key), column)


class _TableColumn(Mapping):
    """``table["col"]``: one column as a label -> value mapping, read like a pandas Series.

    ``column[label]`` and ``label in column`` select by row label (KeyError(label));
    ``column.index`` is the table's index; iteration yields the values in row order.
    """

    __slots__ = ('_table', '_column', '_position')

    def __init__(self, table, column):
        self._table, self._column = table, column
        self._position = table._columns.index(column)

    def __getitem__(self, label):
        return _cell(self._table, _find_row(self._table, label), self._position)

    def __contains__(self, label):
        return label in self._table.index

    def __iter__(self):
        table = self._table
        storage = _storage_of(table)
        if storage[0] == 'contiguous':
            values = storage[2][:, self._position]
            return iter(tuple(values) if table._preserve_scalar_types else values.tolist())
        if storage[0] == 'rectangular':
            return iter(storage[2].tolist())
        position = self._position
        return iter(tuple(values[position] for values in table._values))

    def __len__(self):
        return _row_count(self._table)

    @property
    def index(self):
        return self._table.index

    def _series(self):
        """The column as a pandas Series, a copy of the table's materialised frame (pandas passthrough)."""
        return _pandas_frame(self._table)[self._column].copy(deep=False)

    def __getattr__(self, name):
        """A public name outside the column vocabulary (``sum``, ``str``, ``between``, ``loc``, ``all``...)
        is read from a pandas Series copy, as ``Table`` does with a DataFrame. Materialising expands
        compact storage for good; formulas read the vocabulary instead (pandas passthrough for tests and
        reports, never a per-step read)."""
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._series(), name)


def _column_operator(name):
    def operator(self, *args):
        args = tuple(a._series() if isinstance(a, _TableColumn) else a for a in args)
        return getattr(self._series(), name)(*args)
    operator.__name__ = name
    operator.__doc__ = f'pandas ``Series.{name}`` on a copy of the column (passthrough).'
    return operator


# Comparison, arithmetic and logical operators of a pandas Series (they raised TypeError before).
# ``==`` and ``!=`` keep the Mapping equality of the column (pinned, D35).
for _name in ('__lt__', '__le__', '__gt__', '__ge__', '__add__', '__radd__', '__sub__', '__rsub__',
              '__mul__', '__rmul__', '__truediv__', '__rtruediv__', '__floordiv__', '__rfloordiv__',
              '__mod__', '__rmod__', '__pow__', '__rpow__', '__and__', '__rand__', '__or__', '__ror__',
              '__xor__', '__rxor__', '__neg__', '__pos__', '__abs__', '__invert__'):
    setattr(_TableColumn, _name, _column_operator(_name))
del _name


def _index_position(index, size):
    """``index[i]`` position as pandas: negative counts from the end; IndexError past the end."""
    from operator import index as as_index
    position = as_index(index)
    if position < 0:
        position += size
    if not 0 <= position < size:
        raise IndexError(f'index {as_index(index)} is out of bounds for axis 0 with size {size}')
    return position


def _below_first(label):
    # D25a: one behaviour in live, exported and compiled code (pandas returns NaN).
    raise KeyError(label)


def _asof(self, label):
    """See :meth:`_TableIndex.asof`."""
    table = self._table
    storage = _storage_of(table)
    if storage[0] == 'contiguous':
        keys = table._keys
        if type(label) is int:
            if label >= keys[-1]:
                return keys[-1]
            if label < keys[0]:
                _below_first(label)
            return keys[label - keys[0]]
        position = bisect_right(keys, label) - 1
        if position < 0:
            _below_first(label)
        return keys[position]
    if storage[0] == 'rows':
        increasing = table._native_cache.get('asof_increasing')
        if increasing is None:
            keys = table._keys
            increasing = table._native_cache.setdefault('asof_increasing', bool(
                keys and table._ordered is not None and keys == table._ordered
                and not any(type(key) is tuple for key in keys)))
        if increasing:
            keys = table._keys
            position = bisect_right(keys, label) - 1
            if position < 0:
                _below_first(label)
            return keys[position]
    result = self._pandas_index().asof(label)
    if type(result) is float and result != result:
        _below_first(label)
    return result


class _TableIndex:
    """Read-only view of the table's complete row keys in input order.

    The pandas ``Index`` vocabulary of the target syntax runs on the table's own storage:
    ``k in index`` (a partial key of a multi-level table is contained when some row starts
    with it, as pandas), ``len``, iteration, ``min()``, ``max()``, ``asof(k)``,
    ``searchsorted(k, side=...)`` and ``index[i]``. Any other public attribute is read from
    the pandas index of the materialised DataFrame (expands compact storage).
    """

    __slots__ = ('_table',)

    def __init__(self, table):
        self._table = table

    def __contains__(self, key):
        table = self._table
        levels = _levels(table)
        if levels > 1:
            prefix = key if type(key) is tuple else (key,)
            if 0 < len(prefix) < levels:
                return _has_prefix(table, prefix)
        try:
            _find_row(table, key)
        except KeyError:
            return False
        return True

    def __iter__(self):
        return _iter_keys(self._table)

    def __len__(self):
        return _row_count(self._table)

    def __getitem__(self, i):
        """``index[i]``: the i-th label in row order; negative i counts from the end."""
        if type(i) is bool or not hasattr(type(i), '__index__'):
            return self._pandas_index()[i]
        table = self._table
        size = _row_count(table)
        position = _index_position(i, size)
        storage = _storage_of(table)
        if storage[0] == 'contiguous':
            return storage[1] + position
        return _key_sequence(table)[position]

    def _pandas_index(self):
        return _pandas_frame(self._table).index

    def max_int(self) -> int:
        return int(self.max())

    def max_float(self) -> float:
        return float(self.max())

    def _extreme(self, last, args, kwargs):
        if args or kwargs:
            index = self._pandas_index()
            return (index.max if last else index.min)(*args, **kwargs)
        table = self._table
        storage = _storage_of(table)
        if storage[0] == 'contiguous':
            keys = table._keys
            return keys[-1] if last else keys[0]
        if storage[0] == 'rectangular':
            # a full product: the lexicographic extreme is the tuple of per-axis extremes
            pick = max if last else min
            return tuple(pick(axis) for axis in storage[1])
        keys = table._keys
        if not keys:
            return float('nan')
        ordered = table._ordered
        if ordered is not None:
            return ordered[-1] if last else ordered[0]
        return max(keys) if last else min(keys)

    def max(self, *args, **kwargs):
        """The largest row key, as ``pandas.Index.max`` (a Python scalar)."""
        return self._extreme(True, args, kwargs)

    def min(self, *args, **kwargs):
        """The smallest row key, as ``pandas.Index.min`` (a Python scalar)."""
        return self._extreme(False, args, kwargs)

    def asof(self, label):
        """The last label <= ``label``, as ``pandas.Index.asof``, except below the first label.

        Below the first label this raises ``KeyError(label)`` where pandas returns NaN
        (decision D25a): live, exported and compiled code then agree, and an integer
        key never turns into a float. Scalar labels stored in increasing order are
        searched directly; any other index (tuple labels, labels not in increasing
        order) runs the pandas method on the materialised index, with its errors.
        """
        return _asof(self, label)

    def searchsorted(self, value, side='left', sorter=None):
        """Insertion position of ``value`` in the (sorted) labels, as ``pandas.Index.searchsorted``.

        A Python int. Successor read: ``sub.index[sub.index.searchsorted(x, side="right")]``.
        Array values and ``sorter`` run the pandas method.
        """
        if side not in ('left', 'right'):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        if sorter is not None or type(value) is list or type(value).__name__ == 'ndarray':
            return self._pandas_index().searchsorted(value, side=side, sorter=sorter)
        from bisect import bisect_left
        table = self._table
        storage = _storage_of(table)
        if storage[0] == 'contiguous' and type(value) is int:
            start, size = storage[1], len(storage[2])
            return min(max(value - start + (side == 'right'), 0), size)
        keys = _key_sequence(table)
        return (bisect_right if side == 'right' else bisect_left)(keys, value)

    def __getattr__(self, name):
        """Read-only pandas passthrough to the materialised index."""
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._pandas_index(), name)


# ---------------------------------------------------------------------------
# Contracts for the compiler (round 3, W1.H). Neither expands compact storage.

def table_storage(table):
    """The table's own storage, for the compiler's table handle.

    Returns one of
      ``('contiguous', start, data, columns, spec)``   int keys ``start .. start + len(data) - 1``;
                                                       ``data`` a read-only C-ordered ndarray of
                                                       shape (rows, len(columns))
      ``('rectangular', axes, data, columns, spec)``   keys = the product of ``axes`` (tuples of
                                                       labels, in stored order, last axis fastest);
                                                       ``data`` a read-only 1-D float64 ndarray, one
                                                       column
      ``('rows', keys, values, ordered, columns, spec)`` keys and value tuples in row order;
                                                       ``ordered`` the sorted keys or None
    ``columns`` is the tuple of column names and ``spec`` the pandas index spec Record
    (``multi``, ``boolean_modes``, ...) or None. The objects are the table's own; do not
    change them. Sub-tables (``loc[prefix]``) report their own storage.
    """
    if not isinstance(table, Table):
        raise TypeError('table_storage requires a Table')
    storage = _storage_of(table)
    if storage[0] == 'contiguous':
        return ('contiguous', storage[1], storage[2], table._columns, table._index_error)
    if storage[0] == 'rectangular':
        return ('rectangular', storage[1], storage[2], table._columns, table._index_error)
    return ('rows', table._keys, table._values, table._ordered, table._columns, table._index_error)


def _value_kind(value):
    kind = type(value)
    if kind is bool:
        return 'bool'
    if kind is int:
        return 'int'
    if kind is float:
        return 'float'
    if kind is str:
        return 'str'
    from enum import Enum
    if isinstance(value, Enum):
        return f'enum:{kind.__module__}.{kind.__qualname__}'
    if kind.__module__ == 'numpy':
        import numpy as np
        if isinstance(value, np.bool_):
            return 'bool'
        if isinstance(value, np.integer):
            return 'int'
        if isinstance(value, np.floating):
            return 'float'
    return None


def _common_kind(values):
    kinds = {_value_kind(value) for value in values}
    return kinds.pop() if len(kinds) == 1 else None


def _dtype_kind(dtype):
    return {'f': 'float', 'i': 'int', 'u': 'int', 'b': 'bool'}.get(dtype.kind)


def table_schema(table):
    """Key-level and column kinds of a table, read without expanding compact storage.

    ``{'storage': 'contiguous' | 'rectangular' | 'rows', 'rows': row count,
       'keys': (kind of level 0, ...), 'columns': {name: kind}}``
    A kind is ``'bool'``, ``'int'``, ``'float'``, ``'str'``, ``'enum:<module>.<qualname>'``
    for enum members kept as labels or values, or None (mixed, missing or other values,
    or an empty table). Tables prepared from pandas hold IntEnum labels as plain ints
    (pandas stores them as int64), so their levels report ``'int'``.
    """
    if not isinstance(table, Table):
        raise TypeError('table_schema requires a Table')
    storage = _storage_of(table)
    columns = table._columns
    if storage[0] == 'contiguous':
        keys = ('int',)
        kind = _dtype_kind(storage[2].dtype)
        column_kinds = {name: kind for name in columns}
        if storage[2].dtype.kind == 'O':
            column_kinds = {name: _common_kind(storage[2][:, i]) for i, name in enumerate(columns)}
    elif storage[0] == 'rectangular':
        keys = tuple(_common_kind(axis) for axis in storage[1])
        column_kinds = {columns[0]: _dtype_kind(storage[2].dtype)}
    else:
        rows = table._keys
        levels = _levels(table)
        if levels > 1:
            keys = tuple(_common_kind(key[level] for key in rows) for level in range(levels))
        else:
            keys = (_common_kind(rows),)
        values = table._values
        column_kinds = {name: _common_kind(row[i] for row in values) for i, name in enumerate(columns)}
    return {'storage': storage[0], 'rows': _row_count(table), 'keys': keys, 'columns': column_kinds}
