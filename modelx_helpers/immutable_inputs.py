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


class Record(Mapping):
    __modelx_cython_type__ = (1, 'modelx_cython.runtime._native_inputs',
                             'owned-table-1', 'modelx_helpers.immutable_inputs.Record')
    __slots__ = ('_items', '_index', '_native_groups', '_metadata', '_preserve_scalar_types')

    def __init__(self, values, *, preserve_scalar_types=False):
        items = tuple((key, value if preserve_scalar_types else freeze(value))
                      for key, value in values.items())
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

    @property
    def loc(self):
        """Pandas-spelled reads through grouped tables: ``grouped.loc[(g1, ..., gn, key), "col"]``."""
        return _RecordLoc(self)

    def __iter__(self):
        return iter(self._index)

    def __len__(self):
        return len(self._index)

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


class _RecordLoc:
    """Follow a full tuple key through grouped Records to the leaf Table."""

    __slots__ = ('_record',)

    def __init__(self, record):
        self._record = record

    def __getitem__(self, item):
        column = None
        if type(item) is tuple and len(item) == 2 and type(item[1]) is str and type(item[0]) is tuple:
            item, column = item
        parts = list(item) if type(item) is tuple else [item]
        value = self._record
        while parts and isinstance(value, Record):
            value = value[parts.pop(0)]
        if column is None:
            return value.loc[tuple(parts) if len(parts) > 1 else parts[0]] if parts else value
        if not isinstance(value, Table) or not parts:
            raise KeyError(item)
        return value.loc[tuple(parts) if len(parts) > 1 else parts[0], column]


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

    def max(self):
        """The largest row key, as ``pandas.Index.max``."""
        return max(self._table._keys)

    def min(self):
        """The smallest row key, as ``pandas.Index.min``."""
        return min(self._table._keys)

    def asof(self, label):
        """The last label <= ``label``, NaN below the first one, as ``pandas.Index.asof``.

        Scalar labels stored in increasing order are searched directly; any other index (tuple
        labels, labels not in increasing order) runs the pandas method on the materialised index,
        with its results and errors.
        """
        table = self._table
        increasing = table._native_cache.get('asof_increasing')
        if increasing is None:
            increasing = table._native_cache.setdefault('asof_increasing', bool(
                table._keys and table._ordered is not None and table._keys == table._ordered
                and not any(type(key) is tuple for key in table._keys)))
        if not increasing:
            return self.__getattr__('asof')(label)
        position = bisect_right(table._keys, label) - 1
        return table._keys[position] if position >= 0 else float('nan')

    def __getattr__(self, name):
        """Read-only pandas passthrough to the materialised index."""
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._table.__getattr__('index'), name)


class _TableLoc:
    """Label-based reads with pandas spelling: ``table.loc[key, "col"]`` and ``table.loc[key]``.

    ``loc[key, column]`` returns the stored value, like ``Table.lookup``. A tuple key
    addresses a table with a multi-part index. ``loc[key]`` selects the row as a Record.
    Only these two forms are supported; anything else raises ``TypeError``.
    """

    __slots__ = ('_table',)

    def __init__(self, table):
        self._table = table

    def __getitem__(self, item):
        table = self._table
        if type(item) is tuple and len(item) == 2 and type(item[1]) is str and item[1] in table._columns:
            return table.lookup(item[0], item[1])
        if type(item) is slice or (type(item) is tuple and any(type(part) is slice for part in item)):
            raise TypeError('Table.loc supports loc[key, "column"] and loc[key], not slices')
        prefix = _key_prefix(table, item)
        if prefix is not None:
            return _sub_table(table, item, prefix)
        metadata = table._metadata
        if metadata is not None and metadata[0] == 'frame' and metadata[4] is not None:
            return table.slice(item)
        # Without pandas row metadata, the row is a Record of the stored values. The key is
        # resolved once and the stored row is zipped with the columns at C speed; the values
        # are the table's own (already frozen) objects, so no per-column lookup is repeated.
        if table._preserve_scalar_types:
            # Raw (possibly mutable) cells: Record construction freezes them; keep that copy.
            return Record(dict(zip(table._columns, (table.lookup(item, column) for column in table._columns))))
        spec = table._index_error
        if spec is None:
            position = table._positions[item]
        else:
            if not _valid_index_key(item, spec):
                raise KeyError(item)
            try:
                position = table._positions[_index_key(item, spec)]
            except KeyError:
                raise KeyError(item) from None
        return _stored_row(table, position)


def _key_prefix(table, item):
    """The prefix tuple when ``item`` selects fewer levels than a multi-level table has, else None."""
    keys = table._keys
    if not keys or type(keys[0]) is not tuple:
        return None
    prefix = item if type(item) is tuple else (item,)
    return prefix if 0 < len(prefix) < len(keys[0]) else None


def _sub_table(table, item, prefix):
    """``table.loc[prefix]`` on a multi-level table, as pandas: the matching rows with the selected
    levels dropped (a scalar key when one level is left); KeyError(item) when nothing matches.

    The child is a pure function of this immutable table and the prefix VALUE, so it is built once
    and shared (Decision 13.3): the first selection at a prefix depth groups the rows by prefix in
    one pass; each child is built on its first selection. Both live in ``_native_cache`` (owned by
    the table, uncapped), set with ``setdefault`` as the ``iterrows`` rows are.
    """
    count, spec = len(prefix), table._index_error
    modes = spec['boolean_modes'] if spec is not None else ()
    for part, mode in zip(prefix, modes):
        boolean = type(part) is bool or type(part).__name__ == 'bool_'
        if (mode == 1 and not boolean) or (mode == 2 and boolean):
            raise KeyError(item)
    cache = table._native_cache
    groups = cache.get(('prefix_rows', count))
    if groups is None:
        grouping = {}
        for position, key in enumerate(table._keys):
            grouping.setdefault(key[:count], []).append(position)
        groups = cache.setdefault(('prefix_rows', count), grouping)
    children = cache.get(('prefix_children', count))
    if children is None:
        children = cache.setdefault(('prefix_children', count), {})
    child = children.get(prefix)
    if child is not None:
        return child
    positions = groups.get(prefix)
    if positions is None:
        raise KeyError(item)
    keys, values = table._keys, table._values
    sub_spec = None
    if spec is not None:
        sub_spec = dict(spec._index)
        sub_spec.update(multi=len(keys[0]) - count > 1, boolean_modes=tuple(modes[count:]))
    child = Table([keys[p][count:] if len(keys[p]) - count > 1 else keys[p][count] for p in positions],
                  [values[p] for p in positions], table._columns, sub_spec,
                  preserve_scalar_types=table._preserve_scalar_types)
    return children.setdefault(prefix, child)


class _TableColumn(Mapping):
    """``table["col"]``: one column as a label -> value mapping, read like a pandas Series.

    ``column[label]`` and ``label in column`` select by row label; ``column.index`` is the table's
    index (``max()``, ``min()``, ``asof()``); iteration yields the values, as a Series does.
    """

    __slots__ = ('_table', '_column')

    def __init__(self, table, column):
        self._table, self._column = table, column

    def __getitem__(self, label):
        return self._table.lookup(label, self._column)

    def __contains__(self, label):
        return self._table.contains(label)

    def __iter__(self):
        position = self._table._columns.index(self._column)
        return iter(tuple(values[position] for values in self._table._values))

    def __len__(self):
        return len(self._table._keys)

    @property
    def index(self):
        return self._table.index


def _stored_row(table, position):
    """The exact Record of a stored row: the table's own (already frozen) values, zipped at C speed."""
    items = tuple(zip(table._columns, table._values[position]))
    result = object.__new__(Record)
    object.__setattr__(result, '_items', items)
    object.__setattr__(result, '_index', MappingProxyType(dict(items)))
    object.__setattr__(result, '_native_groups', {})
    object.__setattr__(result, '_metadata', None)
    object.__setattr__(result, '_preserve_scalar_types', False)
    return result


class _TableAt:
    """``table.at[key, column]``: the stored value, as ``table.loc[key, column]`` (pandas ``.at``)."""

    __slots__ = ('_table',)

    def __init__(self, table):
        self._table = table

    def __getitem__(self, item):
        if not (type(item) is tuple and len(item) == 2):
            raise TypeError('Table.at supports at[key, "column"]')
        key, column = item
        if type(column) is not str or column not in self._table._columns:
            raise KeyError(column)
        return self._table.lookup(key, column)


class Table(Mapping):
    __modelx_cython_type__ = (1, 'modelx_cython.runtime._native_inputs',
                             'owned-table-1', 'modelx_helpers.immutable_inputs.Table')
    __slots__ = ('_keys', '_values', '_columns', '_positions', '_ordered', '_native_cache', '_index_error', '_metadata', '_preserve_scalar_types')

    def __init__(self, keys, values, columns, index_error=None, *, preserve_scalar_types=False):
        keys, columns = tuple(keys), tuple(columns)
        values = tuple(tuple(value if preserve_scalar_types else freeze(value)
                             for value in row) for row in values)
        if not columns or len(set(columns)) != len(columns):
            raise ValueError('table columns must be nonempty and unique')
        if any(type(column) is not str for column in columns):
            raise TypeError('table columns must be strings')
        if len(keys) != len(values) or any(len(row) != len(columns) for row in values):
            raise ValueError('table dimensions do not match')
        position_keys = keys if index_error is None else tuple(_index_key(key, index_error) for key in keys)
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
        if self._index_error is None:
            position = self._positions[key]
        else:
            if not _valid_index_key(key, self._index_error):
                raise KeyError(key)
            try:
                position = self._positions[_index_key(key, self._index_error)]
            except KeyError:
                raise KeyError(key) from None
        column = self._columns.index(column) if isinstance(column, str) else column
        return self._values[position][column]

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

    def iterrows(self):
        """``(label, row)`` pairs in row order, like ``DataFrame.iterrows()`` but without pandas.

        Each row is the Record ``loc[label]`` returns: with ``rows=True`` the pandas-coerced row,
        otherwise the stored values. The rows are built once per table and shared (Records are
        immutable), so iterating in a per-policy formula costs no pandas and no per-row copies.
        """
        rows = self._native_cache.get('iterrows')
        if rows is None:
            metadata = self._metadata
            if metadata is not None and metadata[0] == 'frame' and metadata[4] is not None:
                rows = tuple((key, self.slice(key)) for key in self._keys)
            elif self._preserve_scalar_types:
                rows = tuple((key, Record(dict(zip(self._columns, values))))
                             for key, values in zip(self._keys, self._values))
            else:
                rows = tuple((key, _stored_row(self, position)) for position, key in enumerate(self._keys))
            rows = self._native_cache.setdefault('iterrows', rows)
        return iter(rows)

    def previous(self, key, column):
        if self._ordered is None:
            raise TypeError('predecessor lookup requires ordered keys')
        index = bisect_right(self._ordered, key) - 1
        if index < 0:
            raise ValueError(f'No key at or before {key!r}')
        return self.lookup(self._ordered[index], column)

    def contains(self, key):
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

    def __getattr__(self, name):
        """Read-only pandas passthrough for anything outside the table vocabulary.

        ``table.iterrows()``, ``table.iloc[-1]``, ``table.columns`` and the like run on a
        DataFrame materialised from the table, at pandas speed. The DataFrame is built once
        and handed out as a copy-on-write copy, so changing it cannot change the table.
        """
        if name.startswith('_'):
            raise AttributeError(name)
        frame = self._native_cache.get('pandas_frame')
        if frame is None:
            from .table_operations import to_pandas
            try:
                frame = to_pandas(self)
            except (TypeError, ValueError) as exc:   # hasattr() and getattr(..., default) expect AttributeError
                raise AttributeError(name) from exc
            frame = self._native_cache.setdefault('pandas_frame', frame)
        return getattr(frame.copy(deep=False), name)

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
