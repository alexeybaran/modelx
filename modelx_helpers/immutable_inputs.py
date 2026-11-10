"""Owned immutable records and keyed tables for model inputs.

Construction is explicit: existing dictionaries and pandas objects are never
silently frozen. Keys and authored enum objects retain their Python identity.
The explicit preserve_scalar_types option owns structure while retaining raw
cell references, including mutable object cells; it is a shallow snapshot.
"""
from bisect import bisect_right
from collections.abc import Mapping
from types import MappingProxyType


class Record(Mapping):
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


class Table(Mapping):
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
            try:
                if not _valid_index_key(key, self._index_error):
                    raise KeyError(key)
                position = self._positions[_index_key(key, self._index_error)]
            except KeyError:
                raise KeyError(_index_error_key(key, self._index_error)) from None
        column = self._columns.index(column) if isinstance(column, str) else column
        return self._values[position][column]

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
    if isinstance(value, (Record, Table)):
        return value
    if type(value) is dict:
        if value.keys() == {'keys', 'values', 'columns'}:
            return Table(value['keys'], value['values'], value['columns'])
        return Record(value)
    if type(value) in (tuple, list):
        return tuple(freeze(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if type(value).__module__.startswith('numpy'):
        import numpy as np
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value
    # These pandas missing scalars are immutable, pickleable singletons. Keep
    # their identity: replacing NA/NaT with NaN changes conversion exceptions.
    if type(value).__module__.startswith('pandas.'):
        from pandas import NA, NaT
        if value is NA or value is NaT:
            return value
    raise TypeError(f'unsupported immutable input value: {type(value).__name__}')


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
    if len(parts) != len(spec['levels']):
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


def _index_error_key(key, spec):
    if not spec.get('encoded_errors', True):
        return key
    if not spec['multi'] or type(key) is not tuple or len(key) != len(spec['levels']):
        return key
    if not _valid_index_key(key, spec):
        return key
    result = 0
    for part, level, offset in zip(key, spec['levels'], spec['offsets']):
        if _missing_index_part(part):
            code = spec['bias'] - 1
        else:
            try:
                code = level.index(part) + spec['bias']
            except ValueError:
                return key
        result |= code << offset
    return result
