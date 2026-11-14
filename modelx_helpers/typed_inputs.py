"""Python reference operations for named numeric tables in scalar models."""

from bisect import bisect_right
import math
from .immutable_inputs import Table, _MISSING


_FLOOR = math.floor


class EnumDefinitions(tuple):
    """Names of model-code enum bindings; compiled definitions require rebuilding."""
    __modelx_cython_enum_definitions__ = True

    def __new__(cls, names):
        if isinstance(names, str):
            raise TypeError('enum definitions require a sequence of Reference names')
        names = tuple(names)
        if (any(type(name) is not str or not name.isidentifier() or name.startswith('_')
                for name in names) or len(set(names)) != len(names)):
            raise ValueError('enum definition names must be unique Reference identifiers')
        return tuple.__new__(cls, names)


def _column(table, column):
    return table['columns'].index(column) if isinstance(column, str) else column


def lookup(table, key, column, default=_MISSING):
    if isinstance(table, Table):
        # A pandas scalar selection validates its column before its row. Keep
        # this error outside missing-key/default handling. Bare Tables retain
        # their original row-first contract and ValueError for absent columns.
        if (isinstance(column, str) and table._index_error is not None
                and table._index_error._index.get('pandas_columns', False)):
            name = column
            try:
                column = table._columns.index(name)
            except ValueError:
                raise KeyError(name) from None
        try:
            return table.lookup(key, column)
        except KeyError:
            if default is _MISSING:
                raise
            return default
    try:
        index = table['keys'].index(key)
    except ValueError:
        if default is _MISSING:
            raise KeyError(key) from None
        return default
    return table['values'][index][_column(table, column)]


def previous(table, key, column, default=_MISSING):
    if isinstance(table, Table):
        if not table['keys'] or key < first_key(table):
            if default is _MISSING:
                raise ValueError(f'No key at or before {key!r}')
            return default
        return table.previous(key, column)
    index = bisect_right(table['keys'], key) - 1
    if index < 0:
        if default is _MISSING:
            raise ValueError(f'No key at or before {key!r}')
        return default
    return table['values'][index][_column(table, column)]


def contains(table, key):
    if isinstance(table, Table):
        return table.contains(key)
    return key in table['keys']


def key_count(table):
    return len(table['keys'])


def key_at(table, index):
    if index < 0:
        raise IndexError(index)
    return table['keys'][index]


def first_key(table):
    return min(table['keys'])


def last_key(table):
    return max(table['keys'])


def subtable(table, key):
    return table[key]


def has_group(table, key):
    return key in table


def sum_f64(values):
    return sum(values, 0.0)


def prepared_interpolation(table, column, years, lookup_fn, last_fn, math_module):
    """Interpolate a numeric column on adjacent integer rows, or return None."""
    if (lookup_fn is not _native_operations['lookup']
            or last_fn is not _native_operations['last_key']
            or math_module is not math or math.floor is not _FLOOR
            or type(table) is not Table or not isinstance(column, int)
            or type(years) is not float or not -(1 << 63) <= years < (1 << 63)):
        return None
    column = int.__index__(column)
    keys = table['keys']
    if (not 0 <= column < len(table['columns']) or not keys
            or not all(type(key) is int and -(1 << 63) <= key < (1 << 63) for key in keys)
            or not all(type(value) is float for row in table['values'] for value in row)):
        return None
    lower, upper = _FLOOR(years), last_fn(table)
    try:
        base = lookup_fn(table, min(lower, upper), column)
        if lower >= upper:
            return base
        end = lookup_fn(table, lower + 1, column)
    except KeyError:
        return None
    return base + (end - base) * (years - lower)


# Provider operations kept by modelx_cython.runtime._native_inputs.
for _operation in (lookup, first_key, last_key):
    _operation.__modelx_native_operation__ = _operation.__name__
    _operation.__modelx_cython_helper__ = ('modelx_cython.runtime._native_inputs', _operation.__name__)

_native_operations = {function.__name__: function for function in
                      (lookup, first_key, last_key, subtable, has_group, sum_f64,
                       prepared_interpolation)}


# The ordered-key producer is one function, shared with generator consumers.
from .table_keys import keys


# Explicit operations: their .pxd return declarations drive native typing.
_TYPED_LOOKUP = lookup
_LOOKUP_INT, _LOOKUP_FLOAT = int, float
def _typed_value(source, key, column=_MISSING, default=_MISSING):
    from .immutable_inputs import Record
    if isinstance(source, Record):
        if column is not _MISSING:
            raise TypeError('Record lookup does not accept a column; use default=')
        try:
            return source[key]
        except KeyError:
            if default is _MISSING:
                raise
            return default
    if column is _MISSING:
        raise TypeError('Table lookup requires a column')
    return _TYPED_LOOKUP(source, key, column, default)


def lookup_float(source, key, column=_MISSING, default=_MISSING) -> float:
    """Read and convert a Table value or Record field to a float64 number."""
    return _LOOKUP_FLOAT(_typed_value(source, key, column, default))


def lookup_int(source, key, column=_MISSING, default=_MISSING) -> int:
    """Read a signed-int64 value; preserve int conversion and check its range."""
    result = _LOOKUP_INT(_typed_value(source, key, column, default))
    if not -(1 << 63) <= result < (1 << 63):
        raise OverflowError('lookup_int result outside signed 64-bit range')
    return result


for _operation in (lookup_int, lookup_float):
    _native_operations[_operation.__name__] = _operation


def lookup_object(source, key, column=_MISSING, default=_MISSING) -> object:
    """Read the original scalar, including missing sentinels, without conversion."""
    return _typed_value(source, key, column, default)

_native_operations['lookup_object'] = lookup_object
