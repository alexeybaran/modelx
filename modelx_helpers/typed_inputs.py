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


def sum_helper(values):
    return sum(values, 0.0)


def any_true(values):
    """Return whether the iterable yields any truthy value."""
    return any(values)


def all_true(values):
    """Return whether every value in the iterable is truthy."""
    return all(values)


def max_value(values):
    """Return the first greatest value, preserving the iterable's value type."""
    return max(values)


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


for _operation in (lookup, previous, contains, key_count, key_at, first_key,
                   last_key, subtable, has_group, sum_f64, sum_helper, any_true, all_true, max_value,
                   prepared_interpolation):
    _operation.__modelx_native_operation__ = _operation.__name__
    _operation.__modelx_cython_helper__ = ('modelx_cython.runtime._native_inputs', _operation.__name__)

_native_operations = {function.__name__: function for function in
                      (lookup, previous, contains, key_count, key_at, first_key,
                       last_key, subtable, has_group, sum_f64, sum_helper, any_true, all_true, max_value,
                       prepared_interpolation)}

sum_f64.__modelx_cython_reduction__ = 'f64_sum'
any_true.__modelx_cython_reduction__ = 'bool_any'
all_true.__modelx_cython_reduction__ = 'bool_all'
max_value.__modelx_cython_reduction__ = 'f64_max'
lookup.__modelx_cython_numeric__ = 'lookup_f64'
lookup.__modelx_cython_clamped__ = 'lookup_clamped_f64'
lookup.__modelx_cython_clamped_context__ = 'lookup_clamped_context_f64'
lookup.__modelx_cython_optional__ = 'optional_lookup_f64'
lookup.__modelx_cython_grouped__ = ('group_probe', 'group_value', 'group_context_probe')
lookup.__modelx_cython_bounded__ = 'bounded_lookup_f64'
previous.__modelx_cython_numeric__ = 'previous_f64'

lookup.__modelx_cython_bound_numeric__ = ('numeric_resource_ready', 'bound_lookup_f64', 'prepare_lookup_f64')
previous.__modelx_cython_bound_numeric__ = ('numeric_resource_ready', 'bound_previous_f64', 'prepare_previous_f64')

lookup.__modelx_cython_bound_bounded__ = 'bound_bounded_f64'
lookup.__modelx_cython_bound_grouped__ = ('bound_group_probe', 'bound_group_value', 'group_context_value')

# The ordered-key producer is one function, shared with generator consumers.
from .table_keys import keys
lookup.__modelx_cython_float_numeric__ = 'lookup_float_f64'
lookup.__modelx_cython_keyed_float_numeric__ = (
    'lookup_key0_float_f64', 'lookup_key1_float_f64', 'lookup_key2_float_f64',
    'lookup_key3_float_f64', 'lookup_key4_float_f64')


# Explicit operations: their .pxd return declarations drive native typing.
_TYPED_LOOKUP = lookup
_LOOKUP_INT, _LOOKUP_FLOAT, _LOOKUP_BOOL, _LOOKUP_STR = int, float, bool, str
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


def lookup_bool(source, key, column=_MISSING, default=_MISSING) -> bool:
    """Read a value using Python bool conversion, including NA errors."""
    return _LOOKUP_BOOL(_typed_value(source, key, column, default))


def lookup_str(source, key, column=_MISSING, default=_MISSING) -> str:
    """Read a value using Python str conversion."""
    return _LOOKUP_STR(_typed_value(source, key, column, default))


for _operation in (lookup_int, lookup_float, lookup_bool, lookup_str):
    _operation.__modelx_native_operation__ = _operation.__name__
    _operation.__modelx_cython_helper__ = ('modelx_cython.runtime._native_inputs', _operation.__name__)
    _native_operations[_operation.__name__] = _operation



def lookup_object(source, key, column=_MISSING, default=_MISSING) -> object:
    """Read the original scalar, including missing sentinels, without conversion."""
    return _typed_value(source, key, column, default)

lookup_object.__modelx_native_operation__ = 'lookup_object'
lookup_object.__modelx_cython_helper__ = ('modelx_cython.runtime._native_inputs', 'lookup_object')
_native_operations['lookup_object'] = lookup_object


def is_missing(predicate, value) -> bool:
    """Test a scalar, preserving the current pandas isna callable explicitly.

    Pass the predicate first so it is resolved before evaluating the value,
    just as in predicate(value). Noncanonical predicates retain their behavior.
    """
    return _LOOKUP_BOOL(predicate(value))


is_missing.__modelx_native_operation__ = 'is_missing'
is_missing.__modelx_cython_helper__ = ('modelx_cython.runtime._native_inputs', 'is_missing')
_native_operations['is_missing'] = is_missing

# Shared declared-call ABI, owned by the generic compiler helper registry.
# Physical results describe these .pxd dispatchers, including the existing
# type-compatible replacement contract; they do not infer callback bodies.
for _operation, _result in ((lookup_float, 'float64'), (lookup_int, 'int64'),
                           (lookup_bool, 'bool'), (lookup_str, 'unicode'),
                           (lookup_object, 'object'), (is_missing, 'bool')):
    _operation.__modelx_cython_call_layout__ = 'keywords'
    _operation.__modelx_cython_scalar_result__ = _result
