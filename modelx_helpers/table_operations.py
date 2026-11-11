"""Explicit row/column expressions over the existing owned Table and Record.

Preparation and final pandas materialization are visible boundaries. Calculation
selection has direct Cython provider declarations, without formula recognition.
"""
from copy import deepcopy
from types import MappingProxyType

from .immutable_inputs import (Table, Record, _index_key, _index_error_key,
                               _valid_index_key)

_MISSING = object()


def row(table, key):
    """Select a unique row from an ungrouped table prepared with rows=True.

    Pandas scalar coercion follows the columns selected during preparation.
    """
    if not isinstance(table, Table):
        raise TypeError('row requires an owned Table')
    metadata = table._metadata
    if metadata is None or metadata[0] != 'frame':
        raise ValueError('row requires table(..., rows=True) with pandas row metadata')
    if metadata[4] is None:
        raise ValueError('row requires table(..., rows=True)')
    if type(key) is bool and table._index_error is not None:
        spec = table._index_error._index
        if not spec.get('row_bool_labels', spec['boolean_modes'][0] == 1):
            raise KeyError(f'{key}: boolean label can not be used without a boolean index')
    try:
        if table._index_error is not None and not _valid_index_key(key, table._index_error):
            raise KeyError(key)
        position = table._positions[_index_key(key, table._index_error)]
    except KeyError:
        raise KeyError(_index_error_key(key, table._index_error)) from None
    if len(metadata) > 10 and metadata[10] is not None:
        return metadata[10][position]
    # Older tables only have scalar tuples.
    result = object.__new__(Record)
    items = tuple(zip(table._columns, metadata[4][position]))
    object.__setattr__(result, '_items', items)
    object.__setattr__(result, '_index', MappingProxyType(dict(items)))
    object.__setattr__(result, '_native_groups', {})
    object.__setattr__(result, '_preserve_scalar_types',
                       getattr(table, '_preserve_scalar_types', False))
    object.__setattr__(result, '_metadata', ('row', metadata[2], metadata[5][position],
                                           metadata[8][position], metadata[6], metadata[7]))
    return result


def column(table, name, *, report=False):
    """Materialize ordered immutable values with existing column scalar types.

    This is a value sequence, not a public Series. No aggregation or missing-value
    policy is implied. With report=True, select an existing Record carrying the
    positional values and preserved axes/dtype/name/attrs/flags for to_pandas().
    This internal report carrier does not provide a general labelled Series API.
    """
    if not isinstance(table, Table):
        raise TypeError('column requires an owned Table')
    try:
        position = table._columns.index(name)
    except ValueError:
        raise KeyError(name) from None
    values = tuple(values[position] for values in table._values)
    if not report:
        return values
    meta = table._metadata
    if meta is None or meta[0] != 'frame':
        raise ValueError('column report requires table(..., report=True) metadata')
    result = object.__new__(Record)
    items = (('values', values),)
    object.__setattr__(result, '_items', items)
    object.__setattr__(result, '_index', MappingProxyType(dict(items)))
    object.__setattr__(result, '_native_groups', {})
    object.__setattr__(result, '_preserve_scalar_types',
                       getattr(table, '_preserve_scalar_types', False))
    object.__setattr__(result, '_metadata', ('series', meta[1], meta[3][position],
                                          name, meta[6], meta[7]))
    return result


def to_pandas(value, *, index_name=_MISSING):
    """Materialize owned report metadata into the public DataFrame or Series.

    Python/pandas execution at this output boundary is explicit and timed. Row
    and column calculations precede it. Untagged records are not guessed to be
    report rows, and tables without preserved metadata are rejected.

    A column selected with report=True restores a Series using its original
    index, dtype and column name. No filtering, ordering or reduction occurs.
    """
    import pandas as pd
    def axis(spec):
        if type(spec) is tuple and spec[0] == 'labels':
            return pd.Index(spec[1], name=spec[2], dtype=spec[3])
        if type(spec) is tuple and spec[0] == 'range':
            return pd.RangeIndex(spec[1], spec[2], spec[3], name=spec[4])
        return spec.copy(deep=True)
    if type(value) is Table:
        meta = value._metadata
        if meta is None or meta[0] != 'frame':
            raise ValueError('to_pandas requires table(..., report=True) metadata')
        index, columns, dtypes = axis(meta[1]), axis(meta[2]), meta[3]
        import numpy as np
        if all(dtype == np.dtype('float64') for dtype in dtypes):
            # One matrix/constructor avoids a temporary Series per column. Keep
            # column-major storage, as in the dict-of-Series construction, so
            # later pandas/NumPy reductions retain their numerical dispatch.
            numbers = np.array(value._values, dtype='float64', order='F').reshape(
                (len(index), len(columns)), order='F')
            result = pd.DataFrame(numbers, index=index, columns=columns, copy=False)
        else:
            result = pd.DataFrame({name: pd.Series([row[i] for row in value._values],
                                   index=index, dtype=dtypes[i])
                                   for i, name in enumerate(value._columns)}, index=index)
            result.columns = columns
        result.attrs = deepcopy(meta[6])
        result.flags.allows_duplicate_labels = meta[7]
    elif type(value) is Record and value._metadata is not None and value._metadata[0] in ('row', 'series'):
        meta = value._metadata
        values = value._index['values'] if meta[0] == 'series' else [entry[1] for entry in value._items]
        result = pd.Series(values,
                           index=axis(meta[1]), dtype=meta[2], name=meta[3])
        result.attrs = deepcopy(meta[4])
        result.flags.allows_duplicate_labels = meta[5]
    else:
        raise TypeError('to_pandas requires an owned report Table or row Record')
    if index_name is not _MISSING:
        result.index = result.index.rename(index_name)
    return result


def group_sum(table, labels, *, first=(), last=()):
    """Group by explicit labels, summing flows and selecting nonmissing states.

    First columns lead, then last columns, then summed columns in source order.
    Ordinary positional labels and pandas Series alignment retain pandas rules.
    Native compilation admits its declared numeric domains; other inputs keep
    this reference behavior and remain explicit native-coverage gaps.
    """
    import pandas as pd
    if not isinstance(table, Table):
        raise TypeError('group_sum requires an owned Table')
    frame = to_pandas(table)
    first = (first,) if isinstance(first, str) else tuple(first)
    last = (last,) if isinstance(last, str) else tuple(last)
    requested = first + last
    if len(set(requested)) != len(requested):
        raise ValueError('first and last columns must be distinct')
    for name in requested:
        if name not in table._columns:
            raise KeyError(name)
    if isinstance(labels, (tuple, list)):
        if len(labels) != len(frame):
            raise ValueError('group labels must match table length')
        labels = pd.Index(labels)
    remaining = [name for name in table._columns if name not in requested]
    out = frame[remaining].groupby(labels).sum()
    for name in reversed(last):
        out.insert(0, name, frame[name].groupby(labels).last())
    for name in reversed(first):
        out.insert(0, name, frame[name].groupby(labels).first())
    from .pandas_inputs import table as prepare
    return prepare(out, report=True)


for _function in (row, column, group_sum):
    _function.__modelx_native_operation__ = _function.__name__
    _function.__modelx_cython_helper__ = ('modelx_cython.runtime._native_inputs', _function.__name__)
    _function.__modelx_cython_call_layout__ = 'keywords'
    _function.__modelx_cython_scalar_result__ = 'object'
