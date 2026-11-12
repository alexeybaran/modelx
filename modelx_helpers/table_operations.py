"""Explicit row/column expressions over the existing owned Table and Record.

Preparation and final pandas materialization are visible boundaries. Calculation
selection has direct Cython provider declarations, without formula recognition.
"""
from copy import deepcopy
from types import MappingProxyType

from .immutable_inputs import Table, Record, _copy_container, _index_key, _valid_index_key

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
        raise KeyError(key) from None
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
    metadata = ('series', meta[1], meta[3][position], name, meta[6], meta[7])
    if len(meta) > 11:
        containers = {i: values[position] for i, values in meta[11].items()
                      if position in values}
        if containers:
            metadata = (*metadata, containers)
    object.__setattr__(result, '_metadata', metadata)
    return result


def _axis(spec):
    import pandas as pd
    if type(spec) is tuple and spec[0] == 'labels':
        return pd.Index(spec[1], name=spec[2], dtype=spec[3], tupleize_cols=False)
    if type(spec) is tuple and spec[0] == 'range':
        return pd.RangeIndex(spec[1], spec[2], spec[3], name=spec[4])
    return spec.copy(deep=True)


def _frame(values, index, columns, dtypes):
    import numpy as np
    import pandas as pd
    if all(dtype == np.dtype('float64') for dtype in dtypes):
        # Retain column-major storage used by the report reductions.
        numbers = np.array(values, dtype='float64', order='F').reshape(
            (len(index), len(columns)), order='F')
        return pd.DataFrame(numbers, index=index, columns=columns, copy=False)
    result = pd.DataFrame({name: pd.Series([row[i] for row in values],
                           index=index, dtype=dtypes[i])
                           for i, name in enumerate(columns)}, index=index)
    result.columns = columns
    return result


def _restore_containers(values, containers):
    return tuple(_copy_container(containers[i]) if i in containers else value
                 for i, value in enumerate(values))


def _table_rows(value):
    meta = value._metadata
    if meta is not None:
        if meta[0] == 'frame' and len(meta) > 11:
            containers = meta[11]
        elif meta[0] == 'axes' and len(meta) > 4:
            containers = meta[4]
        else:
            return value._values
        return tuple(_restore_containers(row, containers[i]) if i in containers else row
                     for i, row in enumerate(value._values))
    return value._values


def _record_values(value):
    meta = value._metadata
    values = (value._index['values'] if meta is not None and meta[0] == 'series'
              else tuple(value._index.values()))
    if meta is not None and meta[0] in ('row', 'series') and len(meta) > 6:
        return _restore_containers(values, meta[6])
    return values


def _grouped_rows(value):
    if isinstance(value, Table):
        yield from _table_rows(value)
    else:
        for _, child in value._items:
            yield from _grouped_rows(child)


def to_pandas(value, *, index_name=_MISSING):
    """Materialize an owned Table or Record as a pandas DataFrame or Series.

    Frames keep the selected column and input row order. Grouped tables restore
    their MultiIndex. Preparation options control retained report attributes
    and row selection, while values and labels are available for every table.
    """
    import pandas as pd
    if not isinstance(value, (Table, Record)):
        owner = getattr(value, '__modelx_table_owner__', None)
        if callable(owner):
            value = owner()
    if not isinstance(value, (Table, Record)):
        raise TypeError('to_pandas requires an owned Table or Record')

    meta = value._metadata
    kind = meta[0] if meta is not None else None
    if isinstance(value, Table):
        if kind in ('frame', 'axes'):
            result = _frame(_table_rows(value), _axis(meta[1]), _axis(meta[2]), meta[3])
        else:
            # Tables built from Python mappings and older prepared tables
            # carry keys and values without source axis metadata.
            spec = value._index_error
            if spec is not None and spec['multi']:
                index = pd.MultiIndex.from_tuples(value._keys) if value._keys else (
                    pd.MultiIndex.from_arrays([[] for _ in spec['boolean_modes']]))
            else:
                index = pd.Index(value._keys, dtype=object, tupleize_cols=False)
            columns = pd.Index(value._columns)
            result = _frame(value._values, index, columns, (object,) * len(columns))
        if kind == 'frame':
            result.attrs = deepcopy(meta[6])
            result.flags.allows_duplicate_labels = meta[7]
    elif kind in ('row', 'series'):
        result = pd.Series(_record_values(value),
                           index=_axis(meta[1]), dtype=meta[2], name=meta[3])
        result.attrs = deepcopy(meta[4])
        result.flags.allows_duplicate_labels = meta[5]
    elif kind == 'grouped':
        rows = tuple(_grouped_rows(value))
        result = _frame(tuple(rows[i] for i in meta[4]), _axis(meta[1]), _axis(meta[2]), meta[3])
    elif kind == 'records':
        rows = tuple(_record_values(row) for _, row in value._items)
        result = _frame(rows, _axis(meta[1]), _axis(meta[2]), meta[3])
    else:
        result = pd.Series(list(value._index.values()),
                           index=pd.Index(tuple(value), dtype=object, tupleize_cols=False),
                           dtype=object)
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
