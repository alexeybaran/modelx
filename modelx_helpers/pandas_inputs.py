"""Explicit pandas-to-owned-input preparation for ordinary modelx Data Cells.

Calculation helpers live in :mod:`modelx_helpers.typed_inputs` and operate on the
existing Table/Record structures in every execution mode. Public pandas inputs
remain separate. Preparation follows normal modelx dependency/cache rules.
"""
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd

from .immutable_inputs import Record, Table, _copy_container, freeze


def isna(value):
    """Test for missing values with pandas scalar and array semantics."""
    return pd.isna(value)


isna.__modelx_cython_helper__ = ('modelx_cython.runtime._native_inputs', 'isna')
isna.__modelx_cython_scalar_result__ = 'object'


def _scalar(value):
    # itertuples preserves per-column scalar values; iterrows would coerce a
    # numeric mixed row to one dtype before the explicit preparation boundary.
    if isinstance(value, np.generic):
        value = value.item()
    return freeze(value)


def _key(value):
    if isinstance(value, tuple):
        return tuple(_key(part) for part in value)
    return _scalar(value)


def _container_values(values, positions):
    return {i: _copy_container(values[i]) for i in positions
            if type(values[i]) in (list, tuple, dict)}


def _record_metadata(result, metadata, values):
    dtype = metadata[2]
    if (not result._preserve_scalar_types
            and (dtype is object or getattr(dtype, 'kind', None) == 'O')):
        values = tuple(values)
        containers = _container_values(values, range(len(values)))
        if containers:
            metadata = (*metadata, containers)
    object.__setattr__(result, '_metadata', metadata)


def table(frame, *, columns=None, groups=(), indexing='loc', rows=False, report=False,
          preserve_scalar_types=False):
    """Build an existing Table, optionally nested in explicit group Records.

    ``groups`` names index levels to group, in the authored nesting order. The
    remaining index levels form leaf keys. Row order and named column order are
    retained. This constructor does not sort, fill missing values, aggregate
    duplicates or coerce every numeric column to float. Those are separate
    operations, never implicit preparation policies.

    ``indexing='at'`` requires complete, ungrouped keys. Lookup errors identify
    the requested label, and boolean labels retain pandas selection rules.

    The normal Table contract requires unique leaf keys and nonempty unique
    string columns. Unsupported scalar values raise rather than silently lose
    their dtype/missing semantics. Public DataFrames are not modified.

    ``rows=True`` additionally owns pandas-style row scalar coercion for row()
    selection. Lookup-only tables omit this second matrix. Row coercion follows
    the selected columns. Grouped report/row metadata retains remaining levels.
    All tables own the axes and column dtypes needed by to_pandas().
    ``report=True`` also retains attributes and enables column reports.
    ``rows=True`` includes the report metadata and prepared rows.

    ``preserve_scalar_types=True`` retains actual column-selected scalar types
    and object references, deferring value conversion to selected typed lookups.
    Only the structure is owned: mutable object cells retain pandas' shallow
    identity. Keys, dimensions and uniqueness are still validated eagerly.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('table requires a pandas DataFrame')
    if indexing not in ('loc', 'at'):
        raise ValueError('indexing must be loc or at')
    if indexing == 'at' and groups:
        raise ValueError('at preparation requires complete keys')
    index_error = _pandas_index_spec(frame.index) if not groups else None
    selected = list(frame.columns) if columns is None else list(columns)
    if not selected or any(type(name) is not str for name in selected):
        raise ValueError('table columns must be nonempty strings')
    if len(set(selected)) != len(selected) or not frame.columns.is_unique:
        raise ValueError('table columns must be unique')
    selected_frame = frame if columns is None and type(frame) is pd.DataFrame else frame.loc[:, selected]
    dtypes = tuple(selected_frame.dtypes)
    object_columns = tuple(i for i, dtype in enumerate(dtypes)
                           if getattr(dtype, 'kind', None) == 'O'
                           and not isinstance(dtype, pd.StringDtype))
    names = list(frame.index.names)
    groups = tuple(groups)
    if len(set(groups)) != len(groups):
        raise ValueError('group levels must be unique')
    if any(names.count(level) != 1 for level in groups):
        raise ValueError('group levels must name unique index levels')
    grouped_positions = [names.index(level) for level in groups]
    remaining_positions = [i for i in range(frame.index.nlevels)
                           if i not in grouped_positions]
    if groups and not remaining_positions:
        raise ValueError('grouping must leave at least one leaf key level')
    root = OrderedDict() if groups else []
    if preserve_scalar_types:
        # Series positional selection retains NumPy scalars; itertuples may
        # convert those scalars to Python builtins. Neither path coerces rows.
        raw_columns = tuple(selected_frame[name] for name in selected)
        source_rows = (tuple(column.iloc[position] for column in raw_columns)
                       for position in range(len(selected_frame)))
    else:
        source_rows = selected_frame.itertuples(index=False, name=None)
    original_objects = {}
    for position, (label, row) in enumerate(zip(frame.index, source_rows)):
        parts = tuple(label) if isinstance(frame.index, pd.MultiIndex) else (label,)
        parts = tuple(_key(part) for part in parts)
        leaf_parts = tuple(parts[i] for i in remaining_positions)
        key = leaf_parts[0] if len(leaf_parts) == 1 else leaf_parts
        value = tuple(row) if preserve_scalar_types else tuple(_scalar(item) for item in row)
        if object_columns and not preserve_scalar_types:
            containers = _container_values(row, object_columns)
            if containers:
                original_objects[position] = containers
        if not groups:
            root.append((key, value, position))
            continue
        branch = root
        for group_position in grouped_positions[:-1]:
            branch = branch.setdefault(parts[group_position], OrderedDict())
        branch.setdefault(parts[grouped_positions[-1]], []).append((key, value, position))

    axes = _axes_metadata(selected_frame, dtypes) if groups else None

    def build(branch, depth):
        if depth == len(groups):
            positions = tuple(entry[2] for entry in branch) if groups else None
            leaf_spec = index_error
            if groups and (rows or report):
                # Reconstruction axes are separate from scalar selection.
                # Plain grouped leaves use ordinary Table keys.
                leaf_index = axes[1].take(list(positions)).droplevel(grouped_positions)
                leaf_spec = _pandas_index_spec(leaf_index)
            result = Table([entry[0] for entry in branch],
                           [entry[1] for entry in branch], selected, leaf_spec,
                           preserve_scalar_types=preserve_scalar_types)
            if rows or report:
                leaf = selected_frame
                if groups:
                    leaf = selected_frame.iloc[list(positions)].copy(deep=False)
                    leaf.index = leaf_index
                metadata = _report_metadata(leaf, rows=rows,
                                            preserve_scalar_types=preserve_scalar_types)
            else:
                if groups:
                    index = ('subset', axes[1], positions, tuple(grouped_positions))
                    metadata = ('axes', index, *axes[2:])
                else:
                    metadata = _axes_metadata(selected_frame, dtypes)
            if original_objects:
                containers = ({i: original_objects[pos] for i, pos in enumerate(positions)
                               if pos in original_objects} if groups else original_objects)
                if containers:
                    metadata = (*metadata, containers)
            object.__setattr__(result, '_metadata', metadata)
            return result, positions

        children = {}
        positions = []
        for key, child in branch.items():
            children[key], child_positions = build(child, depth + 1)
            positions.extend(child_positions)
        result = Record(children)
        # The tree groups rows together. Retain a permutation to restore
        # input order, including interleaved groups and non-leading levels.
        order = tuple(sorted(range(len(positions)), key=positions.__getitem__))
        index = axes[1]
        if depth:
            index = ('subset', index, tuple(positions[i] for i in order),
                     tuple(grouped_positions[:depth]))
        object.__setattr__(result, '_metadata', ('grouped', index, *axes[2:], order))
        return result, positions

    return build(root, 0)[0]


def _axes_metadata(frame, dtypes=None):
    if dtypes is None:
        dtypes = tuple(frame.dtypes)
    return ('axes', frame.index.copy(deep=True), frame.columns.copy(deep=True), dtypes)


def _report_metadata(frame, *, rows, preserve_scalar_types):
    # Own row Records here so policy calculations can read the prepared row.
    columns = frame.columns.copy(deep=True)
    attrs = deepcopy(frame.attrs)
    allows_duplicates = frame.flags.allows_duplicate_labels
    row_values, row_records, row_dtypes, row_names = [], [], [], []
    if rows:
        for position in range(len(frame)):
            public_row = frame.iloc[position]
            values = {name: public_row[name] for name in columns}
            selected = Record(values,
                              preserve_scalar_types=preserve_scalar_types)
            _record_metadata(selected, ('row', columns, public_row.dtype,
                                        public_row.name, attrs, allows_duplicates), values.values())
            row_values.append(tuple(value for _, value in selected._items))
            row_records.append(selected)
            row_dtypes.append(public_row.dtype)
            row_names.append(public_row.name)
    return ('frame', frame.index.copy(deep=True), columns,
            tuple(frame.dtypes), tuple(row_values) if rows else None,
            tuple(row_dtypes), attrs, allows_duplicates,
            tuple(row_names), (frame.columns.dtype, frame.columns.name),
            tuple(row_records) if rows else None)


def records(frame):
    """Prepare owned row Records keyed by the frame index, without row coercion."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('records requires a pandas DataFrame')
    if not frame.index.is_unique:
        raise ValueError('duplicate record key')
    columns = tuple(frame.columns)
    if not frame.columns.is_unique:
        raise ValueError('record columns must be unique')
    axes = _axes_metadata(frame)
    result = {}
    for key, row in zip(frame.index, frame.itertuples(index=False, name=None)):
        owned = Record({column: _scalar(value) for column, value in zip(columns, row)})
        _record_metadata(owned, ('row', axes[2], object, key, {}, True), row)
        result[_key(key)] = owned
    owned = Record(result)
    object.__setattr__(owned, '_metadata', ('records', *axes[1:]))
    return owned


def record(row, *, preserve_scalar_types=False):
    """Prepare a Record from an explicitly selected public pandas Series.

    A model can keep its public model_point Cell unchanged and make a separate
    preparation Cell depend on that value. Its mixed-row coercion/name semantics
    stay in the public pandas selection; calculation reads the owned fields.
    preserve_scalar_types=True retains Series-selected scalar/object references
    without freezing unrelated values; typed conversion occurs at actual lookup.
    """
    if not isinstance(row, pd.Series):
        raise TypeError('record requires a pandas Series')
    if not row.index.is_unique:
        raise ValueError('record fields must be unique')
    if preserve_scalar_types:
        # Series.items() can yield Python scalars while Series[key] returns a
        # numpy scalar. Preserve the actual public scalar-selection contract.
        values = {key: row[key] for key in row.index}
        result = Record(values, preserve_scalar_types=True)
    else:
        values = dict(row.items())
        result = Record({key: _scalar(value) for key, value in values.items()})
    _record_metadata(result, ('row', row.index.copy(deep=True), row.dtype, row.name,
                             deepcopy(row.attrs), row.flags.allows_duplicate_labels), values.values())
    return result


def index(labels):
    """Own public Index labels for internal ordered traversal.

    The public Index Cell remains unchanged. Unique labels retain the existing
    Table representation with position data. Duplicate labels use an existing
    Record containing the owned key tuple, preserving order and multiplicity
    without weakening Table's unique-lookup-key invariant. This is a traversal
    input, not an implicit conversion of ambiguous labels into unique lookups.
    """
    if not isinstance(labels, pd.Index):
        raise TypeError('index requires a pandas Index')
    keys = tuple(_key(label) for label in labels)
    if not labels.is_unique:
        result = Record({'keys': keys})
        object.__setattr__(result, '_metadata', ('index',))
        return result
    return Table(keys,
                 tuple((position,) for position in range(len(labels))),
                 ('position',), _pandas_index_spec(labels))



def _pandas_index_spec(index):
    """Capture the label domains used by pandas scalar selection."""
    multi = isinstance(index, pd.MultiIndex)
    levels = index.levels if multi else (index,)
    modes = []
    for level in levels:
        values = tuple(_key(value) for value in level)
        if values and all(type(value) is bool for value in values):
            modes.append(1)
        elif any(type(value) is bool for value in values):
            modes.append(0)
        else:
            modes.append(2)
    return Record({
        'multi': multi, 'pandas_columns': True,
        'row_bool_labels': pd.api.types.is_bool_dtype(index.dtype),
        'boolean_modes': tuple(modes)})


table.__modelx_native_operation__ = 'table'
table.__modelx_cython_helper__ = ('modelx_cython.runtime._native_inputs', 'table')
table.__modelx_cython_call_layout__ = 'keywords'
table.__modelx_cython_scalar_result__ = 'object'


def select_row(frame, key):
    """Select a public pandas row at the owning Cell's input boundary.

    Delegate the original .loc semantics, including labels, scalar coercion,
    duplicate selections and errors. Keeping selection in model_point makes a
    cleared Projection observe edits to its public input table. Financial
    formulas can consume record(model_point(), preserve_scalar_types=True).
    """
    return frame.loc[key]
