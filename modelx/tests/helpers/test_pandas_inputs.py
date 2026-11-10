"""Preparation contracts for the canonical existing Table/Record representation."""
from copy import deepcopy
import math
import pickle

import numpy as np
import pandas as pd
import pytest

from modelx_helpers import pandas_inputs as inputs, typed_inputs as h
from modelx_helpers.immutable_inputs import Table, Record


def test_mixed_values_owned_and_no_iterrows_numeric_coercion():
    source = pd.DataFrame({'age': np.array([2**53 + 1, 42], dtype='int64'),
                           'rate': [0.1, float('nan')],
                           'label': ['M', 'F'], 'flag': [True, False]}, index=[8, 2])
    before = source.copy(deep=True)
    table = inputs.table(source)
    assert type(table) is Table
    assert table['keys'] == (8, 2)
    assert type(h.lookup(table, 8, 'age')) is int
    assert h.lookup(table, 8, 'age') == 2**53 + 1
    assert type(h.lookup(table, 8, 'flag')) is bool
    assert h.lookup(table, 2, 'label') == 'F'
    assert math.isnan(h.lookup(table, 2, 'rate'))
    source.loc[8, 'rate'] = 9.0
    assert h.lookup(table, 8, 'rate') == 0.1
    assert h.lookup(inputs.table(source), 8, 'rate') == 9.0
    pd.testing.assert_frame_equal(before, source.assign(rate=[0.1, float('nan')]))
    assert deepcopy(table) is table
    restored = pickle.loads(pickle.dumps(table))
    assert h.lookup(restored, 8, 'age') == 2**53 + 1


def test_explicit_group_order_and_remaining_tuple_keys():
    index = pd.MultiIndex.from_tuples([('M', 'N', 42), ('F', 'Y', 41), ('M', 'N', 40)],
                                     names=['sex', 'smoker', 'age'])
    frame = pd.DataFrame({'rate': [0.2, 0.3, 0.1]}, index=index)
    groups = inputs.table(frame, groups=('sex', 'smoker'))
    assert type(groups) is Record and tuple(groups) == ('M', 'F')
    assert groups['M']['N']['keys'] == (42, 40)
    assert h.lookup(groups['M']['N'], 40, 'rate') == 0.1
    assert h.previous(groups['M']['N'], 41, 'rate') == 0.1
    partly = inputs.table(frame, groups=('sex',))
    assert partly['M']['keys'] == (('N', 42), ('N', 40))
    assert h.lookup(partly['M'], ('N', 42), 'rate') == 0.2
    restored = pickle.loads(pickle.dumps(groups))
    assert h.lookup(restored['F']['Y'], 41, 'rate') == 0.3


def test_empty_frames_and_declared_column_order():
    source = pd.DataFrame({'a': [], 'b': []})
    table = inputs.table(source, columns=('b', 'a'))
    assert table['keys'] == () and table['columns'] == ('b', 'a')
    assert h.lookup(table, 0, 'a', 7.0) == 7.0
    with pytest.raises(KeyError): h.lookup(table, 0, 'a')
    assert len(inputs.records(source)) == 0


def test_records_preserve_scalar_types_and_nulls():
    source = pd.DataFrame({'integer': [2**53 + 1], 'rate': [0.1], 'name': ['M'],
                           'optional': [None], 'flag': [True]}, index=['p'])
    rows = inputs.records(source)
    row = rows['p']
    assert type(rows) is Record and type(row) is Record
    assert type(row['integer']) is int and row['integer'] == 2**53 + 1
    assert row['optional'] is None and row['flag'] is True
    assert pickle.loads(pickle.dumps(rows))['p']['name'] == 'M'


@pytest.mark.parametrize('frame,kwargs,error', [
    (pd.DataFrame({'a': [1, 2]}, index=[1, 1]), {}, 'duplicate table key'),
    (pd.DataFrame([[1, 2]], columns=['a', 'a']), {}, 'unique'),
    (pd.DataFrame({'a': [1]}), {'groups': ('absent',)}, 'unique index levels'),
    (pd.DataFrame({'a': [1]}, index=pd.Index([1], name='age')), {'groups': ('age',)}, 'leave at least'),
    (pd.DataFrame({'a': [1]}), {'columns': ()}, 'nonempty'),
])
def test_constructor_errors_are_explicit_not_silent_coercions(frame, kwargs, error):
    with pytest.raises(ValueError, match=error): inputs.table(frame, **kwargs)


def test_public_row_can_remain_series_with_separate_owned_record():
    frame = pd.DataFrame({'age': [42], 'rate': [0.1]})
    row = frame.loc[0]
    assert row.dtype == np.dtype('float64')
    prepared = inputs.record(row)
    assert type(prepared['age']) is float and prepared['age'] == row['age']
    row['rate'] = 0.9
    assert prepared['rate'] == 0.1
    assert inputs.record(row)['rate'] == 0.9


@pytest.mark.parametrize('dtype,values', [('int64', [42, 43]), ('float64', [0.1, 0.2]),
                                        ('bool', [True, False]), ('Int64', [42, pd.NA])])
def test_explicit_record_preserves_public_scalar_selection_types(dtype, values):
    row = pd.Series(values, index=['a', 'b'], dtype=dtype)
    prepared = inputs.record(row, preserve_scalar_types=True)
    for restored in [prepared, pickle.loads(pickle.dumps(prepared))]:
        for key in row.index:
            actual, expected = restored[key], row[key]
            assert type(actual) is type(expected)
            if expected is pd.NA:
                assert actual is expected
            else:
                assert actual == expected
