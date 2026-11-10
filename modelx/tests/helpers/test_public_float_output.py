"""Bulk public materialization retains schema, ownership and reduction results."""
import warnings

import numpy as np
import pandas as pd
import pytest

from modelx_helpers import pandas_inputs as inputs
from modelx_helpers.table_operations import to_pandas


@pytest.mark.parametrize('height', [0, 1, 4, 127, 257])
def test_float_output_reductions_and_independent_public_values(height):
    values = np.resize(np.array([1e16, 1., -1e16, 3., np.nan, np.inf, -np.inf, -0.]), (height, 3))
    source = pd.DataFrame({name: values[:, i] for i, name in enumerate(['a', 'b', 'c'])},
                          index=pd.Index(range(height), name='t'))
    source.columns.name = 'cashflow'
    source.attrs = {'basis': ['original']}
    source.flags.allows_duplicate_labels = False
    owned = inputs.table(source, report=True)
    actual = to_pandas(owned)
    pd.testing.assert_frame_equal(actual, source, check_exact=True)
    assert actual.attrs == source.attrs
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        pd.testing.assert_series_equal(actual.sum(), source.sum(), check_exact=True)
    if height:
        actual.iloc[0, 0] = 9.0
        assert to_pandas(owned).iloc[0, 0] == source.iloc[0, 0]
    actual.attrs['basis'].append('edited')
    assert to_pandas(owned).attrs == {'basis': ['original']}


@pytest.mark.parametrize('empty', [False, True])
def test_mixed_nullable_and_object_columns_keep_exact_output_contract(empty):
    source = pd.DataFrame({'amount': pd.Series([None, 0.1], dtype=object),
                          'count': pd.Series([42, pd.NA], dtype='Int64'),
                          'flag': [True, False], 'rate': [0.1, np.nan]})
    if empty:
        source = source.iloc[:0]
    actual = to_pandas(inputs.table(source, report=True))
    pd.testing.assert_frame_equal(actual, source, check_exact=True)
    assert not len(source) or actual.loc[0, 'amount'] is None
