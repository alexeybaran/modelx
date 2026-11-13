"""Pandas-spelled reads on prepared tables: Table.loc, .at, T["c"], iterrows and the index.

Ported from the syntax-experiment prototype (modelx 7e20e10) with two adaptations (D27):
the grouped ``Record.loc`` test reads the same keys from a flat table (grouped tables are
outside the target syntax and ``Record.loc`` is not ported), and ``index.asof`` below the
first label raises KeyError instead of returning NaN (D25a).
"""
import pandas as pd
import pytest

from modelx_helpers import pdn


@pytest.fixture
def frames():
    plain = pd.DataFrame({'mort_rate': [0.1, 0.2, 0.3], 'years': [1, 2, 3]},
                         index=pd.Index([30, 31, 32], name='age'))
    grouped = pd.DataFrame({'plan': ['A', 'A', 'B'], 'sex': ['M', 'F', 'M'], 'year': [1, 1, 1],
                            'premium_pp': [1.0, 2.0, 3.0]}).set_index(['plan', 'sex', 'year'])
    return plain, grouped


def test_loc_scalar_matches_pandas(frames):
    plain, _ = frames
    table = pdn.table(plain)
    for age in (30, 31, 32):
        for column in ('mort_rate', 'years'):
            assert table.loc[age, column] == plain.loc[age, column]
    assert 31 in table.index and 33 not in table.index
    assert table.index.max() == plain.index.max() and table.index.min() == plain.index.min()


def test_loc_missing_and_bool_keys(frames):
    plain, _ = frames
    table = pdn.table(plain)
    with pytest.raises(KeyError):
        table.loc[33, 'mort_rate']
    with pytest.raises(KeyError):
        table.loc[True, 'mort_rate']     # bool is not the integer label 1
    with pytest.raises(TypeError):
        table.loc[30:31]


def test_loc_row(frames):
    plain, _ = frames
    assert dict(pdn.table(plain).loc[31]) == {'mort_rate': 0.2, 'years': 2}
    assert pdn.table(plain, rows=True).loc[31]['years'] == 2


def test_multi_level_full_key_matches_pandas(frames):
    """Adapted (D27): the prototype read these keys through a grouped Record (groups=)."""
    _, grouped = frames
    table = pdn.table(grouped, columns=('premium_pp',))
    for key in grouped.index:
        assert table.loc[key, 'premium_pp'] == grouped.loc[key, 'premium_pp']
    assert table.loc[('A', 'F')].loc[1, 'premium_pp'] == 2.0


def test_loc_row_is_the_stored_record(frames):
    """loc[key] without rows=True: an exact Record of the stored row, built without per-column lookups."""
    import pickle
    from modelx_helpers.immutable_inputs import Record, Table
    plain, _ = frames
    table = pdn.table(plain)
    row = table.loc[31]
    expected = Record(dict(zip(table._columns, (table.lookup(31, c) for c in table._columns))))
    assert type(row) is Record and row == expected and row._items == expected._items
    assert pickle.loads(pickle.dumps(row)) == expected
    with pytest.raises(KeyError):
        row['nope']
    with pytest.raises(KeyError):
        table.loc[33]
    with pytest.raises(KeyError):
        table.loc[True]                   # bool is not the integer label 1
    bare = Table([1, 2], [(1, 2.0), (3, 4.0)], ['p', 'q'])
    assert dict(bare.loc[2]) == {'p': 3, 'q': 4.0}
    with pytest.raises(KeyError):
        bare.loc[5]


def test_iterrows_without_pandas():
    """Table.iterrows(): (label, row Record) pairs; values equal to DataFrame.iterrows(), rows shared."""
    from modelx_helpers.immutable_inputs import Record
    frame = pd.DataFrame({'jump_lo': [0.0, 1.5, 3], 'jump_hi': [1.5, 3.0, 99],
                          'rate': [0.1, 0.3, 0.6], 'note': ['a', 'b', 'c']})
    for table in (pdn.table(frame), pdn.table(frame, rows=True)):
        got = list(table.iterrows())
        assert [label for label, _ in got] == list(frame.index)
        for (label, row), (_, expected) in zip(got, frame.iterrows()):
            assert type(row) is Record
            assert all(row[c] == expected[c] for c in frame.columns)
        assert all(a is b for (_, a), (_, b) in zip(got, table.iterrows()))   # built once


@pytest.mark.parametrize('labels, queries', [
    ([1, 3, 7], (0, 1, 2, 5, 9)),            # below first, exact, between, above last
    ([1.5, 2.5], (0, 1, 2, 9)),
    (['a', 'c'], ('0', 'a', 'b', 'z')),
    ([7, 1, 3], (0, 2, 9)),                  # not monotonic: pandas raises, so does the helper
])
def test_index_asof_matches_pandas(labels, queries):
    """Adapted (D27): below the first label the helper raises KeyError(label) where pandas
    returns NaN (D25a); everywhere else it matches pandas, errors included."""
    import math
    frame = pd.DataFrame({'v': range(len(labels))}, index=pd.Index(labels, name='k'))
    table = pdn.table(frame)
    def outcome(read):
        try:
            value = read()
        except Exception as exc:          # noqa: BLE001 - compare the exception types
            return type(exc)
        return 'nan' if isinstance(value, float) and math.isnan(value) else value
    for label in queries:
        expected = outcome(lambda: frame.index.asof(label))
        if expected == 'nan':
            with pytest.raises(KeyError) as error:
                table.index.asof(label)
            assert error.value.args == (label,)
            continue
        assert outcome(lambda: table.index.asof(label)) == expected, label


def _prefix_frame():
    return pd.DataFrame({'s': ['M', 'M', 'M', 'F', 'F'], 'c': ['A', 'A', 'B', 'A', 'A'],
                         'y': [1, 2, 1, 1, 3], 'v': [1., 2., 3., 4., 5.], 'w': [9., 8., 7., 6., 5.]}
                        ).set_index(['s', 'c', 'y'])


@pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')
@pytest.mark.parametrize('prefix', ['M', ('M',), ('M', 'A'), ('F', 'A'), 'X', ('M', 'Z')])
def test_partial_key_selection_matches_pandas(prefix):
    frame = _prefix_frame()
    table = pdn.table(frame)
    try:
        expected = frame.loc[prefix]
    except KeyError:
        with pytest.raises(KeyError):
            table.loc[prefix]
        return
    sub = table.loc[prefix]
    assert type(sub) is type(table)
    assert list(sub._keys) == list(expected.index)
    assert [list(values) for values in sub._values] == expected.values.tolist()
    assert sub.index.max() == expected.index.max() and sub.index.min() == expected.index.min()
    if not isinstance(sub._keys[0], tuple):
        for k in (0, 1, 2, 3, 9):
            if k < expected.index.min():
                with pytest.raises(KeyError):
                    sub.index.asof(k)       # pandas: NaN (D25a)
            else:
                assert sub.index.asof(k) == expected.index.asof(k)
            assert (k in sub.index) == (k in expected.index)
            column, series = sub['w'], expected['w']
            assert (k in column) == (k in series)
            if k in series:
                assert sub.loc[k, 'v'] == expected.loc[k, 'v'] and column[k] == series[k]
            else:
                with pytest.raises(KeyError):
                    column[k]
        assert list(sub['w']) == list(expected['w']) and sub['w'].index.max() == expected['w'].index.max()


def test_partial_key_selection_is_shared_by_prefix_value():
    import enum

    class Sex(enum.IntEnum):
        F = 0
        M = 1
    frame = pd.DataFrame({'a': [0, 0, 1, 1], 'y': [1, 2, 1, 3], 'v': [1., 2., 3., 4.]}).set_index(['a', 'y'])
    table = pdn.table(frame)
    assert table.loc[0] is table.loc[0] is table.loc[Sex.F] is table.loc[(0,)]
    assert table.loc[1] is not table.loc[0]
    assert list(table.loc[Sex.M]._keys) == [1, 3]
    text = pdn.table(_prefix_frame())
    key = ''.join(['M', ''])        # an equal str value, not the same object
    assert text.loc[(key, 'A')] is text.loc[('M', 'A')] and text.loc[('M', 'B')] is not text.loc[('M', 'A')]
    with pytest.raises(KeyError):
        text.loc[('M', 'Z')]
