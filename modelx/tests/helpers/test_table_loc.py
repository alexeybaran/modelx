"""Pandas-spelled reads on prepared tables: Table.loc, Record.loc and index max/min."""
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


def test_grouped_full_key_matches_pandas(frames):
    _, grouped = frames
    table = pdn.table(grouped, columns=('premium_pp',), groups=grouped.index.names[:-1])
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
        assert outcome(lambda: table.index.asof(label)) == outcome(lambda: frame.index.asof(label)), label
