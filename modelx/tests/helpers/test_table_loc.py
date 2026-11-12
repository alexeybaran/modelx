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
