"""Storage form x operation x key type matrix of the target-syntax read vocabulary.

pandas is the oracle: each read is run on the prepared Table and on the DataFrame it was
prepared from, and the outcomes (value, or exception type and arguments) must agree.
Deliberate differences are asserted explicitly where they occur:
  * ``index.asof`` below the first label raises KeyError(label); pandas returns NaN (D25a);
  * a missing complete key in the row form ``T.loc[(a, b)]`` raises KeyError((a, b)); pandas
    reads the tuple as (row, column) and raises KeyError(b) (D27b);
  * a key that is not a row label raises KeyError(key) even where pandas raises another
    error (out-of-range integers, too many key parts);
  * values are Python scalars where pandas returns numpy scalars (compared by value).
Compact forms (contiguous, rectangular and their sub-tables) are also checked for
non-expansion: no read may set the lazy ``_keys``/``_values``/``_positions``/``_ordered``.

Forms (design section 4): C contiguous (negative first key), R rectangular (unsorted
axes), F1 float rows with a sparse int key, F2 float rows on three levels (int-coded enum,
str, int), G mixed columns, S sub-tables (of R: rectangular, contiguous and float-row
children; of F2), O one-row tables.
"""
import enum
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from modelx_helpers import pdn
from modelx_helpers.immutable_inputs import Record, Table, table_schema, table_storage

pytestmark = pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')

LAZY = ('_keys', '_values', '_positions', '_ordered')


class Sex(enum.IntEnum):
    F = 0
    M = 1


# ---------------------------------------------------------------- forms

def _frame_c():
    return pd.DataFrame({'q': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]}, index=pd.Index(range(-3, 4), name='k'))


def _frame_r():
    index = pd.MultiIndex.from_product([[2, 1], ['b', 'a', 'c']], names=['p', 's'])
    return pd.DataFrame({'q': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]}, index=index)


def _frame_r3():
    index = pd.MultiIndex.from_product([[1, 0], ['y', 'x'], [5, 6, 7]], names=['a', 'b', 'c'])
    return pd.DataFrame({'q': np.arange(12.0) / 8}, index=index)


def _frame_f1():
    return pd.DataFrame({'a': [0.1, 0.3, 0.7, 2.0], 'b': [1.0, 3.0, 7.0, 20.0]},
                        index=pd.Index([1, 3, 7, 20], name='k'))


def _frame_f2():
    index = pd.MultiIndex.from_tuples([(0, 'x', 1), (0, 'x', 2), (0, 'y', 1), (1, 'x', 3), (1, 'y', 1),
                                       (1, 'y', 2)], names=['sex', 'cls', 'year'])
    return pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'b': [0.5, 0.25, 0.125, 1.5, 2.5, 3.5]},
                        index=index)


def _frame_g():
    return pd.DataFrame({'f': [0.25, 0.5, 0.75], 'n': [3, 2, 1], 'flag': [True, False, True],
                         's': ['MALE', 'FEMALE', 'MALE']}, index=pd.Index([10, 20, 30], name='k'))


def _frame_o_c():
    return pd.DataFrame({'q': [0.125]}, index=pd.Index([5], name='k'))


def _frame_o_r():
    return pd.DataFrame({'q': [9.5]}, index=pd.MultiIndex.from_product([[7], ['x']]))


def _frame_o_g():
    return pd.DataFrame({'f': [1.25], 'n': [4], 's': ['A']}, index=pd.Index([2], name='k'))


def _sub(make, prefix):
    def build():
        frame, table = make()
        return frame.loc[prefix], table.loc[prefix]
    return build


def _plain(frame_fn):
    def build():
        frame = frame_fn()
        return frame, pdn.table(frame)
    return build


FORMS = {
    # name: (builder, expected storage, compact)
    'C': (_plain(_frame_c), 'contiguous', True),
    'R': (_plain(_frame_r), 'rectangular', True),
    'F1': (_plain(_frame_f1), 'rows', False),
    'F2': (_plain(_frame_f2), 'rows', False),
    'G': (_plain(_frame_g), 'rows', False),
    'S_R3': (_sub(_plain(_frame_r3), 1), 'rectangular', True),          # 2 levels left: R child
    'S_R3C': (_sub(_plain(_frame_r3), (0, 'x')), 'contiguous', True),   # int run left: C child
    'S_RF': (_sub(_plain(_frame_r), 2), 'rows', False),                 # str level left: float rows
    'S_F2': (_sub(_plain(_frame_f2), 1), 'rows', False),                # 2 levels left
    'S_F2F': (_sub(_plain(_frame_f2), (0, 'x')), 'rows', False),        # 1 level left
    'O_C': (_plain(_frame_o_c), 'contiguous', True),
    'O_R': (_plain(_frame_o_r), 'rectangular', True),
    'O_G': (_plain(_frame_o_g), 'rows', False),
}

MULTI = {'R', 'F2', 'S_R3', 'S_F2', 'O_R'}
SINGLE = set(FORMS) - MULTI
NUMERIC_EXTRA = {'G', 'O_G'}    # forms with int/bool/str columns

OPS = {
    1: 'loc[k, "c"]', 2: 'at[k, "c"]', 3: 'int/bool column', 4: 'k in T.index',
    5: 'index.min/max and clamp', 6: 'loc[prefix] then reads', 7: 'row = loc[k]; row["c"]',
    8: 'asof', 9: 'searchsorted / index[i]', 10: 'errors', 11: 'record field reads',
    12: 'T["c"][k]',
}


def _square(form, op):
    """'test' or the reason the square does not apply."""
    if op == 3 and form not in NUMERIC_EXTRA:
        return 'n/a: float columns only'
    if op == 6 and form not in MULTI:
        return 'n/a: single-level index has no prefix'
    if op == 8 and form in MULTI:
        return 'n/a: asof on tuple labels is the pandas passthrough, outside the vocabulary'
    if op == 11 and form not in NUMERIC_EXTRA:
        return 'n/a: record fields of mixed rows are read on G forms'
    return 'test'


SQUARES = {(form, op): _square(form, op) for form in FORMS for op in OPS}
TESTED = sorted(square for square, state in SQUARES.items() if state == 'test')


def test_matrix_is_complete():
    """Every (form, operation) square is tested or carries a reason."""
    assert set(SQUARES) == {(form, op) for form in FORMS for op in OPS}
    for square, state in SQUARES.items():
        assert state == 'test' or state.startswith('n/a: '), square
    # the live-helper squares of design section 4 are all present
    assert {form[0] for form in FORMS} == {'C', 'R', 'F', 'G', 'S', 'O'}
    assert len(TESTED) > 100


# ---------------------------------------------------------------- oracle helpers

def _norm(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return tuple(_norm(part) for part in value)
    if isinstance(value, float) and value != value:
        return 'nan'
    return value


def outcome(read):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return ('ok', _norm(read()))
    except Exception as exc:     # noqa: BLE001 - compare the exception types and arguments
        return (type(exc).__name__, tuple(_norm(arg) for arg in exc.args))


def _is_set(table, name):
    try:
        Table.__dict__[name].__get__(table)
    except AttributeError:
        return False
    return True


def _build(form):
    builder, storage, compact = FORMS[form]
    frame, table = builder()
    assert table_storage(table)[0] == storage, form
    return frame, table, compact


def _labels(frame):
    return list(frame.index)


def _levels(frame):
    return frame.index.nlevels


def _keys_for(frame):
    """Labels, misses and the key-type axis (int, enum, str, tuple of mixed parts, numpy
    int64/uint64/float32/float64, bool, float on an int index)."""
    labels = _labels(frame)
    if _levels(frame) == 1 and type(labels[0]) is str:
        return labels + ['zz', '', 0, True, None, float('nan'), np.str_(labels[0])]
    if _levels(frame) == 1:
        keys = labels + [labels[0] - 1, labels[-1] + 1, 'x', True, False, None, float('nan')]
        first = labels[0]
        keys += [float(first), first + 0.5, np.int64(first), np.float32(first), np.float64(first),
                 np.float64(first + 0.5), np.bool_(True)]
        if first >= 0:
            keys.append(np.uint64(first))
        if first in (0, 1):
            keys.append(Sex(first))
        return keys
    keys = list(labels)
    first = labels[0]
    keys.append(tuple(np.int64(part) if type(part) is int else part for part in first))
    keys.append(tuple(np.float64(part) if type(part) is int else part for part in first))
    if type(first[0]) is int and first[0] in (0, 1):
        keys.append((Sex(first[0]),) + tuple(first[1:]))
    keys.append(tuple('zz' if type(part) is str else part + 100 for part in first))
    keys.append((True,) + tuple(first[1:]))
    keys.append(first[:-1] + ('zz',))
    return keys


def _first_column(frame):
    return frame.columns[0]


def _check_not_expanded(table, compact):
    """Contiguous tables own their key tuple; everything else of a compact table is lazy."""
    if compact:
        kind = table_storage(table)[0]
        for name in LAZY if kind == 'rectangular' else ('_values', '_positions'):
            assert not _is_set(table, name), name


# ---------------------------------------------------------------- squares

@pytest.mark.parametrize('form, op', TESTED)
def test_square(form, op):
    frame, table, compact = _build(form)
    run = globals()[f'_op{op}']
    run(frame, table)
    _check_not_expanded(table, compact)


def _row_form_difference(table, key):
    """D27b: a complete tuple key raises KeyError(key) in the helper."""
    with pytest.raises(KeyError) as error:
        table.loc[key]
    assert error.value.args == (key,)


def _op1(frame, table):
    for column in list(frame.columns) + ['nope']:
        for key in _keys_for(frame):
            got = outcome(lambda: table.loc[key, column])
            expected = outcome(lambda: frame.loc[key, column])
            if expected[0] != 'ok' and got[0] == 'KeyError' and expected[0] != 'KeyError':
                assert got == ('KeyError', (_norm(key),)), (key, column, expected)
                continue
            if _levels(frame) > 1 and expected[0] == 'ok' and isinstance(expected[1], pd.Series):
                continue        # a non-complete key: pandas returns a Series, see op 6
            assert got == expected, (key, column)
            if got[0] == 'ok' and column in frame.columns:
                assert type(table.loc[key, column]) is type(_norm(frame.loc[key, column]))


def _op2(frame, table):
    for column in list(frame.columns) + ['nope']:
        for key in _keys_for(frame):
            got = outcome(lambda: table.at[key, column])
            expected = outcome(lambda: frame.at[key, column])
            if expected[0] != 'ok' and expected[0] != 'KeyError':
                assert got == ('KeyError', (_norm(key),)), (key, column, expected)
                continue
            assert got == expected, (key, column)


def _op3(frame, table):
    for key in _labels(frame):
        assert int(table.loc[key, 'n']) == int(frame.loc[key, 'n'])
        assert type(table.loc[key, 'n']) is int
        if 'flag' in frame.columns:
            assert bool(table.loc[key, 'flag']) is bool(frame.loc[key, 'flag'])
            assert type(table.loc[key, 'flag']) is bool
        assert table.loc[key, 's'] == frame.loc[key, 's']
    schema = table_schema(table)
    assert schema['columns']['n'] == 'int' and schema['columns']['s'] == 'str'


def _op4(frame, table):
    keys = _keys_for(frame)
    if _levels(frame) > 1:
        labels = _labels(frame)
        keys += [labels[0][:1], labels[0][0], labels[-1][:-1], ('zz',), labels[0][:1] + ('zz',)]
    for key in keys:
        got = outcome(lambda: key in table.index)
        expected = outcome(lambda: key in frame.index)
        if expected[0] != 'ok':
            assert got == ('ok', False), key      # pandas raises on unhashable/odd keys
            continue
        assert got == expected, key
    assert len(table.index) == len(frame.index)
    assert [_norm(k) for k in table.index] == [_norm(k) for k in frame.index]


def _op5(frame, table):
    assert _norm(table.index.min()) == _norm(frame.index.min())
    assert _norm(table.index.max()) == _norm(frame.index.max())
    if _levels(frame) == 1 and type(_labels(frame)[0]) is int:
        lo, hi = int(table.index.min()), int(table.index.max())
        column = _first_column(frame)
        for x in range(lo - 3, hi + 4):
            clamped = min(max(x, int(table.index.min())), int(table.index.max()))
            assert clamped == min(max(x, int(frame.index.min())), int(frame.index.max()))
            assert outcome(lambda: table.loc[clamped, column]) == outcome(lambda: frame.loc[clamped, column])
        assert table.index.max_int() == int(frame.index.max())


def _op6(frame, table):
    labels = _labels(frame)
    prefixes = {label[:n] for label in labels for n in range(1, _levels(frame))}
    prefixes |= {p[0] for p in prefixes if len(p) == 1}
    prefixes |= {('zz',), labels[0][:1] + ('zz',), (True,)}
    first = labels[0][0]
    if type(first) is int:
        prefixes |= {np.int64(first), float(first)}
        if first in (0, 1):
            prefixes.add(Sex(first))
    for prefix in prefixes:
        expected = outcome(lambda: frame.loc[prefix])
        got = outcome(lambda: table.loc[prefix])
        if expected[0] != 'ok':
            # KeyError(prefix as passed); pandas unwraps a 1-tuple or names one part
            assert got == ('KeyError', (_norm(prefix),)), prefix
            continue
        sub, want = table.loc[prefix], frame.loc[prefix]
        assert sub is table.loc[prefix]                     # per-prefix memo: shared child
        assert type(sub) is Table
        assert [_norm(k) for k in sub.index] == [_norm(k) for k in want.index]
        for key in want.index:
            for column in want.columns:
                assert sub.loc[key, column] == want.loc[key, column]
        assert _norm(sub.index.max()) == _norm(want.index.max())
        storage = table_storage(table)
        if storage[0] == 'rectangular':
            child = table_storage(sub)
            assert child[0] in ('rectangular', 'contiguous', 'rows')
            if child[0] != 'rows':
                assert np.shares_memory(child[2], storage[2])
            if child[0] != 'rows':
                _check_not_expanded(sub, True)


def _op7(frame, table):
    row = None
    for key in _labels(frame):
        row = table.loc[key]                                # reassignment of the row local
        want = frame.loc[key]
        assert type(row) is Record
        assert list(row) == list(want.index)
        for column in frame.columns:
            assert row[column] == _norm(want[column])
    with pytest.raises(KeyError) as error:
        row['nope']
    assert error.value.args == ('nope',)
    if _levels(frame) == 1:
        missing = _miss(_labels(frame)[-1])
        with pytest.raises(KeyError) as error:
            table.loc[missing]
        assert error.value.args == (missing,)
    else:
        missing = tuple('zz' if type(p) is str else p + 100 for p in _labels(frame)[0])
        _row_form_difference(table, missing)


def _miss(label):
    return label + '0' if type(label) is str else label + 100


def _op8(frame, table):
    labels = _labels(frame)
    column = _first_column(frame)
    if type(labels[0]) is str:
        probes = labels + [label + '0' for label in labels] + ['', 'zz', 0]
    else:
        probes = [labels[0] - 1, labels[0] - 0.5] + labels + [label + 0.5 for label in labels]
        probes += [labels[-1] + 10, np.int64(labels[0]), np.float64(labels[-1] + 0.5)]
    for x in probes:
        expected = outcome(lambda: frame.index.asof(x))
        if expected == ('ok', 'nan'):
            with pytest.raises(KeyError) as error:           # D25a
                table.index.asof(x)
            assert error.value.args == (x,)
            continue
        assert outcome(lambda: table.index.asof(x)) == expected, x
        if expected[0] != 'ok':
            continue
        k = table.index.asof(x)
        assert table.loc[k, column] == frame.loc[frame.index.asof(x), column]     # as a row key
        if type(k) is int:
            assert 2.0 * table.index.asof(x) + 1 == 2.0 * frame.index.asof(x) + 1  # as a value


def _op9(frame, table):
    n = len(frame.index)
    for i in list(range(-n - 2, n + 2)) + [np.int64(0), np.intp(-1)]:
        assert outcome(lambda: table.index[i]) == outcome(lambda: frame.index[i]), i
    if _levels(frame) > 1 or not frame.index.is_monotonic_increasing:
        return      # searchsorted presumes sorted labels
    labels = _labels(frame)
    if type(labels[0]) is str:
        probes = [''] + labels + [label + '0' for label in labels]
    else:
        probes = [labels[0] - 1] + labels + [label + 0.5 for label in labels] + [labels[-1] + 1]
    for x in probes:
        for side in ('left', 'right'):
            got = table.index.searchsorted(x, side=side)
            assert got == frame.index.searchsorted(x, side=side), (x, side)
            assert type(got) is int
        # successor read (D25b)
        expected = outcome(lambda: frame.index[frame.index.searchsorted(x, side='right')])
        assert outcome(lambda: table.index[table.index.searchsorted(x, side='right')]) == expected
    with pytest.raises(ValueError):
        table.index.searchsorted(labels[0], side='middle')


def _op10(frame, table):
    labels = _labels(frame)
    column = _first_column(frame)
    with pytest.raises(KeyError) as error:
        table.loc[labels[0], 'nope']
    assert error.value.args == ('nope',)
    if _levels(frame) > 1:
        missing = tuple('zz' if type(p) is str else p + 100 for p in labels[0])
        with pytest.raises(KeyError) as error:
            table.loc[missing, column]
        assert error.value.args == (missing,)
        with pytest.raises(KeyError):
            table.loc[labels[0] + (1,), column]               # too many parts
        return
    misses = (2 ** 63, -2 ** 63, 2 ** 63 - 1, -2 ** 63 - 1, 2 ** 64, np.uint64(2 ** 64 - 1),
              np.int64(-2 ** 63), np.int64(2 ** 63 - 1), 'x', True)
    if type(labels[0]) is int:
        misses = (labels[0] - 1, labels[-1] + 1) + misses
    for key in misses:
        if key in frame.index:
            continue
        with pytest.raises(KeyError) as error:
            table.loc[key, column]
        assert error.value.args[0] is key
        assert key not in table.index
    if type(labels[0]) is int:
        with pytest.raises(KeyError) as error:
            table.index.asof(labels[0] - 1)                   # below the first key (D25a)
        assert error.value.args == (labels[0] - 1,)
    with pytest.raises(TypeError):
        table.loc[labels[0]:labels[-1]]


def _op11(frame, table):
    for key in _labels(frame):
        row = table.loc[key]
        want = frame.loc[key]
        assert float(row['f']) * 2 + 1 == float(want['f']) * 2 + 1
        assert int(row['n']) == int(want['n'])
        assert (row['s'] == 'A' or row['s'] == 'MALE') == (want['s'] in ('A', 'MALE'))
        assert dict(row) == {column: table.loc[key, column] for column in frame.columns}
        assert row is not None and Record.__getitem__(row, 'f') == table.at[key, 'f']


def _op12(frame, table):
    for column in frame.columns:
        got, want = table[column], frame[column]
        assert [_norm(v) for v in got] == [_norm(v) for v in want]
        assert len(got) == len(want)
        for key in _keys_for(frame):
            expected = outcome(lambda: want[key])
            if expected[0] == 'ok' and isinstance(want[key], pd.Series):
                continue
            if expected[0] != 'ok' or (_levels(frame) == 1 and key not in want.index):
                assert outcome(lambda: got[key])[0] == 'KeyError', key
                continue
            assert outcome(lambda: got[key]) == expected, key
            assert key in got
    with pytest.raises(KeyError) as error:
        table['nope']
    assert error.value.args == ('nope',)


# ---------------------------------------------------------------- contracts and passthrough

@pytest.mark.parametrize('form', sorted(FORMS))
def test_contracts_do_not_expand(form):
    frame, table, compact = _build(form)
    storage = table_storage(table)
    assert storage[-2] == tuple(frame.columns)
    schema = table_schema(table)
    assert schema['storage'] == storage[0]
    assert schema['rows'] == len(frame)
    assert len(schema['keys']) == _levels(frame)
    assert list(schema['columns']) == list(frame.columns)
    if storage[0] == 'contiguous':
        _, start, data, _, _ = storage
        assert start == frame.index[0] and data.shape == (len(frame), len(frame.columns))
        assert not data.flags.writeable
    elif storage[0] == 'rectangular':
        _, axes, data, _, _ = storage
        assert len(axes) == _levels(frame) and data.ndim == 1
    else:
        _, keys, values, ordered, _, _ = storage
        assert len(keys) == len(values) == len(frame)
    _check_not_expanded(table, compact)


def test_schema_kinds():
    schema = table_schema(pdn.table(_frame_g()))
    assert schema == {'storage': 'rows', 'rows': 3, 'keys': ('int',),
                      'columns': {'f': 'float', 'n': 'int', 'flag': 'bool', 's': 'str'}}
    schema = table_schema(pdn.table(_frame_f2()))
    assert schema['keys'] == ('int', 'str', 'int')
    # enum labels kept by a Table built from Python keys
    table = Table([(Sex.F, 'x'), (Sex.M, 'x')], [(1.0,), (2.0,)], ['v'])
    assert table_schema(table)['keys'][0] == f'enum:{Sex.__module__}.{Sex.__qualname__}'
    # pandas turns IntEnum labels into int64, so prepared tables report 'int' (D27)
    frame = pd.DataFrame({'v': [1.0, 2.0]}, index=pd.Index([Sex.F, Sex.M]))
    assert table_schema(pdn.table(frame))['keys'] == ('int',)
    with pytest.raises(TypeError):
        table_storage(frame)


def test_passthrough_and_private_names():
    frame = _frame_r()
    table = pdn.table(frame)
    with pytest.raises(AttributeError):
        table._nope
    with pytest.raises(AttributeError):
        table.index._nope
    assert table.shape == frame.shape
    assert list(table.columns) == list(frame.columns)
    assert table.iloc[-1]['q'] == frame.iloc[-1]['q']
    assert table.index.names == frame.index.names
    shown = table.iloc[:, 0]
    with pytest.raises(ValueError):
        shown.values[0] = -1.0          # copy-on-write view of the read-only storage
    assert table.loc[(2, 'b'), 'q'] == 10.0


@pytest.mark.parametrize('form', sorted(FORMS))
def test_iterrows(form):
    frame, table, compact = _build(form)
    got = list(table.iterrows())
    assert [_norm(k) for k, _ in got] == [_norm(k) for k in frame.index]
    for (_, row), (_, want) in zip(got, frame.iterrows()):
        assert type(row) is Record
        assert all(row[c] == want[c] for c in frame.columns)
    assert all(a is b for (_, a), (_, b) in zip(got, table.iterrows()))
    _check_not_expanded(table, compact)


def test_legacy_vocabulary_unchanged():
    """Round-2 helper names keep working on the same tables (until milestone 4)."""
    frame = _frame_f1()
    table = pdn.table(frame)
    assert table.lookup_float(3, 'a') == 0.3
    assert table.lookup(7, 'b') == 7.0
    assert table.contains(20) and not table.contains(2)
    assert table.previous(5, 'a') == 0.3
    assert pdn.lookup_float(table, 3, 'a') == 0.3
    grouped = pdn.table(_frame_f2(), groups=['sex'])
    assert type(grouped) is Record
    rect = pdn.table(_frame_r())
    assert rect.lookup((1, 'c'), 'q') == 15.0
    assert (1, 'c') in rect.index and 1 in rect.index and (1,) in rect.index
    assert not rect.contains(1)               # Table.contains keeps complete-key semantics (D27a)
