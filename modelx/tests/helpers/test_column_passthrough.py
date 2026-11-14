"""Pandas passthrough of a table column and a Record (round 3, D35 (A)).

A column (``table["col"]``) answers Series operators and attributes outside its vocabulary from a
pandas Series copy; ``==`` and ``!=`` keep the Mapping equality of the column (pinned). A Record
gives its field names as ``.index``, as a pandas row does. pandas is the oracle.
"""
import operator

import pandas as pd
import pytest

from modelx_helpers import pdn
from modelx_helpers.immutable_inputs import Record


@pytest.fixture
def frames():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0], "s": ["x1", "y2", "x3"],
                          "f": [True, False, True]}, index=pd.Index([10, 11, 12], name="k"))
    return frame, pdn.table(frame)


@pytest.mark.parametrize("op", [operator.lt, operator.le, operator.gt, operator.ge, operator.add,
                                operator.sub, operator.mul, operator.truediv, operator.floordiv,
                                operator.mod, operator.pow])
def test_column_operators_follow_pandas(frames, op):
    frame, table = frames
    pd.testing.assert_series_equal(op(table["a"], 2.0), op(frame["a"], 2.0))
    pd.testing.assert_series_equal(op(2.0, table["a"]), op(2.0, frame["a"]))
    pd.testing.assert_series_equal(op(table["a"], table["b"]), op(frame["a"], frame["b"]))


def test_column_logical_and_unary_operators(frames):
    frame, table = frames
    pd.testing.assert_series_equal(table["f"] & (table["a"] > 1), frame["f"] & (frame["a"] > 1))
    pd.testing.assert_series_equal(table["f"] | (table["a"] > 2), frame["f"] | (frame["a"] > 2))
    pd.testing.assert_series_equal(table["f"] ^ True, frame["f"] ^ True)
    pd.testing.assert_series_equal(~table["f"], ~frame["f"])
    pd.testing.assert_series_equal(-table["a"], -frame["a"])
    pd.testing.assert_series_equal(abs(table["a"]), abs(frame["a"]))


def test_column_attributes_follow_pandas(frames):
    frame, table = frames
    assert table["a"].sum() == frame["a"].sum()
    assert (table["a"] > 0).all() and not (table["a"] > 2).all()
    assert table["s"].str.startswith("x").tolist() == [True, False, True]
    assert table["a"].between(1.5, 3.0).tolist() == [False, True, True]
    assert table["a"].loc[11] == 2.0
    assert table["a"].min() == 1.0 and table["a"].max() == 3.0


def test_column_vocabulary_and_equality_are_unchanged(frames):
    _, table = frames
    column = table["a"]
    assert column[11] == 2.0 and 11 in column and list(column) == [1.0, 2.0, 3.0] and len(column) == 3
    assert "__eq__" not in vars(type(column))       # == / != keep the Mapping equality (pinned, D35)
    with pytest.raises(AttributeError):
        column._private


def test_passthrough_does_not_change_the_table(frames):
    _, table = frames
    series = table["a"].sort_values(ascending=False)
    series.iloc[0] = -1.0
    assert table.loc[12, "a"] == 3.0
    assert list(table["a"]) == [1.0, 2.0, 3.0]


def test_record_index_is_the_field_names(frames):
    frame, table = frames
    row = table.loc[11]
    assert isinstance(row, Record)
    assert list(row.index) == list(frame.loc[11].index)
    assert list(Record({"x": 1, "y": 2}).index) == ["x", "y"]
