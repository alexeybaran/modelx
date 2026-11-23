import math

import pytest

from modelx.export._mx_collect_semantics import Accumulator, PackedFloatSeries, numeric_keys
from modelx.core.collect import CollectResult


def add_values(total, row):
    for i, value in enumerate(row):
        total[i] += value


def packed(pairs):
    series = PackedFloatSeries(tuple(k for k, _ in pairs), add_values)
    for i, (_, value) in enumerate(pairs):
        series.values[i] = value
    return series


@pytest.mark.parametrize('forms', [(packed, dict), (dict, packed), (packed, packed)])
def test_numeric_series_match_ordered_dictionary_fold(forms):
    output = Accumulator(CollectResult)
    for i, make in enumerate(forms):
        output.add(i, {'sum': {'cash': make([(2, -0.0), (1, float(i))])}})
    result = output.result().sum.cash
    assert list(result) == [2, 1]
    assert math.copysign(1.0, result(2)) == -1
    assert result[1] == 1

    with pytest.raises(ValueError, match=r"\('sum', 'cash'\)"):
        output.add(2, {'sum': {'cash': packed([(1, 0.0), (2, 0.0)])}})


def test_numeric_domains_preserve_native_keys_and_decline_irregular_shapes():
    layouts = {}
    domain = (1, 3)
    assert numeric_keys(domain, layouts) is domain
    assert numeric_keys(domain, layouts) is domain
    assert numeric_keys(range(0), layouts) == ()
    for other in ([1, 3], (2, 2), (1, True), (1 << 64,), range(1 << 100)):
        assert numeric_keys(other, layouts) is None
