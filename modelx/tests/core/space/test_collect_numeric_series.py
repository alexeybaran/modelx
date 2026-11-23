import math
from threading import local
from types import SimpleNamespace

import pytest

from modelx.export._mx_collect_semantics import Accumulator, PackedFloatSeries, numeric_keys
from modelx.export import _mx_sys as runtime
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
    offsets = numeric_keys(range(-2, 4, 2), layouts)
    assert offsets == (-2, 0, 2)
    assert numeric_keys(range(-2, 4, 2), layouts) is offsets
    assert numeric_keys(range(2, -4, -2), layouts) == (2, 0, -2)
    for other in ([1, 3], (2, 2), (1, True), (1 << 64,), range(1 << 100)):
        assert numeric_keys(other, layouts) is None


def test_pending_series_reduces_before_publication_and_retries(monkeypatch):
    monkeypatch.setattr(runtime.sys, '_is_gil_enabled', lambda: False, raising=False)
    base = SimpleNamespace(_model=SimpleNamespace(_mx_collect_dispatch_state=local(),
                                                  _mx_collect_workers_override=2))
    cache = {}

    def evaluate(coords):
        pairs = [(0, float(sum(coords))), (1, 1.0)]
        if coords == (2,):
            pairs.reverse()
            return runtime.CollectResult({'sum': runtime.CollectResult({'values': dict(pairs)})})
        return runtime.CollectResult({'sum': runtime._PendingCollectSum({'values': packed(pairs)})})

    def collect(evaluator):
        return runtime.collect_from_space(base, (1, 2), 'point', (), (), cache=cache,
                                          evaluate_serial=evaluator, reuse_in_parallel=True,
                                          private_ordered=True)

    with pytest.raises(ValueError, match='structure changed'):
        collect(evaluate)
    assert cache == {}

    def compatible(coords):
        if coords == (2,):
            return runtime.CollectResult({'sum': runtime.CollectResult({'values': {0: 2.0, 1: 1.0}})})
        return evaluate(coords)

    result = collect(compatible)
    assert dict(result.sum.values) == {0: 3.0, 1: 2.0}
    assert result.sum.values(1) == 2.0
    assert collect(compatible) is result
