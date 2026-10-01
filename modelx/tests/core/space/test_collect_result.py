"""Detached collection containers must survive copying and serialization."""
from copy import deepcopy
import pickle

import numpy as np
import pytest

from modelx.core.collect import CollectResult as LiveResult
from modelx.export._mx_sys import CollectResult as ExportResult


@pytest.mark.parametrize('cls', [LiveResult, ExportResult])
@pytest.mark.parametrize('copy', [deepcopy, lambda x: pickle.loads(pickle.dumps(x))])
def test_nested_collect_result_round_trip(cls, copy):
    original = cls({'sum': cls({'amount': 3.0}),
                    'individual': {1: cls({'values': np.array([1.0, 2.0]),
                                          'label': 'first', 'optional': None})}})
    restored = copy(original)
    assert restored.sum.amount == 3.0
    assert restored.individual[1].label == 'first'
    assert restored.individual[1].optional is None
    np.testing.assert_array_equal(restored.individual[1].values, [1.0, 2.0])
    restored.individual[1].values[0] = 99.0
    assert original.individual[1].values[0] == 1.0
    with pytest.raises(AttributeError):
        restored.missing


@pytest.mark.parametrize('cls', [LiveResult, ExportResult])
def test_uninitialized_result_attribute_does_not_recurse(cls):
    raw = cls.__new__(cls)
    with pytest.raises(AttributeError):
        getattr(raw, '__setstate__')
