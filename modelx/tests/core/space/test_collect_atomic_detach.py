"""Exported collection detachment preserves mutable/custom ownership rules."""
import pytest
from modelx.export import _mx_sys


@pytest.mark.parametrize('value', [None, False, 0, -1, 1.25, -0.0, 'label', b'label', 1j])
def test_exact_immutable_values_need_no_copy(value, monkeypatch):
    def unexpected(value):
        raise AssertionError('Immutable value reached deepcopy')
    monkeypatch.setattr(_mx_sys, 'deepcopy', unexpected)
    assert _mx_sys._mx_collect_detach(value) is value


def test_nested_mutable_outputs_remain_detached():
    value = {'one': [1.0, {'two': [3]}]}
    result = _mx_sys._mx_collect_detach(value)
    value['one'][1]['two'].append(4)
    assert result == {'one': [1.0, {'two': [3]}]}
    assert result is not value
    assert result['one'] is not value['one']


def test_builtin_subclass_keeps_custom_copy_and_errors():
    class Custom(float):
        def __deepcopy__(self, memo):
            raise ValueError('custom copy')
    with pytest.raises(ValueError, match='custom copy'):
        _mx_sys._mx_collect_detach(Custom(1.25))


def test_mutable_custom_object_is_copied():
    class Box:
        def __init__(self):
            self.values = [1]
    source = Box()
    result = _mx_sys._mx_collect_detach(source)
    source.values.append(2)
    assert result.values == [1]
    assert result is not source
