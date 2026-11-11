from copy import deepcopy
from enum import IntEnum
import pickle

import pytest

from modelx_helpers.immutable_inputs import Record, Table, freeze
from modelx_helpers import typed_inputs as h


class Kind(IntEnum):
    BASE = 4


def test_owned_records_identity_and_pickle():
    source = {'value': 2.0, 'kind': Kind.BASE, 'optional': None, 'nested': [1, 2]}
    record = Record(source)
    source['value'] = 99
    assert record['value'] == 2.0
    assert record['kind'] is Kind.BASE
    assert record['nested'] == (1, 2)
    assert deepcopy(record) is record
    assert pickle.loads(pickle.dumps(record)) == record
    with pytest.raises(TypeError):
        record['value'] = 3
    with pytest.raises(AttributeError):
        record._items = ()
    with pytest.raises(TypeError):
        record._index['value'] = 3


def test_record_enum_lookup_with_an_optional_default():
    record = Record({'valid': 'BASE', 'unknown': 'OTHER', 'invalid': []},
                    preserve_scalar_types=True)
    assert record.lookup_enum('valid', Kind) is Kind.BASE
    for key in ('unknown', 'invalid', 'missing'):
        assert record.lookup_enum(key, Kind, default=None) is None
    with pytest.raises(KeyError):
        record.lookup_enum('unknown', Kind)
    with pytest.raises(TypeError):
        record.lookup_enum('invalid', Kind)


def test_keyed_table_order_defaults_and_errors():
    table = Table([12, -3, 7], [[2.0, None], [1.0, 'x'], [3.0, 'y']], ['rate', 'tag'])
    assert [h.key_at(table, i) for i in range(h.key_count(table))] == [12, -3, 7]
    assert h.lookup(table, -3, 'tag') == 'x'
    assert h.lookup(table, 12, 'tag', 'missing') is None
    assert h.lookup(table, 11, 'rate', 9.0) == 9.0
    assert h.previous(table, 11, 'rate') == 3.0
    assert h.previous(table, -4, 'rate', 0.0) == 0.0
    assert h.first_key(table) == -3 and h.last_key(table) == 12
    assert h.contains(table, 7) and not h.contains(table, 0)
    assert 7 in table.index and 0 not in table.index
    assert list(table.index) == [12, -3, 7]
    assert table.index.max_int() == 12
    assert type(table.index.max_float()) is float and table.index.max_float() == 12.0
    with pytest.raises(KeyError) as error:
        h.lookup(table, 11, 'rate')
    assert error.value.args == (11,)
    with pytest.raises(ValueError, match='No key'):
        h.previous(table, -4, 'rate')
    with pytest.raises(IndexError):
        h.key_at(table, -1)
    with pytest.raises(IndexError):
        h.key_at(table, 3)
    with pytest.raises(ValueError, match='duplicate'):
        Table([1, 1], [[1.0], [2.0]], ['value'])
    restored = pickle.loads(pickle.dumps(table))
    assert dict(restored) == dict(table)
    assert deepcopy(table) is table


def test_nested_groups_and_raw_helpers_are_unchanged():
    raw = {'keys': [1, 4], 'values': [[2.0], [8.0]], 'columns': ('rate',)}
    group = freeze({'fund-b': raw, 'fund-a': raw})
    nested = Record({'plan': group})
    assert nested.slice('plan', 'fund-b') is group['fund-b']
    assert nested.slice('plan') is group
    with pytest.raises(KeyError, match='absent'):
        nested.slice('plan', 'absent')
    assert tuple(group) == ('fund-b', 'fund-a')
    for table in (raw, group['fund-b']):
        assert h.lookup(table, 4, 'rate') == 8.0
        assert h.previous(table, 3, 'rate') == 2.0
    raw['values'][0][0] = 999
    assert h.lookup(group['fund-b'], 1, 'rate') == 2.0
