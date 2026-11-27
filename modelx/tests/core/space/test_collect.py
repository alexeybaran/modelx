import copy
import pickle

import numpy as np
import pytest

import modelx as mx
from modelx.core.errors import FormulaError
from modelx.core.space import UserSpaceImpl
from modelx.core.execution.trace import TraceGraph


def outputs():
    return {'individual': {'pv': pv(), 'cf': {t: cf(t) for t in range(T)}},
            'sum': {'pv': pv(), 'cf': {t: cf(t) for t in range(T)},
                    'arr': np.array([cf(t) for t in range(T)])}}


@pytest.fixture
def model():
    m = mx.new_model()
    P = m.new_space('P', formula=lambda i, scen=1: None)
    P.T = 4
    P.rate = 0.03
    P.np = np
    P.new_cells('ids', lambda: tuple(range(1, 8)))

    @mx.defcells(P)
    def cf(t):
        return (i % 3 + 1) * scen * 1.07 ** t / 3

    @mx.defcells(P)
    def pv():
        return sum(cf(t) / (1 + rate) ** t for t in range(T))

    P.set_collect(over={'i': P.ids}, outputs=outputs)
    yield m
    m._impl._check_sanity()
    m.close()


def loop(P, *args, delete=True):
    args = args or (1,)
    """collect() written by hand."""
    individual, total = {}, None
    for i in P.ids():
        created = (i, *args) not in P._impl.param_spaces
        item = P(i, *args)
        out = item.collect_outputs()
        individual[i] = out['individual']
        s = out['sum']
        if total is None:
            total = {'pv': s['pv'], 'cf': dict(s['cf']), 'arr': s['arr'].copy()}
        else:
            total['pv'] = total['pv'] + s['pv']
            for t in total['cf']:
                total['cf'][t] = total['cf'][t] + s['cf'][t]
            total['arr'] = total['arr'] + s['arr']
        if delete and created:
            del P[(i, *args)]
    return individual, total


def assert_same(result, expected):
    individual, total = expected
    assert list(result.individual) == list(individual)
    for i, values in individual.items():
        assert result.individual[i].pv == values['pv']
        assert dict(result.individual[i].cf) == values['cf']
    assert result.sum.pv == total['pv']
    assert dict(result.sum.cf) == total['cf']
    assert result.sum.arr.tobytes() == total['arr'].tobytes()


@pytest.mark.parametrize("args, kwargs, scen", [
    ((), {}, 1), ((2,), {}, 2), ((), {'scen': 2}, 2)])
def test_collect_equals_a_loop_bit_for_bit(model, args, kwargs, scen):
    P = model.P
    result = P.collect(*args, **kwargs)
    assert not P.itemspaces
    expected = loop(P, scen)
    assert_same(result, expected)
    # Same values when the ItemSpaces exist, which are then kept
    loop(P, scen, delete=False)
    items = dict(P._impl.param_spaces)
    assert_same(P.collect(*args, **kwargs), expected)
    assert dict(P._impl.param_spaces) == items


def test_result_sections_and_calls(model):
    r = model.P.collect()
    assert r.sum.cf(2) == r.sum.cf[2]
    assert r.individual[3].cf(1) == r.individual[3].cf[1]
    assert sorted(dir(r)) == sorted(set(object.__dir__(r)) | {'individual', 'sum'})
    with pytest.raises(AttributeError):
        r.other
    with pytest.raises(KeyError):
        r.sum.cf(99)


def test_result_is_cached_and_kept_by_defaults(model):
    P = model.P
    r = P.collect()
    assert P.collect() is r
    assert P.collect(1) is r
    assert P.collect(scen=1) is r
    assert P.collect(2) is not r


@pytest.mark.parametrize("copier", [
    lambda r: pickle.loads(pickle.dumps(r)), copy.copy, copy.deepcopy])
def test_result_pickles_and_copies(model, copier):
    r = model.P.collect()
    c = copier(r)
    assert c.sum.pv == r.sum.pv
    assert c.individual[2].cf(3) == r.individual[2].cf(3)
    assert c.sum.arr.tobytes() == r.sum.arr.tobytes()


def test_existing_itemspaces_keep_their_own_cache(model, monkeypatch):
    P = model.P
    P.seen = seen = []
    P.pv.formula = lambda: seen.append(i) or sum(cf(t) / (1 + rate) ** t for t in range(T))
    P.new_cells('outside', lambda: i * 10)
    items = [P[i] for i in P.ids()]
    for item in items:
        item.pv()
        item.outside()
    seen.clear()
    calls = []
    original = TraceGraph.__iter__

    def no_graph_scan(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(TraceGraph, '__iter__', no_graph_scan)
    pv_values = [item.pv() for item in items]
    r = P.collect()
    assert not calls
    assert [r.individual[i].pv for i in P.ids()] == pv_values
    assert seen == list(P.ids())
    assert [P[i] for i in P.ids()] == items
    assert all(len(item.collect_outputs) == 0 for item in items)
    assert [item.outside() for item in items] == [i * 10 for i in P.ids()]


def test_collect_never_publishes_private_itemspaces(model):
    P = model.P
    P.seen = seen = []
    P.collect_outputs.formula = lambda: (
        seen.append(len(_space.parent._impl.param_spaces)) or {'sum': {'pv': pv()}})
    P.collect()
    assert seen == [0] * 7
    assert not P.itemspaces


def test_collect_follows_new_updated_and_removed_nested_inputs(model):
    P = model.P
    Child = P.new_space('Child', formula=lambda j: None)
    Child.new_cells('amount', lambda t: i + j + t)
    P.collect_outputs.formula = lambda: {'sum': {'amount': Child[10].amount(1)}}
    child = P[1].Child
    nested = child[10]
    assert nested.amount(1) == 12
    assert P.collect().sum.amount == 105

    nested.amount[1] = 100
    assert P.collect().sum.amount == 193
    nested.amount[1] = 200
    assert P.collect().sum.amount == 293
    nested.amount.clear_at(1)
    assert P.collect().sum.amount == 105
    assert child is P[1].Child
    assert nested is P[1].Child[10]
    assert not len(nested.amount)


def test_collect_can_call_another_collect_with_durable_arguments(model):
    P = model.P
    Q = model.new_space('Q', formula=lambda j, amount: None, refs={'keys': (1, 2)})
    Q.new_cells('value', lambda: j * amount)
    Q.set_collect(over={'j': 'keys'}, outputs=lambda: {'sum': {'value': value()}})
    P.Q = Q
    P.collect_outputs.formula = lambda: {'sum': {'value': Q.collect(i).sum.value}}
    assert P.collect().sum.value == 84
    Q.value.formula = lambda: 2 * j * amount
    assert P.collect().sum.value == 168
    assert not P.itemspaces
    assert not Q.itemspaces


@pytest.mark.parametrize('warm_private', [False, True])
@pytest.mark.parametrize('nested_collect', [False, True])
def test_cached_external_calls_cannot_read_a_private_item(model, warm_private, nested_collect):
    P = model.P
    Q = model.new_space('Q', formula=lambda j: None, refs={'keys': (1,)})
    P.Q = Q
    P.warm_private = warm_private
    P.bridge = Q.bridge = {}
    P.new_cells('value', lambda: i)
    Q.new_cells('read', lambda: bridge['callback']())
    if nested_collect:
        Q.set_collect(over={'j': 'keys'}, outputs=lambda: {'sum': {'value': read()}})
        P.new_cells('external', lambda: Q.collect().sum.value)
    else:
        P.new_cells('external', lambda: Q.read())

    def checked_outputs():
        if warm_private:
            value()
        bridge['callback'] = value
        try:
            return {'sum': {'value': external()}}
        finally:
            del bridge['callback']

    P.collect_outputs.formula = checked_outputs
    with pytest.raises(FormulaError, match='cached external formula cannot read a temporary Collect ItemSpace'):
        P.collect()
    assert len(Q.read) == 0
    assert not P.itemspaces
    assert not Q.itemspaces


def test_cached_external_calls_cannot_store_private_item_keys(model):
    P = model.P
    Q = model.new_space('Q')
    P.Q = Q
    Q.new_cells('read', lambda item: item.cf(0))
    P.collect_outputs.formula = lambda: {'sum': {'value': Q.read(_space)}}
    with pytest.raises(FormulaError, match='cached external formula cannot take a temporary Collect ItemSpace'):
        P.collect()
    assert len(Q.read) == 0
    assert Q.read(P[1]) == P[1].cf(0)


@pytest.mark.parametrize("edit", [
    lambda P, m: setattr(P, 'rate', 0.05),
    lambda P, m: setattr(P, 'T', 3),
    lambda P, m: setattr(m, 'G', 1),
    lambda P, m: setattr(P.ids, 'formula', "lambda: (1, 2)"),
    lambda P, m: setattr(P.cf, 'formula', "lambda t: t"),
    lambda P, m: setattr(P, 'formula', "lambda i, scen=2: None"),
    lambda P, m: P.new_cells('extra', "lambda: 1"),
    lambda P, m: P.clear_items(),
])
def test_edits_clear_the_result(model, edit):
    P = model.P
    r = P.collect()
    edit(P, model)
    r2 = P.collect(1)
    assert r2 is not r
    assert_same(r2, loop(P))


def test_edit_of_a_base_clears_the_result():
    m = mx.new_model()
    B = m.new_space('B')
    B.new_cells('x', lambda: 1)
    P = m.new_space('P', bases=B, formula=lambda i: None)
    P.ids = (1, 2)
    P.set_collect(over={'i': 'ids'}, outputs=lambda: {'sum': {'x': x() * i}})
    assert P.collect().sum.x == 3
    B.x.formula = lambda: 10
    assert P.collect().sum.x == 30
    m.close()


def test_dependencies_outside_the_itemspaces_are_kept():
    m = mx.new_model()
    Rates = m.new_space('Rates')
    Rates.new_cells('rate', lambda: 2)
    Rates.factor = 1
    P = m.new_space('P', formula=lambda i: None)
    P.ids = (1, 2, 3)
    P.Rates = Rates
    P.new_cells('x', lambda: i * Rates.rate() * Rates.factor)
    P.set_collect(over={'i': 'ids'}, outputs=lambda: {'sum': {'x': x()}})
    Q = m.new_space('Q')
    Q.P = P
    Q.new_cells('total', lambda: P.collect().sum.x)
    assert Q.total() == 12
    Rates.rate.formula = lambda: 3
    assert Q.total() == 18
    Rates.factor = 2
    assert Q.total() == 36
    P.ids = (1, 2)
    assert Q.total() == 18
    assert not P.itemspaces
    assert Q.total.precedents()[0]._get_repr().startswith(P.fullname + '.collect(')
    m._impl._check_sanity()
    m.close()


def test_error_leaves_no_itemspace_and_no_result(model):
    P = model.P
    P.cf.formula = lambda t: 1 / (i - 3)
    with pytest.raises(FormulaError) as info:
        P.collect()
    assert 'ZeroDivisionError' in str(info.value)
    assert not P.itemspaces
    P.cf.formula = lambda t: i * 1.0
    assert P.collect().sum.pv == loop(P)[1]['pv']


def test_collect_is_defined_only_where_declared(model):
    P = model.P
    Q = model.new_space('Q', formula=lambda i: None)
    assert hasattr(P, 'collect')
    assert not hasattr(Q, 'collect')
    assert not hasattr(P[1], 'collect')
    with pytest.raises(AttributeError):
        Q.collect
    P.set_collect(None)
    assert not hasattr(P, 'collect')
    assert 'collect_outputs' in P.cells
    with pytest.raises(TypeError):
        P._mx_collect()


def test_a_member_named_collect_comes_first(model):
    P = model.P
    P.collect = 5
    assert P.collect == 5
    assert P._mx_collect().sum.pv == loop(P)[1]['pv']


def test_declaration_is_checked():
    m = mx.new_model()
    P = m.new_space('P', formula=lambda i, *a: None)
    P.ids = (1, 2)
    out = lambda: {'sum': {'x': 1}}
    with pytest.raises(TypeError):
        P.set_collect(over={'i': 'ids'}, outputs=out)      # *args
    P.formula = lambda i, j: None
    with pytest.raises(TypeError):
        P.set_collect(over={'i': (1, 2)}, outputs=out)     # a plain iterable
    with pytest.raises(TypeError):
        P.set_collect(over={'i': 'ids', 'j': 'ids'}, outputs=out)
    with pytest.raises(ValueError):
        P.set_collect(over={'k': 'ids'}, outputs=out)
    with pytest.raises(ValueError):
        P.set_collect(over={'i': 'missing'}, outputs=out)
    with pytest.raises(ValueError):
        P.set_collect(over={'i': m.new_space('Q').new_cells('c')}, outputs=out)
    with pytest.raises(TypeError):
        P.set_collect(over={'i': 'ids'})                   # no outputs
    with pytest.raises(TypeError):
        P.set_collect(over={'i': 'ids'}, outputs=lambda t: {})
    assert 'collect_outputs' not in P.cells
    assert not hasattr(P, 'collect')
    with pytest.raises(TypeError):
        m.new_space('R', formula=lambda i: None, collect_over={'i': 'ids'})
    with pytest.raises(ValueError):
        m.new_space('R', formula=lambda i: None,
                    collect_over={'i': 'ids'}, collect_outputs=out)
    assert 'R' not in m.spaces
    m.close()


def test_new_space_declares_collect():
    m = mx.new_model()
    B = m.new_space('B')
    B.new_cells('ids', lambda: (1, 2, 3))
    P = m.new_space('P', bases=B, formula=lambda i: None, collect_over={'i': 'ids'},
                    collect_outputs=lambda: {'sum': {'i': i}})
    assert P.collect().sum.i == 6
    S = m.new_space('S')
    C = S.new_space('C', formula=lambda i: None, refs={'ids': [4, 5]},
                    collect_over={'i': 'ids'},
                    collect_outputs=lambda: {'individual': {'i': i}})
    assert list(C.collect().individual) == [4, 5]
    m.close()


def test_a_stale_declaration_raises_until_set_again(model):
    P = model.P
    P.formula = lambda j: None
    with pytest.raises(ValueError, match='stale'):
        P.collect()
    P.formula = lambda i, scen=1: None
    assert P.collect().sum.pv == loop(P)[1]['pv']
    del P.ids
    with pytest.raises(ValueError, match='stale'):
        P.collect()
    P.new_cells('ids', lambda: (1, 2))
    P.collect()
    del P.collect_outputs
    with pytest.raises(ValueError, match='stale'):
        P.collect()


@pytest.mark.parametrize("domain, error", [
    ((), ValueError), ((1, 1), ValueError), (({},), TypeError),
    ({1, 2}, TypeError), ('ab', TypeError), (iter([1]), TypeError)])
def test_invalid_domains(model, domain, error):
    P = model.P
    P.dom = domain
    P.ids.formula = lambda: dom
    with pytest.raises(FormulaError) as info:
        P.collect()
    assert error.__name__ in str(info.value)
    assert not P.itemspaces


@pytest.mark.parametrize("out, error", [
    ("{'sum': {'x': [1.0, 2.0]}}", TypeError),
    ("{'sum': {'x': 'a'}}", TypeError),
    ("{'sum': {'x': {1: 1}}}", None),
    ("{'other': {'x': 1}}", TypeError),
    ("{}", TypeError),
    ("{'sum': {}}", TypeError),
    ("{'individual': {'x': [_space]}}", TypeError),
])
def test_invalid_outputs(out, error):
    m = mx.new_model()
    P = m.new_space('P', formula=lambda i: None)
    P.ids = (1, 2)
    P.set_collect(over={'i': 'ids'}, outputs="lambda: " + out)
    if error is None:
        assert P.collect().sum.x == {1: 2}
    else:
        with pytest.raises(FormulaError) as info:
            P.collect()
        assert error.__name__ in str(info.value)
        assert not P.itemspaces
    m.close()


def test_sum_structure_must_not_change():
    m = mx.new_model()
    P = m.new_space('P', formula=lambda i: None)
    P.ids = (1, 2)
    P.set_collect(over={'i': 'ids'},
                  outputs=lambda: {'sum': {'x': {t: 1 for t in range(i)}}})
    with pytest.raises(FormulaError, match='structure'):
        P.collect()
    m.close()


def test_outputs_are_copies(model):
    P = model.P
    P.new_cells('data', lambda: {'v': [i]})
    P.collect_outputs.formula = lambda: {'individual': {'d': data()}}
    r = P.collect()
    P[1].data()['v'].append(0)
    assert r.individual[1].d == {'v': [1]}


def test_deleting_the_space_unregisters_it(model):
    P = model.P
    assert model._impl.collect_spaces == {P._impl}
    del model.P
    assert not model._impl.collect_spaces


def test_edits_of_a_model_without_collect_never_read_a_declaration(monkeypatch):
    m = mx.new_model()
    for k in range(5):
        s = m.new_space(formula=lambda i: None)
        s[1]
    slot = UserSpaceImpl.collect_runner
    reads = []

    class Counting:
        def __get__(self, obj, owner=None):
            reads.append(obj)
            return slot.__get__(obj, owner)

        def __set__(self, obj, value):
            slot.__set__(obj, value)

    monkeypatch.setattr(UserSpaceImpl, 'collect_runner', Counting())
    for s in m.spaces.values():
        s.new_cells('x', lambda: 1)
        s.y = 2
    m.new_space('New')
    m.z = 3
    assert not reads
    m.close()
