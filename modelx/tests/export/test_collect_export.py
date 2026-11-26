import importlib
import itertools

import numpy as np
import pytest

import modelx as mx
from modelx.export import _mx_sys

_counter = itertools.count()


def export(model, tmp_path, monkeypatch, **kwargs):
    package = 'collect_export_%d' % next(_counter)
    model.export(tmp_path / package, **kwargs)
    monkeypatch.syspath_prepend(str(tmp_path))
    return importlib.import_module(package).mx_model


@pytest.fixture
def model():
    m = mx.new_model('CollectExport')
    P = m.new_space('P', formula='lambda i, scen=1: {"refs": {"w": i * 0.5}}',
                    refs={'w': 0, 'np': np, 'T': 4})
    P.new_cells('ids', lambda: (3, 1, 2, 7))
    P.new_cells('cf', lambda t: (i + w) * scen * 1.03 ** t)
    P.new_cells('pv', lambda: sum(cf(t) / 1.01 ** t for t in range(T)))
    C = P.new_space('C')
    C.new_cells('y', lambda: i * 10)
    P.set_collect(over={'i': P.ids}, outputs=lambda: {
        'individual': {'pv': pv(), 'y': C.y()},
        'sum': {'pv': pv(), 'cf': {t: cf(t) for t in range(T)},
                'arr': np.array([cf(t) for t in range(T)])}})
    yield m
    m.close()


def loop(P, scen):
    individual, total = {}, None
    for i in P.ids():
        item = P[i, scen]
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
    return individual, total


def assert_same(result, expected):
    individual, total = expected
    assert list(result.individual) == list(individual)
    for i, values in individual.items():
        assert result.individual[i].pv == values['pv']
        assert result.individual[i].y == values['y']
    assert result.sum.pv == total['pv']
    assert dict(result.sum.cf) == total['cf']
    assert result.sum.arr.tobytes() == total['arr'].tobytes()


@pytest.mark.parametrize("use_slots", [True, False])
def test_exported_collect_equals_live_and_a_loop(model, tmp_path, monkeypatch, use_slots):
    e = export(model, tmp_path, monkeypatch, use_slots=use_slots)
    for args, kwargs, scen in [((), {}, 1), ((2,), {}, 2), ((), {'scen': 2}, 2)]:
        expected = loop(e.P, scen)
        e.P._mx_itemspaces.clear()
        exported = e.P.collect(*args, **kwargs)
        assert not e.P._mx_itemspaces
        assert_same(exported, expected)
        assert_same(model.P.collect(*args, **kwargs), expected)
        assert e.P.collect(*args, **kwargs) is exported
        assert exported.sum.cf(2) == exported.sum.cf[2]


def test_existing_itemspaces_are_used_and_kept(model, tmp_path, monkeypatch):
    e = export(model, tmp_path, monkeypatch)
    items = {i: e.P[i] for i in (1, 7)}
    expected = loop(e.P, 1)
    e.P._mx_itemspaces.clear()
    e.P._mx_itemspaces.update({(i, 1): item for i, item in items.items()})
    calls = []
    cls = type(e.P)
    original = cls._mx_collect_item

    def item(self, i, scen=1):
        root = original(self, i, scen)
        calls.append(root is items.get(i))
        return root

    monkeypatch.setattr(cls, '_mx_collect_item', item)
    assert_same(e.P.collect(), expected)
    assert calls == [False, True, False, True]
    assert set(e.P._mx_itemspaces) == {(1, 1), (7, 1)}


def test_reference_domain_and_a_member_named_collect(tmp_path, monkeypatch):
    m = mx.new_model()
    P = m.new_space('P', formula=lambda i: None, refs={'keys': (2, 4)})
    P.set_collect(over={'i': 'keys'}, outputs=lambda: {'sum': {'i': i}})
    Q = m.new_space('Q', formula=lambda i: None, refs={'keys': (5,), 'collect': 1})
    Q.set_collect(over={'i': 'keys'}, outputs=lambda: {'individual': {'i': i}})
    e = export(m, tmp_path, monkeypatch)
    assert e.P.collect().sum.i == 6 == P.collect().sum.i
    assert e.Q.collect == 1 == Q.collect
    assert list(e.Q._mx_collect().individual) == [5] == list(Q._mx_collect().individual)
    m.close()


def test_itemspaces_have_no_collect_live_or_exported(tmp_path, monkeypatch):
    m = mx.new_model()
    P = m.new_space('P', formula=lambda i: None, refs={'keys': (2, 4)})
    P.set_collect(over={'i': 'keys'}, outputs=lambda: {'sum': {'i': i}})
    Q = m.new_space('Q', formula=lambda i: None, refs={'keys': (5,), 'collect': 1})
    Q.set_collect(over={'i': 'keys'}, outputs=lambda: {'individual': {'i': i}})
    e = export(m, tmp_path, monkeypatch)
    for space, name in [(P, 'collect'), (e.P, 'collect'),
                        (Q, '_mx_collect'), (e.Q, '_mx_collect')]:
        assert hasattr(space, name)
        item = space[space.keys[0]]
        assert not hasattr(item, name)
        with pytest.raises(AttributeError, match="Attribute '%s' is not in" % name):
            getattr(item, name)
    m.close()


def test_errors_propagate_and_are_not_cached(model, tmp_path, monkeypatch):
    model.P.ids.formula = lambda: (1, 1)
    e = export(model, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='duplicate'):
        e.P.collect()
    assert not e.P._mx_collect_cache


def test_a_stale_declaration_is_not_exported(model, tmp_path):
    model.P.formula = lambda j: None
    with pytest.raises(ValueError, match='stale'):
        model.export(tmp_path / 'stale_nomx')
    assert not (tmp_path / 'stale_nomx').exists()


def test_a_model_without_collect_has_no_collect_runtime(tmp_path):
    m = mx.new_model()
    m.new_space('P', formula=lambda i: None).new_cells('x', 'lambda: i')
    m.export(tmp_path / 'plain_nomx')
    assert not (tmp_path / 'plain_nomx' / '_mx_collect.py').exists()
    assert 'collect' not in (tmp_path / 'plain_nomx' / '_mx_classes.py').read_text()
    m.close()


# The collect hook (doc/source/reference/collect_hook.rst)

def install_hook(monkeypatch, space, hook, version=1):
    cls = type(space)
    monkeypatch.setattr(cls, '_mx_collect_hook', hook, raising=False)
    if version is not None:
        monkeypatch.setattr(cls, '_mx_collect_hook_version', version, raising=False)


def test_hook_takes_over_collect(model, tmp_path, monkeypatch):
    e = export(model, tmp_path, monkeypatch)
    collect_module = importlib.import_module(type(e).__module__.rsplit('.', 1)[0] + '._mx_collect')
    calls = []

    def hook(space, coords, args):
        calls.append((space, coords, args))
        return collect_module.CollectResult({'sum': collect_module.CollectResult({'n': len(coords)})})

    install_hook(monkeypatch, e.P, hook)
    assert _mx_sys.COLLECT_HOOK_VERSIONS == (1,)
    r = e.P.collect(scen=2)
    assert r.sum.n == 4
    assert calls == [(e.P, (3, 1, 2, 7), (2,))]
    assert e.P.collect(2) is r
    assert len(calls) == 1
    assert not e.P._mx_itemspaces


def test_hook_can_decline(model, tmp_path, monkeypatch):
    e = export(model, tmp_path, monkeypatch)
    install_hook(monkeypatch, e.P, lambda space, coords, args: None)
    assert_same(e.P.collect(), loop(model.P, 1))


@pytest.mark.parametrize("version", [None, 0, 2, '1'])
def test_hook_of_an_unknown_version_raises(model, tmp_path, monkeypatch, version):
    e = export(model, tmp_path, monkeypatch)
    install_hook(monkeypatch, e.P, lambda space, coords, args: None, version)
    with pytest.raises(RuntimeError, match='version'):
        e.P.collect()


def test_hook_must_return_a_collect_result(model, tmp_path, monkeypatch):
    e = export(model, tmp_path, monkeypatch)
    install_hook(monkeypatch, e.P, lambda space, coords, args: {'sum': {}})
    with pytest.raises(TypeError, match='CollectResult'):
        e.P.collect()
    assert not e.P._mx_collect_cache
