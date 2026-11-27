import gc
import importlib
import itertools
import weakref

import numpy as np
import pytest

import modelx as mx

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


def test_exported_collect_keeps_direct_itemspaces_separate(model, tmp_path, monkeypatch):
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
    assert calls == [False, False, False, False]
    assert set(e.P._mx_itemspaces) == {(1, 1), (7, 1)}
    assert all(e.P[i] is item for i, item in items.items())


@pytest.mark.parametrize('use_slots', [True, False])
def test_export_preserves_authored_inputs_in_both_routes(model, tmp_path, monkeypatch, use_slots):
    P = model.P
    P.ids[()] = (3, 1)
    P.C.formula = lambda j: None
    P.C.y.formula = lambda: 10 * i + j
    P.C.new_cells('outside', lambda: i + j)
    P.collect_outputs.formula = lambda: {'sum': {'y': C[10].y()}}
    P[3].C[10].y[()] = 100
    P[3].C[10].outside[()] = 900
    assert P[3].C[10].y.is_input()
    assert P[3].C[10].outside.is_input()

    e = export(model, tmp_path, monkeypatch, use_slots=use_slots)
    item = e.P[3]
    nested = item.C[10]
    assert e.P.ids() == (3, 1)
    assert nested.y() == 100
    assert nested.outside() == 900
    assert e.P.collect().sum.y == P.collect().sum.y == 120
    assert e.P[3] is item
    assert item.C[10] is nested
    assert list(e.P._mx_itemspaces.values()) == [item]


def test_exported_collect_releases_an_item_when_outputs_raise(model, tmp_path, monkeypatch):
    model.P.collect_outputs.formula = lambda: {'sum': {'x': 1 / (i - 3)}}
    e = export(model, tmp_path, monkeypatch)
    direct = e.P[3]
    expected = direct.pv()
    roots = []
    cls = type(e.P)
    original = cls._mx_collect_item

    def item(self, i, scen=1):
        root = original(self, i, scen)
        roots.append(root)
        return root

    monkeypatch.setattr(cls, '_mx_collect_item', item)
    with pytest.raises(ZeroDivisionError):
        e.P.collect()
    assert len(roots) == 1
    assert roots[0]._parent is None
    assert roots[0]._space is None
    assert not e.P._mx_collect_cache
    assert e.P[3] is direct
    assert direct.pv() == expected


def test_exported_collect_releases_relative_space_and_cell_references(model, tmp_path, monkeypatch):
    P = model.P
    P.relref(root_alias=P)
    P.C.relref(child_alias=P.C, cell_alias=P.C.y)
    P.collect_outputs.formula = lambda: {
        'sum': {'value': root_alias.C.child_alias.cell_alias()}}
    e = export(model, tmp_path, monkeypatch)
    direct = e.P[3]
    child = direct.C
    classes = (type(direct), type(child))

    def live_items():
        return sum(type(obj) in classes for obj in gc.get_objects())

    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        before = live_items()
        assert e.P.collect().sum.value == 130
        assert live_items() == before
    finally:
        if was_enabled:
            gc.enable()

    assert e.P[3] is direct and direct.root_alias is direct
    assert direct.C is child and child.child_alias is child
    assert child.cell_alias.__self__ is child


def test_exported_collect_releases_each_item_before_constructing_the_next(model, tmp_path, monkeypatch):
    model.P.ids.formula = lambda: tuple(range(1, 129))
    model.P.collect_outputs.formula = lambda: {'sum': {'value': C.y()}}
    e = export(model, tmp_path, monkeypatch, use_slots=False)
    direct = e.P[7]
    roots = []
    original = type(e.P)._mx_collect_item

    def item(self, i, scen=1):
        if roots:
            assert roots[-1]() is None
        result = original(self, i, scen)
        roots.append(weakref.ref(result))
        return result

    monkeypatch.setattr(type(e.P), '_mx_collect_item', item)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        assert e.P.collect().sum.value == 10 * sum(range(1, 129))
        assert len(roots) == 128 and roots[-1]() is None
        assert list(e.P._mx_itemspaces.values()) == [direct]
        assert e.P[7] is direct and direct.C.y() == 70
    finally:
        if was_enabled:
            gc.enable()


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


def test_a_model_without_collect_has_no_collect_runtime(tmp_path, monkeypatch):
    m = mx.new_model()
    try:
        m.new_space('P', formula=lambda i: None).new_cells('x', 'lambda: i')
        package = tmp_path / 'plain_nomx'
        m.export(package)
        assert not (package / '_mx_collect.py').exists()
        assert 'collect' not in (package / '_mx_classes.py').read_text()
        monkeypatch.syspath_prepend(str(tmp_path))
        runtime = importlib.import_module('plain_nomx._mx_sys')
        assert not hasattr(runtime, 'COLLECT_HOOK_VERSIONS')
        assert not hasattr(runtime, 'collect_from_space')
        assert not hasattr(runtime, 'release_collect_item')
        assert importlib.import_module('plain_nomx').mx_model.P[4].x() == 4
    finally:
        m.close()


def test_nested_collect_declaration_enables_exported_runtime(tmp_path, monkeypatch):
    model = mx.new_model()
    try:
        parent = model.new_space('Parent')
        child = parent.new_space('Child', formula='lambda i: None', refs={'ids': (2, 4)})
        child.set_collect(over={'i': 'ids'}, outputs='lambda: {"sum": {"value": i}}')
        exported = export(model, tmp_path, monkeypatch)
        assert exported.Parent.Child.collect().sum.value == 6
        assert not exported.Parent.Child._mx_itemspaces
    finally:
        model.close()


# The collect hook (doc/source/reference/collect_hook.rst)

def install_hook(monkeypatch, space, hook, version=1):
    cls = type(space)
    monkeypatch.setattr(cls, '_mx_collect_hook', hook, raising=False)
    if version is not None:
        monkeypatch.setattr(cls, '_mx_collect_hook_version', version, raising=False)


def test_hook_takes_over_collect(model, tmp_path, monkeypatch):
    e = export(model, tmp_path, monkeypatch)
    package = type(e).__module__.rsplit('.', 1)[0]
    collect_module = importlib.import_module(package + '._mx_collect')
    runtime = importlib.import_module(package + '._mx_sys')
    calls = []

    def hook(space, coords, args):
        calls.append((space, coords, args))
        return collect_module.CollectResult({'sum': collect_module.CollectResult({'n': len(coords)})})

    install_hook(monkeypatch, e.P, hook)
    assert runtime.COLLECT_HOOK_VERSIONS == (1,)
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


def test_a_repeated_read_of_the_same_domain_tuple_is_not_normalized(tmp_path, monkeypatch):
    """A cached result is read again without normalizing the domain only when the arguments are
    equal and the domain is the same tuple object; an equal but new tuple, or another domain, is
    normalized and gives its own result."""
    m = mx.new_model()
    P = m.new_space('P', formula=lambda i, s=1: None, refs={'keys': (2, 4)})
    P.set_collect(over={'i': 'keys'}, outputs=lambda: {'sum': {'v': i * s}})
    e = export(m, tmp_path, monkeypatch)
    collect_module = importlib.import_module(type(e).__module__.rsplit('.', 1)[0] + '._mx_collect')
    calls = []
    normalize = collect_module.normalize_domain
    monkeypatch.setattr(collect_module, 'normalize_domain', lambda value: calls.append(value) or normalize(value))
    r = e.P.collect()
    assert r.sum.v == 6 and len(calls) == 1
    assert e.P.collect() is r and e.P.collect(1) is r and len(calls) == 1     # the same tuple, equal args
    assert e.P.collect(2).sum.v == 12 and len(calls) == 2                       # other args
    e.P.keys = e.P.keys                                      # the same object again
    assert e.P.collect() is r and len(calls) == 2
    e.P.keys = tuple([2, 4])                                 # an equal, new tuple: normalized, same result
    assert e.P.collect() is r and len(calls) == 3
    assert e.P.collect() is r and len(calls) == 3
    e.P.keys = (2, 4, 5)                                     # another domain
    assert e.P.collect().sum.v == 11 and len(calls) == 4
    e.P.keys = [2, 4]                                        # not a tuple: normalized at every read
    assert e.P.collect() is r and e.P.collect() is r and len(calls) == 6
    m.close()
