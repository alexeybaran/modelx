import concurrent.futures as futures
import gc
import importlib
import itertools
import sys
import threading
import weakref

import numpy as np
import pytest

import modelx as mx
from modelx.export import _mx_sys

_counter = itertools.count()


class TimedFuture(futures.Future):
    def result(self, timeout=5):
        return super().result(timeout)


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


def test_parallel_export_overlaps_private_items_and_shares_locked_dependencies(
        model, tmp_path, monkeypatch):
    if getattr(sys, '_is_gil_enabled', lambda: True)():
        pytest.skip('requires free-threaded Python with the GIL off')

    P = model.P
    P[3].cf[0] = 81.0
    shared = model.new_space('Shared')
    table = shared.new_space('Table', formula=lambda key: None, refs={'keys': (7, 9)})
    table.new_cells('offset', lambda: 2 * key)
    table.set_collect(over={'key': 'keys'}, outputs=lambda: {'sum': {'value': offset()}})
    P.shared = shared
    P.collect_outputs.formula = lambda: {
        'individual': {'series': {0: cf(0)}, 'values': [i, 'policy']},
        'sum': {'value': C.y() + shared.Table.collect().sum.value + shared.Table[7].offset()}}
    serial = export(model, tmp_path, monkeypatch, locked_spaces=['P', 'Shared'])
    expected = serial.P.collect()
    e = export(model, tmp_path, monkeypatch, locked_spaces=['P', 'Shared'])
    e._mx_collect_workers = 4
    direct = e.P[3]
    assert direct.cf(0) == 81.0

    table_type = type(e.Shared.Table)
    calls = []
    original_offset = table_type._f_offset

    def offset(item):
        assert item._mx_lock is e._mx_lock and e._mx_lock._is_owned()
        calls.append(item)
        return original_offset(item)

    monkeypatch.setattr(table_type, '_f_offset', offset)
    original_outputs = type(e.P)._f_collect_outputs
    barrier = threading.Barrier(4, timeout=5)
    threads = set()
    private_roots = []
    guard = threading.Lock()

    def outputs(item):
        assert item._mx_lock is item.C._mx_lock
        assert item._mx_lock is not e._mx_lock
        assert item.shared._mx_lock is e._mx_lock
        with guard:
            threads.add(threading.get_ident())
            private_roots.append(item)
        barrier.wait()
        return original_outputs(item)

    monkeypatch.setattr(type(e.P), '_f_collect_outputs', outputs)
    observed = e.P.collect()
    assert e.P.collect() is observed
    assert len(threads) == len(private_roots) == 4
    assert all(root._parent is None and root._space is None for root in private_roots)
    assert observed.sum.value == expected.sum.value
    assert set(observed.individual) == set(expected.individual)
    for i, value in observed.individual.items():
        assert value.series(0) == value.series[0] == expected.individual[i].series[0]
        assert value.values == expected.individual[i].values
    assert e.P[3] is direct and direct.cf(0) == 81.0
    assert direct._mx_lock is direct.C._mx_lock is e._mx_lock
    assert list(e.P._mx_itemspaces.values()) == [direct]
    assert sorted(item.key for item in calls) == [7, 7, 9]
    assert list(e.Shared.Table._mx_itemspaces.values()) == [e.Shared.Table[7]]
    assert not sys._is_gil_enabled()


def test_parallel_export_collect_same_key_shares_in_flight_result(model, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, '_is_gil_enabled', lambda: False, raising=False)
    e = export(model, tmp_path, monkeypatch)
    e._mx_collect_workers = 4
    entered = threading.Event()
    release = threading.Event()
    waiting = threading.Event()
    created = []
    space_type = type(e.P)
    make_item = space_type._mx_collect_item

    def item(space, i, scen=1):
        created.append(i)
        entered.set()
        assert release.wait(5)
        return make_item(space, i, scen)

    class ObservedFuture(TimedFuture):
        def result(self, timeout=5):
            waiting.set()
            return super().result(timeout)

    monkeypatch.setattr(space_type, '_mx_collect_item', item)
    monkeypatch.setattr(futures, 'Future', ObservedFuture)
    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(e.P.collect)
        try:
            assert entered.wait(5)
            assert not e.P._mx_collect_cache
            second = pool.submit(e.P.collect, scen=1)
            assert waiting.wait(5)
        finally:
            release.set()
        assert first.result(timeout=5) is second.result(timeout=5)
    assert sorted(created) == [1, 2, 3, 7]


@pytest.mark.parametrize(('workers', 'gil_enabled'), [(1, False), (4, True), (4, False)])
def test_export_collect_error_allows_retry(workers, gil_enabled, model, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, '_is_gil_enabled', lambda: gil_enabled, raising=False)
    e = export(model, tmp_path, monkeypatch)
    e._mx_collect_workers = workers
    direct = e.P[3]
    space_type = type(e.P)
    make_item = space_type._mx_collect_item
    original_outputs = space_type._f_collect_outputs
    roots = []

    def item(space, i, scen=1):
        root = make_item(space, i, scen)
        roots.append(root)
        return root

    def outputs(item):
        if item.i == 3:
            raise ValueError('bad output')
        return original_outputs(item)

    monkeypatch.setattr(space_type, '_mx_collect_item', item)
    monkeypatch.setattr(space_type, '_f_collect_outputs', outputs)
    with pytest.raises(ValueError, match='bad output'):
        e.P.collect()
    assert not e.P._mx_collect_cache
    assert roots and all(root._parent is None and root._space is None for root in roots)
    if workers == 1 or gil_enabled:
        assert len(roots) == 1
    assert e.P[3] is direct
    monkeypatch.setattr(space_type, '_f_collect_outputs', original_outputs)
    observed = e.P.collect()
    expected = model.P.collect()
    assert observed.sum.pv == pytest.approx(expected.sum.pv)
    assert e.P.collect() is observed


def test_parallel_export_collect_does_not_wait_with_the_model_lock(model, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, '_is_gil_enabled', lambda: False, raising=False)
    shared = model.new_space('Shared')
    shared.new_cells('value', lambda i: i)
    model.P.shared = shared
    model.P.collect_outputs.formula = lambda: {'sum': {'value': shared.value(i)}}
    e = export(model, tmp_path, monkeypatch, locked_spaces=['Shared'])
    e._mx_collect_workers = 4
    entered = threading.Event()
    ready_to_read = threading.Event()
    reading = threading.Event()
    original_outputs = type(e.P)._f_collect_outputs
    original_value = type(e.Shared).value

    def outputs(item):
        if item.i == 3:
            entered.set()
            assert ready_to_read.wait(5)
        return original_outputs(item)

    def value(space, i):
        if i == 3:
            reading.set()
        return original_value(space, i)

    monkeypatch.setattr(futures, 'Future', TimedFuture)
    monkeypatch.setattr(type(e.P), '_f_collect_outputs', outputs)
    monkeypatch.setattr(type(e.Shared), 'value', value)
    with futures.ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(e.P.collect)
        assert entered.wait(5)
        with e._mx_lock:
            ready_to_read.set()
            assert reading.wait(5)
            with pytest.raises(RuntimeError, match='holding the model lock'):
                e.P.collect()
        assert running.result(timeout=5).sum.value == 13


def test_parallel_export_detects_a_cycle_between_running_collects(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, '_is_gil_enabled', lambda: False, raising=False)
    monkeypatch.setattr(futures, 'Future', TimedFuture)
    model = mx.new_model()
    try:
        A = model.new_space('A', formula=lambda i: None, refs={'keys': (1, 2)})
        B = model.new_space('B', formula=lambda i: None, refs={'keys': (1, 2)})
        A.other = B
        B.other = A
        for space in (A, B):
            space.set_collect(over={'i': 'keys'}, outputs=lambda: {
                'sum': {'value': other.collect().sum.value if i == 1 else i}})
        e = export(model, tmp_path, monkeypatch)
    finally:
        model.close()
    e._mx_collect_workers = 4
    barrier = threading.Barrier(2, timeout=5)

    def wait_for_other(outputs):
        def wrapped(item):
            if item.i == 1:
                barrier.wait()
            return outputs(item)
        return wrapped

    for space in (e.A, e.B):
        cls = type(space)
        monkeypatch.setattr(cls, '_f_collect_outputs', wait_for_other(cls._f_collect_outputs))
    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        calls = (pool.submit(e.A.collect), pool.submit(e.B.collect))
        for call in calls:
            with pytest.raises(RuntimeError, match='Circular dependency'):
                call.result(timeout=5)
    assert not e.A._mx_collect_cache and not e.B._mx_collect_cache


def test_custom_cache_key_can_reenter_collect_from_equality(model, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, '_is_gil_enabled', lambda: False, raising=False)
    e = export(model, tmp_path, monkeypatch)
    e._mx_collect_workers = 4
    collect_module = importlib.import_module(type(e).__module__.rsplit('.', 1)[0] + '._mx_collect')
    caller = threading.get_ident()
    build_threads = set()
    make_item = type(e.P)._mx_collect_item
    reentries = []

    def item(space, i, scen=1):
        build_threads.add(threading.get_ident())
        return make_item(space, i, scen)

    class Scenario(int):
        def __hash__(self):
            return int.__hash__(self)

        def __eq__(self, other):
            if type(other) is int and other == 1:
                assert collect_module._PENDING_LOCK.acquire(blocking=False)
                collect_module._PENDING_LOCK.release()
                reentries.append(e.P.collect(99))
            return self is other

    monkeypatch.setattr(type(e.P), '_mx_collect_item', item)
    custom = Scenario(1)
    serial = e.P.collect(custom)
    assert build_threads == {caller}
    pooled = e.P.collect(1)
    assert pooled is not serial
    assert pooled.sum.pv == pytest.approx(serial.sum.pv)
    assert reentries and all(result is e.P.collect(99) for result in reentries)
    assert e.P.collect(custom) is serial


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
    e._mx_collect_workers = 4
    monkeypatch.setattr(sys, '_is_gil_enabled', lambda: False, raising=False)
    collect_module = importlib.import_module(type(e).__module__.rsplit('.', 1)[0] + '._mx_collect')

    def unexpected_pool(*args):
        pytest.fail('A compiled hook that declines should use the serial fallback')

    monkeypatch.setattr(collect_module, '_collect_once', unexpected_pool)
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
