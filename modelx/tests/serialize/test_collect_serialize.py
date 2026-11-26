import json

import pytest

import modelx as mx


@pytest.fixture
def model():
    m = mx.new_model('CollectModel')
    P = m.new_space('P', formula=lambda i, scen=1: None)
    P.new_cells('ids', lambda: (1, 2, 3))
    P.new_cells('x', lambda: i * scen * 1.5)
    P.set_collect(over={'i': P.ids},
                  outputs=lambda: {'individual': {'x': x()}, 'sum': {'x': x()}})
    S = m.new_space('S')
    C = S.new_space('C', formula=lambda j: None, refs={'keys': ['a', 'b']},
                    collect_over={'j': 'keys'},
                    collect_outputs=lambda: {'individual': {'j': j * 2}})
    yield m
    m.close()


@pytest.mark.parametrize("name", ["saved", "saved.zip"])
def test_declaration_is_saved_and_read_back(model, tmp_path, name):
    model.write(tmp_path / name)
    m2 = mx.read_model(tmp_path / name, name='Read')
    for path in ('P', 'S.C'):
        runner = model._get_object(path)._impl.collect_runner
        runner2 = m2._get_object(path)._impl.collect_runner
        assert (runner2.parameter, runner2.domain) == (runner.parameter, runner.domain)
    r, r2 = model.P.collect(2), m2.P.collect(2)
    assert r2.sum.x == r.sum.x and list(r2.individual) == list(r.individual)
    assert m2.S.C.collect().individual['b'].j == 'bb'
    m2.close()


def test_saved_file(model, tmp_path):
    model.write(tmp_path / 'saved')
    assert json.loads((tmp_path / 'saved' / 'P' / '_collect.json').read_text()) == {
        'parameter': 'i', 'domain': 'ids'}
    assert not (tmp_path / 'saved' / 'S' / '_collect.json').exists()
    model.P.set_collect(None)
    model.write(tmp_path / 'saved2')
    assert not (tmp_path / 'saved2' / 'P' / '_collect.json').exists()


def test_a_stale_declaration_is_not_written(model, tmp_path):
    model.write(tmp_path / 'saved')
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*'))
    model.P.formula = lambda j: None
    with pytest.raises(ValueError, match='stale'):
        model.write(tmp_path / 'saved')
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*')) == before
    m2 = mx.read_model(tmp_path / 'saved', name='Kept')
    assert m2.P.collect().sum.x == 9.0
    m2.close()


def test_an_invalid_declaration_fails_the_read(model, tmp_path):
    model.write(tmp_path / 'saved')
    (tmp_path / 'saved' / 'P' / '_collect.json').write_text(
        json.dumps({'parameter': 'k', 'domain': 'ids'}))
    names = set(mx.get_models())
    with pytest.raises(ValueError, match='invalid collect declaration'):
        mx.read_model(tmp_path / 'saved', name='Bad')
    assert set(mx.get_models()) == names
