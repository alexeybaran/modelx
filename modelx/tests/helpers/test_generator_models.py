"""Model adaptations preserve live/export behavior and binding boundaries."""
import subprocess
import sys

import modelx as mx

from modelx_helpers.generator_models import apply_generators


def test_boolean_overlay_preserves_results_and_shadowed_bindings(tmp_path):
    model = mx.new_model('GeneratorOverlay')
    try:
        s = model.new_space('Projection')
        s.new_cells('every', 'lambda n: all(i < 3 for i in range(n))')
        s.new_cells('some', 'lambda n: any(i == 2 for i in range(n))')
        s.new_cells('local', '''def local(n):
    all = lambda values: 17
    return all(i for i in range(n))
''')
        shadowed = model.new_space('Shadowed', refs={'all': sum})
        shadowed.new_cells('total', 'lambda n: all(i for i in range(n))')
        assert s.every(0) is True and s.every(4) is False
        assert s.some(0) is False and s.some(4) is True
        original = s.local.formula.source
        baseline = tmp_path / 'baseline_model'
        mx.write_model(model, baseline, backup=False)
        receipt = apply_generators(model)
        assert {r['cell'] for r in receipt} == {'GeneratorOverlay.Projection.every',
                                               'GeneratorOverlay.Projection.some'}
        assert s.local.formula.source == original
        assert s.local(8) == 17 and shadowed.total(4) == 6
        assert s.every(0) is True and s.every(5) is False
        assert s.some(0) is False and s.some(5) is True
        assert not apply_generators(model)
        saved = tmp_path / 'saved_model'
        mx.write_model(model, saved, backup=False)
        assert {p.relative_to(saved) for p in saved.rglob('*') if p.is_file()} == {
            p.relative_to(baseline) for p in baseline.rglob('*') if p.is_file()}
        reread = mx.read_model(saved, name='RereadOverlay')
        try:
            assert reread.Projection.every(7) is False
            assert reread.Projection.some(7) is True
        finally:
            reread.close()
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    subprocess.run([sys.executable, '-c', '''from exported import mx_model
p = mx_model.Projection
assert p.every(0) is True and p.every(7) is False
assert p.some(0) is False and p.some(7) is True
assert p.local(7) == 17 and mx_model.Shadowed.total(4) == 6
'''], cwd=tmp_path, check=True)


def test_namespace_preserves_formulas_only_serialization(tmp_path):
    model = mx.new_model('FormulaOnly')
    try:
        s = model.new_space('Projection')
        s.new_cells('check', 'lambda n: all(i < 2 for i in range(n))')
        assert len(apply_generators(model)) == 1
        saved = tmp_path / 'model'
        mx.write_model(model, saved, backup=False)
        assert not (saved / '_data').exists()
        assert not list(saved.rglob('*.pickle'))
        reread = mx.read_model(saved, name='RereadFormulaOnly')
        try:
            assert reread.Projection.check(0) is True
            assert reread.Projection.check(3) is False
        finally:
            reread.close()
    finally:
        model.close()
