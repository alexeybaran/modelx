"""Exporting the same model twice writes the same files.

Each export runs in its own process, so values that depend on memory
addresses (``id()``) or on string hashing differ between the two runs
unless the exporter avoids them.
"""
import os
import subprocess
import sys
import textwrap

import modelx as mx

LABELS_SOURCE = """\
from enum import IntEnum


class Plan(IntEnum):
    T10 = 0
    T20 = 1
"""

EXPORT_SCRIPT = textwrap.dedent("""\
    import sys
    import modelx as mx
    m = mx.read_model(sys.argv[1])
    mx.export_model(m, sys.argv[2])
    m.close()
""")


def build_model(src_dir):
    import pandas as pd

    src = src_dir / 'labels_src.py'
    src.write_text(LABELS_SOURCE, encoding='utf-8')

    m = mx.new_model('DeterministicExport')
    s = m.new_space('Space1')
    m.new_module(name='labels', path='labels.py', module=src)
    # CSV, not Excel: openpyxl stamps each written workbook with the time.
    m.new_pandas('df1', 'df1.csv', pd.DataFrame({'x': [1, 2, 3]}), 'csv')
    m.new_pandas('df2', 'df2.csv', pd.DataFrame({'y': [4., 5.]}), 'csv')

    # Pickled values of several kinds, one shared by two References,
    # and a dict keyed by a class from the user module.
    shared = {'a': 1, 'b': [1, 2]}
    s.shared1 = shared
    s.shared2 = shared
    s.tup = (1, 'two', 3.0)
    s.lst = [{'k': 'v'}, (4, 5)]
    m.mtable = {'p': 0.1, 'q': 0.2}
    s.Plan = m.labels.Plan
    s.terms = {m.labels.Plan.T10: 10, m.labels.Plan.T20: 20}
    s.df1 = m.df1
    s.df2 = m.df2
    c = s.new_space('Child', formula=lambda i: None)
    c.ref = ['child', 'value']
    s.new_cells('total', formula=lambda: sum(terms.values()) + df1['x'].sum())
    return m


def export_in_subprocess(model_dir, export_dir, hashseed):
    # The child imports the same modelx as this process.
    env = dict(os.environ)
    env['PYTHONHASHSEED'] = str(hashseed)
    env['PYTHONPATH'] = os.pathsep.join(
        [os.path.dirname(os.path.dirname(os.path.abspath(mx.__file__)))]
        + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    subprocess.run(
        [sys.executable, '-c', EXPORT_SCRIPT, str(model_dir), str(export_dir)],
        env=env, check=True, cwd=str(model_dir.parent))


def read_tree(path):
    result = {}
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        for name in filenames:
            p = os.path.join(dirpath, name)
            with open(p, 'rb') as f:
                result[os.path.relpath(p, path)] = f.read()
    return result


def test_export_twice_is_identical(tmp_path):
    m = build_model(tmp_path)
    model_dir = tmp_path / 'saved'
    try:
        m.write(model_dir)
        expected_total = m.Space1.total()
    finally:
        m.close()

    first = tmp_path / 'first' / 'DeterministicExport'
    second = tmp_path / 'second' / 'DeterministicExport'
    export_in_subprocess(model_dir, first, hashseed=1)
    export_in_subprocess(model_dir, second, hashseed=2)

    files1, files2 = read_tree(first), read_tree(second)
    assert '_mx_pickled' in files1
    assert sorted(files1) == sorted(files2)
    differing = [k for k in files1 if files1[k] != files2[k]]
    assert not differing

    # The exported package still loads what the model holds.
    sys.path.insert(0, str(first.parent))
    try:
        from DeterministicExport import mx_model
        s = mx_model.Space1
        assert s.shared1 is s.shared2
        assert s.shared1 == {'a': 1, 'b': [1, 2]}
        assert s.tup == (1, 'two', 3.0)
        assert s.lst == [{'k': 'v'}, (4, 5)]
        assert mx_model.mtable == {'p': 0.1, 'q': 0.2}
        assert all(type(k) is s.Plan for k in s.terms)
        assert list(s.df1['x']) == [1, 2, 3]
        assert list(s.df2['y']) == [4., 5.]
        assert s.total() == expected_total
    finally:
        sys.path.pop(0)
        for name in list(sys.modules):
            if name == 'DeterministicExport' or name.startswith(
                    'DeterministicExport.'):
                del sys.modules[name]
