"""The portable helper package has no engine or accelerator import dependency."""
import subprocess
import sys


def test_reference_api_imports_without_engine_or_optional_backends():
    subprocess.run([sys.executable, '-c', '''
import importlib.abc
import pickle
import sys

class RejectBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'modelx', 'modelx_cython', 'lifelib', 'pandas', 'numpy'}:
            raise AssertionError('unexpected backend import: ' + fullname)

sys.meta_path.insert(0, RejectBackends())
from modelx_helpers import expressions, expression_streams, typed_inputs
from modelx_helpers.immutable_inputs import Record, Table
assert expressions.sum_float is expression_streams.sum_float
assert expressions.lookup_float is typed_inputs.lookup_float
record = Record({'x': 1.5})
assert type(pickle.loads(pickle.dumps(record))) is Record
table = Table((1,), ((1.5,),), ('x',))
assert type(pickle.loads(pickle.dumps(table))) is Table
assert expressions.lookup_float(table, 1, 'x') == 1.5
assert Record.__module__ == Table.__module__ == 'modelx_helpers.immutable_inputs'
'''], check=True)
