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
from modelx_helpers import typed_inputs
from modelx_helpers.immutable_inputs import Record, Table
record = Record({'x': 1.5})
assert type(pickle.loads(pickle.dumps(record))) is Record
table = Table((1,), ((1.5,),), ('x',))
assert type(pickle.loads(pickle.dumps(table))) is Table
assert typed_inputs.lookup_float(table, 1, 'x') == 1.5
assert Record.__module__ == Table.__module__ == 'modelx_helpers.immutable_inputs'
'''], check=True)
