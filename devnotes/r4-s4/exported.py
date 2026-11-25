"""Import the exported pure-Python package with lifelib NOT importable and compare."""
import sys, pickle, pathlib, importlib.util
folder, expected = pathlib.Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(folder))
assert importlib.util.find_spec("lifelib") is None and importlib.util.find_spec("modelx") is None
print("lifelib importable:", False, "| modelx importable:", False)
from exported import mx_model
from results import collect, compare
p = mx_model.Projection
assert p.PolsTiming is mx_model.enums.PolsTiming
print("EXPORT: PolsTiming =", p.PolsTiming, "module", p.PolsTiming.__module__)
print("EXPORT:", compare(pickle.load(open(expected, "rb")), collect(mx_model)))
