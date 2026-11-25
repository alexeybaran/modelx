"""Run the compiled package (exported_cy) with lifelib NOT importable and compare; then call the enum-valued cells."""
import sys, pickle, pathlib, importlib.util, traceback
folder, expected = pathlib.Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(folder))
assert importlib.util.find_spec("lifelib") is None
from exported_cy import mx_model
from results import collect, compare
try:
    print("COMPILED:", compare(pickle.load(open(expected, "rb")), collect(mx_model)))
except Exception as e:
    print("COMPILED results FAILED:", type(e).__name__, str(e)[:200])
item = mx_model.Projection[1]
for name in ("sex", "smoker", "prem_freq", "surplus_use"):
    try:
        print("COMPILED:", name, "->", repr(getattr(item, name)()))
    except Exception as e:
        print("COMPILED:", name, "FAILED", type(e).__name__, str(e)[:120])
