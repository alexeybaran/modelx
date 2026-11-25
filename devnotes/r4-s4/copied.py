"""Reload a copy of the converted model with lifelib NOT importable; then edit the copy's enums.py."""
import sys, pickle, pathlib, importlib.util
import modelx as mx
from results import collect, compare
folder, expected = pathlib.Path(sys.argv[1]), sys.argv[2]
assert importlib.util.find_spec("lifelib") is None, "lifelib must not be importable"
print("lifelib importable:", importlib.util.find_spec("lifelib") is not None)

m = mx.read_model(folder / "KLV_DE_S")
p = m.Projection
assert p.PolsTiming is m.enums.PolsTiming and m.Data.Sex is m.enums.Sex
print("COPY:", compare(pickle.load(open(expected, "rb")), collect(m)))
m.close()

# edit the copy's stored enum module: add a member, reload
path = folder / "KLV_DE_S" / "enums.py"
text = path.read_text(encoding="utf-8")
path.write_text(text.replace("    AFT_LAPSE = 2\n", "    AFT_LAPSE = 2\n    EDITED = 7\n", 1), encoding="utf-8")
m = mx.read_model(folder / "KLV_DE_S")
print("EDIT: Projection.PolsTiming.EDITED =", repr(m.Projection.PolsTiming.EDITED),
      "| members:", [x.name for x in m.Projection.PolsTiming])
print("EDIT:", compare(pickle.load(open(expected, "rb")), collect(m)))
path.write_text(text, encoding="utf-8")
