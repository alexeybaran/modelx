"""Convert KLV_DE_S: enum classes move into a module stored with the model; formulas unchanged.

Run with lifelib importable (the shipped model's References are pickled by lifelib import path).
"""
import sys, enum, pickle, shutil, pathlib
import modelx as mx
from results import collect, compare
src, dst, expected = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]

m = mx.read_model(src / "KLV_DE_S")
# 1. store the enum module's source with the model (saved as KLV_DE_S/enums.py)
m.new_module(name="enums", path="enums.py", module=src / "enums.py")
# 2. rebind every Reference holding a class of the package enums module to the stored module's class
changed = []
for space in m.spaces.values():
    for name, value in list(space._own_refs.items()):
        if isinstance(value, type) and issubclass(value, enum.Enum) and value.__module__.endswith(".enums"):
            setattr(space, name, getattr(m.enums, value.__qualname__))
            changed.append(space.name + "." + name)
print("rebound:", ", ".join(changed))

# live check
print("LIVE:", compare(pickle.load(open(expected, "rb")), collect(m)))

# save next to the inputs; the package enums.py is NOT copied
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.suffix == ".csv" or f.name in ("sample.py", "spec.py"):
        shutil.copy(f, dst / f.name)
m.write(dst / "KLV_DE_S")
print("saved to", dst / "KLV_DE_S")
