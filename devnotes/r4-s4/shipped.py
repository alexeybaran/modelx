"""Run the shipped KLV_DE_S (lifelib importable) and store its results."""
import sys, pickle, pathlib
import modelx as mx
from results import collect
folder, out = pathlib.Path(sys.argv[1]), sys.argv[2]
m = mx.read_model(folder / "KLV_DE_S")
res = collect(m)
pickle.dump(res, open(out, "wb"))
print("shipped: stored", len(res), "model points; net_cf sum", sum(float(v["result_cf"]["net_cf"].sum()) for v in res.values()))
