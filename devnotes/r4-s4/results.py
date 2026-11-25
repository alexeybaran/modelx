"""Shared: the result frames of every model point of the shipped book."""
import pandas as pd

CHECKS = ["check_net_cf", "check_pols_roll_fwd", "check_res_roll_fwd", "check_surplus_roll_fwd",
          "check_surr_floor", "check_surr_nonneg", "check_equivalence"]


def collect(model):
    out = {}
    for key in map(int, model.Data.model_point_table().index):
        item = model.Projection[key]
        out[key] = {"result_cf": item.result_cf(), "result_cf_annual": item.result_cf_annual(),
                    "result_surplus": item.result_surplus(),
                    "checks": {c: getattr(item, c)() for c in CHECKS}}
    return out


def compare(expected, actual):
    assert expected.keys() == actual.keys(), (expected.keys(), actual.keys())
    for key in expected:
        for name in ("result_cf", "result_cf_annual", "result_surplus"):
            pd.testing.assert_frame_equal(expected[key][name], actual[key][name], check_exact=True)
        assert expected[key]["checks"] == actual[key]["checks"], key
    net = sum(float(v["result_cf"]["net_cf"].sum()) for v in actual.values())
    return "%d model points: result_cf (incl. net_cf), result_cf_annual, result_surplus and %d check cells " \
           "bit-identical; sum of net_cf over book = %.6f" % (len(actual), len(CHECKS), net)
