import math
from seemc_taichi.validation import YieldStats, compare_yields


def test_compare_yields():
    r = YieldStats(1.0, 0.2, 1.2, 0.01, 0.005)
    c = YieldStats(1.01, 0.19, 1.20, 0.01, 0.005)
    d = compare_yields(r, c)
    assert math.isclose(d["tey_delta"], 0.0, abs_tol=1e-12)
    assert d["sey_pull"] > 0
    assert d["bsey_pull"] < 0
