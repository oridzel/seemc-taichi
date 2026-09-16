import math

import numpy as np

from seemc_taichi.plane import per_primary_emission_counts, yield_sem_from_counts


def test_per_primary_emission_counts_partition():
    result = {
        "n_primaries": 4,
        "emission_root_primary_id": np.array([0, 0, 1, 3, 3], dtype=np.int32),
        "emission_energy_ev": np.array([5.0, 100.0, 50.0, 2.0, 500.0]),
    }
    total, se, bse = per_primary_emission_counts(result, 50.0)
    assert total.tolist() == [2, 1, 0, 2]
    assert se.tolist() == [1, 1, 0, 1]
    assert bse.tolist() == [1, 0, 0, 1]
    assert np.array_equal(total, se + bse)


def test_yield_sem_from_counts():
    x = np.array([0, 1, 2, 1], dtype=float)
    expected = float(x.std(ddof=0) / math.sqrt(x.size))
    assert math.isclose(yield_sem_from_counts(x), expected)
    assert yield_sem_from_counts([3]) == 0.0
