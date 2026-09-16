from types import SimpleNamespace

import numpy as np

from seemc_taichi.reference_sweep import (
    deterministic_case_seed,
    emission_arrays,
    reference_emission_filename,
    summarize_reference_emissions,
)


def _emission(**kw):
    defaults = dict(
        energy=10.0,
        uvw=(0.0, 0.0, -1.0),
        electron_id=2,
        parent_id=1,
        root_primary_id=0,
        generation=1,
        is_cascade=True,
        emission_mechanism="transport_escape",
        birth_depth=3.5,
        barrier_reflection_probability=None,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_reference_filename():
    assert (
        reference_emission_filename("Si", 1000.0, 30.0)
        == "Si_1000eV_a30deg_reference_emissions.npz"
    )


def test_case_seed_is_point_stable():
    a = deterministic_case_seed(123, 0.0, 100.0)
    b = deterministic_case_seed(123, 0.0, 100.0)
    c = deterministic_case_seed(123, 0.0, 500.0)
    assert a == b
    assert a != c


def test_emission_arrays_match_taichi_core_layout():
    items = [
        _emission(),
        _emission(
            energy=100.0,
            electron_id=0,
            parent_id=None,
            generation=0,
            is_cascade=False,
            emission_mechanism="incoming_barrier_reflection",
            uvw=(0.6, 0.0, -0.8),
        ),
    ]
    out = emission_arrays(items, n_primaries=3)
    assert out["emission_energy_ev"].shape == (2,)
    assert out["emission_ux"].shape == (2,)
    assert out["emission_parent_id"].tolist() == [1, -1]
    assert out["emission_generation"].tolist() == [1, 0]
    assert out["emission_is_cascade"].tolist() == [1, 0]
    assert out["emission_mechanism"].tolist() == [1, 2]
    norms = np.sqrt(
        out["emission_ux"] ** 2
        + out["emission_uy"] ** 2
        + out["emission_uz"] ** 2
    )
    assert np.allclose(norms, 1.0)


def test_summary_uses_same_50ev_partition_and_lineage():
    items = [
        _emission(energy=10.0),
        _emission(energy=50.0),
        _emission(
            energy=51.0,
            electron_id=0,
            parent_id=None,
            generation=0,
            is_cascade=False,
        ),
    ]
    arrays = emission_arrays(items, n_primaries=4)
    s = summarize_reference_emissions(arrays, n_primaries=4, cutoff_ev=50.0)
    assert s["tey"] == 0.75
    assert s["sey"] == 0.5
    assert s["bsey"] == 0.25
    assert s["cascade_yield"] == 0.5
    assert s["primary_yield"] == 0.25
