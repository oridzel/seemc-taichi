import math

from seemc_taichi.sweep import (
    build_sweep_row,
    csv_row_is_valid,
    emission_filename,
    load_csv_rows,
    point_key,
    write_csv_rows,
)


def _result(overflow=False, **diag_overrides):
    diag = {
        "elastic_events": 100,
        "inelastic_events": 20,
        "escapes": 6,
        "sey_50ev": 2,
        "bse_50ev": 4,
        "cascade_emissions": 3,
        "primary_emissions": 3,
        "emission_buffer_overflow": 0,
        "step_limit_hit": 0,
        "generation_limit_hit": 0,
        "omega_cdf_empty": 0,
        "q_cdf_empty": 0,
    }
    diag.update(diag_overrides)
    return {
        "n_primaries": 10,
        "incident_energy_vac_ev": 1000.0,
        "alpha_deg": 30.0,
        "allocated": 42,
        "stored": 42,
        "overflow": overflow,
        "wave_count": 2,
        "kernel_chunks": 5,
        "diagnostics": diag,
        "tey": 0.6,
        "sey_50ev": 0.2,
        "bsey_50ev": 0.4,
        "cascade_yield": 0.3,
        "primary_yield": 0.3,
    }


def _row(result):
    return build_sweep_row(
        result,
        material="Si",
        material_kind="nonconductor",
        arch="metal",
        precision="f32",
        barrier_model="abrupt",
        inner_potential_ev=17.59,
        bse_cutoff_ev=50.0,
        capacity=1000,
        steps_per_chunk=256,
        seed=1,
        elapsed_s=0.5,
    )


def test_valid_row_has_yields_and_rates():
    row = _row(_result())
    assert row["valid"] is True
    assert row["status"] == "ok"
    assert row["tey"] == 0.6
    assert row["collision_events"] == 120
    assert row["collision_events_per_s"] == 240.0
    assert row["primaries_per_s"] == 20.0


def test_invalid_row_marks_primary_yields_nan_but_keeps_raw():
    row = _row(_result(overflow=True))
    assert row["valid"] is False
    assert row["status"] == "particle_overflow"
    assert math.isnan(row["tey"])
    assert row["raw_tey"] == 0.6


def test_csv_round_trip_and_latest_point(tmp_path):
    path = tmp_path / "sweep.csv"
    row = _row(_result())
    rows = {point_key(1000.0, 30.0): row}
    write_csv_rows(path, rows)
    loaded = load_csv_rows(path)
    key = point_key(1000.0, 30.0)
    assert key in loaded
    assert csv_row_is_valid(loaded[key])


def test_emission_filename_is_stable():
    assert emission_filename("Si", 1000.0, 30.0) == "Si_1000eV_a30deg_emissions.npz"
