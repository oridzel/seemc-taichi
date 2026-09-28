from pathlib import Path

import numpy as np

from seemc_taichi.trapezoid import TrapezoidGeometryConfig
from seemc_taichi.trapezoid_scan import (
    plot_raster,
    run_line_scan,
    run_raster_scan,
    save_scan_npz,
)


class _FakeTi:
    @staticmethod
    def sync():
        return None


class _FakeEngine:
    def __init__(self):
        self.ti = _FakeTi()
        self.calls = []

    def run_trapezoid(self, **kwargs):
        self.calls.append(dict(kwargs))
        n = int(kwargs["n"])
        roots = np.arange(n, dtype=np.int32)
        x = float(kwargs["nominal_x"])
        y = float(kwargs["nominal_y"])
        return {
            "incident_energy_vac_ev": float(kwargs["energy_vac_ev"]),
            "n_primaries": n,
            "emission_energy_ev": np.full(n, 900.0),
            "emission_root_primary_id": roots,
            "emission_generation": np.zeros(n, dtype=np.int32),
            "emission_is_cascade": np.zeros(n, dtype=np.int32),
            "emission_inelastic_count": np.zeros(n, dtype=np.int32),
            "launch_mean_x_angstrom": x,
            "launch_mean_y_angstrom": y,
            "launch_surface_z_mean_angstrom": -500.0,
            "local_incidence_mean_deg": 0.0,
            "local_incidence_sem_deg": 0.0,
            "allocated": n,
            "stored": n,
            "overflow": False,
            "wave_count": 1,
            "kernel_chunks": 1,
            "diagnostics": {
                "elastic_events": 0,
                "inelastic_events": 0,
                "emission_buffer_overflow": 0,
                "step_limit_hit": 0,
                "generation_limit_hit": 0,
                "omega_cdf_empty": 0,
                "q_cdf_empty": 0,
            },
        }


def _run_synthetic_raster():
    engine = _FakeEngine()
    geometry = TrapezoidGeometryConfig(
        top_width=500.0,
        bottom_width=700.0,
        height=500.0,
    )
    rows, per_primary, trajectories = run_raster_scan(
        engine,
        geometry,
        energy_ev=1000.0,
        x_positions_angstrom=[-10.0, 10.0],
        y_positions_angstrom=[-5.0, 5.0],
        primaries_per_pixel=3,
        progress=False,
        store_per_primary_counts=True,
    )
    return engine, rows, per_primary, trajectories


def test_raster_is_y_major_and_passes_nominal_y_to_engine():
    engine, rows, per_primary, trajectories = _run_synthetic_raster()
    assert [
        (call["nominal_x"], call["nominal_y"]) for call in engine.calls
    ] == [(-10.0, -5.0), (10.0, -5.0), (-10.0, 5.0), (10.0, 5.0)]
    assert [(row["pixel_id"], row["iy"], row["ix"]) for row in rows] == [
        (0, 0, 0),
        (1, 0, 1),
        (2, 1, 0),
        (3, 1, 1),
    ]
    assert per_primary["tey"].shape == (2, 2, 3)
    assert np.all(per_primary["tey"] == 1)
    assert trajectories == {}


def test_raster_npz_contains_canonical_2d_maps(tmp_path):
    _engine, rows, per_primary, _trajectories = _run_synthetic_raster()
    path = Path(tmp_path) / "raster.npz"
    save_scan_npz(
        path,
        rows,
        per_primary,
        metadata={"profile_channels": ["cascade_all", "primary_all", "tey"]},
    )
    with np.load(path, allow_pickle=False) as data:
        assert data["x_angstrom"].tolist() == [-10.0, 10.0]
        assert data["y_angstrom"].tolist() == [-5.0, 5.0]
        assert data["yield__tey"].shape == (2, 2)
        assert data["sem__tey"].shape == (2, 2)
        assert data["count__tey"].shape == (2, 2)
        assert data["tey_counts_per_primary"].shape == (2, 2, 3)
        assert np.all(data["count__tey"] == 3)
        assert np.all(data["yield__primary_all"] == 1.0)


def test_raster_can_discard_per_primary_vectors(tmp_path):
    engine = _FakeEngine()
    geometry = TrapezoidGeometryConfig(
        top_width=500.0,
        bottom_width=700.0,
        height=500.0,
    )
    rows, per_primary, _trajectories = run_raster_scan(
        engine,
        geometry,
        energy_ev=1000.0,
        x_positions_angstrom=[-10.0, 10.0],
        y_positions_angstrom=[-5.0, 5.0],
        primaries_per_pixel=3,
        progress=False,
        store_per_primary_counts=False,
    )
    assert per_primary["tey"].size == 0
    assert [row["count__tey"] for row in rows] == [3, 3, 3, 3]
    path = Path(tmp_path) / "compact_raster.npz"
    save_scan_npz(path, rows, per_primary, metadata={})
    with np.load(path, allow_pickle=False) as data:
        assert "tey_counts_per_primary" not in data.files
        assert np.all(data["count__tey"] == 3)


def test_raster_plot_smoke(tmp_path):
    _engine, rows, _per_primary, _trajectories = _run_synthetic_raster()
    path = Path(tmp_path) / "raster.png"
    plot_raster(path, rows, channels=("tey", "primary_all"))
    assert path.exists()
    assert path.stat().st_size > 0


def test_line_scan_wrapper_preserves_v08_per_primary_shape():
    engine = _FakeEngine()
    geometry = TrapezoidGeometryConfig(
        top_width=500.0,
        bottom_width=700.0,
        height=500.0,
    )
    _rows, per_primary, _trajectories = run_line_scan(
        engine,
        geometry,
        energy_ev=1000.0,
        x_positions_angstrom=[-10.0, 10.0],
        primaries_per_pixel=3,
        progress=False,
    )
    assert per_primary["tey"].shape == (2, 3)
