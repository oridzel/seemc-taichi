from pathlib import Path

import numpy as np

from seemc_taichi.scan_archive import RasterTrajectoryArchive, archive_from_scan
from seemc_taichi.scan_anim_cli import animate_trapezoidal_scan


def _synthetic_archive():
    channels = ("cascade_all", "primary_all", "tey")
    rows = []
    trajectories = {}
    for pixel_id, x_nm in enumerate((-50.0, 50.0)):
        rows.append({
            "x_nm": x_nm,
            "cascade_all": 1.0 + pixel_id,
            "cascade_all_sem": 0.1,
            "primary_all": 0.5,
            "primary_all_sem": 0.05,
            "tey": 1.5 + pixel_id,
            "tey_sem": 0.12,
        })
        trajectories[pixel_id] = {
            "trajectory_x_angstrom": np.asarray([10.0 * x_nm, 10.0 * x_nm + 5.0]),
            "trajectory_y_angstrom": np.zeros(2),
            "trajectory_z_angstrom": np.asarray([-500.0, -520.0]),
            "trajectory_energy_ev": np.asarray([1000.0, 800.0]),
            "trajectory_time_fs": np.asarray([0.0, 0.01]),
            "trajectory_electron_id": np.asarray([0, 0], dtype=np.int32),
            "trajectory_parent_id": np.asarray([-1, -1], dtype=np.int32),
            "trajectory_root_primary_id": np.asarray([0, 0], dtype=np.int32),
            "trajectory_generation": np.asarray([0, 0], dtype=np.int32),
            "trajectory_inelastic_count": np.asarray([0, 1], dtype=np.int32),
            "trajectory_event": np.asarray([1, 9], dtype=np.int32),
            "trajectory_surface_code": np.asarray([1, 1], dtype=np.int32),
            "trajectory_step": np.asarray([0, 1], dtype=np.int32),
            "emission_electron_id": np.asarray([0], dtype=np.int32),
            "emission_root_primary_id": np.asarray([0], dtype=np.int32),
            "emission_energy_ev": np.asarray([800.0]),
            "emission_ux": np.asarray([0.2]),
            "emission_uy": np.asarray([0.0]),
            "emission_uz": np.asarray([-0.98]),
            "emission_generation": np.asarray([0], dtype=np.int32),
            "emission_inelastic_count": np.asarray([1], dtype=np.int32),
        }
    metadata = {
        "profile_channels": list(channels),
        "energy_ev": 1000.0,
        "sample_name": "Si",
        "lle_max_loss_ev": 50.0,
        "config": {"energy_ev": 1000.0},
        "top_width_nm": 50.0,
        "bottom_width_nm": 70.0,
        "height_nm": 50.0,
        "geometry": {
            "type": "TrapezoidalLineArray",
            "top_width": 500.0,
            "bottom_width": 700.0,
            "height": 500.0,
            "center_x": 0.0,
            "substrate_z": 0.0,
            "n_lines": 3,
            "pitch": 1000.0,
            "line_centers": [-1000.0, 0.0, 1000.0],
        },
    }
    return archive_from_scan(rows, trajectories, metadata=metadata)


def test_combined_archive_round_trip(tmp_path):
    path = Path(tmp_path) / "scan.trajectories.npz"
    original = _synthetic_archive()
    original.save_npz(path)
    loaded = RasterTrajectoryArchive.load_npz(path)
    assert loaded.n_points == 4
    assert loaded.n_cascades == 2
    assert loaded.recorded_pixels.tolist() == [0, 1]
    assert loaded.profile_channels.tolist() == [
        "cascade_all", "primary_all", "tey"
    ]
    assert np.allclose(loaded.profile_yields[:, 0, 0], [1.0, 0.5, 1.5])


def test_whole_scan_animation_smoke(tmp_path):
    output = Path(tmp_path) / "scan.gif"
    animate_trapezoidal_scan(
        _synthetic_archive(),
        output,
        fps=2,
        frames_per_pixel=2,
        pause_frames=0,
        color_by="energy",
        vacuum_flight_nm=2.0,
        dpi=40,
    )
    assert output.exists()
    assert output.stat().st_size > 0
