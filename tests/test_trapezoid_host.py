import math
import numpy as np

from seemc_taichi.trapezoid import TrapezoidGeometryConfig
from seemc_taichi.trapezoid_scan import (
    emission_masks,
    counts_per_primary,
    yield_and_sem,
    nominal_surface_info,
)


def test_trapezoid_launch_surfaces_and_normals():
    g = TrapezoidGeometryConfig(top_width=500.0, bottom_width=700.0, height=500.0)

    z, nx, nz, code = g.launch_surface(0.0)
    assert code == 1
    assert z == -500.0
    assert (nx, nz) == (0.0, -1.0)

    z, nx, nz, code = g.launch_surface(300.0)
    assert code == 2
    assert -500.0 < z < 0.0
    assert nx > 0.0 and nz < 0.0
    assert math.isclose(nx * nx + nz * nz, 1.0, rel_tol=0, abs_tol=1e-14)

    zl, nxl, nzl, codel = g.launch_surface(-300.0)
    assert codel == 3
    assert math.isclose(zl, z)
    assert math.isclose(nxl, -nx)
    assert math.isclose(nzl, nz)

    z, nx, nz, code = g.launch_surface(400.0)
    assert code == 4
    assert z == 0.0
    assert (nx, nz) == (0.0, -1.0)


def test_nominal_local_incidence():
    g = TrapezoidGeometryConfig(top_width=500.0, bottom_width=700.0, height=500.0)
    _, name, angle = nominal_surface_info(g, 0.0)
    assert name == "top"
    assert angle == 0.0
    _, name, angle = nominal_surface_info(g, 300.0)
    assert name == "right_sidewall"
    expected = math.degrees(math.atan2(500.0, 100.0))
    assert math.isclose(angle, expected, rel_tol=0, abs_tol=1e-12)


def test_centered_three_line_array_and_neighbor_hit():
    g = TrapezoidGeometryConfig(
        top_width=500.0,
        bottom_width=700.0,
        height=500.0,
        n_lines=3,
        pitch=1000.0,
    ).validate()
    assert g.line_centers == (-1000.0, 0.0, 1000.0)
    assert g.span == 2700.0

    # Every line center launches on a top face; the midpoint of a trench is
    # exposed substrate.
    for center in g.line_centers:
        z, nx, nz, code = g.launch_surface(center)
        assert (z, nx, nz, code) == (-500.0, 0.0, -1.0, 1)
    assert g.launch_surface(-500.0) == (0.0, 0.0, -1.0, 4)

    # A lateral vacuum ray leaving the right wall of the left line reaches the
    # left wall of the central line.  This is the interaction the old
    # one-trapezoid Taichi geometry missed.
    hit = g.first_vacuum_hit(-650.0, -250.0, 1.0, 0.0)
    assert hit is not None
    distance, nx, nz, code, line_index = hit
    assert math.isclose(distance, 350.0, rel_tol=0, abs_tol=1e-12)
    assert line_index == 1
    assert code == 3
    assert nx < 0.0 and nz < 0.0

    # A ray travelling away from the complete structure escapes.
    assert g.first_vacuum_hit(0.0, -500.0, 0.0, -1.0) is None


def synthetic_result():
    # 4 primaries, 8 emissions.  Cascade entries deliberately exercise the
    # current rule: only generation=1 with own inelastic_count=0 is SE1.
    return {
        "incident_energy_vac_ev": 1000.0,
        "n_primaries": 4,
        "emission_energy_ev": np.array([900, 20, 15, 60, 10, 800, 30, 25.0]),
        "emission_root_primary_id": np.array([0, 0, 1, 1, 2, 2, 3, 3]),
        "emission_generation": np.array([0, 1, 1, 1, 2, 0, 2, 1], np.int32),
        "emission_is_cascade": np.array([0, 1, 1, 1, 1, 0, 1, 1], np.int32),
        "emission_inelastic_count": np.array([2, 0, 1, 0, 0, 4, 1, 0], np.int32),
    }


def test_current_se1_se2_definition_partitions_cascade():
    r = synthetic_result()
    masks = emission_masks(r, cutoff_ev=50.0, lle_max_loss_ev=50.0)
    assert np.flatnonzero(masks["se1"]).tolist() == [1, 3, 7]
    assert np.flatnonzero(masks["se2"]).tolist() == [2, 4, 6]
    assert np.array_equal(masks["cascade_all"], masks["se1"] | masks["se2"])


def test_per_primary_counts_and_sem():
    r = synthetic_result()
    masks = emission_masks(r)
    counts = counts_per_primary(r, masks["tey"])
    assert counts.tolist() == [2, 2, 2, 2]
    mean, sem = yield_and_sem(counts)
    assert mean == 2.0
    assert sem == 0.0
