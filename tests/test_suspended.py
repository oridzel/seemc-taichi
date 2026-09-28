import math
from dataclasses import replace

import numpy as np
import pytest

from seemc_taichi.trapezoid import TrapezoidGeometryConfig, TrapezoidTransportEngine
from seemc_taichi.trapezoid_scan import classify_result, nominal_surface_info
from seemc_taichi.bulk import BulkPhysicsConfig
from seemc_taichi.surface import SurfacePhysicsConfig
from test_tables import make_tables


def test_suspended_host_intersections():
    g = TrapezoidGeometryConfig(500, 520, 300, n_lines=3, pitch=1000,
                                suspended=True)
    assert g.launch_surface(500)[3] == 0
    assert nominal_surface_info(g, 500)[1] == 'vacuum_gap'
    assert math.isnan(nominal_surface_info(g, 500)[2])
    # A primary in a gap sees no material down to infinity.
    assert g.first_vacuum_hit(500, -500, 0, 1) is None
    # The underside has the opposite normal to the supported substrate.
    assert g.first_vacuum_hit(0, 100, 0, -1) == (100, 0, 1, 5, 1)
    assert g.first_vacuum_hit(0, 0, 0, 1) is None
    # A vacuum electron can still hit an adjacent suspended line.
    hit = g.first_vacuum_hit(-700, -150, 1, 0)
    assert hit[3:] == (3, 1)
    assert hit[0] == pytest.approx(445)
    supported = replace(g, suspended=False)
    assert supported.launch_surface(500)[3] == 4
    assert supported.first_vacuum_hit(500, -500, 0, 1)[3] == 4


@pytest.mark.parametrize('precision', ['f64', 'f32'])
def test_suspended_ballistic_transport_and_gap(precision):
    ti = pytest.importorskip('taichi')
    ti.reset()
    fp = getattr(ti, precision)
    ti.init(arch=ti.cpu, default_fp=fp, random_seed=17, cpu_max_num_threads=2)
    # Synthetic tables isolate boundary transport, not material physics:
    # huge mean free path, no inelastic scattering, classical interface.
    host = make_tables([10., 100., 1000., 40000.])
    host = replace(host, inv_emfp=np.full(4, 1e-12), emfp=np.full(4, 1e12),
                   inv_imfp=np.zeros(4), inv_imfp_se=np.zeros(4),
                   inv_imfp_pl=np.zeros(4))
    g = TrapezoidGeometryConfig(500, 500, 300, suspended=True)
    e = TrapezoidTransportEngine(ti, fp, host, BulkPhysicsConfig(),
            SurfacePhysicsConfig(host.inner_potential_ev, 'classical', 0, True),
            g, capacity=128, trajectory_capacity=512)
    for energy in [900., 30000.]:
        r = e.run_trapezoid(n=32, energy_vac_ev=energy, nominal_x=0,
                           copy_emissions=True, copy_state=True)
        assert r['gap_primary_fraction'] == 0
        assert np.all(r['emission_surface_code'] == 5)
        assert np.all(r['emission_uz'] > 0)
        np.testing.assert_allclose(r['emission_energy_ev'], energy, rtol=1e-6)
        assert np.all(r['alive'] == 0)
        assert np.allclose(r['z'], 0)
        c = classify_result(r)
        assert c['forward_primary_all'] == 1
        assert c['back_bsey_50ev'] == 0
        assert c['sey_50ev'] == 0
        assert r['diagnostics']['surface_encounters'] == 32
        assert r['diagnostics']['internal_reflections'] == 0

    r = e.run_trapezoid(n=32, energy_vac_ev=30000, nominal_x=500,
                       copy_emissions=True)
    assert r['gap_primary_fraction'] == 1
    assert r['tey'] == 0
    assert len(r['emission_energy_ev']) == 0
    assert r['diagnostics']['incoming_barrier_encounters'] == 0
    assert classify_result(r)['forward_primary_all'] == 0

    # A Gaussian beam centered at an edge splits between material and vacuum.
    r = e.run_trapezoid(n=100, energy_vac_ev=900, nominal_x=250,
                       beam_fwhm=10, seed=12, copy_emissions=True)
    assert 0 < r['gap_primary_fraction'] < 1
    assert r['tey'] + r['gap_primary_fraction'] == pytest.approx(1)
    assert np.all(r['emission_surface_code'] == 5)
    ti.reset()
