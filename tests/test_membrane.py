from dataclasses import replace
import numpy as np
import pytest
from seemc_taichi.trapezoid import TrapezoidGeometryConfig, TrapezoidTransportEngine
from seemc_taichi.bulk import BulkPhysicsConfig
from seemc_taichi.surface import SurfacePhysicsConfig
from test_tables import make_tables


def test_membrane_geometry():
    g = TrapezoidGeometryConfig(570, 780, 200, membrane_thickness=1130).validate()
    assert g.launch_surface(1000) == (0, 0, -1, 4)
    assert g.first_vacuum_hit(1000, -100, 0, 1) == (100, 0, -1, 4, -1)
    assert g.first_vacuum_hit(0, 1200, 0, -1) == (70, 0, 1, 5, -1)
    assert g.first_vacuum_hit(1000, 1200, 0, -1) == (70, 0, 1, 5, -1)
    assert g.first_vacuum_hit(0, 1200, 0, 1) is None
    for t in [0, -1, float('inf'), float('nan')]:
        with pytest.raises(ValueError):
            replace(g, membrane_thickness=t).validate()
    with pytest.raises(ValueError):
        replace(g, suspended=True).validate()


@pytest.mark.parametrize('precision', ['f32', 'f64'])
def test_membrane_ballistic(precision):
    ti = pytest.importorskip('taichi')
    ti.reset()
    fp = getattr(ti, precision)
    ti.init(arch=ti.cpu, default_fp=fp, random_seed=17, cpu_max_num_threads=2)
    host = make_tables([10., 100., 1000., 40000.])
    host = replace(host, inv_emfp=np.full(4, 1e-12), emfp=np.full(4, 1e12),
                   inv_imfp=np.zeros(4), inv_imfp_se=np.zeros(4), inv_imfp_pl=np.zeros(4))
    g = TrapezoidGeometryConfig(570, 780, 200, membrane_thickness=1130)
    e = TrapezoidTransportEngine(ti, fp, host, BulkPhysicsConfig(),
          SurfacePhysicsConfig(host.inner_potential_ev, 'classical', 0, True),
          g, capacity=128, trajectory_capacity=512)
    for energy in [900., 30000.]:
        for x in [0, 330, 1000]:  # ridge, slope, exposed membrane
            r = e.run_trapezoid(n=32, energy_vac_ev=energy, nominal_x=x,
                               copy_emissions=True, copy_state=True)
            assert r['gap_primary_fraction'] == 0
            assert np.all(r['emission_surface_code'] == 5)
            assert np.all(r['emission_uz'] > 0)
            np.testing.assert_allclose(r['emission_energy_ev'], energy, rtol=1e-6)
            np.testing.assert_allclose(r['z'], 1130, atol=.01)
            assert r['diagnostics']['surface_encounters'] == 32
    ti.reset()
