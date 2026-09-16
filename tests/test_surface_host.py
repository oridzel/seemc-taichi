import math

from seemc_taichi.surface import (
    SurfacePhysicsConfig,
    analytic_isotropic_escape_probability,
    barrier_transmission_host,
    incoming_barrier_transmission_host,
)


def test_classical_limits():
    p = SurfacePhysicsConfig(13.4, "classical", 0.0, True)
    assert barrier_transmission_host(13.4, p) == 0.0
    assert barrier_transmission_host(20.0, p) == 1.0
    assert incoming_barrier_transmission_host(0.0, p) == 1.0


def test_abrupt_transmission_bounds():
    p = SurfacePhysicsConfig(13.4, "abrupt", 0.0, True)
    for e in (13.5, 20.0, 100.0, 1000.0):
        t = barrier_transmission_host(e, p)
        assert 0.0 <= t <= 1.0
    assert 0.0 <= incoming_barrier_transmission_host(500.0, p) <= 1.0


def test_expqm_between_abrupt_and_classical():
    abrupt = SurfacePhysicsConfig(13.4, "abrupt", 0.0, True)
    expqm = SurfacePhysicsConfig(13.4, "expqm", 1.0, True)
    e = 50.0
    ta = barrier_transmission_host(e, abrupt)
    te = barrier_transmission_host(e, expqm)
    assert 0.0 <= ta <= te <= 1.0


def test_isotropic_escape_probability_bounds():
    p = SurfacePhysicsConfig(13.4, "abrupt", 0.0, True)
    prob = analytic_isotropic_escape_probability(100.0, p, n_mu=1001)
    assert 0.0 < prob < 0.5
