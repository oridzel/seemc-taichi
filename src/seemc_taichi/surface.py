"""SEEMC-Taichi v0.4 planar surface-barrier primitives.

This module ports the validated planar SEEMC solid/vacuum boundary physics
without yet coupling it to the mixed bulk collision kernel.  The separation is
intentional: outgoing escape/reflection and incoming primary
reflection/refraction can be validated independently before surface crossings
are introduced into the transport loop.

Plane convention used by the reference implementation:
    solid  : z > 0
    vacuum : z < 0
    outward normal : (0, 0, -1)
"""

import math
from dataclasses import dataclass

import numpy as np

H2EV = 27.21184
A0_ANG = 0.529177


@dataclass(frozen=True)
class SurfacePhysicsConfig:
    inner_potential_ev: float
    barrier_model: str = "abrupt"
    barrier_width_angstrom: float = 0.0
    incoming_barrier_reflection: bool = True

    @classmethod
    def from_sample(cls, sample):
        cfg = sample.cfg
        return cls(
            inner_potential_ev=float(sample.Ui),
            barrier_model=str(getattr(cfg, "barrier_model", "abrupt")),
            barrier_width_angstrom=float(getattr(cfg, "barrier_width", 0.0)),
            incoming_barrier_reflection=bool(
                getattr(cfg, "incoming_barrier_reflection", True)
            ),
        )

    def validate(self):
        model = {"quantum": "abrupt", "sigmoid": "expqm", "jmonsel": "expqm"}.get(
            self.barrier_model, self.barrier_model
        )
        if model not in {"abrupt", "classical", "expqm"}:
            raise ValueError(f"unsupported barrier_model: {self.barrier_model}")
        if not math.isfinite(self.inner_potential_ev) or self.inner_potential_ev < 0.0:
            raise ValueError("inner_potential_ev must be finite and non-negative")
        if model == "expqm" and self.barrier_width_angstrom <= 0.0:
            raise ValueError("expqm requires barrier_width_angstrom > 0")
        return model


def _sinh_ratio_host(a, b):
    if b <= 0.0:
        return 1.0
    if a <= 0.0:
        return 0.0
    d = a - b
    if d < -700.0:
        return 0.0
    return math.exp(d) * math.expm1(-2.0 * a) / math.expm1(-2.0 * b)


def barrier_transmission_host(E_perp, physics: SurfacePhysicsConfig):
    """Solid -> vacuum transmission for perpendicular solid-state energy."""
    model = physics.validate()
    Ui = float(physics.inner_potential_ev)
    E_perp = float(E_perp)
    if E_perp <= Ui:
        return 0.0
    if model == "classical":
        return 1.0
    k1 = math.sqrt(2.0 * E_perp / H2EV) / A0_ANG
    k2 = math.sqrt(2.0 * (E_perp - Ui) / H2EV) / A0_ANG
    if model == "abrupt":
        return 4.0 * k1 * k2 / ((k1 + k2) ** 2)
    w = float(physics.barrier_width_angstrom)
    r = _sinh_ratio_host(
        0.5 * math.pi * w * (k1 - k2),
        0.5 * math.pi * w * (k1 + k2),
    )
    return max(0.0, min(1.0, 1.0 - r * r))


def incoming_barrier_transmission_host(E_perp_vac, physics: SurfacePhysicsConfig):
    """Vacuum -> solid reciprocal barrier transmission."""
    model = physics.validate()
    Ui = float(physics.inner_potential_ev)
    E_perp_vac = float(E_perp_vac)
    if model == "classical" or Ui == 0.0 or not physics.incoming_barrier_reflection:
        return 1.0
    k_v = math.sqrt(max(2.0 * E_perp_vac / H2EV, 0.0)) / A0_ANG
    k_s = math.sqrt(max(2.0 * (E_perp_vac + Ui) / H2EV, 0.0)) / A0_ANG
    if model == "abrupt":
        denom = (k_v + k_s) ** 2
        return 0.0 if denom <= 0.0 else max(0.0, min(1.0, 4.0 * k_v * k_s / denom))
    w = float(physics.barrier_width_angstrom)
    r = _sinh_ratio_host(
        0.5 * math.pi * w * abs(k_s - k_v),
        0.5 * math.pi * w * (k_s + k_v),
    )
    return max(0.0, min(1.0, 1.0 - r * r))


def analytic_isotropic_escape_probability(E_s, physics: SurfacePhysicsConfig, n_mu=20001):
    """0.5*integral_0^1 T(E_s mu^2)dmu, matching reference validation."""
    mu = np.linspace(0.0, 1.0, int(n_mu))
    vals = np.array(
        [barrier_transmission_host(float(E_s) * float(m) * float(m), physics) for m in mu],
        dtype=float,
    )
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return 0.5 * float(trapz(vals, mu))


class SurfaceSamples:
    def __init__(self, ti, capacity, fp):
        self.escaped = ti.field(dtype=ti.i32, shape=capacity)
        self.reflected = ti.field(dtype=ti.i32, shape=capacity)
        self.energy_out = ti.field(dtype=fp, shape=capacity)
        self.ux = ti.field(dtype=fp, shape=capacity)
        self.uy = ti.field(dtype=fp, shape=capacity)
        self.uz = ti.field(dtype=fp, shape=capacity)


class SurfaceCounters:
    def __init__(self, ti):
        names = (
            "surface_encounters", "escapes", "internal_reflections",
            "incoming_barrier_encounters", "incoming_barrier_reflections",
            "incoming_barrier_transmissions",
        )
        self.names = names
        for name in names:
            setattr(self, name, ti.field(dtype=ti.i32, shape=()))

    def as_dict(self):
        return {name: int(getattr(self, name)[None]) for name in self.names}


def build_surface_kernels(ti, fp, physics, samples, counters):
    model = physics.validate()
    Ui = float(physics.inner_potential_ev)
    w = float(physics.barrier_width_angstrom)
    incoming_reflection = bool(physics.incoming_barrier_reflection)
    is_classical = model == "classical"
    is_abrupt = model == "abrupt"

    @ti.func
    def sinh_ratio(a: fp, b: fp) -> fp:
        result = fp(1.0)
        if b > fp(0.0):
            result = fp(0.0)
            if a > fp(0.0):
                d = a - b
                if d >= fp(-80.0):
                    result = ti.exp(d) * ti.expm1(-fp(2.0) * a) / ti.expm1(-fp(2.0) * b)
        return result

    @ti.func
    def barrier_T(E_perp: fp) -> fp:
        T = fp(0.0)
        if E_perp > fp(Ui):
            if ti.static(is_classical):
                T = fp(1.0)
            else:
                k1 = ti.sqrt(fp(2.0) * E_perp / fp(H2EV)) / fp(A0_ANG)
                k2 = ti.sqrt(fp(2.0) * (E_perp - fp(Ui)) / fp(H2EV)) / fp(A0_ANG)
                if ti.static(is_abrupt):
                    denom = (k1 + k2) * (k1 + k2)
                    if denom > fp(0.0):
                        T = fp(4.0) * k1 * k2 / denom
                else:
                    a = fp(0.5 * math.pi * w) * (k1 - k2)
                    b = fp(0.5 * math.pi * w) * (k1 + k2)
                    r = sinh_ratio(a, b)
                    T = ti.max(fp(0.0), ti.min(fp(1.0), fp(1.0) - r * r))
        return T

    @ti.func
    def incoming_T(E_perp_vac: fp) -> fp:
        T = fp(1.0)
        if ti.static(incoming_reflection and (not is_classical) and Ui > 0.0):
            kv = ti.sqrt(ti.max(fp(2.0) * E_perp_vac / fp(H2EV), fp(0.0))) / fp(A0_ANG)
            ks = ti.sqrt(ti.max(fp(2.0) * (E_perp_vac + fp(Ui)) / fp(H2EV), fp(0.0))) / fp(A0_ANG)
            if ti.static(is_abrupt):
                denom = (kv + ks) * (kv + ks)
                T = fp(0.0)
                if denom > fp(0.0):
                    T = fp(4.0) * kv * ks / denom
            else:
                a = fp(0.5 * math.pi * w) * ti.abs(ks - kv)
                b = fp(0.5 * math.pi * w) * (ks + kv)
                r = sinh_ratio(a, b)
                T = ti.max(fp(0.0), ti.min(fp(1.0), fp(1.0) - r * r))
        return T

    @ti.kernel
    def reset():
        counters.surface_encounters[None] = ti.i32(0)
        counters.escapes[None] = ti.i32(0)
        counters.internal_reflections[None] = ti.i32(0)
        counters.incoming_barrier_encounters[None] = ti.i32(0)
        counters.incoming_barrier_reflections[None] = ti.i32(0)
        counters.incoming_barrier_transmissions[None] = ti.i32(0)

    @ti.kernel
    def outgoing_isotropic(n: ti.i32, E_s: fp):
        for i in range(n):
            samples.escaped[i] = ti.i32(0)
            samples.reflected[i] = ti.i32(0)
            samples.energy_out[i] = E_s
            mu = fp(2.0) * ti.random(fp) - fp(1.0)
            phi = fp(2.0 * math.pi) * ti.random(fp)
            st = ti.sqrt(ti.max(fp(0.0), fp(1.0) - mu * mu))
            ux = st * ti.cos(phi)
            uy = st * ti.sin(phi)
            uz = mu
            samples.ux[i] = ux
            samples.uy[i] = uy
            samples.uz[i] = uz

            # Outward normal is -z; only uz < 0 is an outward encounter.
            if uz < fp(0.0):
                ti.atomic_add(counters.surface_encounters[None], ti.i32(1))
                Eperp = E_s * uz * uz
                T = barrier_T(Eperp)
                if T > fp(0.0) and (T >= fp(1.0) or ti.random(fp) < T):
                    Ev = E_s - fp(Ui)
                    scale = ti.sqrt(E_s / Ev)
                    ux2 = ux * scale
                    uy2 = uy * scale
                    uz2 = -ti.sqrt(ti.max(fp(0.0), fp(1.0) - ux2 * ux2 - uy2 * uy2))
                    samples.ux[i] = ux2
                    samples.uy[i] = uy2
                    samples.uz[i] = uz2
                    samples.energy_out[i] = Ev
                    samples.escaped[i] = ti.i32(1)
                    ti.atomic_add(counters.escapes[None], ti.i32(1))
                else:
                    samples.uz[i] = -uz
                    samples.reflected[i] = ti.i32(1)
                    ti.atomic_add(counters.internal_reflections[None], ti.i32(1))

    @ti.kernel
    def incoming_fixed(n: ti.i32, E_vac: fp, alpha_rad: fp):
        sin_a = ti.sin(alpha_rad)
        cos_a = ti.cos(alpha_rad)
        Eperp = E_vac * cos_a * cos_a
        Es = E_vac + fp(Ui)
        solid_sin = ti.sqrt(ti.max(E_vac, fp(0.0)) / Es) * sin_a
        solid_sin = ti.min(solid_sin, fp(1.0))
        solid_cos = ti.sqrt(ti.max(fp(0.0), fp(1.0) - solid_sin * solid_sin))
        T = incoming_T(Eperp)
        for i in range(n):
            ti.atomic_add(counters.incoming_barrier_encounters[None], ti.i32(1))
            samples.escaped[i] = ti.i32(0)
            samples.reflected[i] = ti.i32(0)
            if T < fp(1.0) and ti.random(fp) >= T:
                # Specularly reflected back into vacuum: z component changes sign.
                samples.ux[i] = sin_a
                samples.uy[i] = fp(0.0)
                samples.uz[i] = -cos_a
                samples.energy_out[i] = E_vac
                samples.reflected[i] = ti.i32(1)
                ti.atomic_add(counters.incoming_barrier_reflections[None], ti.i32(1))
            else:
                samples.ux[i] = solid_sin
                samples.uy[i] = fp(0.0)
                samples.uz[i] = solid_cos
                samples.energy_out[i] = Es
                ti.atomic_add(counters.incoming_barrier_transmissions[None], ti.i32(1))

    return reset, outgoing_isotropic, incoming_fixed


class SurfaceBarrierEngine:
    def __init__(self, ti, fp, physics: SurfacePhysicsConfig, capacity: int):
        self.ti = ti
        self.fp = fp
        self.physics = physics
        self.capacity = int(capacity)
        self.samples = SurfaceSamples(ti, self.capacity, fp)
        self.counters = SurfaceCounters(ti)
        self.reset_kernel, self.outgoing_kernel, self.incoming_kernel = build_surface_kernels(
            ti, fp, physics, self.samples, self.counters
        )

    def _copy(self, n):
        n = int(n)
        return {
            "escaped": self.samples.escaped.to_numpy()[:n],
            "reflected": self.samples.reflected.to_numpy()[:n],
            "energy_out_ev": self.samples.energy_out.to_numpy()[:n],
            "ux": self.samples.ux.to_numpy()[:n],
            "uy": self.samples.uy.to_numpy()[:n],
            "uz": self.samples.uz.to_numpy()[:n],
            "diagnostics": self.counters.as_dict(),
        }

    def sample_outgoing_isotropic(self, n, energy_s_ev, copy=True):
        if int(n) > self.capacity:
            raise ValueError("n exceeds capacity")
        self.reset_kernel()
        self.outgoing_kernel(int(n), float(energy_s_ev))
        self.ti.sync()
        if copy:
            return self._copy(n)
        return {"diagnostics": self.counters.as_dict()}

    def sample_incoming(self, n, energy_vac_ev, alpha_deg=0.0, copy=True):
        if int(n) > self.capacity:
            raise ValueError("n exceeds capacity")
        self.reset_kernel()
        self.incoming_kernel(int(n), float(energy_vac_ev), math.radians(float(alpha_deg)))
        self.ti.sync()
        if copy:
            return self._copy(n)
        return {"diagnostics": self.counters.as_dict()}
