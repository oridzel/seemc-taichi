"""Layered planar SEEMC transport on Taichi: a thin film on a semi-infinite substrate.

Geometry (same convention as ``plane.py``): vacuum z < 0, film 0 < z < t,
substrate z > t, outward normal (0, 0, -1).

Each layer uses the collision physics from ``plane.py``. The block between
the two marker comments below comes from ``build_plane_kernels``
(seemc-taichi 0.9.3) and is instantiated once per material:

  * a secondary electron is born in the layer of its parent;
  * the "cannot escape, do not queue" threshold of a secondary is the layer's
    ``kill_energy`` instead of ``Ui`` (identical to ``Ui`` in the film).
  * the inelastic sampling retries run in a loop to keep the compiled kernel
    smaller.

What is new is the film/substrate interface at z = t:

  * Energies stay ``E_s`` (measured from the valence-band bottom of the layer
    the electron is in).  Crossing from film to substrate changes it by
    ``dV = E_s(substrate) - E_s(film)`` at fixed total energy:
        alignment "vacuum":  dV = Ui(substrate) - Ui(film)   (common vacuum level)
        alignment "fermi":   dV = E_F(substrate) - E_F(film) (common Fermi level)
  * The momentum parallel to the interface is conserved (refraction), and the
    electron is specularly reflected when its perpendicular energy cannot
    supply -dV, or stochastically with the quantum reflection probability of
    the potential step ("abrupt": 4 k1 k2/(k1+k2)^2; "expqm": the same smooth
    step as the vacuum barrier, with its own width; "classical": T = 1).
  * As at the vacuum surface, a leg truncated at the interface has no
    collision; the next leg draws a fresh free path in the new layer.
  * Only the vacuum surface z = 0 (film side) emits.  With
    ``track_subbarrier=False`` an electron is terminated once its energy is
    below the vacuum level of the film surface, in either layer.
"""

import math
from types import SimpleNamespace

import numpy as np

from .device_tables import DeviceTransportTables
from .bulk import BulkPhysicsConfig, InelasticSamples
from .particle_pool import ParticlePool
from .primitives import make_primitives
from .surface import SurfacePhysicsConfig
from .plane import (
    PlaneCounters, EmissionBuffer, PlaneHistory, PlaneTransportEngine,
    H2EV, C_AU, A0_ANG,
)

LAYERED_COUNTERS = (
    "interface_encounters", "interface_transmissions", "interface_reflections",
    "film_to_substrate", "substrate_to_film", "terminated_below_vacuum_level",
)


class LayeredPlaneCounters(PlaneCounters):
    def __init__(self, ti):
        super().__init__(ti)
        for name in LAYERED_COUNTERS:
            setattr(self, name, ti.field(dtype=ti.i32, shape=()))
        self.names = tuple(self.names) + LAYERED_COUNTERS


def interface_step_ev(film_tables, substrate_tables, alignment="vacuum"):
    """Kinetic-energy change dV [eV] of an electron going film -> substrate."""
    if alignment == "vacuum":
        return float(substrate_tables.inner_potential_ev) - float(film_tables.inner_potential_ev)
    if alignment == "fermi":
        return float(substrate_tables.e_fermi_ev) - float(film_tables.e_fermi_ev)
    raise ValueError("interface alignment must be 'vacuum' or 'fermi'")


def _make_material_physics(
    ti, fp, pool, layer, tables, physics, surface, surface_model, counters,
    samples, emissions, history, bse_cutoff_ev, kill_energy,
):
    """Instantiate the plane.py collision physics for one material."""
    physics.validate()
    (
        _, interp1, stochastic_energy_bin, inverse_cdf_column,
        _, interp_matrix_column, inverse_cdf_matrix_column, interp2_bilinear,
    ) = make_primitives(ti, fp)
    kill_energy = float(kill_energy)

    # ---- BEGIN collision physics from plane.build_plane_kernels (0.9.3) ----
    is_metal = bool(tables.is_metal)
    vacuum_ref = physics.emfp_energy_ref == "vacuum"
    fermi_imfp_ref = physics.imfp_energy_ref == "fermi"
    low_model = physics.elastic_low_energy_model
    low_elsepa = low_model == "elsepa"
    low_linear = low_model == "linear"
    low_browning = low_model == "browning"
    use_mao = physics.se_channel_rule == "mao"
    pauli_fallback = physics.on_pauli_block == "fallback"
    binary_isotropic = physics.se_direction_model == "isotropic"
    plasmon_isotropic = physics.plasmon_se_direction == "isotropic"
    semiconductor_isotropic = physics.semiconductor_se_direction == "isotropic"
    track_subbarrier = bool(physics.track_subbarrier)

    ui = float(tables.inner_potential_ev)
    ef = float(tables.e_fermi_ev)
    ef_feg = float(tables.e_fermi_feg_ev)
    evb = float(tables.e_vb_ev)
    gap = float(tables.band_gap_ev)
    ecbm = float(tables.e_cbm_ev)
    kf = math.sqrt(max(2.0 * ef_feg / H2EV, 0.0))
    z_eff = float(physics.atomic_number)
    z17 = z_eff ** 1.7 if z_eff > 0.0 else 0.0
    elastic_min = float(physics.elastic_min_energy_ev)
    cutoff = float(physics.elastic_cutoff_energy_ev)
    max_generation = int(physics.max_generation)
    n_q_sample = int(physics.n_q_sample)
    two_pi = float(2.0 * math.pi)
    barrier_width = float(surface.barrier_width_angstrom)
    incoming_reflection = bool(surface.incoming_barrier_reflection)
    barrier_classical = surface_model == "classical"
    barrier_abrupt = surface_model == "abrupt"
    bse_cutoff = float(bse_cutoff_ev)

    @ti.func
    def clamp_energy_grid(x: fp) -> fp:
        y = ti.max(x, tables.energy[0])
        y = ti.min(y, tables.energy[tables.nE - 1])
        return y

    @ti.func
    def emfp_abscissa(E_s: fp) -> fp:
        x = E_s
        if ti.static(vacuum_ref):
            x = ti.max(E_s - fp(ui), fp(elastic_min))
        return clamp_energy_grid(x)

    @ti.func
    def imfp_abscissa(E_s: fp) -> fp:
        x = E_s
        if ti.static(fermi_imfp_ref):
            x = E_s - fp(ef)
        return clamp_energy_grid(x)

    @ti.func
    def browning_sigma(E_ev: fp) -> fp:
        Ekev = ti.max(E_ev, fp(1.0e-9)) / fp(1000.0)
        rt = ti.sqrt(Ekev)
        value = fp(3.0e-18) * fp(z17) / (
            Ekev + fp(0.005) * fp(z17) * rt
            + fp(0.0007) * fp(z_eff * z_eff) / rt
        )
        return value

    @ti.func
    def elastic_sigma_scale(E_s: fp) -> fp:
        scale = fp(1.0)
        if ti.static(low_linear):
            if E_s < fp(cutoff):
                if E_s <= fp(ef):
                    scale = fp(0.0)
                else:
                    scale = (E_s - fp(ef)) / ti.max(fp(cutoff - ef), fp(1.0e-9))
        elif ti.static(low_browning):
            if E_s < fp(cutoff):
                scale = browning_sigma(E_s) / browning_sigma(fp(cutoff))
        return scale

    @ti.func
    def inverse_emfp(E_s: fp) -> fp:
        inv_e = fp(0.0)
        if ti.static(low_elsepa):
            inv_e = interp1(tables.energy, tables.inv_emfp, ti.i32(tables.nE), emfp_abscissa(E_s))
        else:
            if E_s >= fp(cutoff):
                inv_e = interp1(tables.energy, tables.inv_emfp, ti.i32(tables.nE), emfp_abscissa(E_s))
            else:
                ec = emfp_abscissa(fp(cutoff))
                inv_c = interp1(tables.energy, tables.inv_emfp, ti.i32(tables.nE), ec)
                inv_e = inv_c * elastic_sigma_scale(E_s)
        return inv_e

    @ti.func
    def inelastic_threshold() -> fp:
        threshold = fp(ef)
        if ti.static(not is_metal):
            threshold = fp(evb + 2.0 * gap)
        return threshold

    @ti.func
    def inverse_imfp(E_s: fp) -> fp:
        inv_i = fp(0.0)
        if E_s > inelastic_threshold():
            inv_i = interp1(tables.energy, tables.inv_imfp, ti.i32(tables.nE), imfp_abscissa(E_s))
        return inv_i

    @ti.func
    def dcs_abscissa(E_s: fp) -> fp:
        x = E_s
        if ti.static(not low_elsepa):
            x = ti.max(E_s, fp(cutoff))
        return emfp_abscissa(x)

    @ti.func
    def rotate_direction(ux: fp, uy: fp, uz: fp, polar: fp, azimuth: fp):
        sin_psi = ti.sin(polar)
        cos_psi = ti.cos(polar)
        sin_fi = ti.sin(azimuth)
        cos_fi = ti.cos(azimuth)
        cos_theta = uz
        sin_theta = ti.sqrt(ti.max(ux * ux + uy * uy, fp(0.0)))
        cos_phi = fp(1.0)
        sin_phi = fp(0.0)
        if sin_theta > fp(1.0e-12):
            cos_phi = ux / sin_theta
            sin_phi = uy / sin_theta
        h0 = sin_psi * cos_fi
        h1 = sin_theta * cos_psi + h0 * cos_theta
        h2 = sin_psi * sin_fi
        ox = h1 * cos_phi - h2 * sin_phi
        oy = h1 * sin_phi + h2 * cos_phi
        oz = cos_theta * cos_psi - h0 * sin_theta
        norm = ti.sqrt(ti.max(ox * ox + oy * oy + oz * oz, fp(1.0e-30)))
        return ox / norm, oy / norm, oz / norm

    @ti.func
    def isotropic_direction():
        mu = fp(2.0) * ti.random(fp) - fp(1.0)
        st = ti.sqrt(ti.max(fp(1.0) - mu * mu, fp(0.0)))
        phi = fp(two_pi) * ti.random(fp)
        return st * ti.cos(phi), st * ti.sin(phi), mu

    @ti.func
    def k_rel(E_ev: fp) -> fp:
        e = ti.max(E_ev, fp(0.0)) / fp(H2EV)
        k = ti.sqrt(e * (fp(2.0) + e / fp(C_AU * C_AU)))
        return k

    @ti.func
    def choose_channel(E_s: fp) -> ti.i32:
        ch = ti.i32(0)
        if E_s > inelastic_threshold():
            x = imfp_abscissa(E_s)
            inv_pl = interp1(tables.energy, tables.inv_imfp_pl, ti.i32(tables.nE), x)
            inv_se = interp1(tables.energy, tables.inv_imfp_se, ti.i32(tables.nE), x)
            total = inv_pl + inv_se
            if total > fp(0.0):
                ch = ti.i32(1)
                if ti.random(fp) < inv_pl / total:
                    ch = ti.i32(2)
        return ch

    @ti.func
    def omega_minimum() -> fp:
        w = fp(0.0)
        if ti.static(not is_metal):
            w = fp(gap)
        return w

    @ti.func
    def omega_maximum(E_s: fp) -> fp:
        w = E_s - fp(ef)
        if ti.static(not is_metal):
            w = E_s - fp(gap + evb)
        return w

    @ti.func
    def sample_omega_channel(wgrid, cdf, nrow: ti.i32, E_s: fp):
        valid = ti.i32(0)
        omega = fp(0.0)
        e_lookup = imfp_abscissa(E_s)
        col = stochastic_energy_bin(tables.energy, ti.i32(tables.nE), e_lookup)
        wlo = ti.max(omega_minimum(), wgrid[0, col])
        whi = ti.min(omega_maximum(E_s), wgrid[nrow - 1, col])
        if whi > wlo:
            c0 = interp_matrix_column(wgrid, cdf, nrow, col, wlo)
            c1 = interp_matrix_column(wgrid, cdf, nrow, col, whi)
            if c1 > c0:
                target = c0 + ti.random(fp) * (c1 - c0)
                omega = inverse_cdf_matrix_column(cdf, wgrid, nrow, col, target)
                omega = ti.max(wlo, ti.min(whi, omega))
                if omega > fp(0.0):
                    valid = ti.i32(1)
        return valid, omega

    @ti.func
    def sample_omega(ch: ti.i32, E_s: fp):
        valid = ti.i32(0)
        omega = fp(0.0)
        if ch == ti.i32(1):
            valid, omega = sample_omega_channel(
                tables.w_se, tables.cdf_se, ti.i32(tables.nLossSE), E_s
            )
        elif ch == ti.i32(2):
            valid, omega = sample_omega_channel(
                tables.w_pl, tables.cdf_pl, ti.i32(tables.nLossPL), E_s
            )
        return valid, omega

    @ti.func
    def sample_q_channel(elf, E_s: fp, omega: fp):
        valid = ti.i32(0)
        clipped = ti.i32(0)
        q = fp(0.0)
        k = fp(0.0)
        kp = fp(0.0)

        tprime = E_s
        if ti.static(not is_metal):
            tprime = E_s - fp(gap)
        k = k_rel(tprime)
        kp = k_rel(ti.max(tprime - omega, fp(0.0)))
        qminus = ti.abs(k - kp)
        qplus = k + kp

        if qminus > fp(0.0) and qplus > qminus:
            ql_raw = ti.log(qminus)
            qh_raw = ti.log(qplus)
            ql = ti.max(ql_raw, tables.qlog[0])
            qh = ti.min(qh_raw, tables.qlog[tables.nQ - 1])
            if ql != ql_raw or qh != qh_raw:
                clipped = ti.i32(1)
            if qh > ql:
                wh = omega / fp(H2EV)
                wh = ti.max(wh, tables.omega_h[0])
                wh = ti.min(wh, tables.omega_h[tables.nOmega - 1])
                dq = (qh - ql) / ti.cast(ti.i32(n_q_sample) - ti.i32(1), fp)

                total = fp(0.0)
                f0 = interp2_bilinear(
                    tables.omega_h, tables.qlog, elf,
                    ti.i32(tables.nOmega), ti.i32(tables.nQ), wh, ql,
                )
                for s in range(n_q_sample - 1):
                    x1 = ql + ti.cast(s + ti.i32(1), fp) * dq
                    f1 = interp2_bilinear(
                        tables.omega_h, tables.qlog, elf,
                        ti.i32(tables.nOmega), ti.i32(tables.nQ), wh, x1,
                    )
                    total += fp(0.5) * (f0 + f1) * dq
                    f0 = f1

                if total > fp(0.0):
                    target = ti.random(fp) * total
                    cum = fp(0.0)
                    found = ti.i32(0)
                    qlog_sample = ql
                    f0 = interp2_bilinear(
                        tables.omega_h, tables.qlog, elf,
                        ti.i32(tables.nOmega), ti.i32(tables.nQ), wh, ql,
                    )
                    for s in range(n_q_sample - 1):
                        x0 = ql + ti.cast(s, fp) * dq
                        x1 = x0 + dq
                        f1 = interp2_bilinear(
                            tables.omega_h, tables.qlog, elf,
                            ti.i32(tables.nOmega), ti.i32(tables.nQ), wh, x1,
                        )
                        area = fp(0.5) * (f0 + f1) * dq
                        if found == ti.i32(0) and area > fp(0.0) and target <= cum + area:
                            frac = (target - cum) / area
                            frac = ti.max(fp(0.0), ti.min(fp(1.0), frac))
                            qlog_sample = x0 + frac * dq
                            found = ti.i32(1)
                        cum += area
                        f0 = f1
                    if found == ti.i32(1):
                        q = ti.exp(qlog_sample)
                        q = ti.max(qminus, ti.min(qplus, q))
                        valid = ti.i32(1)
        return valid, q, k, kp, clipped

    @ti.func
    def sample_q(ch: ti.i32, E_s: fp, omega: fp):
        valid = ti.i32(0)
        q = fp(0.0)
        k = fp(0.0)
        kp = fp(0.0)
        clipped = ti.i32(0)
        if ch == ti.i32(1):
            valid, q, k, kp, clipped = sample_q_channel(tables.elf_se, E_s, omega)
        elif ch == ti.i32(2):
            valid, q, k, kp, clipped = sample_q_channel(tables.elf_pl, E_s, omega)
        return valid, q, k, kp, clipped

    @ti.func
    def sample_omega_q_retry(ch: ti.i32, E_s: fp):
        """Sample a consistent inelastic loss and conditional momentum transfer.

        A very rare f32/table-interpolation edge case can produce an omega for
        which the numerically sampled conditional q distribution has zero
        weight.  Keep the already-selected inelastic channel fixed and redraw
        omega (and therefore q) a few times before declaring the event invalid.
        """
        valid = ti.i32(0)
        any_omega = ti.i32(0)
        clipped_any = ti.i32(0)
        omega = fp(0.0)
        q = fp(0.0)
        k = fp(0.0)
        kp = fp(0.0)

        # Each layer inlines the q sampler. Keep one copy per layer in the
        # compiled kernel while allowing up to four draws, as in plane.py.
        attempt = ti.i32(0)
        while valid == ti.i32(0) and attempt < ti.i32(4):
            attempt += ti.i32(1)
            ok_w, w_try = sample_omega(ch, E_s)
            if ok_w == ti.i32(1):
                any_omega = ti.i32(1)
                ok_q, q_try, k_try, kp_try, clipped = sample_q(ch, E_s, w_try)
                if clipped == ti.i32(1):
                    clipped_any = ti.i32(1)
                if ok_q == ti.i32(1):
                    omega = w_try
                    q = q_try
                    k = k_try
                    kp = kp_try
                    valid = ti.i32(1)

        return valid, any_omega, omega, q, k, kp, clipped_any

    @ti.func
    def q_hat(ux0: fp, uy0: fp, uz0: fp, theta_p: fp, phi_p: fp, k: fp, kp: fp):
        theta_q = ti.atan2(kp * ti.sin(theta_p), k - kp * ti.cos(theta_p))
        qx, qy, qz = rotate_direction(
            ux0, uy0, uz0, theta_q, phi_p + fp(math.pi)
        )
        return qx, qy, qz

    @ti.func
    def sample_joint_dos(e_ref: fp, omega: fp) -> fp:
        ref = ti.max(e_ref, fp(1.0e-6))
        fmax = ti.sqrt(ref * (ref + omega))
        accepted = ti.i32(0)
        value = fp(0.5) * ref
        tries = ti.i32(0)
        while tries < ti.i32(200) and accepted == ti.i32(0):
            e = ti.random(fp) * ref
            if ti.random(fp) * fmax <= ti.sqrt(ti.max(e * (e + omega), fp(0.0))):
                value = e
                accepted = ti.i32(1)
            tries += ti.i32(1)
        return value

    @ti.func
    def sample_semiconductor_valence(omega: fp):
        valid = ti.i32(0)
        epsilon = fp(0.0)
        lower = ti.max(fp(0.0), fp(ecbm) - omega)
        upper = fp(evb)
        if lower <= upper:
            fmax = ti.sqrt(ti.max(upper * (upper + omega), fp(0.0)))
            if fmax <= fp(0.0):
                epsilon = upper
                valid = ti.i32(1)
            else:
                accepted = ti.i32(0)
                tries = ti.i32(0)
                epsilon = fp(0.5) * (lower + upper)
                while tries < ti.i32(200) and accepted == ti.i32(0):
                    e = lower + ti.random(fp) * (upper - lower)
                    weight = ti.sqrt(ti.max(e * (e + omega), fp(0.0)))
                    if ti.random(fp) * fmax <= weight:
                        epsilon = e
                        accepted = ti.i32(1)
                    tries += ti.i32(1)
                valid = ti.i32(1)
        return valid, epsilon

    @ti.func
    def sample_binary_target(omega: fp, q: fp):
        valid = ti.i32(0)
        r = fp(0.0)
        kz = fp(0.0)
        if q > fp(0.0):
            omega_h = omega / fp(H2EV)
            kz = (fp(2.0) * omega_h - q * q) / (fp(2.0) * q)
            rout2 = fp(kf * kf) - kz * kz
            if rout2 > fp(0.0):
                rout = ti.sqrt(rout2)
                rin2 = fp(kf * kf) - (kz + q) * (kz + q)
                rin = fp(0.0)
                if rin2 > fp(0.0):
                    rin = ti.sqrt(rin2)
                if rin < rout:
                    u = ti.random(fp)
                    r = ti.sqrt(rin * rin + u * (rout * rout - rin * rin))
                    valid = ti.i32(1)
        return valid, r, kz

    @ti.func
    def sample_secondary(
        mechanism: ti.i32, omega: fp, q: fp, k: fp, kp: fp,
        ux0: fp, uy0: fp, uz0: fp, theta_p: fp, phi_p: fp,
    ):
        # mechanism: 1=binary, 2=plasmon, 3=semiconductor
        valid = ti.i32(0)
        fallback = ti.i32(0)
        Esec = fp(0.0)
        sx = fp(0.0)
        sy = fp(0.0)
        sz = fp(1.0)

        if ti.static(not is_metal):
            ok, epsv = sample_semiconductor_valence(omega)
            if ok == ti.i32(1):
                Esec = ti.max(epsv + omega, fp(ecbm))
                if ti.static(semiconductor_isotropic):
                    sx, sy, sz = isotropic_direction()
                else:
                    sx, sy, sz = q_hat(ux0, uy0, uz0, theta_p, phi_p, k, kp)
                valid = ti.i32(1)
        else:
            if mechanism == ti.i32(1):
                ok, r, kz = sample_binary_target(omega, q)
                if ok == ti.i32(1):
                    kfz = kz + q
                    Esec = fp(0.5 * H2EV) * (r * r + kfz * kfz)
                    if ti.static(binary_isotropic):
                        sx, sy, sz = isotropic_direction()
                    else:
                        qx, qy, qz = q_hat(ux0, uy0, uz0, theta_p, phi_p, k, kp)
                        psi = fp(two_pi) * ti.random(fp)
                        theta_f = ti.atan2(r, kfz)
                        sx, sy, sz = rotate_direction(qx, qy, qz, theta_f, psi)
                    valid = ti.i32(1)
                else:
                    ti.atomic_add(counters.se_blocked_pauli[None], ti.i32(1))
                    if ti.static(pauli_fallback):
                        fallback = ti.i32(1)
                        ti.atomic_add(counters.se_pauli_fallback[None], ti.i32(1))
                        Ei = sample_joint_dos(fp(ef_feg), omega)
                        Esec = Ei + omega
                        if ti.static(plasmon_isotropic):
                            sx, sy, sz = isotropic_direction()
                        else:
                            qx, qy, qz = q_hat(ux0, uy0, uz0, theta_p, phi_p, k, kp)
                            ki = ti.sqrt(ti.max(fp(2.0) * Ei / fp(H2EV), fp(0.0)))
                            mu = fp(2.0) * ti.random(fp) - fp(1.0)
                            kz2 = ki * mu + q
                            rr = ki * ti.sqrt(ti.max(fp(1.0) - mu * mu, fp(0.0)))
                            sx, sy, sz = rotate_direction(
                                qx, qy, qz, ti.atan2(rr, kz2), fp(two_pi) * ti.random(fp)
                            )
                        valid = ti.i32(1)
            else:
                Ei = sample_joint_dos(fp(ef_feg), omega)
                Esec = Ei + omega
                if ti.static(plasmon_isotropic):
                    sx, sy, sz = isotropic_direction()
                else:
                    qx, qy, qz = q_hat(ux0, uy0, uz0, theta_p, phi_p, k, kp)
                    ki = ti.sqrt(ti.max(fp(2.0) * Ei / fp(H2EV), fp(0.0)))
                    mu = fp(2.0) * ti.random(fp) - fp(1.0)
                    kz2 = ki * mu + q
                    rr = ki * ti.sqrt(ti.max(fp(1.0) - mu * mu, fp(0.0)))
                    sx, sy, sz = rotate_direction(
                        qx, qy, qz, ti.atan2(rr, kz2), fp(two_pi) * ti.random(fp)
                    )
                valid = ti.i32(1)
        return valid, fallback, Esec, sx, sy, sz

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
        if E_perp > fp(ui):
            if ti.static(barrier_classical):
                T = fp(1.0)
            else:
                k1 = ti.sqrt(fp(2.0) * E_perp / fp(H2EV)) / fp(A0_ANG)
                k2 = ti.sqrt(fp(2.0) * (E_perp - fp(ui)) / fp(H2EV)) / fp(A0_ANG)
                if ti.static(barrier_abrupt):
                    denom = (k1 + k2) * (k1 + k2)
                    if denom > fp(0.0):
                        T = fp(4.0) * k1 * k2 / denom
                else:
                    a = fp(0.5 * math.pi * barrier_width) * (k1 - k2)
                    b = fp(0.5 * math.pi * barrier_width) * (k1 + k2)
                    r = sinh_ratio(a, b)
                    T = ti.max(fp(0.0), ti.min(fp(1.0), fp(1.0) - r * r))
        return T

    @ti.func
    def incoming_T(E_perp_vac: fp) -> fp:
        T = fp(1.0)
        if ti.static(incoming_reflection and (not barrier_classical) and ui > 0.0):
            kv = ti.sqrt(ti.max(fp(2.0) * E_perp_vac / fp(H2EV), fp(0.0))) / fp(A0_ANG)
            ks = ti.sqrt(ti.max(fp(2.0) * (E_perp_vac + fp(ui)) / fp(H2EV), fp(0.0))) / fp(A0_ANG)
            if ti.static(barrier_abrupt):
                denom = (kv + ks) * (kv + ks)
                T = fp(0.0)
                if denom > fp(0.0):
                    T = fp(4.0) * kv * ks / denom
            else:
                a = fp(0.5 * math.pi * barrier_width) * ti.abs(ks - kv)
                b = fp(0.5 * math.pi * barrier_width) * (ks + kv)
                r = sinh_ratio(a, b)
                T = ti.max(fp(0.0), ti.min(fp(1.0), fp(1.0) - r * r))
        return T

    @ti.func
    def record_emission(i: ti.i32, energy_vac: fp, ux: fp, uy: fp, uz: fp, mechanism: ti.i32):
        slot = ti.cast(ti.atomic_add(emissions.n_emitted[None], ti.i32(1)), ti.i32)
        if slot < ti.i32(emissions.capacity):
            emissions.energy[slot] = energy_vac
            emissions.x[slot] = pool.x[i]
            emissions.y[slot] = pool.y[i]
            emissions.z[slot] = pool.z[i]
            emissions.ux[slot] = ux
            emissions.uy[slot] = uy
            emissions.uz[slot] = uz
            emissions.electron_id[slot] = i
            emissions.parent_id[slot] = pool.parent_id[i]
            emissions.root_primary_id[slot] = pool.root_primary_id[i]
            emissions.generation[slot] = pool.generation[i]
            cascade = ti.i32(0)
            if pool.generation[i] > ti.i32(0):
                cascade = ti.i32(1)
                ti.atomic_add(counters.cascade_emissions[None], ti.i32(1))
            else:
                ti.atomic_add(counters.primary_emissions[None], ti.i32(1))
            emissions.is_cascade[slot] = cascade
            emissions.mechanism[slot] = mechanism
            emissions.own_elastic_count[slot] = history.own_elastic_count[i]
            emissions.own_inelastic_count[slot] = history.own_inelastic_count[i]
            emissions.path_elastic_count[slot] = history.path_elastic_count[i]
            emissions.path_inelastic_count[slot] = history.path_inelastic_count[i]
            emissions.birth_x[slot] = history.birth_x[i]
            emissions.birth_y[slot] = history.birth_y[i]
            emissions.birth_z[slot] = history.birth_z[i]
            if energy_vac <= fp(bse_cutoff):
                ti.atomic_add(counters.sey_50ev[None], ti.i32(1))
            else:
                ti.atomic_add(counters.bse_50ev[None], ti.i32(1))
        else:
            emissions.overflow[None] = ti.i32(1)
            ti.atomic_add(counters.emission_buffer_overflow[None], ti.i32(1))

    @ti.func
    def do_inelastic(i: ti.i32, allow_spawn: ti.i32):
        ok_event = ti.i32(0)
        ch = choose_channel(pool.energy[i])
        if ch != ti.i32(0):
            ok_sample, any_omega, omega, q, k, kp, clipped = sample_omega_q_retry(
                ch, pool.energy[i]
            )
            if clipped == ti.i32(1):
                ti.atomic_add(counters.q_window_clipped[None], ti.i32(1))
            if ok_sample == ti.i32(0):
                if any_omega == ti.i32(0):
                    ti.atomic_add(counters.omega_cdf_empty[None], ti.i32(1))
                else:
                    ti.atomic_add(counters.q_cdf_empty[None], ti.i32(1))
            else:
                ok_event = ti.i32(1)
                ti.atomic_add(counters.inelastic_events[None], ti.i32(1))
                history.own_inelastic_count[i] += ti.i32(1)
                history.path_inelastic_count[i] += ti.i32(1)

                sampled_ch = ch
                mechanism = ch  # se->binary (1), pl->plasmon (2)
                if ti.static(not is_metal):
                    mechanism = ti.i32(3)
                elif ti.static(use_mao):
                    wh = omega / fp(H2EV)
                    root = ti.sqrt(fp(kf * kf) + fp(2.0) * wh)
                    qminus = root - fp(kf)
                    qplus = root + fp(kf)
                    mechanism = ti.i32(2)
                    if q >= qminus and q <= qplus:
                        mechanism = ti.i32(1)
                    if mechanism != sampled_ch:
                        ti.atomic_add(counters.channel_reclassified[None], ti.i32(1))

                cos_tp = (k * k + kp * kp - q * q) / ti.max(
                    fp(2.0) * k * kp, fp(1.0e-30)
                )
                cos_tp = ti.max(fp(-1.0), ti.min(fp(1.0), cos_tp))
                theta_p = ti.acos(cos_tp)
                phi_p = fp(two_pi) * ti.random(fp)
                ux0 = pool.ux[i]
                uy0 = pool.uy[i]
                uz0 = pool.uz[i]
                ux1, uy1, uz1 = rotate_direction(ux0, uy0, uz0, theta_p, phi_p)
                pool.ux[i] = ux1
                pool.uy[i] = uy1
                pool.uz[i] = uz1
                pool.energy[i] = ti.max(pool.energy[i] - omega, fp(0.0))

                sec_valid, _, Esec, sx, sy, sz = sample_secondary(
                    mechanism, omega, q, k, kp,
                    ux0, uy0, uz0, theta_p, phi_p,
                )

                samples.valid[i] = ti.i32(1)
                samples.sampled_channel[i] = sampled_ch
                samples.mechanism[i] = mechanism
                samples.omega[i] = omega
                samples.q[i] = q
                samples.theta_projectile[i] = theta_p
                samples.secondary_valid[i] = sec_valid
                samples.secondary_energy[i] = Esec

                if sec_valid == ti.i32(1) and allow_spawn == ti.i32(1):
                    child_generation = pool.generation[i] + ti.i32(1)
                    if child_generation > ti.i32(max_generation):
                        ti.atomic_add(counters.generation_limit_hit[None], ti.i32(1))
                    else:
                        enqueue = ti.i32(1)
                        if ti.static(not track_subbarrier):
                            if Esec <= fp(kill_energy):  # LAYERED (was: ui)
                                enqueue = ti.i32(0)
                                ti.atomic_add(counters.se_below_barrier[None], ti.i32(1))
                        if enqueue == ti.i32(1):
                            slot = ti.cast(
                                ti.atomic_add(pool.n_allocated[None], ti.i32(1)), ti.i32
                            )
                            if slot < ti.i32(pool.capacity):
                                pool.x[slot] = pool.x[i]
                                pool.y[slot] = pool.y[i]
                                pool.z[slot] = pool.z[i]
                                pool.ux[slot] = sx
                                pool.uy[slot] = sy
                                pool.uz[slot] = sz
                                pool.energy[slot] = Esec
                                pool.parent_id[slot] = i
                                pool.root_primary_id[slot] = pool.root_primary_id[i]
                                pool.generation[slot] = child_generation
                                pool.alive[slot] = ti.i32(1)
                                pool.steps[slot] = ti.i32(0)
                                layer[slot] = layer[i]  # LAYERED
                                history.own_elastic_count[slot] = ti.i32(0)
                                history.own_inelastic_count[slot] = ti.i32(0)
                                history.path_elastic_count[slot] = history.path_elastic_count[i]
                                history.path_inelastic_count[slot] = history.path_inelastic_count[i]
                                history.birth_x[slot] = pool.x[i]
                                history.birth_y[slot] = pool.y[i]
                                history.birth_z[slot] = pool.z[i]
                                ti.atomic_add(counters.secondaries_queued[None], ti.i32(1))
                            else:
                                pool.overflow[None] = ti.i32(1)
        return ok_event

    # ---- END collision physics from plane.build_plane_kernels ----

    @ti.func
    def do_elastic(i: ti.i32):
        # Same statements as the elastic branch of plane.process_wave.
        E_s = pool.energy[i]
        e_dcs = dcs_abscissa(E_s)
        col = stochastic_energy_bin(tables.energy, ti.i32(tables.nE), e_dcs)
        polar = inverse_cdf_column(
            tables.elastic_cdf, tables.elastic_theta,
            ti.i32(tables.nTheta), col, ti.random(fp),
        )
        azimuth = fp(two_pi) * ti.random(fp)
        ux, uy, uz = rotate_direction(
            pool.ux[i], pool.uy[i], pool.uz[i], polar, azimuth
        )
        pool.ux[i] = ux
        pool.uy[i] = uy
        pool.uz[i] = uz
        history.own_elastic_count[i] += ti.i32(1)
        history.path_elastic_count[i] += ti.i32(1)
        ti.atomic_add(counters.elastic_events[None], ti.i32(1))

    return SimpleNamespace(
        ui=ui, kill_energy=kill_energy,
        inverse_emfp=inverse_emfp, inverse_imfp=inverse_imfp,
        do_elastic=do_elastic, do_inelastic=do_inelastic,
        barrier_T=barrier_T, incoming_T=incoming_T,
        record_emission=record_emission, sinh_ratio=sinh_ratio,
    )


def build_layered_plane_kernels(
    ti, pool, layer, film_tables, substrate_tables, fp, film_physics,
    substrate_physics, surface, counters, samples, emissions, history,
    active_counter, bse_cutoff_ev, thickness_field, interface_dv_ev,
    interface_model, interface_width_angstrom,
):
    """``thickness_field`` is a 0-d Taichi field holding the film thickness in
    Angstrom.  It is read at run time, so one compiled engine serves any
    thickness (compilation of the two-material kernel takes minutes)."""
    surface_model = surface.validate()
    if abs(float(surface.inner_potential_ev) - float(film_tables.inner_potential_ev)) > 1.0e-9:
        raise ValueError("surface inner potential must match the FILM material-table Ui")
    if bool(film_physics.track_subbarrier) != bool(substrate_physics.track_subbarrier):
        raise ValueError("film and substrate must use the same track_subbarrier setting")
    if interface_model not in {"abrupt", "classical", "expqm"}:
        raise ValueError("interface_model must be 'abrupt', 'classical' or 'expqm'")
    if interface_model == "expqm" and not interface_width_angstrom > 0.0:
        raise ValueError("expqm interface requires interface_width_angstrom > 0")

    dv = float(interface_dv_ev)                    # E_s(substrate) - E_s(film)
    ui_film = float(film_tables.inner_potential_ev)
    # Energy below which an electron can never leave through the film surface.
    kill_film = ui_film
    kill_sub = ui_film + dv
    track_subbarrier = bool(film_physics.track_subbarrier)
    iface_classical = interface_model == "classical"
    iface_abrupt = interface_model == "abrupt"
    iface_width = float(interface_width_angstrom)

    common = dict(
        ti=ti, fp=fp, pool=pool, layer=layer, surface=surface,
        surface_model=surface_model, counters=counters, samples=samples,
        emissions=emissions, history=history, bse_cutoff_ev=bse_cutoff_ev,
    )
    F = _make_material_physics(
        tables=film_tables, physics=film_physics, kill_energy=kill_film, **common
    )
    S = _make_material_physics(
        tables=substrate_tables, physics=substrate_physics, kill_energy=kill_sub, **common
    )

    @ti.func
    def interface_T(E_perp: fp, step: fp) -> fp:
        """Transmission through a potential step; ``step`` is the kinetic gain."""
        T = fp(0.0)
        if E_perp + step > fp(0.0) and E_perp > fp(0.0):
            if ti.static(iface_classical):
                T = fp(1.0)
            else:
                k1 = ti.sqrt(fp(2.0) * E_perp / fp(H2EV)) / fp(A0_ANG)
                k2 = ti.sqrt(fp(2.0) * (E_perp + step) / fp(H2EV)) / fp(A0_ANG)
                if ti.static(iface_abrupt):
                    T = fp(4.0) * k1 * k2 / ((k1 + k2) * (k1 + k2))
                else:
                    a = fp(0.5 * math.pi * iface_width) * ti.abs(k1 - k2)
                    b = fp(0.5 * math.pi * iface_width) * (k1 + k2)
                    r = F.sinh_ratio(a, b)
                    T = ti.max(fp(0.0), ti.min(fp(1.0), fp(1.0) - r * r))
        return T

    counter_fields = [getattr(counters, name) for name in counters.names]

    @ti.kernel
    def seed_primaries(n: ti.i32, energy_vac_ev: fp, sin_alpha: fp, cos_alpha: fp):
        pool.n_allocated[None] = n
        pool.overflow[None] = ti.i32(0)
        emissions.n_emitted[None] = ti.i32(0)
        emissions.overflow[None] = ti.i32(0)
        active_counter[None] = ti.i32(0)
        for field in ti.static(counter_fields):
            field[None] = ti.i32(0)

        # Primaries enter through the film surface: same statements as plane.py.
        E_s = energy_vac_ev + fp(ui_film)
        solid_sin = ti.sqrt(ti.max(energy_vac_ev, fp(0.0)) / E_s) * sin_alpha
        solid_sin = ti.min(solid_sin, fp(1.0))
        solid_cos = ti.sqrt(ti.max(fp(0.0), fp(1.0) - solid_sin * solid_sin))
        Eperp_vac = energy_vac_ev * cos_alpha * cos_alpha
        Tin = F.incoming_T(Eperp_vac)

        for i in range(n):
            pool.x[i] = fp(0.0)
            pool.y[i] = fp(0.0)
            pool.z[i] = fp(0.0)
            layer[i] = ti.i32(0)
            pool.parent_id[i] = ti.i32(-1)
            pool.root_primary_id[i] = i
            pool.generation[i] = ti.i32(0)
            pool.steps[i] = ti.i32(0)
            history.own_elastic_count[i] = ti.i32(0)
            history.own_inelastic_count[i] = ti.i32(0)
            history.path_elastic_count[i] = ti.i32(0)
            history.path_inelastic_count[i] = ti.i32(0)
            history.birth_x[i] = fp(0.0)
            history.birth_y[i] = fp(0.0)
            history.birth_z[i] = fp(0.0)
            samples.valid[i] = ti.i32(0)
            samples.sampled_channel[i] = ti.i32(0)
            samples.mechanism[i] = ti.i32(0)
            samples.omega[i] = fp(0.0)
            samples.q[i] = fp(0.0)
            samples.theta_projectile[i] = fp(0.0)
            samples.secondary_valid[i] = ti.i32(0)
            samples.secondary_energy[i] = fp(0.0)

            ti.atomic_add(counters.incoming_barrier_encounters[None], ti.i32(1))
            reflected = ti.i32(0)
            if Tin < fp(1.0) and ti.random(fp) >= Tin:
                reflected = ti.i32(1)

            if reflected == ti.i32(1):
                pool.ux[i] = sin_alpha
                pool.uy[i] = fp(0.0)
                pool.uz[i] = -cos_alpha
                pool.energy[i] = energy_vac_ev
                pool.alive[i] = ti.i32(0)
                ti.atomic_add(counters.incoming_barrier_reflections[None], ti.i32(1))
                ti.atomic_add(counters.escapes[None], ti.i32(1))
                F.record_emission(
                    i, energy_vac_ev, sin_alpha, fp(0.0), -cos_alpha, ti.i32(2)
                )
            else:
                pool.ux[i] = solid_sin
                pool.uy[i] = fp(0.0)
                pool.uz[i] = solid_cos
                pool.energy[i] = E_s
                pool.alive[i] = ti.i32(1)
                ti.atomic_add(counters.incoming_barrier_transmissions[None], ti.i32(1))

    @ti.kernel
    def process_wave(begin: ti.i32, end: ti.i32, steps_per_chunk: ti.i32):
        active_counter[None] = ti.i32(0)
        for i in range(begin, end):
            if pool.alive[i] == ti.i32(1):
                local_steps = ti.i32(0)
                running = ti.i32(1)
                while running == ti.i32(1) and local_steps < steps_per_chunk:
                    E_s = pool.energy[i]
                    in_film = ti.i32(0)
                    if layer[i] == ti.i32(0):
                        in_film = ti.i32(1)

                    thickness = thickness_field[None]
                    if ti.static(not track_subbarrier):
                        kill = fp(kill_sub)
                        if in_film == ti.i32(1):
                            kill = fp(kill_film)
                        if E_s <= kill:
                            running = ti.i32(0)
                            ti.atomic_add(
                                counters.terminated_below_vacuum_level[None], ti.i32(1)
                            )

                    if running == ti.i32(1):
                        inv_e = fp(0.0)
                        inv_i = fp(0.0)
                        if in_film == ti.i32(1):
                            inv_e = F.inverse_emfp(E_s)
                            inv_i = F.inverse_imfp(E_s)
                        else:
                            inv_e = S.inverse_emfp(E_s)
                            inv_i = S.inverse_imfp(E_s)
                        inv_tot = inv_e + inv_i

                        # As in plane.py, an electron with no scattering rate at
                        # all is terminated (counted in no_scattering_rate).
                        flight = fp(0.0)
                        if inv_tot > fp(0.0):
                            flight = -ti.log(ti.max(ti.random(fp), fp(1.0e-15))) / inv_tot

                        # boundary: 0 none, 1 vacuum surface z=0, 2 interface z=t
                        boundary = ti.i32(0)
                        distance = fp(0.0)
                        uz = pool.uz[i]
                        if in_film == ti.i32(1):
                            if uz < fp(0.0):
                                distance = ti.max(pool.z[i] / (-uz), fp(0.0))
                                if distance <= flight:
                                    boundary = ti.i32(1)
                            elif uz > fp(0.0):
                                distance = ti.max((thickness - pool.z[i]) / uz, fp(0.0))
                                if distance <= flight:
                                    boundary = ti.i32(2)
                        else:
                            if uz < fp(0.0):
                                distance = ti.max((pool.z[i] - thickness) / (-uz), fp(0.0))
                                if distance <= flight:
                                    boundary = ti.i32(2)

                        if inv_tot <= fp(0.0):
                            ti.atomic_add(counters.no_scattering_rate[None], ti.i32(1))
                            running = ti.i32(0)

                        elif boundary == ti.i32(1):
                            # ---- vacuum surface (film side): same as plane.py ----
                            pool.x[i] += distance * pool.ux[i]
                            pool.y[i] += distance * pool.uy[i]
                            pool.z[i] = fp(0.0)
                            pool.steps[i] += ti.i32(1)
                            local_steps += ti.i32(1)
                            ti.atomic_add(counters.surface_encounters[None], ti.i32(1))

                            outward_cos = -pool.uz[i]
                            Eperp = E_s * outward_cos * outward_cos
                            T = F.barrier_T(Eperp)
                            escaped = ti.i32(0)
                            if T > fp(0.0):
                                if T >= fp(1.0) or ti.random(fp) < T:
                                    escaped = ti.i32(1)

                            if escaped == ti.i32(1):
                                Ev = E_s - fp(ui_film)
                                scale = ti.sqrt(E_s / Ev)
                                ux2 = pool.ux[i] * scale
                                uy2 = pool.uy[i] * scale
                                uz2 = -ti.sqrt(ti.max(
                                    fp(0.0), fp(1.0) - ux2 * ux2 - uy2 * uy2
                                ))
                                pool.ux[i] = ux2
                                pool.uy[i] = uy2
                                pool.uz[i] = uz2
                                pool.energy[i] = Ev
                                ti.atomic_add(counters.escapes[None], ti.i32(1))
                                F.record_emission(i, Ev, ux2, uy2, uz2, ti.i32(1))
                                running = ti.i32(0)
                            else:
                                pool.uz[i] = ti.abs(pool.uz[i])
                                ti.atomic_add(counters.internal_reflections[None], ti.i32(1))

                        elif boundary == ti.i32(2):
                            # ---- film/substrate interface ----
                            pool.x[i] += distance * pool.ux[i]
                            pool.y[i] += distance * pool.uy[i]
                            pool.z[i] = thickness
                            pool.steps[i] += ti.i32(1)
                            local_steps += ti.i32(1)
                            ti.atomic_add(counters.interface_encounters[None], ti.i32(1))

                            step = fp(-dv)              # substrate -> film
                            if in_film == ti.i32(1):
                                step = fp(dv)           # film -> substrate
                            cosn = ti.abs(pool.uz[i])
                            Eperp = E_s * cosn * cosn
                            T = interface_T(Eperp, step)
                            crossed = ti.i32(0)
                            if T > fp(0.0):
                                if T >= fp(1.0) or ti.random(fp) < T:
                                    crossed = ti.i32(1)

                            if crossed == ti.i32(1):
                                E2 = E_s + step
                                scale = ti.sqrt(E_s / E2)
                                ux2 = pool.ux[i] * scale
                                uy2 = pool.uy[i] * scale
                                uz2 = ti.sqrt(ti.max(
                                    fp(0.0), fp(1.0) - ux2 * ux2 - uy2 * uy2
                                ))
                                if pool.uz[i] < fp(0.0):
                                    uz2 = -uz2
                                pool.ux[i] = ux2
                                pool.uy[i] = uy2
                                pool.uz[i] = uz2
                                pool.energy[i] = E2
                                ti.atomic_add(counters.interface_transmissions[None], ti.i32(1))
                                if in_film == ti.i32(1):
                                    layer[i] = ti.i32(1)
                                    ti.atomic_add(counters.film_to_substrate[None], ti.i32(1))
                                else:
                                    layer[i] = ti.i32(0)
                                    ti.atomic_add(counters.substrate_to_film[None], ti.i32(1))
                            else:
                                pool.uz[i] = -pool.uz[i]
                                ti.atomic_add(counters.interface_reflections[None], ti.i32(1))

                        else:
                            # ---- collision inside the current layer ----
                            pool.x[i] += flight * pool.ux[i]
                            pool.y[i] += flight * pool.uy[i]
                            pool.z[i] += flight * pool.uz[i]
                            # Clamp round-off only; boundaries were handled above.
                            if in_film == ti.i32(1):
                                pool.z[i] = ti.max(fp(0.0), ti.min(thickness, pool.z[i]))
                            else:
                                pool.z[i] = ti.max(thickness, pool.z[i])
                            pool.steps[i] += ti.i32(1)
                            local_steps += ti.i32(1)

                            if ti.random(fp) < inv_e / inv_tot:
                                if in_film == ti.i32(1):
                                    F.do_elastic(i)
                                else:
                                    S.do_elastic(i)
                            else:
                                if in_film == ti.i32(1):
                                    F.do_inelastic(i, ti.i32(1))
                                else:
                                    S.do_inelastic(i, ti.i32(1))

                        if pool.overflow[None] == ti.i32(1):
                            running = ti.i32(0)

                if running == ti.i32(1) and local_steps >= steps_per_chunk:
                    pool.alive[i] = ti.i32(1)
                    ti.atomic_add(active_counter[None], ti.i32(1))
                    ti.atomic_add(counters.step_chunk_continuations[None], ti.i32(1))
                else:
                    pool.alive[i] = ti.i32(0)

    return seed_primaries, process_wave


class LayeredPlaneTransportEngine(PlaneTransportEngine):
    """Film-on-substrate planar engine with the ``PlaneTransportEngine`` interface.

    ``run_plane`` returns the same result dictionary (yields, diagnostics and,
    on request, event-level emissions), so the sweep CLI and the sampler-library
    adapter work unchanged.  Extra diagnostics: the ``interface_*`` counters.
    """

    def __init__(
        self, ti, fp, film_tables, substrate_tables,
        film_physics: BulkPhysicsConfig, substrate_physics: BulkPhysicsConfig,
        surface: SurfacePhysicsConfig, capacity: int, *,
        film_thickness_nm: float, bse_cutoff_ev: float = 50.0,
        interface_alignment: str = "vacuum", interface_model: str | None = None,
        interface_width_angstrom: float | None = None,
    ):
        self.ti = ti
        self.fp = fp
        self.capacity = int(capacity)
        self.bse_cutoff_ev = float(bse_cutoff_ev)
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        self.thickness_field = ti.field(dtype=fp, shape=())
        self.set_film_thickness_nm(film_thickness_nm)
        self.interface_alignment = str(interface_alignment)
        surface_model = surface.validate()
        self.interface_model = str(interface_model or surface_model)
        self.interface_width_angstrom = float(
            surface.barrier_width_angstrom
            if interface_width_angstrom is None else interface_width_angstrom
        )
        self.interface_dv_ev = interface_step_ev(
            film_tables, substrate_tables, self.interface_alignment
        )

        self.pool = ParticlePool(ti, self.capacity, fp)
        self.layer = ti.field(dtype=ti.i32, shape=self.capacity)   # 0 film, 1 substrate
        self.tables = DeviceTransportTables(ti, film_tables, fp)
        self.substrate_tables = DeviceTransportTables(ti, substrate_tables, fp)
        self.counters = LayeredPlaneCounters(ti)
        self.samples = InelasticSamples(ti, self.capacity, fp)
        self.emissions = EmissionBuffer(ti, self.capacity, fp)
        self.history = PlaneHistory(ti, self.capacity, fp)
        self.active_counter = ti.field(dtype=ti.i32, shape=())
        self.seed_kernel, self.process_wave_kernel = build_layered_plane_kernels(
            ti, self.pool, self.layer, self.tables, self.substrate_tables, fp,
            film_physics, substrate_physics, surface, self.counters, self.samples,
            self.emissions, self.history, self.active_counter, self.bse_cutoff_ev,
            self.thickness_field, self.interface_dv_ev,
            self.interface_model, self.interface_width_angstrom,
        )
        self.force_inelastic_kernel = None

    def set_film_thickness_nm(self, film_thickness_nm):
        t = float(film_thickness_nm)
        if not (math.isfinite(t) and t > 0.0):
            raise ValueError("film thickness must be positive; use plane.py for a bare substrate")
        self.film_thickness_nm = t
        self.thickness_field[None] = 10.0 * t

    def describe(self):
        return {
            "film_thickness_nm": self.film_thickness_nm,
            "interface_alignment": self.interface_alignment,
            "interface_dV_eV": self.interface_dv_ev,
            "interface_model": self.interface_model,
            "interface_width_angstrom": self.interface_width_angstrom,
        }
