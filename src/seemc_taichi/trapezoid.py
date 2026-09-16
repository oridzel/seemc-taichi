"""Taichi transport for a raised trapezoidal SEM line on a bulk substrate.

Coordinates follow seemc-imaging: vacuum is toward negative z, the substrate
occupies z >= 0, and a trapezoidal line occupies -height <= z <= 0.  The line
is infinite along y.  Every transport leg is truncated at the first exposed
solid/vacuum boundary before collision physics is applied.
"""

import math
from dataclasses import dataclass
import numpy as np

from .device_tables import DeviceTransportTables
from .bulk import BulkPhysicsConfig, InelasticSamples
from .particle_pool import ParticlePool
from .primitives import make_primitives
from .tables import extract_reference_tables
from .surface import SurfacePhysicsConfig

H2EV = 27.21184
C_AU = 137.035999084
A0_ANG = 0.529177


@dataclass(frozen=True)
class TrapezoidGeometryConfig:
    """One trapezoidal line on a semi-infinite substrate, in Angstrom."""

    top_width: float
    bottom_width: float
    height: float
    center_x: float = 0.0

    def validate(self):
        vals = tuple(float(v) for v in (self.top_width, self.bottom_width, self.height, self.center_x))
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("trapezoid geometry values must be finite")
        if self.top_width <= 0 or self.bottom_width <= 0 or self.height <= 0:
            raise ValueError("top_width, bottom_width, and height must be positive")
        if self.bottom_width < self.top_width:
            raise ValueError("bottom_width must be >= top_width")
        return self

    @property
    def half_top(self):
        return 0.5 * float(self.top_width)

    @property
    def half_bottom(self):
        return 0.5 * float(self.bottom_width)

    def launch_surface(self, x):
        """Normal-incidence first surface: (z, outward_nx, outward_nz, code)."""
        self.validate()
        dx = float(x) - float(self.center_x)
        a, b, h = self.half_top, self.half_bottom, float(self.height)
        adx = abs(dx)
        if adx <= a or b <= a:
            return -h, 0.0, -1.0, 1
        if adx < b:
            z = -h + h * (adx - a) / (b - a)
            norm = math.hypot(h, b - a)
            nx = h / norm if dx >= 0 else -h / norm
            nz = -(b - a) / norm
            return z, nx, nz, 2 if dx >= 0 else 3
        return 0.0, 0.0, -1.0, 4



class TrapezoidCounters:
    def __init__(self, ti):
        names = (
            "elastic_events", "inelastic_events", "secondaries_queued",
            "se_below_barrier", "se_blocked_pauli", "se_pauli_fallback",
            "channel_reclassified", "omega_cdf_empty", "q_window_clipped",
            "q_cdf_empty", "step_limit_hit", "generation_limit_hit",
            "no_scattering_rate",
            "surface_encounters", "escapes", "internal_reflections",
            "incoming_barrier_encounters", "incoming_barrier_reflections",
            "incoming_barrier_transmissions", "step_chunk_continuations",
            "sey_50ev", "bse_50ev", "cascade_emissions",
            "primary_emissions", "emission_buffer_overflow",
        )
        self.names = names
        for name in names:
            setattr(self, name, ti.field(dtype=ti.i32, shape=()))

    def as_dict(self):
        return {name: int(getattr(self, name)[None]) for name in self.names}


class TrapezoidEmissionBuffer:
    """Event-level trapezoid emissions; each particle can emit at most once."""

    def __init__(self, ti, capacity, fp):
        self.capacity = int(capacity)
        self.n_emitted = ti.field(dtype=ti.i32, shape=())
        self.overflow = ti.field(dtype=ti.i32, shape=())
        self.energy = ti.field(dtype=fp, shape=self.capacity)
        self.ux = ti.field(dtype=fp, shape=self.capacity)
        self.uy = ti.field(dtype=fp, shape=self.capacity)
        self.uz = ti.field(dtype=fp, shape=self.capacity)
        self.electron_id = ti.field(dtype=ti.i32, shape=self.capacity)
        self.parent_id = ti.field(dtype=ti.i32, shape=self.capacity)
        self.root_primary_id = ti.field(dtype=ti.i32, shape=self.capacity)
        self.generation = ti.field(dtype=ti.i32, shape=self.capacity)
        self.is_cascade = ti.field(dtype=ti.i32, shape=self.capacity)
        # 1 = ordinary transport escape; 2 = incoming-barrier reflection.
        self.mechanism = ti.field(dtype=ti.i32, shape=self.capacity)
        self.inelastic_count = ti.field(dtype=ti.i32, shape=self.capacity)
        # 1 top, 2 right wall, 3 left wall, 4 substrate.
        self.surface_code = ti.field(dtype=ti.i32, shape=self.capacity)


class TrapezoidTrajectoryBuffer:
    """Sparse point records for a small traced subset of primaries/cascades."""

    def __init__(self, ti, capacity, fp):
        self.capacity = int(capacity)
        self.n_points = ti.field(dtype=ti.i32, shape=())
        self.overflow = ti.field(dtype=ti.i32, shape=())
        self.trace_enabled = ti.field(dtype=ti.i32, shape=())
        self.trace_n_primaries = ti.field(dtype=ti.i32, shape=())
        self.x = ti.field(dtype=fp, shape=self.capacity)
        self.y = ti.field(dtype=fp, shape=self.capacity)
        self.z = ti.field(dtype=fp, shape=self.capacity)
        self.energy = ti.field(dtype=fp, shape=self.capacity)
        self.electron_id = ti.field(dtype=ti.i32, shape=self.capacity)
        self.parent_id = ti.field(dtype=ti.i32, shape=self.capacity)
        self.root_primary_id = ti.field(dtype=ti.i32, shape=self.capacity)
        self.generation = ti.field(dtype=ti.i32, shape=self.capacity)
        self.inelastic_count = ti.field(dtype=ti.i32, shape=self.capacity)
        self.event = ti.field(dtype=ti.i32, shape=self.capacity)
        self.surface_code = ti.field(dtype=ti.i32, shape=self.capacity)
        self.step = ti.field(dtype=ti.i32, shape=self.capacity)


def build_trapezoid_kernels(ti, pool, tables, fp, physics, surface, geometry, counters, samples, emissions, trajectories, active_counter, launch_x_input, launch_y_input, bse_cutoff_ev):
    physics.validate()
    geometry.validate()
    surface_model = surface.validate()
    if abs(float(surface.inner_potential_ev) - float(tables.inner_potential_ev)) > 1.0e-9:
        raise ValueError("surface inner potential must match material-table Ui")
    (
        _, interp1, stochastic_energy_bin, inverse_cdf_column,
        _, interp_matrix_column, inverse_cdf_matrix_column, interp2_bilinear,
    ) = make_primitives(ti, fp)

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
    geom_a = 0.5 * float(geometry.top_width)
    geom_b = 0.5 * float(geometry.bottom_width)
    geom_h = float(geometry.height)
    geom_cx = float(geometry.center_x)
    geom_slope = geom_b - geom_a
    geom_norm = math.hypot(geom_h, geom_slope)
    right_nx = geom_h / geom_norm
    right_nz = -geom_slope / geom_norm
    left_nx = -right_nx
    left_nz = right_nz
    geom_eps = 1.0e-7

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

        for _attempt in ti.static(range(4)):
            if valid == ti.i32(0):
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
    def launch_surface_for_x(x: fp):
        """Normal-incidence first exposed surface for a beam coordinate x."""
        z = fp(0.0)
        nx = fp(0.0)
        nz = fp(-1.0)
        code = ti.i32(4)
        dx = x - fp(geom_cx)
        adx = ti.abs(dx)
        if adx <= fp(geom_a) or ti.static(geom_slope <= 0.0):
            z = fp(-geom_h)
            code = ti.i32(1)
        elif adx < fp(geom_b):
            z = fp(-geom_h) + fp(geom_h) * (adx - fp(geom_a)) / fp(geom_slope)
            if dx >= fp(0.0):
                nx = fp(right_nx)
                nz = fp(right_nz)
                code = ti.i32(2)
            else:
                nx = fp(left_nx)
                nz = fp(left_nz)
                code = ti.i32(3)
        return z, nx, nz, code

    @ti.func
    def first_surface_hit(x: fp, z: fp, ux: fp, uz: fp, max_distance: fp):
        """Nearest exposed solid->vacuum crossing along a leg from inside solid."""
        hit = ti.i32(0)
        best = max_distance + fp(1.0)
        nx_best = fp(0.0)
        nz_best = fp(-1.0)
        code = ti.i32(0)

        # Top face z=-h, |x-c| <= a, outward n=(0,-1).
        if uz < fp(-1.0e-15):
            t = (fp(-geom_h) - z) / uz
            if t >= fp(-geom_eps) and t <= max_distance + fp(geom_eps):
                xh = x + t * ux
                if ti.abs(xh - fp(geom_cx)) <= fp(geom_a + geom_eps):
                    tt = ti.max(t, fp(0.0))
                    if tt < best:
                        hit = ti.i32(1)
                        best = tt
                        nx_best = fp(0.0)
                        nz_best = fp(-1.0)
                        code = ti.i32(1)

        # Right sidewall.  Candidate only when moving outward (d dot n > 0).
        denom = ux * fp(right_nx) + uz * fp(right_nz)
        if denom > fp(1.0e-15):
            signed = (x - fp(geom_cx + geom_a)) * fp(right_nx) + (z + fp(geom_h)) * fp(right_nz)
            t = -signed / denom
            if t >= fp(-geom_eps) and t <= max_distance + fp(geom_eps):
                zh = z + t * uz
                if zh >= fp(-geom_h - geom_eps) and zh <= fp(geom_eps):
                    tt = ti.max(t, fp(0.0))
                    if tt < best:
                        hit = ti.i32(1)
                        best = tt
                        nx_best = fp(right_nx)
                        nz_best = fp(right_nz)
                        code = ti.i32(2)

        # Left sidewall.
        denom = ux * fp(left_nx) + uz * fp(left_nz)
        if denom > fp(1.0e-15):
            signed = (x - fp(geom_cx - geom_a)) * fp(left_nx) + (z + fp(geom_h)) * fp(left_nz)
            t = -signed / denom
            if t >= fp(-geom_eps) and t <= max_distance + fp(geom_eps):
                zh = z + t * uz
                if zh >= fp(-geom_h - geom_eps) and zh <= fp(geom_eps):
                    tt = ti.max(t, fp(0.0))
                    if tt < best:
                        hit = ti.i32(1)
                        best = tt
                        nx_best = fp(left_nx)
                        nz_best = fp(left_nz)
                        code = ti.i32(3)

        # Exposed substrate z=0 only outside the buried trapezoid base.
        if uz < fp(-1.0e-15):
            t = -z / uz
            if t >= fp(-geom_eps) and t <= max_distance + fp(geom_eps):
                xh = x + t * ux
                if ti.abs(xh - fp(geom_cx)) >= fp(geom_b - geom_eps):
                    tt = ti.max(t, fp(0.0))
                    if tt < best:
                        hit = ti.i32(1)
                        best = tt
                        nx_best = fp(0.0)
                        nz_best = fp(-1.0)
                        code = ti.i32(4)
        return hit, best, nx_best, nz_best, code

    @ti.func
    def reflect_about_normal(ux: fp, uy: fp, uz: fp, nx: fp, nz: fp):
        dotn = ux * nx + uz * nz
        rx = ux - fp(2.0) * dotn * nx
        ry = uy
        rz = uz - fp(2.0) * dotn * nz
        norm = ti.sqrt(ti.max(rx * rx + ry * ry + rz * rz, fp(1.0e-30)))
        return rx / norm, ry / norm, rz / norm

    @ti.func
    def transmit_outgoing(ux: fp, uy: fp, uz: fp, E_s: fp, nx: fp, nz: fp):
        """Parallel momentum conservation for solid -> vacuum."""
        Ev = E_s - fp(ui)
        dn = ux * nx + uz * nz
        tx = ux - dn * nx
        ty = uy
        tz = uz - dn * nz
        scale = ti.sqrt(E_s / ti.max(Ev, fp(1.0e-30)))
        tx *= scale
        ty *= scale
        tz *= scale
        tang2 = tx * tx + ty * ty + tz * tz
        normal_mag = ti.sqrt(ti.max(fp(0.0), fp(1.0) - tang2))
        ox = tx + normal_mag * nx
        oy = ty
        oz = tz + normal_mag * nz
        norm = ti.sqrt(ti.max(ox * ox + oy * oy + oz * oz, fp(1.0e-30)))
        return Ev, ox / norm, oy / norm, oz / norm

    @ti.func
    def transmit_incoming(ux: fp, uy: fp, uz: fp, E_v: fp, nx: fp, nz: fp):
        """Parallel momentum conservation for vacuum -> solid."""
        E_s = E_v + fp(ui)
        dn = ux * nx + uz * nz
        tx = ux - dn * nx
        ty = uy
        tz = uz - dn * nz
        scale = ti.sqrt(ti.max(E_v, fp(0.0)) / E_s)
        tx *= scale
        ty *= scale
        tz *= scale
        tang2 = tx * tx + ty * ty + tz * tz
        inward_mag = ti.sqrt(ti.max(fp(0.0), fp(1.0) - tang2))
        ox = tx - inward_mag * nx
        oy = ty
        oz = tz - inward_mag * nz
        norm = ti.sqrt(ti.max(ox * ox + oy * oy + oz * oz, fp(1.0e-30)))
        return E_s, ox / norm, oy / norm, oz / norm

    @ti.func
    def record_trajectory(i: ti.i32, event_code: ti.i32, surface_code: ti.i32):
        if trajectories.trace_enabled[None] == ti.i32(1):
            root = pool.root_primary_id[i]
            if root >= ti.i32(0) and root < trajectories.trace_n_primaries[None]:
                slot = ti.cast(ti.atomic_add(trajectories.n_points[None], ti.i32(1)), ti.i32)
                if slot < ti.i32(trajectories.capacity):
                    trajectories.x[slot] = pool.x[i]
                    trajectories.y[slot] = pool.y[i]
                    trajectories.z[slot] = pool.z[i]
                    trajectories.energy[slot] = pool.energy[i]
                    trajectories.electron_id[slot] = i
                    trajectories.parent_id[slot] = pool.parent_id[i]
                    trajectories.root_primary_id[slot] = root
                    trajectories.generation[slot] = pool.generation[i]
                    trajectories.inelastic_count[slot] = pool.inelastic_count[i]
                    trajectories.event[slot] = event_code
                    trajectories.surface_code[slot] = surface_code
                    trajectories.step[slot] = pool.steps[i]
                else:
                    trajectories.overflow[None] = ti.i32(1)

    @ti.func
    def record_emission(i: ti.i32, energy_vac: fp, ux: fp, uy: fp, uz: fp, mechanism: ti.i32, surface_code: ti.i32):
        slot = ti.cast(ti.atomic_add(emissions.n_emitted[None], ti.i32(1)), ti.i32)
        if slot < ti.i32(emissions.capacity):
            emissions.energy[slot] = energy_vac
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
            emissions.inelastic_count[slot] = pool.inelastic_count[i]
            emissions.surface_code[slot] = surface_code
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
                pool.inelastic_count[i] += ti.i32(1)
                record_trajectory(i, ti.i32(4), ti.i32(0))

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
                            if Esec <= fp(ui):
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
                                pool.inelastic_count[slot] = ti.i32(0)
                                record_trajectory(slot, ti.i32(5), ti.i32(0))
                                ti.atomic_add(counters.secondaries_queued[None], ti.i32(1))
                            else:
                                pool.overflow[None] = ti.i32(1)
        return ok_event

    @ti.kernel
    def seed_primaries(n: ti.i32, energy_vac_ev: fp):
        pool.n_allocated[None] = n
        pool.overflow[None] = ti.i32(0)
        emissions.n_emitted[None] = ti.i32(0)
        emissions.overflow[None] = ti.i32(0)
        trajectories.n_points[None] = ti.i32(0)
        trajectories.overflow[None] = ti.i32(0)
        active_counter[None] = ti.i32(0)
        counters.elastic_events[None] = ti.i32(0)
        counters.inelastic_events[None] = ti.i32(0)
        counters.secondaries_queued[None] = ti.i32(0)
        counters.se_below_barrier[None] = ti.i32(0)
        counters.se_blocked_pauli[None] = ti.i32(0)
        counters.se_pauli_fallback[None] = ti.i32(0)
        counters.channel_reclassified[None] = ti.i32(0)
        counters.omega_cdf_empty[None] = ti.i32(0)
        counters.q_window_clipped[None] = ti.i32(0)
        counters.q_cdf_empty[None] = ti.i32(0)
        counters.step_limit_hit[None] = ti.i32(0)
        counters.generation_limit_hit[None] = ti.i32(0)
        counters.no_scattering_rate[None] = ti.i32(0)
        counters.surface_encounters[None] = ti.i32(0)
        counters.escapes[None] = ti.i32(0)
        counters.internal_reflections[None] = ti.i32(0)
        counters.incoming_barrier_encounters[None] = ti.i32(0)
        counters.incoming_barrier_reflections[None] = ti.i32(0)
        counters.incoming_barrier_transmissions[None] = ti.i32(0)
        counters.step_chunk_continuations[None] = ti.i32(0)
        counters.sey_50ev[None] = ti.i32(0)
        counters.bse_50ev[None] = ti.i32(0)
        counters.cascade_emissions[None] = ti.i32(0)
        counters.primary_emissions[None] = ti.i32(0)
        counters.emission_buffer_overflow[None] = ti.i32(0)

        for i in range(n):
            x0 = launch_x_input[i]
            y0 = launch_y_input[i]
            z0, nx, nz, surf_code = launch_surface_for_x(x0)
            pool.x[i] = x0
            pool.y[i] = y0
            pool.z[i] = z0
            pool.parent_id[i] = ti.i32(-1)
            pool.root_primary_id[i] = i
            pool.generation[i] = ti.i32(0)
            pool.steps[i] = ti.i32(0)
            pool.inelastic_count[i] = ti.i32(0)
            samples.valid[i] = ti.i32(0)
            samples.sampled_channel[i] = ti.i32(0)
            samples.mechanism[i] = ti.i32(0)
            samples.omega[i] = fp(0.0)
            samples.q[i] = fp(0.0)
            samples.theta_projectile[i] = fp(0.0)
            samples.secondary_valid[i] = ti.i32(0)
            samples.secondary_energy[i] = fp(0.0)

            # Normal incident beam d=(0,0,+1).  Local normal is solid->vacuum.
            Eperp_vac = energy_vac_ev * nz * nz
            Tin = incoming_T(Eperp_vac)
            ti.atomic_add(counters.incoming_barrier_encounters[None], ti.i32(1))
            reflected = ti.i32(0)
            if Tin < fp(1.0) and ti.random(fp) >= Tin:
                reflected = ti.i32(1)

            if reflected == ti.i32(1):
                rx, ry, rz = reflect_about_normal(fp(0.0), fp(0.0), fp(1.0), nx, nz)
                pool.ux[i] = rx
                pool.uy[i] = ry
                pool.uz[i] = rz
                pool.energy[i] = energy_vac_ev
                pool.alive[i] = ti.i32(0)
                record_trajectory(i, ti.i32(1), surf_code)
                ti.atomic_add(counters.incoming_barrier_reflections[None], ti.i32(1))
                ti.atomic_add(counters.escapes[None], ti.i32(1))
                record_trajectory(i, ti.i32(9), surf_code)
                record_emission(i, energy_vac_ev, rx, ry, rz, ti.i32(2), surf_code)
            else:
                E_s, sx, sy, sz = transmit_incoming(
                    fp(0.0), fp(0.0), fp(1.0), energy_vac_ev, nx, nz
                )
                pool.ux[i] = sx
                pool.uy[i] = sy
                pool.uz[i] = sz
                pool.energy[i] = E_s
                pool.alive[i] = ti.i32(1)
                record_trajectory(i, ti.i32(1), surf_code)
                ti.atomic_add(counters.incoming_barrier_transmissions[None], ti.i32(1))

    @ti.kernel
    def force_one_inelastic(n: ti.i32):
        for i in range(n):
            if pool.alive[i] == ti.i32(1):
                do_inelastic(i, ti.i32(0))

    @ti.kernel
    def process_wave(begin: ti.i32, end: ti.i32, steps_per_chunk: ti.i32):
        active_counter[None] = ti.i32(0)
        for i in range(begin, end):
            if pool.alive[i] == ti.i32(1):
                local_steps = ti.i32(0)
                running = ti.i32(1)
                while running == ti.i32(1) and local_steps < steps_per_chunk:
                    E_s = pool.energy[i]
                    if ti.static(not track_subbarrier):
                        if E_s <= fp(ui):
                            record_trajectory(i, ti.i32(10), ti.i32(0))
                            running = ti.i32(0)

                    if running == ti.i32(1):
                        inv_e = inverse_emfp(E_s)
                        inv_i = inverse_imfp(E_s)
                        inv_tot = inv_e + inv_i
                        if inv_tot <= fp(0.0):
                            ti.atomic_add(counters.no_scattering_rate[None], ti.i32(1))
                            record_trajectory(i, ti.i32(10), ti.i32(0))
                            running = ti.i32(0)
                        else:
                            flight = -ti.log(ti.max(ti.random(fp), fp(1.0e-15))) / inv_tot

                            # Truncate at the nearest exposed trapezoid/substrate
                            # boundary.  A surface-truncated leg never collides.
                            hit_surface, distance_surface, nx, nz, surf_code = first_surface_hit(
                                pool.x[i], pool.z[i], pool.ux[i], pool.uz[i], flight
                            )

                            if hit_surface == ti.i32(1):
                                pool.x[i] += distance_surface * pool.ux[i]
                                pool.y[i] += distance_surface * pool.uy[i]
                                pool.z[i] += distance_surface * pool.uz[i]
                                pool.steps[i] += ti.i32(1)
                                local_steps += ti.i32(1)
                                ti.atomic_add(counters.surface_encounters[None], ti.i32(1))
                                record_trajectory(i, ti.i32(6), surf_code)

                                outward_cos = pool.ux[i] * nx + pool.uz[i] * nz
                                Eperp = E_s * outward_cos * outward_cos
                                T = barrier_T(Eperp)
                                escaped = ti.i32(0)
                                if T > fp(0.0):
                                    if T >= fp(1.0) or ti.random(fp) < T:
                                        escaped = ti.i32(1)

                                if escaped == ti.i32(1):
                                    Ev, ux2, uy2, uz2 = transmit_outgoing(
                                        pool.ux[i], pool.uy[i], pool.uz[i], E_s, nx, nz
                                    )
                                    pool.ux[i] = ux2
                                    pool.uy[i] = uy2
                                    pool.uz[i] = uz2
                                    pool.energy[i] = Ev
                                    ti.atomic_add(counters.escapes[None], ti.i32(1))
                                    record_trajectory(i, ti.i32(9), surf_code)
                                    record_emission(
                                        i, Ev, ux2, uy2, uz2, ti.i32(1), surf_code
                                    )
                                    running = ti.i32(0)
                                else:
                                    rx, ry, rz = reflect_about_normal(
                                        pool.ux[i], pool.uy[i], pool.uz[i], nx, nz
                                    )
                                    pool.ux[i] = rx
                                    pool.uy[i] = ry
                                    pool.uz[i] = rz
                                    record_trajectory(i, ti.i32(7), surf_code)
                                    ti.atomic_add(counters.internal_reflections[None], ti.i32(1))
                            else:
                                pool.x[i] += flight * pool.ux[i]
                                pool.y[i] += flight * pool.uy[i]
                                pool.z[i] += flight * pool.uz[i]
                                pool.steps[i] += ti.i32(1)
                                local_steps += ti.i32(1)

                                if ti.random(fp) < inv_e / inv_tot:
                                    e_dcs = dcs_abscissa(E_s)
                                    col = stochastic_energy_bin(
                                        tables.energy, ti.i32(tables.nE), e_dcs
                                    )
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
                                    record_trajectory(i, ti.i32(3), ti.i32(0))
                                    ti.atomic_add(counters.elastic_events[None], ti.i32(1))
                                else:
                                    do_inelastic(i, ti.i32(1))

                            if pool.overflow[None] == ti.i32(1):
                                running = ti.i32(0)

                if running == ti.i32(1) and local_steps >= steps_per_chunk:
                    # This is a scheduling continuation, not a physical death.
                    pool.alive[i] = ti.i32(1)
                    ti.atomic_add(active_counter[None], ti.i32(1))
                    ti.atomic_add(counters.step_chunk_continuations[None], ti.i32(1))
                else:
                    pool.alive[i] = ti.i32(0)

    return seed_primaries, force_one_inelastic, process_wave


class TrapezoidTransportEngine:
    """Persistent Taichi SEEMC engine for one raised trapezoidal line."""

    def __init__(
        self, ti, fp, host_tables, physics: BulkPhysicsConfig,
        surface: SurfacePhysicsConfig, geometry: TrapezoidGeometryConfig,
        capacity: int, bse_cutoff_ev: float = 50.0, trajectory_capacity: int | None = None,
    ):
        self.ti = ti
        self.fp = fp
        self.capacity = int(capacity)
        self.bse_cutoff_ev = float(bse_cutoff_ev)
        self.geometry = geometry.validate()
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        self.pool = ParticlePool(ti, self.capacity, fp)
        self.tables = DeviceTransportTables(ti, host_tables, fp)
        self.counters = TrapezoidCounters(ti)
        self.samples = InelasticSamples(ti, self.capacity, fp)
        self.emissions = TrapezoidEmissionBuffer(ti, self.capacity, fp)
        self.trajectory_capacity = int(self.capacity if trajectory_capacity is None else trajectory_capacity)
        if self.trajectory_capacity < 1:
            raise ValueError("trajectory_capacity must be positive")
        self.trajectories = TrapezoidTrajectoryBuffer(ti, self.trajectory_capacity, fp)
        self.active_counter = ti.field(dtype=ti.i32, shape=())
        self.launch_x_input = ti.field(dtype=fp, shape=self.capacity)
        self.launch_y_input = ti.field(dtype=fp, shape=self.capacity)
        (
            self.seed_kernel,
            self.force_inelastic_kernel,
            self.process_wave_kernel,
        ) = build_trapezoid_kernels(
            ti, self.pool, self.tables, fp, physics, surface, self.geometry,
            self.counters, self.samples, self.emissions, self.trajectories, self.active_counter,
            self.launch_x_input, self.launch_y_input, self.bse_cutoff_ev,
        )

    def seed(self, n, energy_vac_ev, *, nominal_x=0.0, nominal_y=0.0,
             beam_fwhm=0.0, seed=12345):
        """Seed one normal-incidence beam position with Gaussian spot sampling."""
        n = int(n)
        if n < 1 or n > self.capacity:
            raise ValueError("n must be positive and <= capacity")
        fwhm = float(beam_fwhm)
        if not math.isfinite(fwhm) or fwhm < 0.0:
            raise ValueError("beam_fwhm must be finite and non-negative")
        sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0))) if fwhm > 0 else 0.0
        rng = np.random.default_rng(int(seed))
        if sigma > 0.0:
            xs = float(nominal_x) + rng.normal(0.0, sigma, size=n)
            ys = float(nominal_y) + rng.normal(0.0, sigma, size=n)
        else:
            xs = np.full(n, float(nominal_x), dtype=float)
            ys = np.full(n, float(nominal_y), dtype=float)
        # Copy only the active prefix; fields are persistent across pixels.
        full_x = np.zeros(self.capacity, dtype=np.float64 if self.fp == self.ti.f64 else np.float32)
        full_y = np.zeros_like(full_x)
        full_x[:n] = xs
        full_y[:n] = ys
        self.launch_x_input.from_numpy(full_x)
        self.launch_y_input.from_numpy(full_y)
        self.seed_kernel(n, float(energy_vac_ev))
        self.ti.sync()

        # Host-side launch statistics are cheap and useful for edge pixels where
        # a finite Gaussian spot can straddle top/sidewall/substrate surfaces.
        zvals = np.empty(n, dtype=float)
        inc = np.empty(n, dtype=float)
        surface_codes = np.empty(n, dtype=np.int32)
        for j, xj in enumerate(xs):
            zj, _nxj, nzj, codej = self.geometry.launch_surface(float(xj))
            zvals[j] = zj
            surface_codes[j] = int(codej)
            inc[j] = math.degrees(math.acos(max(-1.0, min(1.0, -float(nzj)))))
        return {
            "launch_mean_x_angstrom": float(np.mean(xs)),
            "launch_mean_y_angstrom": float(np.mean(ys)),
            "launch_surface_z_mean_angstrom": float(np.mean(zvals)),
            "local_incidence_mean_deg": float(np.mean(inc)),
            "local_incidence_sem_deg": (
                0.0 if n < 2 else float(np.std(inc, ddof=0) / math.sqrt(n))
            ),
            "launch_surface_codes": surface_codes,
        }


    def _copy_emissions(self):
        n_emit_total = int(self.emissions.n_emitted[None])
        n_emit = min(n_emit_total, self.emissions.capacity)
        e = self.emissions
        return {
            "n_emitted_total": n_emit_total,
            "n_emitted_stored": n_emit,
            "emission_overflow": bool(e.overflow[None]),
            "emission_energy_ev": e.energy.to_numpy()[:n_emit],
            "emission_ux": e.ux.to_numpy()[:n_emit],
            "emission_uy": e.uy.to_numpy()[:n_emit],
            "emission_uz": e.uz.to_numpy()[:n_emit],
            "emission_electron_id": e.electron_id.to_numpy()[:n_emit],
            "emission_parent_id": e.parent_id.to_numpy()[:n_emit],
            "emission_root_primary_id": e.root_primary_id.to_numpy()[:n_emit],
            "emission_generation": e.generation.to_numpy()[:n_emit],
            "emission_is_cascade": e.is_cascade.to_numpy()[:n_emit],
            "emission_mechanism": e.mechanism.to_numpy()[:n_emit],
            "emission_inelastic_count": e.inelastic_count.to_numpy()[:n_emit],
            "emission_surface_code": e.surface_code.to_numpy()[:n_emit],
        }

    def _copy_trajectories(self):
        n_total = int(self.trajectories.n_points[None])
        n_pts = min(n_total, self.trajectories.capacity)
        t = self.trajectories
        return {
            "trajectory_n_points_total": n_total,
            "trajectory_n_points_stored": n_pts,
            "trajectory_overflow": bool(t.overflow[None]),
            "trajectory_x_angstrom": t.x.to_numpy()[:n_pts],
            "trajectory_y_angstrom": t.y.to_numpy()[:n_pts],
            "trajectory_z_angstrom": t.z.to_numpy()[:n_pts],
            "trajectory_energy_ev": t.energy.to_numpy()[:n_pts],
            "trajectory_electron_id": t.electron_id.to_numpy()[:n_pts],
            "trajectory_parent_id": t.parent_id.to_numpy()[:n_pts],
            "trajectory_root_primary_id": t.root_primary_id.to_numpy()[:n_pts],
            "trajectory_generation": t.generation.to_numpy()[:n_pts],
            "trajectory_inelastic_count": t.inelastic_count.to_numpy()[:n_pts],
            "trajectory_event": t.event.to_numpy()[:n_pts],
            "trajectory_surface_code": t.surface_code.to_numpy()[:n_pts],
            "trajectory_step": t.step.to_numpy()[:n_pts],
        }

    def run_trapezoid(
        self, *, n, energy_vac_ev, nominal_x=0.0, nominal_y=0.0,
        beam_fwhm=0.0, seed=12345, steps_per_chunk=256,
        max_chunk_repeats_per_wave=100000, copy_emissions=False, copy_state=False,
        copy_trajectories=False, trace_n_primaries=0,
    ):
        """Run independent trapezoid primaries and all queued cascade electrons.

        ``steps_per_chunk`` is only a GPU scheduling bound.  Reaching it leaves
        the electron alive and the same allocation wave is relaunched until all
        electrons in that wave physically terminate.  It is therefore not a
        transport cutoff.
        """
        steps_per_chunk = int(steps_per_chunk)
        if steps_per_chunk < 1:
            raise ValueError("steps_per_chunk must be positive")
        self.trajectories.trace_enabled[None] = int(bool(copy_trajectories and int(trace_n_primaries) > 0))
        self.trajectories.trace_n_primaries[None] = int(max(0, int(trace_n_primaries)))
        launch_meta = self.seed(
            n, energy_vac_ev, nominal_x=nominal_x, nominal_y=nominal_y,
            beam_fwhm=beam_fwhm, seed=seed,
        )

        begin = 0
        wave_count = 0
        kernel_chunks = 0
        while begin < self.pool.allocated():
            end = min(self.pool.allocated(), self.capacity)
            repeats = 0
            while True:
                self.process_wave_kernel(
                    int(begin), int(end), int(steps_per_chunk)
                )
                self.ti.sync()
                kernel_chunks += 1
                repeats += 1
                if int(self.pool.overflow[None]):
                    break
                if int(self.active_counter[None]) == 0:
                    break
                if repeats >= int(max_chunk_repeats_per_wave):
                    self.counters.step_limit_hit[None] = int(
                        self.counters.step_limit_hit[None]
                    ) + 1
                    raise RuntimeError(
                        "wave did not physically terminate before the host safety "
                        "limit; increase max_chunk_repeats_per_wave"
                    )
            wave_count += 1
            if int(self.pool.overflow[None]):
                break
            begin = end

        self.ti.sync()
        allocated = self.pool.allocated()
        stored = min(allocated, self.capacity)
        diag = self.counters.as_dict()
        n_primary = int(n)
        if diag["escapes"] != diag["sey_50ev"] + diag["bse_50ev"]:
            raise RuntimeError("SEY/BSEY counters do not partition trapezoid TEY")
        if diag["escapes"] != diag["cascade_emissions"] + diag["primary_emissions"]:
            raise RuntimeError("cascade/primary counters do not partition trapezoid TEY")
        if int(self.emissions.n_emitted[None]) != diag["escapes"]:
            raise RuntimeError("emission buffer count does not match escape counter")
        result = {
            "n_primaries": n_primary,
            "incident_energy_vac_ev": float(energy_vac_ev),
            "nominal_x_angstrom": float(nominal_x),
            "beam_fwhm_angstrom": float(beam_fwhm),
            "allocated": allocated,
            "stored": stored,
            "overflow": bool(self.pool.overflow[None]),
            "wave_count": wave_count,
            "kernel_chunks": kernel_chunks,
            "diagnostics": diag,
            "tey": diag["escapes"] / n_primary,
            "sey_50ev": diag["sey_50ev"] / n_primary,
            "bsey_50ev": diag["bse_50ev"] / n_primary,
            "cascade_yield": diag["cascade_emissions"] / n_primary,
            "primary_yield": diag["primary_emissions"] / n_primary,
            "launch_mean_x_angstrom": launch_meta["launch_mean_x_angstrom"],
            "launch_mean_y_angstrom": launch_meta["launch_mean_y_angstrom"],
            "launch_surface_z_mean_angstrom": launch_meta["launch_surface_z_mean_angstrom"],
            "local_incidence_mean_deg": launch_meta["local_incidence_mean_deg"],
            "local_incidence_sem_deg": launch_meta["local_incidence_sem_deg"],
            "trace_n_primaries": int(max(0, int(trace_n_primaries))),
        }
        if copy_emissions:
            result.update(self._copy_emissions())
        if copy_trajectories:
            result.update(self._copy_trajectories())
        if copy_state:
            p = self.pool
            result.update({
                "x": p.x.to_numpy()[:stored], "y": p.y.to_numpy()[:stored],
                "z": p.z.to_numpy()[:stored], "ux": p.ux.to_numpy()[:stored],
                "uy": p.uy.to_numpy()[:stored], "uz": p.uz.to_numpy()[:stored],
                "energy_ev": p.energy.to_numpy()[:stored],
                "alive": p.alive.to_numpy()[:stored],
                "parent_id": p.parent_id.to_numpy()[:stored],
                "root_primary_id": p.root_primary_id.to_numpy()[:stored],
                "generation": p.generation.to_numpy()[:stored],
                "steps": p.steps.to_numpy()[:stored],
                "inelastic_count": p.inelastic_count.to_numpy()[:stored],
            })
        return result


def per_primary_emission_counts(result, cutoff_ev=50.0):
    """Return total/SE/BSE counts per root primary from copied emissions."""
    roots = np.asarray(result["emission_root_primary_id"], dtype=np.int64)
    energy = np.asarray(result["emission_energy_ev"], dtype=float)
    n = int(result["n_primaries"])
    valid = (roots >= 0) & (roots < n)
    roots = roots[valid]
    energy = energy[valid]
    total = np.bincount(roots, minlength=n).astype(np.int64)
    se = np.bincount(roots[energy <= float(cutoff_ev)], minlength=n).astype(np.int64)
    bse = np.bincount(roots[energy > float(cutoff_ev)], minlength=n).astype(np.int64)
    return total, se, bse


def yield_sem_from_counts(counts):
    a = np.asarray(counts, dtype=float)
    if a.size < 2:
        return 0.0
    return float(a.std(ddof=0) / math.sqrt(a.size))


def load_reference_plane_tables(database_path, material="Si", config=None):
    try:
        from seemc_imaging.transport import MCConfig, Sample
    except ImportError as exc:
        raise RuntimeError("Install seemc-imaging or use the reference extra") from exc
    cfg = config or MCConfig()
    sample = Sample(material, db_path=str(database_path), config=cfg)
    return sample, extract_reference_tables(sample)


def summarize_inelastic_samples(result):
    valid = np.asarray(result["valid"], bool)
    out = {"valid_fraction": float(np.mean(valid))}
    if np.any(valid):
        ch = np.asarray(result["sampled_channel"])[valid]
        mech = np.asarray(result["mechanism"])[valid]
        omega = np.asarray(result["omega_ev"])[valid]
        q = np.asarray(result["q_a0inv"])[valid]
        sec_valid = np.asarray(result["secondary_valid"])[valid].astype(bool)
        out.update({
            "sampled_se_fraction": float(np.mean(ch == 1)),
            "mechanism_binary_fraction": float(np.mean(mech == 1)),
            "mean_omega_ev": float(np.mean(omega)),
            "mean_q_a0inv": float(np.mean(q)),
            "secondary_valid_fraction": float(np.mean(sec_valid)),
        })
        if np.any(sec_valid):
            sec = np.asarray(result["secondary_energy_ev"])[valid][sec_valid]
            out["mean_secondary_energy_ev"] = float(np.mean(sec))
    return out
