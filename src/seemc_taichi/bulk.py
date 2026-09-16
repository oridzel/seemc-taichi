"""v0.3 bulk elastic + inelastic SEEMC transport on Taichi.

This module intentionally stops at the solid/vacuum boundary problem.  It ports
collision physics and cascade creation first, using the same material tables and
energy conventions as the Python reference.  Surface intersection/barrier
transport is the next milestone.
"""

import math
from dataclasses import dataclass
import numpy as np

from .device_tables import DeviceTransportTables
from .particle_pool import ParticlePool
from .primitives import make_primitives
from .tables import extract_reference_tables

H2EV = 27.21184
C_AU = 137.035999084


@dataclass(frozen=True)
class BulkPhysicsConfig:
    emfp_energy_ref: str = "vb_bottom"
    imfp_energy_ref: str = "vb_bottom"
    elastic_min_energy_ev: float = 5.0
    elastic_low_energy_model: str = "elsepa"
    elastic_cutoff_energy_ev: float = 50.0
    atomic_number: float = 0.0

    se_channel_rule: str = "mao"
    on_pauli_block: str = "fallback"
    se_direction_model: str = "momentum"
    plasmon_se_direction: str = "isotropic"
    semiconductor_se_direction: str = "isotropic"

    track_subbarrier: bool = False
    max_generation: int = 100
    n_q_sample: int = 64

    @classmethod
    def from_sample(cls, sample):
        cfg = sample.cfg
        z = getattr(sample, "Z_eff", None)
        return cls(
            emfp_energy_ref=str(cfg.emfp_energy_ref),
            imfp_energy_ref=str(cfg.imfp_energy_ref),
            elastic_min_energy_ev=float(cfg.elastic_min_energy),
            elastic_low_energy_model=str(cfg.elastic_low_energy_model),
            elastic_cutoff_energy_ev=float(cfg.elastic_cutoff_energy),
            atomic_number=0.0 if z is None else float(z),
            se_channel_rule=str(getattr(cfg, "se_channel_rule", "mao")),
            on_pauli_block=str(getattr(cfg, "on_pauli_block", "fallback")),
            se_direction_model=str(getattr(cfg, "se_direction_model", "momentum")),
            plasmon_se_direction=str(getattr(cfg, "plasmon_se_direction", "isotropic")),
            semiconductor_se_direction=str(
                getattr(cfg, "semiconductor_se_direction", "isotropic")
            ),
            track_subbarrier=bool(getattr(cfg, "track_subbarrier", False)),
            max_generation=int(getattr(cfg, "max_generation", 100)),
            n_q_sample=int(getattr(cfg, "n_q_sample", 64)),
        )

    def validate(self):
        if self.emfp_energy_ref not in {"vb_bottom", "vacuum"}:
            raise ValueError("emfp_energy_ref must be 'vb_bottom' or 'vacuum'")
        if self.imfp_energy_ref not in {"vb_bottom", "fermi"}:
            raise ValueError("imfp_energy_ref must be 'vb_bottom' or 'fermi'")
        if self.elastic_low_energy_model not in {"elsepa", "browning", "linear"}:
            raise ValueError("unsupported elastic_low_energy_model")
        if self.elastic_low_energy_model == "browning" and self.atomic_number <= 0.0:
            raise ValueError("Browning elastic model requires atomic_number/Z_eff")
        if self.se_channel_rule not in {"mao", "table"}:
            raise ValueError("se_channel_rule must be 'mao' or 'table'")
        if self.on_pauli_block not in {"fallback", "drop"}:
            raise ValueError("on_pauli_block must be 'fallback' or 'drop'")
        if self.se_direction_model not in {"momentum", "isotropic"}:
            raise ValueError("bad se_direction_model")
        if self.plasmon_se_direction not in {"momentum", "isotropic"}:
            raise ValueError("bad plasmon_se_direction")
        if self.semiconductor_se_direction not in {"momentum", "isotropic"}:
            raise ValueError("bad semiconductor_se_direction")
        if self.n_q_sample < 8:
            raise ValueError("n_q_sample must be >= 8")


class BulkCounters:
    def __init__(self, ti):
        names = (
            "elastic_events", "inelastic_events", "secondaries_queued",
            "se_below_barrier", "se_blocked_pauli", "se_pauli_fallback",
            "channel_reclassified", "omega_cdf_empty", "q_window_clipped",
            "q_cdf_empty", "step_limit_hit", "generation_limit_hit",
            "no_scattering_rate",
        )
        self.names = names
        for name in names:
            setattr(self, name, ti.field(dtype=ti.i32, shape=()))

    def as_dict(self):
        return {name: int(getattr(self, name)[None]) for name in self.names}


class InelasticSamples:
    """Per-primary last inelastic sample, primarily for v0.3 validation."""

    def __init__(self, ti, capacity, fp):
        self.valid = ti.field(dtype=ti.i32, shape=capacity)
        self.sampled_channel = ti.field(dtype=ti.i32, shape=capacity)  # 1=se, 2=pl
        self.mechanism = ti.field(dtype=ti.i32, shape=capacity)        # 1=binary, 2=plasmon, 3=semicond
        self.omega = ti.field(dtype=fp, shape=capacity)
        self.q = ti.field(dtype=fp, shape=capacity)
        self.theta_projectile = ti.field(dtype=fp, shape=capacity)
        self.secondary_valid = ti.field(dtype=ti.i32, shape=capacity)
        self.secondary_energy = ti.field(dtype=fp, shape=capacity)


def build_bulk_kernels(ti, pool, tables, fp, physics, counters, samples):
    physics.validate()
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
                                ti.atomic_add(counters.secondaries_queued[None], ti.i32(1))
                            else:
                                pool.overflow[None] = ti.i32(1)
        return ok_event

    @ti.kernel
    def seed_primaries(n: ti.i32, energy_ev: fp, sin_alpha: fp, cos_alpha: fp):
        pool.n_allocated[None] = n
        pool.overflow[None] = ti.i32(0)
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
        for i in range(n):
            pool.x[i] = fp(0.0)
            pool.y[i] = fp(0.0)
            pool.z[i] = fp(0.0)
            pool.ux[i] = sin_alpha
            pool.uy[i] = fp(0.0)
            pool.uz[i] = cos_alpha
            pool.energy[i] = energy_ev
            pool.parent_id[i] = ti.i32(-1)
            pool.root_primary_id[i] = i
            pool.generation[i] = ti.i32(0)
            pool.alive[i] = ti.i32(1)
            pool.steps[i] = ti.i32(0)
            samples.valid[i] = ti.i32(0)
            samples.sampled_channel[i] = ti.i32(0)
            samples.mechanism[i] = ti.i32(0)
            samples.omega[i] = fp(0.0)
            samples.q[i] = fp(0.0)
            samples.theta_projectile[i] = fp(0.0)
            samples.secondary_valid[i] = ti.i32(0)
            samples.secondary_energy[i] = fp(0.0)

    @ti.kernel
    def force_one_inelastic(n: ti.i32):
        for i in range(n):
            do_inelastic(i, ti.i32(0))

    @ti.kernel
    def process_wave(begin: ti.i32, end: ti.i32, max_steps: ti.i32):
        for i in range(begin, end):
            if pool.alive[i] == ti.i32(1):
                local_steps = ti.i32(0)
                running = ti.i32(1)
                while running == ti.i32(1) and local_steps < max_steps:
                    E_s = pool.energy[i]
                    if ti.static(not track_subbarrier):
                        if E_s <= fp(ui):
                            running = ti.i32(0)
                    if running == ti.i32(1):
                        inv_e = inverse_emfp(E_s)
                        inv_i = inverse_imfp(E_s)
                        inv_tot = inv_e + inv_i
                        if inv_tot <= fp(0.0):
                            ti.atomic_add(counters.no_scattering_rate[None], ti.i32(1))
                            running = ti.i32(0)
                        else:
                            flight = -ti.log(ti.max(ti.random(fp), fp(1.0e-15))) / inv_tot
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
                                ti.atomic_add(counters.elastic_events[None], ti.i32(1))
                            else:
                                do_inelastic(i, ti.i32(1))

                            if pool.overflow[None] == ti.i32(1):
                                running = ti.i32(0)
                if running == ti.i32(1) and local_steps >= max_steps:
                    ti.atomic_add(counters.step_limit_hit[None], ti.i32(1))
                pool.alive[i] = ti.i32(0)

    return seed_primaries, force_one_inelastic, process_wave


class BulkTransportEngine:
    """Persistent v0.3 bulk transport engine with dynamic cascade allocation."""

    def __init__(self, ti, fp, host_tables, physics: BulkPhysicsConfig, capacity: int):
        self.ti = ti
        self.fp = fp
        self.capacity = int(capacity)
        self.pool = ParticlePool(ti, self.capacity, fp)
        self.tables = DeviceTransportTables(ti, host_tables, fp)
        self.counters = BulkCounters(ti)
        self.samples = InelasticSamples(ti, self.capacity, fp)
        (
            self.seed_kernel,
            self.force_inelastic_kernel,
            self.process_wave_kernel,
        ) = build_bulk_kernels(
            ti, self.pool, self.tables, fp, physics, self.counters, self.samples
        )

    def seed(self, n, energy_ev, alpha_deg=0.0):
        n = int(n)
        if n < 1 or n > self.capacity:
            raise ValueError("n must be positive and <= capacity")
        a = math.radians(float(alpha_deg))
        self.seed_kernel(n, float(energy_ev), math.sin(a), math.cos(a))

    def sample_inelastic_once(self, *, n, energy_ev, alpha_deg=0.0, copy=True):
        self.seed(n, energy_ev, alpha_deg)
        self.force_inelastic_kernel(int(n))
        self.ti.sync()
        result = {"n": int(n), "diagnostics": self.counters.as_dict()}
        if copy:
            m = int(n)
            result.update({
                "valid": self.samples.valid.to_numpy()[:m],
                "sampled_channel": self.samples.sampled_channel.to_numpy()[:m],
                "mechanism": self.samples.mechanism.to_numpy()[:m],
                "omega_ev": self.samples.omega.to_numpy()[:m],
                "q_a0inv": self.samples.q.to_numpy()[:m],
                "theta_projectile_rad": self.samples.theta_projectile.to_numpy()[:m],
                "secondary_valid": self.samples.secondary_valid.to_numpy()[:m],
                "secondary_energy_ev": self.samples.secondary_energy.to_numpy()[:m],
            })
        return result

    def run_bulk_cascade(
        self, *, n, energy_ev, alpha_deg=0.0, max_steps_per_particle=1000,
        copy_state=False,
    ):
        self.seed(n, energy_ev, alpha_deg)
        begin = 0
        while begin < self.pool.allocated():
            end = min(self.pool.allocated(), self.capacity)
            self.process_wave_kernel(int(begin), int(end), int(max_steps_per_particle))
            self.ti.sync()
            begin = end
            if int(self.pool.overflow[None]):
                break

        allocated = self.pool.allocated()
        stored = min(allocated, self.capacity)
        out = {
            "n_primaries": int(n),
            "allocated": allocated,
            "stored": stored,
            "overflow": bool(self.pool.overflow[None]),
            "diagnostics": self.counters.as_dict(),
        }
        if copy_state:
            p = self.pool
            out.update({
                "x": p.x.to_numpy()[:stored], "y": p.y.to_numpy()[:stored],
                "z": p.z.to_numpy()[:stored], "ux": p.ux.to_numpy()[:stored],
                "uy": p.uy.to_numpy()[:stored], "uz": p.uz.to_numpy()[:stored],
                "energy_ev": p.energy.to_numpy()[:stored],
                "parent_id": p.parent_id.to_numpy()[:stored],
                "root_primary_id": p.root_primary_id.to_numpy()[:stored],
                "generation": p.generation.to_numpy()[:stored],
                "steps": p.steps.to_numpy()[:stored],
            })
        return out


def load_reference_bulk_tables(database_path, material="Si", config=None):
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
