"""v0.2 real elastic transport using validated SEEMC EMFP and DECS tables."""

import math
from dataclasses import dataclass
import numpy as np

from .device_tables import DeviceElasticTables
from .particle_pool import ParticlePool
from .primitives import make_primitives
from .tables import extract_reference_tables


@dataclass(frozen=True)
class ElasticPhysicsConfig:
    """Subset of reference ``MCConfig`` needed by the elastic-only kernel."""

    emfp_energy_ref: str = "vb_bottom"
    elastic_min_energy_ev: float = 5.0
    elastic_low_energy_model: str = "elsepa"
    elastic_cutoff_energy_ev: float = 50.0
    inner_potential_ev: float = 0.0
    e_fermi_ev: float = 0.0
    atomic_number: float = 0.0

    @classmethod
    def from_sample(cls, sample):
        cfg = sample.cfg
        z = getattr(sample, "Z_eff", None)
        return cls(
            emfp_energy_ref=str(cfg.emfp_energy_ref),
            elastic_min_energy_ev=float(cfg.elastic_min_energy),
            elastic_low_energy_model=str(cfg.elastic_low_energy_model),
            elastic_cutoff_energy_ev=float(cfg.elastic_cutoff_energy),
            inner_potential_ev=float(sample.Ui),
            e_fermi_ev=float(sample.e_fermi),
            atomic_number=0.0 if z is None else float(z),
        )

    def validate(self):
        if self.emfp_energy_ref not in {"vb_bottom", "vacuum"}:
            raise ValueError("emfp_energy_ref must be 'vb_bottom' or 'vacuum'")
        if self.elastic_low_energy_model not in {"elsepa", "browning", "linear"}:
            raise ValueError("unsupported elastic_low_energy_model")
        if self.elastic_low_energy_model == "browning" and self.atomic_number <= 0.0:
            raise ValueError("Browning elastic model requires atomic_number/Z_eff")


def build_elastic_kernels(ti, pool, tables, fp, physics: ElasticPhysicsConfig, event_count):
    """Compile seed + repeated real elastic-collision kernels.

    v0.2 is deliberately bulk-only: every accepted step is elastic, energy is
    unchanged, and no surface or inelastic channel is present.  EMFP lookup,
    low-energy cross-section scaling, stochastic DCS-bin selection, inverse-CDF
    sampling and direction rotation mirror the current reference implementation.
    """
    physics.validate()
    (
        _, interp1, stochastic_energy_bin, inverse_cdf_column,
        _, _, _, _,
    ) = make_primitives(ti, fp)

    vacuum_ref = physics.emfp_energy_ref == "vacuum"
    model = physics.elastic_low_energy_model
    low_model_elsepa = model == "elsepa"
    low_model_linear = model == "linear"
    low_model_browning = model == "browning"
    ui = float(physics.inner_potential_ev)
    ef = float(physics.e_fermi_ev)
    z_eff = float(physics.atomic_number)
    z17 = z_eff ** 1.7 if z_eff > 0.0 else 0.0
    emin_el = float(physics.elastic_min_energy_ev)
    cutoff = float(physics.elastic_cutoff_energy_ev)
    two_pi = float(2.0 * math.pi)

    @ti.func
    def clip_energy_abscissa(E_s: fp) -> fp:
        E_lookup = E_s
        if ti.static(vacuum_ref):
            E_lookup = ti.max(E_s - fp(ui), fp(emin_el))
        E_lookup = ti.max(E_lookup, tables.energy[0])
        E_lookup = ti.min(E_lookup, tables.energy[tables.nE - 1])
        return E_lookup

    @ti.func
    def browning_sigma(E_ev: fp) -> fp:
        # Browning et al. JAP 76 (1994); only ratios are used below.
        Ekev = ti.max(E_ev, fp(1.0e-9)) / fp(1000.0)
        rt = ti.sqrt(Ekev)
        return fp(3.0e-18) * fp(z17) / (
            Ekev + fp(0.005) * fp(z17) * rt + fp(0.0007) * fp(z_eff * z_eff) / rt
        )

    @ti.func
    def elastic_sigma_scale(E_s: fp) -> fp:
        scale = fp(1.0)
        if ti.static(low_model_linear):
            if E_s < fp(cutoff):
                if E_s <= fp(ef):
                    scale = fp(0.0)
                else:
                    scale = (E_s - fp(ef)) / ti.max(fp(cutoff - ef), fp(1.0e-9))
        elif ti.static(low_model_browning):
            if E_s < fp(cutoff):
                scale = browning_sigma(E_s) / browning_sigma(fp(cutoff))
        return scale

    @ti.func
    def inverse_emfp(E_s: fp) -> fp:
        inv_e = fp(0.0)
        if ti.static(low_model_elsepa):
            e_mfp = clip_energy_abscissa(E_s)
            inv_e = interp1(tables.energy, tables.inv_emfp, ti.i32(tables.nE), e_mfp)
        else:
            # Reference behavior: anchor ELSEPA at the cutoff, then rescale the
            # total elastic cross section below it. Above cutoff use the table.
            if E_s >= fp(cutoff):
                e_mfp = clip_energy_abscissa(E_s)
                inv_e = interp1(tables.energy, tables.inv_emfp, ti.i32(tables.nE), e_mfp)
            else:
                e_c = clip_energy_abscissa(fp(cutoff))
                inv_c = interp1(tables.energy, tables.inv_emfp, ti.i32(tables.nE), e_c)
                inv_e = inv_c * elastic_sigma_scale(E_s)
        return inv_e

    @ti.func
    def dcs_lookup_abscissa(E_s: fp) -> fp:
        E_look = E_s
        if ti.static(not low_model_elsepa):
            E_look = ti.max(E_s, fp(cutoff))
        return clip_energy_abscissa(E_look)

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

    @ti.kernel
    def seed_primaries(n: ti.i32, energy_ev: fp, sin_alpha: fp, cos_alpha: fp):
        pool.n_allocated[None] = n
        pool.overflow[None] = ti.i32(0)
        event_count[None] = ti.i32(0)
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

    @ti.kernel
    def elastic_steps(n: ti.i32, n_steps: ti.i32):
        for i in range(n):
            E_s = pool.energy[i]
            local_events = ti.i32(0)
            for _ in range(n_steps):
                inv_e = inverse_emfp(E_s)
                if inv_e > fp(0.0):
                    uflight = ti.max(ti.random(fp), fp(1.0e-15))
                    flight = -ti.log(uflight) / inv_e
                    pool.x[i] += flight * pool.ux[i]
                    pool.y[i] += flight * pool.uy[i]
                    pool.z[i] += flight * pool.uz[i]

                    e_dcs = dcs_lookup_abscissa(E_s)
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
                    pool.steps[i] += ti.i32(1)
                    local_events += ti.i32(1)
            if local_events > 0:
                ti.atomic_add(event_count[None], local_events)

    return seed_primaries, elastic_steps


class ElasticTransportEngine:
    """Persistent device allocation/kernels so repeated runs are truly warmed."""

    def __init__(self, ti, fp, host_tables, physics: ElasticPhysicsConfig, capacity: int):
        self.ti = ti
        self.fp = fp
        self.capacity = int(capacity)
        self.pool = ParticlePool(ti, self.capacity, fp)
        self.tables = DeviceElasticTables(ti, host_tables, fp)
        self.event_count = ti.field(dtype=ti.i32, shape=())
        self.seed_kernel, self.step_kernel = build_elastic_kernels(
            ti, self.pool, self.tables, fp, physics, self.event_count
        )

    def run(self, *, n, energy_ev, alpha_deg=0.0, collisions=100, copy_state=True):
        n = int(n)
        collisions = int(collisions)
        if n < 1 or n > self.capacity or collisions < 1:
            raise ValueError("invalid n/collisions or n exceeds engine capacity")
        a = math.radians(float(alpha_deg))
        self.seed_kernel(n, float(energy_ev), math.sin(a), math.cos(a))
        self.step_kernel(n, collisions)
        self.ti.sync()
        result = {
            "n": n,
            "collisions_requested": collisions,
            "collisions_completed": int(self.event_count[None]),
        }
        if copy_state:
            p = self.pool
            result.update({
                "x": p.x.to_numpy()[:n], "y": p.y.to_numpy()[:n], "z": p.z.to_numpy()[:n],
                "ux": p.ux.to_numpy()[:n], "uy": p.uy.to_numpy()[:n], "uz": p.uz.to_numpy()[:n],
                "energy_ev": p.energy.to_numpy()[:n], "steps": p.steps.to_numpy()[:n],
            })
        return result


def run_elastic_transport(ti, fp, host_tables, *, physics, n=100_000,
                          energy_ev=1000.0, alpha_deg=0.0, collisions=100,
                          capacity=None, copy_state=True):
    """Convenience one-shot wrapper; benchmarks should reuse ElasticTransportEngine."""
    capacity = int(n if capacity is None else capacity)
    engine = ElasticTransportEngine(ti, fp, host_tables, physics, capacity)
    return engine.run(n=n, energy_ev=energy_ev, alpha_deg=alpha_deg,
                      collisions=collisions, copy_state=copy_state)


def load_reference_host_tables(database_path, material="Si", config=None):
    try:
        from seemc_imaging.transport import MCConfig, Sample
    except ImportError as exc:
        raise RuntimeError(
            "Install seemc-imaging>=0.7.4 or this package with the reference extra"
        ) from exc
    cfg = config or MCConfig()
    sample = Sample(material, db_path=str(database_path), config=cfg)
    return sample, extract_reference_tables(sample)


def elastic_summary(result):
    u = np.column_stack([result["ux"], result["uy"], result["uz"]])
    r = np.column_stack([result["x"], result["y"], result["z"]])
    norms = np.linalg.norm(u, axis=1)
    displacement = np.linalg.norm(r, axis=1)
    return {
        "mean_ux": float(np.mean(u[:, 0])),
        "mean_uy": float(np.mean(u[:, 1])),
        "mean_uz": float(np.mean(u[:, 2])),
        "mean_direction_norm": float(np.mean(norms)),
        "max_direction_norm_error": float(np.max(np.abs(norms - 1.0))),
        "mean_displacement_A": float(np.mean(displacement)),
        "rms_displacement_A": float(np.sqrt(np.mean(displacement ** 2))),
        "mean_steps": float(np.mean(result["steps"])),
    }
