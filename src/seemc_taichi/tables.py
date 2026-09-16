from __future__ import annotations

from dataclasses import dataclass
import numpy as np

H2EV = 27.21184


@dataclass(frozen=True)
class HostMaterialTables:
    """Contiguous host-side copy of the validated SEEMC lookup tables.

    v0.3 adds the full electronic-loss data needed for a real inelastic event:
    the total/channel inverse IMFPs, channel DIIMFP cumulative distributions,
    and the channel-resolved ELF on (omega, log q).  Energies stored on
    particles remain ``E_s``: eV measured from the valence-band bottom.
    """

    energy_ev: np.ndarray
    inv_emfp: np.ndarray
    inv_imfp: np.ndarray
    inv_imfp_se: np.ndarray
    inv_imfp_pl: np.ndarray
    emfp: np.ndarray

    elastic_theta_rad: np.ndarray
    elastic_cdf: np.ndarray

    diimfp_se_eloss_ev: np.ndarray
    diimfp_se_cdf: np.ndarray
    diimfp_pl_eloss_ev: np.ndarray
    diimfp_pl_cdf: np.ndarray

    omega_hartree: np.ndarray
    qlog_a0inv: np.ndarray
    elf_se: np.ndarray
    elf_pl: np.ndarray

    is_metal: bool
    e_fermi_ev: float
    e_fermi_feg_ev: float
    e_vb_ev: float
    band_gap_ev: float
    e_cbm_ev: float
    inner_potential_ev: float

    def validate(self) -> None:
        e = np.asarray(self.energy_ev)
        if e.ndim != 1 or e.size < 2 or not np.all(np.isfinite(e)):
            raise ValueError("energy grid must be a finite one-dimensional array")
        if not np.all(np.diff(e) > 0):
            raise ValueError("energy grid must be strictly increasing")
        n = e.size

        for name in ("inv_emfp", "inv_imfp", "inv_imfp_se", "inv_imfp_pl", "emfp"):
            a = np.asarray(getattr(self, name))
            if a.shape != (n,):
                raise ValueError(f"{name} has shape {a.shape}, expected {(n,)}")
            if np.any(~np.isfinite(a)) or np.any(a < 0):
                raise ValueError(f"{name} must be finite and non-negative after extraction")

        th = np.asarray(self.elastic_theta_rad)
        cdf = np.asarray(self.elastic_cdf)
        if th.ndim != 1 or th.size < 2 or not np.all(np.diff(th) > 0):
            raise ValueError("elastic theta grid must be strictly increasing")
        if cdf.shape != (th.size, n):
            raise ValueError("elastic CDF shape does not match theta x energy")
        if np.any(~np.isfinite(cdf)):
            raise ValueError("elastic CDF contains non-finite values")
        if np.any(np.diff(cdf, axis=0) < -1e-12):
            raise ValueError("elastic CDF must be non-decreasing")
        if not np.allclose(cdf[0, :], 0.0, atol=1e-10):
            raise ValueError("elastic CDF must start at zero")
        if not np.allclose(cdf[-1, :], 1.0, atol=1e-8):
            raise ValueError("elastic CDF must end at one")

        for prefix in ("se", "pl"):
            w = np.asarray(getattr(self, f"diimfp_{prefix}_eloss_ev"))
            cc = np.asarray(getattr(self, f"diimfp_{prefix}_cdf"))
            if w.ndim != 2 or w.shape[1] != n:
                raise ValueError(f"diimfp_{prefix}_eloss_ev must be (Nloss, Nenergy)")
            if cc.shape != w.shape:
                raise ValueError(f"diimfp_{prefix}_cdf must match its loss grid")
            if np.any(~np.isfinite(w)) or np.any(~np.isfinite(cc)):
                raise ValueError(f"{prefix} DIIMFP lookup tables must be finite")
            if np.any(np.diff(w, axis=0) < 0.0):
                raise ValueError(f"{prefix} DIIMFP loss grid must be non-decreasing")
            if np.any(np.diff(cc, axis=0) < -1e-12):
                raise ValueError(f"{prefix} DIIMFP CDF must be non-decreasing")

        wh = np.asarray(self.omega_hartree)
        ql = np.asarray(self.qlog_a0inv)
        if wh.ndim != 1 or wh.size < 2 or not np.all(np.diff(wh) > 0):
            raise ValueError("ELF omega grid must be strictly increasing")
        if ql.ndim != 1 or ql.size < 2 or not np.all(np.diff(ql) > 0):
            raise ValueError("ELF log-q grid must be strictly increasing")
        for name in ("elf_se", "elf_pl"):
            a = np.asarray(getattr(self, name))
            if a.shape != (wh.size, ql.size):
                raise ValueError(
                    f"{name} has shape {a.shape}, expected {(wh.size, ql.size)}"
                )
            if np.any(~np.isfinite(a)) or np.any(a < 0):
                raise ValueError(f"{name} must be finite and non-negative")

        scalars = (
            self.e_fermi_ev, self.e_fermi_feg_ev, self.e_vb_ev,
            self.band_gap_ev, self.e_cbm_ev, self.inner_potential_ev,
        )
        if not np.all(np.isfinite(np.asarray(scalars, float))):
            raise ValueError("material energy metadata must be finite")


def _contig(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def _finite_nonnegative(a):
    out = np.asarray(a, dtype=np.float64).copy()
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out[out < 0.0] = 0.0
    return np.ascontiguousarray(out)


def extract_reference_tables(sample) -> HostMaterialTables:
    """Extract the current ``seemc_imaging.transport.Sample`` tables.

    Reference-dependent transformations (energy references, channel rules,
    semiconductor thresholds) are *not* baked into arrays here; v0.3 carries
    those model choices separately in ``BulkPhysicsConfig`` so the Taichi
    implementation can be compared option-for-option with the Python kernel.
    """
    required = (
        "Egrid", "material_data", "_elastic_theta", "_elastic_cdf",
        "_w_se", "_cdf_se", "_w_pl", "_cdf_pl", "_qlog_grid",
        "e_fermi", "Ui",
    )
    missing = [name for name in required if not hasattr(sample, name)]
    if missing:
        raise TypeError(f"object is not a compatible SEEMC Sample; missing {missing}")

    md = sample.material_data
    emfp = _finite_nonnegative(md["emfp"])
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_emfp = np.where(emfp > 0.0, 1.0 / emfp, 0.0)
    inv_emfp = _finite_nonnegative(inv_emfp)

    if hasattr(sample, "inv_imfp_table"):
        inv_imfp = _finite_nonnegative(sample.inv_imfp_table)
    elif "imfp" in md:
        imfp = np.asarray(md["imfp"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_imfp = np.where(imfp > 0.0, 1.0 / imfp, 0.0)
        inv_imfp = _finite_nonnegative(inv_imfp)
    else:
        inv_imfp = _finite_nonnegative(
            np.asarray(md["inv_imfp_se"], float) + np.asarray(md["inv_imfp_pl"], float)
        )

    inv_se = _finite_nonnegative(md["inv_imfp_se"])
    inv_pl = _finite_nonnegative(md["inv_imfp_pl"])

    if hasattr(sample, "_omega_h_grid"):
        omega_h = _contig(sample._omega_h_grid)
    else:
        omega_h = _contig(np.asarray(md["omega"], float) / H2EV)
    qlog = _contig(sample._qlog_grid)

    elf_se = np.asarray(md["elf_se"], dtype=float)
    elf_pl = np.asarray(md["elf_pl"], dtype=float)
    wanted = (omega_h.size, qlog.size)
    if elf_se.shape != wanted:
        if elf_se.T.shape == wanted:
            elf_se = elf_se.T
            elf_pl = elf_pl.T
        else:
            raise ValueError(
                f"ELF shape {elf_se.shape} is incompatible with omega/q grids {wanted}"
            )
    elf_se = _finite_nonnegative(elf_se)
    elf_pl = _finite_nonnegative(elf_pl)

    is_metal = bool(getattr(sample, "is_metal", md.get("is_metal", True)))
    e_fermi = float(sample.e_fermi)
    e_fermi_feg = float(getattr(sample, "e_fermi_feg", e_fermi))
    e_vb = float(getattr(sample, "e_vb", md.get("e_vb", md.get("width_of_the_valence_band", 0.0))))
    gap = float(getattr(sample, "band_gap", md.get("band_gap", md.get("e_gap", 0.0))))
    e_cbm = float(getattr(sample, "e_cbm", e_vb + gap))

    out = HostMaterialTables(
        energy_ev=_contig(sample.Egrid),
        inv_emfp=inv_emfp,
        inv_imfp=inv_imfp,
        inv_imfp_se=inv_se,
        inv_imfp_pl=inv_pl,
        emfp=emfp,
        elastic_theta_rad=_contig(sample._elastic_theta),
        elastic_cdf=_contig(sample._elastic_cdf),
        diimfp_se_eloss_ev=_contig(sample._w_se),
        diimfp_se_cdf=_finite_nonnegative(sample._cdf_se),
        diimfp_pl_eloss_ev=_contig(sample._w_pl),
        diimfp_pl_cdf=_finite_nonnegative(sample._cdf_pl),
        omega_hartree=omega_h,
        qlog_a0inv=qlog,
        elf_se=elf_se,
        elf_pl=elf_pl,
        is_metal=is_metal,
        e_fermi_ev=e_fermi,
        e_fermi_feg_ev=e_fermi_feg,
        e_vb_ev=e_vb,
        band_gap_ev=gap,
        e_cbm_ev=e_cbm,
        inner_potential_ev=float(sample.Ui),
    )
    out.validate()
    return out
