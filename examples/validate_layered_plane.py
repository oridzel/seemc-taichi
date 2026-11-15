"""Limit checks for the film-on-substrate planar engine with the real database.

  python3 examples/validate_layered_plane.py ../MaterialDatabase.pkl \
      --film C --substrate W --arch cuda --precision f64 --energy 1000 --n 20000

Three runs are compared with the validated single-material plane engine:

  1. film on a substrate of the SAME material, any thickness  -> must equal plane(film)
  2. a very thick film on the real substrate                   -> must equal plane(film)
  3. the requested thickness on the real substrate             -> the new result

Runs 1 and 2 are statistical identities; a pull beyond ~3 in either is a
failure.  Compilation of the two-material kernels takes several minutes per
engine on the first run.
"""
import argparse
import math

import numpy as np

from seemc_taichi.config import BackendConfig
from seemc_taichi.layered_plane import LayeredPlaneTransportEngine
from seemc_taichi.plane import (
    BulkPhysicsConfig, PlaneTransportEngine, load_reference_plane_tables,
    per_primary_emission_counts, yield_sem_from_counts,
)
from seemc_taichi.runtime import init_taichi
from seemc_taichi.surface import SurfacePhysicsConfig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("database")
    p.add_argument("--film", default="C")
    p.add_argument("--substrate", default="W")
    p.add_argument("--film-thickness-nm", type=float, default=70.0)
    p.add_argument("--interface-alignment", choices=("vacuum", "fermi"), default="vacuum")
    p.add_argument("--arch", default="cpu")
    p.add_argument("--precision", choices=("f32", "f64"), default="f64")
    p.add_argument("--energy", type=float, default=1000.0)
    p.add_argument("--angle", type=float, default=0.0)
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--capacity", type=int, default=5_000_000)
    p.add_argument("--seed", type=int, default=20261114)
    a = p.parse_args()

    film_sample, film = load_reference_plane_tables(a.database, a.film)
    sub_sample, sub = load_reference_plane_tables(a.database, a.substrate)
    film_bulk = BulkPhysicsConfig.from_sample(film_sample)
    sub_bulk = BulkPhysicsConfig.from_sample(sub_sample)
    surface = SurfacePhysicsConfig.from_sample(film_sample)
    cutoff = float(getattr(film_sample.cfg, "bse_cutoff_ev", 50.0))

    ti = init_taichi(BackendConfig(arch=a.arch, max_particles=a.capacity,
                                   default_fp=a.precision, random_seed=a.seed))
    fp = ti.f32 if a.precision == "f32" else ti.f64

    def run(tag, engine):
        r = engine.run_plane(n=a.n, energy_vac_ev=a.energy, alpha_deg=a.angle, copy_emissions=True)
        if r["overflow"] or r["emission_overflow"]:
            raise RuntimeError(f"{tag}: overflow; increase --capacity")
        total, se, bse = per_primary_emission_counts(r, cutoff)
        e = np.asarray(r["emission_energy_ev"])
        per_primary_energy = np.bincount(r["emission_root_primary_id"], weights=e, minlength=a.n)
        d = r["diagnostics"]
        out = dict(tey=total.mean(), tey_sem=yield_sem_from_counts(total),
                   sey=se.mean(), sey_sem=yield_sem_from_counts(se),
                   bsey=bse.mean(), bsey_sem=yield_sem_from_counts(bse))
        print(f"{tag:42s} TEY {out['tey']:.4f}+-{out['tey_sem']:.4f}  "
              f"SEY {out['sey']:.4f}+-{out['sey_sem']:.4f}  BSEY {out['bsey']:.4f}+-{out['bsey_sem']:.4f}  "
              f"max(sum E - E0) {per_primary_energy.max() - a.energy:+.3g} eV  "
              f"interface: {d.get('interface_encounters', 0)} hits, "
              f"{d.get('interface_reflections', 0)} reflected")
        return out

    def pull(x, y, k):
        s = math.hypot(x[k + "_sem"], y[k + "_sem"])
        return (x[k] - y[k]) / s if s > 0 else 0.0

    def layered(substrate_tables, substrate_bulk, thickness):
        return LayeredPlaneTransportEngine(
            ti, fp, film, substrate_tables, film_bulk, substrate_bulk, surface, a.capacity,
            film_thickness_nm=thickness, bse_cutoff_ev=cutoff,
            interface_alignment=a.interface_alignment,
        )

    ref = run(f"plane {a.film}", PlaneTransportEngine(ti, fp, film, film_bulk, surface, a.capacity, bse_cutoff_ev=cutoff))
    same = run(f"{a.film} {a.film_thickness_nm:g} nm on {a.film}", layered(film, film_bulk, a.film_thickness_nm))
    eng = layered(sub, sub_bulk, 1.0e6)
    print("interface:", eng.describe())
    thick = run(f"{a.film} 1e6 nm on {a.substrate}", eng)
    eng.set_film_thickness_nm(a.film_thickness_nm)
    new = run(f"{a.film} {a.film_thickness_nm:g} nm on {a.substrate}", eng)

    print()
    ok = True
    for tag, x in (("same-material film", same), ("very thick film", thick)):
        pulls = {k: pull(x, ref, k) for k in ("tey", "sey", "bsey")}
        worst = max(abs(v) for v in pulls.values())
        ok &= worst < 3.5
        print(f"{tag:20s} vs plane {a.film}: pulls " + ", ".join(f"{k} {v:+.2f}" for k, v in pulls.items())
              + ("   OK" if worst < 3.5 else "   FAIL"))
    print(f"\n{a.film} {a.film_thickness_nm:g} nm on {a.substrate} relative to bulk {a.film}: "
          f"TEY x{new['tey'] / ref['tey']:.3f}, SEY x{new['sey'] / ref['sey']:.3f}, BSEY x{new['bsey'] / ref['bsey']:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
