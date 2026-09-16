import argparse
import math
import time

import numpy as np

from seemc_taichi.config import BackendConfig
from seemc_taichi.plane import (
    BulkPhysicsConfig,
    PlaneTransportEngine,
    load_reference_plane_tables,
    per_primary_emission_counts,
    yield_sem_from_counts,
)
from seemc_taichi.runtime import init_taichi
from seemc_taichi.surface import SurfacePhysicsConfig

p = argparse.ArgumentParser(
    description="Compare v0.5 planar Taichi yields against Python SEEMC"
)
p.add_argument("database")
p.add_argument("--material", default="Si")
p.add_argument("--arch", choices=("cpu", "metal", "cuda", "vulkan", "gpu"), default="cpu")
p.add_argument("--precision", choices=("f32", "f64"), default=None)
p.add_argument("--energy", type=float, default=1000.0)
p.add_argument("--angle", type=float, default=0.0)
p.add_argument("--n", type=int, default=2000, help="primaries in each implementation")
p.add_argument("--capacity", type=int, default=200_000)
p.add_argument("--steps-per-chunk", type=int, default=256)
p.add_argument("--seed", type=int, default=20260915)
p.add_argument("--bins", type=int, default=100)
args = p.parse_args()

precision = args.precision or ("f32" if args.arch == "metal" else "f64")
if args.arch == "metal" and precision == "f64":
    p.error("Metal requires f32 on this Taichi build")
if not 0.0 <= args.angle < 90.0:
    p.error("--angle must be in [0, 90)")

sample, host = load_reference_plane_tables(args.database, args.material)
bulk = BulkPhysicsConfig.from_sample(sample)
surface = SurfacePhysicsConfig.from_sample(sample)
cutoff = float(getattr(sample.cfg, "bse_cutoff_ev", 50.0))

# ---------------- reference Python SEEMC ----------------
try:
    from seemc_imaging.geometry import Plane
    from seemc_imaging.transport import MCConfig, Sample, simulate_trajectory
except ImportError as exc:
    raise RuntimeError("validate_plane_yield requires seemc-imaging") from exc

ref_cfg = MCConfig(**{**sample.cfg.__dict__, "collect_spectra": True})
ref_cfg.validate()
ref_sample = Sample(args.material, db_path=str(args.database), config=ref_cfg)
plane = Plane()
a = math.radians(args.angle)
vacuum = (math.sin(a), 0.0, math.cos(a))
outward = (0.0, 0.0, -1.0)
ref_counts = np.zeros((args.n, 3), dtype=np.int64)
ref_energies = []

t0 = time.perf_counter()
for i in range(args.n):
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0, i]))
    rr = simulate_trajectory(
        ref_sample,
        float(args.energy),
        a,
        rng,
        trajectory_id=i,
        geometry=plane,
        vacuum_direction=vacuum,
        surface_normal=outward,
    )
    ref_counts[i, 0] = int(rr.tey)
    ref_counts[i, 1] = int(rr.sey_50ev)
    ref_counts[i, 2] = int(rr.bse_50ev)
    if rr.emissions:
        ref_energies.extend(float(e.energy) for e in rr.emissions)
ref_elapsed = time.perf_counter() - t0

# ---------------- Taichi plane ----------------
cfg = BackendConfig(
    arch=args.arch,
    max_particles=args.capacity,
    default_fp=precision,
    default_ip="i32",
    random_seed=args.seed,
)
ti = init_taichi(cfg)
fp = ti.f32 if precision == "f32" else ti.f64
engine = PlaneTransportEngine(
    ti, fp, host, bulk, surface, args.capacity, bse_cutoff_ev=cutoff
)

# Compile first; then reseed and run the requested independent batch.
engine.run_plane(
    n=min(args.n, 64),
    energy_vac_ev=args.energy,
    alpha_deg=args.angle,
    steps_per_chunk=min(args.steps_per_chunk, 32),
    copy_emissions=False,
)
ti.sync()

t1 = time.perf_counter()
dev = engine.run_plane(
    n=args.n,
    energy_vac_ev=args.energy,
    alpha_deg=args.angle,
    steps_per_chunk=args.steps_per_chunk,
    copy_emissions=True,
)
ti.sync()
dev_elapsed = time.perf_counter() - t1

if dev["overflow"] or dev.get("emission_overflow", False):
    raise RuntimeError(
        "Taichi pool/emission buffer overflowed; increase --capacity before comparing yields"
    )

dev_total, dev_se, dev_bse = per_primary_emission_counts(dev, cutoff)
dev_counts = np.column_stack((dev_total, dev_se, dev_bse))

names = ("TEY", f"SEY(E<={cutoff:g})", f"BSEY(E>{cutoff:g})")
print(f"material: {args.material} ({'metal' if sample.is_metal else 'nonconductor'})")
print(f"E0 / angle: {args.energy:g} eV / {args.angle:g} deg")
print(f"reference: Python SEEMC; device: {args.arch}/{precision}")
print(f"n per implementation: {args.n:,}")
print()
print(f"{'quantity':<18s} {'reference':>12s} {'Taichi':>12s} {'delta':>12s} {'combined pull':>15s}")
for j, name in enumerate(names):
    ref_mean = float(ref_counts[:, j].mean())
    dev_mean = float(dev_counts[:, j].mean())
    ref_sem = yield_sem_from_counts(ref_counts[:, j])
    dev_sem = yield_sem_from_counts(dev_counts[:, j])
    combined = math.sqrt(ref_sem * ref_sem + dev_sem * dev_sem)
    pull = (dev_mean - ref_mean) / combined if combined > 0 else float("nan")
    print(f"{name:<18s} {ref_mean:12.7f} {dev_mean:12.7f} {dev_mean-ref_mean:+12.7f} {pull:15.3f}")

ref_e = np.asarray(ref_energies, dtype=float)
dev_e = np.asarray(dev["emission_energy_ev"], dtype=float)
if ref_e.size and dev_e.size:
    emax = max(float(args.energy), float(ref_e.max()), float(dev_e.max()))
    edges = np.linspace(0.0, emax, args.bins + 1)
    hr, _ = np.histogram(ref_e, bins=edges)
    hd, _ = np.histogram(dev_e, bins=edges)
    pr = hr / max(hr.sum(), 1)
    pd = hd / max(hd.sum(), 1)
    tv = 0.5 * float(np.abs(pr - pd).sum())
    print()
    print(f"emission-energy histogram TV: {tv:.6f}")

norm = np.sqrt(
    np.asarray(dev["emission_ux"]) ** 2
    + np.asarray(dev["emission_uy"]) ** 2
    + np.asarray(dev["emission_uz"]) ** 2
)
print(f"device emitted direction max norm error: {np.max(np.abs(norm-1.0)) if norm.size else 0.0:.6g}")
print()
print(f"reference elapsed: {ref_elapsed:.6f} s")
print(f"Taichi elapsed (warmed): {dev_elapsed:.6f} s")
if dev_elapsed > 0:
    print(f"primary throughput: {args.n/dev_elapsed:,.0f} primaries/s")
print(f"device allocated/stored: {dev['allocated']:,} / {dev['stored']:,}")
print("device diagnostics:")
for key, value in dev["diagnostics"].items():
    print(f"  {key:<30s} {value:,}")
