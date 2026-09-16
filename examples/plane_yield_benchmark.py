import argparse
import time

from seemc_taichi.config import BackendConfig
from seemc_taichi.plane import (
    BulkPhysicsConfig,
    PlaneTransportEngine,
    load_reference_plane_tables,
)
from seemc_taichi.runtime import init_taichi
from seemc_taichi.surface import SurfacePhysicsConfig

p = argparse.ArgumentParser(
    description="SEEMC-Taichi v0.5 planar TEY/SEY/BSEY benchmark"
)
p.add_argument("database")
p.add_argument("--material", default="Si")
p.add_argument("--arch", choices=("cpu", "metal", "cuda", "vulkan", "gpu"), default="cpu")
p.add_argument("--precision", choices=("f32", "f64"), default=None)
p.add_argument("--energy", type=float, default=1000.0, help="incident vacuum energy E0 [eV]")
p.add_argument("--angle", type=float, default=0.0, help="incidence angle from surface normal [deg]")
p.add_argument("--n", type=int, default=10_000)
p.add_argument("--capacity", type=int, default=500_000)
p.add_argument("--steps-per-chunk", type=int, default=256)
p.add_argument("--seed", type=int, default=20260915)
p.add_argument("--copy-emissions", action="store_true")
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

# Compile all kernels on a small real planar cascade.  It is deliberately
# excluded from the reported timing.
warm_n = min(args.n, 128)
engine.run_plane(
    n=warm_n,
    energy_vac_ev=args.energy,
    alpha_deg=args.angle,
    steps_per_chunk=min(args.steps_per_chunk, 32),
    copy_emissions=False,
)
ti.sync()

start = time.perf_counter()
r = engine.run_plane(
    n=args.n,
    energy_vac_ev=args.energy,
    alpha_deg=args.angle,
    steps_per_chunk=args.steps_per_chunk,
    copy_emissions=args.copy_emissions,
)
ti.sync()
elapsed = time.perf_counter() - start

diag = r["diagnostics"]
collisions = diag["elastic_events"] + diag["inelastic_events"]
print(f"material: {args.material} ({'metal' if sample.is_metal else 'nonconductor'})")
print(f"incident E0 (vacuum): {args.energy:g} eV")
print(f"incidence angle: {args.angle:g} deg")
print(f"arch / precision: {args.arch} / {precision}")
print(f"barrier: {surface.barrier_model}; Ui={surface.inner_potential_ev:.9g} eV")
print(f"primaries: {args.n:,}")
print(f"capacity: {args.capacity:,}")
print(f"allocated/stored: {r['allocated']:,} / {r['stored']:,}")
print(f"particle overflow: {r['overflow']}")
print(f"wave count / kernel chunks: {r['wave_count']:,} / {r['kernel_chunks']:,}")
print()
print(f"TEY:  {r['tey']:.8f}")
print(f"SEY (E <= {cutoff:g} eV): {r['sey_50ev']:.8f}")
print(f"BSEY (E > {cutoff:g} eV): {r['bsey_50ev']:.8f}")
print(f"cascade-emission yield: {r['cascade_yield']:.8f}")
print(f"primary-emission yield: {r['primary_yield']:.8f}")
print()
print(f"elastic events: {diag['elastic_events']:,}")
print(f"inelastic events: {diag['inelastic_events']:,}")
print(f"secondaries queued: {diag['secondaries_queued']:,}")
print(f"surface encounters: {diag['surface_encounters']:,}")
print(f"escapes: {diag['escapes']:,}")
print(f"elapsed: {elapsed:.6f} s")
if elapsed > 0:
    print(f"completed collision events/s: {collisions / elapsed:,.0f}")
    print(f"primaries/s: {args.n / elapsed:,.0f}")
print("diagnostics:")
for key, value in diag.items():
    print(f"  {key:<30s} {value:,}")

if r["overflow"]:
    print("WARNING: particle pool overflowed; yields are truncated and must not be used")
if diag["emission_buffer_overflow"]:
    print("WARNING: emission buffer overflowed; copied emissions/yields are incomplete")
if diag["step_limit_hit"]:
    print("WARNING: host safety limit was reached; this is not a valid completed run")
