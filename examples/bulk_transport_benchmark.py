import argparse
import time

from seemc_taichi.bulk import BulkPhysicsConfig, BulkTransportEngine, load_reference_bulk_tables
from seemc_taichi.config import BackendConfig
from seemc_taichi.runtime import init_taichi

p = argparse.ArgumentParser(description="SEEMC-Taichi v0.3 mixed bulk/cascade benchmark")
p.add_argument("database")
p.add_argument("--material", default="Si")
p.add_argument("--arch", choices=("cpu", "metal", "cuda", "vulkan", "gpu"), default="cpu")
p.add_argument("--precision", choices=("f32", "f64"), default=None)
p.add_argument("--energy", type=float, default=1000.0)
p.add_argument("--angle", type=float, default=0.0)
p.add_argument("--n", type=int, default=10_000)
p.add_argument("--capacity", type=int, default=200_000)
p.add_argument("--steps", type=int, default=250, help="maximum collision attempts per particle in bulk")
p.add_argument("--seed", type=int, default=20260915)
p.add_argument("--copy", action="store_true")
args = p.parse_args()

precision = args.precision or ("f32" if args.arch == "metal" else "f64")
if args.arch == "metal" and precision == "f64":
    p.error("Metal requires f32 on this Taichi build")

sample, host = load_reference_bulk_tables(args.database, args.material)
physics = BulkPhysicsConfig.from_sample(sample)
cfg = BackendConfig(
    arch=args.arch,
    max_particles=args.capacity,
    default_fp=precision,
    default_ip="i32",
    random_seed=args.seed,
)
ti = init_taichi(cfg)
fp = ti.f32 if precision == "f32" else ti.f64
engine = BulkTransportEngine(ti, fp, host, physics, args.capacity)

# JIT pass on a small batch but the same kernels/tables.
engine.run_bulk_cascade(
    n=min(args.n, 256), energy_ev=args.energy, alpha_deg=args.angle,
    max_steps_per_particle=min(args.steps, 8), copy_state=False,
)
ti.sync()

start = time.perf_counter()
r = engine.run_bulk_cascade(
    n=args.n, energy_ev=args.energy, alpha_deg=args.angle,
    max_steps_per_particle=args.steps, copy_state=args.copy,
)
ti.sync()
elapsed = time.perf_counter() - start

diag = r["diagnostics"]
events = diag["elastic_events"] + diag["inelastic_events"]
print(f"material: {args.material} ({'metal' if sample.is_metal else 'nonconductor'})")
print(f"energy E_s: {args.energy:g} eV")
print(f"arch / precision: {args.arch} / {precision}")
print(f"primaries: {args.n:,}")
print(f"capacity: {args.capacity:,}")
print(f"allocated/stored: {r['allocated']:,} / {r['stored']:,}")
print(f"overflow: {r['overflow']}")
print(f"elastic events: {diag['elastic_events']:,}")
print(f"inelastic events: {diag['inelastic_events']:,}")
print(f"secondaries queued: {diag['secondaries_queued']:,}")
print(f"elapsed: {elapsed:.6f} s")
print(f"completed collision events/s: {events / elapsed:,.0f}" if elapsed > 0 else "completed collision events/s: inf")
print("diagnostics:")
for key, value in diag.items():
    print(f"  {key:<24s} {value:,}")

if r["overflow"]:
    print("WARNING: particle pool overflowed; throughput is for a partial cascade, not a completed-cascade benchmark")
if diag["step_limit_hit"]:
    print("WARNING: one or more electrons reached --steps and were terminated by the v0.3 safety cap")
