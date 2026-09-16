import argparse
import time

from seemc_taichi.config import BackendConfig
from seemc_taichi.elastic import (
    ElasticPhysicsConfig,
    ElasticTransportEngine,
    elastic_summary,
    load_reference_host_tables,
)
from seemc_taichi.runtime import init_taichi


p = argparse.ArgumentParser(description="SEEMC-Taichi v0.2 real elastic transport benchmark")
p.add_argument("database")
p.add_argument("--material", default="Si")
p.add_argument("--arch", choices=("cpu", "metal", "cuda", "vulkan", "gpu"), default="cpu")
p.add_argument("--precision", choices=("f32", "f64"), default=None)
p.add_argument("--energy", type=float, default=1000.0)
p.add_argument("--angle", type=float, default=0.0)
p.add_argument("--n", type=int, default=100_000)
p.add_argument("--collisions", type=int, default=100)
p.add_argument("--seed", type=int, default=20260819)
p.add_argument("--no-copy", action="store_true", help="benchmark kernels without copying particle arrays back")
args = p.parse_args()

precision = args.precision or ("f32" if args.arch == "metal" else "f64")
if args.arch == "metal" and precision == "f64":
    p.error("Taichi Metal on this platform does not support f64; use --precision f32")

# Build the validated reference Sample before ti.init; it also carries the
# exact elastic energy-reference configuration we mirror on device.
sample, host = load_reference_host_tables(args.database, args.material)
physics = ElasticPhysicsConfig.from_sample(sample)

cfg = BackendConfig(
    arch=args.arch,
    max_particles=args.n,
    default_fp=precision,
    default_ip="i32",
    random_seed=args.seed,
)
ti = init_taichi(cfg)
fp = ti.f32 if precision == "f32" else ti.f64

engine = ElasticTransportEngine(ti, fp, host, physics, capacity=args.n)

# First call includes JIT compilation. The second reuses exactly the same
# fields and compiled kernels, so it is a genuine warmed measurement.
t0 = time.perf_counter()
r0 = engine.run(n=args.n, energy_ev=args.energy, alpha_deg=args.angle,
                collisions=args.collisions, copy_state=not args.no_copy)
ti.sync()
jit_elapsed = time.perf_counter() - t0

t0 = time.perf_counter()
r = engine.run(n=args.n, energy_ev=args.energy, alpha_deg=args.angle,
               collisions=args.collisions, copy_state=not args.no_copy)
ti.sync()
elapsed = time.perf_counter() - t0

events = int(r["collisions_completed"])
print(f"material: {args.material}")
print(f"energy E_s: {args.energy:g} eV")
print(f"arch / precision: {args.arch} / {precision}")
print(f"EMFP reference: {physics.emfp_energy_ref}")
print(f"elastic model: {physics.elastic_low_energy_model}")
print(f"primaries: {args.n:,}")
print(f"collisions/primary: {args.collisions:,}")
print(f"elastic events completed: {events:,}")
print(f"first call (JIT + run): {jit_elapsed:.6f} s")
print(f"warmed elapsed: {elapsed:.6f} s")
print(f"elastic events/s: {events / elapsed:,.0f}")
if not args.no_copy:
    for k, v in elastic_summary(r).items():
        print(f"{k}: {v:.9g}")
