import argparse
import time

from seemc_taichi.config import BackendConfig
from seemc_taichi.runtime import init_taichi
from seemc_taichi.skeleton import run_pool_demo

p = argparse.ArgumentParser()
p.add_argument("--arch", default="cpu")
p.add_argument("--precision", choices=("f32", "f64"), default=None)
p.add_argument("--n", type=int, default=100_000)
p.add_argument("--capacity", type=int, default=1_000_000)
p.add_argument("--spawn", type=float, default=0.25)
args = p.parse_args()
precision = args.precision or ("f32" if args.arch == "metal" else "f64")
if args.arch == "metal" and precision == "f64":
    p.error("Metal does not support f64 in Taichi 1.7.4 on this platform")

ti = init_taichi(BackendConfig(
    arch=args.arch, max_particles=args.capacity,
    default_fp=precision, default_ip="i32",
))
fp = ti.f32 if precision == "f32" else ti.f64

t0 = time.perf_counter()
r = run_pool_demo(ti, fp=fp, n=args.n, capacity=args.capacity, spawn_probability=args.spawn)
ti.sync()
dt = time.perf_counter() - t0
print(r)
print(f"precision: {precision}")
print(f"elapsed: {dt:.6f} s")
print(f"stored particles/s: {r['stored']/dt:,.0f}")
