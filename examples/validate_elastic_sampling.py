"""Compare Taichi elastic statistics with a NumPy implementation of the same tables."""

import argparse
import math
import numpy as np

from seemc_taichi.config import BackendConfig
from seemc_taichi.elastic import ElasticPhysicsConfig, load_reference_host_tables, run_elastic_transport
from seemc_taichi.runtime import init_taichi
from seemc_taichi.validation import histogram_distance


def invert_cdf(cdf, x, u):
    j = np.searchsorted(cdf, u, side="right") - 1
    j = int(np.clip(j, 0, len(x) - 2))
    c0, c1 = cdf[j], cdf[j + 1]
    if c1 <= c0:
        return float(x[j])
    t = (u - c0) / (c1 - c0)
    return float(x[j] + t * (x[j + 1] - x[j]))


def numpy_one_collision(sample, host, n, energy, seed):
    rng = np.random.default_rng(seed)
    E = float(energy)
    inv_emfp = sample.inverse_mfps(E)[0]
    flights = -np.log(np.maximum(rng.random(n), 1e-15)) / inv_emfp
    theta = np.empty(n)
    for k in range(n):
        theta[k] = sample.sample_elastic_theta(E, rng)
    return flights, theta


p = argparse.ArgumentParser()
p.add_argument("database")
p.add_argument("--material", default="Si")
p.add_argument("--arch", choices=("cpu", "metal"), default="cpu")
p.add_argument("--precision", choices=("f32", "f64"), default=None)
p.add_argument("--energy", type=float, default=1000.0)
p.add_argument("--n", type=int, default=200_000)
p.add_argument("--seed", type=int, default=20260819)
args = p.parse_args()
precision = args.precision or ("f32" if args.arch == "metal" else "f64")
if args.arch == "metal" and precision == "f64":
    p.error("Metal requires f32")

sample, host = load_reference_host_tables(args.database, args.material)
physics = ElasticPhysicsConfig.from_sample(sample)

# CPU reference distribution from the actual Sample methods.
fl_ref, th_ref = numpy_one_collision(sample, host, args.n, args.energy, args.seed + 1)

# Device distribution: one collision from +z, so theta = arccos(final uz).
ti = init_taichi(BackendConfig(
    arch=args.arch, default_fp=precision, default_ip="i32", random_seed=args.seed,
    max_particles=args.n,
))
fp = ti.f32 if precision == "f32" else ti.f64
r = run_elastic_transport(
    ti, fp, host, physics=physics, n=args.n, energy_ev=args.energy,
    alpha_deg=0.0, collisions=1, copy_state=True,
)
th_dev = np.arccos(np.clip(r["uz"].astype(float), -1.0, 1.0))
fl_dev = np.sqrt(r["x"].astype(float)**2 + r["y"].astype(float)**2 + r["z"].astype(float)**2)

expected_inv_emfp = sample.inverse_mfps(args.energy)[0]
expected_mfp = 1.0 / expected_inv_emfp if expected_inv_emfp > 0.0 else float("inf")
print(f"expected EMFP: {expected_mfp:.9g} A")
print(f"mean flight reference: {fl_ref.mean():.9g} A")
print(f"mean flight device:    {fl_dev.mean():.9g} A")
print(f"flight relative error: {(fl_dev.mean()-expected_mfp)/expected_mfp:.4%}")
print(f"mean theta reference: {np.degrees(th_ref.mean()):.9g} deg")
print(f"mean theta device:    {np.degrees(th_dev.mean()):.9g} deg")
print(f"theta histogram TV:   {histogram_distance(th_ref, th_dev, bins=180, value_range=(0, math.pi)):.6g}")
print(f"direction norm max error: {np.max(np.abs(np.sqrt(r['ux']**2+r['uy']**2+r['uz']**2)-1)):.6g}")
