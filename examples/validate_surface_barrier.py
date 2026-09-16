import argparse
import math
import numpy as np

from seemc_taichi.bulk import load_reference_bulk_tables
from seemc_taichi.config import BackendConfig
from seemc_taichi.runtime import init_taichi
from seemc_taichi.surface import (
    SurfaceBarrierEngine,
    SurfacePhysicsConfig,
    analytic_isotropic_escape_probability,
    incoming_barrier_transmission_host,
)

p = argparse.ArgumentParser(description="Validate v0.4 planar surface-barrier primitives")
p.add_argument("database")
p.add_argument("--material", default="Si")
p.add_argument("--arch", choices=("cpu", "metal", "cuda", "vulkan", "gpu"), default="cpu")
p.add_argument("--precision", choices=("f32", "f64"), default=None)
p.add_argument("--energy-solid", type=float, default=100.0,
               help="solid-state E_s for outgoing isotropic escape test")
p.add_argument("--energy-vac", type=float, default=1000.0,
               help="vacuum primary energy for incoming barrier test")
p.add_argument("--angle", type=float, default=0.0)
p.add_argument("--n", type=int, default=200_000)
p.add_argument("--seed", type=int, default=20260915)
args = p.parse_args()

precision = args.precision or ("f32" if args.arch == "metal" else "f64")
if args.arch == "metal" and precision == "f64":
    p.error("Metal requires f32 on this Taichi build")

sample, _ = load_reference_bulk_tables(args.database, args.material)
physics = SurfacePhysicsConfig.from_sample(sample)

cfg = BackendConfig(
    arch=args.arch,
    max_particles=args.n,
    default_fp=precision,
    default_ip="i32",
    random_seed=args.seed,
)
ti = init_taichi(cfg)
fp = ti.f32 if precision == "f32" else ti.f64
engine = SurfaceBarrierEngine(ti, fp, physics, args.n)

out = engine.sample_outgoing_isotropic(args.n, args.energy_solid, copy=True)
escaped = np.asarray(out["escaped"], bool)
mc = float(escaped.mean())
analytic = analytic_isotropic_escape_probability(args.energy_solid, physics)
sigma = math.sqrt(max(mc * (1.0 - mc), 0.0) / args.n)

norm = np.sqrt(out["ux"]**2 + out["uy"]**2 + out["uz"]**2)
print(f"material: {args.material} ({'metal' if sample.is_metal else 'nonconductor'})")
print(f"arch / precision: {args.arch} / {precision}")
print(f"barrier model: {physics.barrier_model}")
print(f"Ui: {physics.inner_potential_ev:.9g} eV")
print()
print("outgoing isotropic surface test")
print(f"  E_s: {args.energy_solid:g} eV")
print(f"  MC escape probability:       {mc:.8f}")
print(f"  analytic escape probability: {analytic:.8f}")
print(f"  difference / MC sigma:       {(mc-analytic)/sigma if sigma > 0 else float('nan'):.3f}")
print(f"  direction norm max error:    {np.max(np.abs(norm-1.0)):.6g}")
print(f"  diagnostics: {out['diagnostics']}")

inc = engine.sample_incoming(args.n, args.energy_vac, args.angle, copy=True)
reflected = np.asarray(inc["reflected"], bool)
mc_R = float(reflected.mean())
mu = math.cos(math.radians(args.angle))
T_host = incoming_barrier_transmission_host(args.energy_vac * mu * mu, physics)
R_host = 1.0 - T_host
sigR = math.sqrt(max(mc_R * (1.0 - mc_R), 0.0) / args.n)

print()
print("incoming primary barrier test")
print(f"  E_vac: {args.energy_vac:g} eV")
print(f"  angle: {args.angle:g} deg")
print(f"  MC reflection probability:   {mc_R:.8f}")
print(f"  expected reflection:         {R_host:.8f}")
print(f"  difference / MC sigma:       {(mc_R-R_host)/sigR if sigR > 0 else float('nan'):.3f}")
print(f"  diagnostics: {inc['diagnostics']}")
