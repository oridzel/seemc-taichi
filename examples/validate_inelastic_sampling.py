import argparse
from collections import defaultdict
import math
import numpy as np

from seemc_taichi.bulk import (
    BulkPhysicsConfig,
    BulkTransportEngine,
    load_reference_bulk_tables,
    summarize_inelastic_samples,
)
from seemc_taichi.config import BackendConfig
from seemc_taichi.runtime import init_taichi

H2EV = 27.21184


def tv_distance(a, b, bins):
    ha, _ = np.histogram(a, bins=bins, density=False)
    hb, _ = np.histogram(b, bins=bins, density=False)
    pa = ha / max(ha.sum(), 1)
    pb = hb / max(hb.sum(), 1)
    return 0.5 * float(np.abs(pa - pb).sum())


def reference_samples(sample, E_s, n, seed):
    rng = np.random.default_rng(seed)
    diag = defaultdict(int)
    chs, mechs, omega, qs, esec = [], [], [], [], []
    for _ in range(int(n)):
        ch = sample.choose_channel(float(E_s), rng)
        if ch is None:
            continue
        w = sample.sample_energy_loss(ch, float(E_s), rng, diag)
        if w is None:
            continue
        qres = sample.sample_q(ch, float(E_s), w, rng, diag)
        if qres is None:
            continue
        q, k, kp = qres
        mech = ch
        if sample.is_metal and sample.cfg.se_channel_rule == "mao":
            qm, qp = sample.mao_q_boundaries(w)
            mech = "se" if qm <= q <= qp else "pl"

        sec = np.nan
        if not sample.is_metal:
            eps = sample.sample_semiconductor_target_energy(w, rng)
            if eps is not None:
                sec = max(eps + w, sample.e_cbm)
        elif mech == "se":
            target = sample.sample_target_electron(w, q, rng, diag)
            if target is not None:
                r, kz = target
                sec = 0.5 * (r * r + (kz + q) ** 2) * H2EV
            elif sample.cfg.on_pauli_block == "fallback":
                sec = sample.sample_plasmon_target_energy(w, rng) + w
        else:
            sec = sample.sample_plasmon_target_energy(w, rng) + w

        chs.append(1 if ch == "se" else 2)
        mechs.append(3 if not sample.is_metal else (1 if mech == "se" else 2))
        omega.append(w)
        qs.append(q)
        esec.append(sec)

    return {
        "sampled_channel": np.asarray(chs, np.int32),
        "mechanism": np.asarray(mechs, np.int32),
        "omega_ev": np.asarray(omega, float),
        "q_a0inv": np.asarray(qs, float),
        "secondary_energy_ev": np.asarray(esec, float),
        "diagnostics": dict(diag),
    }


p = argparse.ArgumentParser(description="Validate v0.3 Taichi inelastic event sampling")
p.add_argument("database")
p.add_argument("--material", default="Si")
p.add_argument("--arch", choices=("cpu", "metal", "cuda", "vulkan", "gpu"), default="cpu")
p.add_argument("--precision", choices=("f32", "f64"), default=None)
p.add_argument("--energy", type=float, default=1000.0)
p.add_argument("--n", type=int, default=20_000)
p.add_argument("--seed", type=int, default=20260915)
args = p.parse_args()

precision = args.precision or ("f32" if args.arch == "metal" else "f64")
if args.arch == "metal" and precision == "f64":
    p.error("Metal requires f32 on this Taichi build")

sample, host = load_reference_bulk_tables(args.database, args.material)
physics = BulkPhysicsConfig.from_sample(sample)

ref = reference_samples(sample, args.energy, args.n, args.seed + 1)

cfg = BackendConfig(
    arch=args.arch,
    max_particles=args.n,
    default_fp=precision,
    default_ip="i32",
    random_seed=args.seed,
)
ti = init_taichi(cfg)
fp = ti.f32 if precision == "f32" else ti.f64
engine = BulkTransportEngine(ti, fp, host, physics, capacity=args.n)
dev = engine.sample_inelastic_once(n=args.n, energy_ev=args.energy, copy=True)

mask = np.asarray(dev["valid"], bool)
dev_omega = np.asarray(dev["omega_ev"])[mask]
dev_q = np.asarray(dev["q_a0inv"])[mask]
dev_ch = np.asarray(dev["sampled_channel"])[mask]
dev_mech = np.asarray(dev["mechanism"])[mask]
dev_sec_mask = np.asarray(dev["secondary_valid"])[mask].astype(bool)
dev_sec = np.asarray(dev["secondary_energy_ev"])[mask][dev_sec_mask]

ref_sec = ref["secondary_energy_ev"]
ref_sec = ref_sec[np.isfinite(ref_sec)]

print(f"material: {args.material} ({'metal' if sample.is_metal else 'nonconductor'})")
print(f"energy E_s: {args.energy:g} eV")
print(f"arch / precision: {args.arch} / {precision}")
print(f"requested samples: {args.n:,}")
print(f"reference valid: {len(ref['omega_ev']):,}")
print(f"device valid:    {len(dev_omega):,}")
print()

if len(ref["omega_ev"]) and len(dev_omega):
    print(f"sampled SE-channel fraction ref/dev: {np.mean(ref['sampled_channel']==1):.6f} / {np.mean(dev_ch==1):.6f}")
    print(f"binary-mechanism fraction ref/dev:   {np.mean(ref['mechanism']==1):.6f} / {np.mean(dev_mech==1):.6f}")
    print(f"mean omega ref/dev: {np.mean(ref['omega_ev']):.6f} / {np.mean(dev_omega):.6f} eV")
    print(f"mean q ref/dev:     {np.mean(ref['q_a0inv']):.6f} / {np.mean(dev_q):.6f} a0^-1")

    omin = min(float(np.min(ref['omega_ev'])), float(np.min(dev_omega)))
    omax = max(float(np.max(ref['omega_ev'])), float(np.max(dev_omega)))
    qmin = min(float(np.min(ref['q_a0inv'])), float(np.min(dev_q)))
    qmax = max(float(np.max(ref['q_a0inv'])), float(np.max(dev_q)))
    obins = np.linspace(omin, omax, 81) if omax > omin else np.array([omin, omin + 1.0])
    qbins = np.linspace(qmin, qmax, 81) if qmax > qmin else np.array([qmin, qmin + 1.0])
    print(f"omega histogram TV: {tv_distance(ref['omega_ev'], dev_omega, obins):.6f}")
    print(f"q histogram TV:     {tv_distance(ref['q_a0inv'], dev_q, qbins):.6f}")

if len(ref_sec) and len(dev_sec):
    print(f"mean secondary E ref/dev: {np.mean(ref_sec):.6f} / {np.mean(dev_sec):.6f} eV")
    smin = min(float(np.min(ref_sec)), float(np.min(dev_sec)))
    smax = max(float(np.max(ref_sec)), float(np.max(dev_sec)))
    sbins = np.linspace(smin, smax, 81) if smax > smin else np.array([smin, smin + 1.0])
    print(f"secondary-E histogram TV: {tv_distance(ref_sec, dev_sec, sbins):.6f}")

print()
print("device diagnostics:", dev["diagnostics"])
print("device summary:", summarize_inelastic_samples(dev))
