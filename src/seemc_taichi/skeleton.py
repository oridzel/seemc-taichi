# v0.1 infrastructure demo retained for regression testing.
# Do not enable postponed annotations in this module: Taichi 1.7.4 kernel
# annotations must remain concrete dtype objects.

import math

from .particle_pool import ParticlePool


def build_skeleton(ti, pool: ParticlePool, fp):
    @ti.kernel
    def seed_primaries(n: ti.i32, energy_ev: fp, sin_alpha: fp, cos_alpha: fp):
        pool.n_allocated[None] = n
        pool.overflow[None] = ti.i32(0)
        for i in range(n):
            pool.x[i] = 0.0
            pool.y[i] = 0.0
            pool.z[i] = 0.0
            pool.ux[i] = sin_alpha
            pool.uy[i] = 0.0
            pool.uz[i] = cos_alpha
            pool.energy[i] = energy_ev
            pool.parent_id[i] = ti.i32(-1)
            pool.root_primary_id[i] = i
            pool.generation[i] = ti.i32(0)
            pool.alive[i] = ti.i32(1)
            pool.steps[i] = ti.i32(0)

    @ti.kernel
    def process_wave(begin: ti.i32, end: ti.i32, spawn_probability: fp, max_generation: ti.i32):
        for i in range(begin, end):
            if pool.alive[i] == ti.i32(1):
                flight = -ti.log(ti.max(ti.random(fp), fp(1.0e-15)))
                pool.x[i] += flight * pool.ux[i]
                pool.y[i] += flight * pool.uy[i]
                pool.z[i] += flight * pool.uz[i]
                pool.steps[i] += ti.i32(1)
                if pool.generation[i] < max_generation and ti.random(fp) < spawn_probability:
                    slot = ti.cast(ti.atomic_add(pool.n_allocated[None], ti.i32(1)), ti.i32)
                    if slot < pool.capacity:
                        phi = fp(2.0 * math.pi) * ti.random(fp)
                        mu = fp(2.0) * ti.random(fp) - fp(1.0)
                        s = ti.sqrt(ti.max(fp(0.0), fp(1.0) - mu * mu))
                        pool.x[slot] = pool.x[i]
                        pool.y[slot] = pool.y[i]
                        pool.z[slot] = pool.z[i]
                        pool.ux[slot] = s * ti.cos(phi)
                        pool.uy[slot] = s * ti.sin(phi)
                        pool.uz[slot] = mu
                        pool.energy[slot] = fp(0.5) * pool.energy[i]
                        pool.parent_id[slot] = i
                        pool.root_primary_id[slot] = pool.root_primary_id[i]
                        pool.generation[slot] = pool.generation[i] + ti.i32(1)
                        pool.alive[slot] = ti.i32(1)
                        pool.steps[slot] = ti.i32(0)
                    else:
                        pool.overflow[None] = ti.i32(1)
                pool.alive[i] = ti.i32(0)
    return seed_primaries, process_wave


def run_pool_demo(ti, fp, n=100_000, capacity=1_000_000, energy_ev=1000.0,
                  alpha_deg=0.0, spawn_probability=0.25, max_generation=4):
    pool = ParticlePool(ti, capacity, fp)
    seed, wave = build_skeleton(ti, pool, fp)
    a = math.radians(float(alpha_deg))
    seed(int(n), float(energy_ev), math.sin(a), math.cos(a))
    begin = 0
    while begin < pool.allocated():
        end = min(pool.allocated(), capacity)
        wave(int(begin), int(end), float(spawn_probability), int(max_generation))
        begin = end
        if int(pool.overflow[None]):
            break
    m = min(pool.allocated(), capacity)
    return {
        "allocated": pool.allocated(),
        "stored": m,
        "overflow": bool(pool.overflow[None]),
        "generation": pool.generation.to_numpy()[:m],
        "root_primary_id": pool.root_primary_id.to_numpy()[:m],
        "parent_id": pool.parent_id.to_numpy()[:m],
        "steps": pool.steps.to_numpy()[:m],
    }
