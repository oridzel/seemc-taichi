from __future__ import annotations


class ParticlePool:
    """Structure-of-arrays particle state for Taichi transport.

    Positions are Angstrom, directions are unit vectors, and energy is E_s in eV
    (measured from the bottom of the valence band while inside the solid).
    Integer IDs/counters are deliberately i32 for Metal portability.
    """

    def __init__(self, ti, capacity: int, fp):
        self.ti = ti
        self.capacity = int(capacity)
        self.fp = fp
        if self.capacity < 1:
            raise ValueError("capacity must be positive")

        self.x = ti.field(dtype=fp, shape=self.capacity)
        self.y = ti.field(dtype=fp, shape=self.capacity)
        self.z = ti.field(dtype=fp, shape=self.capacity)
        self.ux = ti.field(dtype=fp, shape=self.capacity)
        self.uy = ti.field(dtype=fp, shape=self.capacity)
        self.uz = ti.field(dtype=fp, shape=self.capacity)
        self.energy = ti.field(dtype=fp, shape=self.capacity)

        self.parent_id = ti.field(dtype=ti.i32, shape=self.capacity)
        self.root_primary_id = ti.field(dtype=ti.i32, shape=self.capacity)
        self.generation = ti.field(dtype=ti.i32, shape=self.capacity)
        self.alive = ti.field(dtype=ti.i32, shape=self.capacity)
        self.steps = ti.field(dtype=ti.i32, shape=self.capacity)
        # Number of inelastic collisions experienced by this electron itself.
        # Used by the trapezoid imaging backend for the current SE1/SE2 rule.
        self.inelastic_count = ti.field(dtype=ti.i32, shape=self.capacity)

        self.n_allocated = ti.field(dtype=ti.i32, shape=())
        self.overflow = ti.field(dtype=ti.i32, shape=())

    def reset(self):
        self.n_allocated[None] = 0
        self.overflow[None] = 0

    def allocated(self) -> int:
        return int(self.n_allocated[None])
