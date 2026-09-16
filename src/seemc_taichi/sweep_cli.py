from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from .config import BackendConfig
from .plane import BulkPhysicsConfig, PlaneTransportEngine, load_reference_plane_tables
from .runtime import init_taichi
from .surface import SurfacePhysicsConfig
from .sweep import (
    build_sweep_row,
    csv_row_is_valid,
    emission_filename,
    load_csv_rows,
    point_key,
    write_csv_rows,
)

DEFAULT_ENERGIES = (100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch planar SEEMC-Taichi energy/angle yield sweep"
    )
    p.add_argument("database")
    p.add_argument("--material", default="Si")
    p.add_argument("--arch", choices=("cpu", "metal", "cuda", "vulkan", "gpu"), default="cpu")
    p.add_argument("--precision", choices=("f32", "f64"), default=None)
    p.add_argument(
        "--energies", type=float, nargs="+", default=list(DEFAULT_ENERGIES),
        help="incident vacuum energies [eV] (default: 100 200 500 1000 2000 5000 10000)",
    )
    p.add_argument(
        "--angles", type=float, nargs="+", default=[0.0],
        help="incidence angles from surface normal [deg] (default: 0)",
    )
    # Backward-friendly singular form for quick one-angle usage.
    p.add_argument("--angle", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--capacity", type=int, default=5_000_000)
    p.add_argument("--steps-per-chunk", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--output", default=None, help="CSV output path")
    p.add_argument(
        "--save-emissions", action="store_true",
        help="save emitted energy/directions/lineage to one compressed NPZ per point",
    )
    p.add_argument(
        "--emissions-dir", default=None,
        help="directory for --save-emissions (default: <output stem>_emissions)",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="keep the CSV, skip valid completed points, and retry invalid points",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="overwrite an existing output CSV instead of refusing to start",
    )
    return p


def _default_output(material: str, arch: str, precision: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in material)
    return f"{safe}_plane_yield_sweep_{arch}_{precision}.csv"


def _save_emissions(path: Path, result: dict, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        material=np.asarray(row["material"]),
        energy_eV=np.asarray(float(row["energy_eV"])),
        angle_deg=np.asarray(float(row["angle_deg"])),
        n_primaries=np.asarray(int(row["n_primaries"])),
        bse_cutoff_eV=np.asarray(float(row["bse_cutoff_eV"])),
        valid=np.asarray(bool(row["valid"])),
        emission_energy_ev=result["emission_energy_ev"],
        emission_ux=result["emission_ux"],
        emission_uy=result["emission_uy"],
        emission_uz=result["emission_uz"],
        emission_electron_id=result["emission_electron_id"],
        emission_parent_id=result["emission_parent_id"],
        emission_root_primary_id=result["emission_root_primary_id"],
        emission_generation=result["emission_generation"],
        emission_is_cascade=result["emission_is_cascade"],
        emission_mechanism=result["emission_mechanism"],
    )


def main(argv=None) -> int:
    p = _parser()
    args = p.parse_args(argv)

    precision = args.precision or ("f32" if args.arch == "metal" else "f64")
    if args.arch == "metal" and precision == "f64":
        p.error("Metal requires f32 on this Taichi build")
    if args.n < 1:
        p.error("--n must be positive")
    if args.capacity < args.n:
        p.error("--capacity must be >= --n")
    if args.steps_per_chunk < 1:
        p.error("--steps-per-chunk must be positive")

    energies = [float(x) for x in args.energies]
    angles = [float(args.angle)] if args.angle is not None else [float(x) for x in args.angles]
    if not energies or any((not math.isfinite(x) or x <= 0.0) for x in energies):
        p.error("all energies must be finite and > 0")
    if not angles or any((not math.isfinite(x) or not 0.0 <= x < 90.0) for x in angles):
        p.error("all angles must lie in [0, 90)")

    output = Path(args.output or _default_output(args.material, args.arch, precision))
    if output.exists() and not args.resume and not args.overwrite:
        p.error(f"{output} already exists; use --resume or --overwrite")

    existing = load_csv_rows(output) if (args.resume and output.exists()) else {}
    if existing:
        exemplar = next(iter(existing.values()))
        expected = {
            "material": str(args.material),
            "arch": str(args.arch),
            "precision": str(precision),
            "n_primaries": str(int(args.n)),
        }
        for name, want in expected.items():
            got = str(exemplar.get(name, ""))
            if got != want:
                p.error(
                    f"cannot --resume {output}: existing {name}={got!r}, "
                    f"requested {name}={want!r}; choose a new --output or --overwrite"
                )
    rows = dict(existing)

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

    points = [(energy, angle) for angle in angles for energy in energies]
    pending = []
    for energy, angle in points:
        old = rows.get(point_key(energy, angle))
        if args.resume and old is not None and csv_row_is_valid(old):
            print(f"SKIP {energy:g} eV @ {angle:g} deg (valid row already in {output})")
        else:
            pending.append((energy, angle))

    if not pending:
        print(f"Nothing to run; all requested points are already valid in {output}")
        return 0

    # Compile on a small real cascade at the first pending point.  This is not timed.
    warm_energy, warm_angle = pending[0]
    warm_n = min(args.n, 128)
    engine.run_plane(
        n=warm_n,
        energy_vac_ev=warm_energy,
        alpha_deg=warm_angle,
        steps_per_chunk=min(args.steps_per_chunk, 32),
        copy_emissions=False,
    )
    ti.sync()

    emissions_dir = None
    if args.save_emissions:
        emissions_dir = Path(args.emissions_dir) if args.emissions_dir else output.with_suffix("").with_name(output.stem + "_emissions")
        emissions_dir.mkdir(parents=True, exist_ok=True)

    kind = "metal" if sample.is_metal else "nonconductor"
    print(f"material: {args.material} ({kind})")
    print(f"arch / precision: {args.arch} / {precision}")
    print(f"barrier: {surface.barrier_model}; Ui={surface.inner_potential_ev:.9g} eV")
    print(f"points to run: {len(pending)} / {len(points)}")
    print(f"output: {output}")
    if emissions_dir is not None:
        print(f"emissions: {emissions_dir}")
    print()

    for j, (energy, angle) in enumerate(pending, start=1):
        print(f"[{j}/{len(pending)}] {energy:g} eV @ {angle:g} deg")
        start = time.perf_counter()
        result = engine.run_plane(
            n=args.n,
            energy_vac_ev=energy,
            alpha_deg=angle,
            steps_per_chunk=args.steps_per_chunk,
            copy_emissions=args.save_emissions,
        )
        ti.sync()
        elapsed = time.perf_counter() - start

        row = build_sweep_row(
            result,
            material=args.material,
            material_kind=kind,
            arch=args.arch,
            precision=precision,
            barrier_model=surface.barrier_model,
            inner_potential_ev=surface.inner_potential_ev,
            bse_cutoff_ev=cutoff,
            capacity=args.capacity,
            steps_per_chunk=args.steps_per_chunk,
            seed=args.seed,
            elapsed_s=elapsed,
        )
        rows[point_key(energy, angle)] = row
        write_csv_rows(output, rows)

        if args.save_emissions:
            epath = emissions_dir / emission_filename(args.material, energy, angle)
            _save_emissions(epath, result, row)

        validity = "VALID" if row["valid"] else f"INVALID ({row['status']})"
        print(
            f"  {validity}: TEY={row['raw_tey']:.6f} "
            f"SEY={row['raw_sey']:.6f} BSEY={row['raw_bsey']:.6f} "
            f"allocated={row['allocated']:,} elapsed={elapsed:.3f}s "
            f"primaries/s={row['primaries_per_s']:,.0f}"
        )
        print(f"  wrote {output}")

    n_valid = sum(csv_row_is_valid(row) for row in rows.values())
    print()
    print(f"Sweep complete: {n_valid} valid point(s) in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
