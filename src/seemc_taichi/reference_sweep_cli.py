from __future__ import annotations

import argparse
from dataclasses import replace
import math
import time
from pathlib import Path

import numpy as np

from .reference_sweep import (
    csv_row_is_valid,
    deterministic_case_seed,
    emission_arrays,
    load_csv_rows,
    plane_directions,
    point_key,
    reference_emission_filename,
    save_reference_emissions,
    summarize_reference_emissions,
    write_csv_rows,
)

DEFAULT_ENERGIES = (100.0, 500.0, 1000.0, 5000.0)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export Python SEEMC planar emissions for Taichi parity comparisons"
    )
    p.add_argument("database")
    p.add_argument("--material", default="Si")
    p.add_argument(
        "--energies", type=float, nargs="+", default=list(DEFAULT_ENERGIES),
        help="incident vacuum energies [eV] (default: 100 500 1000 5000)",
    )
    p.add_argument("--angles", type=float, nargs="+", default=[0.0])
    p.add_argument("--angle", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--n", type=int, default=5000, help="primaries per energy/angle point")
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument(
        "--workers", type=int, default=1,
        help="reference SEEMC workers; >1 uses its multiprocessing path",
    )
    p.add_argument(
        "--output-dir", default="reference_emissions",
        help="directory for reference emission NPZ files",
    )
    p.add_argument(
        "--output", default=None,
        help="summary CSV path (default: <output-dir>/reference_yields.csv)",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--progress", action="store_true",
        help="show seemc-imaging reference simulation progress",
    )
    return p


def main(argv=None) -> int:
    p = _parser()
    args = p.parse_args(argv)

    if args.n < 1:
        p.error("--n must be positive")
    if args.workers < 1:
        p.error("--workers must be positive")

    energies = [float(v) for v in args.energies]
    angles = [float(args.angle)] if args.angle is not None else [float(v) for v in args.angles]
    if not energies or any((not math.isfinite(v) or v <= 0.0) for v in energies):
        p.error("all energies must be finite and > 0")
    if not angles or any((not math.isfinite(v) or not 0.0 <= v < 90.0) for v in angles):
        p.error("all angles must lie in [0, 90)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else out_dir / "reference_yields.csv"
    if output.exists() and not args.resume and not args.overwrite:
        p.error(f"{output} already exists; use --resume or --overwrite")

    try:
        from seemc_imaging.geometry import Plane
        from seemc_imaging.transport import MCConfig, Sample, SEEMC
    except ImportError as exc:
        raise RuntimeError(
            "reference_emission_sweep requires seemc-imaging; install with "
            "python3 -m pip install -e '.[reference]'"
        ) from exc

    base_cfg = MCConfig()
    base_sample = Sample(args.material, db_path=str(args.database), config=base_cfg)
    # Preserve every material/config convention used by the reference sample,
    # but force event-level emissions on.
    try:
        ref_cfg = replace(base_sample.cfg, collect_spectra=True)
    except TypeError:
        ref_cfg = MCConfig(**{**base_sample.cfg.__dict__, "collect_spectra": True})
    ref_cfg.validate()
    cutoff = float(getattr(ref_cfg, "bse_cutoff_ev", 50.0))
    kind = "metal" if base_sample.is_metal else "nonconductor"
    barrier_model = str(getattr(ref_cfg, "barrier_model", "abrupt"))
    ui = float(base_sample.Ui)

    existing = load_csv_rows(output) if (args.resume and output.exists()) else {}
    if existing:
        exemplar = next(iter(existing.values()))
        expected = {
            "material": str(args.material),
            "n_primaries": str(int(args.n)),
        }
        for name, want in expected.items():
            got = str(exemplar.get(name, ""))
            if got != want:
                p.error(
                    f"cannot --resume {output}: existing {name}={got!r}, "
                    f"requested {name}={want!r}"
                )
    rows = dict(existing)

    points = [(energy, angle) for angle in angles for energy in energies]
    pending = []
    for energy, angle in points:
        key = point_key(energy, angle)
        npz_path = out_dir / reference_emission_filename(args.material, energy, angle)
        old = rows.get(key)
        if args.resume and old is not None and csv_row_is_valid(old) and npz_path.exists():
            print(f"SKIP {energy:g} eV @ {angle:g} deg")
        else:
            pending.append((energy, angle))

    if not pending:
        print(f"Nothing to run; all requested reference points are complete in {out_dir}")
        return 0

    print(f"material: {args.material} ({kind})")
    print(f"reference backend: Python seemc-imaging")
    print(f"barrier: {barrier_model}; Ui={ui:.9g} eV")
    print(f"primaries/point: {args.n:,}; workers: {args.workers}")
    print(f"points to run: {len(pending)} / {len(points)}")
    print(f"NPZ output: {out_dir}")
    print(f"summary CSV: {output}")
    print()

    for j, (energy, angle) in enumerate(pending, start=1):
        print(f"[{j}/{len(pending)}] {energy:g} eV @ {angle:g} deg")
        case_seed = deterministic_case_seed(args.seed, angle, energy)
        vacuum, outward = plane_directions(angle)
        start = time.perf_counter()
        try:
            model = SEEMC(
                [float(energy)],
                args.material,
                math.radians(float(angle)),
                int(args.n),
                db_path=str(args.database),
                config=ref_cfg,
                seed=int(case_seed),
                history=False,
                geometry=Plane(),
                vacuum_direction=vacuum,
                surface_normal=outward,
            ).run_simulation(
                use_parallel=int(args.workers) > 1,
                workers=int(args.workers),
                progress=bool(args.progress),
                verbose=False,
            )
            elapsed = time.perf_counter() - start
            emissions = model.emissions[0]
            arrays = emission_arrays(emissions, args.n)
            summary = summarize_reference_emissions(
                arrays, n_primaries=args.n, cutoff_ev=cutoff
            )

            # Cross-check exported raw emissions against the reference model's
            # own aggregate yields. Fail rather than writing inconsistent data.
            model_sey = float(model.sey_50ev[0])
            model_bsey = float(model.bse_50ev[0])
            model_tey = model_sey + model_bsey
            for name, raw, expected_value in (
                ("TEY", summary["tey"], model_tey),
                ("SEY", summary["sey"], model_sey),
                ("BSEY", summary["bsey"], model_bsey),
            ):
                if not math.isclose(float(raw), expected_value, rel_tol=0.0, abs_tol=1.0e-12):
                    raise RuntimeError(
                        f"exported {name}={raw:.12g} disagrees with reference "
                        f"model {expected_value:.12g}"
                    )

            npz_path = out_dir / reference_emission_filename(args.material, energy, angle)
            save_reference_emissions(
                npz_path,
                arrays,
                material=args.material,
                energy_ev=energy,
                angle_deg=angle,
                n_primaries=args.n,
                cutoff_ev=cutoff,
                seed=args.seed,
                case_seed=case_seed,
                valid=True,
            )

            row = {
                "material": args.material,
                "material_kind": kind,
                "barrier_model": barrier_model,
                "inner_potential_eV": ui,
                "bse_cutoff_eV": cutoff,
                "energy_eV": energy,
                "angle_deg": angle,
                "n_primaries": int(args.n),
                "seed": int(args.seed),
                "case_seed": int(case_seed),
                "workers": int(args.workers),
                "valid": True,
                "status": "ok",
                **summary,
                "elapsed_s": elapsed,
                "primaries_per_s": args.n / elapsed if elapsed > 0 else math.nan,
            }
            rows[point_key(energy, angle)] = row
            write_csv_rows(output, rows)
            print(
                f"  VALID: TEY={summary['tey']:.6f} SEY={summary['sey']:.6f} "
                f"BSEY={summary['bsey']:.6f} emitted={summary['n_emitted']:,} "
                f"elapsed={elapsed:.3f}s"
            )
            print(f"  wrote {npz_path}")
        except Exception as exc:
            elapsed = time.perf_counter() - start
            row = {
                "material": args.material,
                "material_kind": kind,
                "barrier_model": barrier_model,
                "inner_potential_eV": ui,
                "bse_cutoff_eV": cutoff,
                "energy_eV": energy,
                "angle_deg": angle,
                "n_primaries": int(args.n),
                "seed": int(args.seed),
                "case_seed": int(case_seed),
                "workers": int(args.workers),
                "valid": False,
                "status": f"error:{type(exc).__name__}:{exc}",
                "elapsed_s": elapsed,
                "primaries_per_s": math.nan,
            }
            rows[point_key(energy, angle)] = row
            write_csv_rows(output, rows)
            print(f"  INVALID: {type(exc).__name__}: {exc}")
            print(f"  wrote failure row to {output}")

    valid_count = sum(csv_row_is_valid(row) for row in rows.values())
    print()
    print(f"Reference sweep complete: {valid_count} valid point(s) in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
