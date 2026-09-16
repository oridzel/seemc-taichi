from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .bulk import BulkPhysicsConfig
from .config import BackendConfig
from .runtime import init_taichi
from .surface import SurfacePhysicsConfig
from .trapezoid import (
    TrapezoidGeometryConfig,
    TrapezoidTransportEngine,
    load_reference_plane_tables,
)
from .trapezoid_scan import plot_scan, run_line_scan, save_scan_csv, save_scan_npz, save_trajectory_npz


def _parser():
    p = argparse.ArgumentParser(
        description="Taichi-accelerated SEEMC line scan over a raised trapezoidal feature"
    )
    p.add_argument("database")
    p.add_argument("--material", default="Si")
    p.add_argument("--arch", choices=("cpu", "metal", "cuda", "vulkan", "gpu"), default="metal")
    p.add_argument("--precision", choices=("f32", "f64"), default=None)
    p.add_argument("--energy-ev", type=float, default=1000.0)
    p.add_argument("--top-width-nm", type=float, default=50.0)
    p.add_argument("--bottom-width-nm", type=float, default=70.0)
    p.add_argument("--height-nm", type=float, default=50.0)
    p.add_argument("--scan-width-nm", type=float, default=150.0)
    p.add_argument("--pixels", type=int, default=201)
    p.add_argument("--primaries-per-pixel", type=int, default=1000)
    p.add_argument("--beam-fwhm-nm", type=float, default=2.0)
    p.add_argument("--capacity", type=int, default=500_000, help="particle slots per pixel")
    p.add_argument("--steps-per-chunk", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--bse-cutoff-ev", type=float, default=None)
    p.add_argument("--lle-max-loss-ev", type=float, default=50.0)
    p.add_argument("--trace-pixel", type=int, action="append", default=[], help="pixel index to record trajectories for; may be given multiple times")
    p.add_argument("--trace-x-nm", type=float, action="append", default=[], help="beam x position to trace; nearest scan pixel is selected; may be repeated")
    p.add_argument("--trace-primaries", type=int, default=0, help="number of root primaries to trace for each traced pixel")
    p.add_argument("--trajectory-capacity", type=int, default=200000, help="maximum stored trajectory points per traced pixel")
    p.add_argument("--output-prefix", type=Path, default=Path("trapezoid_taichi"))
    p.add_argument("--plot", action="store_true")
    return p


def main(argv=None):
    p = _parser()
    args = p.parse_args(argv)
    precision = args.precision or ("f32" if args.arch == "metal" else "f64")
    if args.arch == "metal" and precision == "f64":
        p.error("Metal requires f32 on this Taichi build")
    positive = {
        "--energy-ev": args.energy_ev,
        "--top-width-nm": args.top_width_nm,
        "--bottom-width-nm": args.bottom_width_nm,
        "--height-nm": args.height_nm,
        "--scan-width-nm": args.scan_width_nm,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            p.error(f"{name} must be finite and positive")
    if args.bottom_width_nm < args.top_width_nm:
        p.error("--bottom-width-nm must be >= --top-width-nm")
    if args.pixels < 2 or args.primaries_per_pixel < 1:
        p.error("--pixels must be >=2 and --primaries-per-pixel must be positive")
    if args.beam_fwhm_nm < 0.0:
        p.error("--beam-fwhm-nm must be non-negative")
    if args.capacity < args.primaries_per_pixel:
        p.error("--capacity must be >= --primaries-per-pixel")
    if args.steps_per_chunk < 1:
        p.error("--steps-per-chunk must be positive")
    if args.trace_primaries < 0:
        p.error("--trace-primaries must be non-negative")
    if args.trajectory_capacity < 1:
        p.error("--trajectory-capacity must be positive")
    for pix in args.trace_pixel:
        if pix < 0 or pix >= args.pixels:
            p.error("--trace-pixel values must be in [0, pixels)")
    for xpos in args.trace_x_nm:
        if not math.isfinite(xpos):
            p.error("--trace-x-nm values must be finite")

    sample, host = load_reference_plane_tables(args.database, args.material)
    bulk = BulkPhysicsConfig.from_sample(sample)
    surface = SurfacePhysicsConfig.from_sample(sample)
    cutoff = float(args.bse_cutoff_ev if args.bse_cutoff_ev is not None else getattr(sample.cfg, "bse_cutoff_ev", 50.0))
    geometry = TrapezoidGeometryConfig(
        top_width=10.0 * args.top_width_nm,
        bottom_width=10.0 * args.bottom_width_nm,
        height=10.0 * args.height_nm,
    ).validate()

    backend = BackendConfig(
        arch=args.arch,
        max_particles=args.capacity,
        max_emissions=args.capacity,
        default_fp=precision,
        default_ip="i32",
        random_seed=args.seed,
    )
    ti = init_taichi(backend)
    fp = ti.f32 if precision == "f32" else ti.f64
    engine = TrapezoidTransportEngine(
        ti, fp, host, bulk, surface, geometry, args.capacity,
        bse_cutoff_ev=cutoff, trajectory_capacity=args.trajectory_capacity,
    )

    # One tiny compile/warm-up at the center.  The real scan starts from a clean seed.
    engine.run_trapezoid(
        n=min(args.primaries_per_pixel, 64),
        energy_vac_ev=args.energy_ev,
        nominal_x=0.0,
        beam_fwhm=10.0 * args.beam_fwhm_nm,
        seed=args.seed,
        steps_per_chunk=min(args.steps_per_chunk, 32),
        copy_emissions=False,
    )
    ti.sync()

    # User-facing scan width is in nm; the transport engine uses Angstrom.
    x_nm = np.linspace(-0.5 * args.scan_width_nm, 0.5 * args.scan_width_nm, args.pixels)
    x = 10.0 * x_nm
    trace_pixels = set(int(v) for v in args.trace_pixel)
    for xpos in args.trace_x_nm:
        trace_pixels.add(int(np.argmin(np.abs(x_nm - float(xpos)))))
    if trace_pixels:
        print("Tracing pixels:", ", ".join(f"{i} (x={x_nm[i]:+.3f} nm)" for i in sorted(trace_pixels)))
    rows, per_primary, trajectories = run_line_scan(
        engine,
        geometry,
        energy_ev=args.energy_ev,
        x_positions_angstrom=x,
        primaries_per_pixel=args.primaries_per_pixel,
        beam_fwhm_angstrom=10.0 * args.beam_fwhm_nm,
        seed=args.seed,
        steps_per_chunk=args.steps_per_chunk,
        cutoff_ev=cutoff,
        lle_max_loss_ev=args.lle_max_loss_ev,
        progress=True,
        trajectory_pixels=trace_pixels,
        trace_n_primaries=args.trace_primaries,
    )

    prefix = args.output_prefix
    csv_path = prefix.parent / f"{prefix.name}.csv"
    npz_path = prefix.parent / f"{prefix.name}.npz"
    metadata = {
        "package": "seemc-taichi",
        "backend": args.arch,
        "precision": precision,
        "material": args.material,
        "energy_ev": args.energy_ev,
        "top_width_nm": args.top_width_nm,
        "bottom_width_nm": args.bottom_width_nm,
        "height_nm": args.height_nm,
        "scan_width_nm": args.scan_width_nm,
        "pixels": args.pixels,
        "primaries_per_pixel": args.primaries_per_pixel,
        "beam_fwhm_nm": args.beam_fwhm_nm,
        "capacity_per_pixel": args.capacity,
        "seed": args.seed,
        "bse_cutoff_ev": cutoff,
        "lle_max_loss_ev": args.lle_max_loss_ev,
        "trace_pixels": sorted(trace_pixels),
        "trace_x_nm_requested": list(args.trace_x_nm),
        "trace_primaries": args.trace_primaries,
        "trajectory_capacity": args.trajectory_capacity,
        "barrier_model": surface.barrier_model,
        "inner_potential_ev": surface.inner_potential_ev,
        "se1_definition": "generation==1 and own_inelastic_count==0",
        "geometry_convention": "vacuum negative z; top z=-height; substrate z=0; line infinite y",
    }
    save_scan_csv(csv_path, rows)
    save_scan_npz(npz_path, rows, per_primary, metadata=metadata)
    print(f"Wrote {csv_path}")
    print(f"Wrote {npz_path}")
    for pix, traj in sorted(trajectories.items()):
        traj_path = prefix.parent / f"{prefix.name}_pixel{pix:03d}_trajectories.npz"
        save_trajectory_npz(traj_path, traj, metadata=metadata)
        print(f"Wrote {traj_path}")

    invalid = [row for row in rows if not row["valid"]]
    if invalid:
        print(f"WARNING: {len(invalid)} pixel(s) invalid; inspect status in {csv_path}")
    if args.plot:
        png_path = prefix.parent / f"{prefix.name}.png"
        plot_scan(
            png_path,
            rows,
            title=(f"{args.material}, {args.energy_ev:g} eV, "
                   f"{args.primaries_per_pixel:,} primaries/pixel, "
                   f"{args.beam_fwhm_nm:g} nm FWHM"),
        )
        print(f"Wrote {png_path}")
    return 0 if not invalid else 2


if __name__ == "__main__":
    raise SystemExit(main())
