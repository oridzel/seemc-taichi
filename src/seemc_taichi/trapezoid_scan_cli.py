"""Taichi line scans and x-y rasters across one or more trapezoidal lines.

The command follows the established ``seemc-imaging`` raster workflow.  A
one-row scan may also write a combined ``.trajectories.npz`` archive for the
whole-scan animator.  A two-dimensional raster simulates every x-y pixel
independently and writes CSV, self-describing NPZ maps, and optional heatmaps;
trajectory recording is intentionally disabled for this imaging mode.
"""

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
from .trapezoid_scan import (
    CHANNELS,
    plot_scan,
    plot_raster,
    run_raster_scan,
    save_scan_csv,
    save_scan_npz,
    save_trajectory_archive_npz,
)


def _derived_outputs(csv_path: Path):
    csv_path = Path(csv_path)
    if csv_path.suffix.lower() != ".csv":
        csv_path = csv_path.with_suffix(".csv")
    stem = csv_path.with_suffix("")
    return csv_path, Path(f"{stem}.npz"), Path(f"{stem}.trajectories.npz")


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("database", type=Path)
    p.add_argument("--material", default="Cu")
    p.add_argument(
        "--arch", choices=("cpu", "metal", "cuda", "vulkan", "gpu"),
        default="metal",
    )
    p.add_argument("--precision", choices=("f32", "f64"), default=None)
    p.add_argument("--energy-ev", type=float, default=1000.0)

    p.add_argument("--top-width-nm", type=float, default=50.0)
    p.add_argument("--bottom-width-nm", type=float, default=70.0)
    p.add_argument("--height-nm", type=float, default=50.0)
    support = p.add_mutually_exclusive_group()
    support.add_argument("--membrane-thickness-nm", type=float, default=None,
                         help="finite support thickness below z=0; omitted means bulk substrate")
    support.add_argument("--suspended", action="store_true",
                   help="finite-thickness lines with vacuum below and in the gaps")
    p.add_argument("--n-lines", type=int, default=1)
    p.add_argument(
        "--pitch-nm", type=float, default=100.0,
        help="centre-to-centre pitch of adjacent lines",
    )
    p.add_argument(
        "--field-width-nm", "--scan-width-nm", dest="field_width_nm",
        type=float, default=None,
        help=(
            "total x scan width; default covers the complete line array plus "
            "40 nm of substrate on each side (--scan-width-nm is an alias)"
        ),
    )
    p.add_argument(
        "--field-height-nm", type=float, default=None,
        help=(
            "total y scan height; for ny>1 the default equals the x field "
            "width"
        ),
    )

    p.add_argument("--nx", type=int, default=None, help="number of x pixels")
    p.add_argument("--ny", type=int, default=1, help="number of y pixels")
    p.add_argument(
        "--pixels", type=int, default=None,
        help="legacy alias for --nx (default 101)",
    )
    p.add_argument(
        "--trajectories", "--primaries-per-pixel", dest="trajectories",
        type=int, default=100,
        help="Monte Carlo primaries per scan pixel",
    )
    p.add_argument(
        "--beam-fwhm-nm", type=float, default=0.0,
        help="Gaussian beam FWHM; zero uses a point beam",
    )
    p.add_argument("--no-per-primary-counts", action="store_true",
                   help="retain yields, SEMs and total counts, without large per-primary arrays")
    p.add_argument("--seed", type=int, default=20260811)

    # Accepted for command compatibility. Taichi parallelizes each pixel on
    # the selected device; pixels reuse one persistent engine sequentially.
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--workers", type=int)

    p.add_argument(
        "--record-primaries-per-pixel", type=int, default=10,
        help=(
            "primaries retained per pixel in the trajectory archive "
            "(default 10; all primaries still contribute to yields)"
        ),
    )
    p.add_argument(
        "--record-all-trajectories", action="store_true",
        help="retain every simulated primary in the trajectory archive",
    )
    p.add_argument(
        "--no-record-trajectories", action="store_true",
        help="skip the animation trajectory archive",
    )
    p.add_argument(
        "--trajectory-stride", type=int, default=1,
        help="retain every Nth point of each electron track, preserving endpoints",
    )
    p.add_argument(
        "--trajectory-max-points", type=int,
        help="maximum retained points per electron after striding",
    )

    p.add_argument("--capacity", type=int, default=500_000,
                   help="particle slots available for each scan pixel")
    p.add_argument("--steps-per-chunk", type=int, default=256)
    p.add_argument(
        "--trajectory-capacity", type=int, default=200_000,
        help="maximum raw trajectory event points stored for each pixel",
    )
    p.add_argument("--bse-cutoff-ev", type=float, default=None)
    p.add_argument("--lle-max-loss-ev", type=float, default=50.0)

    # v0.7 compatibility selectors. Recording now covers every pixel by
    # default; explicitly supplying one of these selects a sparse subset.
    p.add_argument("--trace-pixel", type=int, action="append", default=[])
    p.add_argument("--trace-x-nm", type=float, action="append", default=[])
    p.add_argument("--trace-all-pixels", action="store_true")
    p.add_argument("--trace-every", type=int, default=0)
    p.add_argument("--trace-primaries", type=int, default=None)

    p.add_argument(
        "--output", type=Path, default=None,
        help=(
            "CSV output path; companion .npz and .trajectories.npz paths are "
            "derived automatically (defaults: line_scan.csv for ny=1, "
            "trapezoid_raster.csv otherwise)"
        ),
    )
    p.add_argument("--raster-output", type=Path)
    p.add_argument("--trajectory-output", type=Path)
    p.add_argument(
        "--output-prefix", type=Path,
        help="legacy alias deriving PREFIX.csv, PREFIX.npz and PREFIX.trajectories.npz",
    )
    p.add_argument("--plot", action="store_true")
    p.add_argument(
        "--plot-channels",
        default=(
            "cascade_all,primary_all,tey,se1,se2,lle_primary,"
            "non_lle_primary,sey_50ev,bsey_50ev"
        ),
        help="comma-separated channels in the --plot overview",
    )
    p.add_argument("--plot-cmap", default="gray", help="matplotlib heatmap colormap")
    return p


def _validate_args(p, args):
    precision = args.precision or ("f32" if args.arch == "metal" else "f64")
    if args.arch == "metal" and precision == "f64":
        p.error("Metal requires f32 on this Taichi build")
    for name, value in {
        "--energy-ev": args.energy_ev,
        "--top-width-nm": args.top_width_nm,
        "--bottom-width-nm": args.bottom_width_nm,
        "--height-nm": args.height_nm,
        "--pitch-nm": args.pitch_nm,
    }.items():
        if not math.isfinite(value) or value <= 0.0:
            p.error(f"{name} must be finite and positive")
    if args.bottom_width_nm < args.top_width_nm:
        p.error("--bottom-width-nm must be >= --top-width-nm")
    if args.n_lines < 1:
        p.error("--n-lines must be >=1")
    if args.n_lines > 1 and args.pitch_nm < args.bottom_width_nm:
        p.error("--pitch-nm must be >= --bottom-width-nm to avoid overlap")
    if args.nx is not None and args.pixels is not None and args.nx != args.pixels:
        p.error("--nx and legacy --pixels must match when both are supplied")
    args.nx = args.nx if args.nx is not None else args.pixels
    args.nx = 101 if args.nx is None else int(args.nx)
    # Keep the resolved legacy attribute available to downstream compatibility
    # selectors without making it the canonical name in new metadata.
    args.pixels = args.nx
    if args.nx < 2:
        p.error("--nx must be >=2")
    if args.ny < 1:
        p.error("--ny must be >=1")
    if args.trajectories < 1:
        p.error("--trajectories must be positive")
    if not math.isfinite(args.beam_fwhm_nm) or args.beam_fwhm_nm < 0.0:
        p.error("--beam-fwhm-nm must be finite and non-negative")
    if args.capacity < args.trajectories:
        p.error("--capacity must be >= --trajectories")
    if args.steps_per_chunk < 1:
        p.error("--steps-per-chunk must be positive")
    if args.trajectory_capacity < 1:
        p.error("--trajectory-capacity must be positive")
    if args.trajectory_stride < 1:
        p.error("--trajectory-stride must be positive")
    if args.trajectory_max_points is not None and args.trajectory_max_points < 2:
        p.error("--trajectory-max-points must be at least 2")
    if args.record_primaries_per_pixel < 1:
        p.error("--record-primaries-per-pixel must be positive")
    if args.record_all_trajectories and args.no_record_trajectories:
        p.error(
            "--record-all-trajectories and --no-record-trajectories are mutually exclusive"
        )
    if args.trace_primaries is not None and args.trace_primaries < 1:
        p.error("--trace-primaries must be positive")
    if args.trace_every < 0:
        p.error("--trace-every must be non-negative")
    if args.workers is not None and args.workers < 1:
        p.error("--workers must be positive")
    if args.output is not None and args.output_prefix is not None:
        p.error("use either --output or --output-prefix, not both")
    plot_channels = tuple(
        value.strip() for value in str(args.plot_channels).split(",") if value.strip()
    )
    if not plot_channels:
        p.error("--plot-channels must contain at least one channel")
    unknown_channels = [value for value in plot_channels if value not in CHANNELS]
    if unknown_channels:
        p.error("unknown --plot-channels: " + ", ".join(unknown_channels))
    args.plot_channels = plot_channels
    return precision


def main(argv=None):
    p = _parser()
    args = p.parse_args(argv)
    precision = _validate_args(p, args)
    if args.membrane_thickness_nm is not None and (
        not math.isfinite(args.membrane_thickness_nm) or args.membrane_thickness_nm <= 0
    ):
        p.error("--membrane-thickness-nm must be finite and positive")

    geometry = TrapezoidGeometryConfig(
        top_width=10.0 * args.top_width_nm,
        bottom_width=10.0 * args.bottom_width_nm,
        height=10.0 * args.height_nm,
        n_lines=args.n_lines,
        pitch=10.0 * args.pitch_nm,
        suspended=args.suspended,
        membrane_thickness=(None if args.membrane_thickness_nm is None
                            else 10.0 * args.membrane_thickness_nm),
    ).validate()
    array_span_nm = geometry.span / 10.0
    field_width_nm = (
        float(args.field_width_nm)
        if args.field_width_nm is not None
        else array_span_nm + 80.0
    )
    if not math.isfinite(field_width_nm) or field_width_nm <= 0.0:
        p.error("--field-width-nm must be finite and positive")
    if field_width_nm < array_span_nm:
        p.error(
            f"--field-width-nm={field_width_nm:g} does not cover the full "
            f"line-array span of {array_span_nm:g} nm"
        )
    requested_field_height_nm = (
        None if args.field_height_nm is None else float(args.field_height_nm)
    )
    if requested_field_height_nm is not None and (
        not math.isfinite(requested_field_height_nm)
        or requested_field_height_nm <= 0.0
    ):
        p.error("--field-height-nm must be finite and positive")
    field_height_nm = (
        0.0
        if args.ny == 1
        else (
            requested_field_height_nm
            if requested_field_height_nm is not None
            else field_width_nm
        )
    )
    x_nm = np.linspace(-0.5 * field_width_nm, 0.5 * field_width_nm, args.nx)
    y_nm = (
        np.asarray([0.0])
        if args.ny == 1
        else np.linspace(-0.5 * field_height_nm, 0.5 * field_height_nm, args.ny)
    )

    if args.ny > 1 and (
        args.record_all_trajectories
        or args.trace_pixel
        or args.trace_x_nm
        or args.trace_all_pixels
        or args.trace_every
        or args.trace_primaries is not None
        or args.trajectory_output is not None
    ):
        p.error(
            "trajectory recording and trace selectors require --ny 1; "
            "use --no-record-trajectories for a 2D image"
        )
    record_trajectories = args.ny == 1 and not args.no_record_trajectories
    record_n = (
        args.trajectories
        if args.record_all_trajectories
        else min(args.record_primaries_per_pixel, args.trajectories)
    )
    if args.trace_primaries is not None:
        record_n = min(int(args.trace_primaries), args.trajectories)

    sparse_requested = bool(
        args.trace_pixel or args.trace_x_nm or args.trace_every
    ) and not args.trace_all_pixels
    if record_trajectories and sparse_requested:
        trace_pixels = set(int(value) for value in args.trace_pixel)
        if args.trace_every:
            trace_pixels.update(range(0, args.nx, args.trace_every))
        for value in args.trace_x_nm:
            if not math.isfinite(value):
                p.error("--trace-x-nm values must be finite")
            trace_pixels.add(int(np.argmin(np.abs(x_nm - float(value)))))
    elif record_trajectories:
        trace_pixels = set(range(args.nx))
    else:
        trace_pixels = set()
    if any(value < 0 or value >= args.nx for value in trace_pixels):
        p.error("--trace-pixel values must be in [0, nx)")

    sample, host = load_reference_plane_tables(args.database, args.material)
    bulk = BulkPhysicsConfig.from_sample(sample)
    surface = SurfacePhysicsConfig.from_sample(sample)
    cutoff = float(
        args.bse_cutoff_ev
        if args.bse_cutoff_ev is not None
        else getattr(sample.cfg, "bse_cutoff_ev", 50.0)
    )
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
    effective_trajectory_capacity = (
        args.trajectory_capacity if record_trajectories else 1
    )
    engine = TrapezoidTransportEngine(
        ti, fp, host, bulk, surface, geometry, args.capacity,
        bse_cutoff_ev=cutoff,
        trajectory_capacity=effective_trajectory_capacity,
    )

    # Compile once with trajectory recording disabled. The measured scan then
    # starts from deterministic per-pixel seeds.
    engine.run_trapezoid(
        n=min(args.trajectories, 64),
        energy_vac_ev=args.energy_ev,
        nominal_x=0.0,
        beam_fwhm=10.0 * args.beam_fwhm_nm,
        seed=args.seed,
        steps_per_chunk=min(args.steps_per_chunk, 32),
        copy_emissions=False,
    )
    ti.sync()

    if args.parallel or args.workers is not None:
        print(
            "Note: Taichi parallelizes trajectories on the selected device; "
            "scan pixels reuse one engine sequentially."
        )
    if args.ny > 1 and not args.no_record_trajectories:
        print(
            "Note: trajectory recording is disabled automatically for a 2D "
            "raster; all primaries contribute to the image maps."
        )
    rows, per_primary, trajectories = run_raster_scan(
        engine,
        geometry,
        energy_ev=args.energy_ev,
        x_positions_angstrom=10.0 * x_nm,
        y_positions_angstrom=10.0 * y_nm,
        primaries_per_pixel=args.trajectories,
        beam_fwhm_angstrom=10.0 * args.beam_fwhm_nm,
        seed=args.seed,
        steps_per_chunk=args.steps_per_chunk,
        cutoff_ev=cutoff,
        lle_max_loss_ev=args.lle_max_loss_ev,
        progress=True,
        trajectory_pixels=trace_pixels,
        trace_n_primaries=record_n if record_trajectories else 0,
        trajectory_stride=args.trajectory_stride,
        trajectory_max_points=args.trajectory_max_points,
        store_per_primary_counts=args.ny == 1 and not args.no_per_primary_counts,
    )

    if args.output_prefix is not None:
        csv_path = args.output_prefix.parent / f"{args.output_prefix.name}.csv"
    else:
        csv_path = args.output or Path(
            "line_scan.csv" if args.ny == 1 else "trapezoid_raster.csv"
        )
    csv_path, default_raster, default_trajectory = _derived_outputs(csv_path)
    raster_path = args.raster_output or default_raster
    trajectory_path = args.trajectory_output or default_trajectory

    geometry_metadata = {
        "type": "TrapezoidalLineArray",
        "suspended": geometry.suspended,
        "membrane_thickness": geometry.membrane_thickness,
        "top_width": geometry.top_width,
        "bottom_width": geometry.bottom_width,
        "height": geometry.height,
        "center_x": geometry.center_x,
        "substrate_z": 0.0,
        "n_lines": geometry.n_lines,
        "pitch": geometry.effective_pitch,
        "line_centers": list(geometry.line_centers),
        "span": geometry.span,
    }
    metadata = {
        "package": "seemc-taichi",
        "backend": args.arch,
        "precision": precision,
        "sample_name": args.material,
        "material": args.material,
        "energy_ev": args.energy_ev,
        "top_width_nm": args.top_width_nm,
        "bottom_width_nm": args.bottom_width_nm,
        "height_nm": args.height_nm,
        "n_lines": args.n_lines,
        "pitch_nm": args.pitch_nm,
        "line_centers_nm": [value / 10.0 for value in geometry.line_centers],
        "array_span_nm": array_span_nm,
        "field_width_nm": field_width_nm,
        "field_height_nm": field_height_nm,
        "scan_width_nm": field_width_nm,
        "nx": args.nx,
        "ny": args.ny,
        "n_pixels": args.nx * args.ny,
        "pixels": args.nx,
        "primaries_per_pixel": args.trajectories,
        "beam_fwhm_nm": args.beam_fwhm_nm,
        "capacity_per_pixel": args.capacity,
        "seed": args.seed,
        "bse_cutoff_ev": cutoff,
        "lle_max_loss_ev": args.lle_max_loss_ev,
        "recorded_pixels": sorted(trace_pixels),
        "record_trajectories": record_trajectories,
        "stores_per_primary_counts": args.ny == 1 and not args.no_per_primary_counts,
        "record_primaries_per_pixel": record_n if record_trajectories else 0,
        "trajectory_stride": args.trajectory_stride,
        "trajectory_max_points": args.trajectory_max_points,
        "trajectory_capacity_per_pixel": effective_trajectory_capacity,
        "barrier_model": surface.barrier_model,
        "inner_potential_ev": surface.inner_potential_ev,
        "suspended": geometry.suspended,
        "membrane_thickness": geometry.membrane_thickness,
        "signal_convention": "all-direction legacy channels exclude vacuum misses; use back_* for uz<0",
        "se1_definition": "generation==1 and own_inelastic_count==0",
        "geometry_convention": (
            "line tops z=-height; bottom z=0; vacuum outside finite lines; lines infinite y"
            if geometry.suspended else
            ("line tops z=-height; membrane from z=0 to membrane_thickness; vacuum below; lines infinite y"
             if geometry.membrane_thickness is not None else
             "vacuum negative z; line tops z=-height; substrate z=0; lines infinite y")
        ),
        "geometry": geometry_metadata,
        "config": {
            "energy_ev": args.energy_ev,
            "x_positions_angstrom": list(10.0 * x_nm),
            "y_positions_angstrom": list(10.0 * y_nm),
            "primaries_per_pixel": args.trajectories,
            "beam_fwhm_angstrom": [
                10.0 * args.beam_fwhm_nm,
                10.0 * args.beam_fwhm_nm,
            ],
            "seed": args.seed,
        },
        "profile_channels": list(CHANNELS),
    }

    save_scan_csv(csv_path, rows)
    save_scan_npz(raster_path, rows, per_primary, metadata=metadata)
    print(f"Wrote CSV:              {csv_path}")
    print(f"Wrote raster NPZ:       {raster_path}")
    if record_trajectories:
        save_trajectory_archive_npz(
            trajectory_path, rows, trajectories, metadata=metadata
        )
        print(f"Wrote animation NPZ:    {trajectory_path}")
        print(
            f"Recorded {record_n} of {args.trajectories} primaries/pixel "
            "for animation; all primaries contributed to yields."
        )
        overflow_pixels = [
            pixel for pixel, payload in trajectories.items()
            if payload.get("trajectory_overflow", False)
        ]
        if overflow_pixels:
            print(
                "WARNING: trajectory buffer overflow at pixels "
                + ", ".join(str(value) for value in overflow_pixels)
                + "; increase --trajectory-capacity for complete animations."
            )

    centers_nm = [value / 10.0 for value in geometry.line_centers]
    print(
        f"Geometry: {args.n_lines} line(s), pitch={args.pitch_nm:g} nm, "
        f"centers={centers_nm} nm"
    )
    if args.ny == 1:
        print(f"Scan: ny=1, nx={args.nx}, field width={field_width_nm:g} nm")
    else:
        print(
            f"Scan: ny={args.ny}, nx={args.nx}, "
            f"field={field_width_nm:g} x {field_height_nm:g} nm"
        )
    print(
        "Channels: legacy totals include all emission directions; "
        "back_* means final uz<0, forward_* means uz>=0. "
        "Vacuum misses are excluded; see gap_primary_fraction."
    )

    if args.plot:
        png_path = csv_path.with_suffix(".png")
        plotter = plot_scan if args.ny == 1 else plot_raster
        plot_kwargs = {"channels": args.plot_channels}
        if args.ny > 1:
            plot_kwargs.update(
                channels=args.plot_channels,
                cmap=args.plot_cmap,
            )
        plotter(
            png_path,
            rows,
            title=(
                f"{args.material}, {args.energy_ev:g} eV, {args.n_lines} line(s), "
                f"{args.trajectories:,} primaries/pixel, "
                f"{args.beam_fwhm_nm:g} nm FWHM"
            ),
            **plot_kwargs,
        )
        print(f"Wrote {png_path}")

    invalid = [row for row in rows if not row["valid"]]
    if invalid:
        print(f"WARNING: {len(invalid)} pixel(s) invalid; inspect status in {csv_path}")
    return 0 if not invalid else 2


if __name__ == "__main__":
    raise SystemExit(main())
