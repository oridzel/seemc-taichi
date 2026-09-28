from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np

from . import __version__
from .config import BackendConfig
from .plane import BulkPhysicsConfig, PlaneTransportEngine, load_reference_plane_tables
from .plane_psf import (
    CLASSIFICATION_DESCRIPTIONS,
    classify_emissions,
    emission_xy_nm,
    plot_mechanism_detail,
    plot_radial_psf,
    plot_seed_radius_depth,
    population_summary,
    radial_bin_edges,
    radial_profiles,
    seed_radius_depth_summary,
    write_profile_csv,
    write_rows_csv,
)
from .runtime import init_taichi
from .surface import SurfacePhysicsConfig


EVENT_KEYS = (
    "emission_energy_ev",
    "emission_x_angstrom",
    "emission_y_angstrom",
    "emission_z_angstrom",
    "emission_ux",
    "emission_uy",
    "emission_uz",
    "emission_electron_id",
    "emission_parent_id",
    "emission_root_primary_id",
    "emission_generation",
    "emission_is_cascade",
    "emission_mechanism",
    "emission_own_elastic_count",
    "emission_own_inelastic_count",
    "emission_path_elastic_count",
    "emission_path_inelastic_count",
    "emission_birth_x_angstrom",
    "emission_birth_y_angstrom",
    "emission_birth_z_angstrom",
)


def _safe_token(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text


def _default_prefix(material: str, energy_ev: float, angle_deg: float) -> Path:
    material = "".join(c if c.isalnum() or c in "-_" else "_" for c in material)
    return Path(
        f"{material}_{_safe_token(energy_ev)}eV_a{_safe_token(angle_deg)}deg_plane_psf"
    )


def _add_analysis_arguments(parser):
    parser.add_argument(
        "--beam-sigma-nm", type=float, default=0.0,
        help=(
            "post-process the point response with a circular Gaussian beam of "
            "this x/y sigma; use 1 for the attached paper's stated sigma"
        ),
    )
    parser.add_argument("--convolution-seed", type=int, default=20260921)
    parser.add_argument("--r-min-nm", type=float, default=0.02)
    parser.add_argument(
        "--r-max-nm", type=float, default=None,
        help="outer PSF radius; default is 1.01 times the largest recorded radius",
    )
    parser.add_argument("--radial-bins", type=int, default=140)
    parser.add_argument(
        "--min-count-per-bin", type=int, default=10,
        help=(
            "show a radial-PSF bin only when this many population events fall "
            "in it; the CSV always retains every raw bin"
        ),
    )
    parser.add_argument(
        "--seed-bins", type=int, default=90,
        help="bins per transformed coordinate for seed-radius/depth density maps",
    )
    parser.add_argument(
        "--cutoff-ev", type=float, default=None,
        help="override the saved SE/BSE cutoff during classification",
    )


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run or re-analyze a point-beam planar SEEMC simulation with "
            "collision-history-resolved radial PSFs"
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run planar transport and write raw events plus PSFs")
    run.add_argument("database")
    run.add_argument("--material", default="Si")
    run.add_argument("--energy-ev", type=float, default=10_000.0)
    run.add_argument("--angle-deg", type=float, default=0.0)
    run.add_argument(
        "--arch", choices=("cpu", "metal", "cuda", "vulkan", "gpu", "opengl"),
        default="cpu",
    )
    run.add_argument("--precision", choices=("f32", "f64"), default=None)
    run.add_argument("--n", type=int, default=100_000, help="total incident primaries")
    run.add_argument(
        "--batch-size", type=int, default=10_000,
        help="primaries transported per device batch",
    )
    run.add_argument(
        "--capacity", type=int, default=1_000_000,
        help="particle/cascade capacity for one batch",
    )
    run.add_argument("--steps-per-chunk", type=int, default=256)
    run.add_argument("--seed", type=int, default=20260921)
    run.add_argument("--output-prefix", type=Path, default=None)
    _add_analysis_arguments(run)

    plot = sub.add_parser("plot", help="reclassify and replot an existing event archive")
    plot.add_argument("events_npz", type=Path)
    plot.add_argument("--output-prefix", type=Path, default=None)
    _add_analysis_arguments(plot)
    return parser


def _load_archive(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def _save_archive_atomic(path: Path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _decode_metadata(data):
    raw = np.asarray(data.get("metadata_json", np.asarray("{}"))).item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _analysis_paths(prefix: Path):
    return {
        "summary": prefix.with_name(prefix.name + "_population_summary.csv"),
        "profile": prefix.with_name(prefix.name + "_radial_psf.csv"),
        "plot": prefix.with_name(prefix.name + "_radial_psf.png"),
        "detail_plot": prefix.with_name(prefix.name + "_mechanism_detail.png"),
        "seed_summary": prefix.with_name(
            prefix.name + "_seed_radius_depth_summary.csv"
        ),
        "seed_plot": prefix.with_name(prefix.name + "_seed_radius_depth.png"),
    }


def _analyze(data, prefix: Path, args):
    metadata = _decode_metadata(data)
    n_primaries = int(np.asarray(data.get("n_primaries", metadata.get("n_primaries"))).item())
    saved_cutoff = float(
        np.asarray(data.get("bse_cutoff_ev", metadata.get("bse_cutoff_ev", 50.0))).item()
    )
    cutoff = saved_cutoff if args.cutoff_ev is None else float(args.cutoff_ev)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive")
    if args.min_count_per_bin < 1:
        raise ValueError("--min-count-per-bin must be positive")
    if args.seed_bins < 20:
        raise ValueError("--seed-bins must be at least 20")

    masks = classify_emissions(data, cutoff_ev=cutoff)
    x_nm, y_nm = emission_xy_nm(
        data, beam_sigma_nm=args.beam_sigma_nm, seed=args.convolution_seed
    )
    radius_nm = np.hypot(x_nm, y_nm)
    if args.r_max_nm is None:
        observed = float(np.max(radius_nm)) if radius_nm.size else args.r_min_nm * 10.0
        r_max = max(args.r_min_nm * 10.0, observed * 1.01)
    else:
        r_max = float(args.r_max_nm)
    edges = radial_bin_edges(args.r_min_nm, r_max, args.radial_bins)
    profiles = radial_profiles(radius_nm, masks, edges, n_primaries)
    rows = population_summary(data, radius_nm, masks, n_primaries)
    seed_rows = seed_radius_depth_summary(data, masks)

    paths = _analysis_paths(prefix)
    write_rows_csv(paths["summary"], rows)
    write_profile_csv(paths["profile"], profiles)
    write_rows_csv(paths["seed_summary"], seed_rows)
    material = str(metadata.get("material", "material"))
    energy = float(metadata.get("energy_vac_ev", metadata.get("energy_ev", float("nan"))))
    angle = float(metadata.get("angle_deg", float("nan")))
    title = f"{material}, {energy:g} eV, incidence {angle:g} deg; cutoff {cutoff:g} eV"
    plot_radial_psf(
        paths["plot"], profiles, masks,
        n_primaries=n_primaries,
        title=title,
        beam_sigma_nm=args.beam_sigma_nm,
        min_count_per_bin=args.min_count_per_bin,
    )
    plot_mechanism_detail(
        paths["detail_plot"], profiles,
        n_primaries=n_primaries,
        title=title,
        beam_sigma_nm=args.beam_sigma_nm,
        min_count_per_bin=args.min_count_per_bin,
    )
    plot_seed_radius_depth(
        paths["seed_plot"], data, masks,
        n_primaries=n_primaries,
        title=title,
        n_bins=args.seed_bins,
    )

    lookup = {row["population"]: row for row in rows}
    print(f"analysis cutoff: {cutoff:g} eV")
    print(f"beam convolution sigma: {float(args.beam_sigma_nm):g} nm")
    print(f"observed maximum radius: {(float(np.max(radius_nm)) if radius_nm.size else 0.0):.6g} nm")
    for name in (
        "se_total", "se_single_inelastic_origin", "se_multiple_origin",
        "se_mult_gen1_after_elastic", "se_mult_gen1_after_prior_inelastic",
        "se_mult_generation_2plus",
        "se1_current_rule", "se2_current_rule",
        "bse_total", "bse_single_elastic", "bse_plural_elastic_only",
        "bse_elastic_plus_inelastic", "bse_other_high_energy_cascade",
        "bse_other_primary_inelastic_only", "combined_single", "combined_multiple",
    ):
        row = lookup[name]
        print(
            f"  {name}: count={row['count']:,}, "
            f"yield={row['yield_per_primary']:.8g}, r90={row['r90_nm']:.6g} nm"
        )
    print(f"wrote {paths['summary']}")
    print(f"wrote {paths['profile']}")
    print(f"wrote {paths['plot']}")
    print(f"wrote {paths['detail_plot']}")
    print(f"wrote {paths['seed_summary']}")
    print(f"wrote {paths['seed_plot']}")
    return paths


def _run(args):
    if args.n < 1 or args.batch_size < 1:
        raise ValueError("--n and --batch-size must be positive")
    if args.capacity < min(args.n, args.batch_size):
        raise ValueError("--capacity must be >= the number of primaries in one batch")
    if args.energy_ev <= 0.0:
        raise ValueError("--energy-ev must be positive")
    if not 0.0 <= args.angle_deg < 90.0:
        raise ValueError("--angle-deg must lie in [0, 90)")
    if args.steps_per_chunk < 1:
        raise ValueError("--steps-per-chunk must be positive")

    precision = args.precision or ("f32" if args.arch in {"metal", "cuda", "gpu"} else "f64")
    if args.arch == "metal" and precision == "f64":
        raise ValueError("Metal requires f32 on the supported Taichi build")
    prefix = args.output_prefix or _default_prefix(
        args.material, args.energy_ev, args.angle_deg
    )
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    events_path = prefix.with_name(prefix.name + "_events.npz")

    try:
        sample, host = load_reference_plane_tables(args.database, args.material)
    except ValueError as exc:
        detail = str(exc)
        if "shape" in detail and "expected" in detail:
            raise RuntimeError(
                "The material database is incompatible with this SEEMC reference "
                "loader: its transport arrays do not share one energy-grid length. "
                "Use the current matching MaterialDatabase.pkl and "
                "seemc-imaging>=0.7.5. Original error: " + detail
            ) from exc
        raise
    bulk = BulkPhysicsConfig.from_sample(sample)
    surface = SurfacePhysicsConfig.from_sample(sample)
    cutoff = float(getattr(sample.cfg, "bse_cutoff_ev", 50.0))
    if args.cutoff_ev is not None:
        cutoff = float(args.cutoff_ev)

    cfg = BackendConfig(
        arch=args.arch,
        max_particles=args.capacity,
        max_emissions=args.capacity,
        default_fp=precision,
        default_ip="i32",
        random_seed=args.seed,
    )
    ti = init_taichi(cfg)
    fp = ti.f32 if precision == "f32" else ti.f64
    engine = PlaneTransportEngine(
        ti, fp, host, bulk, surface, args.capacity, bse_cutoff_ev=cutoff
    )

    warm_n = min(64, args.batch_size, args.n)
    engine.run_plane(
        n=warm_n,
        energy_vac_ev=args.energy_ev,
        alpha_deg=args.angle_deg,
        steps_per_chunk=min(32, args.steps_per_chunk),
        copy_emissions=False,
    )
    ti.sync()

    accum = {key: [] for key in EVENT_KEYS}
    batch_index_parts = []
    diagnostic_sum = None
    primary_offset = 0
    particle_offset = 0
    elapsed_total = 0.0
    n_batches = int(math.ceil(args.n / args.batch_size))

    print(f"material: {args.material}; energy: {args.energy_ev:g} eV; angle: {args.angle_deg:g} deg")
    print(f"arch / precision: {args.arch} / {precision}")
    print(f"primaries: {args.n:,} in {n_batches} batch(es); capacity per batch: {args.capacity:,}")
    for batch in range(n_batches):
        n_batch = min(args.batch_size, args.n - primary_offset)
        start = time.perf_counter()
        result = engine.run_plane(
            n=n_batch,
            energy_vac_ev=args.energy_ev,
            alpha_deg=args.angle_deg,
            steps_per_chunk=args.steps_per_chunk,
            copy_emissions=True,
        )
        ti.sync()
        elapsed = time.perf_counter() - start
        elapsed_total += elapsed
        diag = result["diagnostics"]
        if result["overflow"] or result["emission_overflow"] or diag["emission_buffer_overflow"]:
            raise RuntimeError(
                f"batch {batch + 1} overflowed; reduce --batch-size or increase --capacity"
            )
        if diag["step_limit_hit"] or diag["generation_limit_hit"]:
            raise RuntimeError(
                f"batch {batch + 1} hit a transport safety limit: {diag}"
            )

        if diagnostic_sum is None:
            diagnostic_sum = {name: 0 for name in diag}
        for name, value in diag.items():
            diagnostic_sum[name] += int(value)

        local = {key: np.asarray(result[key]).copy() for key in EVENT_KEYS}
        local["emission_root_primary_id"] += primary_offset
        local["emission_electron_id"] += particle_offset
        parent = local["emission_parent_id"]
        parent[parent >= 0] += particle_offset
        for key in EVENT_KEYS:
            accum[key].append(local[key])
        batch_index_parts.append(
            np.full(local["emission_energy_ev"].size, batch, dtype=np.int32)
        )

        primary_offset += n_batch
        particle_offset += int(result["allocated"])
        print(
            f"[{batch + 1}/{n_batches}] n={n_batch:,}, emissions={local['emission_energy_ev'].size:,}, "
            f"allocated={int(result['allocated']):,}, elapsed={elapsed:.3f}s"
        )

    merged = {
        key: (
            np.concatenate(parts)
            if parts
            else np.asarray([], dtype=np.float64)
        )
        for key, parts in accum.items()
    }
    merged["emission_batch_index"] = (
        np.concatenate(batch_index_parts)
        if batch_index_parts
        else np.asarray([], dtype=np.int32)
    )
    n_emit = int(merged["emission_energy_ev"].size)
    if diagnostic_sum is None or n_emit != diagnostic_sum["escapes"]:
        raise RuntimeError("merged emission count does not match accumulated escape count")

    metadata = {
        "schema": "seemc-plane-psf-events-v1",
        "seemc_taichi_version": __version__,
        "material": str(args.material),
        "material_kind": "metal" if sample.is_metal else "nonconductor",
        "energy_vac_ev": float(args.energy_ev),
        "angle_deg": float(args.angle_deg),
        "n_primaries": int(args.n),
        "batch_size": int(args.batch_size),
        "n_batches": n_batches,
        "capacity_per_batch": int(args.capacity),
        "arch": str(args.arch),
        "precision": str(precision),
        "seed": int(args.seed),
        "steps_per_chunk": int(args.steps_per_chunk),
        "bse_cutoff_ev": cutoff,
        "point_beam": True,
        "position_unit": "angstrom",
        "surface_convention": "solid z>0; vacuum z<0; emission surface z=0",
        "bulk_physics": asdict(bulk),
        "surface_physics": asdict(surface),
        "diagnostics": diagnostic_sum,
        "classification_descriptions": CLASSIFICATION_DESCRIPTIONS,
        "elapsed_s": elapsed_total,
        "primaries_per_s": args.n / elapsed_total if elapsed_total > 0.0 else None,
    }
    _save_archive_atomic(
        events_path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        n_primaries=np.asarray(args.n, dtype=np.int64),
        bse_cutoff_ev=np.asarray(cutoff, dtype=float),
        **merged,
    )
    print(f"wrote {events_path}")
    _analyze(merged | {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "n_primaries": np.asarray(args.n, dtype=np.int64),
        "bse_cutoff_ev": np.asarray(cutoff, dtype=float),
    }, prefix, args)
    return 0


def _plot(args):
    data = _load_archive(args.events_npz)
    prefix = args.output_prefix
    if prefix is None:
        stem = args.events_npz.with_suffix("")
        name = stem.name[:-7] if stem.name.endswith("_events") else stem.name
        prefix = stem.with_name(name)
    _analyze(data, Path(prefix), args)
    return 0


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    return _plot(args)


if __name__ == "__main__":
    raise SystemExit(main())
