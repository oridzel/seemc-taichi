"""High-statistics Taichi line scans over a raised trapezoidal SEM feature.

The transport itself is performed by :class:`TrapezoidTransportEngine`.
This module is deliberately host-side: it turns emitted-electron records into
per-primary signal counts, Monte-Carlo SEMs, a line-scan CSV, and a compact NPZ.

Signal taxonomy used here follows the current SEEMC imaging work:

* ``tey``: every emitted electron.
* ``sey_50ev`` / ``bsey_50ev``: conventional emitted-energy split.
* ``cascade_all``: every emitted cascade electron (full secondary signal).
* ``primary_all``: every emitted original incident electron (full BSE signal).
* ``se1``: generation-1 cascade electron that escaped without an inelastic
  collision of its own.  Elastic collisions do not disqualify it.
* ``se2``: every other emitted cascade electron.

The SE1/SE2 definition is intentionally independent of the 50 eV detector cut.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from .scan_archive import archive_from_scan
from .sweep import result_is_valid


CHANNELS = (
    "tey",
    "sey_50ev",
    "bsey_50ev",
    "cascade_all",
    "primary_all",
    "se1",
    "se2",
    "generation_1",
    "generation_2plus",
    "lle_primary",
    "non_lle_primary",
)

CHANNEL_DEFINITIONS = {
    "tey": "All emitted electrons.",
    "sey_50ev": "Emitted electrons with vacuum energy <= the BSE cutoff.",
    "bsey_50ev": "Emitted electrons with vacuum energy > the BSE cutoff.",
    "cascade_all": "All emitted cascade-origin electrons.",
    "primary_all": "All emitted original incident primaries.",
    "se1": (
        "Generation-1 cascade electron that escaped without any inelastic "
        "collision of its own; elastic collisions are allowed."
    ),
    "se2": "Every other emitted cascade-origin electron.",
    "generation_1": "All emitted generation-1 cascade electrons.",
    "generation_2plus": "All emitted cascade electrons with generation >= 2.",
    "lle_primary": "Emitted original primary with vacuum energy loss < LLE threshold.",
    "non_lle_primary": "Emitted original primary with vacuum energy loss >= LLE threshold.",
}

SURFACE_NAMES = {
    1: "top",
    2: "right_sidewall",
    3: "left_sidewall",
    4: "substrate",
}

TRAJECTORY_EVENT_NAMES = {
    1: "launch",
    3: "elastic",
    4: "inelastic",
    5: "spawn",
    6: "surface_hit",
    7: "internal_reflection",
    8: "vacuum_surface_hit",
    9: "escape",
    10: "terminate",
    11: "incoming_barrier_reflection",
    12: "geometry_reentry",
}

_REST_ENERGY_EV = 510_998.95069
_C_ANGSTROM_PER_FS = 2_997.92458


def _emission_arrays(result: dict):
    energy = np.asarray(result["emission_energy_ev"], dtype=float)
    roots = np.asarray(result["emission_root_primary_id"], dtype=np.int64)
    generation = np.asarray(result["emission_generation"], dtype=np.int32)
    cascade = np.asarray(result["emission_is_cascade"], dtype=np.int32).astype(bool)
    inelastic = np.asarray(result["emission_inelastic_count"], dtype=np.int32)
    n = len(energy)
    for name, arr in {
        "root_primary_id": roots,
        "generation": generation,
        "is_cascade": cascade,
        "inelastic_count": inelastic,
    }.items():
        if len(arr) != n:
            raise ValueError(f"emission {name} length does not match energy length")
    return energy, roots, generation, cascade, inelastic


def emission_masks(
    result: dict,
    *,
    cutoff_ev: float = 50.0,
    lle_max_loss_ev: float = 50.0,
) -> dict[str, np.ndarray]:
    """Boolean emission masks for all line-scan channels."""
    energy, _roots, generation, cascade, inelastic = _emission_arrays(result)
    primary = ~cascade
    se1 = cascade & (generation == 1) & (inelastic == 0)
    se2 = cascade & ~se1
    incident = float(result["incident_energy_vac_ev"])
    loss = incident - energy
    masks = {
        "tey": np.ones(len(energy), dtype=bool),
        "sey_50ev": energy <= float(cutoff_ev),
        "bsey_50ev": energy > float(cutoff_ev),
        "cascade_all": cascade,
        "primary_all": primary,
        "se1": se1,
        "se2": se2,
        "generation_1": cascade & (generation == 1),
        "generation_2plus": cascade & (generation >= 2),
        "lle_primary": primary & (loss < float(lle_max_loss_ev)),
        "non_lle_primary": primary & (loss >= float(lle_max_loss_ev)),
    }
    if not np.array_equal(masks["cascade_all"], masks["se1"] | masks["se2"]):
        raise RuntimeError("SE1/SE2 masks do not partition cascade emissions")
    if np.any(masks["se1"] & masks["se2"]):
        raise RuntimeError("SE1/SE2 masks overlap")
    return masks


def counts_per_primary(result: dict, mask: np.ndarray) -> np.ndarray:
    roots = np.asarray(result["emission_root_primary_id"], dtype=np.int64)
    n = int(result["n_primaries"])
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != roots.shape:
        raise ValueError("emission mask has the wrong shape")
    selected = roots[mask]
    if selected.size and (selected.min() < 0 or selected.max() >= n):
        raise ValueError("emission root-primary IDs are outside the primary range")
    return np.bincount(selected, minlength=n).astype(np.int64)


def yield_and_sem(counts: np.ndarray) -> tuple[float, float]:
    values = np.asarray(counts, dtype=float)
    if values.size == 0:
        return math.nan, math.nan
    mean = float(values.mean())
    sem = 0.0 if values.size < 2 else float(values.std(ddof=0) / math.sqrt(values.size))
    return mean, sem


def classify_result(
    result: dict,
    *,
    cutoff_ev: float = 50.0,
    lle_max_loss_ev: float = 50.0,
) -> dict[str, float | int | np.ndarray]:
    """Convert copied emissions into yields, SEMs, and per-primary counts."""
    masks = emission_masks(
        result,
        cutoff_ev=cutoff_ev,
        lle_max_loss_ev=lle_max_loss_ev,
    )
    out: dict[str, float | int | np.ndarray] = {}
    for channel in CHANNELS:
        counts = counts_per_primary(result, masks[channel])
        mean, sem = yield_and_sem(counts)
        out[channel] = mean
        out[f"{channel}_sem"] = sem
        out[f"{channel}_counts_per_primary"] = counts

    # Independent invariants beyond those already checked inside the engine.
    tol = 1.0e-12
    if abs(float(out["tey"]) - float(out["sey_50ev"]) - float(out["bsey_50ev"])) > tol:
        raise RuntimeError("TEY is not partitioned by the conventional SE/BSE channels")
    if abs(float(out["tey"]) - float(out["cascade_all"]) - float(out["primary_all"])) > tol:
        raise RuntimeError("TEY is not partitioned by cascade/primary channels")
    if abs(float(out["cascade_all"]) - float(out["se1"]) - float(out["se2"])) > tol:
        raise RuntimeError("cascade signal is not partitioned by SE1/SE2")
    return out


def _pixel_seed(base_seed: int, pixel_index: int) -> int:
    seq = np.random.SeedSequence([int(base_seed), int(pixel_index)])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def nominal_surface_info(geometry, x_angstrom: float) -> tuple[float, str, float]:
    z, _nx, nz, code = geometry.launch_surface(float(x_angstrom))
    cosine = max(-1.0, min(1.0, -float(nz)))
    incidence = math.degrees(math.acos(cosine))
    return float(z), SURFACE_NAMES[int(code)], incidence


def run_line_scan(
    engine,
    geometry,
    *,
    energy_ev: float,
    x_positions_angstrom,
    primaries_per_pixel: int,
    beam_fwhm_angstrom: float = 0.0,
    seed: int = 20260916,
    steps_per_chunk: int = 256,
    cutoff_ev: float = 50.0,
    lle_max_loss_ev: float = 50.0,
    progress: bool = True,
    trajectory_pixels: set[int] | None = None,
    trace_n_primaries: int = 0,
    trajectory_stride: int = 1,
    trajectory_max_points: int | None = None,
) -> tuple[list[dict], dict[str, np.ndarray], dict[int, dict]]:
    """Run a normal-incidence one-dimensional trapezoidal line scan."""
    x_positions = np.asarray(x_positions_angstrom, dtype=float)
    if x_positions.ndim != 1 or x_positions.size < 1 or not np.all(np.isfinite(x_positions)):
        raise ValueError("x_positions_angstrom must be a finite one-dimensional array")
    n = int(primaries_per_pixel)
    if n < 1:
        raise ValueError("primaries_per_pixel must be positive")
    trajectory_stride = int(trajectory_stride)
    if trajectory_stride < 1:
        raise ValueError("trajectory_stride must be positive")
    if trajectory_max_points is not None and int(trajectory_max_points) < 2:
        raise ValueError("trajectory_max_points must be at least 2")

    rows: list[dict] = []
    per_primary: dict[str, list[np.ndarray]] = {name: [] for name in CHANNELS}
    trajectory_pixels = set() if trajectory_pixels is None else set(int(v) for v in trajectory_pixels)
    trajectories: dict[int, dict] = {}
    total_start = time.perf_counter()

    for ix, x in enumerate(x_positions):
        pseed = _pixel_seed(seed, ix)
        start = time.perf_counter()
        trace_this_pixel = ix in trajectory_pixels and int(trace_n_primaries) > 0
        result = engine.run_trapezoid(
            n=n,
            energy_vac_ev=float(energy_ev),
            nominal_x=float(x),
            nominal_y=0.0,
            beam_fwhm=float(beam_fwhm_angstrom),
            seed=pseed,
            steps_per_chunk=int(steps_per_chunk),
            copy_emissions=True,
            copy_trajectories=trace_this_pixel,
            trace_n_primaries=int(trace_n_primaries) if trace_this_pixel else 0,
        )
        engine.ti.sync()
        elapsed = time.perf_counter() - start
        valid, status = result_is_valid(result)
        classified = classify_result(
            result,
            cutoff_ev=cutoff_ev,
            lle_max_loss_ev=lle_max_loss_ev,
        )
        z_nom, surface_name, incidence_nom = nominal_surface_info(geometry, float(x))
        diag = result["diagnostics"]
        collision_events = int(diag["elastic_events"]) + int(diag["inelastic_events"])

        row = {
            "pixel_id": ix,
            "x_nm": float(x) / 10.0,
            "nominal_surface_z_nm": z_nom / 10.0,
            "nominal_surface_id": surface_name,
            "nominal_local_incidence_deg": incidence_nom,
            "launch_mean_x_nm": float(result.get("launch_mean_x_angstrom", x)) / 10.0,
            "launch_mean_y_nm": float(result.get("launch_mean_y_angstrom", 0.0)) / 10.0,
            "launch_surface_z_mean_nm": float(result.get("launch_surface_z_mean_angstrom", z_nom)) / 10.0,
            "local_incidence_mean_deg": float(result.get("local_incidence_mean_deg", incidence_nom)),
            "local_incidence_sem_deg": float(result.get("local_incidence_sem_deg", 0.0)),
            "n_primaries": n,
            "valid": bool(valid),
            "status": status,
            "elapsed_s": elapsed,
            "primaries_per_s": n / elapsed if elapsed > 0 else math.nan,
            "collision_events": collision_events,
            "collision_events_per_s": collision_events / elapsed if elapsed > 0 else math.nan,
            "allocated": int(result["allocated"]),
            "stored": int(result["stored"]),
            "particle_overflow": bool(result["overflow"]),
            "wave_count": int(result["wave_count"]),
            "kernel_chunks": int(result["kernel_chunks"]),
        }
        for channel in CHANNELS:
            value = float(classified[channel])
            sem = float(classified[f"{channel}_sem"])
            row[channel] = value if valid else math.nan
            row[f"{channel}_sem"] = sem if valid else math.nan
            row[f"raw_{channel}"] = value
            per_primary[channel].append(
                np.asarray(classified[f"{channel}_counts_per_primary"], dtype=np.int32)
            )
        for name, value in diag.items():
            key = f"diag_{name}" if name in row else name
            row[key] = int(value)
        if trace_this_pixel:
            trajectories[ix] = extract_trajectory_payload(
                result,
                pixel_id=ix,
                x_nm=row["x_nm"],
                stride=trajectory_stride,
                max_points=trajectory_max_points,
            )
        rows.append(row)

        if progress:
            tag = "VALID" if valid else f"INVALID:{status}"
            print(
                f"[{ix + 1}/{len(x_positions)}] x={x / 10.0:+.3f} nm "
                f"{surface_name} {tag}  TEY={float(classified['tey']):.4f} "
                f"SE1={float(classified['se1']):.4f} "
                f"SE2={float(classified['se2']):.4f} "
                f"BSE={float(classified['primary_all']):.4f}  {elapsed:.3f}s"
            )

    per_primary_arrays = {
        channel: np.stack(values, axis=0) if values else np.empty((0, n), dtype=np.int32)
        for channel, values in per_primary.items()
    }
    if progress:
        print(f"line scan complete in {time.perf_counter() - total_start:.3f} s")
    return rows, per_primary_arrays, trajectories


def _flight_time_fs(distance_angstrom: float, energy_ev: float) -> float:
    if distance_angstrom <= 0.0 or energy_ev <= 0.0:
        return 0.0
    gamma = 1.0 + float(energy_ev) / _REST_ENERGY_EV
    beta2 = max(1.0 - 1.0 / (gamma * gamma), 0.0)
    if beta2 <= 0.0:
        return 0.0
    return float(distance_angstrom) / (_C_ANGSTROM_PER_FS * math.sqrt(beta2))


def _trajectory_times(result: dict) -> np.ndarray:
    """Approximate physical time for every stored Taichi trajectory point.

    Taichi records event states rather than clock time.  Free-flight time is
    reconstructed from consecutive points and instantaneous energy.  A child
    begins at the closest recorded point on its parent, which preserves the
    causal cascade timing needed by the SEEMC-imaging style animation.
    """
    x = np.asarray(result["trajectory_x_angstrom"], dtype=float)
    y = np.asarray(result["trajectory_y_angstrom"], dtype=float)
    z = np.asarray(result["trajectory_z_angstrom"], dtype=float)
    energy = np.asarray(result["trajectory_energy_ev"], dtype=float)
    electron = np.asarray(result["trajectory_electron_id"], dtype=np.int64)
    parent = np.asarray(result["trajectory_parent_id"], dtype=np.int64)
    times = np.zeros(len(x), dtype=float)
    if not len(x):
        return times

    indices_by_electron = {
        int(eid): np.flatnonzero(electron == eid) for eid in np.unique(electron)
    }
    for eid in sorted(indices_by_electron):
        indices = indices_by_electron[eid]
        first = int(indices[0])
        parent_id = int(parent[first])
        birth_time = 0.0
        if parent_id in indices_by_electron:
            parent_indices = indices_by_electron[parent_id]
            dx = x[parent_indices] - x[first]
            dy = y[parent_indices] - y[first]
            dz = z[parent_indices] - z[first]
            nearest = int(parent_indices[np.argmin(dx * dx + dy * dy + dz * dz)])
            birth_time = float(times[nearest])
        times[first] = birth_time
        for previous, current in zip(indices[:-1], indices[1:]):
            distance = math.sqrt(
                (x[current] - x[previous]) ** 2
                + (y[current] - y[previous]) ** 2
                + (z[current] - z[previous]) ** 2
            )
            segment_energy = max(float(energy[previous]), float(energy[current]), 0.0)
            times[current] = times[previous] + _flight_time_fs(
                distance, segment_energy
            )
    return times


def _decimation_indices(
    electron_id: np.ndarray,
    *,
    stride: int,
    max_points: int | None,
) -> np.ndarray:
    keep = []
    for eid in np.unique(electron_id):
        indices = np.flatnonzero(electron_id == eid)
        selected = indices[::stride]
        if len(indices) and (not len(selected) or selected[-1] != indices[-1]):
            selected = np.append(selected, indices[-1])
        if max_points is not None and len(selected) > int(max_points):
            positions = np.rint(
                np.linspace(0, len(selected) - 1, int(max_points))
            ).astype(int)
            selected = selected[np.unique(positions)]
        keep.extend(int(value) for value in selected)
    return np.asarray(sorted(keep), dtype=np.int64)


def extract_trajectory_payload(
    result: dict,
    *,
    pixel_id: int,
    x_nm: float,
    stride: int = 1,
    max_points: int | None = None,
) -> dict:
    if "trajectory_x_angstrom" not in result:
        raise ValueError("result does not contain copied trajectories")
    electron = np.asarray(result["trajectory_electron_id"], dtype=np.int32)
    indices = _decimation_indices(
        electron, stride=int(stride), max_points=max_points
    )
    times = _trajectory_times(result)
    trace_n = int(result.get("trace_n_primaries", 0))
    emission_roots = np.asarray(result["emission_root_primary_id"], dtype=np.int32)
    emission_keep = emission_roots < trace_n
    payload = {
        "pixel_id": int(pixel_id),
        "x_nm": float(x_nm),
        "trace_n_primaries": trace_n,
        "trajectory_overflow": bool(result.get("trajectory_overflow", False)),
        "trajectory_event_names_json": json.dumps(TRAJECTORY_EVENT_NAMES, sort_keys=True),
        "trajectory_surface_names_json": json.dumps(SURFACE_NAMES, sort_keys=True),
        "trajectory_x_angstrom": np.asarray(result["trajectory_x_angstrom"], dtype=float)[indices],
        "trajectory_y_angstrom": np.asarray(result["trajectory_y_angstrom"], dtype=float)[indices],
        "trajectory_z_angstrom": np.asarray(result["trajectory_z_angstrom"], dtype=float)[indices],
        "trajectory_energy_ev": np.asarray(result["trajectory_energy_ev"], dtype=float)[indices],
        "trajectory_time_fs": times[indices],
        "trajectory_electron_id": electron[indices],
        "trajectory_parent_id": np.asarray(result["trajectory_parent_id"], dtype=np.int32)[indices],
        "trajectory_root_primary_id": np.asarray(result["trajectory_root_primary_id"], dtype=np.int32)[indices],
        "trajectory_generation": np.asarray(result["trajectory_generation"], dtype=np.int32)[indices],
        "trajectory_inelastic_count": np.asarray(result["trajectory_inelastic_count"], dtype=np.int32)[indices],
        "trajectory_event": np.asarray(result["trajectory_event"], dtype=np.int32)[indices],
        "trajectory_surface_code": np.asarray(result["trajectory_surface_code"], dtype=np.int32)[indices],
        "trajectory_step": np.asarray(result["trajectory_step"], dtype=np.int32)[indices],
        "emission_electron_id": np.asarray(result["emission_electron_id"], dtype=np.int32)[emission_keep],
        "emission_root_primary_id": emission_roots[emission_keep],
        "emission_energy_ev": np.asarray(result["emission_energy_ev"], dtype=float)[emission_keep],
        "emission_ux": np.asarray(result["emission_ux"], dtype=float)[emission_keep],
        "emission_uy": np.asarray(result["emission_uy"], dtype=float)[emission_keep],
        "emission_uz": np.asarray(result["emission_uz"], dtype=float)[emission_keep],
        "emission_generation": np.asarray(result["emission_generation"], dtype=np.int32)[emission_keep],
        "emission_inelastic_count": np.asarray(result["emission_inelastic_count"], dtype=np.int32)[emission_keep],
    }
    return payload


def save_trajectory_npz(path: str | Path, trajectory: dict, *, metadata: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(trajectory)
    if metadata is not None:
        payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **payload)
    return path


def save_trajectory_archive_npz(
    path: str | Path,
    rows: list[dict],
    trajectories: dict[int, dict],
    *,
    metadata: dict,
) -> Path:
    """Save all recorded scan pixels and profiles to one animation archive."""
    archive = archive_from_scan(rows, trajectories, metadata=metadata)
    return archive.save_npz(path)


def save_scan_csv(path: str | Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot save an empty line scan")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_scan_npz(
    path: str | Path,
    rows: list[dict],
    per_primary: dict[str, np.ndarray],
    *,
    metadata: dict,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot save an empty line scan")
    payload: dict[str, np.ndarray] = {
        "x_nm": np.asarray([row["x_nm"] for row in rows], dtype=float),
        "valid": np.asarray([row["valid"] for row in rows], dtype=bool),
        "nominal_surface_z_nm": np.asarray([row["nominal_surface_z_nm"] for row in rows], dtype=float),
        "nominal_local_incidence_deg": np.asarray([row["nominal_local_incidence_deg"] for row in rows], dtype=float),
        "local_incidence_mean_deg": np.asarray([row["local_incidence_mean_deg"] for row in rows], dtype=float),
        "local_incidence_sem_deg": np.asarray([row["local_incidence_sem_deg"] for row in rows], dtype=float),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "channel_definitions_json": np.asarray(json.dumps(CHANNEL_DEFINITIONS, sort_keys=True)),
    }
    for channel in CHANNELS:
        payload[channel] = np.asarray([row[channel] for row in rows], dtype=float)
        payload[f"{channel}_sem"] = np.asarray([row[f"{channel}_sem"] for row in rows], dtype=float)
        payload[f"{channel}_counts_per_primary"] = np.asarray(per_primary[channel], dtype=np.int32)
    np.savez_compressed(path, **payload)
    return path


def plot_scan(path: str | Path, rows: list[dict], *, title: str | None = None) -> Path:
    """Quick diagnostic plot; matplotlib is optional."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional convenience
        raise RuntimeError("--plot requires matplotlib") from exc
    x = np.asarray([row["x_nm"] for row in rows], dtype=float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for channel, label in (
        ("tey", "TEY"),
        ("cascade_all", "Cascade all"),
        ("primary_all", "Primary all"),
        ("se1", "SE1"),
        ("se2", "SE2"),
    ):
        y = np.asarray([row[channel] for row in rows], dtype=float)
        sem = np.asarray([row[f"{channel}_sem"] for row in rows], dtype=float)
        ax.plot(x, y, label=label)
        ax.fill_between(x, y - 1.96 * sem, y + 1.96 * sem, alpha=0.12)
    ax.set_xlabel("Beam x (nm)")
    ax.set_ylabel("Emitted electrons / primary")
    ax.set_title(title or "SEEMC-Taichi trapezoidal line scan")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
