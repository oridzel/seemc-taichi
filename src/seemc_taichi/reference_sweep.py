from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np


REFERENCE_CSV_FIELDS = (
    "material",
    "material_kind",
    "barrier_model",
    "inner_potential_eV",
    "bse_cutoff_eV",
    "energy_eV",
    "angle_deg",
    "n_primaries",
    "seed",
    "case_seed",
    "workers",
    "valid",
    "status",
    "tey",
    "sey",
    "bsey",
    "cascade_yield",
    "primary_yield",
    "n_emitted",
    "n_cascade_emitted",
    "n_primary_emitted",
    "elapsed_s",
    "primaries_per_s",
)


def point_key(energy_ev: float, angle_deg: float) -> tuple[float, float]:
    return (round(float(energy_ev), 12), round(float(angle_deg), 12))


def deterministic_case_seed(base_seed: int, angle_deg: float, energy_ev: float) -> int:
    angle_key = int(round(float(angle_deg) * 1_000_000.0))
    energy_key = int(round(float(energy_ev) * 1_000.0))
    sequence = np.random.SeedSequence(
        [int(base_seed), angle_key & 0xFFFFFFFF, energy_key & 0xFFFFFFFF]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def reference_emission_filename(material: str, energy_ev: float, angle_deg: float) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(material))
    return f"{safe}_{_token(energy_ev)}eV_a{_token(angle_deg)}deg_reference_emissions.npz"


def plane_directions(angle_deg: float):
    alpha = math.radians(float(angle_deg))
    vacuum = (math.sin(alpha), 0.0, math.cos(alpha))
    outward = (0.0, 0.0, -1.0)
    return vacuum, outward


def emission_arrays(emissions, n_primaries: int) -> dict[str, np.ndarray]:
    """Convert reference ``Emission`` objects into Taichi-compatible arrays.

    The numeric mechanism code matches the accelerated buffer:
      1 = ordinary transport escape
      2 = incoming-barrier reflection
    A string label is stored as an additional reference-only field.
    """
    emissions = list(emissions or ())
    n = len(emissions)

    energy = np.empty(n, dtype=np.float64)
    direction = np.empty((n, 3), dtype=np.float64)
    electron_id = np.empty(n, dtype=np.int64)
    parent_id = np.empty(n, dtype=np.int64)
    root_id = np.empty(n, dtype=np.int64)
    generation = np.empty(n, dtype=np.int32)
    is_cascade = np.empty(n, dtype=np.int32)
    mechanism = np.empty(n, dtype=np.int32)
    mechanism_label = []
    birth_depth = np.empty(n, dtype=np.float64)
    barrier_r = np.empty(n, dtype=np.float64)

    for j, item in enumerate(emissions):
        energy[j] = float(item.energy)
        uvw = np.asarray(item.uvw, dtype=float)
        if uvw.shape != (3,):
            raise ValueError(f"emission {j} has invalid direction shape {uvw.shape}")
        norm = float(np.linalg.norm(uvw))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"emission {j} has invalid direction norm {norm}")
        direction[j] = uvw / norm

        electron_id[j] = int(getattr(item, "electron_id", -1))
        parent = getattr(item, "parent_id", None)
        parent_id[j] = -1 if parent is None else int(parent)
        root_id[j] = int(getattr(item, "root_primary_id", -1))
        generation[j] = int(getattr(item, "generation", 0))
        is_cascade[j] = int(bool(getattr(item, "is_cascade", generation[j] > 0)))

        label = str(getattr(item, "emission_mechanism", "transport_escape"))
        mechanism_label.append(label)
        mechanism[j] = 2 if label == "incoming_barrier_reflection" else 1
        birth_depth[j] = float(getattr(item, "birth_depth", math.nan))
        r = getattr(item, "barrier_reflection_probability", None)
        barrier_r[j] = math.nan if r is None else float(r)

    if n:
        if np.any(~np.isfinite(energy)) or np.any(energy < 0.0):
            raise ValueError("reference emissions contain invalid energies")
        if np.any(root_id < 0) or np.any(root_id >= int(n_primaries)):
            lo, hi = int(root_id.min()), int(root_id.max())
            raise ValueError(
                f"reference root_primary_id outside [0,{n_primaries}): [{lo},{hi}]; "
                "use seemc-imaging 0.7.4 or newer"
            )
        if np.any(generation < 0):
            raise ValueError("reference emissions contain negative generations")
        # ``is_cascade`` and generation should agree for the current reference.
        expected_cascade = generation > 0
        if not np.array_equal(is_cascade.astype(bool), expected_cascade):
            raise ValueError("reference is_cascade disagrees with generation")

    return {
        "emission_energy_ev": energy,
        "emission_ux": direction[:, 0] if n else np.empty(0, dtype=np.float64),
        "emission_uy": direction[:, 1] if n else np.empty(0, dtype=np.float64),
        "emission_uz": direction[:, 2] if n else np.empty(0, dtype=np.float64),
        "emission_electron_id": electron_id,
        "emission_parent_id": parent_id,
        "emission_root_primary_id": root_id,
        "emission_generation": generation,
        "emission_is_cascade": is_cascade,
        "emission_mechanism": mechanism,
        "emission_mechanism_label": np.asarray(mechanism_label, dtype="U64"),
        "emission_birth_depth_angstrom": birth_depth,
        "emission_barrier_reflection_probability": barrier_r,
    }


def summarize_reference_emissions(
    arrays: dict[str, np.ndarray],
    *,
    n_primaries: int,
    cutoff_ev: float,
) -> dict[str, float | int]:
    n_primary = int(n_primaries)
    if n_primary < 1:
        raise ValueError("n_primaries must be positive")
    energy = np.asarray(arrays["emission_energy_ev"], dtype=float)
    cascade = np.asarray(arrays["emission_is_cascade"], dtype=np.int32).astype(bool)
    is_se = energy <= float(cutoff_ev) + 1.0e-12
    n_emit = int(energy.size)
    n_cascade = int(np.count_nonzero(cascade))
    n_primary_emit = n_emit - n_cascade
    return {
        "n_emitted": n_emit,
        "n_cascade_emitted": n_cascade,
        "n_primary_emitted": n_primary_emit,
        "tey": n_emit / n_primary,
        "sey": int(np.count_nonzero(is_se)) / n_primary,
        "bsey": int(np.count_nonzero(~is_se)) / n_primary,
        "cascade_yield": n_cascade / n_primary,
        "primary_yield": n_primary_emit / n_primary,
    }


def save_reference_emissions(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    *,
    material: str,
    energy_ev: float,
    angle_deg: float,
    n_primaries: int,
    cutoff_ev: float,
    seed: int,
    case_seed: int,
    valid: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "material": np.asarray(str(material)),
        "energy_eV": np.asarray(float(energy_ev)),
        "angle_deg": np.asarray(float(angle_deg)),
        "n_primaries": np.asarray(int(n_primaries), dtype=np.int64),
        "bse_cutoff_eV": np.asarray(float(cutoff_ev)),
        "seed": np.asarray(int(seed), dtype=np.int64),
        "case_seed": np.asarray(int(case_seed), dtype=np.uint64),
        "valid": np.asarray(bool(valid)),
        **arrays,
    }
    with tmp.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, path)
    return path


def load_csv_rows(path: str | Path) -> dict[tuple[float, float], dict]:
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[tuple[float, float], dict] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"energy_eV", "angle_deg"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} is not a reference emission sweep CSV")
        for row in reader:
            out[point_key(float(row["energy_eV"]), float(row["angle_deg"]))] = row
    return out


def csv_row_is_valid(row: dict) -> bool:
    return str(row.get("valid", "")).strip().lower() in {"1", "true", "yes", "y"}


def write_csv_rows(path: str | Path, rows: dict[tuple[float, float], dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    ordered = sorted(
        rows.values(),
        key=lambda row: (float(row["angle_deg"]), float(row["energy_eV"])),
    )
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REFERENCE_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            writer.writerow({name: row.get(name, "") for name in REFERENCE_CSV_FIELDS})
        f.flush()
    os.replace(tmp, path)
