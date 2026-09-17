from __future__ import annotations

import csv
import math
from pathlib import Path


DIAGNOSTIC_FIELDS = (
    "elastic_events",
    "inelastic_events",
    "secondaries_queued",
    "se_below_barrier",
    "se_blocked_pauli",
    "se_pauli_fallback",
    "channel_reclassified",
    "omega_cdf_empty",
    "q_window_clipped",
    "q_cdf_empty",
    "step_limit_hit",
    "generation_limit_hit",
    "no_scattering_rate",
    "surface_encounters",
    "escapes",
    "internal_reflections",
    "incoming_barrier_encounters",
    "incoming_barrier_reflections",
    "incoming_barrier_transmissions",
    "vacuum_surface_hits",
    "geometry_reentries",
    "vacuum_hop_limit_hit",
    "step_chunk_continuations",
    "sey_50ev",
    "bse_50ev",
    "cascade_emissions",
    "primary_emissions",
    "emission_buffer_overflow",
)

CSV_FIELDS = (
    "material",
    "material_kind",
    "arch",
    "precision",
    "barrier_model",
    "inner_potential_eV",
    "bse_cutoff_eV",
    "energy_eV",
    "angle_deg",
    "n_primaries",
    "capacity",
    "steps_per_chunk",
    "seed",
    "valid",
    "status",
    "tey",
    "sey",
    "bsey",
    "cascade_yield",
    "primary_yield",
    "raw_tey",
    "raw_sey",
    "raw_bsey",
    "raw_cascade_yield",
    "raw_primary_yield",
    "allocated",
    "stored",
    "particle_overflow",
    "wave_count",
    "kernel_chunks",
    "elapsed_s",
    "collision_events",
    "collision_events_per_s",
    "primaries_per_s",
) + DIAGNOSTIC_FIELDS


def point_key(energy_ev: float, angle_deg: float) -> tuple[float, float]:
    """Stable key for one sweep point."""
    return (round(float(energy_ev), 12), round(float(angle_deg), 12))


def result_is_valid(result: dict) -> tuple[bool, str]:
    diag = result["diagnostics"]
    problems: list[str] = []
    if bool(result.get("overflow", False)):
        problems.append("particle_overflow")
    if int(diag.get("emission_buffer_overflow", 0)):
        problems.append("emission_buffer_overflow")
    if int(diag.get("step_limit_hit", 0)):
        problems.append("step_limit_hit")
    if int(diag.get("generation_limit_hit", 0)):
        problems.append("generation_limit_hit")
    if int(diag.get("omega_cdf_empty", 0)):
        problems.append("omega_cdf_empty")
    if int(diag.get("q_cdf_empty", 0)):
        problems.append("q_cdf_empty")
    valid = not problems
    return valid, "ok" if valid else ";".join(problems)


def build_sweep_row(
    result: dict,
    *,
    material: str,
    material_kind: str,
    arch: str,
    precision: str,
    barrier_model: str,
    inner_potential_ev: float,
    bse_cutoff_ev: float,
    capacity: int,
    steps_per_chunk: int,
    seed: int,
    elapsed_s: float,
) -> dict:
    diag = result["diagnostics"]
    valid, status = result_is_valid(result)
    collisions = int(diag.get("elastic_events", 0)) + int(diag.get("inelastic_events", 0))
    n = int(result["n_primaries"])

    raw_tey = float(result["tey"])
    raw_sey = float(result["sey_50ev"])
    raw_bsey = float(result["bsey_50ev"])
    raw_cascade = float(result["cascade_yield"])
    raw_primary = float(result["primary_yield"])
    maybe = (lambda x: x if valid else math.nan)

    row = {
        "material": material,
        "material_kind": material_kind,
        "arch": arch,
        "precision": precision,
        "barrier_model": barrier_model,
        "inner_potential_eV": float(inner_potential_ev),
        "bse_cutoff_eV": float(bse_cutoff_ev),
        "energy_eV": float(result["incident_energy_vac_ev"]),
        "angle_deg": float(result["alpha_deg"]),
        "n_primaries": n,
        "capacity": int(capacity),
        "steps_per_chunk": int(steps_per_chunk),
        "seed": int(seed),
        "valid": bool(valid),
        "status": status,
        "tey": maybe(raw_tey),
        "sey": maybe(raw_sey),
        "bsey": maybe(raw_bsey),
        "cascade_yield": maybe(raw_cascade),
        "primary_yield": maybe(raw_primary),
        "raw_tey": raw_tey,
        "raw_sey": raw_sey,
        "raw_bsey": raw_bsey,
        "raw_cascade_yield": raw_cascade,
        "raw_primary_yield": raw_primary,
        "allocated": int(result["allocated"]),
        "stored": int(result["stored"]),
        "particle_overflow": bool(result["overflow"]),
        "wave_count": int(result["wave_count"]),
        "kernel_chunks": int(result["kernel_chunks"]),
        "elapsed_s": float(elapsed_s),
        "collision_events": collisions,
        "collision_events_per_s": collisions / elapsed_s if elapsed_s > 0 else math.nan,
        "primaries_per_s": n / elapsed_s if elapsed_s > 0 else math.nan,
    }
    for name in DIAGNOSTIC_FIELDS:
        row[name] = int(diag.get(name, 0))
    return row


def load_csv_rows(path: str | Path) -> dict[tuple[float, float], dict]:
    path = Path(path)
    if not path.exists():
        return {}
    rows: dict[tuple[float, float], dict] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"energy_eV", "angle_deg"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} is not a SEEMC-Taichi sweep CSV")
        for row in reader:
            rows[point_key(float(row["energy_eV"]), float(row["angle_deg"]))] = row
    return rows


def csv_row_is_valid(row: dict) -> bool:
    value = str(row.get("valid", "")).strip().lower()
    return value in {"1", "true", "yes", "y"}


def write_csv_rows(path: str | Path, rows: dict[tuple[float, float], dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    ordered = sorted(
        rows.values(),
        key=lambda r: (float(r["angle_deg"]), float(r["energy_eV"])),
    )
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            writer.writerow({name: row.get(name, "") for name in CSV_FIELDS})
        f.flush()
    tmp.replace(path)


def emission_filename(material: str, energy_ev: float, angle_deg: float) -> str:
    def token(x: float) -> str:
        return f"{float(x):g}".replace("-", "m").replace(".", "p")
    safe_material = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(material))
    return f"{safe_material}_{token(energy_ev)}eV_a{token(angle_deg)}deg_emissions.npz"
