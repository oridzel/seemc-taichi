"""Host-side population classification and radial PSFs for planar emissions.

The transport run deliberately saves primitive event facts rather than a single
hard-coded "single/multiple" label.  This module supplies one physically explicit
classification while keeping the raw counters available for alternative rules.
"""

from __future__ import annotations

from collections import OrderedDict
import csv
import json
from pathlib import Path

import numpy as np


CLASSIFICATION_DESCRIPTIONS = OrderedDict(
    (
        ("all_emitted", "Every recorded vacuum emission."),
        (
            "direct_barrier_reflection",
            "Incident primary reflected at the incoming surface barrier without solid transport.",
        ),
        ("transport_total", "All emissions after transport inside the solid."),
        ("se_total", "Transport emission with vacuum energy <= the SE/BSE cutoff."),
        (
            "se_single_inelastic_origin",
            "Generation-1 SE born in the first inelastic collision of a primary that had no prior elastic collision; later SE transport is allowed.",
        ),
        (
            "se_strict_single_inelastic",
            "SE single-inelastic-origin subset that escapes without any collision of its own.",
        ),
        (
            "se_multiple_origin",
            "All other conventional SE emissions, including deflected-primary, later-generation, and rescattered-primary origins.",
        ),
        (
            "se_mult_gen1_after_elastic",
            "Multiple-origin generation-1 SE whose primary parent had at least one prior elastic collision.",
        ),
        (
            "se_mult_gen1_after_prior_inelastic",
            "Multiple-origin generation-1 SE created after at least two primary inelastic collisions and no prior elastic collision.",
        ),
        (
            "se_mult_generation_2plus",
            "Multiple-origin SE with generation >= 2.",
        ),
        (
            "se_mult_primary",
            "Primary electron emitted below the conventional SE/BSE cutoff.",
        ),
        (
            "se_mult_other",
            "Residual multiple-origin SE not assigned to the explicit lineage groups.",
        ),
        (
            "se1_current_rule",
            "Current imaging SE1 rule: generation == 1 and the emitted electron has zero own inelastic collisions.",
        ),
        (
            "se1_single_origin",
            "Current-rule SE1 that is also single-inelastic-origin.",
        ),
        (
            "se1_multiple_origin",
            "Current-rule SE1 that belongs to the multiple-origin population.",
        ),
        ("se2_current_rule", "Conventional SE not assigned to the current SE1 rule."),
        (
            "se2_single_origin",
            "Current-rule SE2 that is nevertheless single-inelastic-origin because it scattered inelastically after birth.",
        ),
        (
            "se2_multiple_origin",
            "Current-rule SE2 that belongs to the multiple-origin population.",
        ),
        (
            "bse_total",
            "Transport emission with vacuum energy above the SE/BSE cutoff.",
        ),
        (
            "bse_single_elastic",
            "Escaping primary with exactly one elastic and zero inelastic collisions.",
        ),
        (
            "bse_plural_elastic_only",
            "Escaping primary with at least two elastic and zero inelastic collisions.",
        ),
        (
            "bse_elastic_plus_inelastic",
            "Escaping primary with at least one elastic and at least one inelastic collision.",
        ),
        (
            "bse_other",
            "Remaining transport BSEs, such as high-energy cascade electrons or paths with no elastic collision.",
        ),
        (
            "bse_other_high_energy_cascade",
            "BSE-energy emitted cascade electron with generation >= 1.",
        ),
        (
            "bse_other_primary_inelastic_only",
            "Escaping primary BSE with one or more inelastic collisions and no elastic collision.",
        ),
        (
            "bse_other_primary_no_collision",
            "Escaping primary BSE with no recorded bulk collision.",
        ),
        (
            "bse_other_unclassified",
            "Residual BSE not assigned to an explicit BSE mechanism group.",
        ),
        (
            "combined_single",
            "SE single-inelastic-origin plus BSE single-elastic populations.",
        ),
        (
            "combined_multiple",
            "All other transport emissions; direct incoming-barrier reflections are excluded.",
        ),
    )
)


REQUIRED_EVENT_KEYS = (
    "emission_energy_ev",
    "emission_x_angstrom",
    "emission_y_angstrom",
    "emission_generation",
    "emission_mechanism",
    "emission_own_elastic_count",
    "emission_own_inelastic_count",
    "emission_path_elastic_count",
    "emission_path_inelastic_count",
)


SEED_ANALYSIS_POPULATIONS = (
    "se_total",
    "se_single_inelastic_origin",
    "se_multiple_origin",
    "se1_current_rule",
    "se2_current_rule",
    "se_mult_gen1_after_elastic",
    "se_mult_gen1_after_prior_inelastic",
    "se_mult_generation_2plus",
    "se_mult_primary",
)


def _array(data, key, dtype=None):
    if key not in data:
        raise KeyError(f"event archive is missing {key!r}")
    out = np.asarray(data[key], dtype=dtype)
    if out.ndim != 1:
        raise ValueError(f"{key} must be one-dimensional")
    return out


def validate_event_arrays(data) -> int:
    """Validate the event-level arrays and return the emission count."""
    arrays = [_array(data, key) for key in REQUIRED_EVENT_KEYS]
    sizes = {a.size for a in arrays}
    if len(sizes) != 1:
        raise ValueError("event arrays do not have a common length")
    n = sizes.pop()
    own_e = _array(data, "emission_own_elastic_count", np.int64)
    own_i = _array(data, "emission_own_inelastic_count", np.int64)
    path_e = _array(data, "emission_path_elastic_count", np.int64)
    path_i = _array(data, "emission_path_inelastic_count", np.int64)
    if np.any(own_e < 0) or np.any(own_i < 0):
        raise ValueError("own collision counts must be nonnegative")
    if np.any(path_e < own_e) or np.any(path_i < own_i):
        raise ValueError("causal-path counts cannot be smaller than own counts")
    return int(n)


def _require_partition(total, parts, label):
    membership = np.zeros(np.asarray(total).shape, dtype=np.int16)
    for part in parts:
        membership += np.asarray(part, dtype=np.int16)
    if not np.array_equal(membership, np.asarray(total, dtype=np.int16)):
        raise RuntimeError(f"{label} populations do not form an exclusive partition")


def classify_emissions(data, cutoff_ev: float = 50.0):
    """Return ordered boolean masks for reconfigurable PSF populations.

    A generation-1 secondary inherits the parent's causal counters *after* its
    creation collision.  Consequently ``path - own == (0 elastic, 1 inelastic)``
    identifies birth in the first inelastic collision of an unscattered primary,
    even when the emitted secondary later scatters on its way to the surface.
    """
    n = validate_event_arrays(data)
    energy = _array(data, "emission_energy_ev", float)
    generation = _array(data, "emission_generation", np.int64)
    mechanism = _array(data, "emission_mechanism", np.int64)
    own_e = _array(data, "emission_own_elastic_count", np.int64)
    own_i = _array(data, "emission_own_inelastic_count", np.int64)
    path_e = _array(data, "emission_path_elastic_count", np.int64)
    path_i = _array(data, "emission_path_inelastic_count", np.int64)

    all_emitted = np.ones(n, dtype=bool)
    direct = mechanism == 2
    transport = ~direct
    se = transport & (energy <= float(cutoff_ev))
    bse = transport & (energy > float(cutoff_ev))

    inherited_e = path_e - own_e
    inherited_i = path_i - own_i
    se_single_origin = (
        se
        & (generation == 1)
        & (inherited_e == 0)
        & (inherited_i == 1)
    )
    se_strict = se_single_origin & (own_e == 0) & (own_i == 0)
    se_multiple = se & ~se_single_origin
    se_mult_gen1_elastic = se_multiple & (generation == 1) & (inherited_e >= 1)
    se_mult_gen1_prior_i = (
        se_multiple
        & (generation == 1)
        & (inherited_e == 0)
        & (inherited_i >= 2)
    )
    se_mult_gen2plus = se_multiple & (generation >= 2)
    se_mult_primary = se_multiple & (generation == 0)
    se_mult_assigned = (
        se_mult_gen1_elastic
        | se_mult_gen1_prior_i
        | se_mult_gen2plus
        | se_mult_primary
    )
    se_mult_other = se_multiple & ~se_mult_assigned

    se1 = se & (generation == 1) & (own_i == 0)
    se2 = se & ~se1
    se1_single = se1 & se_single_origin
    se1_multiple = se1 & se_multiple
    se2_single = se2 & se_single_origin
    se2_multiple = se2 & se_multiple

    primary_bse = bse & (generation == 0)
    bse_single = primary_bse & (path_e == 1) & (path_i == 0)
    bse_plural = primary_bse & (path_e >= 2) & (path_i == 0)
    bse_mixed = primary_bse & (path_e >= 1) & (path_i >= 1)
    bse_other = bse & ~(bse_single | bse_plural | bse_mixed)
    bse_other_cascade = bse_other & (generation >= 1)
    bse_other_primary_i = (
        bse_other & (generation == 0) & (path_e == 0) & (path_i >= 1)
    )
    bse_other_primary_none = (
        bse_other & (generation == 0) & (path_e == 0) & (path_i == 0)
    )
    bse_other_assigned = (
        bse_other_cascade | bse_other_primary_i | bse_other_primary_none
    )
    bse_other_unclassified = bse_other & ~bse_other_assigned

    combined_single = se_single_origin | bse_single
    combined_multiple = transport & ~combined_single

    masks = OrderedDict(
        (
            ("all_emitted", all_emitted),
            ("direct_barrier_reflection", direct),
            ("transport_total", transport),
            ("se_total", se),
            ("se_single_inelastic_origin", se_single_origin),
            ("se_strict_single_inelastic", se_strict),
            ("se_multiple_origin", se_multiple),
            ("se_mult_gen1_after_elastic", se_mult_gen1_elastic),
            ("se_mult_gen1_after_prior_inelastic", se_mult_gen1_prior_i),
            ("se_mult_generation_2plus", se_mult_gen2plus),
            ("se_mult_primary", se_mult_primary),
            ("se_mult_other", se_mult_other),
            ("se1_current_rule", se1),
            ("se1_single_origin", se1_single),
            ("se1_multiple_origin", se1_multiple),
            ("se2_current_rule", se2),
            ("se2_single_origin", se2_single),
            ("se2_multiple_origin", se2_multiple),
            ("bse_total", bse),
            ("bse_single_elastic", bse_single),
            ("bse_plural_elastic_only", bse_plural),
            ("bse_elastic_plus_inelastic", bse_mixed),
            ("bse_other", bse_other),
            ("bse_other_high_energy_cascade", bse_other_cascade),
            ("bse_other_primary_inelastic_only", bse_other_primary_i),
            ("bse_other_primary_no_collision", bse_other_primary_none),
            ("bse_other_unclassified", bse_other_unclassified),
            ("combined_single", combined_single),
            ("combined_multiple", combined_multiple),
        )
    )

    _require_partition(se, (se_single_origin, se_multiple), "SE origin")
    _require_partition(
        se_multiple,
        (
            se_mult_gen1_elastic,
            se_mult_gen1_prior_i,
            se_mult_gen2plus,
            se_mult_primary,
            se_mult_other,
        ),
        "SE multiple-origin",
    )
    _require_partition(se, (se1, se2), "current SE1/SE2")
    _require_partition(
        se, (se1_single, se1_multiple, se2_single, se2_multiple),
        "SE origin/current-rule cross",
    )
    _require_partition(bse, (bse_single, bse_plural, bse_mixed, bse_other), "BSE")
    _require_partition(
        bse_other,
        (
            bse_other_cascade,
            bse_other_primary_i,
            bse_other_primary_none,
            bse_other_unclassified,
        ),
        "BSE other",
    )
    _require_partition(
        transport, (combined_single, combined_multiple), "combined transport"
    )
    return masks


def emission_xy_nm(data, beam_sigma_nm: float = 0.0, seed: int = 20260921):
    """Return surface positions, optionally convolved with a 2D Gaussian beam.

    The raw transport always represents a point impact at (0, 0).  For a plane,
    translational invariance makes adding independent Gaussian launch offsets in
    post-processing exactly equivalent to simulating a finite Gaussian spot.
    """
    sigma = float(beam_sigma_nm)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("beam_sigma_nm must be finite and nonnegative")
    x = 0.1 * _array(data, "emission_x_angstrom", float)
    y = 0.1 * _array(data, "emission_y_angstrom", float)
    if sigma > 0.0 and x.size:
        rng = np.random.default_rng(int(seed))
        x = x + rng.normal(0.0, sigma, x.size)
        y = y + rng.normal(0.0, sigma, y.size)
    return x, y


def birth_radius_depth_nm(data):
    """Return electron birth radius and depth relative to the point impact."""
    n = validate_event_arrays(data)
    birth_x = _array(data, "emission_birth_x_angstrom", float)
    birth_y = _array(data, "emission_birth_y_angstrom", float)
    birth_z = _array(data, "emission_birth_z_angstrom", float)
    if birth_x.size != n or birth_y.size != n or birth_z.size != n:
        raise ValueError("birth-position arrays do not match the emission count")
    if not (
        np.all(np.isfinite(birth_x))
        and np.all(np.isfinite(birth_y))
        and np.all(np.isfinite(birth_z))
    ):
        raise ValueError("birth-position arrays must be finite")
    depth_nm = 0.1 * birth_z
    if np.any(depth_nm < -1.0e-6):
        raise ValueError("birth depth is negative under the z>0 solid convention")
    return 0.1 * np.hypot(birth_x, birth_y), np.maximum(depth_nm, 0.0)


def seed_radius_depth_summary(data, masks, populations=SEED_ANALYSIS_POPULATIONS):
    """Summarize where emitted SE populations were created inside the solid."""
    birth_radius, birth_depth = birth_radius_depth_nm(data)
    se_count = int(np.count_nonzero(masks["se_total"]))
    rows = []
    for name in populations:
        if name not in masks:
            raise KeyError(f"population masks are missing {name!r}")
        mask = np.asarray(masks[name], dtype=bool)
        if mask.shape != birth_radius.shape:
            raise ValueError(f"mask {name!r} does not match birth-position arrays")
        radius = birth_radius[mask]
        depth = birth_depth[mask]
        count = int(radius.size)
        correlation = float("nan")
        if count >= 2:
            log_radius = np.log1p(radius)
            log_depth = np.log1p(depth)
            if np.std(log_radius) > 0.0 and np.std(log_depth) > 0.0:
                correlation = float(np.corrcoef(log_radius, log_depth)[0, 1])

        def fraction(condition):
            return float(np.mean(condition)) if count else float("nan")

        rows.append(
            {
                "population": name,
                "description": CLASSIFICATION_DESCRIPTIONS[name],
                "count": count,
                "fraction_of_se": count / float(se_count) if se_count else float("nan"),
                "fraction_birth_radius_zero": fraction(radius == 0.0),
                "mean_birth_radius_nm": float(np.mean(radius)) if count else float("nan"),
                "r50_birth_nm": _quantile_or_nan(radius, 0.50),
                "r90_birth_nm": _quantile_or_nan(radius, 0.90),
                "r95_birth_nm": _quantile_or_nan(radius, 0.95),
                "mean_birth_depth_nm": float(np.mean(depth)) if count else float("nan"),
                "z50_birth_nm": _quantile_or_nan(depth, 0.50),
                "z90_birth_nm": _quantile_or_nan(depth, 0.90),
                "z95_birth_nm": _quantile_or_nan(depth, 0.95),
                "fraction_birth_radius_le_1nm": fraction(radius <= 1.0),
                "fraction_birth_radius_ge_100nm": fraction(radius >= 100.0),
                "fraction_birth_depth_le_1nm": fraction(depth <= 1.0),
                "fraction_birth_depth_le_2nm": fraction(depth <= 2.0),
                "fraction_shallow_le_2nm_and_far_ge_100nm": fraction(
                    (depth <= 2.0) & (radius >= 100.0)
                ),
                "log1p_radius_depth_pearson": correlation,
            }
        )
    return rows


def radial_bin_edges(r_min_nm: float, r_max_nm: float, n_bins: int):
    """Return zero plus geometrically spaced positive annular edges."""
    r_min = float(r_min_nm)
    r_max = float(r_max_nm)
    n_bins = int(n_bins)
    if not np.isfinite(r_min) or r_min <= 0.0:
        raise ValueError("r_min_nm must be finite and > 0")
    if not np.isfinite(r_max) or r_max <= r_min:
        raise ValueError("r_max_nm must be finite and > r_min_nm")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    return np.concatenate(([0.0], np.geomspace(r_min, r_max, n_bins)))


def radial_profiles(radius_nm, masks, edges_nm, n_primaries: int):
    """Compute annular PSFs in electrons / nm^2 / incident primary."""
    radius = np.asarray(radius_nm, dtype=float)
    edges = np.asarray(edges_nm, dtype=float)
    n_primaries = int(n_primaries)
    if radius.ndim != 1:
        raise ValueError("radius_nm must be one-dimensional")
    if n_primaries < 1:
        raise ValueError("n_primaries must be positive")
    if edges.ndim != 1 or edges.size < 3 or edges[0] != 0.0:
        raise ValueError("edges_nm must start at zero and define at least two bins")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("edges_nm must be strictly increasing")

    inner = edges[:-1]
    outer = edges[1:]
    area = np.pi * (outer * outer - inner * inner)
    center = (2.0 / 3.0) * (
        (outer ** 3 - inner ** 3) / (outer * outer - inner * inner)
    )
    out = OrderedDict(
        (
            ("radius_inner_nm", inner),
            ("radius_outer_nm", outer),
            ("radius_center_nm", center),
            ("annulus_area_nm2", area),
        )
    )
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != radius.shape:
            raise ValueError(f"mask {name!r} does not match radius array")
        count = np.histogram(radius[mask], bins=edges)[0].astype(np.int64)
        scale = area * float(n_primaries)
        out[f"count__{name}"] = count
        out[f"psf__{name}"] = count / scale
        out[f"poisson_sem__{name}"] = np.sqrt(count) / scale
        out[f"relative_poisson_sem__{name}"] = np.divide(
            1.0,
            np.sqrt(count),
            out=np.full(count.shape, np.nan, dtype=float),
            where=count > 0,
        )
    return out


def _quantile_or_nan(values, q):
    values = np.asarray(values, dtype=float)
    return float(np.quantile(values, q)) if values.size else float("nan")


def population_summary(data, radius_nm, masks, n_primaries: int):
    """Return one portable row per population."""
    radius = np.asarray(radius_nm, dtype=float)
    energy = _array(data, "emission_energy_ev", float)
    birth_z = np.asarray(
        data.get("emission_birth_z_angstrom", np.full(radius.size, np.nan)),
        dtype=float,
    )
    n_primaries = int(n_primaries)
    family = {
        "se_single_inelastic_origin": "se_total",
        "se_strict_single_inelastic": "se_total",
        "se_multiple_origin": "se_total",
        "se_mult_gen1_after_elastic": "se_multiple_origin",
        "se_mult_gen1_after_prior_inelastic": "se_multiple_origin",
        "se_mult_generation_2plus": "se_multiple_origin",
        "se_mult_primary": "se_multiple_origin",
        "se_mult_other": "se_multiple_origin",
        "se1_current_rule": "se_total",
        "se1_single_origin": "se1_current_rule",
        "se1_multiple_origin": "se1_current_rule",
        "se2_current_rule": "se_total",
        "se2_single_origin": "se2_current_rule",
        "se2_multiple_origin": "se2_current_rule",
        "bse_single_elastic": "bse_total",
        "bse_plural_elastic_only": "bse_total",
        "bse_elastic_plus_inelastic": "bse_total",
        "bse_other": "bse_total",
        "bse_other_high_energy_cascade": "bse_other",
        "bse_other_primary_inelastic_only": "bse_other",
        "bse_other_primary_no_collision": "bse_other",
        "bse_other_unclassified": "bse_other",
        "combined_single": "transport_total",
        "combined_multiple": "transport_total",
        "direct_barrier_reflection": "all_emitted",
    }
    counts = {name: int(np.count_nonzero(mask)) for name, mask in masks.items()}
    rows = []
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        parent = family.get(name, name)
        denom = counts.get(parent, counts[name])
        rows.append(
            {
                "population": name,
                "description": CLASSIFICATION_DESCRIPTIONS[name],
                "count": counts[name],
                "yield_per_primary": counts[name] / float(n_primaries),
                "yield_poisson_sem": np.sqrt(counts[name]) / float(n_primaries),
                "relative_counting_sem": (
                    1.0 / np.sqrt(counts[name]) if counts[name] else float("nan")
                ),
                "fraction_of_family": (
                    counts[name] / float(denom) if denom else float("nan")
                ),
                "mean_energy_ev": (
                    float(np.mean(energy[mask])) if counts[name] else float("nan")
                ),
                "r50_nm": _quantile_or_nan(radius[mask], 0.50),
                "r90_nm": _quantile_or_nan(radius[mask], 0.90),
                "r95_nm": _quantile_or_nan(radius[mask], 0.95),
                "mean_birth_depth_nm": (
                    float(np.nanmean(0.1 * birth_z[mask]))
                    if counts[name] and np.any(np.isfinite(birth_z[mask]))
                    else float("nan")
                ),
            }
        )
    return rows


def write_rows_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty row table")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_profile_csv(path, profiles):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(profiles)
    columns = [np.asarray(profiles[name]) for name in names]
    if len({column.size for column in columns}) != 1:
        raise ValueError("profile columns do not have a common length")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(zip(*columns))
    return path


def archive_metadata(data):
    """Decode metadata_json from a loaded NPZ-like mapping."""
    raw = data.get("metadata_json", np.asarray("{}"))
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _plot_psf_curve(
    ax,
    radius,
    profiles,
    name,
    label,
    color,
    *,
    min_count_per_bin,
    ls="-",
    lw=2.0,
):
    values = np.asarray(profiles[f"psf__{name}"], dtype=float)
    count = np.asarray(profiles[f"count__{name}"], dtype=np.int64)
    reliable = (
        (count >= int(min_count_per_bin)) & np.isfinite(values) & (values > 0.0)
    )
    plotted = np.where(reliable, values, np.nan)
    if np.any(reliable):
        ax.plot(radius, plotted, label=label, color=color, ls=ls, lw=lw)
    elif int(np.sum(count)) > 0:
        ax.plot(
            [], [], label=f"{label} (sparse; n={int(np.sum(count))})",
            color=color, ls=ls, lw=lw,
        )


def _style_psf_axis(ax, panel_title):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Emission radius r (nm)")
    ax.set_ylabel(r"Radial PSF (electrons nm$^{-2}$ primary$^{-1}$)")
    ax.set_title(panel_title, loc="left", fontweight="bold")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, fontsize=8)


def plot_radial_psf(
    path,
    profiles,
    masks,
    *,
    n_primaries: int,
    title: str,
    beam_sigma_nm: float = 0.0,
    min_count_per_bin: int = 10,
):
    """Write the four-panel population-resolved radial PSF figure."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    radius = np.asarray(profiles["radius_center_nm"], dtype=float)

    colors = {
        "total": "#161616",
        "single": "#1261a0",
        "strict": "#00a6a6",
        "multiple": "#d1495b",
        "plural": "#e59500",
        "mixed": "#7a5195",
        "other": "#777777",
        "reflection": "#9c27b0",
    }

    min_count_per_bin = int(min_count_per_bin)
    if min_count_per_bin < 1:
        raise ValueError("min_count_per_bin must be positive")

    def curve(ax, name, label, color, ls="-", lw=2.0):
        _plot_psf_curve(
            ax, radius, profiles, name, label, color,
            min_count_per_bin=min_count_per_bin, ls=ls, lw=lw,
        )

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6), constrained_layout=True)
    ax = axes[0, 0]
    curve(ax, "se_total", "All SE (E <= cutoff)", colors["total"], lw=2.4)
    curve(ax, "se_single_inelastic_origin", "Single-inelastic origin", colors["single"])
    curve(ax, "se_strict_single_inelastic", "Strict ballistic subset", colors["strict"], "--", 1.6)
    curve(ax, "se_multiple_origin", "Other / multiple origin", colors["multiple"])
    _style_psf_axis(ax, "(a) Secondary-electron populations")

    ax = axes[0, 1]
    curve(ax, "bse_total", "All BSE (E > cutoff)", colors["total"], lw=2.4)
    curve(ax, "bse_single_elastic", "Exactly 1 elastic, 0 inelastic", colors["single"])
    curve(ax, "bse_plural_elastic_only", ">=2 elastic, 0 inelastic", colors["plural"])
    curve(ax, "bse_elastic_plus_inelastic", "Elastic + inelastic", colors["mixed"])
    curve(ax, "bse_other", "Other BSE", colors["other"], ":", 1.6)
    _style_psf_axis(ax, "(b) Backscattered-electron populations")

    ax = axes[1, 0]
    curve(ax, "transport_total", "All transported emissions", colors["total"], lw=2.4)
    curve(ax, "combined_single", "Combined single", colors["single"])
    curve(ax, "combined_multiple", "Combined multiple", colors["multiple"])
    if np.count_nonzero(masks["direct_barrier_reflection"]):
        curve(ax, "direct_barrier_reflection", "Direct barrier reflection", colors["reflection"], ":", 1.6)
    _style_psf_axis(ax, "(c) Combined populations")

    ax = axes[1, 1]
    for name, label, color, ls in (
        ("se_single_inelastic_origin", "SE single-inelastic origin", colors["single"], "-"),
        ("se_multiple_origin", "SE other / multiple", colors["multiple"], "-"),
        ("bse_single_elastic", "BSE single elastic", colors["strict"], "--"),
        ("bse_plural_elastic_only", "BSE plural elastic only", colors["plural"], "--"),
        ("bse_elastic_plus_inelastic", "BSE elastic + inelastic", colors["mixed"], "-."),
    ):
        count = np.asarray(profiles[f"count__{name}"], dtype=float)
        total = float(np.sum(count))
        if total > 0.0:
            ax.plot(radius, np.cumsum(count) / total, label=label, color=color, ls=ls, lw=1.9)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Emission radius r (nm)")
    ax.set_ylabel("Encircled fraction of population")
    ax.set_title("(d) Population radial extent", loc="left", fontweight="bold")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, fontsize=8)

    beam_note = "point beam" if float(beam_sigma_nm) == 0.0 else f"Gaussian convolution sigma={float(beam_sigma_nm):g} nm"
    fig.suptitle(
        f"{title}\nN={int(n_primaries):,} incident primaries; {beam_note}; "
        f"PSF bins shown for count >= {min_count_per_bin}",
        fontsize=15,
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_mechanism_detail(
    path,
    profiles,
    *,
    n_primaries: int,
    title: str,
    beam_sigma_nm: float = 0.0,
    min_count_per_bin: int = 10,
):
    """Write refined SE1/SE2, SE-lineage, and BSE-other PSF panels."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    radius = np.asarray(profiles["radius_center_nm"], dtype=float)
    min_count_per_bin = int(min_count_per_bin)
    if min_count_per_bin < 1:
        raise ValueError("min_count_per_bin must be positive")

    colors = {
        "total": "#161616",
        "blue": "#1261a0",
        "red": "#d1495b",
        "orange": "#e59500",
        "purple": "#7a5195",
        "green": "#2a9d8f",
        "gray": "#777777",
    }

    def curve(ax, name, label, color, ls="-", lw=2.0):
        _plot_psf_curve(
            ax, radius, profiles, name, label, color,
            min_count_per_bin=min_count_per_bin, ls=ls, lw=lw,
        )

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6), constrained_layout=True)

    ax = axes[0, 0]
    curve(ax, "se_total", "All SE", colors["total"], lw=2.4)
    curve(ax, "se1_current_rule", "Current SE1", colors["blue"])
    curve(ax, "se2_current_rule", "Current SE2", colors["red"])
    _style_psf_axis(ax, "(a) Current imaging SE1/SE2")

    ax = axes[0, 1]
    curve(ax, "se_multiple_origin", "All multiple-origin SE", colors["total"], lw=2.4)
    curve(
        ax, "se_mult_gen1_after_elastic", "Generation 1 after parent elastic",
        colors["orange"],
    )
    curve(
        ax, "se_mult_gen1_after_prior_inelastic",
        "Generation 1 after prior inelastic", colors["green"], "--",
    )
    curve(
        ax, "se_mult_generation_2plus", "Generation >=2",
        colors["purple"],
    )
    curve(ax, "se_mult_primary", "Low-energy primary", colors["gray"], ":", 1.6)
    _style_psf_axis(ax, "(b) Multiple-origin SE decomposition")

    ax = axes[1, 0]
    curve(ax, "bse_other", "All other BSE", colors["total"], lw=2.4)
    curve(
        ax, "bse_other_high_energy_cascade", "High-energy cascade",
        colors["purple"],
    )
    curve(
        ax, "bse_other_primary_inelastic_only", "Primary, inelastic only",
        colors["orange"], "--",
    )
    curve(
        ax, "bse_other_primary_no_collision", "Primary, no bulk collision",
        colors["gray"], ":", 1.6,
    )
    curve(
        ax, "bse_other_unclassified", "Unclassified residual",
        colors["red"], ":", 1.6,
    )
    _style_psf_axis(ax, "(c) Other-BSE decomposition")

    ax = axes[1, 1]
    for name, label, color, ls in (
        ("se1_current_rule", "Current SE1", colors["blue"], "-"),
        ("se2_current_rule", "Current SE2", colors["red"], "-"),
        (
            "se_single_inelastic_origin", "Single-inelastic origin",
            colors["green"], "--",
        ),
        ("se_multiple_origin", "Multiple origin", colors["purple"], "--"),
    ):
        count = np.asarray(profiles[f"count__{name}"], dtype=float)
        total = float(np.sum(count))
        if total > 0.0:
            ax.plot(
                radius, np.cumsum(count) / total, label=label,
                color=color, ls=ls, lw=1.9,
            )
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Emission radius r (nm)")
    ax.set_ylabel("Encircled fraction of population")
    ax.set_title("(d) Current-rule versus origin-rule extent", loc="left", fontweight="bold")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, fontsize=8)

    beam_note = "point beam" if float(beam_sigma_nm) == 0.0 else f"Gaussian convolution sigma={float(beam_sigma_nm):g} nm"
    fig.suptitle(
        f"{title}\nN={int(n_primaries):,}; {beam_note}; "
        f"PSF bins shown for count >= {min_count_per_bin}",
        fontsize=15,
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_seed_radius_depth(
    path,
    data,
    masks,
    *,
    n_primaries: int,
    title: str,
    n_bins: int = 90,
):
    """Write four seed-radius-versus-depth density panels for SE populations."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_bins = int(n_bins)
    if n_bins < 20:
        raise ValueError("seed density requires at least 20 bins")

    birth_radius, birth_depth = birth_radius_depth_nm(data)
    se_mask = np.asarray(masks["se_total"], dtype=bool)
    if not np.any(se_mask):
        raise ValueError("seed density cannot be plotted without emitted SEs")

    radius_scale = 0.01
    depth_scale = 0.01

    def forward(values, scale):
        return np.log10(1.0 + np.asarray(values, dtype=float) / scale)

    u_all = forward(birth_radius[se_mask], radius_scale)
    v_all = forward(birth_depth[se_mask], depth_scale)
    u_edges = np.linspace(0.0, max(float(np.max(u_all)) * 1.001, 1.0), n_bins + 1)
    v_edges = np.linspace(0.0, max(float(np.max(v_all)) * 1.001, 1.0), n_bins + 1)

    panels = (
        ("se_single_inelastic_origin", "(a) Single-inelastic-origin SE"),
        ("se_multiple_origin", "(b) Multiple-origin SE"),
        ("se1_current_rule", "(c) Current imaging SE1"),
        ("se2_current_rule", "(d) Current imaging SE2"),
    )
    histograms = []
    maximum = 1
    for name, _ in panels:
        mask = np.asarray(masks[name], dtype=bool)
        hist, _, _ = np.histogram2d(
            forward(birth_radius[mask], radius_scale),
            forward(birth_depth[mask], depth_scale),
            bins=(u_edges, v_edges),
        )
        histograms.append(hist)
        maximum = max(maximum, int(np.max(hist)))

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6), constrained_layout=True)
    mesh = None
    for ax, (name, panel_title), hist in zip(axes.ravel(), panels, histograms):
        masked = np.ma.masked_less_equal(hist.T, 0.0)
        mesh = ax.pcolormesh(
            u_edges, v_edges, masked, shading="auto", cmap="magma",
            norm=LogNorm(vmin=1.0, vmax=float(maximum)), rasterized=True,
        )
        mask = np.asarray(masks[name], dtype=bool)
        count = int(np.count_nonzero(mask))
        shallow_far = (
            float(np.mean((birth_depth[mask] <= 2.0) & (birth_radius[mask] >= 100.0)))
            if count else float("nan")
        )
        ax.axvline(forward(100.0, radius_scale), color="white", ls="--", lw=1.0, alpha=0.8)
        ax.axhline(forward(2.0, depth_scale), color="white", ls="--", lw=1.0, alpha=0.8)
        ax.set_title(panel_title, loc="left", fontweight="bold")
        ax.text(
            0.98, 0.96,
            f"n={count:,}\nshallow and far={100.0 * shallow_far:.1f}%" if count else "n=0",
            transform=ax.transAxes, ha="right", va="top", color="white", fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 3},
        )
        radius_ticks = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
        depth_ticks = [0.0, 0.1, 1.0, 10.0, 100.0]
        radius_ticks = [value for value in radius_ticks if value <= float(np.max(birth_radius[se_mask])) * 1.05]
        depth_ticks = [value for value in depth_ticks if value <= float(np.max(birth_depth[se_mask])) * 1.05]
        ax.set_xticks([forward(value, radius_scale) for value in radius_ticks])
        ax.set_xticklabels([f"{value:g}" for value in radius_ticks])
        ax.set_yticks([forward(value, depth_scale) for value in depth_ticks])
        ax.set_yticklabels([f"{value:g}" for value in depth_ticks])
        ax.set_xlabel("Birth radius from impact (nm)")
        ax.set_ylabel("Birth depth (nm)")

    if mesh is not None:
        colorbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02)
        colorbar.set_label("Emissions per bin")
    fig.suptitle(
        f"{title}\nN={int(n_primaries):,}; intrinsic point-response birth coordinates; "
        "log-like axes include zero\nDashed guides: radius 100 nm, depth 2 nm",
        fontsize=14,
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path
