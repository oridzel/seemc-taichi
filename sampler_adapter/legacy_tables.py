"""CSV export functions copied unchanged from the supplied seemc-imaging source."""

from __future__ import annotations

import math, os, csv

from pathlib import Path

from typing import Sequence

import numpy as np

SEY_FILENAME = "SEYFromPlane_SEVaccum_t0nmCuFPA.csv"

BSEY_FILENAME = "BSEYFromPlane_SEVaccum_t0nmCuFPA.csv"

SE_ENERGY_FILENAME = "SEeEFromPlaneSampler_SEVaccum_t0nmCuFPA.csv"

BSE_ENERGY_FILENAME = "BSEeEFromPlaneSampler_SEVaccum_t0nmCuFPA.csv"

SE_THETA_FILENAME = "SEThetaFromPlaneSampler_uncoatedCuFPA.csv"

BSE_THETA_FILENAME = "BSEThetaFromPlaneSampler_uncoatedCuFPA.csv"

def cosine_probability_grid(size: int = 513) -> np.ndarray:
    """Return a common inverse-CDF grid with extra resolution in both tails."""
    size = int(size)
    if size < 3:
        raise ValueError("probability-grid size must be at least 3")
    u = np.linspace(0.0, 1.0, size)
    return 0.5 * (1.0 - np.cos(np.pi * u))

def _number_token(value: float) -> str:
    token = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return token.replace("-", "m").replace(".", "p")

def angle_directory_name(angle_deg: float) -> str:
    return f"alpha_{_number_token(angle_deg)}deg"

def _inverse_cdf(samples: np.ndarray, probabilities: np.ndarray,
                 lower: float, upper: float) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        raise ValueError(
            "cannot construct an inverse CDF from zero emissions; increase "
            "the number of primaries for this angle-energy case"
        )
    quantiles = np.quantile(samples, probabilities, method="linear")
    quantiles = np.clip(quantiles, lower, upper)
    quantiles[0] = lower
    quantiles[-1] = upper
    return np.maximum.accumulate(quantiles)

def _atomic_csv(path: Path, description_lines: Sequence[str], header,
                rows: Iterable[Sequence[object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for line in description_lines:
            writer.writerow([line])
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(temporary, path)

def _format_number(value: float) -> str:
    return format(float(value), ".17g")

def export_angle_tables(directory, cases: Sequence[PlaneSamplerCase],
                        *, material: str, probabilities: np.ndarray) -> None:
    """Write the six legacy-compatible CSV tables for one incidence angle."""
    if not cases:
        raise ValueError("at least one case is required")
    cases = sorted(cases, key=lambda item: item.incident_energy_ev)
    angle = cases[0].incidence_angle_deg
    cutoff = cases[0].energy_cutoff_ev
    for case in cases:
        case.validate()
        if not math.isclose(case.incidence_angle_deg, angle, abs_tol=1.0e-12):
            raise ValueError("all cases in one export must have the same angle")
        if not math.isclose(case.energy_cutoff_ev, cutoff, abs_tol=1.0e-12):
            raise ValueError("all cases in one export must have the same cutoff")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    angle_text = _format_number(angle)

    yield_description = (
        f"{{label}} vs. energy for {material} substrate tilted "
        f"{angle_text} degrees."
    )
    _atomic_csv(
        directory / SEY_FILENAME,
        [yield_description.format(label="SEY")],
        ("beamE (eV)", "SEY"),
        ((_format_number(case.incident_energy_ev), _format_number(case.sey))
         for case in cases),
    )
    _atomic_csv(
        directory / BSEY_FILENAME,
        [yield_description.format(label="BSEY")],
        ("beamE (eV)", "BSEY"),
        ((_format_number(case.incident_energy_ev), _format_number(case.bsey))
         for case in cases),
    )

    second_line_energy = (
        "eE = sampler[T,r] with r uniform on [0,1]; Interpolate this "
        "tabulation to obtain sampler."
    )
    second_line_theta = (
        "theta = sampler[T,r] with r uniform on [0,1]; Interpolate this "
        "tabulation to obtain sampler."
    )

    def distribution_rows(kind: str, value: str):
        for case in cases:
            if kind == "SE":
                samples = case.se_energy_ev if value == "energy" else case.se_theta_deg
                lower = 0.0
                upper = (
                    min(cutoff, case.incident_energy_ev)
                    if value == "energy" else 90.0 + angle
                )
            else:
                samples = case.bse_energy_ev if value == "energy" else case.bse_theta_deg
                lower = cutoff if value == "energy" else 0.0
                upper = case.incident_energy_ev if value == "energy" else 90.0 + angle
            # A zero-yield class has no conditional distribution.  Keep its
            # yield row (SEY/BSEY=0), but omit sampler rows for that energy.
            if samples.size == 0:
                continue
            quantiles = _inverse_cdf(samples, probabilities, lower, upper)
            for probability, quantile in zip(probabilities, quantiles):
                yield (
                    _format_number(case.incident_energy_ev),
                    _format_number(probability),
                    _format_number(quantile),
                )

    for kind, filename in (("SE", SE_ENERGY_FILENAME), ("BSE", BSE_ENERGY_FILENAME)):
        _atomic_csv(
            directory / filename,
            [f"Sampler for emission energy (eE) of {kind} from a planar "
             f"{material} substrate tilted {angle_text} degrees.",
             second_line_energy],
            ("beamE (eV)", "r", "eE (eV)"),
            distribution_rows(kind, "energy"),
        )
    for kind, filename in (("SE", SE_THETA_FILENAME), ("BSE", BSE_THETA_FILENAME)):
        _atomic_csv(
            directory / filename,
            [f"Sampler for beam-relative polar angle of {kind} from a planar "
             f"{material} substrate tilted {angle_text} degrees.",
             second_line_theta],
            ("beamE (eV)", "r", "theta (deg)"),
            distribution_rows(kind, "theta"),
        )

    readme = directory / "readme.txt"
    temporary = readme.with_suffix(".txt.tmp")
    temporary.write_text(
        f"Planar {material} emission samplers at {angle_text} degrees incidence.\n"
        f"SE means emitted energy < {cutoff:g} eV; BSE means emitted energy "
        f">= {cutoff:g} eV.\n"
        "Theta is measured from the beam-back direction (opposite the incident "
        "vacuum ray), not from the sample normal.\n"
        f"Physical polar support is 0 to {90.0 + angle:g} degrees.\n"
        "The legacy theta CSV is a marginal distribution only. For oblique "
        "incidence use the event-level SE/BSE JointFromPlaneSampler NPZ files, "
        "which preserve E, theta, phi, and direction correlations.\n",
        encoding="utf-8",
    )
    os.replace(temporary, readme)
