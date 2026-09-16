from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class YieldStats:
    sey: float
    bsey: float
    tey: float
    sey_sem: float | None = None
    bsey_sem: float | None = None


def compare_yields(reference: YieldStats, candidate: YieldStats, sigma_floor=1e-12):
    """Return absolute/relative differences and SEM-normalized pulls when available."""
    out = {}
    for key in ("sey", "bsey", "tey"):
        a = float(getattr(reference, key))
        b = float(getattr(candidate, key))
        out[f"{key}_delta"] = b - a
        out[f"{key}_relative"] = (b - a) / a if a != 0 else math.nan

    for key in ("sey", "bsey"):
        sr = getattr(reference, f"{key}_sem")
        sc = getattr(candidate, f"{key}_sem")
        if sr is not None and sc is not None:
            denom = max(math.hypot(float(sr), float(sc)), sigma_floor)
            out[f"{key}_pull"] = (float(getattr(candidate, key)) - float(getattr(reference, key))) / denom
    return out


def histogram_distance(a, b, bins=100, value_range=None):
    """Total-variation distance between normalized 1-D histograms."""
    ha, edges = np.histogram(np.asarray(a), bins=bins, range=value_range, density=False)
    hb, _ = np.histogram(np.asarray(b), bins=edges, density=False)
    if ha.sum() == 0 or hb.sum() == 0:
        return math.nan
    pa = ha / ha.sum()
    pb = hb / hb.sum()
    return 0.5 * float(np.abs(pa - pb).sum())
