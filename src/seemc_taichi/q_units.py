"""Momentum units of the material database, and the one place where a
reference material sample is created.

Every seemc-taichi loader builds its tables from a ``seemc_imaging`` ``Sample``.
The momentum-transfer grid of a database entry can be stored in inverse Bohr
radii (``a0^-1``) or inverse Angstrom (``A^-1``), and ``MCConfig`` defaults to
``a0^-1`` whatever the database says.  Reading an ``A^-1`` database with the
default therefore runs without any error and gives wrong inelastic kinematics
(and wrong secondary yields).

``make_reference_sample`` removes the default.  The unit is taken from the
database entry (key ``q_unit``); an explicit request must agree with it; and a
database entry without the key is refused unless the unit is given
explicitly.  Use ``tools/stamp_q_unit.py`` to add the key to an old database.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

Q_UNITS = ("a0^-1", "A^-1")
_ALIASES = {
    "a0^-1": "a0^-1", "a0-1": "a0^-1", "1/a0": "a0^-1", "bohr^-1": "a0^-1",
    "1/bohr": "a0^-1", "a.u.": "a0^-1", "au": "a0^-1",
    "a^-1": "A^-1", "a-1": "A^-1", "1/a": "A^-1", "angstrom^-1": "A^-1",
    "1/angstrom": "A^-1", "å^-1": "A^-1", "1/å": "A^-1", "å⁻¹": "A^-1", "a⁻¹": "A^-1",
}


class QUnitError(RuntimeError):
    pass


def normalise_q_unit(unit) -> str:
    """Canonical spelling of a momentum unit, or raise."""
    if unit in Q_UNITS:           # exact, case-sensitive: "A^-1" vs "a0^-1"
        return unit
    key = str(unit).strip().lower().replace(" ", "")
    if key in _ALIASES:
        return _ALIASES[key]
    raise QUnitError(f"unknown momentum unit {unit!r}; use one of {Q_UNITS}")


def _entries(database):
    if isinstance(database, dict):
        if "name" in database and not all(isinstance(v, dict) for v in database.values()):
            return [database]
        return [dict(v, name=v.get("name", k)) for k, v in database.items() if isinstance(v, dict)]
    return [e for e in database if isinstance(e, dict)]


@lru_cache(maxsize=8)
def _database_units(path: str, mtime_ns: int, size: int) -> dict:
    with open(path, "rb") as f:
        database = pickle.load(f)
    out = {}
    for entry in _entries(database):
        name = entry.get("name")
        if name is None:
            continue
        unit = entry.get("q_unit")
        out[str(name)] = None if unit is None else normalise_q_unit(unit)
    return out


def database_q_unit(database_path, material) -> str | None:
    """Unit recorded in the database entry of ``material`` (None if absent)."""
    path = Path(database_path)
    st = path.stat()
    units = _database_units(str(path), st.st_mtime_ns, st.st_size)
    if str(material) not in units:
        raise QUnitError(
            f"material {material!r} not found in {path} (entries: {sorted(units)})"
        )
    return units[str(material)]


def resolve_q_unit(database_path, material, q_unit=None) -> str:
    """The momentum unit to use for ``material``; never a silent default."""
    recorded = database_q_unit(database_path, material)
    requested = None if q_unit is None else normalise_q_unit(q_unit)
    if requested is None:
        env = os.environ.get("SEEMC_Q_UNIT")
        if env and recorded is None:
            requested = normalise_q_unit(env)
    if recorded is not None:
        if requested is not None and requested != recorded:
            raise QUnitError(
                f"{material}: database {Path(database_path).name} records q_unit="
                f"{recorded!r} but {requested!r} was requested. Fix the request or the database."
            )
        return recorded
    if requested is not None:
        return requested
    raise QUnitError(
        f"{material}: the entry in {Path(database_path).name} has no 'q_unit' key, so the "
        f"momentum unit is unknown. Pass q_unit explicitly (--q-unit on the command line, "
        f"or SEEMC_Q_UNIT in the environment), or record it once with "
        f"tools/stamp_q_unit.py. Units: {Q_UNITS}."
    )


def make_reference_sample(database_path, material="Si", config=None, q_unit=None):
    """``seemc_imaging`` Sample with the momentum unit of the database entry."""
    try:
        from seemc_imaging.transport import MCConfig, Sample
    except ImportError as exc:
        raise RuntimeError(
            "Install seemc-imaging or this package with the reference extra"
        ) from exc
    unit = resolve_q_unit(database_path, material, q_unit)
    cfg = config or MCConfig()
    if not hasattr(cfg, "q_unit"):
        raise QUnitError(
            "this seemc-imaging version has no MCConfig.q_unit; it cannot read a "
            f"database in {unit!r}. Update seemc-imaging."
        )
    if cfg.q_unit != unit:
        cfg = replace(cfg, q_unit=unit)
    sample = Sample(material, db_path=str(database_path), config=cfg)
    used = getattr(getattr(sample, "cfg", cfg), "q_unit", None)
    if used != unit:
        raise QUnitError(
            f"{material}: requested q_unit={unit!r} but the sample was built with {used!r}"
        )
    return sample
