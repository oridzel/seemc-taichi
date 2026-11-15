from __future__ import annotations

import math


def require_reference():
    try:
        from seemc_imaging.geometry import Plane
        from seemc_imaging.transport import MCConfig, SEEMC, Sample
    except ImportError as exc:
        raise RuntimeError(
            "Reference comparison requires seemc-imaging>=0.7.4. "
            "Install this package with the `reference` extra."
        ) from exc
    return Plane, MCConfig, SEEMC, Sample


def run_reference_plane(database_path, material, energy_ev, angle_deg, n_primaries,
                        *, config=None, seed=12345, workers=1):
    Plane, MCConfig, SEEMC, _ = require_reference()
    # Momentum unit from the database entry, never the MCConfig default.
    from dataclasses import replace
    from .q_units import resolve_q_unit
    config = config or MCConfig()
    unit = resolve_q_unit(database_path, material)
    if getattr(config, "q_unit", None) != unit:
        config = replace(config, q_unit=unit)
    alpha = math.radians(float(angle_deg))
    vacuum = (math.sin(alpha), 0.0, math.cos(alpha))
    outward = (0.0, 0.0, -1.0)
    model = SEEMC(
        [float(energy_ev)], material, alpha, int(n_primaries),
        db_path=str(database_path), config=config, seed=int(seed), history=False,
        geometry=Plane(), vacuum_direction=vacuum, surface_normal=outward,
    ).run_simulation(
        use_parallel=int(workers) > 1,
        workers=int(workers), progress=False, verbose=False,
    )
    return model
