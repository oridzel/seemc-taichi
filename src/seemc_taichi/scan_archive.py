"""Combined, pickle-free trajectory archive for Taichi line scans.

The layout intentionally mirrors the role of ``RasterTrajectoryArchive`` in
``seemc-imaging``: one file contains the scan coordinates and profiles plus a
recorded subset of cascades for every pixel.  Electron and root IDs are local
to a pixel, so ``(pixel_id, electron_id)`` and ``(pixel_id, root_primary_id)``
are the stable identifiers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ARCHIVE_FORMAT = "seemc-taichi-raster-trajectories-v1"


@dataclass
class RasterTrajectoryArchive:
    metadata: dict
    x_angstrom: np.ndarray
    y_angstrom: np.ndarray
    profile_channels: np.ndarray
    profile_yields: np.ndarray
    profile_sems: np.ndarray
    point_pixel_id: np.ndarray
    point_x_angstrom: np.ndarray
    point_y_angstrom: np.ndarray
    point_z_angstrom: np.ndarray
    point_energy_ev: np.ndarray
    point_time_fs: np.ndarray
    point_electron_id: np.ndarray
    point_parent_id: np.ndarray
    point_root_primary_id: np.ndarray
    point_generation: np.ndarray
    point_inelastic_count: np.ndarray
    point_event: np.ndarray
    point_surface_code: np.ndarray
    point_step: np.ndarray
    emission_pixel_id: np.ndarray
    emission_electron_id: np.ndarray
    emission_root_primary_id: np.ndarray
    emission_energy_ev: np.ndarray
    emission_ux: np.ndarray
    emission_uy: np.ndarray
    emission_uz: np.ndarray
    emission_generation: np.ndarray
    emission_inelastic_count: np.ndarray

    def validate(self):
        self.x_angstrom = np.asarray(self.x_angstrom, dtype=float)
        self.y_angstrom = np.asarray(self.y_angstrom, dtype=float)
        self.profile_channels = np.asarray(self.profile_channels, dtype="U")
        self.profile_yields = np.asarray(self.profile_yields, dtype=float)
        self.profile_sems = np.asarray(self.profile_sems, dtype=float)
        if self.x_angstrom.ndim != 1 or not len(self.x_angstrom):
            raise ValueError("x_angstrom must be a non-empty one-dimensional array")
        if self.y_angstrom.shape != (1,):
            raise ValueError("trajectory animation requires a one-row scan")
        expected_profile_shape = (
            len(self.profile_channels), len(self.y_angstrom), len(self.x_angstrom)
        )
        if self.profile_yields.shape != expected_profile_shape:
            raise ValueError(
                f"profile_yields has shape {self.profile_yields.shape}, expected "
                f"{expected_profile_shape}"
            )
        if self.profile_sems.shape != expected_profile_shape:
            raise ValueError(
                f"profile_sems has shape {self.profile_sems.shape}, expected "
                f"{expected_profile_shape}"
            )

        point_arrays = (
            self.point_pixel_id,
            self.point_x_angstrom,
            self.point_y_angstrom,
            self.point_z_angstrom,
            self.point_energy_ev,
            self.point_time_fs,
            self.point_electron_id,
            self.point_parent_id,
            self.point_root_primary_id,
            self.point_generation,
            self.point_inelastic_count,
            self.point_event,
            self.point_surface_code,
            self.point_step,
        )
        point_length = len(np.asarray(self.point_pixel_id))
        if any(np.asarray(value).ndim != 1 or len(value) != point_length for value in point_arrays):
            raise ValueError("trajectory point arrays must be one-dimensional and equal-length")
        if point_length:
            pixel = np.asarray(self.point_pixel_id, dtype=np.int64)
            if pixel.min() < 0 or pixel.max() >= len(self.x_angstrom):
                raise ValueError("trajectory point pixel IDs are outside the scan")
            times = np.asarray(self.point_time_fs, dtype=float)
            if np.any(~np.isfinite(times)) or np.any(times < 0.0):
                raise ValueError("trajectory times must be finite and non-negative")

        emission_arrays = (
            self.emission_pixel_id,
            self.emission_electron_id,
            self.emission_root_primary_id,
            self.emission_energy_ev,
            self.emission_ux,
            self.emission_uy,
            self.emission_uz,
            self.emission_generation,
            self.emission_inelastic_count,
        )
        emission_length = len(np.asarray(self.emission_pixel_id))
        if any(
            np.asarray(value).ndim != 1 or len(value) != emission_length
            for value in emission_arrays
        ):
            raise ValueError("emission arrays must be one-dimensional and equal-length")
        return self

    @property
    def n_points(self):
        return len(self.point_pixel_id)

    @property
    def recorded_pixels(self):
        return np.unique(np.asarray(self.point_pixel_id, dtype=np.int32))

    @property
    def n_cascades(self):
        if not self.n_points:
            return 0
        pairs = np.column_stack((self.point_pixel_id, self.point_root_primary_id))
        return len(np.unique(pairs, axis=0))

    def save_npz(self, path: str | Path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata)
        metadata["format"] = ARCHIVE_FORMAT
        payload = {
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
            "x_angstrom": np.asarray(self.x_angstrom, dtype=float),
            "y_angstrom": np.asarray(self.y_angstrom, dtype=float),
            "profile_channels": np.asarray(self.profile_channels, dtype="U"),
            "profile_yields": np.asarray(self.profile_yields, dtype=float),
            "profile_sems": np.asarray(self.profile_sems, dtype=float),
            "trajectory_pixel_id": np.asarray(self.point_pixel_id, dtype=np.int32),
            "trajectory_x_angstrom": np.asarray(self.point_x_angstrom, dtype=float),
            "trajectory_y_angstrom": np.asarray(self.point_y_angstrom, dtype=float),
            "trajectory_z_angstrom": np.asarray(self.point_z_angstrom, dtype=float),
            "trajectory_energy_ev": np.asarray(self.point_energy_ev, dtype=float),
            "trajectory_time_fs": np.asarray(self.point_time_fs, dtype=float),
            "trajectory_electron_id": np.asarray(self.point_electron_id, dtype=np.int32),
            "trajectory_parent_id": np.asarray(self.point_parent_id, dtype=np.int32),
            "trajectory_root_primary_id": np.asarray(
                self.point_root_primary_id, dtype=np.int32
            ),
            "trajectory_generation": np.asarray(self.point_generation, dtype=np.int32),
            "trajectory_inelastic_count": np.asarray(
                self.point_inelastic_count, dtype=np.int32
            ),
            "trajectory_event": np.asarray(self.point_event, dtype=np.int32),
            "trajectory_surface_code": np.asarray(
                self.point_surface_code, dtype=np.int32
            ),
            "trajectory_step": np.asarray(self.point_step, dtype=np.int32),
            "emission_pixel_id": np.asarray(self.emission_pixel_id, dtype=np.int32),
            "emission_electron_id": np.asarray(
                self.emission_electron_id, dtype=np.int32
            ),
            "emission_root_primary_id": np.asarray(
                self.emission_root_primary_id, dtype=np.int32
            ),
            "emission_energy_ev": np.asarray(self.emission_energy_ev, dtype=float),
            "emission_ux": np.asarray(self.emission_ux, dtype=float),
            "emission_uy": np.asarray(self.emission_uy, dtype=float),
            "emission_uz": np.asarray(self.emission_uz, dtype=float),
            "emission_generation": np.asarray(
                self.emission_generation, dtype=np.int32
            ),
            "emission_inelastic_count": np.asarray(
                self.emission_inelastic_count, dtype=np.int32
            ),
        }
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load_npz(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as data:
            raw_metadata = np.asarray(data["metadata_json"])
            if raw_metadata.shape == ():
                raw_metadata = raw_metadata.item()
            metadata = json.loads(str(raw_metadata))
            archive_format = metadata.get("format")
            if archive_format != ARCHIVE_FORMAT:
                raise ValueError(
                    f"unsupported trajectory archive format {archive_format!r}; "
                    f"expected {ARCHIVE_FORMAT!r}"
                )
            archive = cls(
                metadata=metadata,
                x_angstrom=np.asarray(data["x_angstrom"], dtype=float),
                y_angstrom=np.asarray(data["y_angstrom"], dtype=float),
                profile_channels=np.asarray(data["profile_channels"], dtype="U"),
                profile_yields=np.asarray(data["profile_yields"], dtype=float),
                profile_sems=np.asarray(data["profile_sems"], dtype=float),
                point_pixel_id=np.asarray(data["trajectory_pixel_id"], dtype=np.int32),
                point_x_angstrom=np.asarray(data["trajectory_x_angstrom"], dtype=float),
                point_y_angstrom=np.asarray(data["trajectory_y_angstrom"], dtype=float),
                point_z_angstrom=np.asarray(data["trajectory_z_angstrom"], dtype=float),
                point_energy_ev=np.asarray(data["trajectory_energy_ev"], dtype=float),
                point_time_fs=np.asarray(data["trajectory_time_fs"], dtype=float),
                point_electron_id=np.asarray(
                    data["trajectory_electron_id"], dtype=np.int32
                ),
                point_parent_id=np.asarray(data["trajectory_parent_id"], dtype=np.int32),
                point_root_primary_id=np.asarray(
                    data["trajectory_root_primary_id"], dtype=np.int32
                ),
                point_generation=np.asarray(
                    data["trajectory_generation"], dtype=np.int32
                ),
                point_inelastic_count=np.asarray(
                    data["trajectory_inelastic_count"], dtype=np.int32
                ),
                point_event=np.asarray(data["trajectory_event"], dtype=np.int32),
                point_surface_code=np.asarray(
                    data["trajectory_surface_code"], dtype=np.int32
                ),
                point_step=np.asarray(data["trajectory_step"], dtype=np.int32),
                emission_pixel_id=np.asarray(data["emission_pixel_id"], dtype=np.int32),
                emission_electron_id=np.asarray(
                    data["emission_electron_id"], dtype=np.int32
                ),
                emission_root_primary_id=np.asarray(
                    data["emission_root_primary_id"], dtype=np.int32
                ),
                emission_energy_ev=np.asarray(data["emission_energy_ev"], dtype=float),
                emission_ux=np.asarray(data["emission_ux"], dtype=float),
                emission_uy=np.asarray(data["emission_uy"], dtype=float),
                emission_uz=np.asarray(data["emission_uz"], dtype=float),
                emission_generation=np.asarray(
                    data["emission_generation"], dtype=np.int32
                ),
                emission_inelastic_count=np.asarray(
                    data["emission_inelastic_count"], dtype=np.int32
                ),
            )
        return archive.validate()


def _empty(dtype):
    return np.empty(0, dtype=dtype)


def archive_from_scan(
    rows: list[dict],
    trajectories: dict[int, dict],
    *,
    metadata: dict,
) -> RasterTrajectoryArchive:
    """Build one animation-ready archive from all recorded scan pixels."""
    if not rows:
        raise ValueError("cannot create a trajectory archive from an empty scan")
    channels = tuple(str(value) for value in metadata.get("profile_channels", ()))
    if not channels:
        raise ValueError("metadata.profile_channels must not be empty")
    x_angstrom = 10.0 * np.asarray([row["x_nm"] for row in rows], dtype=float)
    yields = np.asarray(
        [[row[channel] for row in rows] for channel in channels], dtype=float
    )[:, None, :]
    sems = np.asarray(
        [[row[f"{channel}_sem"] for row in rows] for channel in channels], dtype=float
    )[:, None, :]

    point_specs = {
        "point_x_angstrom": ("trajectory_x_angstrom", float),
        "point_y_angstrom": ("trajectory_y_angstrom", float),
        "point_z_angstrom": ("trajectory_z_angstrom", float),
        "point_energy_ev": ("trajectory_energy_ev", float),
        "point_time_fs": ("trajectory_time_fs", float),
        "point_electron_id": ("trajectory_electron_id", np.int32),
        "point_parent_id": ("trajectory_parent_id", np.int32),
        "point_root_primary_id": ("trajectory_root_primary_id", np.int32),
        "point_generation": ("trajectory_generation", np.int32),
        "point_inelastic_count": ("trajectory_inelastic_count", np.int32),
        "point_event": ("trajectory_event", np.int32),
        "point_surface_code": ("trajectory_surface_code", np.int32),
        "point_step": ("trajectory_step", np.int32),
    }
    emission_specs = {
        "emission_electron_id": ("emission_electron_id", np.int32),
        "emission_root_primary_id": ("emission_root_primary_id", np.int32),
        "emission_energy_ev": ("emission_energy_ev", float),
        "emission_ux": ("emission_ux", float),
        "emission_uy": ("emission_uy", float),
        "emission_uz": ("emission_uz", float),
        "emission_generation": ("emission_generation", np.int32),
        "emission_inelastic_count": ("emission_inelastic_count", np.int32),
    }
    point_parts = {name: [] for name in point_specs}
    emission_parts = {name: [] for name in emission_specs}
    point_pixels = []
    emission_pixels = []
    for pixel_id, payload in sorted(trajectories.items()):
        n_points = len(np.asarray(payload["trajectory_x_angstrom"]))
        point_pixels.append(np.full(n_points, int(pixel_id), dtype=np.int32))
        for name, (source, dtype) in point_specs.items():
            point_parts[name].append(np.asarray(payload[source], dtype=dtype))
        n_emissions = len(np.asarray(payload["emission_electron_id"]))
        emission_pixels.append(np.full(n_emissions, int(pixel_id), dtype=np.int32))
        for name, (source, dtype) in emission_specs.items():
            emission_parts[name].append(np.asarray(payload[source], dtype=dtype))

    def joined(parts, dtype):
        return np.concatenate(parts).astype(dtype, copy=False) if parts else _empty(dtype)

    kwargs = {
        name: joined(point_parts[name], dtype)
        for name, (_source, dtype) in point_specs.items()
    }
    kwargs.update({
        name: joined(emission_parts[name], dtype)
        for name, (_source, dtype) in emission_specs.items()
    })
    archive = RasterTrajectoryArchive(
        metadata=dict(metadata),
        x_angstrom=x_angstrom,
        y_angstrom=np.asarray([0.0]),
        profile_channels=np.asarray(channels, dtype="U"),
        profile_yields=yields,
        profile_sems=sems,
        point_pixel_id=joined(point_pixels, np.int32),
        emission_pixel_id=joined(emission_pixels, np.int32),
        **kwargs,
    )
    return archive.validate()


__all__ = ["ARCHIVE_FORMAT", "RasterTrajectoryArchive", "archive_from_scan"]
