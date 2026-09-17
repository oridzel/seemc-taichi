"""Animate a recorded Taichi scan over one or more trapezoidal lines."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .scan_archive import RasterTrajectoryArchive


_REST_ENERGY_EV = 510_998.95069
_C_ANGSTROM_PER_FS = 2_997.92458

POPULATION_COLORS = {
    "se1": "#39d98a",
    "se2": "#f9c74f",
    "lle_primary": "#43aaef",
    "non_lle_primary": "#c77dff",
    "cascade_absorbed": "#98a2b3",
    "primary_absorbed": "#e5e7eb",
}

POPULATION_LABELS = {
    "se1": "SE1",
    "se2": "SE2",
    "lle_primary": "Low-loss primary",
    "non_lle_primary": "Non-LLE primary",
    "cascade_absorbed": "Absorbed cascade",
    "primary_absorbed": "Absorbed primary",
}

PROFILE_STYLES = {
    "cascade_all": ("Full SE", "#39d98a"),
    "primary_all": ("Full BSE (LLE + non-LLE)", "#43aaef"),
    "tey": ("Total measured", "#f8fafc"),
    "sey_50ev": ("SE, E <= cutoff", "#39d98a"),
    "bsey_50ev": ("BSE, E > cutoff", "#43aaef"),
    "se1": ("SE1", "#39d98a"),
    "se2": ("SE2", "#f9c74f"),
    "lle_primary": ("Low-loss primary", "#43aaef"),
    "non_lle_primary": ("Non-LLE primary", "#c77dff"),
}

PROFILE_PRESETS = {
    "physical": ("cascade_all", "primary_all", "tey"),
    "populations": ("se1", "se2", "lle_primary", "non_lle_primary"),
    "conventional": ("sey_50ev", "bsey_50ev"),
    "tey_se_bse": ("tey", "sey_50ev", "bsey_50ev"),
}


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trajectories", type=Path)
    p.add_argument("--output", type=Path, default=Path("trapezoid_scan.mp4"))
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frames-per-pixel", type=int, default=16)
    p.add_argument("--pause-frames", type=int, default=4)
    p.add_argument(
        "--pixel-stride", type=int, default=1,
        help="animate every Nth recorded beam position",
    )
    p.add_argument(
        "--color-by", choices=("energy", "population"), default="energy"
    )
    p.add_argument("--tail-fraction", type=float, default=0.45)
    p.add_argument("--vacuum-flight-nm", type=float, default=35.0)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--title")
    p.add_argument(
        "--n-lines", type=int,
        help="display override for older archives without line-array metadata",
    )
    p.add_argument(
        "--pitch-nm", type=float,
        help="display override for center-to-center line pitch in nm",
    )
    p.add_argument(
        "--line-centers-nm", type=float, nargs="+",
        help="explicit display line centers in nm, e.g. -100 0 100",
    )
    p.add_argument(
        "--profile-channels", default="cascade_all,primary_all,tey",
        help=(
            "lower panel (default: Full SE, Full BSE including LLE+non-LLE, "
            "and Total measured); accepts a preset or comma-separated channels"
        ),
    )
    return p


def _validate_args(parser, args):
    if args.fps < 1:
        parser.error("--fps must be positive")
    if args.frames_per_pixel < 2:
        parser.error("--frames-per-pixel must be at least 2")
    if args.pause_frames < 0:
        parser.error("--pause-frames must be non-negative")
    if args.pixel_stride < 1:
        parser.error("--pixel-stride must be positive")
    if not 0.0 < args.tail_fraction <= 1.0:
        parser.error("--tail-fraction must lie in (0, 1]")
    if not math.isfinite(args.vacuum_flight_nm) or args.vacuum_flight_nm < 0.0:
        parser.error("--vacuum-flight-nm must be finite and non-negative")
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    if args.n_lines is not None and args.n_lines < 1:
        parser.error("--n-lines must be >=1")
    if args.pitch_nm is not None and (
        not math.isfinite(args.pitch_nm) or args.pitch_nm <= 0.0
    ):
        parser.error("--pitch-nm must be finite and positive")
    if args.line_centers_nm is not None and not all(
        math.isfinite(value) for value in args.line_centers_nm
    ):
        parser.error("--line-centers-nm values must be finite")
    if args.output.suffix.lower() not in {".mp4", ".gif"}:
        parser.error("--output must end in .mp4 or .gif")


def _resolve_profile_channels(value, available):
    if isinstance(value, str) and value in PROFILE_PRESETS:
        channels = PROFILE_PRESETS[value]
    elif isinstance(value, str):
        channels = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        channels = tuple(str(part) for part in value)
    if not channels:
        raise ValueError("profile_channels must contain at least one channel")
    if len(channels) > 6:
        raise ValueError("profile_channels supports at most six channels")
    unknown = [channel for channel in channels if channel not in available]
    if unknown:
        raise ValueError(
            f"unknown profile channels {unknown}; available channels: {list(available)}"
        )
    return channels


def _geometry_values(archive, args):
    metadata = archive.metadata
    geometry = metadata.get("geometry", {})

    def dimension_nm(geometry_key, metadata_key):
        if geometry_key in geometry:
            return float(geometry[geometry_key]) / 10.0
        if metadata_key in metadata:
            return float(metadata[metadata_key])
        raise ValueError(
            f"trajectory archive is missing geometry value {geometry_key!r}"
        )

    top = dimension_nm("top_width", "top_width_nm")
    bottom = dimension_nm("bottom_width", "bottom_width_nm")
    height = dimension_nm("height", "height_nm")
    center = float(geometry.get("center_x", 0.0)) / 10.0
    substrate_height = -float(geometry.get("substrate_z", 0.0)) / 10.0

    metadata_centers = geometry.get("line_centers")
    if args.line_centers_nm is not None:
        centers = np.asarray(args.line_centers_nm, dtype=float)
    elif (
        metadata_centers is not None
        and args.n_lines is None
        and args.pitch_nm is None
    ):
        centers = np.asarray(metadata_centers, dtype=float) / 10.0
    else:
        n_lines = int(
            args.n_lines
            if args.n_lines is not None
            else geometry.get("n_lines", metadata.get("n_lines", 1))
        )
        pitch = float(
            args.pitch_nm
            if args.pitch_nm is not None
            else (
                float(geometry["pitch"]) / 10.0
                if "pitch" in geometry
                else metadata.get("pitch_nm", bottom)
            )
        )
        centers = center + (
            np.arange(n_lines, dtype=float) - 0.5 * (n_lines - 1)
        ) * pitch
    if len(centers) < 1:
        raise ValueError("line-center list must not be empty")
    if args.n_lines is not None and len(centers) != args.n_lines:
        raise ValueError("--n-lines does not match --line-centers-nm")
    if len(centers) > 1 and np.any(np.diff(np.sort(centers)) < bottom):
        raise ValueError("display line centers overlap for the configured bottom width")
    return {
        "top_width": top,
        "bottom_width": bottom,
        "height": height,
        "centers": np.asarray(centers, dtype=float),
        "substrate_height": substrate_height,
    }


def _surface_height_nm(x_nm, geometry):
    top = geometry["top_width"]
    bottom = geometry["bottom_width"]
    height = geometry["height"]
    substrate = geometry["substrate_height"]
    half_top = 0.5 * top
    half_bottom = 0.5 * bottom
    best = substrate
    for center in geometry["centers"]:
        distance = abs(float(x_nm) - float(center))
        if distance <= half_top:
            best = max(best, substrate + height)
        elif distance < half_bottom and half_bottom > half_top:
            fraction = (half_bottom - distance) / (half_bottom - half_top)
            best = max(best, substrate + height * fraction)
    return best


def _flight_time_fs(distance_angstrom, energy_ev):
    if distance_angstrom <= 0.0 or energy_ev <= 0.0:
        return 0.0
    gamma = 1.0 + float(energy_ev) / _REST_ENERGY_EV
    beta2 = max(1.0 - 1.0 / (gamma * gamma), 0.0)
    if beta2 <= 0.0:
        return 0.0
    return float(distance_angstrom) / (_C_ANGSTROM_PER_FS * math.sqrt(beta2))


def _collapse_equal_times(points):
    if len(points) < 2:
        return points
    reverse_unique = np.unique(points[::-1, 4], return_index=True)[1]
    keep = np.sort(len(points) - 1 - reverse_unique)
    return points[keep]


def _emission_lookup(archive):
    lookup = {}
    for index in range(len(archive.emission_pixel_id)):
        key = (
            int(archive.emission_pixel_id[index]),
            int(archive.emission_electron_id[index]),
        )
        lookup[key] = index
    return lookup


def _electron_population(archive, point_indices, emission_index):
    generation = int(np.max(archive.point_generation[point_indices]))
    own_inelastic = int(np.max(archive.point_inelastic_count[point_indices]))
    if emission_index is None:
        return "primary_absorbed" if generation == 0 else "cascade_absorbed"
    if generation == 1 and own_inelastic == 0:
        return "se1"
    if generation > 0:
        return "se2"
    if "energy_ev" in archive.metadata:
        incident = float(archive.metadata["energy_ev"])
    else:
        incident = float(archive.metadata.get("config", {})["energy_ev"])
    loss = incident - float(archive.emission_energy_ev[emission_index])
    threshold = float(archive.metadata.get("lle_max_loss_ev", 50.0))
    return "lle_primary" if loss < threshold else "non_lle_primary"


def _build_tracks(archive, vacuum_flight_nm):
    emission_lookup = _emission_lookup(archive)
    tracks_by_pixel = {}
    for pixel_id in archive.recorded_pixels:
        pixel_id = int(pixel_id)
        pixel_mask = archive.point_pixel_id == pixel_id
        electrons = np.unique(archive.point_electron_id[pixel_mask])
        pixel_tracks = []
        for electron_id in electrons:
            indices = np.flatnonzero(
                pixel_mask & (archive.point_electron_id == electron_id)
            )
            points = np.column_stack((
                archive.point_x_angstrom[indices],
                archive.point_y_angstrom[indices],
                archive.point_z_angstrom[indices],
                archive.point_energy_ev[indices],
                archive.point_time_fs[indices],
            ))
            points = _collapse_equal_times(points)
            emission_index = emission_lookup.get((pixel_id, int(electron_id)))
            if emission_index is not None and len(points) and vacuum_flight_nm > 0.0:
                distance = 10.0 * float(vacuum_flight_nm)
                terminal = points[-1].copy()
                terminal[0] += distance * archive.emission_ux[emission_index]
                terminal[1] += distance * archive.emission_uy[emission_index]
                terminal[2] += distance * archive.emission_uz[emission_index]
                terminal[3] = archive.emission_energy_ev[emission_index]
                terminal[4] += _flight_time_fs(distance, terminal[3])
                points = np.vstack((points, terminal))
            pixel_tracks.append({
                "electron_id": int(electron_id),
                "root_primary_id": int(archive.point_root_primary_id[indices[0]]),
                "population": _electron_population(
                    archive, indices, emission_index
                ),
                "points": points,
            })
        tracks_by_pixel[pixel_id] = pixel_tracks
    return tracks_by_pixel


def animate_trapezoidal_scan(
    archive,
    output,
    *,
    fps=30,
    frames_per_pixel=16,
    pause_frames=4,
    pixel_stride=1,
    color_by="energy",
    tail_fraction=0.45,
    vacuum_flight_nm=35.0,
    dpi=150,
    title=None,
    profile_channels="cascade_all,primary_all,tey",
    n_lines=None,
    pitch_nm=None,
    line_centers_nm=None,
):
    """Render the SEEMC-imaging style whole-scan MP4 or GIF."""
    frames_per_pixel = int(frames_per_pixel)
    pause_frames = int(pause_frames)
    pixel_stride = int(pixel_stride)
    fps = int(fps)
    dpi = int(dpi)
    if frames_per_pixel < 2 or pause_frames < 0 or pixel_stride < 1 or fps < 1:
        raise ValueError("invalid frame, pause, pixel-stride, or fps setting")
    if color_by not in {"energy", "population"}:
        raise ValueError("color_by must be 'energy' or 'population'")
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    if not math.isfinite(float(vacuum_flight_nm)) or vacuum_flight_nm < 0.0:
        raise ValueError("vacuum_flight_nm must be finite and non-negative")
    if dpi < 1:
        raise ValueError("dpi must be positive")
    if not isinstance(archive, RasterTrajectoryArchive):
        archive = RasterTrajectoryArchive.load_npz(archive)
    archive.validate()
    if archive.n_cascades == 0:
        raise ValueError("trajectory archive contains no recorded cascades")
    args = argparse.Namespace(
        n_lines=n_lines,
        pitch_nm=pitch_nm,
        line_centers_nm=line_centers_nm,
    )
    geometry = _geometry_values(archive, args)
    profile_channels = _resolve_profile_channels(
        profile_channels, tuple(str(value) for value in archive.profile_channels)
    )

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LogNorm, to_rgba
        from matplotlib.cm import ScalarMappable
        from matplotlib.patches import Polygon, Rectangle
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "animation requires matplotlib and Pillow; install seemc-taichi[animation]"
        ) from exc

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix not in {".mp4", ".gif"}:
        raise ValueError("animation output must end in .mp4 or .gif")

    x_nm = archive.x_angstrom / 10.0
    available_pixels = archive.recorded_pixels[::pixel_stride]
    tracks_by_pixel = _build_tracks(archive, vacuum_flight_nm)
    pixel_duration = {
        int(pixel_id): max(
            (
                float(track["points"][-1, 4])
                for track in tracks_by_pixel[int(pixel_id)]
                if len(track["points"])
            ),
            default=1.0e-12,
        )
        for pixel_id in available_pixels
    }

    incident = max(
        float(
            archive.metadata["energy_ev"]
            if "energy_ev" in archive.metadata
            else archive.metadata.get("config", {})["energy_ev"]
        ),
        1.0,
    )
    positive = archive.point_energy_ev[archive.point_energy_ev > 0.0]
    min_energy = max(
        min(float(np.percentile(positive, 1.0)), 1.0) if len(positive) else 0.1,
        0.1,
    )
    energy_norm = LogNorm(vmin=min_energy, vmax=max(incident, min_energy * 1.01))
    energy_cmap = plt.get_cmap("turbo")

    figure = plt.figure(figsize=(11.5, 7.4), facecolor="#070b12")
    grid = figure.add_gridspec(4, 1, height_ratios=(1, 1, 1, 0.82), hspace=0.08)
    axis = figure.add_subplot(grid[:3, 0])
    profile_axis = figure.add_subplot(grid[3, 0], sharex=axis)
    for current_axis in (axis, profile_axis):
        current_axis.set_facecolor("#070b12")
        current_axis.tick_params(colors="#cbd5e1")
        for spine in current_axis.spines.values():
            spine.set_color("#344054")

    height = geometry["height"]
    substrate_height = geometry["substrate_height"]
    half_bottom = 0.5 * geometry["bottom_width"]
    content_left = min(float(x_nm[0]), float(np.min(geometry["centers"])) - half_bottom)
    content_right = max(float(x_nm[-1]), float(np.max(geometry["centers"])) + half_bottom)
    x_margin = max(0.08 * (content_right - content_left), 8.0)
    axis.set_xlim(content_left - x_margin, content_right + x_margin)
    lower_limit = substrate_height - max(0.35 * height, 12.0)
    axis.set_ylim(lower_limit, substrate_height + height + vacuum_flight_nm)
    axis.set_ylabel("Height above line base (nm)", color="#e5e7eb")
    axis.tick_params(labelbottom=False)
    axis.grid(color="#334155", alpha=0.18, linewidth=0.7)

    solid_color = "#2d4057"
    surface_color = "#9fb3c8"
    support = Rectangle(
        (content_left - 2.0 * x_margin, lower_limit),
        content_right - content_left + 4.0 * x_margin,
        substrate_height - lower_limit,
        facecolor=solid_color,
        edgecolor="none",
        zorder=0,
    )
    axis.add_patch(support)
    half_top = 0.5 * geometry["top_width"]
    for center in geometry["centers"]:
        vertices = [
            (center - half_bottom, substrate_height),
            (center - half_top, substrate_height + height),
            (center + half_top, substrate_height + height),
            (center + half_bottom, substrate_height),
        ]
        axis.add_patch(Polygon(
            vertices, closed=True, facecolor=solid_color,
            edgecolor="none", zorder=1,
        ))
        # Draw only exposed top/side surfaces.  Omitting the base edge keeps
        # each line visually continuous with the same-colour substrate.
        axis.plot(
            [value[0] for value in vertices],
            [value[1] for value in vertices],
            color=surface_color, linewidth=1.1, zorder=2,
        )
    exposed = sorted(
        (float(center - half_bottom), float(center + half_bottom))
        for center in geometry["centers"]
    )
    cursor = axis.get_xlim()[0]
    for left, right in exposed:
        if left > cursor:
            axis.plot(
                [cursor, left], [substrate_height, substrate_height],
                color=surface_color, linewidth=1.0, zorder=2,
            )
        cursor = max(cursor, right)
    if cursor < axis.get_xlim()[1]:
        axis.plot(
            [cursor, axis.get_xlim()[1]], [substrate_height, substrate_height],
            color=surface_color, linewidth=1.0, zorder=2,
        )

    beam_line, = axis.plot([], [], color="#59e1ff", linewidth=2.1,
                           alpha=0.9, zorder=4)
    beam_halo, = axis.plot([], [], color="#59e1ff", linewidth=7.0,
                           alpha=0.13, zorder=3)
    beam_spot = axis.scatter([], [], s=62, facecolor="#d8fbff",
                             edgecolor="#59e1ff", linewidth=1.4, zorder=8)
    status = axis.text(
        0.015, 0.965, "", transform=axis.transAxes, va="top", ha="left",
        color="#f8fafc", fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#101827",
              "edgecolor": "#334155", "alpha": 0.88},
    )
    sample_name = archive.metadata.get("sample_name", archive.metadata.get("material", "sample"))
    main_title = title or (
        f"SEEMC-Taichi electron-cascade scan • {sample_name} • {incident / 1000.0:g} keV"
    )
    axis.set_title(main_title, color="#f8fafc", fontsize=14, pad=12)

    profile_axis.set_xlabel("Beam position x (nm)", color="#e5e7eb")
    profile_axis.set_ylabel("Yield", color="#e5e7eb")
    profile_axis.grid(color="#334155", alpha=0.22, linewidth=0.7)
    channel_names = [str(value) for value in archive.profile_channels]
    default_colors = ("#f8fafc", "#39d98a", "#f9c74f", "#43aaef", "#c77dff", "#f9844a")
    profile_lines = []
    profile_values = []
    for index, channel in enumerate(profile_channels):
        label, color = PROFILE_STYLES.get(
            channel, (channel.replace("_", " "), default_colors[index])
        )
        values = archive.profile_yields[channel_names.index(channel), 0]
        line, = profile_axis.plot([], [], color=color, linewidth=1.8, label=label)
        profile_lines.append(line)
        profile_values.append(values)
        profile_axis.plot(x_nm, values, color=color, linewidth=0.8, alpha=0.14)
    finite_profile = [values[np.isfinite(values)] for values in profile_values]
    finite_profile = [values for values in finite_profile if len(values)]
    upper = max(
        (float(np.max(values)) for values in finite_profile),
        default=0.1,
    )
    profile_axis.set_ylim(0.0, max(upper * 1.12, 0.1))
    profile_axis.legend(
        loc="upper right", ncol=len(profile_lines), frameon=False,
        fontsize=8.5, labelcolor="#e5e7eb",
    )
    profile_cursor = profile_axis.axvline(
        x_nm[0], color="#59e1ff", linewidth=1.2, alpha=0.85
    )

    if color_by == "energy":
        mapper = ScalarMappable(norm=energy_norm, cmap=energy_cmap)
        colorbar = figure.colorbar(mapper, ax=axis, pad=0.012, fraction=0.035)
        colorbar.set_label("Instantaneous energy (eV)", color="#e5e7eb")
        colorbar.ax.tick_params(colors="#cbd5e1")
        colorbar.outline.set_edgecolor("#344054")
    else:
        populations = {
            track["population"]
            for tracks in tracks_by_pixel.values()
            for track in tracks
        }
        handles = [
            plt.Line2D(
                [0], [0], color=POPULATION_COLORS[name], lw=2.4,
                label=POPULATION_LABELS[name],
            )
            for name in POPULATION_COLORS if name in populations
        ]
        if handles:
            axis.legend(
                handles=handles, loc="upper right", frameon=False,
                fontsize=8, labelcolor="#e5e7eb", ncol=2,
            )

    dynamic_artists = []
    frames_each = frames_per_pixel + pause_frames
    total_frames = len(available_pixels) * frames_each

    def clear_dynamic():
        while dynamic_artists:
            dynamic_artists.pop().remove()

    def interpolate_head(track, time_fs):
        if time_fs < track[0, 4]:
            return None
        if time_fs >= track[-1, 4]:
            return track[-1, :4]
        upper_index = int(np.searchsorted(track[:, 4], time_fs, side="right"))
        lower_index = max(upper_index - 1, 0)
        t0, t1 = track[lower_index, 4], track[upper_index, 4]
        fraction = 1.0 if t1 <= t0 else (time_fs - t0) / (t1 - t0)
        return track[lower_index, :4] + fraction * (
            track[upper_index, :4] - track[lower_index, :4]
        )

    def update(frame_index):
        clear_dynamic()
        sequence_index = min(frame_index // frames_each, len(available_pixels) - 1)
        local_frame = frame_index % frames_each
        pixel_id = int(available_pixels[sequence_index])
        progress = min(local_frame / max(frames_per_pixel - 1, 1), 1.0)
        time_fs = progress * pixel_duration[pixel_id]
        nominal_x = float(x_nm[pixel_id])
        nominal_height = _surface_height_nm(nominal_x, geometry)
        beam_top = axis.get_ylim()[1]
        beam_line.set_data([nominal_x, nominal_x], [beam_top, nominal_height])
        beam_halo.set_data([nominal_x, nominal_x], [beam_top, nominal_height])
        beam_spot.set_offsets(np.asarray([[nominal_x, nominal_height]]))

        roots = set()
        for track_info in tracks_by_pixel[pixel_id]:
            track = track_info["points"]
            if len(track) == 0 or time_fs < track[0, 4]:
                continue
            roots.add(track_info["root_primary_id"])
            head = interpolate_head(track, time_fs)
            if head is None:
                continue
            visible = track[track[:, 4] <= time_fs]
            if len(visible) == 0 or not np.allclose(visible[-1, :4], head):
                visible = np.vstack((visible, np.r_[head, time_fs]))
            tail_start = time_fs - float(tail_fraction) * pixel_duration[pixel_id]
            visible = visible[visible[:, 4] >= tail_start]
            if len(visible) == 1:
                visible = np.vstack((visible, visible))
            xy = np.column_stack((visible[:, 0] / 10.0, -visible[:, 2] / 10.0))
            segments = np.stack((xy[:-1], xy[1:]), axis=1)
            if color_by == "energy":
                colors = energy_cmap(
                    energy_norm(np.maximum(visible[1:, 3], min_energy))
                )
            else:
                color = POPULATION_COLORS[track_info["population"]]
                colors = np.tile(np.asarray(to_rgba(color)), (len(segments), 1))
            if len(colors):
                colors[:, 3] *= np.linspace(0.12, 0.92, len(colors))
                collection = LineCollection(
                    segments, colors=colors, linewidths=1.45, zorder=5
                )
                axis.add_collection(collection)
                dynamic_artists.append(collection)
            if color_by == "energy":
                head_color = energy_cmap(
                    energy_norm(max(float(head[3]), min_energy))
                )
            else:
                head_color = POPULATION_COLORS[track_info["population"]]
            marker = axis.scatter(
                [head[0] / 10.0], [-head[2] / 10.0], s=16,
                facecolor=head_color, edgecolor="white", linewidth=0.25,
                alpha=0.95, zorder=7,
            )
            dynamic_artists.append(marker)

        for line, values in zip(profile_lines, profile_values):
            line.set_data(x_nm[: pixel_id + 1], values[: pixel_id + 1])
        profile_cursor.set_xdata([nominal_x, nominal_x])
        status.set_text(
            f"pixel {sequence_index + 1}/{len(available_pixels)}   "
            f"x = {nominal_x:+.1f} nm   t = {time_fs:.3f} fs   "
            f"primaries = {len(roots)}"
        )
        return [
            beam_line, beam_halo, beam_spot, profile_cursor, status,
            *profile_lines, *dynamic_artists,
        ]

    animation = FuncAnimation(
        figure, update, frames=total_frames,
        interval=1000.0 / fps, blit=False, repeat=False,
    )
    if suffix == ".mp4":
        if not FFMpegWriter.isAvailable():
            plt.close(figure)
            raise RuntimeError(
                "ffmpeg is unavailable; install ffmpeg or choose a .gif output"
            )
        writer = FFMpegWriter(
            fps=fps, bitrate=3200,
            metadata={"title": main_title, "artist": "SEEMC-Taichi"},
            extra_args=[
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
            ],
        )
    else:
        writer = PillowWriter(fps=fps)
    animation.save(output, writer=writer, dpi=dpi)
    plt.close(figure)
    return output


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    archive = RasterTrajectoryArchive.load_npz(args.trajectories)
    output = animate_trapezoidal_scan(
        archive,
        args.output,
        fps=args.fps,
        frames_per_pixel=args.frames_per_pixel,
        pause_frames=args.pause_frames,
        pixel_stride=args.pixel_stride,
        color_by=args.color_by,
        tail_fraction=args.tail_fraction,
        vacuum_flight_nm=args.vacuum_flight_nm,
        dpi=args.dpi,
        title=args.title,
        profile_channels=args.profile_channels,
        n_lines=args.n_lines,
        pitch_nm=args.pitch_nm,
        line_centers_nm=args.line_centers_nm,
    )
    duration = (
        len(archive.recorded_pixels[::args.pixel_stride])
        * (args.frames_per_pixel + args.pause_frames)
        / args.fps
    )
    print(f"Wrote {output} ({duration:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "POPULATION_COLORS",
    "PROFILE_PRESETS",
    "PROFILE_STYLES",
    "animate_trapezoidal_scan",
]
