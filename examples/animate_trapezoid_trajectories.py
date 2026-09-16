from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EVENT_NAMES = {
    1: "launch",
    3: "elastic",
    4: "inelastic",
    5: "spawn",
    6: "surface hit",
    7: "internal reflection",
    9: "escape",
    10: "terminate",
}

LINEAGE_NAMES = {
    0: "Primary",
    1: "SE1",
    2: "SE2",
}

DEFAULT_PROFILE_CHANNELS = (
    ("tey", "TEY"),
    ("cascade_all", "Cascade all"),
    ("primary_all", "Primary all"),
    ("se1", "SE1"),
    ("se2", "SE2"),
)


def _parser():
    p = argparse.ArgumentParser(
        description=(
            "Animate traced Taichi trapezoid trajectories, optionally linked to "
            "the line-scan profile that produced the traced pixel"
        )
    )
    p.add_argument("trajectory_npz", type=Path)
    p.add_argument("--scan-npz", type=Path, default=None,
                   help="line-scan NPZ; adds a linked profile panel and traced-pixel marker")
    p.add_argument("--top-width-nm", type=float, default=None)
    p.add_argument("--bottom-width-nm", type=float, default=None)
    p.add_argument("--height-nm", type=float, default=None)
    p.add_argument("--output", type=Path, default=Path("trapezoid_trajectories.gif"))
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--frames-per-point", type=int, default=1)
    p.add_argument("--pause-frames", type=int, default=12,
                   help="hold final frame for this many frames")
    p.add_argument("--color-by", choices=("lineage", "generation", "event", "energy"),
                   default="lineage")
    p.add_argument("--max-points", type=int, default=None,
                   help="optional cap on displayed trajectory records")
    p.add_argument("--max-roots", type=int, default=None,
                   help="optional cap on distinct traced root primaries")
    p.add_argument("--x-margin-nm", type=float, default=5.0)
    p.add_argument("--z-margin-nm", type=float, default=5.0)
    return p


def _scalar(data, name, default=None):
    if name not in data:
        return default
    value = np.asarray(data[name])
    if value.shape == ():
        return value.item()
    if value.size == 1:
        return value.reshape(()).item()
    return value


def _metadata(data):
    if "metadata_json" not in data:
        return {}
    raw = _scalar(data, "metadata_json", "{}")
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def _geometry_from_args_or_metadata(args, data):
    meta = _metadata(data)
    top = args.top_width_nm if args.top_width_nm is not None else meta.get("top_width_nm")
    bottom = args.bottom_width_nm if args.bottom_width_nm is not None else meta.get("bottom_width_nm")
    height = args.height_nm if args.height_nm is not None else meta.get("height_nm")
    if top is None or bottom is None or height is None:
        raise ValueError(
            "geometry not found in trajectory metadata; provide --top-width-nm, "
            "--bottom-width-nm, and --height-nm"
        )
    return float(top), float(bottom), float(height)


def trapezoid_outline(top_width_nm: float, bottom_width_nm: float, height_nm: float):
    a = 0.5 * top_width_nm
    b = 0.5 * bottom_width_nm
    h = height_nm
    x = np.array([-b, -a, a, b, -b], dtype=float)
    z = np.array([0.0, -h, -h, 0.0, 0.0], dtype=float)
    return x, z


def classify_electron_lineage(electron_id, generation, inelastic_count):
    """Return per-electron class: 0 primary, 1 SE1, 2 SE2.

    SE1 follows the current SEEMC rule: generation==1 and the electron itself
    never underwent an inelastic collision.  The maximum recorded own
    inelastic count is used so the whole trajectory gets one stable class.
    """
    electron_id = np.asarray(electron_id, dtype=np.int64)
    generation = np.asarray(generation, dtype=np.int32)
    inelastic_count = np.asarray(inelastic_count, dtype=np.int32)
    classes: dict[int, int] = {}
    for eid in np.unique(electron_id):
        m = electron_id == eid
        g = int(np.max(generation[m])) if np.any(m) else 0
        nin = int(np.max(inelastic_count[m])) if np.any(m) else 0
        if g == 0:
            cls = 0
        elif g == 1 and nin == 0:
            cls = 1
        else:
            cls = 2
        classes[int(eid)] = cls
    return classes


def _load_trajectory(path: Path, max_points=None, max_roots=None):
    data = np.load(path, allow_pickle=False)
    fields = {
        "x": np.asarray(data["trajectory_x_angstrom"], dtype=float) / 10.0,
        "y": np.asarray(data["trajectory_y_angstrom"], dtype=float) / 10.0,
        "z": np.asarray(data["trajectory_z_angstrom"], dtype=float) / 10.0,
        "energy": np.asarray(data["trajectory_energy_ev"], dtype=float),
        "electron": np.asarray(data["trajectory_electron_id"], dtype=np.int32),
        "parent": np.asarray(data["trajectory_parent_id"], dtype=np.int32),
        "root": np.asarray(data["trajectory_root_primary_id"], dtype=np.int32),
        "generation": np.asarray(data["trajectory_generation"], dtype=np.int32),
        "inelastic": np.asarray(data["trajectory_inelastic_count"], dtype=np.int32),
        "event": np.asarray(data["trajectory_event"], dtype=np.int32),
        "surface": np.asarray(data["trajectory_surface_code"], dtype=np.int32),
        "step": np.asarray(data["trajectory_step"], dtype=np.int32),
    }
    n = len(fields["x"])
    if any(len(v) != n for v in fields.values()):
        raise ValueError("trajectory arrays have inconsistent lengths")
    keep = np.ones(n, dtype=bool)
    if max_roots is not None:
        roots = np.unique(fields["root"])
        allowed = set(int(v) for v in roots[: max(0, int(max_roots))])
        keep &= np.array([int(v) in allowed for v in fields["root"]], dtype=bool)
    idx = np.flatnonzero(keep)
    if max_points is not None:
        idx = idx[: max(0, int(max_points))]
    for key in fields:
        fields[key] = fields[key][idx]
    fields["lineage_by_electron"] = classify_electron_lineage(
        fields["electron"], fields["generation"], fields["inelastic"]
    )
    fields["pixel_id"] = int(_scalar(data, "pixel_id", -1))
    fields["pixel_x_nm"] = float(_scalar(data, "x_nm", np.nan))
    fields["metadata"] = _metadata(data)
    return fields, data


def _line_segments_for_electron(x, z, electron):
    segments = []
    segment_electron = []
    segment_end_index = []
    for eid in np.unique(electron):
        ind = np.flatnonzero(electron == eid)
        if len(ind) < 2:
            continue
        # Records for an electron are expected in event order.  Preserve stored
        # order because it is also the order used for animation reveal.
        for a, b in zip(ind[:-1], ind[1:]):
            segments.append([[x[a], z[a]], [x[b], z[b]]])
            segment_electron.append(int(eid))
            segment_end_index.append(int(b))
    if segments:
        return np.asarray(segments, dtype=float), np.asarray(segment_electron), np.asarray(segment_end_index)
    return np.empty((0, 2, 2), dtype=float), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)


def _category_colors(plt, values, categories):
    cmap = plt.get_cmap("tab10")
    mapping = {cat: cmap(i % 10) for i, cat in enumerate(categories)}
    return [mapping[v] for v in values], mapping


def _color_payload(plt, fields, segment_electron, segment_end_index, mode):
    electron = fields["electron"]
    generation = fields["generation"]
    event = fields["event"]
    energy = fields["energy"]
    lineage = fields["lineage_by_electron"]

    if mode == "lineage":
        values = [lineage[int(e)] for e in segment_electron]
        cats = [0, 1, 2]
        colors, mapping = _category_colors(plt, values, cats)
        labels = {k: LINEAGE_NAMES[k] for k in cats}
        return colors, None, mapping, labels

    if mode == "generation":
        values = [int(np.max(generation[electron == e])) for e in segment_electron]
        cats = sorted(set(int(v) for v in generation))
        colors, mapping = _category_colors(plt, values, cats)
        labels = {k: f"generation {k}" for k in cats}
        return colors, None, mapping, labels

    if mode == "event":
        values = [int(event[j]) for j in segment_end_index]
        cats = sorted(set(int(v) for v in event))
        colors, mapping = _category_colors(plt, values, cats)
        labels = {k: EVENT_NAMES.get(k, f"event {k}") for k in cats}
        return colors, None, mapping, labels

    # energy: return numeric array for colormap
    values = np.asarray([float(energy[j]) for j in segment_end_index], dtype=float)
    return None, values, None, None


def _make_profile_panel(ax, scan_path: Path, pixel_id: int, pixel_x_nm: float):
    scan = np.load(scan_path, allow_pickle=False)
    x = np.asarray(scan["x_nm"], dtype=float)
    for key, label in DEFAULT_PROFILE_CHANNELS:
        if key in scan:
            ax.plot(x, np.asarray(scan[key], dtype=float), label=label)
    if 0 <= pixel_id < len(x):
        marker_x = float(x[pixel_id])
    else:
        marker_x = float(pixel_x_nm)
    ax.axvline(marker_x, linestyle="--", linewidth=1.5)
    ax.scatter([marker_x], [0.0], marker="^", s=45, clip_on=False)
    ax.set_xlabel("Beam x (nm)")
    ax.set_ylabel("Emitted electrons / primary")
    ax.set_title(f"Line scan — traced pixel {pixel_id} at x={marker_x:+.3f} nm")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    return marker_x


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.fps < 1 or args.frames_per_point < 1 or args.pause_frames < 0:
        raise SystemExit("fps and frames-per-point must be positive; pause-frames must be non-negative")

    fields, raw = _load_trajectory(
        args.trajectory_npz,
        max_points=args.max_points,
        max_roots=args.max_roots,
    )
    x = fields["x"]
    z = fields["z"]
    electron = fields["electron"]
    if len(x) == 0:
        raise SystemExit("trajectory file contains no selected points")

    top, bottom, height = _geometry_from_args_or_metadata(args, raw)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize

    linked = args.scan_npz is not None
    if linked:
        fig, (ax, ax_profile) = plt.subplots(
            1, 2, figsize=(12.8, 5.7), gridspec_kw={"width_ratios": [1.05, 1.0]}
        )
    else:
        fig, ax = plt.subplots(figsize=(7.6, 5.8))
        ax_profile = None

    outline_x, outline_z = trapezoid_outline(top, bottom, height)
    xmin = min(float(np.min(x)), float(np.min(outline_x))) - args.x_margin_nm
    xmax = max(float(np.max(x)), float(np.max(outline_x))) + args.x_margin_nm
    zmin = min(float(np.min(z)), float(np.min(outline_z))) - args.z_margin_nm
    zmax = max(float(np.max(z)), 5.0) + args.z_margin_nm

    ax.plot(outline_x, outline_z, linewidth=2.0)
    ax.fill(outline_x, outline_z, alpha=0.13)
    # Same visual surface with no extra boundary color distinction between line and substrate.
    ax.axhline(0.0, linewidth=1.0)
    if np.isfinite(fields["pixel_x_nm"]):
        ax.axvline(fields["pixel_x_nm"], linestyle=":", linewidth=1.2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmax, zmin)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("z (nm)")
    ax.set_title(f"Traced trajectories — {args.color_by}")
    ax.grid(True, alpha=0.15)

    segments, segment_electron, segment_end = _line_segments_for_electron(x, z, electron)
    static_colors, energy_values, mapping, labels = _color_payload(
        plt, fields, segment_electron, segment_end, args.color_by
    )
    if args.color_by == "energy":
        if len(energy_values):
            norm = Normalize(vmin=max(0.0, float(np.min(energy_values))), vmax=max(1.0, float(np.max(energy_values))))
        else:
            norm = Normalize(vmin=0.0, vmax=1.0)
        lc = LineCollection([], cmap="viridis", norm=norm, linewidths=1.35)
        ax.add_collection(lc)
        sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Electron energy (eV)")
    else:
        lc = LineCollection([], linewidths=1.35)
        ax.add_collection(lc)
        if mapping:
            handles = [
                Line2D([0], [0], linewidth=2.5, color=mapping[k], label=labels[k])
                for k in mapping
            ]
            ax.legend(handles=handles, loc="best", fontsize=8)

    heads = ax.scatter([], [], s=20)
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    if linked:
        _make_profile_panel(
            ax_profile, args.scan_npz, fields["pixel_id"], fields["pixel_x_nm"]
        )

    n_points = len(x)
    base_frames = n_points * args.frames_per_point
    total_frames = base_frames + args.pause_frames

    def update(frame):
        shown = min(n_points, frame // args.frames_per_point + 1)
        if len(segments):
            mseg = segment_end < shown
            lc.set_segments(segments[mseg])
            if args.color_by == "energy":
                lc.set_array(energy_values[mseg])
            else:
                cols = np.asarray(static_colors, dtype=object)[mseg].tolist()
                lc.set_color(cols)

        # Current head for each electron that has appeared.
        px = []
        pz = []
        head_colors = []
        for eid in np.unique(electron[:shown]):
            ind = np.flatnonzero(electron[:shown] == eid)
            if len(ind):
                jh = int(ind[-1])
                px.append(x[jh])
                pz.append(z[jh])
                if args.color_by == "lineage":
                    key = fields["lineage_by_electron"].get(int(eid), 0)
                    head_colors.append(mapping.get(key, (0.2, 0.2, 0.2, 1.0)))
                elif args.color_by == "generation":
                    key = int(fields["generation"][jh])
                    head_colors.append(mapping.get(key, (0.2, 0.2, 0.2, 1.0)))
                elif args.color_by == "event":
                    key = int(fields["event"][jh])
                    head_colors.append(mapping.get(key, (0.2, 0.2, 0.2, 1.0)))
                else:
                    head_colors.append(plt.get_cmap("viridis")(norm(float(fields["energy"][jh]))))
        if px:
            heads.set_offsets(np.c_[px, pz])
            heads.set_color(head_colors)
        else:
            heads.set_offsets(np.empty((0, 2)))

        j = shown - 1
        ev_name = EVENT_NAMES.get(int(fields["event"][j]), str(int(fields["event"][j])))
        lin = fields["lineage_by_electron"].get(int(electron[j]), 0)
        text.set_text(
            f"pixel {fields['pixel_id']}  x={fields['pixel_x_nm']:+.3f} nm\n"
            f"point {shown}/{n_points}  root={int(fields['root'][j])}  "
            f"electron={int(electron[j])}\n"
            f"gen={int(fields['generation'][j])}  {LINEAGE_NAMES.get(lin, '?')}  "
            f"event={ev_name}  E={fields['energy'][j]:.2f} eV"
        )
        return [lc, heads, text]

    ani = FuncAnimation(
        fig, update, frames=total_frames,
        interval=1000.0 / float(args.fps), blit=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.output.suffix.lower()
    if suffix == ".mp4":
        writer = FFMpegWriter(fps=args.fps)
    else:
        writer = PillowWriter(fps=args.fps)
    ani.save(args.output, writer=writer)
    plt.close(fig)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
