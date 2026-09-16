from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .trajectory_anim_cli import (
    EVENT_NAMES,
    LINEAGE_NAMES,
    DEFAULT_PROFILE_CHANNELS,
    _load_trajectory,
    _line_segments_for_electron,
    _color_payload,
    trapezoid_outline,
)


def _parser():
    p = argparse.ArgumentParser(
        description=(
            "Animate the full trapezoid line scan, optionally showing traced "
            "trajectories for the current pixel while the scan profile builds"
        )
    )
    p.add_argument("scan_npz", type=Path)
    p.add_argument("--trajectory-dir", type=Path, default=None,
                   help="directory containing <prefix>_pixelXXX_trajectories.npz files; defaults to scan_npz directory")
    p.add_argument("--trajectory-pattern", type=str, default=None,
                   help="glob pattern for trajectory files; defaults to '<scanstem>_pixel*_trajectories.npz'")
    p.add_argument("--output", type=Path, default=Path("trapezoid_scan.gif"))
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--frames-per-pixel", type=int, default=10,
                   help="number of animation frames allocated to each scan pixel")
    p.add_argument("--pause-frames", type=int, default=18,
                   help="hold final frame for this many frames")
    p.add_argument("--color-by", choices=("lineage", "generation", "event", "energy"),
                   default="lineage")
    p.add_argument("--max-points-per-pixel", type=int, default=None,
                   help="optional cap on loaded trajectory records for each pixel")
    p.add_argument("--max-roots", type=int, default=None,
                   help="optional cap on traced root primaries per pixel in the animation")
    p.add_argument("--x-margin-nm", type=float, default=5.0)
    p.add_argument("--z-margin-nm", type=float, default=5.0)
    return p


def _metadata_from_scan(scan):
    if "metadata_json" not in scan:
        return {}
    raw = np.asarray(scan["metadata_json"])
    if raw.shape == ():
        raw = raw.item()
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def _infer_traj_files(scan_path: Path, trajectory_dir: Path | None, pattern: str | None):
    tdir = scan_path.parent if trajectory_dir is None else trajectory_dir
    patt = pattern or f"{scan_path.stem}_pixel*_trajectories.npz"
    return sorted(tdir.glob(patt))


def _load_trajectories(files, max_points_per_pixel=None, max_roots=None):
    out = {}
    for path in files:
        fields, _ = _load_trajectory(path, max_points=max_points_per_pixel, max_roots=max_roots)
        out[int(fields["pixel_id"])] = fields
    return out


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.fps < 1 or args.frames_per_pixel < 1 or args.pause_frames < 0:
        raise SystemExit("fps and frames-per-pixel must be positive; pause-frames must be non-negative")

    scan = np.load(args.scan_npz, allow_pickle=False)
    x_nm = np.asarray(scan["x_nm"], dtype=float)
    meta = _metadata_from_scan(scan)
    top = float(meta["top_width_nm"])
    bottom = float(meta["bottom_width_nm"])
    height = float(meta["height_nm"])

    traj_files = _infer_traj_files(args.scan_npz, args.trajectory_dir, args.trajectory_pattern)
    trajectories = _load_trajectories(traj_files, args.max_points_per_pixel, args.max_roots)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    fig, (ax_geo, ax_prof) = plt.subplots(
        1, 2, figsize=(13.2, 5.9), gridspec_kw={"width_ratios": [1.05, 1.15]}
    )

    outline_x, outline_z = trapezoid_outline(top, bottom, height)
    # geometry bounds from structure and any stored trajectories
    tx_all = []
    tz_all = []
    for fields in trajectories.values():
        if len(fields["x"]):
            tx_all.append(fields["x"])
            tz_all.append(fields["z"])
    if tx_all:
        tx = np.concatenate(tx_all)
        tz = np.concatenate(tz_all)
        xmin = min(float(np.min(tx)), float(np.min(outline_x))) - args.x_margin_nm
        xmax = max(float(np.max(tx)), float(np.max(outline_x))) + args.x_margin_nm
        zmin = min(float(np.min(tz)), float(np.min(outline_z))) - args.z_margin_nm
        zmax = max(float(np.max(tz)), 5.0) + args.z_margin_nm
    else:
        xmin = float(np.min(outline_x)) - args.x_margin_nm
        xmax = float(np.max(outline_x)) + args.x_margin_nm
        zmin = float(np.min(outline_z)) - args.z_margin_nm
        zmax = 5.0 + args.z_margin_nm

    ax_geo.plot(outline_x, outline_z, linewidth=2.0)
    ax_geo.fill(outline_x, outline_z, alpha=0.13)
    ax_geo.axhline(0.0, linewidth=1.0)
    ax_geo.set_xlim(xmin, xmax)
    ax_geo.set_ylim(zmax, zmin)
    ax_geo.set_xlabel("x (nm)")
    ax_geo.set_ylabel("z (nm)")
    ax_geo.set_title("Trapezoid scan animation")
    ax_geo.grid(True, alpha=0.15)
    beam_marker = ax_geo.scatter([], [], marker="x", s=80)
    lc = LineCollection([], linewidths=1.25)
    ax_geo.add_collection(lc)
    heads = ax_geo.scatter([], [], s=18)
    text = ax_geo.text(0.02, 0.98, "", transform=ax_geo.transAxes, va="top")

    if args.color_by == "energy":
        norm = Normalize(vmin=0.0, vmax=max(1.0, float(meta.get("energy_ev", 1.0))))
        sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
        sm.set_array([])
        fig.colorbar(sm, ax=ax_geo, label="Electron energy (eV)")
        static_map = None
    else:
        static_map = None
        if args.color_by == "lineage":
            cats = [0, 1, 2]
            cmap = plt.get_cmap("tab10")
            static_map = {cat: cmap(i % 10) for i, cat in enumerate(cats)}
            handles = [Line2D([0], [0], linewidth=2.5, color=static_map[k], label=LINEAGE_NAMES[k]) for k in cats]
            ax_geo.legend(handles=handles, loc="best", fontsize=8)

    # profile panel
    profile_lines = {}
    profile_data = {}
    for key, label in DEFAULT_PROFILE_CHANNELS:
        if key in scan:
            y = np.asarray(scan[key], dtype=float)
            profile_data[key] = y
            (line,) = ax_prof.plot([], [], label=label)
            profile_lines[key] = line
    vline = ax_prof.axvline(x_nm[0], linestyle="--", linewidth=1.5)
    ax_prof.set_xlabel("Beam x (nm)")
    ax_prof.set_ylabel("Emitted electrons / primary")
    ax_prof.set_title("Line scan profile building up")
    ax_prof.grid(True, alpha=0.25)
    ax_prof.legend(fontsize=8, ncol=2)
    prof_text = ax_prof.text(0.02, 0.98, "", transform=ax_prof.transAxes, va="top")

    total_pixels = len(x_nm)
    base_frames = total_pixels * args.frames_per_pixel
    total_frames = base_frames + args.pause_frames

    current_cache = {"pixel": None, "segments": np.empty((0, 2, 2), dtype=float),
                     "segment_end": np.empty(0, dtype=np.int32), "segment_electron": np.empty(0, dtype=np.int32),
                     "fields": None, "static_colors": None, "energy_values": None,
                     "mapping": None}

    def prepare_pixel(pixel_id: int):
        if current_cache["pixel"] == pixel_id:
            return
        current_cache["pixel"] = pixel_id
        fields = trajectories.get(pixel_id)
        current_cache["fields"] = fields
        if fields is None or len(fields["x"]) == 0:
            current_cache["segments"] = np.empty((0, 2, 2), dtype=float)
            current_cache["segment_end"] = np.empty(0, dtype=np.int32)
            current_cache["segment_electron"] = np.empty(0, dtype=np.int32)
            current_cache["static_colors"] = None
            current_cache["energy_values"] = None
            current_cache["mapping"] = None
            return
        segs, seg_e, seg_end = _line_segments_for_electron(fields["x"], fields["z"], fields["electron"])
        current_cache["segments"] = segs
        current_cache["segment_end"] = seg_end
        current_cache["segment_electron"] = seg_e
        static_colors, energy_values, mapping, labels = _color_payload(plt, fields, seg_e, seg_end, args.color_by)
        current_cache["static_colors"] = static_colors
        current_cache["energy_values"] = energy_values
        current_cache["mapping"] = mapping
        if args.color_by == "energy" and energy_values is not None and len(energy_values):
            lc.set_cmap("viridis")
            lc.set_norm(Normalize(vmin=max(0.0, float(np.min(energy_values))), vmax=max(1.0, float(np.max(energy_values)))))

    def update(frame):
        if frame >= base_frames:
            pixel_id = total_pixels - 1
            subframe = args.frames_per_pixel - 1
        else:
            pixel_id = int(frame // args.frames_per_pixel)
            subframe = int(frame % args.frames_per_pixel)
        prepare_pixel(pixel_id)

        # update profile reveal
        shown_pixels = min(total_pixels, pixel_id + 1)
        for key, line in profile_lines.items():
            line.set_data(x_nm[:shown_pixels], profile_data[key][:shown_pixels])
        vline.set_xdata([x_nm[pixel_id], x_nm[pixel_id]])

        tey = float(profile_data.get("tey", np.full(total_pixels, np.nan))[pixel_id])
        se1 = float(profile_data.get("se1", np.full(total_pixels, np.nan))[pixel_id])
        se2 = float(profile_data.get("se2", np.full(total_pixels, np.nan))[pixel_id])
        prof_text.set_text(
            f"pixel {pixel_id+1}/{total_pixels}\n"
            f"x = {x_nm[pixel_id]:+.3f} nm\n"
            f"TEY = {tey:.4f}\nSE1 = {se1:.4f}\nSE2 = {se2:.4f}"
        )

        # geometry + trajectories for current pixel
        beam_marker.set_offsets(np.c_[[x_nm[pixel_id]], [2.0]])
        fields = current_cache["fields"]
        if fields is None or len(fields["x"]) == 0:
            lc.set_segments([])
            heads.set_offsets(np.empty((0, 2)))
            text.set_text(f"pixel {pixel_id+1}/{total_pixels}\nx = {x_nm[pixel_id]:+.3f} nm\nno trajectory trace for this pixel")
            return [lc, heads, beam_marker, vline, prof_text, text, *profile_lines.values()]

        n_points = len(fields["x"])
        shown_points = max(1, int(np.floor((subframe + 1) * n_points / args.frames_per_pixel)))
        segs = current_cache["segments"]
        seg_end = current_cache["segment_end"]
        if len(segs):
            mseg = seg_end < shown_points
            lc.set_segments(segs[mseg])
            if args.color_by == "energy":
                lc.set_array(current_cache["energy_values"][mseg])
            else:
                cols = np.asarray(current_cache["static_colors"], dtype=object)[mseg].tolist()
                lc.set_color(cols)
        px = []
        pz = []
        head_colors = []
        electron = fields["electron"]
        for eid in np.unique(electron[:shown_points]):
            ind = np.flatnonzero(electron[:shown_points] == eid)
            if len(ind):
                jh = int(ind[-1])
                px.append(fields["x"][jh])
                pz.append(fields["z"][jh])
                if args.color_by == "lineage":
                    key = fields["lineage_by_electron"].get(int(eid), 0)
                    head_colors.append(current_cache["mapping"].get(key, (0.2, 0.2, 0.2, 1.0)))
                elif args.color_by == "generation":
                    cmap = plt.get_cmap("tab10")
                    head_colors.append(cmap(int(fields["generation"][jh]) % 10))
                elif args.color_by == "event":
                    cmap = plt.get_cmap("tab10")
                    head_colors.append(cmap(int(fields["event"][jh]) % 10))
                else:
                    norm = lc.norm if hasattr(lc, 'norm') else Normalize(vmin=0.0, vmax=1.0)
                    head_colors.append(plt.get_cmap("viridis")(norm(float(fields["energy"][jh]))))
        if px:
            heads.set_offsets(np.c_[px, pz])
            heads.set_color(head_colors)
        else:
            heads.set_offsets(np.empty((0, 2)))
        j = shown_points - 1
        ev_name = EVENT_NAMES.get(int(fields["event"][j]), str(int(fields["event"][j])))
        lin = fields["lineage_by_electron"].get(int(fields["electron"][j]), 0)
        text.set_text(
            f"pixel {pixel_id+1}/{total_pixels}  x={x_nm[pixel_id]:+.3f} nm\n"
            f"trajectory point {shown_points}/{n_points}\n"
            f"root={int(fields['root'][j])}  electron={int(fields['electron'][j])}\n"
            f"gen={int(fields['generation'][j])}  {LINEAGE_NAMES.get(lin, '?')}\n"
            f"event={ev_name}  E={fields['energy'][j]:.2f} eV"
        )
        return [lc, heads, beam_marker, vline, prof_text, text, *profile_lines.values()]

    ani = FuncAnimation(fig, update, frames=total_frames, interval=1000.0 / float(args.fps), blit=False)
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
