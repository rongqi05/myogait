#!/usr/bin/env python3
"""Reference-motion statistics for the locally available retargeted clips.

The acceptance protocol requires a 10 s episode whose reference has not ended
and a mean forward speed >= 0.6 m/s. This script measures whether the available
reference clips can satisfy those requirements *at all*, independently of the
policy.

    .venv/bin/python scripts/motion_stats.py --out reports/motion_stats.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def segment_stats(t: np.ndarray, xy: np.ndarray, speed: np.ndarray, thresh: float) -> list[dict]:
    moving = speed > thresh
    segs = []
    i = 0
    while i < len(moving):
        if not moving[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(moving) and moving[j + 1]:
            j += 1
        d = float(np.sum(np.linalg.norm(np.diff(xy[i : j + 1], axis=0), axis=1)))
        dur = t[j] - t[i]
        segs.append(
            {
                "start_s": float(t[i]),
                "end_s": float(t[j]),
                "duration_s": float(dur),
                "path_length_m": d,
                "mean_speed": float(d / dur) if dur > 0 else 0.0,
            }
        )
        i = j + 1
    return segs


def analyse(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    q = d["qpos"]
    freq = float(d["frequency"])
    t = np.arange(len(q)) / freq
    names = [str(n) for n in d["site_names"]]
    sx = d["site_xpos"]
    pel = sx[:, names.index("pelvis_mimic"), :]
    xy = pel[:, :2]
    vel = np.diff(xy, axis=0) * freq
    speed = np.linalg.norm(vel, axis=1)
    speed = np.concatenate([[speed[0]], speed])
    path_len = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    net = float(np.linalg.norm(xy[-1] - xy[0]))
    dur = float(t[-1])

    segs = segment_stats(t, xy, speed, 0.2)
    longest = max(segs, key=lambda s: s["duration_s"]) if segs else None

    contacts = {}
    for side in ("left", "right"):
        a = sx[:, names.index(f"{side}_ankle_mimic"), :]
        below = a[:, 2] < a[:, 2].min() + 0.02
        contacts[side] = int(np.sum(np.diff(below.astype(int)) == 1))

    # Precondition for a 10 s acceptance: the first 10 s of the reference must
    # itself be walking at >= 0.6 m/s along the path (net displacement is not
    # used because these clips contain turns).
    i10 = min(len(t) - 1, int(round(10.0 * freq)))
    dur10 = float(t[i10])
    path10 = float(np.sum(np.linalg.norm(np.diff(xy[: i10 + 1], axis=0), axis=1)))
    speed10 = path10 / dur10 if dur10 > 0 else 0.0
    moving10 = float((speed[: i10 + 1] > 0.2).mean())

    return {
        "file": str(path),
        "first_10s": {
            "window_s": dur10,
            "path_length_m": path10,
            "mean_path_speed": speed10,
            "moving_fraction_speed_gt_0.2": moving10,
        },
        "acceptance_precondition_ok": bool(dur >= 10.0 and dur10 >= 9.999 and speed10 >= 0.6),
        "n_frames": int(len(q)),
        "frequency_hz": freq,
        "duration_s": dur,
        "pelvis_height_min": float(pel[:, 2].min()),
        "pelvis_height_max": float(pel[:, 2].max()),
        "net_displacement_m": net,
        "path_length_m": path_len,
        "mean_speed_whole_clip": net / dur if dur else 0.0,
        "mean_instant_speed": float(speed.mean()),
        "max_instant_speed": float(speed.max()),
        "moving_fraction_speed_gt_0.2": float((speed > 0.2).mean()),
        "longest_continuous_walk_segment": longest,
        "all_segments_gt_0.2": segs,
        "foot_contact_onsets": contacts,
        "meets_10s_reference": bool(dur >= 10.0),
        "meets_0.6mps_over_clip": bool(net / dur >= 0.6) if dur else False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=str(Path.home() / ".musclemimic/caches/AMASS/MyoFullBody/gmr"),
    )
    ap.add_argument("--out", default="reports/motion_stats.json")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.root) / "**" / "*.npz"), recursive=True))
    out = [analyse(Path(f)) for f in files]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(
        f"{'motion':44s} {'dur':>6s} {'net_v':>6s} {'path_v':>7s} {'1st10s_v':>8s} "
        f"{'longest_walk':>12s} {'walk_v':>7s} {'precond':>7s}"
    )
    for r in out:
        name = Path(r["file"]).name.replace("_poses.npz", "")
        lw = r["longest_continuous_walk_segment"]
        print(
            f"{name:44s} {r['duration_s']:6.2f} {r['mean_speed_whole_clip']:6.3f} "
            f"{r['path_length_m']/r['duration_s']:7.3f} {r['first_10s']['mean_path_speed']:8.3f} "
            f"{(lw['duration_s'] if lw else 0):12.2f} {(lw['mean_speed'] if lw else 0):7.3f} "
            f"{'OK' if r['acceptance_precondition_ok'] else 'no':>7s}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
