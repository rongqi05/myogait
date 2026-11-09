#!/usr/bin/env python3
"""Straight-line walk audit over locally available retargeted MyoFullBody clips.

For every clip it reports the quantities the straight-walking baseline needs:

  duration, path length, net displacement, lateral extent, turn angle (net and
  maximum heading change inside a 2 s window), pelvis height, foot contacts and
  the longest sub-window whose heading stays within a tolerance.

The longest straight sub-window is what a straight-line reference can actually
provide; a clip that only becomes straight for 6 s cannot be advertised as a
10 s straight walk.

    .venv/bin/python scripts/motion_straightness.py \
        --roots ~/.musclemimic/caches/AMASS/MyoFullBody/gmr \
        --out reports/motion_straightness.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

MOTION_THRESHOLD = 0.2  # m/s, below this the clip is standing still
HEADING_TOL_DEG = 20.0  # tolerance for "walking a straight line"
SMOOTH_S = 0.2


def load(path: Path):
    d = np.load(path, allow_pickle=True)
    names = [str(n) for n in d["site_names"]]
    freq = float(d["frequency"])
    sx = d["site_xpos"]
    pel = sx[:, names.index("pelvis_mimic"), :]
    return {"freq": freq, "t": np.arange(len(pel)) / freq, "pel": pel, "names": names, "sx": sx}


def heading_series(xy: np.ndarray, freq: float, smooth_s: float = SMOOTH_S):
    """Smoothed walking heading (deg) and speed; heading is NaN while standing."""
    w = max(1, int(round(smooth_s * freq)))
    pad = np.pad(xy, ((w, w), (0, 0)), mode="edge")
    kernel = np.ones(2 * w + 1) / (2 * w + 1)
    xs = np.convolve(pad[:, 0], kernel, mode="valid")
    ys = np.convolve(pad[:, 1], kernel, mode="valid")
    dx = np.gradient(xs) * freq
    dy = np.gradient(ys) * freq
    speed = np.hypot(dx, dy)
    h = np.degrees(np.arctan2(dy, dx))
    h[speed < MOTION_THRESHOLD] = np.nan
    return h, speed


def wrap180(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0


def longest_straight_window(heading: np.ndarray, t: np.ndarray, tol: float) -> dict:
    """Longest contiguous moving span whose heading stays within +-tol."""
    best = {"start_s": 0.0, "end_s": 0.0, "duration_s": 0.0}
    n = len(heading)
    i = 0
    while i < n:
        if not np.isfinite(heading[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and np.isfinite(heading[j + 1]):
            seg = heading[i : j + 2]
            ref = wrap180(seg - heading[i])
            if np.max(np.abs(ref - ref.mean())) > tol:
                break
            j += 1
        dur = float(t[j] - t[i])
        if dur > best["duration_s"]:
            best = {"start_s": float(t[i]), "end_s": float(t[j]), "duration_s": dur}
        i = j + 1
    return best


def analyse(path: Path) -> dict:
    m = load(path)
    t, pel, freq, sx, names = m["t"], m["pel"], m["freq"], m["sx"], m["names"]
    xy = pel[:, :2]
    dur = float(t[-1])
    path_len = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    net = float(np.linalg.norm(xy[-1] - xy[0]))
    lateral_extent = float(np.ptp(xy[:, 1]))

    heading, speed = heading_series(xy, freq)
    valid = np.flatnonzero(np.isfinite(heading))
    net_turn = float(wrap180(heading[valid[-1]] - heading[valid[0]])) if valid.size else 0.0

    max_turn_2s = 0.0
    w = max(2, int(round(2.0 * freq)))
    for i in range(0, max(1, len(heading) - w)):
        seg = heading[i : i + w]
        if not np.isfinite(seg).all():
            continue
        max_turn_2s = max(max_turn_2s, float(np.ptp(wrap180(seg - seg[0]))))

    straight = longest_straight_window(heading, t, HEADING_TOL_DEG)

    lz = sx[:, names.index("left_ankle_mimic"), 2]
    rz = sx[:, names.index("right_ankle_mimic"), 2]
    left_down = lz < lz.min() + 0.02
    right_down = rz < rz.min() + 0.02
    moving = speed > MOTION_THRESHOLD
    support = float(np.mean((left_down | right_down)[moving])) if moving.any() else 0.0
    contacts = {s: int(np.sum(np.diff((left_down if s == "left" else right_down).astype(int)) == 1)) for s in ("left", "right")}

    i10 = min(len(t) - 1, int(round(10.0 * freq)))
    path10 = float(np.linalg.norm(np.diff(xy[: i10 + 1], axis=0), axis=1).sum())

    return {
        "file": str(path),
        "parent_dir": path.parent.name,
        "n_frames": int(len(t)),
        "frequency_hz": freq,
        "duration_s": dur,
        "path_length_m": path_len,
        "net_displacement_m": net,
        "mean_path_speed": path_len / dur if dur else 0.0,
        "lateral_extent_m": lateral_extent,
        "net_turn_deg": net_turn,
        "max_turn_in_2s_deg": max_turn_2s,
        "pelvis_height_min": float(pel[:, 2].min()),
        "pelvis_height_range": float(np.ptp(pel[:, 2])),
        "foot_contact_onsets": contacts,
        "support_fraction_while_moving": support,
        "longest_straight_window": straight,
        "first_10s_path_m": path10,
        "first_10s_speed": path10 / t[i10] if i10 else 0.0,
        "is_10s": bool(dur >= 10.0),
        "is_straight": bool(max_turn_2s <= HEADING_TOL_DEG),
        "is_straight_10s": bool(straight["duration_s"] >= 10.0 and max_turn_2s <= HEADING_TOL_DEG),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        nargs="+",
        default=[str(Path.home() / ".musclemimic/caches/AMASS/MyoFullBody/gmr")],
    )
    ap.add_argument("--out", default="reports/motion_straightness.json")
    args = ap.parse_args()

    files: list[str] = []
    for r in args.roots:
        files += glob.glob(str(Path(r).expanduser() / "**" / "*.npz"), recursive=True)
    rows = [analyse(Path(f)) for f in sorted(set(files))]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))

    print(
        f"{'clip':40s} {'dur':>6s} {'path_v':>7s} {'lat':>6s} {'turn2s':>7s} "
        f"{'straight':>9s} {'10s':>4s} {'s10s':>5s} {'support':>8s} {'zmin':>6s}"
    )
    for r in sorted(rows, key=lambda x: -x["longest_straight_window"]["duration_s"]):
        print(
            f"{Path(r['file']).stem[:40]:40s} {r['duration_s']:6.2f} {r['mean_path_speed']:7.3f} "
            f"{r['lateral_extent_m']:6.2f} {r['max_turn_in_2s_deg']:7.1f} "
            f"{r['longest_straight_window']['duration_s']:9.2f} "
            f"{'yes' if r['is_10s'] else 'no':>4s} {'yes' if r['is_straight_10s'] else 'no':>5s} "
            f"{r['support_fraction_while_moving']:8.3f} {r['pelvis_height_min']:6.3f}"
        )

    n10 = sum(r["is_10s"] for r in rows)
    n10s = sum(r["is_straight_10s"] for r in rows)
    print(f"\nclips={len(rows)}  >=10s={n10}  >=10s AND straight={n10s}")
    if rows:
        best = max(rows, key=lambda r: r["longest_straight_window"]["duration_s"])
        print(f"longest straight window: {best['longest_straight_window']['duration_s']:.2f}s in {Path(best['file']).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
