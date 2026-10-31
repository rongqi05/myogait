#!/usr/bin/env python3
"""Audit BVH clips for continuous straight walking (duration, path, heading).

Reads only the HIERARCHY/MOTION header plus the root translation channels, so it
is cheap enough to scan hundreds of files. The checks mirror
scripts/motion_straightness.py so results are comparable with the retargeted
caches: straight = maximum heading change inside any 2 s window <= 20 deg while
moving faster than 0.2 m/s.

    .venv/bin/python scripts/bvh_straightness.py --root ~/Documents/muscle/data/cmu_mocap \
        --out reports/bvh_straightness_cmu.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

MOTION_SPEED = 0.2
HEADING_TOL_DEG = 20.0


def parse_bvh_header(path: Path) -> dict:
    """Return frame count, frame time and the root translation channel indices."""
    root_channels: list[str] = []
    in_root = False
    depth = 0
    frames = None
    frame_time = None
    header_lines = 0
    y_offsets: list[float] = []

    with open(path, "r", errors="replace") as f:
        for line in f:
            header_lines += 1
            s = line.strip()
            low = s.lower()
            if low.startswith("root"):
                in_root = True
                depth = 0
            if in_root and low.startswith("channels"):
                # BVH writes "CHANNELS 6 Xposition Yposition ..." (no colon)
                toks = s.replace(":", " ").split()
                root_channels = toks[2:]
                in_root = False
            if s.startswith("OFFSET"):
                parts = s.split()
                if len(parts) >= 4:
                    y_offsets.append(float(parts[2]))
            if low.startswith("frames:"):
                frames = int(s.split(":")[1])
            if low.startswith("frame time:"):
                frame_time = float(s.split(":")[1])
                break

    if frames is None or frame_time is None:
        raise ValueError(f"no MOTION header in {path}")
    trans_idx = [i for i, c in enumerate(root_channels) if "position" in c.lower()]
    if len(trans_idx) < 2:
        raise ValueError(f"root has no translation channels in {path}: {root_channels}")
    # skeleton height proxy: sum of |y| offsets along the rest pose
    height_units = float(np.abs(np.asarray(y_offsets, dtype=float)).sum()) if y_offsets else float("nan")
    return {
        "frames": frames,
        "frame_time": frame_time,
        "n_channels": len(root_channels),
        "trans_idx": trans_idx,
        "header_lines": header_lines,
        "height_units": height_units,
    }


def read_root_xy(path: Path, hdr: dict) -> np.ndarray:
    """Root translation (x, y) per frame."""
    n = hdr["frames"]
    ch = hdr["n_channels"]
    i0, i1 = hdr["trans_idx"][0], hdr["trans_idx"][1]
    xy = np.zeros((n, 2), dtype=np.float64)
    with open(path, "r", errors="replace") as f:
        for _ in range(hdr["header_lines"]):
            f.readline()
        for k in range(n):
            line = f.readline()
            if not line:
                return xy[:k]
            vals = line.split()
            if len(vals) < ch:
                return xy[:k]
            xy[k, 0] = float(vals[i0])
            xy[k, 1] = float(vals[i1])
    return xy


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def heading_series(xy: np.ndarray, fps: float, smooth_s: float = 0.5):
    """Windowed PCA heading + windowed speed.

    Per-frame finite differences at 120 Hz are ~1 mm and dominated by sway, so
    the direction is estimated from a 0.5 s window instead (same convention as
    the policy metrics).
    """
    w = max(3, int(round(smooth_s * fps)))
    n = len(xy)
    h = np.full(n, np.nan)
    k = np.ones(w) / w
    sp = np.concatenate([[0.0], np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps]) if n > 1 else np.zeros(n)
    speed = np.convolve(sp, k, mode="same")
    for i in range(n):
        a, b = max(0, i - w // 2), min(n, i + w // 2 + 1)
        seg = xy[a:b] - xy[a:b].mean(0)
        if len(seg) < 3 or np.linalg.norm(seg[-1] - seg[0]) < 1e-6:
            continue
        _, _, vt = np.linalg.svd(seg, full_matrices=False)
        v = vt[0]
        if np.dot(v, xy[b - 1] - xy[a]) < 0:
            v = -v
        h[i] = np.degrees(np.arctan2(v[1], v[0]))
    h[speed < MOTION_SPEED] = np.nan
    return h, speed


def longest_straight(heading, t, tol):
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
    hdr = parse_bvh_header(path)
    fps = 1.0 / hdr["frame_time"]
    xy = read_root_xy(path, hdr)
    n = len(xy)
    t = np.arange(n) / fps
    dur = float(t[-1]) if n > 1 else 0.0
    path_len = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if n > 1 else 0.0
    net = float(np.linalg.norm(xy[-1] - xy[0])) if n > 1 else 0.0

    h, speed = heading_series(xy, fps)
    valid = np.flatnonzero(np.isfinite(h))
    max_turn_2s = 0.0
    w2 = max(2, int(round(2.0 * fps)))
    for i in range(0, max(1, len(h) - w2)):
        seg = h[i : i + w2]
        if not np.isfinite(seg).all():
            continue
        max_turn_2s = max(max_turn_2s, float(np.ptp(wrap180(seg - seg[0]))))
    straight = longest_straight(h, t, HEADING_TOL_DEG)

    return {
        "file": str(path),
        "subject": path.parent.name,
        "frames": n,
        "fps": fps,
        "duration_s": dur,
        "path_length_m": path_len,
        "net_displacement_m": net,
        "mean_path_speed": path_len / dur if dur else 0.0,
        "skeleton_units": hdr["height_units"],
        "speed_in_body_heights_per_s": (path_len / dur / hdr["height_units"]) if dur and hdr["height_units"] > 0 else float("nan"),
        "max_turn_in_2s_deg": max_turn_2s,
        "moving_fraction": float(np.mean(speed > MOTION_SPEED)),
        "longest_straight_window": straight,
        "is_10s": bool(dur >= 10.0),
        "is_straight_10s": bool(straight["duration_s"] >= 10.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.root).expanduser() / "**" / "*.bvh"), recursive=True))
    rows, errors = [], []
    for f in files:
        try:
            rows.append(analyse(Path(f)))
        except Exception as exc:
            errors.append({"file": f, "error": str(exc)})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"clips": rows, "errors": errors}, indent=2))

    ok = [r for r in rows if r["is_straight_10s"]]
    print(f"scanned {len(files)} bvh files, parsed {len(rows)}, errors {len(errors)}")
    print(f"duration >= 10 s: {sum(r['is_10s'] for r in rows)}")
    print(f">= 10 s AND straight (<= {HEADING_TOL_DEG} deg in any 2 s): {len(ok)}")
    print()
    print(f"{'clip':22s} {'dur':>6s} {'straight':>9s} {'turn2s':>7s} {'height/s':>9s} {'units':>7s}")
    for r in sorted(rows, key=lambda x: -x["longest_straight_window"]["duration_s"])[:15]:
        w = r["longest_straight_window"]
        print(
            f"{Path(r['file']).stem[:22]:22s} {r['duration_s']:6.2f} {w['duration_s']:9.2f} "
            f"{r['max_turn_in_2s_deg']:7.1f} {r['speed_in_body_heights_per_s']:9.3f} {r['skeleton_units']:7.2f}"
        )
    if ok:
        print("\n--- qualifying clips ---")
        for r in sorted(ok, key=lambda x: -x["longest_straight_window"]["duration_s"]):
            w = r["longest_straight_window"]
            print(
                f"{r['file']}  dur={r['duration_s']:.2f}s straight={w['duration_s']:.2f}s "
                f"[{w['start_s']:.2f},{w['end_s']:.2f}] v={r['mean_path_speed']:.3f} m/s turn2s={r['max_turn_in_2s_deg']:.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
