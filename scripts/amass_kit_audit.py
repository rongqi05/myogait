#!/usr/bin/env python3
"""Audit the AMASS KIT archive for continuous straight walking, without extracting.

Streams the ``.tar.bz2`` once, reads only ``trans`` and ``mocap_framerate`` from
each ``*_poses.npz`` (SMPL-H AMASS layout) and measures motion from the root
translation. The vertical axis is detected per clip because KIT clips are not
consistently y-up.

    .venv/bin/python scripts/amass_kit_audit.py --archive ~/Downloads/KIT.tar.bz2 \
        --out reports/amass_kit_audit.json
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from pathlib import Path

import numpy as np

MOTION_SPEED = 0.2
HEADING_TOL_DEG = 20.0
HEAD_W_S = 0.5


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def windowed_heading(xy: np.ndarray, fps: float, w_s: float = HEAD_W_S):
    w = max(3, int(round(w_s * fps)))
    n = len(xy)
    h = np.full(n, np.nan)
    if n < 3:
        return h, np.zeros(n)
    sp = np.concatenate([[0.0], np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps])
    speed = np.convolve(sp, np.ones(w) / w, mode="same")
    for i in range(n):
        a, b = max(0, i - w // 2), min(n, i + w // 2 + 1)
        seg = xy[a:b] - xy[a:b].mean(0)
        if len(seg) < 3:
            continue
        v = xy[b - 1] - xy[a]
        if np.linalg.norm(v) < 1e-6:
            continue
        _, _, vt = np.linalg.svd(seg, full_matrices=False)
        d = vt[0]
        if np.dot(d, v) < 0:
            d = -d
        h[i] = np.degrees(np.arctan2(d[1], d[0]))
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


def analyse(name: str, trans: np.ndarray, fps: float) -> dict:
    t = np.arange(len(trans)) / fps
    dur = float(t[-1]) if len(t) > 1 else 0.0
    rng = np.ptp(trans, axis=0)
    up = int(np.argmin(rng))
    horiz = [i for i in range(3) if i != up]
    xy = trans[:, horiz]

    path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0
    net = float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) > 1 else 0.0

    h, speed = windowed_heading(xy, fps)
    w2 = max(2, int(round(2.0 * fps)))
    max_turn = 0.0
    for i in range(0, max(1, len(h) - w2)):
        seg = h[i : i + w2]
        if not np.isfinite(seg).all():
            continue
        max_turn = max(max_turn, float(np.ptp(wrap180(seg - seg[0]))))
    straight = longest_straight(h, t, HEADING_TOL_DEG)

    return {
        "file": name,
        "subject": name.split("/")[1] if name.count("/") >= 1 else "?",
        "clip": Path(name).stem.replace("_poses", ""),
        "frames": int(len(trans)),
        "fps": fps,
        "duration_s": dur,
        "up_axis": up,
        "path_length_m": path,
        "net_displacement_m": net,
        "mean_path_speed": path / dur if dur else 0.0,
        "max_turn_in_2s_deg": max_turn,
        "moving_fraction": float(np.mean(speed > MOTION_SPEED)) if len(speed) else 0.0,
        "longest_straight_window": straight,
        "is_10s": bool(dur >= 10.0),
        "is_20s": bool(dur >= 20.0),
        "is_straight_10s": bool(straight["duration_s"] >= 10.0),
        "is_straight_20s": bool(straight["duration_s"] >= 20.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows, errors = [], []
    with tarfile.open(Path(args.archive).expanduser(), "r:bz2") as tf:
        for member in tf:
            if not member.name.endswith(".npz"):
                continue
            try:
                with np.load(io.BytesIO(tf.extractfile(member).read()), allow_pickle=True) as d:
                    trans = np.asarray(d["trans"], dtype=np.float64)
                    fps = float(d["mocap_framerate"])
                rows.append(analyse(member.name, trans, fps))
            except Exception as exc:
                errors.append({"file": member.name, "error": str(exc)})

    out = {"clips": rows, "errors": errors, "n_clips": len(rows)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    walk = [r for r in rows if re.search(r"walk", r["clip"], re.I)]
    print(f"clips={len(rows)} errors={len(errors)}  walk-labelled={len(walk)}")
    print(f"duration>=10s: {sum(r['is_10s'] for r in rows)}   >=20s: {sum(r['is_20s'] for r in rows)}")
    print(f"straight window>=10s: {sum(r['is_straight_10s'] for r in rows)}   >=20s: {sum(r['is_straight_20s'] for r in rows)}")
    print()
    print(f"{'clip':34s} {'dur':>6s} {'v':>6s} {'turn2s':>7s} {'straight':>9s}")
    for r in sorted(rows, key=lambda x: -x["longest_straight_window"]["duration_s"])[:20]:
        print(
            f"{r['clip'][:34]:34s} {r['duration_s']:6.2f} {r['mean_path_speed']:6.3f} "
            f"{r['max_turn_in_2s_deg']:7.1f} {r['longest_straight_window']['duration_s']:9.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
