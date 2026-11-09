#!/usr/bin/env python3
"""Audit an AMASS ``.tar.bz2`` archive for continuous straight walking, without extracting.

Streams the archive once, reads only ``trans`` and ``mocap_framerate`` from each
``*_poses.npz`` (SMPL-H AMASS layout) and measures motion from the root
translation. The vertical axis is detected per clip because AMASS subsets are not
consistently y-up. Works for any AMASS subset, e.g.:

    .venv/bin/python scripts/amass_kit_audit.py --archive ~/Downloads/KIT.tar.bz2 \
        --out reports/amass_kit_audit.json

    .venv/bin/python scripts/amass_kit_audit.py --archive ~/Downloads/CMU.tar.bz2 \
        --out reports/amass_cmu_audit.json --keyword walk
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
    # np.convolve(mode="same") returns max(w, n) samples, so a clip shorter than
    # the smoothing window must shrink the window instead of growing the result.
    w = min(w, n)
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
    best = {"start_s": 0.0, "end_s": 0.0, "duration_s": 0.0, "i": 0, "j": 0}
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
            best = {"start_s": float(t[i]), "end_s": float(t[j]), "duration_s": dur, "i": i, "j": j}
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

    # A long window that barely moves is not usable walking, so measure the
    # window itself rather than only the whole clip.
    i, j = straight["i"], straight["j"]
    wxy = xy[i : j + 1]
    wtime = straight["duration_s"]
    w_net = float(np.linalg.norm(wxy[-1] - wxy[0])) if len(wxy) > 1 else 0.0
    w_path = float(np.linalg.norm(np.diff(wxy, axis=0), axis=1).sum()) if len(wxy) > 1 else 0.0
    wh = h[i : j + 1]
    w_turn = float(np.ptp(wrap180(wh - wh[0]))) if np.isfinite(wh).all() else float("nan")
    straight = dict(
        straight,
        net_displacement_m=w_net,
        path_length_m=w_path,
        mean_speed_mps=w_net / wtime if wtime else 0.0,
        path_speed_mps=w_path / wtime if wtime else 0.0,
        heading_range_deg=w_turn,
        straightness=w_net / w_path if w_path else 0.0,
    )

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
        # gate used by the frozen protocol: >=10 s of window AND walking fast
        # enough to be comparable with the 0.874 m/s reference.
        "is_usable_10s": bool(straight["duration_s"] >= 10.0 and straight["mean_speed_mps"] >= 0.6),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--keyword",
        default="walk",
        help="case-insensitive clip-name keyword counted as '(keyword)-labelled' (default: walk)",
    )
    args = ap.parse_args()

    rows, errors = [], []
    with tarfile.open(Path(args.archive).expanduser(), "r:bz2") as tf:
        for member in tf:
            if not member.name.endswith(".npz"):
                continue
            # AMASS also ships <subject>/shape.npz; it has no motion in it.
            if member.name.endswith("/shape.npz") or member.name.endswith("shape.npz"):
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

    keyword = [r for r in rows if re.search(re.escape(args.keyword), r["clip"], re.I)]
    print(f"clips={len(rows)} errors={len(errors)}  {args.keyword}-labelled={len(keyword)}")
    print(f"duration>=10s: {sum(r['is_10s'] for r in rows)}   >=20s: {sum(r['is_20s'] for r in rows)}")
    print(f"straight window>=10s: {sum(r['is_straight_10s'] for r in rows)}   >=20s: {sum(r['is_straight_20s'] for r in rows)}")
    print(f"usable  >=10s & >=0.6 m/s: {sum(r['is_usable_10s'] for r in rows)}")
    print()
    print(f"{'clip':22s} {'dur':>7s} {'v_clip':>7s} {'straight':>9s} {'w_net':>7s} {'w_speed':>8s} {'w_turn':>7s} {'w_str':>6s}")
    for r in sorted(rows, key=lambda x: -x["longest_straight_window"]["duration_s"])[:20]:
        w = r["longest_straight_window"]
        print(
            f"{r['clip'][:22]:22s} {r['duration_s']:7.2f} {r['mean_path_speed']:7.3f} "
            f"{w['duration_s']:9.2f} {w['net_displacement_m']:7.2f} {w['mean_speed_mps']:8.3f} "
            f"{w['heading_range_deg']:7.1f} {w['straightness']:6.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
