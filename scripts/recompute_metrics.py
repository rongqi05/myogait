#!/usr/bin/env python3
"""Recompute per-rollout metrics from stored rollouts with one metric version.

Used when a metric definition changes: the physics in the ``.npz`` files does not
change, only the derived numbers. Applying it to every reference group with the
same code keeps the comparison honest.

    .venv/bin/python scripts/recompute_metrics.py \
        --group old=runs/A_ref_mm10m2/stochastic --group trained=runs/A_straight_10s/stochastic \
        --out reports/metrics_recomputed.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

HEAD_W = 50  # 0.5 s at 100 Hz
MOTION_SPEED = 0.2


def windowed_heading(xy: np.ndarray, w: int = HEAD_W) -> np.ndarray:
    out = np.full(len(xy), np.nan)
    for i in range(len(xy)):
        a = max(0, i - w // 2)
        b = min(len(xy), i + w // 2 + 1)
        seg = xy[a:b] - xy[a:b].mean(0)
        if len(seg) < 3 or np.linalg.norm(seg[-1] - seg[0]) < 1e-3:
            continue
        _, _, vt = np.linalg.svd(seg, full_matrices=False)
        v = vt[0]
        if np.dot(v, xy[b - 1] - xy[a]) < 0:
            v = -v
        out[i] = np.degrees(np.arctan2(v[1], v[0]))
    return out


def windowed_speed(xy: np.ndarray, dt: float, w: int = HEAD_W) -> np.ndarray:
    if len(xy) < 2:
        return np.zeros(len(xy))
    sp = np.concatenate([[0.0], np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt])
    return np.convolve(sp, np.ones(w) / w, mode="same")


def metrics(path: Path) -> dict:
    d = np.load(path)
    t = d["time"]
    dt = float(np.median(np.diff(t)))
    root = d["root_pos"][:, :2]
    ref = d["ref_qpos"][:, :2]

    hp = windowed_heading(root)
    hr = windowed_heading(ref)
    vp = windowed_speed(root, dt)
    vr = windowed_speed(ref, dt)
    valid = (vp > MOTION_SPEED) & (vr > MOTION_SPEED) & np.isfinite(hp) & np.isfinite(hr)
    err = np.abs((hp[valid] - hr[valid] + 180.0) % 360.0 - 180.0)

    net = float(
        abs(
            np.degrees(
                (np.arctan2(*(root[-1] - root[0])[::-1]) - np.arctan2(*(ref[-1] - ref[0])[::-1]) + np.pi)
                % (2 * np.pi)
                - np.pi
            )
        )
    )

    fwd = 0.0
    ref_fwd = 0.0
    for i in range(1, len(root)):
        dr = ref[i] - ref[i - 1]
        n = np.linalg.norm(dr)
        if n < 1e-9:
            continue
        h = dr / n
        fwd += float(np.dot(root[i] - root[i - 1], h))
        ref_fwd += n

    return {
        "steps": int(len(t)),
        "duration_s": float(t[-1] - t[0]),
        "mean_heading_error_deg": float(err.mean()) if err.size else float("nan"),
        "max_heading_error_deg": float(err.max()) if err.size else float("nan"),
        "p90_heading_error_deg": float(np.percentile(err, 90)) if err.size else float("nan"),
        "net_heading_error_deg": net,
        "valid_heading_frames": int(valid.sum()),
        "low_speed_frames": int((~valid).sum()),
        "policy_speed_mps": float(vp[valid].mean()) if valid.any() else float("nan"),
        "reference_speed_mps": float(vr[valid].mean()) if valid.any() else float("nan"),
        "forward_distance_m": fwd,
        "reference_distance_m": ref_fwd,
        "progress_ratio": fwd / ref_fwd if ref_fwd > 0 else float("nan"),
        "final_root": [float(x) for x in root[-1]],
        "final_ref": [float(x) for x in ref[-1]],
        "final_offset_m": float(np.linalg.norm(root[-1] - ref[-1])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True, help="NAME=DIR (repeatable)")
    ap.add_argument("--out", default="reports/metrics_recomputed.json")
    args = ap.parse_args()

    result = {"metric_version": "heading-v2 (0.5 s windowed PCA, speed gated at 0.2 m/s)", "groups": {}}
    summary = {}
    for spec in args.group:
        name, d = spec.split("=", 1)
        files = sorted(glob.glob(str(Path(d) / "*_seed*.npz")), key=lambda f: int(re.search(r"seed(\d+)", f).group(1)))
        rows = []
        for f in files:
            m = metrics(Path(f))
            m["seed"] = int(re.search(r"seed(\d+)", f).group(1))
            m["file"] = f
            rows.append(m)
        result["groups"][name] = {"dir": d, "n": len(rows), "per_seed": rows}

        def stat(k, fn):
            v = np.array([r[k] for r in rows], dtype=float)
            return float(fn(v)) if v.size else float("nan")

        summary[name] = {
            "n": len(rows),
            "mean_heading_error_deg": stat("mean_heading_error_deg", np.mean),
            "max_heading_error_deg": stat("max_heading_error_deg", np.mean),
            "net_heading_error_deg": stat("net_heading_error_deg", np.mean),
            "policy_speed_mps": stat("policy_speed_mps", np.mean),
            "reference_speed_mps": stat("reference_speed_mps", np.mean),
            "progress_ratio": stat("progress_ratio", np.mean),
            "final_offset_m": stat("final_offset_m", np.mean),
        }

    result["summary"] = summary
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, default=float))

    print(
        f"{'group':10s} {'n':>3s} {'mean_hdg':>9s} {'max_hdg':>8s} {'net_hdg':>8s} "
        f"{'v_pol':>6s} {'v_ref':>6s} {'progress':>8s} {'offset':>7s}"
    )
    for name, s in summary.items():
        print(
            f"{name:10s} {s['n']:3d} {s['mean_heading_error_deg']:9.1f} {s['max_heading_error_deg']:8.1f} "
            f"{s['net_heading_error_deg']:8.1f} {s['policy_speed_mps']:6.3f} {s['reference_speed_mps']:6.3f} "
            f"{s['progress_ratio']:8.3f} {s['final_offset_m']:7.3f}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
