#!/usr/bin/env python3
"""Paired comparison: frozen official policy vs adapted policy, same F0 deficit.

Reads two eval_walk run dirs (A = frozen baseline, B = adapted policy, both at
the same f0_left_lower / reference / seeds / protocol) and prints a per-seed
paired table plus mean/median differences. Also accepts a normal-strength
condition to check for normal-gait regression.

Usage:
    .venv/bin/python scripts/analyze_adapt_pair.py \
        --frozen runs/adapt_baseline/L0.90 --adapted runs/adapt_adapted/L0.90 \
        --normal runs/adapt_adapted_normal/L1.00
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

METRICS = [
    ("survived", "survived_full_window"),
    ("duration_s", "duration_s"),
    ("avg_forward_speed", "avg_forward_speed"),
    ("max_lateral_deviation_m", "max_lateral_deviation"),
    ("max_cross_track_m", "max_cross_track_from_reference"),
    ("net_heading_error_deg", "net_heading_error_deg"),
    ("max_tilt_torso_deg", "max_tilt_torso_deg"),
    ("min_root_height_m", "min_root_height"),
    ("foot_slip_total_m", None),  # computed
    ("end_condition", "end_condition"),
]


def load_seeds(run_dir: str) -> dict[int, dict]:
    out = {}
    for p in sorted(glob.glob(str(Path(run_dir) / "stochastic" / "*.json"))):
        if p.endswith("summary.json"):
            continue
        d = json.loads(Path(p).read_text())
        if isinstance(d, list):
            d = d[-1]
        out[int(d["seed"])] = d
    return out


def foot_slip(d: dict) -> float:
    v = d.get("foot_slip_distance")
    if isinstance(v, dict):
        return float(sum(v.values()))
    return float(v or 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--adapted", required=True)
    ap.add_argument("--normal", default=None, help="adapted policy at f0=1.0 (regression check)")
    args = ap.parse_args()

    frozen = load_seeds(args.frozen)
    adapted = load_seeds(args.adapted)
    seeds = sorted(set(frozen) & set(adapted))
    if len(seeds) < 20:
        print(f"WARNING: only {len(seeds)} common seeds")

    def summarize(ds: dict[int, dict]) -> dict:
        n = len(ds)
        return {
            "n": n,
            "survived": sum(d["survived_full_window"] for d in ds.values()),
            "physical_fall": sum(d["end_condition"] == "physical_fall" for d in ds.values()),
            "median_duration_s": float(np.median([d["duration_s"] for d in ds.values()])),
            "median_avg_forward_speed": float(np.median([d["avg_forward_speed"] for d in ds.values()])),
            "median_max_lateral_deviation_m": float(np.median([d["max_lateral_deviation"] for d in ds.values()])),
            "median_net_heading_error_deg": float(np.median([d["net_heading_error_deg"] for d in ds.values()])),
            "median_min_root_height_m": float(np.median([d["min_root_height"] for d in ds.values()])),
            "median_foot_slip_m": float(np.median([foot_slip(d) for d in ds.values()])),
        }

    sF = summarize(frozen)
    sA = summarize(adapted)
    print("=== SUMMARY ===")
    print(f"{'metric':32s} {'frozen':>12s} {'adapted':>12s}")
    for k in sF:
        print(f"{k:32s} {sF[k]:>12} {sA[k]:>12}")

    print("\n=== PER-SEED PAIRED ===")
    hdr = f"{'seed':>4s} | {'fz_surv':>7s} {'ad_surv':>7s} | {'fz_dur':>6s} {'ad_dur':>6s} | {'fz_lat':>6s} {'ad_lat':>6s} | {'fz_spd':>6s} {'ad_spd':>6s} | {'fz_end':>15s} {'ad_end':>15s}"
    print(hdr)
    dur_diffs, lat_diffs = [], []
    for s in seeds:
        f, a = frozen[s], adapted[s]
        dur_diffs.append(a["duration_s"] - f["duration_s"])
        lat_diffs.append(a["max_lateral_deviation"] - f["max_lateral_deviation"])
        print(
            f"{s:4d} | {str(f['survived_full_window']):>7s} {str(a['survived_full_window']):>7s} | "
            f"{f['duration_s']:6.2f} {a['duration_s']:6.2f} | "
            f"{f['max_lateral_deviation']:6.2f} {a['max_lateral_deviation']:6.2f} | "
            f"{f['avg_forward_speed']:6.2f} {a['avg_forward_speed']:6.2f} | "
            f"{f['end_condition']:>15s} {a['end_condition']:>15s}"
        )

    print("\n=== PAIRED DIFFERENCES (adapted - frozen) ===")
    print(f"  duration_s:        mean={np.mean(dur_diffs):+.3f}  median={np.median(dur_diffs):+.3f}")
    print(f"  max_lateral_dev:   mean={np.mean(lat_diffs):+.3f}  median={np.median(lat_diffs):+.3f}")

    if args.normal:
        normal = load_seeds(args.normal)
        sN = summarize(normal)
        print("\n=== NORMAL-STRENGTH REGRESSION (adapted policy @ f0=1.0) ===")
        print(f"  survived={sN['survived']}/{sN['n']}  median_lat={sN['median_max_lateral_deviation_m']:.3f}  "
              f"median_speed={sN['median_avg_forward_speed']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
