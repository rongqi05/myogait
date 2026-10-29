#!/usr/bin/env python3
"""Render a per-seed markdown table from a run directory produced by eval_walk.py.

    .venv/bin/python scripts/make_table.py runs/C_acceptance/stochastic \
        --out reports/table_walk.md
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

HEADER = (
    "| seed | reason | dur (s) | v_fwd (m/s) | lat dev (m) | x-track (m) | "
    "tilt pelvis/trunk (deg) | min root z (m) | onsets | slip/m L/R | clip clean |"
)
SEP = "|---:|---|---:|---:|---:|---:|---|---:|---:|---|---|"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="directory containing *_seed*.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.run_dir) / "*_seed*.json")), key=lambda f: int(re.search(r"seed(\d+)", f).group(1)))
    if not files:
        raise SystemExit(f"no per-seed json in {args.run_dir}")

    lines = [HEADER, SEP]
    for f in files:
        r = json.load(open(f))
        lines.append(
            "| {seed} | {reason} | {duration_s:.2f} | {avg_forward_speed:.3f} | "
            "{max_lateral_deviation:.3f} | {max_cross_track_from_reference:.3f} | "
            "{max_tilt_pelvis_deg:.1f} / {max_tilt_torso_deg:.1f} | {min_root_height:.3f} | "
            "{n_foot_contact_onsets_total} | {sl:.3f} / {sr:.3f} | {cc} |".format(
                sl=r["slip_per_meter"]["left"],
                sr=r["slip_per_meter"]["right"],
                cc="yes" if r["clip_completed_without_failure"] else "no",
                **r,
            )
        )
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(files)} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
