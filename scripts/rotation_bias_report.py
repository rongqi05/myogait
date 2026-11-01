#!/usr/bin/env python3
"""Decide whether the policy's walking veer is body-fixed or world-fixed.

The reference is rigidly rotated about z, which changes nothing about the gait:
same joint angles, same foot placement relative to the pelvis, same speed. Only
the world heading changes. Because the per-episode veer is measured against the
reference's own heading, it is already relative to the body frame, so:

  * veer roughly constant across rotations -> the veer is fixed in the body /
    reference frame, i.e. the policy or the model is biased. Rotating the world
    cannot help.
  * veer rotating with the world (i.e. drift always toward the same world
    direction) -> something in the setup is world-referenced.

    .venv/bin/python scripts/rotation_bias_report.py \
        --group 000=runs/ROT_000/stochastic --group 090=runs/ROT_090/stochastic \
        --group 180=runs/ROT_180/stochastic --group 270=runs/ROT_270/stochastic
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np


def load_group(directory: Path) -> dict:
    files = sorted(glob.glob(str(directory / "*_seed*.npz")), key=lambda f: int(re.search(r"seed(\d+)", f).group(1)))
    if not files:
        raise SystemExit(f"no seed npz in {directory}")
    rows = []
    for f in files:
        d = np.load(f)
        t = np.asarray(d["time"], float)
        root = np.asarray(d["root_pos"], float)[:, :2]
        ref = np.asarray(d["ref_qpos"], float)[:, :2]

        dref = np.diff(ref, axis=0)
        step = np.linalg.norm(dref, axis=1)
        ok = step > 1e-9
        heading = np.zeros_like(dref)
        heading[ok] = dref[ok] / step[ok, None]
        normal = np.stack([-heading[:, 1], heading[:, 0]], axis=1)

        policy = root - root[0] + ref[0]
        droot = np.diff(policy, axis=0)
        # veer components, both measured against the reference frame
        along = float(np.einsum("ij,ij->i", droot, heading).sum())
        cross = float(np.einsum("ij,ij->i", droot, normal).sum())
        ref_along = float(step[ok].sum())

        # world-frame heading of the reference and of the policy path
        ref_world = float(np.degrees(np.arctan2(*(ref[-1] - ref[0])[::-1])))
        pol_world = float(np.degrees(np.arctan2(*(policy[-1] - policy[0])[::-1])))

        rows.append(
            {
                "seed": int(re.search(r"seed(\d+)", f).group(1)),
                "duration_s": float(t[-1] - t[0]),
                "policy_along_m": along,
                "policy_cross_m": cross,
                "reference_along_m": ref_along,
                "progress_ratio": along / ref_along if ref_along else float("nan"),
                "veer_deg_vs_reference": float(np.degrees(np.arctan2(cross, along))) if along else float("nan"),
                "reference_world_heading_deg": ref_world,
                "policy_world_heading_deg": pol_world,
                "world_crab_deg": float((pol_world - ref_world + 180.0) % 360.0 - 180.0),
            }
        )
    return {"dir": str(directory), "n": len(rows), "per_seed": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append", required=True, help="ROTATION_DEG=DIR (repeatable)")
    ap.add_argument("--out", default="reports/rotation_bias.json")
    args = ap.parse_args()

    groups = {}
    for spec in args.group:
        name, d = spec.split("=", 1)
        groups[name] = load_group(Path(d))

    print(f"{'rot':>5s} {'n':>3s} {'veer_vs_ref':>12s} {'sd':>6s} {'policy_along':>13s} {'progress':>9s} {'world_crab':>11s}")
    rows = []
    for name, g in groups.items():
        v = np.array([r["veer_deg_vs_reference"] for r in g["per_seed"]], float)
        a = np.array([r["policy_along_m"] for r in g["per_seed"]], float)
        pr = np.array([r["progress_ratio"] for r in g["per_seed"]], float)
        wc = np.array([r["world_crab_deg"] for r in g["per_seed"]], float)
        rows.append(
            {
                "rotation_deg": float(name),
                "n": g["n"],
                "veer_vs_reference_mean_deg": float(v.mean()),
                "veer_vs_reference_sd_deg": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "policy_along_mean_m": float(a.mean()),
                "progress_ratio_mean": float(pr.mean()),
                "world_crab_mean_deg": float(wc.mean()),
            }
        )
        print(
            f"{name:>5s} {g['n']:3d} {v.mean():12.2f} {v.std(ddof=1) if len(v) > 1 else 0:6.2f} "
            f"{a.mean():13.3f} {pr.mean():9.3f} {wc.mean():11.2f}"
        )

    veers = np.array([r["veer_vs_reference_mean_deg"] for r in rows])
    worlds = np.array([r["world_crab_mean_deg"] for r in rows])
    verdict = {
        "veer_spread_across_rotations_deg": float(np.ptp(veers)),
        "world_crab_spread_across_rotations_deg": float(np.ptp(worlds)),
        "body_fixed": bool(np.ptp(veers) < 5.0),
        "world_fixed": bool(np.ptp(worlds) < 5.0),
    }
    print()
    print(f"veer-vs-reference spread across rotations: {verdict['veer_spread_across_rotations_deg']:.2f} deg")
    print(f"world-frame crab spread across rotations : {verdict['world_crab_spread_across_rotations_deg']:.2f} deg")
    if verdict["body_fixed"] and not verdict["world_fixed"]:
        print("=> body-fixed veer: the bias belongs to the policy/model, not to the world frame")
    elif verdict["world_fixed"] and not verdict["body_fixed"]:
        print("=> world-fixed veer: something in the setup is world-referenced")
    else:
        print("=> ambiguous; inspect per-seed values")

    out = Path(__file__).resolve().parent.parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"groups": groups, "summary": rows, "verdict": verdict}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
