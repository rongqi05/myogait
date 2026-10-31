#!/usr/bin/env python3
"""Evidence that motion comes from muscle force, not from state injection.

Reads a rollout produced by scripts/eval_walk.py and reports:

  * policy action -> muscle ctrl range / saturation
  * muscle activation (data.act) dynamics
  * actuator force statistics and their correlation with mean activation
  * root speed produced by those forces

    .venv/bin/python scripts/check_muscle_chain.py runs/A_straight_10s/stochastic/stochastic_seed0.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for path in args.npz:
        d = np.load(path)
        ctrl = d["ctrl"]
        act = d["act"]
        force = d["actuator_force"]
        qvel = d["qvel"]

        a_mean = act.mean(axis=1)
        f_mean = force.mean(axis=1)
        corr = float(np.corrcoef(a_mean, f_mean)[0, 1]) if len(a_mean) > 2 else float("nan")

        rows.append(
            {
                "file": str(path),
                "steps": int(len(ctrl)),
                "ctrl_min": float(ctrl.min()),
                "ctrl_max": float(ctrl.max()),
                "ctrl_abs_mean": float(np.abs(ctrl).mean()),
                "ctrl_saturation_fraction": float(np.mean(np.abs(ctrl) > 0.99)),
                "act_min": float(act.min()),
                "act_max": float(act.max()),
                "act_mean": float(act.mean()),
                "act_std_over_time": float(np.std(a_mean)),
                "force_abs_mean": float(np.abs(force).mean()),
                "force_max": float(np.abs(force).max()),
                "corr_activation_vs_force": corr,
                "root_speed_mean": float(np.linalg.norm(qvel[:, :3], axis=1).mean()),
            }
        )

    text = json.dumps({"per_rollout": rows}, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)

    for r in rows:
        print(f"{Path(r['file']).name}  steps={r['steps']}")
        print(
            f"  ctrl[{r['ctrl_min']:+.2f},{r['ctrl_max']:+.2f}] |ctrl|={r['ctrl_abs_mean']:.3f} "
            f"sat={r['ctrl_saturation_fraction']:.3f}"
        )
        print(
            f"  act[{r['act_min']:.3f},{r['act_max']:.3f}] mean={r['act_mean']:.3f} "
            f"std_over_time={r['act_std_over_time']:.3f}"
        )
        print(
            f"  |force| mean={r['force_abs_mean']:.1f} N max={r['force_max']:.1f} N "
            f"corr(act,force)={r['corr_activation_vs_force']:.3f}"
        )
        print(f"  root speed mean={r['root_speed_mean']:.3f} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
