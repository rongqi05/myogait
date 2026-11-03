"""Analyse the left-upper-limb F0 sweep (s_U = 1.0, 0.9, 0.8, 0.7).

Lower limbs and trunk stay at 1.0; only the 32 left-arm muscles are scaled.
Same reference / checkpoint / 20 seeds as the earlier ablation experiment.

Inputs:
  * U1.0 -> runs/f0_ablation/A       (baseline, upper F0 = 1.0)
  * U0.9 -> runs/upper_limb_sweep/U0.9
  * U0.8 -> runs/upper_limb_sweep/U0.8
  * U0.7 -> runs/f0_ablation/B       (upper F0 = 0.7)

Outputs (in runs/upper_limb_sweep/):
  * per_seed.csv     -- one row per seed x level
  * summary.json     -- per-level survival + gait summaries
  * topdown.png      -- top-down trajectories per level/seed

Selection logic for the cane experiment: a level is "feasible" if the frozen
policy still completes the 10 s window in a majority of episodes (>= 10/20)
AND shows a measurable, graded deficit in at least one gait metric relative to
U1.0 (so the cane has something to assist). The chosen range is the set of
consecutive levels meeting both criteria.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _env_builder import build_eval_bundle, env_data, env_model, muscle_groups  # noqa: E402
from three_way_report import path_metrics  # noqa: E402

LEVELS = {"1.0": "runs/f0_ablation/A", "0.9": "runs/upper_limb_sweep/U0.9",
          "0.8": "runs/upper_limb_sweep/U0.8", "0.7": "runs/f0_ablation/B"}
SEEDS = list(range(20))
MIN_SURVIVAL_FOR_FEASIBLE = 10  # >= half of the 20 seeds complete 10 s


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def signed_veer_deg(root_xy, ref_xy):
    """Signed angle (deg) between net policy and net reference displacement; + = left."""
    if len(root_xy) < 2 or len(ref_xy) < 2:
        return float("nan")
    hp = np.arctan2(*(root_xy[-1] - root_xy[0])[::-1])
    hr = np.arctan2(*(ref_xy[-1] - ref_xy[0])[::-1])
    return float(wrap180(np.degrees(hp - hr)))


def final_signed_cross_track(root_xy, ref_xy):
    """Signed terminal lateral offset (m) vs reference path; + = left."""
    dref = np.diff(ref_xy, axis=0)
    step_len = np.linalg.norm(dref, axis=1)
    valid = step_len > 1e-9
    heading = np.zeros_like(dref)
    heading[valid] = dref[valid] / step_len[valid, None]
    normal = np.stack([-heading[:, 1], heading[:, 0]], axis=1)
    policy = root_xy - root_xy[0] + ref_xy[0]
    delta = (policy - ref_xy)[1:]
    return float(np.einsum("i,i->", delta[-1], normal[-1]))


def load_seed(json_path: Path) -> dict:
    meta = json.loads(json_path.read_text())
    if isinstance(meta, list):
        meta = meta[-1]
    d = np.load(json_path.with_suffix(".npz"))
    return {"meta": meta, "npz": d}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/upper_limb_sweep")
    args = ap.parse_args()
    out = Path(args.out)

    # muscle indices for left upper limb (fixed by the model, same for all levels)
    bundle = build_eval_bundle(
        checkpoint_path="artifacts/checkpoints/checkpoint_12500",
        motion_path="CMU/120/cmu120_19_straight_h000",
        start_step=53,
        traj_index=0,
        strict_termination=False,
        horizon=None,
    )
    model = env_model(bundle.env)
    lu = np.asarray(muscle_groups(model)["left_upper"]["indices"], int)

    rows = []
    for level, rel in LEVELS.items():
        jdir = Path(rel) / "stochastic"
        files = [f for f in glob.glob(str(jdir / "*.json")) if not f.endswith("summary.json")]
        assert len(files) == 20, f"{level}: {len(files)} seeds at {rel}"
        for f in files:
            r = load_seed(Path(f))
            meta, d = r["meta"], r["npz"]
            root_xy = np.asarray(d["root_pos"], float)[:, :2]
            ref_xy = np.asarray(d["ref_qpos"], float)[:, :2]
            pm = path_metrics(d)
            rows.append({
                "level": float(level),
                "seed": int(meta["seed"]),
                "end_condition": meta["end_condition"],
                "survived_10s": bool(meta["survived_full_window"]),
                "duration_s": float(meta["duration_s"]),
                "forward_distance_m": float(meta["forward_distance"]),
                "avg_forward_speed": float(meta["avg_forward_speed"]),
                "net_speed": float(meta["net_speed"]),
                "max_lateral_deviation_m": float(meta["max_lateral_deviation"]),
                "max_cross_track_m": float(pm["max_cross_track_m"]),
                "net_cross_drift_m": float(pm["net_cross_drift_m"]),
                "final_cross_track_m": final_signed_cross_track(root_xy, ref_xy),
                "signed_veer_deg": signed_veer_deg(root_xy, ref_xy),
                "mean_heading_error_deg": float(pm["mean_heading_error_deg"]),
                "min_root_height_m": float(meta["min_root_height"]),
                "max_tilt_pelvis_deg": float(meta["max_tilt_pelvis_deg"]),
                "max_tilt_torso_deg": float(meta["max_tilt_torso_deg"]),
                "foot_slip_distance_m": float(sum(meta["foot_slip_distance"].values())) if isinstance(meta["foot_slip_distance"], dict) else float(meta["foot_slip_distance"]),
                "double_support_fraction": float(meta["double_support_fraction"]),
                "mean_activation_left_upper": float(np.mean(d["act"][:, lu])),
                "mean_abs_force_left_upper_N": float(np.mean(np.abs(d["actuator_force"][:, lu]))),
            })

    csv_path = out / "per_seed.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {csv_path}")

    # summary per level
    summary = {}
    for level in ["1.0", "0.9", "0.8", "0.7"]:
        cr = [r for r in rows if r["level"] == float(level)]
        surv = [r for r in cr if r["survived_10s"]]
        summary[level] = {
            "n": len(cr),
            "survived_10s": int(len(surv)),
            "physical_fall": int(sum(r["end_condition"] == "physical_fall" for r in cr)),
            "median_duration_s": float(np.median([r["duration_s"] for r in cr])),
            "median_forward_speed": float(np.median([r["avg_forward_speed"] for r in surv])) if surv else float("nan"),
            "median_max_cross_track_m": float(np.median([r["max_cross_track_m"] for r in cr])),
            "median_net_cross_drift_m": float(np.median([r["net_cross_drift_m"] for r in cr])),
            "median_signed_veer_deg": float(np.median([r["signed_veer_deg"] for r in cr])),
            "median_min_root_height_m": float(np.median([r["min_root_height_m"] for r in cr])),
            "median_max_tilt_torso_deg": float(np.median([r["max_tilt_torso_deg"] for r in cr])),
            "median_foot_slip_m": float(np.median([r["foot_slip_distance_m"] for r in cr])),
            "median_left_upper_activation": float(np.median([r["mean_activation_left_upper"] for r in cr])),
            "median_left_upper_abs_force_N": float(np.median([r["mean_abs_force_left_upper_N"] for r in cr])),
        }

    # feasibility: survived >= MIN and graded deficit vs 1.0 in at least one metric
    base = summary["1.0"]
    for level in ["1.0", "0.9", "0.8", "0.7"]:
        s = summary[level]
        s["feasible"] = bool(
            s["survived_10s"] >= MIN_SURVIVAL_FOR_FEASIBLE
            and level != "1.0"
            and (
                s["median_max_cross_track_m"] > base["median_max_cross_track_m"] * 1.2
                or s["median_signed_veer_deg"] != 0.0
                or abs(s["median_signed_veer_deg"]) > abs(base["median_signed_veer_deg"]) + 5.0
                or s["median_forward_speed"] < base["median_forward_speed"] * 0.9
            )
        )
    feasible_levels = [lvl for lvl in ["0.9", "0.8", "0.7"] if summary[lvl]["feasible"]]
    summary["recommended_range"] = (
        f"{min(feasible_levels)}-{max(feasible_levels)}" if feasible_levels else "none"
    )
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"saved {summary_path}")

    # top-down plots
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    for li, level in enumerate(["1.0", "0.9", "0.8", "0.7"]):
        rel = LEVELS[level]
        for k, s in enumerate([0, 1, 2, 9]):
            ax = axes[li, k]
            d = np.load(str(Path(rel) / "stochastic" / f"stochastic_seed{s}.npz"))
            root = np.asarray(d["root_pos"], float)[:, :2]
            ref = np.asarray(d["ref_qpos"], float)[:, :2]
            ax.plot(ref[:, 0], ref[:, 1], "k--", lw=1, label="ref")
            ax.plot(root[:, 0], root[:, 1], "r-", lw=1, label="policy")
            ax.set_title(f"U{level} s{s}", fontsize=8)
            ax.axis("equal")
    fig.suptitle("Top-down: left-upper-limb F0 sweep (policy red vs reference dashed)")
    fig.tight_layout()
    fig.savefig(out / "topdown.png", dpi=110)
    print(f"saved {out / 'topdown.png'}")

    # print summary table
    print("\nlevel  survived  falls  med_dur   med_fwd  med_xct  med_veer  med_minH  med_tilt  med_slip  feasible")
    for level in ["1.0", "0.9", "0.8", "0.7"]:
        s = summary[level]
        print(f"U{level:<5s} {s['survived_10s']:>3d}/20  {s['physical_fall']:>5d}  "
              f"{s['median_duration_s']:>6.2f}s  {s['median_forward_speed']:>6.3f}  "
              f"{s['median_max_cross_track_m']:>6.3f}m  {s['median_signed_veer_deg']:>+7.1f}  "
              f"{s['median_min_root_height_m']:>6.3f}m  {s['median_max_tilt_torso_deg']:>6.1f}  "
              f"{s['median_foot_slip_m']:>6.3f}m  {s['feasible']}")
    print(f"\nrecommended range for cane experiment: {summary['recommended_range']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
