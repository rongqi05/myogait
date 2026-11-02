"""Analyse the upper/lower-limb F0 ablation experiment (A-E, 20 seeds each).

Produces, from the per-seed JSON + NPZ saved by ``eval_walk.py``:

* ``runs/f0_ablation/per_seed.csv``       -- one row per seed x condition
* ``runs/f0_ablation/common_window.csv``  -- first-2s common-time-window metrics
* ``runs/f0_ablation/muscle_force.csv``   -- upper/lower activation + active/passive force
* ``runs/f0_ablation/summary.json``       -- per-condition summaries + paired diffs
* ``runs/f0_ablation/topdown.png``        -- top-down trajectories per seed/condition

Metric definitions (same as the frozen protocol where applicable):
  * cross-track / net drift / heading error: ``three_way_report.path_metrics``
    (re-anchored to the reference start, reference-path normal; +y = left).
  * signed terminal lateral offset: final (policy - reference) . normal.
  * signed heading change: wrap(final_heading - initial_heading); + = veer left.
  * "active force" = recorded total actuator force - passive force recomputed at
    activation 0 with the same qpos/qvel on a fresh baseline model (passive force
    does not depend on F0 or activation).
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
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _env_builder import build_eval_bundle, env_data, env_model, muscle_groups  # noqa: E402
from three_way_report import path_metrics  # noqa: E402

CONDS = ["A", "B", "C", "D", "E"]
SEEDS = list(range(20))
COMMON_WINDOW = 2.0  # s
FORCE_SAMPLE_HZ = 10.0


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def signed_veer_deg(root_xy, ref_xy):
    """Signed angle (deg) between net policy displacement and net reference
    displacement. + = veer left (matches |.| of ``net_heading_error_deg``)."""
    if len(root_xy) < 2 or len(ref_xy) < 2:
        return float("nan")
    hp = np.arctan2(*(root_xy[-1] - root_xy[0])[::-1])
    hr = np.arctan2(*(ref_xy[-1] - ref_xy[0])[::-1])
    return float(wrap180(np.degrees(hp - hr)))


def final_signed_cross_track(root_xy, ref_xy):
    """Signed terminal lateral offset (m) vs the reference path. + = left."""
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
    ap.add_argument("--runs_dir", default="runs/f0_ablation")
    args = ap.parse_args()
    base = Path(args.runs_dir)

    # -- one baseline model for the passive-force recomputation and indices ----
    bundle = build_eval_bundle(
        checkpoint_path="artifacts/checkpoints/checkpoint_12500",
        motion_path="CMU/120/cmu120_19_straight_h000",
        start_step=53,
        traj_index=0,
        strict_termination=False,
        horizon=None,
    )
    model = env_model(bundle.env)
    data = env_data(bundle.env)
    groups = muscle_groups(model)
    lu = np.asarray(groups["left_upper"]["indices"], int)
    ll = np.asarray(groups["left_lower"]["indices"], int)

    # -- per-seed table --------------------------------------------------------
    per_seed = []
    for cond in CONDS:
        jdir = base / cond / "stochastic"
        files = [f for f in glob.glob(str(jdir / "*.json")) if not f.endswith("summary.json")]
        assert len(files) == 20, f"{cond}: {len(files)} seeds"
        for f in files:
            r = load_seed(Path(f))
            meta, d = r["meta"], r["npz"]
            root_xy = np.asarray(d["root_pos"], float)[:, :2]
            ref_xy = np.asarray(d["ref_qpos"], float)[:, :2]
            dt = float(np.median(np.diff(np.asarray(d["time"], float))))
            pm = path_metrics(d)
            per_seed.append({
                "condition": cond,
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
                "max_heading_error_deg": float(pm["max_heading_error_deg"]),
                "min_root_height_m": float(meta["min_root_height"]),
                "max_tilt_pelvis_deg": float(meta["max_tilt_pelvis_deg"]),
                "max_tilt_torso_deg": float(meta["max_tilt_torso_deg"]),
                "foot_slip_distance_m": float(sum(meta["foot_slip_distance"].values())) if isinstance(meta["foot_slip_distance"], dict) else float(meta["foot_slip_distance"]),
                "double_support_fraction": float(meta["double_support_fraction"]),
                "n_foot_contact_onsets": int(meta["n_foot_contact_onsets_total"]),
                "mean_grf_left_N": float(np.mean(d["contact_left_force"])),
                "mean_grf_right_N": float(np.mean(d["contact_right_force"])),
                "mean_activation_left_upper": float(np.mean(d["act"][:, lu])),
                "mean_activation_left_lower": float(np.mean(d["act"][:, ll])),
                "mean_abs_force_left_upper_N": float(np.mean(np.abs(d["actuator_force"][:, lu]))),
                "mean_abs_force_left_lower_N": float(np.mean(np.abs(d["actuator_force"][:, ll]))),
            })

    csv_path = base / "per_seed.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_seed[0].keys()))
        w.writeheader()
        w.writerows(per_seed)
    print(f"saved {csv_path}")

    # -- summaries + paired diffs ---------------------------------------------
    summary = {}
    for cond in CONDS:
        cr = [r for r in per_seed if r["condition"] == cond]
        summary[cond] = {
            "n": len(cr),
            "survived_10s": int(sum(r["survived_10s"] for r in cr)),
            "physical_fall": int(sum(r["end_condition"] == "physical_fall" for r in cr)),
            "median_duration_s": float(np.median([r["duration_s"] for r in cr])),
            "median_net_cross_drift_m": float(np.median([r["net_cross_drift_m"] for r in cr])),
            "median_signed_veer_deg": float(np.median([r["signed_veer_deg"] for r in cr])),
            "median_max_cross_track_m": float(np.median([r["max_cross_track_m"] for r in cr])),
            "median_final_cross_track_m": float(np.median([r["final_cross_track_m"] for r in cr])),
        }
    a_by_seed = {r["seed"]: r for r in per_seed if r["condition"] == "A"}
    summary["paired_vs_A"] = {}
    for cond in ["B", "C", "D", "E"]:
        cr = {r["seed"]: r for r in per_seed if r["condition"] == cond}
        diffs = [
            {
                "seed": s,
                "duration_s": cr[s]["duration_s"] - a_by_seed[s]["duration_s"],
                "net_cross_drift_m": cr[s]["net_cross_drift_m"] - a_by_seed[s]["net_cross_drift_m"],
                "final_cross_track_m": cr[s]["final_cross_track_m"] - a_by_seed[s]["final_cross_track_m"],
            }
            for s in SEEDS
        ]
        summary["paired_vs_A"][cond] = {
            "mean_duration_s_diff": float(np.mean([v["duration_s"] for v in diffs])),
            "mean_net_cross_drift_diff_m": float(np.mean([v["net_cross_drift_m"] for v in diffs])),
            "per_seed": diffs,
        }

    # -- common window (first 2 s) --------------------------------------------
    common_rows = []
    for cond in CONDS:
        for s in SEEDS:
            d = np.load(str(base / cond / "stochastic" / f"stochastic_seed{s}.npz"))
            t = np.asarray(d["time"], float)
            sel = (t - t[0]) <= COMMON_WINDOW
            root = np.asarray(d["root_pos"], float)[sel, :2]
            ref = np.asarray(d["ref_qpos"], float)[sel, :2]
            dsub = {"time": t[sel], "root_pos": d["root_pos"][sel], "ref_qpos": d["ref_qpos"][sel]}
            pm = path_metrics(dsub)
            common_rows.append({
                "condition": cond,
                "seed": s,
                "window_s": COMMON_WINDOW,
                "max_cross_track_m": float(pm["max_cross_track_m"]),
                "net_cross_drift_m": float(pm["net_cross_drift_m"]),
                "final_cross_track_m": final_signed_cross_track(root, ref),
                "mean_heading_error_deg": float(pm["mean_heading_error_deg"]),
            })
    cw_csv = base / "common_window.csv"
    with cw_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(common_rows[0].keys()))
        w.writeheader()
        w.writerows(common_rows)
    print(f"saved {cw_csv}")

    # -- activation + active/passive force in the common window ---------------
    force_rows = []
    for cond in CONDS:
        for s in SEEDS:
            d = np.load(str(base / cond / "stochastic" / f"stochastic_seed{s}.npz"))
            t = np.asarray(d["time"], float)
            dt = float(np.median(np.diff(t)))
            step = max(1, int(round(1.0 / (FORCE_SAMPLE_HZ * dt))))
            sel = np.arange(0, len(t), step, dtype=int)
            sel = sel[t[sel] - t[0] <= COMMON_WINDOW]
            act_u = float(np.mean(d["act"][sel][:, lu]))
            act_l = float(np.mean(d["act"][sel][:, ll]))
            total_u = float(np.mean(d["actuator_force"][sel][:, lu]))
            total_l = float(np.mean(d["actuator_force"][sel][:, ll]))
            pas_u, pas_l = [], []
            for i in sel:
                data.qpos[:] = d["qpos"][i]
                data.qvel[:] = d["qvel"][i]
                data.act[:] = 0.0
                data.ctrl[:] = 0.0
                mujoco.mj_forward(model, data)
                pas_u.append(float(np.mean(data.actuator_force[lu])))
                pas_l.append(float(np.mean(data.actuator_force[ll])))
            passive_u = float(np.mean(pas_u))
            passive_l = float(np.mean(pas_l))
            force_rows.append({
                "condition": cond,
                "seed": s,
                "left_upper_mean_activation": act_u,
                "left_lower_mean_activation": act_l,
                "left_upper_mean_total_force_N": total_u,
                "left_lower_mean_total_force_N": total_l,
                "left_upper_mean_passive_force_N": passive_u,
                "left_lower_mean_passive_force_N": passive_l,
                "left_upper_mean_active_force_N": total_u - passive_u,
                "left_lower_mean_active_force_N": total_l - passive_l,
            })
    mf_csv = base / "muscle_force.csv"
    with mf_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(force_rows[0].keys()))
        w.writeheader()
        w.writerows(force_rows)
    print(f"saved {mf_csv}")

    summary_path = base / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"saved {summary_path}")

    # -- top-down plots -------------------------------------------------------
    fig, axes = plt.subplots(5, 4, figsize=(20, 22))
    for ci, cond in enumerate(CONDS):
        for k, s in enumerate([0, 1, 2, 13]):
            ax = axes[ci, k]
            d = np.load(str(base / cond / "stochastic" / f"stochastic_seed{s}.npz"))
            root = np.asarray(d["root_pos"], float)[:, :2]
            ref = np.asarray(d["ref_qpos"], float)[:, :2]
            ax.plot(ref[:, 0], ref[:, 1], "k--", lw=1, label="ref")
            ax.plot(root[:, 0], root[:, 1], "r-", lw=1, label="policy")
            ax.set_title(f"{cond} s{s}", fontsize=8)
            ax.axis("equal")
    fig.suptitle("Top-down: policy (red) vs reference (dashed) per condition/seed")
    fig.tight_layout()
    fig.savefig(base / "topdown.png", dpi=110)
    print(f"saved {base / 'topdown.png'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
