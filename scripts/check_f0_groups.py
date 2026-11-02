"""Model-level check that region F0 scaling hits only the intended muscles.

Builds a fresh MyoFullBody model, freezes qpos/qvel/activation, and verifies:

1. the anatomical grouping is complete and mutually exclusive (354 actuators);
2. scaling ``left_upper`` / ``left_lower`` changes ``actuator_gainprm[i, 0]``
   (the peak isometric force F0) for exactly the intended indices;
3. at fixed qpos/qvel/activation the resulting *force* changes only for the
   intended muscles, while every untouched muscle keeps bit-identical force;
4. the relation between the F0 parameter ratio, the total muscle-force ratio,
   and the "active" force ratio (total minus passive force at activation 0).

Muscles that are completely slack at the frozen pose (length outside their
``lengthrange``, zero force at any activation) are listed explicitly rather than
treated as failures, because F0 scaling cannot change a force that is zero.

Usage:
    PYTHONPATH=external/MuscleMimic:scripts .venv/bin/python scripts/check_f0_groups.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import mujoco
import numpy as np

from _env_builder import build_eval_bundle, env_data, env_model, muscle_groups, scale_muscle_groups

CHECKPOINT = "artifacts/checkpoints/checkpoint_12500"
MOTION = "CMU/120/cmu120_19_straight_h000"
START_STEP = 53
ACT = 1.0  # full activation: maximizes active force so scaling is visible


def _bundle():
    return build_eval_bundle(
        checkpoint_path=CHECKPOINT,
        motion_path=MOTION,
        start_step=START_STEP,
        traj_index=0,
        strict_termination=False,
        horizon=None,
    )


def _frozen(bundle):
    model = env_model(bundle.env)
    data = env_data(bundle.env)
    bundle.env.reset()
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return model, data


def forces_at(model, data, act: float) -> np.ndarray:
    data.act[:] = act
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    return data.actuator_force.copy()


def main() -> int:
    model, data = _frozen(_bundle())
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    groups = muscle_groups(model)

    # -- 1. grouping completeness / exclusivity ---------------------------------
    print("== group classification (354 actuators) ==")
    all_idx = []
    for key in ("left_upper", "right_upper", "left_lower", "right_lower", "left_trunk", "right_trunk"):
        g = groups[key]
        all_idx.extend(g["indices"])
        print(f"  {key:12s} n={len(g['indices']):3d}")
    all_idx = sorted(all_idx)
    ok_exclusive = len(all_idx) == len(set(all_idx)) == model.nu
    print(f"  total={len(all_idx)} nu={model.nu} exclusive={ok_exclusive}")
    if not ok_exclusive:
        print("FAIL: grouping is not complete/exclusive")
        return 1

    base_total = forces_at(model, data, ACT)
    base_passive = forces_at(model, data, 0.0)

    report = {"groups": {}, "isolations": {}, "force_ratios": {}, "slack_muscles": {}}
    for key in ("left_upper", "left_lower"):
        model2, data2 = _frozen(_bundle())
        idx = np.asarray(groups[key]["indices"], dtype=int)
        mask = np.zeros(model2.nu, dtype=bool)
        mask[idx] = True
        gain_before = model2.actuator_gainprm[:, 0].copy()
        scale_muscle_groups(model2, **{key: 0.7})

        # parameter-level isolation
        gain_changed = ~np.isclose(model2.actuator_gainprm[:, 0], gain_before)
        param_wrong = int(np.count_nonzero(gain_changed & ~mask))
        param_missed = int(np.count_nonzero(mask & ~gain_changed))

        scaled_total = forces_at(model2, data2, ACT)
        scaled_passive = forces_at(model2, data2, 0.0)

        # force-level isolation
        force_changed = ~np.isclose(scaled_total, base_total, rtol=1e-6, atol=1e-9)
        wrongly_changed = int(np.count_nonzero(force_changed & ~mask))

        # slack muscles inside the group: zero force at both activations
        slack_names = [names[i] for i in idx if abs(base_total[i]) < 1e-6 and abs(base_passive[i]) < 1e-6]
        slack_idx = [names.index(s) for s in slack_names]
        missed = int(np.count_nonzero(mask & ~force_changed & ~np.isin(np.arange(model2.nu), slack_idx)))

        print(f"  scale {key}: gainprm wrong={param_wrong} missed={param_missed}; "
              f"force changed={np.count_nonzero(force_changed)} (non-slack expect {len(idx) - len(slack_idx)}), "
              f"wrongly_changed={wrongly_changed}, slack={slack_names}")
        report["isolations"][key] = {
            "gainprm_wrongly_changed": param_wrong,
            "gainprm_missed": param_missed,
            "force_wrongly_changed": wrongly_changed,
            "force_missed_non_slack": missed,
        }
        report["slack_muscles"][key] = slack_names

        # force ratios over muscles with substantial active force
        active_idx = np.asarray([i for i in idx if i not in slack_idx], dtype=int)
        active_base = base_total[active_idx] - base_passive[active_idx]
        active_scaled = scaled_total[active_idx] - scaled_passive[active_idx]
        sel = np.abs(active_base) > 1.0
        if sel.any():
            ratio_active = float(np.median(active_scaled[sel] / active_base[sel]))
            ratio_total = float(np.median(scaled_total[active_idx][sel] / base_total[active_idx][sel]))
        else:
            ratio_active = ratio_total = float("nan")
        print(f"           F0 ratio=0.7000  total-force ratio(median)={ratio_total:.4f}  "
              f"active-force ratio(median)={ratio_active:.4f}  (n={int(sel.sum())})")
        report["force_ratios"][key] = {
            "f0_ratio": 0.7,
            "total_force_ratio_median": ratio_total,
            "active_force_ratio_median": ratio_active,
            "n_muscles_in_ratio": int(sel.sum()),
        }
        report["groups"][key] = {
            "count": int(len(idx)),
            "names": list(groups[key]["names"]),
            "indices": [int(i) for i in idx],
        }

    ok = all(
        v["gainprm_wrongly_changed"] == 0 and v["gainprm_missed"] == 0 and v["force_wrongly_changed"] == 0
        for v in report["isolations"].values()
    )
    print("PASS" if ok else "FAIL")
    out = pathlib.Path("runs/f0_ablation/group_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"saved -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
