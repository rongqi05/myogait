#!/usr/bin/env python3
"""Full-strength crutch-assisted walking baseline.

Three modes, each run in its own process so the MuJoCo patch state stays clean:

    --mode calibrate     measure the reference arm posture and hand offsets
    --mode run --condition A|B_left|B_right|C|D
    --mode compose       stitch baseline_comparison.mp4 from per-condition videos

Conditions
    A        no crutch
    B_left   crutch follows with the left lateral offset, grips inactive
    B_right  same, mirrored offset
    C        left hand grip active   (ipsilateral to a left-sided deficit)
    D        right hand grip active  (contralateral)

The human policy is the frozen official checkpoint_12500 and is never trained.
Every episode starts once and is never reset; the crutch base is positioned only
during initialisation, then advances by control and dynamics alone.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import resource
import sys
import time
from pathlib import Path

import jax
import mujoco
import numpy as np

import crutch_env
from _env_builder import build_eval_bundle, env_data, env_model, quat_to_tilt_deg
from env_step_snippet import DEFAULTS as CRUTCH_DEFAULTS
from env_step_snippet import CrutchInterface
from eval_walk import make_policy_fn

CKPT = "artifacts/checkpoints/checkpoint_12500"
MOTION = "CMU/136/cmu136_stitched_20s_v2_h000_rot000"
PREFIX = crutch_env.PREFIX
FALL_HEIGHT = 0.6
FALL_TILT_DEG = 60.0
CRUTCH_KEYS = ("forward_offset", "lateral_offset", "k_ff", "k_pos", "k_vel", "v_max",
               "cmd_dv_coef", "frame_skip", "grip_dist_tol", "free_side")
COND_GRIP = {"A": None, "B_left": None, "B_right": None, "C": "left", "D": "right"}
COND_SIDE = {"A": None, "B_left": "left", "B_right": "right", "C": "left", "D": "right"}
ACT_FRAME = 0.30   # offset/height calibration frame, matches the default --settle
ROD_BASE_Z = CRUTCH_DEFAULTS["clearance"] + 2.0 * CRUTCH_DEFAULTS["chassis_half_size"][2]  # chassis top
STARTUP_WINDOW = 0.20   # s after the window opens: "startup" grip load
SUSTAIN_WINDOW = 0.50   # s after the window opens: sustained grip load


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_calibration(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing calibration {path}; run --mode calibrate first")
    return json.loads(path.read_text())


def variant_geometry(calib: dict, variant: str) -> dict:
    """start_step / rod_length / per-side offsets / expected closure error."""
    blk = calib[variant]
    return {
        "start_step": int(blk["start_step"]),
        "rod_length": float(blk["rod_length"]),
        "offsets": blk["offsets_by_side"],
        "dz_at_closure": blk.get("dz_at_closure", {}),
        "palm_z": blk.get("palm_z", {}),
    }


def crutch_cfg(calib: dict, side: str | None, variant: str = "paired") -> dict:
    cfg = {k: CRUTCH_DEFAULTS[k] for k in CRUTCH_KEYS}
    geo = variant_geometry(calib, variant)
    off = geo["offsets"][side] if side else {"forward": 0.0, "lateral": 0.0}
    mount = crutch_env.CrutchMount()   # single source of truth for the palm point
    cfg.update(
        prefix=PREFIX,
        hand_body=dict(mount.hand_body),
        hand_site=dict(mount.hand_grip),
        forward_offset=off["forward"],
        lateral_offset=abs(off["lateral"]),
    )
    return cfg


def build(merged: bool, calib: dict, horizon: int, render_size=None,
          rod_length: float | None = None, start_step: int = 0, palm: bool = False,
          markers: bool = False):
    """Build the env. ``palm`` adds the hand grip sites without the robot.

    Calibration uses ``merged=False, palm=True``: the palm point must be the
    same one the grip will use, but the robot itself perturbs nothing before the
    constraint closes, so it is left out.
    """
    mount = None
    if merged or palm:
        mount = crutch_env.CrutchMount(
            xml=calib.get("crutch_xml", "crutch_robot.xml"),
            rod_length=calib.get("rod_length", 1.00) if rod_length is None else rod_length)
    crutch_env.install(mount, render_size=render_size, attach=merged, markers=markers)
    return build_eval_bundle(checkpoint_path=CKPT, motion_path=MOTION, start_step=int(start_step),
                             traj_index=0, horizon=horizon)


# --------------------------------------------------------------------------- #
# calibrate
# --------------------------------------------------------------------------- #
def _probe_activation(start_step: int, seconds: float, n_warm: int, policy_mode: str,
                      seed: int = 0) -> dict:
    """Reproduce the run protocol up to the closing instant and measure the hands.

    Reset at ``start_step``, advance ``n_warm`` control steps with the same
    policy and seed the runs use, then read the palm grip sites. The human
    dynamics do not depend on the crutch before the constraint closes, so this
    is the exact state the runs will be in when they call ``set_assistance``.
    """
    b = build(merged=False, calib={}, horizon=start_step + n_warm + int(seconds / 0.01) + 5,
              palm=True, start_step=start_step)
    env, m, d = b.env, env_model(b.env), env_data(b.env)
    obs = np.asarray(env.reset())
    pf = make_policy_fn(b.agent_conf, policy_mode == "deterministic")
    rng = jax.random.key(seed)
    palm_names = crutch_env.CrutchMount().hand_grip
    sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
    hp = {s: sid(palm_names[s]) for s in ("left", "right")}
    hw = {s: sid(f"{s}_hand_mimic") for s in ("left", "right")}
    jn = {s: {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
              for n in (f"shoulder_elv_{s[0]}", f"elbow_flex_{s[0]}")} for s in ("left", "right")}

    for _ in range(n_warm):
        rng, _rng = jax.random.split(rng)
        a, _ts = pf(b.agent_state.train_state, obs, _rng)
        obs, *_ = env.step(a)

    root = np.array(d.qpos[:3], copy=True)
    out = {"start_step": int(start_step), "activation_step": int(start_step + n_warm),
           "activation_time_s": round(float(d.time), 3), "palm_z": {}, "wrist_z": {},
           "offset": {}, "reach_ratio": {}, "elbow_flexion_deg": {}, "hand_to_shoulder": {}}
    for side in ("left", "right"):
        pz = np.array(d.site_xpos[hp[side]], copy=True)
        out["palm_z"][side] = round(float(pz[2]), 4)
        out["wrist_z"][side] = round(float(d.site_xpos[hw[side]][2]), 4)
        out["offset"][side] = {"forward": round(float(pz[0] - root[0]), 4),
                               "lateral": round(float(pz[1] - root[1]), 4)}
        sh = np.array(d.xanchor[jn[side][f"shoulder_elv_{side[0]}"]], copy=True)
        el = np.array(d.xanchor[jn[side][f"elbow_flex_{side[0]}"]], copy=True)
        upper = float(np.linalg.norm(el - sh))
        fore = float(np.linalg.norm(pz - el))
        span = upper + fore
        dd = float(np.linalg.norm(pz - sh))
        cos_el = (upper ** 2 + fore ** 2 - dd ** 2) / (2 * upper * fore) if upper * fore > 0 else 1.0
        out["reach_ratio"][side] = round(dd / span if span > 0 else float("nan"), 4)
        out["hand_to_shoulder"][side] = round(dd, 4)
        out["elbow_flexion_deg"][side] = round(
            float(np.degrees(np.pi - np.arccos(np.clip(cos_el, -1, 1)))), 2)
    del b
    gc.collect()
    return out


def calibrate(out_dir: Path, seconds: float, policy_mode: str = "stochastic",
              settle: float = 0.30) -> int:
    """Derive the ball height from the palms at the instant the grip closes.

    ``L_rod = z_palm(at closing) - chassis_top``, with one common rod for both
    sides per variant. The closing instant is ``start_step + warm-up``, i.e. the
    state the runs are in when they activate the constraint, not the frame the
    episode starts on and not a clip-wide average.

    ``paired``     start at frame 0, per-side hand offsets (asymmetric reference)
    ``symmetric``  start frame chosen so the *closing instant* has the most
                   comparable two-hand posture, one height, mirrored offsets
    """
    dt_guess = 0.01
    n_warm = int(round(settle / dt_guess))

    # 1) a palm-only env, just to read dt and confirm the grip sites exist
    b = build(merged=False, calib={}, horizon=int(seconds / 0.01) + 5, palm=True)
    env, m, d = b.env, env_model(b.env), env_data(b.env)
    dt = float(env.dt)
    n_warm = int(round(settle / dt))
    palm_names = crutch_env.CrutchMount().hand_grip
    if any(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, palm_names[s]) < 0 for s in ("left", "right")):
        raise RuntimeError(f"palm grip sites missing: {palm_names}")

    # 2) symmetric start frame: probe the real protocol on a coarse grid and take
    #    the closing instant whose two palms are most comparable. A clip-wide
    #    rollout from frame 0 is not a proxy here -- the posture after 30 steps
    #    from a later start frame differs by centimetres.
    stride = max(1, int(round(0.2 / dt)))
    grid = list(range(0, int(3.0 / dt) + 1, stride))
    probes = {k: _probe_activation(k, seconds, n_warm, policy_mode) for k in grid}
    del b
    gc.collect()

    def posture_cost(pr):
        zl, zr = pr["palm_z"]["left"], pr["palm_z"]["right"]
        yl, yr = pr["offset"]["left"]["lateral"], pr["offset"]["right"]["lateral"]
        return abs(zl - zr) + 0.5 * abs(yl + yr)

    costs = {k: posture_cost(pr) for k, pr in probes.items()}
    k_sym = min(costs, key=lambda k: costs[k])
    p_sym = probes[k_sym]
    p_paired = probes[0]

    def common_rod(probe):
        return round(0.5 * (probe["palm_z"]["left"] + probe["palm_z"]["right"]) - ROD_BASE_Z, 4)

    def closure(probe, rod):
        """Height difference each side will see when the constraint closes."""
        return {s: round(probe["palm_z"][s] - (ROD_BASE_Z + rod), 4) for s in ("left", "right")}

    rod_paired, rod_sym = common_rod(p_paired), common_rod(p_sym)

    calib = {
        "crutch_xml": "crutch_robot.xml",
        "checkpoint": CKPT,
        "motion": MOTION,
        "policy_mode": policy_mode,
        "control_dt": dt,
        "symmetry_grid": grid,
        "clearance": CRUTCH_DEFAULTS["clearance"],
        "chassis_half_z": CRUTCH_DEFAULTS["chassis_half_size"][2],
        "rod_base_z": round(ROD_BASE_Z, 4),
        "warmup_steps": n_warm,
        "palm_site": {"left": palm_names["left"], "right": palm_names["right"]},
        "wrist_site": {"left": "left_hand_mimic", "right": "right_hand_mimic"},
        "paired": {
            "start_step": 0,
            "rod_length": rod_paired,
            "palm_z": p_paired["palm_z"],
            "wrist_z": p_paired["wrist_z"],
            "offsets_by_side": p_paired["offset"],
            "dz_at_closure": closure(p_paired, rod_paired),
            "reach_ratio": p_paired["reach_ratio"],
            "hand_to_shoulder": p_paired["hand_to_shoulder"],
            "elbow_flexion_deg": p_paired["elbow_flexion_deg"],
        },
        "symmetric": {
            "start_step": k_sym,
            "start_time_s": round(k_sym * dt, 3),
            "activation_step": p_sym["activation_step"],
            "activation_time_s": p_sym["activation_time_s"],
            "hand_posture_cost_at_activation": round(float(costs[k_sym]), 4),
            "hand_posture_cost_by_start": {int(k): round(float(v), 4) for k, v in costs.items()},
            "rod_length": rod_sym,
            "palm_z": p_sym["palm_z"],
            "wrist_z": p_sym["wrist_z"],
            "dz_at_closure": closure(p_sym, rod_sym),
            "lateral": round(0.5 * (abs(p_sym["offset"]["left"]["lateral"])
                                    + abs(p_sym["offset"]["right"]["lateral"])), 4),
            "forward": round(0.5 * (p_sym["offset"]["left"]["forward"]
                                    + p_sym["offset"]["right"]["forward"]), 4),
            "offsets_by_side": p_sym["offset"],
            "reach_ratio": p_sym["reach_ratio"],
            "hand_to_shoulder": p_sym["hand_to_shoulder"],
            "elbow_flexion_deg": p_sym["elbow_flexion_deg"],
        },
        "seconds": seconds,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=2))
    print(json.dumps(calib, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# run one condition
# --------------------------------------------------------------------------- #
def run_condition(cond: str, seeds: list[int], seconds: float, out_dir: Path,
                  calib: dict, video: bool, settle: float, mode: str = "stochastic",
                  variant: str = "paired", markers: bool = False,
                  camera: str = "overview") -> int:
    grip_side = COND_GRIP[cond]
    side = COND_SIDE[cond]
    merged = cond != "A"
    geo = variant_geometry(calib, variant)
    dt = float(calib["control_dt"])
    target_steps = int(round(seconds / dt))
    n_warm = int(round(settle / dt))
    # the environment must not truncate before warm-up + window are done
    horizon = n_warm + target_steps + 20

    b = build(merged=merged, calib=calib, horizon=horizon,
              render_size=(1280, 720) if video else None,
              rod_length=geo["rod_length"], start_step=geo["start_step"],
              markers=markers and merged)
    env, m, d = b.env, env_model(b.env), env_data(b.env)
    pf = make_policy_fn(b.agent_conf, mode == "deterministic")
    ci = CrutchInterface(m, d, cfg=crutch_cfg(calib, side, variant)) if merged else None
    layout = crutch_env.layout(m) if merged else None
    if merged:
        print(f"[layout] nq={layout['nq']} nv={layout['nv']} nu={layout['nu']} "
              f"crutch_act={layout['crutch_act_ids']} first_crutch_qpos={layout['first_crutch_qpos']}")
    print(f"[{cond}] variant={variant} start_step={geo['start_step']} rod={geo['rod_length']} "
          f"warmup={n_warm} steps, window={target_steps} steps")

    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pelvis_mimic")
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "upper_body_mimic")
    grip_site = ci.site_grip if merged else -1
    # the same palm point the constraint uses -- not the wrist mimic site
    hand_site_id = ({s: ci.site_hand[s] for s in ("left", "right")} if merged
                    else {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_hand_mimic")
                          for s in ("left", "right")})
    n_muscle_ctrl = layout["first_crutch_qpos"] if merged else 0  # 354 either way

    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = None
    if video:
        renderer = mujoco.Renderer(m, height=720, width=1280)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 1
        if camera == "side":
            cam.distance = 3.2
            cam.azimuth = 90.0
            cam.elevation = -5.0
        else:
            cam.distance = 4.4
            cam.azimuth = 133.0
            cam.elevation = -11.0

    summary = []
    for seed in seeds:
        obs = np.asarray(env.reset())
        rng = jax.random.key(seed)

        # ---- uniform warm-up for every condition: same number of steps, not recorded
        if merged:
            ci.mode, ci.affected_side, ci.active_side = "none", side, None
            ci.cfg["free_side"] = side
            ci._heading = np.array([1.0, 0.0])
        for _ in range(n_warm):
            rng, _rng = jax.random.split(rng)
            a, _ts = pf(b.agent_state.train_state, obs, _rng)
            if merged:
                ci.command(d.qpos[:3], d.qvel[:3])
            obs, *_ = env.step(a)
            if merged:
                ci.observe()

        align_info, act_info = {}, {}
        if merged:
            if grip_side:
                align_info = ci.align_grip(grip_side)
            ci.tare_now()
            if grip_side:
                gap_before = float(np.linalg.norm(d.site_xpos[grip_site] - d.site_xpos[hand_site_id[grip_side]]))
                dz_before = float(d.site_xpos[hand_site_id[grip_side]][2] - d.site_xpos[grip_site][2])
                rel_before = ci.grip_rel_speed(grip_side)
                # the impaired side is always left here: C ipsilateral, D contralateral
                act_info = ci.set_assistance("ipsilateral" if grip_side == "left" else "contralateral", "left")
                act_info["gap_before_activation_m"] = gap_before
                act_info["dz_before_activation_m"] = dz_before
                act_info["rel_speed_before_activation_m_s"] = rel_before
                if not act_info["ok"]:
                    print(f"[fail] {cond} seed {seed}: grip not activated: {act_info}")

        # ---- recorded window: identical length and clock for all conditions ----
        steps = []
        reason = "other"
        frames = []
        for k in range(1, target_steps + 1):
            rng, _rng = jax.random.split(rng)
            a, _ts = pf(b.agent_state.train_state, obs, _rng)
            crutch_out = {}
            if merged:
                ci.command(d.qpos[:3], d.qvel[:3])
            obs, reward, absorbing, done, info = env.step(a)
            if merged:
                crutch_out = ci.observe()
            qpos = np.array(d.qpos, copy=True)
            qvel = np.array(d.qvel, copy=True)
            root = qpos[:3]
            tilt_p = quat_to_tilt_deg(qpos[3:7])
            tv = d.site_xpos[torso] - d.site_xpos[pelvis]
            tilt_t = float(np.degrees(np.arccos(np.clip(tv[2] / max(np.linalg.norm(tv), 1e-9), -1, 1))))
            rec = {
                "t": float(d.time), "walk_t": k * dt, "qpos": qpos, "qvel": qvel,
                "root": root.copy(), "tilt_pelvis": tilt_p, "tilt_torso": tilt_t,
                "reward": float(reward), "done": bool(done),
            }
            if merged:
                rec.update({
                    "chassis_pos": ci._sensor("chassis_pos", 3).copy(),
                    "chassis_vel": crutch_out["v_meas"].copy(),
                    "chassis_acc": crutch_out["a_meas"].copy(),
                    "v_cmd": crutch_out["v_cmd"].copy(),
                    "dv_limited": crutch_out["dv_limited"],
                    "force_world": crutch_out["grip_force_world"].copy(),
                    "force_raw": crutch_out["grip_force_raw"].copy(),
                    "torque_raw": crutch_out["grip_torque_raw"].copy(),
                    "support_up": crutch_out["support_up_N"],
                    "grip_gap": crutch_out["grip_gap_m"][side] if side else float("nan"),
                    "grip_dz": crutch_out["grip_dz_m"][side] if side else float("nan"),
                    "grip_active": bool(crutch_out["eq_active"][side]) if side else False,
                })
            steps.append(rec)
            if renderer is not None and k % 5 == 0:
                renderer.update_scene(d, camera=cam)
                frames.append(renderer.render().copy())

            numeric = not (np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel)))
            fall = root[2] < FALL_HEIGHT or max(tilt_p, tilt_t) > FALL_TILT_DEG
            if numeric:
                reason = "numeric_anomaly"
            elif fall:
                reason = "physical_fall"
            if reason != "other":
                break
        if reason == "other":
            reason = "window_complete" if len(steps) >= target_steps else "incomplete"

        walk_t = np.array([s["walk_t"] for s in steps])
        root_xy = np.array([s["root"][:2] for s in steps])
        fwd = float(root_xy[-1, 0] - root_xy[0, 0])
        dur = float(len(steps) * dt)
        completed = bool(len(steps) >= target_steps)
        rec_summary = {
            "condition": cond, "variant": variant, "seed": seed, "reason": reason,
            "start_step": int(geo["start_step"]), "warmup_s": round(n_warm * dt, 3),
            "duration_s": round(dur, 3),
            "completed_window": completed,
            "forward_x_m": round(fwd, 3),
            "avg_forward_speed": round(fwd / dur, 3) if dur > 0 else 0.0,
            "max_tilt_torso_deg": round(max(s["tilt_torso"] for s in steps), 2),
            "max_tilt_pelvis_deg": round(max(s["tilt_pelvis"] for s in steps), 2),
            "min_root_z": round(min(float(s["root"][2]) for s in steps), 3),
            "max_lateral_abs_m": round(float(np.max(np.abs(root_xy[:, 1] - root_xy[0, 1]))), 3),
            "grip_required": bool(grip_side),
            "grip_activated": bool(act_info.get("ok", False)) if grip_side else False,
            "align": align_info, "assist": act_info,
        }
        if merged:
            fz = np.array([s["support_up"] for s in steps])
            fx = np.array([s["force_world"][0] for s in steps])
            fy = np.array([s["force_world"][1] for s in steps])
            gap = np.array([s["grip_gap"] for s in steps])
            dz = np.array([s["grip_dz"] for s in steps], dtype=float)
            cv = np.array([s["chassis_vel"] for s in steps])
            active = np.array([s["grip_active"] for s in steps], dtype=float) if side else np.zeros(len(steps))
            start = walk_t <= STARTUP_WINDOW
            sust = walk_t >= SUSTAIN_WINDOW
            rec_summary.update({
                "grip_active_fraction": round(float(active.mean()), 4),
                "fz_mean_N": round(float(fz.mean()), 2),
                "fz_min_N": round(float(fz.min()), 2), "fz_max_N": round(float(fz.max()), 2),
                "fz_abs_mean_N": round(float(np.abs(fz).mean()), 2),
                "startup_abs_peak_N": round(float(np.max(np.abs(fz[start]))) if start.any() else 0.0, 2),
                "sustained_abs_mean_N": round(float(np.abs(fz[sust]).mean()) if sust.any() else 0.0, 2),
                "sustained_abs_p95_N": round(float(np.percentile(np.abs(fz[sust]), 95)) if sust.any() else 0.0, 2),
                "fx_abs_mean_N": round(float(np.abs(fx).mean()), 2),
                "fy_abs_mean_N": round(float(np.abs(fy).mean()), 2),
                "grip_gap_max_m": round(float(np.nanmax(gap)), 4),
                "grip_gap_mean_m": round(float(np.nanmean(gap)), 4),
                "dz_start_m": round(float(dz[0]), 4) if len(dz) else float("nan"),
                "dz_mean_m": round(float(np.nanmean(dz)), 4),
                "dz_p05_m": round(float(np.nanpercentile(dz, 5)), 4),
                "dz_p95_m": round(float(np.nanpercentile(dz, 95)), 4),
                "dz_max_abs_m": round(float(np.nanmax(np.abs(dz))), 4),
                "chassis_speed_mean": round(float(np.linalg.norm(cv, axis=1).mean()), 3),
                "chassis_speed_max": round(float(np.linalg.norm(cv, axis=1).max()), 3),
            })
        rec_summary["completed_with_valid_grip"] = bool(
            completed and (not grip_side or (rec_summary["grip_activated"]
                                            and rec_summary.get("grip_active_fraction", 0.0) >= 0.999)))
        summary.append(rec_summary)
        print(json.dumps(rec_summary, ensure_ascii=False))

        np.savez_compressed(
            out_dir / f"{cond}_seed{seed}.npz",
            time=np.array([s["t"] for s in steps]),
            walk_t=walk_t,
            qpos=np.array([s["qpos"] for s in steps]),
            qvel=np.array([s["qvel"] for s in steps]),
            root=np.array([s["root"] for s in steps]),
            tilt_pelvis=np.array([s["tilt_pelvis"] for s in steps]),
            tilt_torso=np.array([s["tilt_torso"] for s in steps]),
            **({"chassis_pos": np.array([s["chassis_pos"] for s in steps]),
                "chassis_vel": np.array([s["chassis_vel"] for s in steps]),
                "force_world": np.array([s["force_world"] for s in steps]),
                "force_raw": np.array([s["force_raw"] for s in steps]),
                "support_up": np.array([s["support_up"] for s in steps]),
                "grip_gap": np.array([s["grip_gap"] for s in steps]),
                "grip_dz": np.array([s["grip_dz"] for s in steps]),
                "grip_active": np.array([s["grip_active"] for s in steps]),
                "v_cmd": np.array([s["v_cmd"] for s in steps])} if merged else {}),
        )
        (out_dir / f"{cond}_seed{seed}.json").write_text(json.dumps(rec_summary, indent=2))
        if renderer is not None and frames:
            import imageio.v2 as iio
            iio.mimwrite(out_dir / f"{cond}_seed{seed}.mp4", frames,
                         fps=int(round(1.0 / (dt * 5))), macro_block_size=1)

    with (out_dir / f"summary_{cond}.csv").open("w", newline="") as fh:
        keys = [k for k in summary[0] if not isinstance(summary[0][k], dict)]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)
    done = sum(1 for r in summary if r["completed_window"])
    valid = sum(1 for r in summary if r["completed_with_valid_grip"])
    print(f"[{cond}] window complete {done}/{len(summary)}   complete+valid grip {valid}/{len(summary)}")
    print("peak rss MB:", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["calibrate", "run", "compose"], default="run")
    ap.add_argument("--condition", default="A")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--settle", type=float, default=0.30, help="s of follow-only before clamping the grip")
    ap.add_argument("--out_dir", default="runs/crutch_baseline")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--policy_mode", choices=["stochastic", "deterministic"], default="stochastic")
    ap.add_argument("--markers", action="store_true", help="draw spheres on the two palm points and the ball grip point")
    ap.add_argument("--camera", choices=["overview", "side"], default="overview")
    ap.add_argument("--variant", choices=["paired", "symmetric"], default="paired",
                    help="paired: per-side offsets (asymmetric reference pose); "
                         "symmetric: one ball height, mirrored lateral offsets, comparable arm posture")
    args = ap.parse_args()

    out = Path(args.out_dir)
    calib_path = Path(args.calib) if args.calib else out / "calibration.json"

    if args.mode == "calibrate":
        return calibrate(out, max(args.seconds, 10.0), args.policy_mode, args.settle)
    if args.mode == "compose":
        return compose(out, args.seeds)
    calib = load_calibration(calib_path)
    t0 = time.time()
    geo = variant_geometry(calib, args.variant)
    print(f"[{args.condition}] policy={args.policy_mode} variant={args.variant} "
          f"seconds={args.seconds} seeds={args.seeds} rod={geo['rod_length']} "
          f"start_step={geo['start_step']} offsets={geo['offsets']} "
          f"expected_dz_at_closure={geo['dz_at_closure']}")
    sub = f"{args.seconds:g}s" if args.variant == "paired" else f"{args.seconds:g}s_{args.variant}"
    rc = run_condition(args.condition, args.seeds, args.seconds, out / sub,
                       calib, args.video, args.settle, args.policy_mode, args.variant,
                       args.markers, args.camera)
    print(f"wall {time.time() - t0:.1f}s")
    return rc


def compose(out: Path, seeds: list[int]) -> int:
    import imageio.v2 as iio
    from PIL import Image, ImageDraw, ImageFont

    FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    seed = seeds[0]
    tag = None
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        if (d / f"A_seed{seed}.mp4").exists():
            tag = d
    if tag is None:
        raise SystemExit("no rendered videos found")
    panels = [("A", "无拐杖"), ("C", "左手持杖（患侧）"), ("D", "右手持杖（健侧）")]
    data, meta = {}, {}
    for c, _ in panels:
        f = tag / f"{c}_seed{seed}.mp4"
        r = iio.get_reader(str(f))
        data[c] = [np.asarray(x) for x in r]
        r.close()
        z = np.load(tag / f"{c}_seed{seed}.npz")
        meta[c] = {
            "t": z["time"],
            "fz": z["support_up"] if "support_up" in z else np.zeros(len(z["time"])),
            "root": z["root"],
        }
    h, w = data["A"][0].shape[:2]
    nmax = max(len(v) for v in data.values())
    canvas_h = h + 74
    out_f = out / f"baseline_comparison_seed{seed}.mp4"
    wr = iio.get_writer(str(out_f), fps=20, macro_block_size=1)
    f_h = ImageFont.truetype(FONT, 26, index=0)
    try:
        for k in range(nmax):
            canv = Image.new("RGB", (w * 3, canvas_h), (16, 17, 20))
            dr = ImageDraw.Draw(canv)
            f_s = ImageFont.truetype(FONT, 20, index=0)
            for i, (c, label) in enumerate(panels):
                frames = data[c]
                img = Image.fromarray(frames[min(k, len(frames) - 1)])
                if k >= len(frames):
                    img = Image.blend(img, Image.new("RGB", (w, h)), 0.45)
                canv.paste(img, (i * w, 46))
                dr.text((i * w + 14, 10), label, font=f_h, fill=(240, 240, 242))
                step = min(k * 5, len(meta[c]["t"]) - 1)
                sim_t = float(meta[c]["t"][step])
                fz = float(meta[c]["fz"][step])
                fz_txt = f"Fz {fz:+7.1f} N" if c != "A" else "Fz    —  (无拐杖)"
                dr.text((i * w + 14, h + 44), f"t={sim_t:5.2f}s   {fz_txt}", font=f_s,
                        fill=(150, 220, 255) if c != "A" else (150, 150, 155))
            wr.append_data(np.asarray(canv))
    finally:
        wr.close()
    print("wrote", out_f)

    # keyframes: same simulation times, side by side, with the real stop time marked
    times = [0.5, 3.0, 6.0, 9.0]
    small_w = w // 2
    small_h = h // 2
    cols, rows = len(times), len(panels)
    pad_top = 44
    sheet = Image.new("RGB", (small_w * cols, pad_top + (small_h + 30) * rows), (16, 17, 20))
    ds = ImageDraw.Draw(sheet)
    f_c = ImageFont.truetype(FONT, 20, index=0)
    f_r = ImageFont.truetype(FONT, 18, index=0)
    for ci_, tv in enumerate(times):
        ds.text((ci_ * small_w + 12, 12), f"t = {tv:.1f} s", font=f_c, fill=(230, 230, 235))
    for ri, (c, label) in enumerate(panels):
        frames = data[c]
        stopped = len(frames) < nmax
        for ci_, tv in enumerate(times):
            k = min(int(round(tv * 20)), len(frames) - 1)
            img = Image.fromarray(frames[k]).resize((small_w, small_h), Image.LANCZOS)
            y0 = pad_top + ri * (small_h + 30)
            sheet.paste(img, (ci_ * small_w, y0))
            step = min(k * 5, len(meta[c]["t"]) - 1)
            fz = float(meta[c]["fz"][step])
            txt = f"{label}   Fz {fz:+.0f} N" if c != "A" else f"{label}   Fz —"
            if stopped and k >= len(frames) - 1:
                txt += "   (已终止)"
            ds.text((ci_ * small_w + 10, y0 + small_h + 4), txt, font=f_r,
                    fill=(150, 220, 255) if c != "A" else (160, 160, 165))
    sheet_f = out / f"baseline_keyframes_seed{seed}.png"
    sheet.save(sheet_f)
    print("wrote", sheet_f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
