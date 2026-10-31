#!/usr/bin/env python3
"""Single-episode, no-reset acceptance rollout for the MuscleMimic MyoFullBody policy.

Each seed produces one independent rollout that starts once at a fixed reference
frame and is *never* reset inside the episode. The script records, per control
step, the full MuJoCo state needed to audit that the motion is produced by the
policy's muscle commands through MuJoCo dynamics.

Run (see README for the environment):

    PYTHONPATH=external/MuscleMimic:scripts .venv/bin/python scripts/eval_walk.py \
        --checkpoint artifacts/checkpoints/checkpoint_12500 \
        --motion_path KIT/314/walking_medium09_poses \
        --mode stochastic --seeds 0 1 2 \
        --out_dir runs/C_acceptance
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from _env_builder import build_eval_bundle, env_data, env_model, quat_to_tilt_deg

# ---------------------------------------------------------------------------
# success thresholds (frozen for this baseline study)
# ---------------------------------------------------------------------------
SUCCESS_SECONDS = 10.0
MIN_FORWARD_SPEED = 0.6  # m/s
MAX_LATERAL_OFFSET = 0.35  # m
MAX_TILT_DEG = 45.0
FALL_HEIGHT = 0.6  # m, matches env.root_height_healthy_range[0]
FALL_TILT_DEG = 60.0

REASONS = (
    "time_limit_success",
    "physical_fall",
    "reference_deviation",
    "clip_end",
    "numeric_anomaly",
    "reset_or_state_override",
    "other",
)


# ---------------------------------------------------------------------------
# roll-out
# ---------------------------------------------------------------------------
def make_policy_fn(agent_conf, deterministic: bool):
    def _act(ts, obs, rng):
        obs_b = jnp.atleast_2d(obs)
        y, updates = agent_conf.network.apply(
            {"params": ts.params, "run_stats": ts.run_stats}, obs_b, mutable=["run_stats"]
        )
        pi, _ = y
        a = pi.mode() if deterministic else pi.sample(seed=rng)
        if hasattr(a, "ndim") and a.ndim > 1 and a.shape[0] == 1:
            a = a[0]
        return a, ts.replace(run_stats=updates["run_stats"])

    return jax.jit(_act)


def foot_geom_sets(model):
    """Map each foot side to the geom ids that can contact the ground."""
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
    sets = {"left": set(), "right": set()}
    for gid in range(model.ngeom):
        bname = body_names[model.geom_bodyid[gid]].lower()
        if not any(k in bname for k in ("calcn", "toes", "talus")):
            continue
        if bname.endswith("_l") or "left" in bname:
            sets["left"].add(gid)
        elif bname.endswith("_r") or "right" in bname:
            sets["right"].add(gid)
    return sets


def foot_body_ids(model, side: str):
    ids = {}
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        low = name.lower()
        if side == "left" and (low in ("calcn_l", "toes_l", "talus_l")):
            ids[low] = i
        if side == "right" and (low in ("calcn_r", "toes_r", "talus_r")):
            ids[low] = i
    return ids


def contact_state(model, data, geom_sets, force_threshold=20.0):
    """Per-side normal force, contact flag and true contact-point slip speed.

    Slip is the tangential velocity of the material point of the foot geom that
    is currently in contact (v_body + omega x r), not the velocity of the foot
    body origin, so heel-toe roll during stance is not counted as slipping.
    """
    out = {
        "left": 0.0,
        "right": 0.0,
        "left_n": 0,
        "right_n": 0,
        "left_slip": 0.0,
        "right_slip": 0.0,
    }
    fbuf = np.zeros(6, dtype=float)
    vbuf = np.zeros(6, dtype=float)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        side = None
        foot_geom = None
        if g1 in geom_sets["left"]:
            side, foot_geom = "left", g1
        elif g2 in geom_sets["left"]:
            side, foot_geom = "left", g2
        elif g1 in geom_sets["right"]:
            side, foot_geom = "right", g1
        elif g2 in geom_sets["right"]:
            side, foot_geom = "right", g2
        if side is None:
            continue
        mujoco.mj_contactForce(model, data, i, fbuf)
        normal = abs(float(fbuf[0]))
        out[side] += normal
        out[f"{side}_n"] += 1

        body_id = int(model.geom_bodyid[foot_geom])
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vbuf, 0)
        omega = vbuf[:3]
        v_lin = vbuf[3:]
        r = np.asarray(c.pos) - np.asarray(data.xpos[body_id])
        v_point = v_lin + np.cross(omega, r)
        n = np.asarray(c.frame[:3])
        v_tan = v_point - n * float(np.dot(v_point, n))
        speed = float(np.linalg.norm(v_tan))
        out[f"{side}_slip"] = max(out[f"{side}_slip"], speed)
    out["left_on"] = out["left"] > force_threshold
    out["right_on"] = out["right"] > force_threshold
    return out


def reference_frame(env, np_backend=np):
    """Current reference frame (qpos, site_xpos) and its index."""
    carry = env._additional_carry
    ref = env.th.get_current_traj_data(carry, np_backend)
    ts = carry.traj_state
    return ref, int(ts.subtraj_step_no), int(ts.traj_no)


class TerminationProbe:
    """Re-evaluate the env terminal handler after a `done` to name the trigger."""

    def __init__(self, env):
        self.env = env
        self.handler = getattr(env, "_terminal_state_handler", None)

    def checks(self, obs, info):
        env = self.env
        handler = self.handler
        data = env_data(env)
        carry = env._additional_carry
        out = {}
        for name, meth in (
            ("mean_site_deviation", "_check_mean_site_deviation"),
            ("root_deviation", "_check_root_deviation"),
            ("root_orientation", "_check_root_orientation"),
        ):
            fn = getattr(handler, meth, None)
            if fn is None:
                continue
            try:
                out[name] = bool(np.asarray(fn(env, data, carry, np)).item())
            except Exception as exc:  # pragma: no cover - defensive
                out[name] = f"error: {exc}"
        # height check lives in HeightBasedTerminalStateHandler
        if hasattr(handler, "root_height_range") and hasattr(handler, "root_free_joint_xml_ind"):
            try:
                h = float(data.qpos[handler.root_free_joint_xml_ind][2])
                lo, hi = handler.root_height_range
                out["root_height"] = h
                out["height_violation"] = bool(h < lo or h > hi)
            except Exception as exc:  # pragma: no cover
                out["height_violation"] = f"error: {exc}"
        return out


def rollout(
    checkpoint: str,
    motion_path: str,
    mode: str,
    seed: int,
    out_dir: Path,
    seconds: float = SUCCESS_SECONDS,
    start_step: int = 0,
    record_video: bool = False,
    video_every: int = 5,
    strict_termination: bool = False,
    horizon: int | None = None,
    perturb: float = 0.0,
    perturb_seed: int = 0,
) -> dict:
    deterministic = mode == "deterministic"
    t_build = time.time()
    bundle = build_eval_bundle(
        checkpoint_path=checkpoint,
        motion_path=motion_path,
        start_step=start_step,
        traj_index=0,
        strict_termination=strict_termination,
        horizon=horizon,
    )
    env = bundle.env
    model = env_model(env)
    data = env_data(env)
    policy_fn = make_policy_fn(bundle.agent_conf, deterministic)
    probe = TerminationProbe(env)
    geom_sets = foot_geom_sets(model)
    if not geom_sets["left"] or not geom_sets["right"]:
        raise RuntimeError(f"could not identify foot geoms: {geom_sets}")

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    pelvis_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pelvis_mimic")
    torso_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "upper_body_mimic")
    if pelvis_site < 0 or torso_site < 0:
        raise RuntimeError("pelvis_mimic / upper_body_mimic sites not found")
    left_foot_ids = foot_body_ids(model, "left")
    right_foot_ids = foot_body_ids(model, "right")
    if not left_foot_ids or not right_foot_ids:
        raise RuntimeError("could not find foot bodies")
    left_foot_track = next(iter(left_foot_ids.values()))
    right_foot_track = next(iter(right_foot_ids.values()))

    # ------------------------------------------------------------------
    # reset once (this is the only reset of the episode)
    # ------------------------------------------------------------------
    reset_count = 0
    obs = env.reset()
    reset_count += 1
    reset_time = float(data.time)

    perturb_info = None
    if perturb > 0.0:
        # Small initial-state perturbation applied ONCE, immediately after the
        # trajectory-based reset. Joints get N(0, perturb) rad / (rad/s); the
        # free root is left untouched so the episode still starts on the
        # reference. The policy must recover on its own.
        prng = np.random.default_rng(perturb_seed)
        nq_j = model.nq - 7
        qpos_j = data.qpos[7:].copy()
        qvel_j = data.qvel[6:].copy()
        data.qpos[7:] = qpos_j + prng.normal(0.0, perturb, size=nq_j)
        data.qvel[6:] = qvel_j + prng.normal(0.0, perturb, size=qvel_j.size)
        mujoco.mj_forward(model, data)
        obs = env._create_observation(model, data, env._additional_carry)[0]
        perturb_info = {
            "sigma_rad": perturb,
            "seed": perturb_seed,
            "max_abs_qpos_delta": float(np.max(np.abs(data.qpos[7:] - qpos_j))),
            "max_abs_qvel_delta": float(np.max(np.abs(data.qvel[6:] - qvel_j))),
        }
    reset_qpos = np.array(data.qpos, copy=True)
    ref0, start_subtraj_step, traj_no = reference_frame(env)
    ref_qpos0 = np.array(ref0.qpos, copy=True)
    traj_len = int(env.th.len_trajectory(traj_no))

    # forward axis = direction of the reference pelvis displacement over the
    # currently available clip, expressed in the initial root frame.
    motion_file = str(getattr(env.th, "traj_path", "") or "")
    split = np.asarray(env.th.traj.data.split_points)
    ref_all_qpos = np.asarray(env.th.traj.data.qpos)[int(split[traj_no]) : int(split[traj_no + 1])]
    ref_root_xy = ref_all_qpos[:, :2]
    d = ref_root_xy[min(len(ref_root_xy) - 1, start_step + 200)] - ref_root_xy[start_step]
    if np.linalg.norm(d) < 1e-6:
        d = ref_root_xy[-1] - ref_root_xy[start_step]
    forward = d / max(np.linalg.norm(d), 1e-9)
    lateral_axis = np.array([-forward[1], forward[0]])

    rng = jax.random.key(seed)
    train_state = bundle.agent_state.train_state

    steps = []
    terminated = False
    done = False
    info = {}
    reason = "other"
    failure_detail = ""
    numeric_anomaly = False
    reset_override = False
    fall_detected = False
    tilt_max_pelvis = 0.0
    tilt_max_torso = 0.0
    height_min = float("inf")
    ref_phase = 0.0

    prev_time = float(data.time)
    prev_qpos = reset_qpos.copy()
    prev_ref_step = start_subtraj_step
    mapper = getattr(getattr(env, "_goal", None), "_site_mapper", None)
    if mapper is not None and mapper.requires_mapping:
        ref_pelvis_idx = int(mapper.model_ids_to_traj_indices([pelvis_site])[0])
        ref_torso_idx = int(mapper.model_ids_to_traj_indices([torso_site])[0])
    else:
        ref_pelvis_idx, ref_torso_idx = pelvis_site, torso_site
    track_id = {"left": left_foot_track, "right": right_foot_track}
    track_name = {"left": next(iter(left_foot_ids)), "right": next(iter(right_foot_ids))}
    prev_foot_pos = {s: {} for s in ("left", "right")}
    for s in ("left", "right"):
        prev_foot_pos[s][track_name[s]] = np.array(data.xpos[track_id[s]], copy=True)

    slip = {"left": 0.0, "right": 0.0}
    contact_time = {"left": 0.0, "right": 0.0}
    contact_onsets = {"left": 0, "right": 0}
    prev_on = {"left": False, "right": False}
    double_support_steps = 0
    n_steps = 0

    video_frames = []
    if record_video:
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 1
        cam.distance = 3.6
        cam.azimuth = 135.0
        cam.elevation = -12.0
    else:
        renderer = None
        cam = None

    timeout_steps = int(round(seconds / float(env.dt))) + 1
    t0 = time.time()
    while n_steps <= timeout_steps:
        rng, _rng = jax.random.split(rng)
        action, train_state = policy_fn(train_state, obs, _rng)
        action = jnp.atleast_2d(action)
        obs, reward, absorbing, done, info = env.step(action)
        n_steps += 1

        t = float(data.time)
        qpos = np.array(data.qpos, copy=True)
        qvel = np.array(data.qvel, copy=True)
        ctrl = np.array(data.ctrl, copy=True)
        act = np.array(data.act, copy=True) if data.act.size else np.zeros(0)
        aforce = np.array(data.actuator_force, copy=True)
        cs = contact_state(model, data, geom_sets)
        qfrc_applied_max = float(np.max(np.abs(data.qfrc_applied))) if data.qfrc_applied.size else 0.0
        xfrc_applied_max = float(np.max(np.abs(data.xfrc_applied))) if data.xfrc_applied.size else 0.0

        # root / torso orientation
        root_pos = qpos[:3].copy()
        root_quat = qpos[3:7].copy()
        tilt_pelvis = quat_to_tilt_deg(root_quat)
        trunk_vec = data.site_xpos[torso_site] - data.site_xpos[pelvis_site]
        trunk_norm = float(np.linalg.norm(trunk_vec))
        tilt_torso = (
            float(np.degrees(np.arccos(np.clip(trunk_vec[2] / trunk_norm, -1.0, 1.0))))
            if trunk_norm > 1e-9
            else float("nan")
        )
        tilt_max_pelvis = max(tilt_max_pelvis, tilt_pelvis)
        tilt_max_torso = max(tilt_max_torso, tilt_torso)
        height_min = min(height_min, float(root_pos[2]))

        ref, ref_step, ref_traj = reference_frame(env)
        ref_phase = ref_step / max(traj_len - 1, 1)
        rt = ref.site_xpos[ref_torso_idx] - ref.site_xpos[ref_pelvis_idx]
        rtn = float(np.linalg.norm(rt))
        ref_trunk_tilt = (
            float(np.degrees(np.arccos(np.clip(rt[2] / rtn, -1.0, 1.0)))) if rtn > 1e-9 else float("nan")
        )

        # per-foot kinematics
        foot_info = {}
        for side in ("left", "right"):
            name = track_name[side]
            bid = track_id[side]
            pos = np.array(data.xpos[bid], copy=True)
            dt = t - prev_time
            vel = (pos - prev_foot_pos[side][name]) / dt if dt > 0 else np.zeros(3)
            prev_foot_pos[side][name] = pos
            foot_info[side] = {"pos": pos, "speed": float(np.linalg.norm(vel[:2]))}
            if cs[f"{side}_on"]:
                contact_time[side] += dt
                slip[side] += cs[f"{side}_slip"] * dt
                if not prev_on[side]:
                    contact_onsets[side] += 1
        if cs["left_on"] and cs["right_on"]:
            double_support_steps += 1
        prev_on = {"left": cs["left_on"], "right": cs["right_on"]}

        # sanity / implicit reset detection
        if t <= prev_time:
            reset_override = True
        if ref_step < prev_ref_step:
            reset_override = True
        if abs(t - prev_time - env.dt) > 1e-6:
            reset_override = True
        if n_steps > 2 and np.array_equal(qpos, reset_qpos):
            reset_override = True
        prev_time, prev_ref_step = t, ref_step

        bad = not (
            np.all(np.isfinite(qpos))
            and np.all(np.isfinite(qvel))
            and np.all(np.isfinite(ctrl))
            and np.isfinite(reward)
        )
        if bad:
            numeric_anomaly = True

        steps.append(
            {
                "step": n_steps - 1,
                "time": t,
                "qpos": qpos,
                "qvel": qvel,
                "ctrl": ctrl,
                "act": act,
                "actuator_force": aforce,
                "root_pos": root_pos,
                "root_quat": root_quat,
                "tilt_pelvis_deg": tilt_pelvis,
                "tilt_torso_deg": tilt_torso,
                "reward": float(reward),
                "ref_step": ref_step,
                "ref_phase": ref_phase,
                "ref_qpos": np.array(ref.qpos, copy=True),
                "ref_site_xpos": np.array(ref.site_xpos, copy=True),
                "contact_left_force": cs["left"],
                "contact_right_force": cs["right"],
                "contact_left": cs["left_on"],
                "contact_right": cs["right_on"],
                "foot_left_speed": foot_info["left"]["speed"],
                "foot_right_speed": foot_info["right"]["speed"],
                "foot_slip_speed_left": cs["left_slip"],
                "foot_slip_speed_right": cs["right_slip"],
                "qfrc_applied_max": qfrc_applied_max,
                "xfrc_applied_max": xfrc_applied_max,
                "ref_trunk_tilt_deg": ref_trunk_tilt,        
                "absorbing": bool(np.asarray(absorbing).item()),
                "done": bool(done),
            }
        )

        if record_video and (n_steps % video_every == 0):
            renderer.update_scene(data, camera=cam)
            video_frames.append((renderer.render().copy(), t - reset_time, n_steps - 1, tilt_pelvis, tilt_torso))

        if bad or reset_override:
            break
        if done:
            terminated = True
            break
        if t - reset_time >= seconds:
            break

    # ------------------------------------------------------------------
    # classify how the episode ended
    # ------------------------------------------------------------------
    duration = float(steps[-1]["time"] - reset_time) if steps else 0.0
    clip_end = False
    if done:
        ref, ref_step, _ = reference_frame(env)
        clip_end = bool(ref_step >= traj_len - 1)
    handler_checks = probe.checks(obs, info) if done else {}

    horizon_reached = bool(horizon is not None and n_steps >= int(horizon)) or bool(
        env.info.horizon is not None and n_steps >= int(env.info.horizon)
    )

    if reset_override:
        reason = "reset_or_state_override"
    elif numeric_anomaly:
        reason = "numeric_anomaly"
    elif height_min < FALL_HEIGHT or max(tilt_max_pelvis, tilt_max_torso) > FALL_TILT_DEG:
        reason = "physical_fall"
    elif done and (
        handler_checks.get("mean_site_deviation") is True
        or handler_checks.get("root_deviation") is True
        or handler_checks.get("root_orientation") is True
        or handler_checks.get("height_violation") is True
    ):
        reason = "reference_deviation"
    elif clip_end:
        reason = "clip_end"
    elif duration >= seconds - 1e-9:
        # Survived the full window. `done` may be true here because the env
        # truncates at `horizon`; that is a time limit, not a failure.
        reason = "time_limit_success"
    elif done and horizon_reached:
        reason = "time_limit_success"
    elif done:
        reason = "other"

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------
    root_xy = np.array([s["root_pos"][:2] for s in steps]) if steps else np.zeros((0, 2))
    ref_xy = np.array([s["ref_qpos"][:2] for s in steps]) if steps else np.zeros((0, 2))
    disp_total = root_xy[-1] - root_xy[0] if len(root_xy) else np.zeros(2)
    net_speed = float(np.linalg.norm(disp_total) / duration) if duration > 0 else 0.0
    # speed projected on the instantaneous reference heading (handles turns)
    forward_dist = 0.0
    lateral_cum = 0.0
    lateral_max = 0.0
    lateral_from_ref_max = 0.0
    cross_track_max = 0.0
    prev_heading = forward
    for i in range(1, len(root_xy)):
        dref = ref_xy[i] - ref_xy[i - 1]
        if np.linalg.norm(dref) > 1e-6:
            prev_heading = dref / np.linalg.norm(dref)
        dstep = root_xy[i] - root_xy[i - 1]
        forward_dist += float(np.dot(dstep, prev_heading))
        lateral_cum += float(np.dot(dstep, np.array([-prev_heading[1], prev_heading[0]])))
        lateral_max = max(lateral_max, abs(lateral_cum))
        # Rotation-invariant cross-track error: the policy-reference offset
        # projected on the *instantaneous* reference normal.
        rel_i = root_xy[i] - ref_xy[i]
        cross_track_max = max(cross_track_max, abs(float(np.dot(rel_i, np.array([-prev_heading[1], prev_heading[0]])))))
    # Heading error, computed on a 0.5 s window with a PCA direction and gated
    # by speed. Per-step displacement at 100 Hz is ~8 mm, so the instantaneous
    # direction is dominated by lateral sway and is undefined when the body is
    # nearly stationary; using it produced spurious 180 deg samples.
    HEAD_W = 50  # 0.5 s at 100 Hz

    def _windowed_heading(xy: np.ndarray) -> np.ndarray:
        out = np.full(len(xy), np.nan)
        for i in range(len(xy)):
            a = max(0, i - HEAD_W // 2)
            b = min(len(xy), i + HEAD_W // 2 + 1)
            seg = xy[a:b] - xy[a:b].mean(0)
            if len(seg) < 3 or np.linalg.norm(seg[-1] - seg[0]) < 1e-3:
                continue
            _, _, vt = np.linalg.svd(seg, full_matrices=False)
            v = vt[0]
            if np.dot(v, xy[b - 1] - xy[a]) < 0:
                v = -v
            out[i] = np.degrees(np.arctan2(v[1], v[0]))
        return out

    def _windowed_speed(xy: np.ndarray) -> np.ndarray:
        if len(xy) < 2:
            return np.zeros(len(xy))
        sp = np.concatenate([[0.0], np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt])
        k = np.ones(HEAD_W) / HEAD_W
        return np.convolve(sp, k, mode="same")

    heading_err = []
    n_low_speed_frames = 0
    if len(root_xy) > HEAD_W:
        hp_w = _windowed_heading(root_xy)
        hr_w = _windowed_heading(ref_xy)
        vp_w = _windowed_speed(root_xy)
        vr_w = _windowed_speed(ref_xy)
        for i in range(len(root_xy)):
            if vp_w[i] < 0.2 or vr_w[i] < 0.2:
                n_low_speed_frames += 1
                continue
            if not (np.isfinite(hp_w[i]) and np.isfinite(hr_w[i])):
                continue
            heading_err.append(abs(float((hp_w[i] - hr_w[i] + 180.0) % 360.0 - 180.0)))
    max_heading_err = max(heading_err) if heading_err else 0.0
    mean_heading_err = float(np.mean(heading_err)) if heading_err else 0.0
    net_heading_err = float(
        abs(
            np.degrees(
                (
                    np.arctan2(*(root_xy[-1] - root_xy[0])[::-1])
                    - np.arctan2(*(ref_xy[-1] - ref_xy[0])[::-1])
                    + np.pi
                )
                % (2 * np.pi)
                - np.pi
            )
        )
    ) if len(root_xy) > 1 else 0.0

    rel = root_xy - ref_xy
    for i in range(len(rel)):
        lateral_from_ref_max = max(lateral_from_ref_max, abs(float(np.dot(rel[i], lateral_axis))))
    # The initial-frame projection is NOT rotation invariant: on a turning clip a
    # perfectly tracked 180 degree turn shows a large value. It is kept only as a
    # straight-line diagnostic; the criterion uses rotation-invariant quantities.
    lateral_dev = max(lateral_max, cross_track_max)
    avg_forward_speed = forward_dist / duration if duration > 0 else 0.0

    ctrl_abs = np.array([np.abs(s["ctrl"]).mean() for s in steps]) if steps else np.zeros(1)
    act_mean = float(np.mean([s["act"].mean() for s in steps])) if steps else 0.0
    act_abs_mean = float(np.mean([np.abs(s["act"]).mean() for s in steps])) if steps else 0.0
    sat_frac = float(np.mean(ctrl_abs > 0.99)) if steps else 0.0

    qfrc_applied_max = max((s["qfrc_applied_max"] for s in steps), default=0.0)
    xfrc_applied_max = max((s["xfrc_applied_max"] for s in steps), default=0.0)
    ref_trunk_tilt_max = max((s["ref_trunk_tilt_deg"] for s in steps if np.isfinite(s["ref_trunk_tilt_deg"])), default=float("nan"))

    pelvis_err = (
        float(np.mean([np.linalg.norm(s["root_pos"][:2] - s["ref_qpos"][:2]) for s in steps]))
        if steps
        else 0.0
    )
    pelvis_err_max = (
        float(np.max([np.linalg.norm(s["root_pos"][:2] - s["ref_qpos"][:2]) for s in steps])) if steps else 0.0
    )

    criteria = {
        "duration_ge_10s": bool(duration >= seconds - 1e-9),
        "no_physical_fall": not (height_min < FALL_HEIGHT or max(tilt_max_pelvis, tilt_max_torso) > FALL_TILT_DEG),
        "no_numeric_anomaly": not numeric_anomaly,
        "no_reset_or_state_override": not reset_override,
        "reference_not_ended": not clip_end,
        "avg_forward_speed_ge_0.6": bool(avg_forward_speed >= MIN_FORWARD_SPEED),
        "max_lateral_le_0.35": bool(lateral_dev <= MAX_LATERAL_OFFSET),
        "max_tilt_le_45": bool(max(tilt_max_pelvis, tilt_max_torso) <= MAX_TILT_DEG),
    }
    success = all(criteria.values())
    blocking = [k for k, v in criteria.items() if not v]

    # gait-window diagnostics (between the first and the last foot-contact onset)
    left_on = np.array([s["contact_left"] for s in steps])
    right_on = np.array([s["contact_right"] for s in steps])
    onset_idx = [
        i
        for i in range(1, len(steps))
        if (left_on[i] and not left_on[i - 1]) or (right_on[i] and not right_on[i - 1])
    ]
    if onset_idx:
        w0, w1 = onset_idx[0], onset_idx[-1]
        walk_steps = steps[w0 : w1 + 1]
        walk_time = walk_steps[-1]["time"] - walk_steps[0]["time"] if len(walk_steps) > 1 else 0.0
        ds = sum(1 for s in walk_steps if s["contact_left"] and s["contact_right"])
        walk_forward = 0.0
        for i in range(w0 + 1, w1 + 1):
            dref = ref_xy[i] - ref_xy[i - 1]
            h = dref / np.linalg.norm(dref) if np.linalg.norm(dref) > 1e-6 else forward
            walk_forward += float(np.dot(root_xy[i] - root_xy[i - 1], h))
        walk_window = [float(steps[w0]["time"] - reset_time), float(steps[w1]["time"] - reset_time)]
        ds_frac = ds / max(len(walk_steps), 1)
        walk_forward_speed = walk_forward / walk_time if walk_time > 0 else 0.0
    else:
        walk_window = [0.0, 0.0]
        ds_frac = 0.0
        walk_forward_speed = 0.0

    slip_per_meter = {
        k: (slip[k] / forward_dist if forward_dist > 0.2 else float("nan")) for k in ("left", "right")
    }
    completed_clip_clean = (
        reason == "clip_end"
        and criteria["no_physical_fall"]
        and criteria["no_numeric_anomaly"]
        and criteria["no_reset_or_state_override"]
        and criteria["max_tilt_le_45"]
        and criteria["max_lateral_le_0.35"]
    )

    end_condition = reason
    if reason == "time_limit_success" and not success:
        failed = [k for k, v in criteria.items() if not v]
        reason = "other"
        failure_detail = "criteria_failed:" + ",".join(failed)

    result = {
        "seed": seed,
        "mode": mode,
        "checkpoint": str(checkpoint),
        "motion_path": motion_path,
        "motion_file": motion_file,
        "start_step": start_step,
        "traj_no": traj_no,
        "traj_len": traj_len,
        "control_dt": float(env.dt),
        "episode_id": 0,
        "reset_count": reset_count,
        "n_control_steps": len(steps),
        "duration_s": duration,
        "reset_time": reset_time,
        "reason": reason,
        "end_condition": end_condition,
        "survived_full_window": bool(duration >= seconds - 1e-9),
        "success": bool(success),
        "failure_detail": failure_detail,
        "criteria": criteria,
        "blocking_criteria": blocking,
        "clip_completed_without_failure": bool(completed_clip_clean),
        "walking_window_s": walk_window,
        "double_support_fraction_walking": ds_frac,
        "walking_window_forward_speed": walk_forward_speed,
        "slip_per_meter": slip_per_meter,
        "handler_checks_at_end": handler_checks,
        "clip_ended": clip_end,
        "perturbation": perturb_info,
        "horizon": int(env.info.horizon),
        "horizon_reached": horizon_reached,
        "env_done": bool(done),
        "avg_forward_speed": avg_forward_speed,
        "net_speed": net_speed,
        "forward_distance": forward_dist,
        "max_lateral_offset": lateral_max,
        "max_lateral_from_reference_initial_frame": lateral_from_ref_max,
        "max_cross_track_from_reference": cross_track_max,
        "max_lateral_deviation": lateral_dev,
        "max_heading_error_deg": max_heading_err,
        "mean_heading_error_deg": mean_heading_err,
        "net_heading_error_deg": net_heading_err,
        "heading_low_speed_frames": int(n_low_speed_frames),
        "max_tilt_pelvis_deg": tilt_max_pelvis,
        "max_tilt_torso_deg": tilt_max_torso,
        "min_root_height": height_min,
        "mean_ctrl_abs": float(ctrl_abs.mean()) if len(ctrl_abs) else 0.0,
        "ctrl_saturation_fraction": sat_frac,
        "mean_activation": act_mean,
        "mean_abs_activation": act_abs_mean,
        "foot_slip_distance": slip,
        "foot_contact_time": contact_time,
        "foot_contact_onsets": contact_onsets,
        "double_support_fraction": double_support_steps / max(len(steps), 1),
        "qfrc_applied_max": qfrc_applied_max,
        "xfrc_applied_max": xfrc_applied_max,
        "ref_trunk_tilt_max_deg": ref_trunk_tilt_max,
        "n_foot_contact_onsets_total": int(contact_onsets["left"] + contact_onsets["right"]),
        "pelvis_xy_error_mean": pelvis_err,
        "pelvis_xy_error_max": pelvis_err_max,
        "reward_sum": float(sum(s["reward"] for s in steps)),
        "ref_start_step": start_subtraj_step,
        "ref_end_step": steps[-1]["ref_step"] if steps else start_subtraj_step,
        "ref_end_phase": float(ref_phase),
        "wall_time_s": time.time() - t0,
        "build_time_s": t0 - t_build,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }

    # ------------------------------------------------------------------
    # persist
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{mode}_seed{seed}"
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2))

    if steps:
        np.savez_compressed(
            out_dir / f"{tag}.npz",
            time=np.array([s["time"] for s in steps]),
            qpos=np.array([s["qpos"] for s in steps]),
            qvel=np.array([s["qvel"] for s in steps]),
            ctrl=np.array([s["ctrl"] for s in steps]),
            act=np.array([s["act"] for s in steps]),
            actuator_force=np.array([s["actuator_force"] for s in steps]),
            root_pos=np.array([s["root_pos"] for s in steps]),
            root_quat=np.array([s["root_quat"] for s in steps]),
            ref_qpos=np.array([s["ref_qpos"] for s in steps]),
            ref_site_xpos=np.array([s["ref_site_xpos"] for s in steps]),
            ref_step=np.array([s["ref_step"] for s in steps]),
            contact_left_force=np.array([s["contact_left_force"] for s in steps]),
            contact_right_force=np.array([s["contact_right_force"] for s in steps]),
            contact_left=np.array([s["contact_left"] for s in steps]),
            contact_right=np.array([s["contact_right"] for s in steps]),
            foot_left_speed=np.array([s["foot_left_speed"] for s in steps]),
            foot_right_speed=np.array([s["foot_right_speed"] for s in steps]),
            foot_slip_speed_left=np.array([s["foot_slip_speed_left"] for s in steps]),
            foot_slip_speed_right=np.array([s["foot_slip_speed_right"] for s in steps]),
            tilt_pelvis_deg=np.array([s["tilt_pelvis_deg"] for s in steps]),
            tilt_torso_deg=np.array([s["tilt_torso_deg"] for s in steps]),
            reward=np.array([s["reward"] for s in steps]),
            done=np.array([s["done"] for s in steps]),
        )

    if record_video and video_frames:
        write_video(
            out_dir / f"{tag}.mp4",
            video_frames,
            fps=int(round(1.0 / (env.dt * video_every))),
            times=result,
        )

    del env, bundle, policy_fn, probe
    gc.collect()
    return result


def write_video(path: Path, frames, fps: int, times: dict) -> None:
    """Write frames whose (frame, t, step) stamps come from the logging loop."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, macro_block_size=1)
    try:
        for f, t, step, tilt_p, tilt_t in frames:
            img = Image.fromarray(f)
            draw = ImageDraw.Draw(img)
            draw.text(
                (10, 10),
                f"{times['mode']} seed={times['seed']} step={step} t={t:.2f}s",
                fill=(255, 255, 0),
            )
            draw.text(
                (10, 30),
                f"pelvis tilt={tilt_p:.1f}deg trunk tilt={tilt_t:.1f}deg",
                fill=(255, 255, 0),
            )
            writer.append_data(np.asarray(img))
    finally:
        writer.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/checkpoints/checkpoint_12500")
    ap.add_argument("--motion_path", default="KIT/314/walking_medium09_poses")
    ap.add_argument("--mode", choices=("stochastic", "deterministic"), default="stochastic")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seconds", type=float, default=SUCCESS_SECONDS)
    ap.add_argument("--start_step", type=int, default=0)
    ap.add_argument("--out_dir", default="runs/C_acceptance")
    ap.add_argument("--video", action="store_true", help="record an mp4 for each seed")
    ap.add_argument("--video_every", type=int, default=5)
    ap.add_argument("--strict_termination", action="store_true")
    ap.add_argument("--perturb", type=float, default=0.0, help="initial joint noise sigma (rad, rad/s)")
    ap.add_argument("--perturb_seed", type=int, default=0)
    ap.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override env horizon (control steps). Default uses the checkpoint value (1000 = 10 s).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        print(f"\n=== {args.mode} seed {seed} ===", flush=True)
        r = rollout(
            checkpoint=args.checkpoint,
            motion_path=args.motion_path,
            mode=args.mode,
            seed=seed,
            out_dir=out_dir,
            seconds=args.seconds,
            start_step=args.start_step,
            record_video=args.video,
            video_every=args.video_every,
            strict_termination=args.strict_termination,
            horizon=args.horizon,
            perturb=args.perturb,
            perturb_seed=args.perturb_seed,
        )
        results.append(r)
        print(
            f"  {r['reason']:24s} dur={r['duration_s']:6.2f}s "
            f"v_forward={r['avg_forward_speed']:.3f} m/s lat={r['max_lateral_offset']:.3f} "
            f"tilt={r['max_tilt_pelvis_deg']:.1f}/{r['max_tilt_torso_deg']:.1f} "
            f"ref_end={r['ref_end_step']}/{r['traj_len']-1}",
            flush=True,
        )

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    keys = [
        "seed",
        "reason",
        "success",
        "duration_s",
        "avg_forward_speed",
        "net_speed",
        "max_lateral_offset",
        "max_cross_track_from_reference",
        "max_lateral_deviation",
        "max_heading_error_deg",
        "mean_heading_error_deg",
        "net_heading_error_deg",
        "max_tilt_pelvis_deg",
        "max_tilt_torso_deg",
        "min_root_height",
        "n_control_steps",
        "ref_end_step",
        "traj_len",
        "foot_contact_onsets",
        "n_foot_contact_onsets_total",
        "clip_completed_without_failure",
        "walking_window_s",
        "double_support_fraction_walking",
        "walking_window_forward_speed",
        "slip_per_meter",
        "foot_slip_distance",
        "ref_trunk_tilt_max_deg",
        "wall_time_s",
        "max_rss_mb",
    ]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in keys}
            for k in ("foot_contact_onsets", "foot_slip_distance", "walking_window_s", "slip_per_meter"):
                row[k] = json.dumps(row.get(k))
            w.writerow(row)

    n_ok = sum(1 for r in results if r["success"])
    n_clip = sum(1 for r in results if r.get("clip_completed_without_failure"))
    print(f"\n{n_ok}/{len(results)} seeds met all criteria")
    print(f"{n_clip}/{len(results)} seeds completed the available reference clip without failure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
