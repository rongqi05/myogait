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


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_calibration(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing calibration {path}; run --mode calibrate first")
    return json.loads(path.read_text())


def crutch_cfg(calib: dict, side: str | None) -> dict:
    cfg = {k: CRUTCH_DEFAULTS[k] for k in CRUTCH_KEYS}
    off = calib["offsets_by_side"][side] if side else calib["offsets"]
    cfg.update(
        prefix=PREFIX,
        hand_body={"left": "radius_l", "right": "radius_r"},
        hand_site={"left": "left_hand_mimic", "right": "right_hand_mimic"},
        forward_offset=off["forward"],
        lateral_offset=abs(off["lateral"]),
    )
    return cfg


def build(merged: bool, calib: dict, horizon: int, render_size=None):
    mount = crutch_env.CrutchMount(xml=calib["crutch_xml"], rod_length=calib["rod_length"]) if merged else None
    crutch_env.install(mount, render_size=render_size)
    return build_eval_bundle(checkpoint_path=CKPT, motion_path=MOTION, start_step=0,
                             traj_index=0, horizon=horizon)


# --------------------------------------------------------------------------- #
# calibrate
# --------------------------------------------------------------------------- #
def calibrate(out_dir: Path, seconds: float) -> int:
    b = build(merged=False, calib={}, horizon=int(seconds / 0.01) + 5)
    env, m, d = b.env, env_model(b.env), env_data(b.env)
    obs = np.asarray(env.reset())
    pf = make_policy_fn(b.agent_conf, mode == "deterministic")
    rng = jax.random.key(0)

    r0 = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in crutch_env.reach(m, d, "left").items()
          if not isinstance(v, np.ndarray)}
    r1 = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in crutch_env.reach(m, d, "right").items()
          if not isinstance(v, np.ndarray)}
    hs = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_hand_mimic") for s in ("left", "right")}

    rows = []
    for _ in range(int(round(seconds / env.dt))):
        a, _ts = pf(b.agent_state.train_state, obs, rng)
        obs, *_ = env.step(a)
        root = np.array(d.qpos[:3], copy=True)
        rows.append([*(np.array(d.site_xpos[hs["left"]]) - root),
                     *(np.array(d.site_xpos[hs["right"]]) - root),
                     float(d.site_xpos[hs["left"]][2]), float(d.site_xpos[hs["right"]][2])])
    arr = np.asarray(rows)
    hand_l, hand_r = arr[:, 0:3], arr[:, 3:6]
    z_l, z_r = arr[:, 6], arr[:, 7]

    k_act = min(len(arr) - 1, int(round(ACT_FRAME / env.dt)))
    hand_z_mean = float(0.5 * (z_l[k_act] + z_r[k_act]))
    calib = {
        "crutch_xml": "crutch_robot.xml",
        "checkpoint": CKPT,
        "motion": MOTION,
        "start_step": 0,
        "control_dt": float(env.dt),
        "hand_z_both_at_t0": round(float(0.5 * (z_l[0] + z_r[0])), 4),
        "hand_z_both_at_activation_frame": round(hand_z_mean, 4),
        "hand_z_left_mean_over_clip": round(float(z_l.mean()), 4),
        "hand_z_right_mean_over_clip": round(float(z_r.mean()), 4),
        "activation_frame_s": ACT_FRAME,
        "clearance": CRUTCH_DEFAULTS["clearance"],
        "chassis_half_z": CRUTCH_DEFAULTS["chassis_half_size"][2],
        "rod_length": round(hand_z_mean - CRUTCH_DEFAULTS["clearance"] - 2 * CRUTCH_DEFAULTS["chassis_half_size"][2], 4),
        "offsets": {"forward": round(float(0.5 * (hand_l[:, 0].mean() + hand_r[:, 0].mean())), 4),
                    "lateral": round(float(0.5 * (abs(hand_l[:, 1]).mean() + abs(hand_r[:, 1]).mean())), 4)},
        "offsets_by_side": {
            "left": {"forward": round(float(hand_l[k_act, 0]), 4), "lateral": round(float(hand_l[k_act, 1]), 4)},
            "right": {"forward": round(float(hand_r[k_act, 0]), 4), "lateral": round(float(hand_r[k_act, 1]), 4)},
        },
        "hand_offset_std": {"forward": round(float(0.5 * (hand_l[:, 0].std() + hand_r[:, 0].std())), 4),
                            "lateral": round(float(0.5 * (hand_l[:, 1].std() + hand_r[:, 1].std())), 4)},
        "reach_at_t0": {"left": r0, "right": r1},
        "seconds": seconds,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=2))
    print(json.dumps(calib, indent=2))
    del b
    gc.collect()
    return 0


# --------------------------------------------------------------------------- #
# run one condition
# --------------------------------------------------------------------------- #
def run_condition(cond: str, seeds: list[int], seconds: float, out_dir: Path,
                  calib: dict, video: bool, settle: float, mode: str = "stochastic") -> int:
    grip_side = COND_GRIP[cond]
    side = COND_SIDE[cond]
    merged = cond != "A"
    horizon = int(round(seconds / calib["control_dt"])) + 5
    b = build(merged=merged, calib=calib, horizon=horizon,
              render_size=(1280, 720) if video else None)
    env, m, d = b.env, env_model(b.env), env_data(b.env)
    pf = make_policy_fn(b.agent_conf, mode == "deterministic")
    ci = CrutchInterface(m, d, cfg=crutch_cfg(calib, side)) if merged else None
    layout = crutch_env.layout(m) if merged else None
    if merged:
        print(f"[layout] nq={layout['nq']} nv={layout['nv']} nu={layout['nu']} "
              f"crutch_act={layout['crutch_act_ids']} first_crutch_qpos={layout['first_crutch_qpos']}")

    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pelvis_mimic")
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "upper_body_mimic")
    grip_site = ci.site_grip if merged else -1
    hand_site_id = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_hand_mimic") for s in ("left", "right")}

    out_dir.mkdir(parents=True, exist_ok=True)
    tgt_site = "torso" if torso >= 0 else "pelvis"
    renderer = None
    if video:
        renderer = mujoco.Renderer(m, height=720, width=1280)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 1
        cam.distance = 4.4
        cam.azimuth = 133.0
        cam.elevation = -11.0
        _ = tgt_site

    summary = []
    for seed in seeds:
        obs = np.asarray(env.reset())
        rng = jax.random.key(seed)
        q0 = np.array(d.qpos, copy=True)
        align_info, act_info = {}, {}
        if merged:
            ci.mode, ci.affected_side, ci.active_side = "none", side, None
            ci.cfg["free_side"] = side
            ci._heading = np.array([1.0, 0.0])
            # follow-only warm-up: the base reaches its offset target before the grip closes
            n_settle = int(round(settle / ci.control_dt))
            for _ in range(n_settle):
                rng, _rng = jax.random.split(rng)
                a, _ts = pf(b.agent_state.train_state, obs, _rng)
                ci.command(d.qpos[:3], d.qvel[:3])
                obs, *_ = env.step(a)
                ci.observe()
            if grip_side:
                align_info = ci.align_grip(grip_side)
            ci.tare_now()
            gap_before = float(np.linalg.norm(d.site_xpos[grip_site] - d.site_xpos[hand_site_id[grip_side]])) if grip_side else float("nan")
            if grip_side:
                # the impaired side is always left here: C is ipsilateral, D contralateral
                mode = "ipsilateral" if grip_side == "left" else "contralateral"
                act_info = ci.set_assistance(mode, "left")
                act_info["gap_before_activation_m"] = gap_before
                if not act_info["ok"]:
                    print(f"[warn] {cond} seed {seed}: grip not activated: {act_info}")

        steps = []
        reason = "other"
        n = 0
        timeout = int(round(seconds / env.dt)) + 1
        frames = []
        q_prev = np.array(d.qpos, copy=True)
        while n <= timeout:
            rng, _rng = jax.random.split(rng)
            a, _ts = pf(b.agent_state.train_state, obs, _rng)
            crutch_out = {}
            if merged:
                ci.command(d.qpos[:3], d.qvel[:3])
            obs, reward, absorbing, done, info = env.step(a)
            if merged:
                crutch_out = ci.observe()
            n += 1
            qpos = np.array(d.qpos, copy=True)
            qvel = np.array(d.qvel, copy=True)
            root = qpos[:3]
            tilt_p = quat_to_tilt_deg(qpos[3:7])
            tv = d.site_xpos[torso] - d.site_xpos[pelvis]
            tilt_t = float(np.degrees(np.arccos(np.clip(tv[2] / max(np.linalg.norm(tv), 1e-9), -1, 1))))
            rec = {
                "t": float(d.time), "qpos": qpos, "qvel": qvel,
                "ctrl_muscle_max": float(np.max(np.abs(d.ctrl[:layout["first_crutch_qpos"] if False else 354]))),
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
                    "eq_active": crutch_out["eq_active"],
                })
            steps.append(rec)
            if renderer is not None and n % 5 == 0:
                renderer.update_scene(d, camera=cam)
                frames.append(renderer.render().copy())

            numeric = not (np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel)))
            fall = root[2] < FALL_HEIGHT or max(tilt_p, tilt_t) > FALL_TILT_DEG
            if numeric:
                reason = "numeric_anomaly"
            elif fall:
                reason = "physical_fall"
            elif done:
                reason = "env_done"
            if reason != "other":
                break
        if reason == "other":
            reason = "time_limit_reached" if n > timeout else "incomplete"

        root_xy = np.array([s["root"][:2] for s in steps])
        fwd = float(root_xy[-1, 0] - root_xy[0, 0])
        dur = float(steps[-1]["t"] - 0.0)
        rec_summary = {
            "condition": cond, "seed": seed, "reason": reason,
            "duration_s": round(dur, 3),
            "completed_target": bool(dur >= seconds - 1e-6),
            "forward_x_m": round(fwd, 3),
            "avg_forward_speed": round(fwd / dur, 3) if dur > 0 else 0.0,
            "max_tilt_torso_deg": round(max(s["tilt_torso"] for s in steps), 2),
            "max_tilt_pelvis_deg": round(max(s["tilt_pelvis"] for s in steps), 2),
            "min_root_z": round(min(float(s["root"][2]) for s in steps), 3),
            "max_lateral_abs_m": round(float(np.max(np.abs(root_xy[:, 1] - root_xy[0, 1]))), 3),
            "align": align_info, "assist": act_info,
        }
        if merged:
            fz = np.array([s["support_up"] for s in steps])
            fx = np.array([s["force_world"][0] for s in steps])
            fy = np.array([s["force_world"][1] for s in steps])
            gap = np.array([s["grip_gap"] for s in steps])
            cv = np.array([s["chassis_vel"] for s in steps])
            rec_summary.update({
                "fz_mean_N": round(float(fz.mean()), 2), "fz_min_N": round(float(fz.min()), 2),
                "fz_max_N": round(float(fz.max()), 2),
                "fz_abs_mean_N": round(float(np.abs(fz).mean()), 2),
                "fz_first_step_N": round(float(fz[0]), 2),
                "fx_abs_mean_N": round(float(np.abs(fx).mean()), 2),
                "fy_abs_mean_N": round(float(np.abs(fy).mean()), 2),
                "grip_gap_max_m": round(float(np.nanmax(gap)), 4),
                "grip_gap_mean_m": round(float(np.nanmean(gap)), 4),
                "chassis_speed_mean": round(float(np.linalg.norm(cv, axis=1).mean()), 3),
                "chassis_speed_max": round(float(np.linalg.norm(cv, axis=1).max()), 3),
            })
        summary.append(rec_summary)
        print(json.dumps(rec_summary, ensure_ascii=False))

        np.savez_compressed(
            out_dir / f"{cond}_seed{seed}.npz",
            time=np.array([s["t"] for s in steps]),
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
                "v_cmd": np.array([s["v_cmd"] for s in steps])} if merged else {}),
        )
        (out_dir / f"{cond}_seed{seed}.json").write_text(json.dumps(rec_summary, indent=2))
        if renderer is not None and frames:
            import imageio.v2 as iio
            fps = int(round(1.0 / (env.dt * 5)))
            iio.mimwrite(out_dir / f"{cond}_seed{seed}.mp4", frames, fps=fps, macro_block_size=1)

    with (out_dir / f"summary_{cond}.csv").open("w", newline="") as fh:
        keys = [k for k in summary[0] if not isinstance(summary[0][k], dict)]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)
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
    args = ap.parse_args()

    out = Path(args.out_dir)
    calib_path = Path(args.calib) if args.calib else out / "calibration.json"

    if args.mode == "calibrate":
        return calibrate(out, max(args.seconds, 10.0))
    if args.mode == "compose":
        return compose(out, args.seeds)
    calib = load_calibration(calib_path)
    t0 = time.time()
    print(f"[{args.condition}] policy={args.policy_mode} seconds={args.seconds} seeds={args.seeds} "
          f"rod_length={calib['rod_length']} offsets={calib['offsets']}")
    rc = run_condition(args.condition, args.seeds, args.seconds, out / f"{args.seconds:g}s",
                       calib, args.video, args.settle, args.policy_mode)
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
