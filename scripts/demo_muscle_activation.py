#!/usr/bin/env python3
"""Render a walking demo with a live muscle-activation panel.

Steps the frozen policy through a single episode (no reset, same loop as
``eval_walk.py``) and, every few control steps, draws a frame composed of:

  * left  : the MyoFullBody scene (MuJoCo tracking camera);
  * right : a per-muscle activation heatmap (all 354 Hill muscles, grouped by
            anatomical region) plus the per-group mean activation bars.

Colour scale: activation 0 (slack) -> 1 (fully active), ``inferno`` colormap.
The output is a time-synchronised mp4, so the heatmap moves in real time with
the gait.

Usage:

    export PYTHONPATH=external/MuscleMimic:scripts MUJOCO_GL=egl
    .venv/bin/python scripts/demo_muscle_activation.py \
        --motion_path KIT/314/walking_medium09_poses --seconds 10 --seed 0 \
        --out reports/video/demo_activation.mp4

Pass ``--npz runs/.../stochastic_seed0.npz`` to replay an already-saved rollout
instead of running the policy (the recorded ``act`` array is visualised).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
os.environ.setdefault("MUJOCO_GL", "egl")

import jax  # noqa: E402
import mujoco  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

from _env_builder import build_eval_bundle, env_data, env_model, muscle_groups  # noqa: E402
from eval_walk import make_policy_fn  # noqa: E402

GROUP_ORDER = ("left_upper", "right_upper", "left_lower", "right_lower", "left_trunk", "right_trunk")
GROUP_LABELS = {
    "left_upper": "L arm",
    "right_upper": "R arm",
    "left_lower": "L leg",
    "right_lower": "R leg",
    "left_trunk": "L trunk",
    "right_trunk": "R trunk",
}
SIDE_COLORS = {"L": "#4c72b0", "R": "#c44e52"}


def build_panel(renderer, data, cam, groups, order, bounds, group_means, t):
    """Compose one frame: body image + activation heatmap + group bars."""
    renderer.update_scene(data, camera=cam)
    body = renderer.render()  # (H, W, 3) uint8

    act = np.asarray(data.act, dtype=float)[order]  # (354,)

    fig = plt.figure(figsize=(13.2, 5.2), dpi=100)
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.35, 0.55], height_ratios=[1.0, 2.2],
        left=0.015, right=0.985, top=0.96, bottom=0.05, wspace=0.10, hspace=0.28,
    )
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(body, interpolation="bilinear")
    ax_img.axis("off")
    ax_img.set_title(f"t = {t:5.2f} s", fontsize=11, pad=4)

    # ---- per-group mean activation (top right) ---------------------------
    ax_bars = fig.add_subplot(gs[0, 1])
    labels = []
    colors = []
    for g in GROUP_ORDER:
        side = GROUP_LABELS[g][0]
        labels.append(GROUP_LABELS[g])
        colors.append(SIDE_COLORS[side])
    y = np.arange(len(GROUP_ORDER))[::-1]
    ax_bars.barh(y, group_means, color=colors, height=0.65)
    ax_bars.set_yticks(y, labels, fontsize=8)
    ax_bars.set_xlim(0, 1)
    ax_bars.set_xlabel("mean activation", fontsize=8)
    ax_bars.tick_params(axis="x", labelsize=8)
    ax_bars.set_title("group mean activation", fontsize=9, pad=4)
    for yi, v in zip(y, group_means):
        ax_bars.text(min(v + 0.03, 0.97), yi, f"{v:.2f}", va="center", fontsize=7)

    # ---- per-muscle heatmap (bottom right) --------------------------------
    ax_hm = fig.add_subplot(gs[1, 1])
    col = act[:, None]  # (354, 1)
    im = ax_hm.imshow(col, aspect="auto", cmap="inferno", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_hm.set_yticks([], [])
    ax_hm.set_xticks([], [])
    # group separators + labels
    b = 0
    for g in GROUP_ORDER:
        n = bounds[g]
        ax_hm.axhline(b + n, color="white", lw=0.6, alpha=0.6)
        ax_hm.text(-0.35, b + n / 2, GROUP_LABELS[g], va="center", ha="right", fontsize=7,
                   transform=ax_hm.get_yaxis_transform())
        b += n
    ax_hm.set_ylim(354, 0)
    ax_hm.set_title("muscle activation (354 muscles)", fontsize=9, pad=4)
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.06, pad=0.08)
    cbar.set_label("activation", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def renderer_camera(model, height=480, width=640):
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1
    cam.distance = 3.6
    cam.azimuth = 135.0
    cam.elevation = -12.0
    return renderer, cam


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="artifacts/checkpoints/checkpoint_12500")
    ap.add_argument("--motion_path", default="KIT/314/walking_medium09_poses")
    ap.add_argument("--mode", choices=("stochastic", "deterministic"), default="stochastic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--start_step", type=int, default=0)
    ap.add_argument("--video_every", type=int, default=5, help="render every Nth control step")
    ap.add_argument("--out", default="reports/video/demo_activation.mp4")
    ap.add_argument("--json_out", default=None, help="optional per-frame group-mean activation JSON")
    ap.add_argument("--npz", default=None, help="replay a saved rollout npz instead of stepping the policy")
    args = ap.parse_args()

    bundle = build_eval_bundle(
        checkpoint_path=args.checkpoint,
        motion_path=args.motion_path,
        start_step=args.start_step,
        traj_index=0,
        strict_termination=False,
        horizon=None,
    )
    model = env_model(bundle.env)
    data = env_data(bundle.env)
    groups = muscle_groups(model)

    order_parts, bounds = [], {}
    for g in GROUP_ORDER:
        idx = np.asarray(groups[g]["indices"], dtype=int)
        order_parts.append(idx)
        bounds[g] = len(idx)
    order = np.concatenate(order_parts)

    renderer, cam = renderer_camera(model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = int(round(1.0 / (float(bundle.env.dt) * args.video_every)))
    writer = imageio.get_writer(str(out), fps=fps, macro_block_size=1)

    group_mean_series = []

    if args.npz:
        d = np.load(args.npz)
        qpos = np.asarray(d["qpos"], float)
        qvel = np.asarray(d["qvel"], float)
        act = np.asarray(d["act"], float)
        t = np.asarray(d["time"], float)
        try:
            for i in range(0, len(qpos), args.video_every):
                data.qpos[:] = qpos[i]
                data.qvel[:] = qvel[i]
                data.act[:] = act[i]
                data.ctrl[:] = 0.0
                mujoco.mj_forward(model, data)
                means = [float(np.mean(data.act[groups[g]["indices"]])) for g in GROUP_ORDER]
                group_mean_series.append({"i": int(i), "t": float(t[i]), **dict(zip(GROUP_ORDER, means))})
                frame = build_panel(renderer, data, cam, groups, order, bounds, np.asarray(means), float(t[i]))
                writer.append_data(frame)
        finally:
            writer.close()
    else:
        policy_fn = make_policy_fn(bundle.agent_conf, deterministic=(args.mode == "deterministic"))
        train_state = bundle.agent_state.train_state
        rng = jax.random.key(args.seed)

        obs = bundle.env.reset()
        n_steps = 0
        timeout_steps = int(round(args.seconds / float(bundle.env.dt))) + 1
        try:
            while n_steps <= timeout_steps:
                rng, _rng = jax.random.split(rng)
                action, train_state = policy_fn(train_state, obs, _rng)
                obs, _, _, done, _ = bundle.env.step(action)
                n_steps += 1
                t = float(data.time)
                if n_steps % args.video_every == 0:
                    means = [float(np.mean(data.act[groups[g]["indices"]])) for g in GROUP_ORDER]
                    group_mean_series.append({"i": int(n_steps - 1), "t": t, **dict(zip(GROUP_ORDER, means))})
                    frame = build_panel(renderer, data, cam, groups, order, bounds, np.asarray(means), t)
                    writer.append_data(frame)
                if done:
                    break
        finally:
            writer.close()

    print(f"saved {out}  ({len(group_mean_series)} frames @ {fps} fps)")

    if args.json_out:
        jp = Path(args.json_out)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(group_mean_series, indent=2))
        print(f"saved {jp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
