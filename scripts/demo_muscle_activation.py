#!/usr/bin/env python3
"""Render a walking demo and a separate grid-style muscle-activation video.

Steps the frozen policy through a single episode (no reset, same loop as
``eval_walk.py``) and writes, as *separate* time-synchronised mp4 files:

  * the walking video (MyoFullBody scene only);
  * a grid-style muscle-activation heatmap: all 354 Hill muscles arranged as
    coloured tiles in a fixed grid, grouped by anatomical region. Each tile is
    one muscle, coloured by its activation (0 slack -> 1 fully active).

Both outputs are produced from the *same* rollout, so they stay in sync when
played side by side.

Usage:

    export PYTHONPATH=external/MuscleMimic:scripts MUJOCO_GL=egl
    .venv/bin/python scripts/demo_muscle_activation.py \
        --motion_path KIT/314/walking_medium09_poses --seconds 10 --seed 0 \
        --out reports/video/demo_walk.mp4 \
        --out_activation reports/video/demo_activation_grid.mp4

``--style side`` additionally writes a single combined frame (body + panel).
``--npz`` replays an already-saved rollout instead of stepping the policy.
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
GRID_COLS = 15  # muscles per row in the activation grid


def _even(frame: np.ndarray) -> np.ndarray:
    """Crop to even width/height (h264 yuv420p requires even dimensions)."""
    h, w = frame.shape[:2]
    return frame[: h // 2 * 2, : w // 2 * 2]


def make_grid_layout(order, bounds, cols=GRID_COLS):
    """Build a fixed grid layout for the 354 muscles (NaN = empty cell)."""
    rows, labels = [], []
    offset = 0
    for g in GROUP_ORDER:
        n = bounds[g]
        nrows = max(1, int(np.ceil(n / cols)))
        block = np.full((nrows, cols), np.nan)
        labels.append((offset + (nrows - 1) / 2.0, GROUP_LABELS[g], n))
        offset += nrows
        rows.append(block)
        if g != GROUP_ORDER[-1]:
            rows.append(np.full((1, cols), np.nan))  # separator row
            offset += 1
    return np.vstack(rows), labels


def fill_grid(layout, order, bounds, act, cols=GRID_COLS):
    """Copy per-muscle activation into the fixed grid layout."""
    grid = layout.copy()
    src = 0
    for gi, g in enumerate(GROUP_ORDER):
        n = bounds[g]
        nrows = max(1, int(np.ceil(n / cols)))
        top = 0
        for gg in GROUP_ORDER[:gi]:
            top += max(1, int(np.ceil(bounds[gg] / cols))) + (1 if gg != GROUP_ORDER[-1] else 0)
        block = np.full((nrows, cols), np.nan)
        block.flat[:n] = act[src : src + n]
        grid[top : top + nrows, :] = block
        src += n
    return grid


def build_grid_frame(act, order, bounds, t):
    """One standalone grid-heatmap frame (no body)."""
    layout, _ = make_grid_layout(order, bounds)
    grid = fill_grid(layout, order, bounds, act)
    nrows, ncols = grid.shape

    fig, ax = plt.subplots(figsize=(max(6.0, ncols * 0.55), max(3.0, nrows * 0.32)), dpi=110)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("0.85")  # empty cells: light grey
    im = ax.imshow(grid, aspect="equal", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    # group separators + labels
    yticks, yticklabels = [], []
    offset = 0
    for g in GROUP_ORDER:
        n = bounds[g]
        nrows_g = max(1, int(np.ceil(n / GRID_COLS)))
        yticks.append(offset + (nrows_g - 1) / 2.0)
        yticklabels.append(GROUP_LABELS[g])
        offset += nrows_g
        if g != GROUP_ORDER[-1]:
            ax.axhline(offset - 0.5, color="white", lw=1.0)
            offset += 1
    ax.set_yticks(yticks, yticklabels, fontsize=8)
    ax.set_xticks([])
    ax.set_title(f"muscle activation (354 muscles)   t = {t:5.2f} s", fontsize=11, pad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("activation", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return _even(frame)


def renderer_camera(model, height=480, width=640):
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1
    cam.distance = 3.6
    cam.azimuth = 135.0
    cam.elevation = -12.0
    return renderer, cam


def body_frame(renderer, data, cam, t):
    """Render one body-only frame with a timestamp strip."""
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"t = {t:5.2f} s", fontsize=11)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return _even(frame)


def side_frame(renderer, data, cam, act, order, bounds, t):
    """Combined body + grid panel (previous style, optional)."""
    renderer.update_scene(data, camera=cam)
    body = renderer.render()  # raw body, no timestamp wrapper

    grid_frame = build_grid_frame(act, order, bounds, t)
    # left-align body to the grid frame height
    gh, gw = grid_frame.shape[:2]
    bh = int(gh * 0.62)
    bw = int(bh * 640 / 480)
    from PIL import Image

    body_img = np.asarray(Image.fromarray(body).resize((bw, bh)))
    left = np.full((gh, bw, 3), 0, dtype=np.uint8)
    left[:bh, :] = body_img
    return _even(np.concatenate([left, grid_frame], axis=1))


def open_writer(path, fps):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(path), fps=fps, macro_block_size=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="artifacts/checkpoints/checkpoint_12500")
    ap.add_argument("--motion_path", default="KIT/314/walking_medium09_poses")
    ap.add_argument("--mode", choices=("stochastic", "deterministic"), default="stochastic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--start_step", type=int, default=0)
    ap.add_argument("--video_every", type=int, default=5, help="render every Nth control step")
    ap.add_argument("--style", choices=("grid", "side"), default="grid",
                    help="grid: standalone activation grid; side: body+grid combined")
    ap.add_argument("--out", default="reports/video/demo_activation_grid.mp4")
    ap.add_argument("--out_body", default=None, help="also write a separate walking video (body only)")
    ap.add_argument("--json_out", default=None, help="per-frame group-mean activation JSON")
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
    fps = int(round(1.0 / (float(bundle.env.dt) * args.video_every)))

    writers = {}
    try:
        if args.style == "grid":
            writers["grid"] = open_writer(args.out, fps)
        else:
            writers["side"] = open_writer(args.out, fps)
        if args.out_body:
            writers["body"] = open_writer(args.out_body, fps)

        group_mean_series = []

        def emit(t):
            act = np.asarray(data.act, dtype=float)[order]
            means = [float(np.mean(data.act[groups[g]["indices"]])) for g in GROUP_ORDER]
            group_mean_series.append({"t": float(t), **dict(zip(GROUP_ORDER, means))})
            if "grid" in writers:
                writers["grid"].append_data(build_grid_frame(act, order, bounds, t))
            if "side" in writers:
                writers["side"].append_data(side_frame(renderer, data, cam, act, order, bounds, t))
            if "body" in writers:
                writers["body"].append_data(body_frame(renderer, data, cam, t))

        if args.npz:
            d = np.load(args.npz)
            qpos = np.asarray(d["qpos"], float)
            qvel = np.asarray(d["qvel"], float)
            act = np.asarray(d["act"], float)
            t = np.asarray(d["time"], float)
            for i in range(0, len(qpos), args.video_every):
                data.qpos[:] = qpos[i]
                data.qvel[:] = qvel[i]
                data.act[:] = act[i]
                data.ctrl[:] = 0.0
                mujoco.mj_forward(model, data)
                emit(float(t[i]))
        else:
            policy_fn = make_policy_fn(bundle.agent_conf, deterministic=(args.mode == "deterministic"))
            train_state = bundle.agent_state.train_state
            rng = jax.random.key(args.seed)

            obs = bundle.env.reset()
            n_steps = 0
            timeout_steps = int(round(args.seconds / float(bundle.env.dt))) + 1
            while n_steps <= timeout_steps:
                rng, _rng = jax.random.split(rng)
                action, train_state = policy_fn(train_state, obs, _rng)
                obs, _, _, done, _ = bundle.env.step(action)
                n_steps += 1
                if n_steps % args.video_every == 0:
                    emit(float(data.time))
                if done:
                    break
    finally:
        for w in writers.values():
            w.close()

    print(f"saved {args.out}  ({len(group_mean_series)} frames @ {fps} fps)")
    if args.out_body:
        print(f"saved {args.out_body}  ({len(group_mean_series)} frames @ {fps} fps)")

    if args.json_out:
        jp = Path(args.json_out)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(group_mean_series, indent=2))
        print(f"saved {jp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
