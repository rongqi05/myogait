"""Render an mp4 video from a saved rollout .npz (exact stored trajectory).

Reads qpos per stored control step, applies it to a freshly built MyoFullBody
model, and renders with the same tracking camera as ``eval_walk.py``. This
replays the *recorded* dynamics (no re-simulation of the policy), so the video
matches the per-seed JSON/NPZ evidence exactly.

Usage:
    PYTHONPATH=external/MuscleMimic:scripts .venv/bin/python scripts/render_npz_video.py \
        --npz runs/f0_ablation/A/stochastic/stochastic_seed0.npz \
        --out reports/video/upper_sweep_U1.0_seed0.mp4 \
        --label "left-upper F0=1.0  seed 0" --every 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _env_builder import build_eval_bundle, env_data, env_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--every", type=int, default=5, help="render every Nth control step")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--checkpoint", default="artifacts/checkpoints/checkpoint_12500")
    ap.add_argument("--motion_path", default="CMU/120/cmu120_19_straight_h000")
    ap.add_argument("--start_step", type=int, default=53)
    args = ap.parse_args()

    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    d = np.load(args.npz)
    qpos = np.asarray(d["qpos"], float)
    qvel = np.asarray(d["qvel"], float)
    t = np.asarray(d["time"], float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.01
    fps = int(round(1.0 / (dt * args.every)))

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

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1
    cam.distance = 3.6
    cam.azimuth = 135.0
    cam.elevation = -12.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=fps, macro_block_size=1)
    try:
        for i in range(0, len(qpos), args.every):
            data.qpos[:] = qpos[i]
            data.qvel[:] = qvel[i]
            data.act[:] = 0.0
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            img = renderer.render().copy()
            im = Image.fromarray(img)
            draw = ImageDraw.Draw(im)
            draw.text((10, 10), f"{args.label}", fill=(255, 255, 0))
            draw.text((10, 30), f"t={t[i]:.2f}s  step={i}", fill=(255, 255, 0))
            writer.append_data(np.asarray(im))
    finally:
        writer.close()
    print(f"saved {out}  ({len(range(0, len(qpos), args.every))} frames @ {fps} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
