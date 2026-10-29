#!/usr/bin/env python3
"""Phase A audit: model, checkpoint, motion and observation/action interfaces.

Run:
    PYTHONPATH=external/MuscleMimic .venv/bin/python scripts/env_audit.py \
        --checkpoint artifacts/checkpoints/checkpoint_12500 \
        --motion_path KIT/314/walking_medium09_poses \
        --out reports/env_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from _env_builder import build_eval_bundle, env_data, env_model


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tree_manifest(root: Path) -> dict:
    """Size + sha256 for every regular file under ``root`` (orbax checkpoints)."""
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".cache" not in p.parts:
            files[str(p.relative_to(root))] = {"size": p.stat().st_size, "sha256": sha256_file(p)}
    return files


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/checkpoints/checkpoint_12500")
    ap.add_argument("--motion_path", default="KIT/314/walking_medium09_poses")
    ap.add_argument("--out", default="reports/env_audit.json")
    args = ap.parse_args()

    bundle = build_eval_bundle(
        checkpoint_path=args.checkpoint,
        motion_path=args.motion_path,
        start_step=0,
    )
    env = bundle.env
    model = env_model(env)
    data = env_data(env)

    from importlib.metadata import version

    import musclemimic_models

    motion_file = None
    if env.th is not None and hasattr(env.th, "traj_path"):
        motion_file = Path(env.th.traj_path)
    if motion_file is None:
        for p in Path.home().glob(f".musclemimic/caches/AMASS/MyoFullBody/gmr/{args.motion_path}.npz"):
            motion_file = p
            break

    obs = env.reset()
    data = env_data(env)
    ref = env.th.get_current_traj_data(env._additional_carry, np)

    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    site_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]
    sensor_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)]

    foot_geoms = {}
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not gname:
            continue
        bname = body_names[model.geom_bodyid[gid]] or ""
        if any(k in bname.lower() for k in ("calcn", "toes", "talus")):
            foot_geoms[gname] = {"body": bname, "geom_id": gid}

    params = bundle.agent_state.train_state.params
    param_shapes = {}

    def _walk(prefix, tree):
        if isinstance(tree, dict):
            for k, v in tree.items():
                _walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif hasattr(tree, "shape"):
            # Only record first/last layer to keep the report small.
            if any(t in prefix for t in ("Dense_0", "Dense_4", "log_std", "Output")):
                param_shapes[prefix] = list(tree.shape)

    _walk("", params)

    th = env.th
    tlen = int(th.len_trajectory(0))
    info = {
        "host": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os_cpu_count(),
        },
        "versions": {
            "mujoco": mujoco.__version__,
            "musclemimic_models": version("musclemimic-models"),
            "musclemimic_models_path": str(Path(musclemimic_models.__file__).parent),
            "musclemimic_src_commit": git_commit(Path("external/MuscleMimic")),
            "musclemimic_models_src_commit": git_commit(Path("external/musclemimic_models")),
            "numpy": np.__version__,
        },
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "target": str(Path(args.checkpoint).resolve()) + " -> " + str(Path(args.checkpoint).resolve().parent / "mm-10m-2"),
            "metadata": bundle.metadata.__dict__ if hasattr(bundle.metadata, "__dict__") else str(bundle.metadata),
            "files": tree_manifest(Path(args.checkpoint).resolve()),
        },
        "motion": {
            "requested": args.motion_path,
            "file": str(motion_file) if motion_file else None,
            "sha256": sha256_file(motion_file) if motion_file and motion_file.exists() else None,
            "frequency_hz": float(th.traj.info.frequency),
            "n_frames": tlen,
            "n_trajectories": int(th.n_trajectories),
            "fixed_start": [int(th.fixed_start_conf[0]), int(th.fixed_start_conf[1])] if th.fixed_start_conf else None,
            "duration_s": tlen * float(env.dt),
        },
        "model": {
            "name": type(env).__name__,
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "njnt": int(model.njnt),
            "nbody": int(model.nbody),
            "nsite": int(model.nsite),
            "ngeom": int(model.ngeom),
            "nsensor": int(model.nsensor),
            "opt_timestep_s": float(model.opt.timestep),
            "n_substeps": int(getattr(env, "_n_substeps", -1)),
            "control_dt_s": float(env.dt),
            "control_hz": 1.0 / float(env.dt),
            "horizon": int(env.info.horizon),
            "n_intermediate_steps": int(getattr(env, "_n_intermediate_steps", 1)),
            "root_free_joint": env.root_free_joint_xml_name,
            "root_height_healthy_range": list(env.root_height_healthy_range),
            "muscle_actuators": int(np.sum(model.actuator_dyntype == int(mujoco.mjtDyn.mjDYN_MUSCLE))),
            "actuator_ctrlrange_min": float(np.min(model.actuator_ctrlrange)),
            "actuator_ctrlrange_max": float(np.max(model.actuator_ctrlrange)),
            "actuator_forcerange_min": float(np.min(model.actuator_forcerange)),
            "actuator_forcerange_max": float(np.max(model.actuator_forcerange)),
            "first_actuators": actuator_names[:5],
            "last_actuators": actuator_names[-5:],
            "first_joints": joint_names[:6],
            "feet_geoms": foot_geoms,
            "touch_sensors": [s for s in sensor_names if s and "touch" in s.lower()],
        },
        "interface": {
            "observation_dim": int(np.asarray(obs).size),
            "observation_space": list(env.info.observation_space.shape),
            "action_dim": int(np.prod(env.info.action_space.shape)),
            "action_space": list(env.info.action_space.shape),
            "action_low": float(np.min(env.info.action_space.low)),
            "action_high": float(np.max(env.info.action_space.high)),
            "network_param_shapes_sample": param_shapes,
            "n_seeds": int(bundle.config.experiment.n_seeds),
            "len_obs_history": int(getattr(bundle.config.experiment, "len_obs_history", 1)),
            "split_goal": bool(getattr(bundle.config.experiment, "split_goal", False)),
        },
        "termination": {
            "type": type(env._terminal_state_handler).__name__,
            "params": {
                k: v
                for k, v in vars(env._terminal_state_handler).items()
                if k
                in (
                    "mean_site_deviation_threshold",
                    "root_deviation_threshold",
                    "root_orientation_threshold",
                    "root_site",
                    "enable_site_check",
                    "site_deviation_mode",
                )
            },
        },
        "terrain": {
            "type": type(getattr(env, "_terrain", None)).__name__,
            "is_dynamic": bool(getattr(getattr(env, "_terrain", None), "is_dynamic", False)),
        },
        "hidden_forces_at_reset": {
            "qfrc_applied_max": float(np.max(np.abs(data.qfrc_applied))) if data.qfrc_applied.size else 0.0,
            "xfrc_applied_max": float(np.max(np.abs(data.xfrc_applied))) if data.xfrc_applied.size else 0.0,
        },
        "initial_state": {
            "qpos_head": np.round(np.asarray(data.qpos[:7]), 6).tolist(),
            "ref_qpos_head": np.round(np.asarray(ref.qpos[:7]), 6).tolist(),
        },
    }
    digest = hashlib.sha256(
        "\n".join(f"{k}:{v['sha256']}" for k, v in info["checkpoint"]["files"].items()).encode()
    ).hexdigest()
    info["checkpoint"]["digest_sha256"] = digest
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps({k: info[k] for k in ("model", "interface", "motion", "termination")}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


def os_cpu_count() -> int:
    import os

    return os.cpu_count() or 0


if __name__ == "__main__":
    raise SystemExit(main())
