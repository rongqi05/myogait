"""Build a MyoFullBody evaluation environment the same way `fullbody/eval.py` does.

The official entrypoint does not expose its environment construction as a
function, so this module mirrors the relevant calls (checkpoint load, temporal
parameters, terminal-state defaults, task factory, agent alignment) without
modifying the upstream source tree.

Upstream code is imported from ``external/MuscleMimic``; set ``PYTHONPATH``
accordingly (see README).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults
from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.runner.eval_utils import (
    align_agent_state,
    apply_temporal_params,
    apply_trajectory_selection,
    load_checkpoint,
)


@dataclass
class EvalBundle:
    env: Any
    agent_conf: Any
    agent_state: Any
    config: Any
    metadata: Any


def _resolve_hf_endpoint() -> None:
    # huggingface.co is not reachable from this host; the mirror serves the same
    # public repos. Only used when an environment variable is not already set.
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def build_eval_bundle(
    checkpoint_path: str,
    motion_path: str | None = None,
    motion_group: str | None = None,
    start_step: int = 0,
    traj_index: int = 0,
    use_mujoco: bool = True,
    strict_termination: bool = False,
    terminal_state_type: str | None = None,
    terminal_state_params: dict[str, float] | None = None,
) -> EvalBundle:
    _resolve_hf_endpoint()

    config, agent_state, metadata = load_checkpoint(checkpoint_path)
    OmegaConf.set_struct(config, False)
    config.experiment.env_params = config.experiment.env_params.copy()

    # Evaluation-time overrides that the official entrypoint performs.
    config.experiment.env_params["headless"] = True
    apply_trajectory_selection(config, traj_index, start_step)
    config.experiment.env_params.th_params["start_from_random_step"] = False
    apply_temporal_params(config)

    if motion_path is not None:
        config.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path = [motion_path]
        config.experiment.task_factory.params.amass_dataset_conf.dataset_group = None
    elif motion_group is not None:
        config.experiment.task_factory.params.amass_dataset_conf.dataset_group = motion_group

    play_env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    apply_eval_terminal_defaults(play_env_params, config, strict_termination)
    if terminal_state_type is not None:
        play_env_params["terminal_state_type"] = terminal_state_type
    if terminal_state_params is not None:
        merged = dict(play_env_params.get("terminal_state_params") or {})
        merged.update(terminal_state_params)
        play_env_params["terminal_state_params"] = merged

    if use_mujoco and "Mjx" in play_env_params.get("env_name", ""):
        play_env_params["env_name"] = play_env_params["env_name"].replace("Mjx", "")

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    env = factory.make(**{**play_env_params, **task_params})

    agent_conf = PPOJax.init_agent_conf(env, config)
    agent_state = align_agent_state(agent_state, agent_conf)
    return EvalBundle(env=env, agent_conf=agent_conf, agent_state=agent_state, config=config, metadata=metadata)


def env_data(env: Any):
    """Return the concrete MuJoCo MjData regardless of wrapper depth."""
    cur = env
    while cur is not None:
        if hasattr(cur, "data") and hasattr(getattr(cur, "data"), "qpos"):
            return cur.data
        cur = getattr(cur, "env", None)
    raise RuntimeError("could not locate MjData on environment")


def env_model(env: Any):
    cur = env
    while cur is not None:
        if hasattr(cur, "model") and hasattr(getattr(cur, "model"), "nq"):
            return cur.model
        cur = getattr(cur, "env", None)
    raise RuntimeError("could not locate MjModel on environment")


def quat_to_tilt_deg(quat_wxyz: np.ndarray) -> float:
    """Angle between the body's +z axis and world +z, in degrees."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=float)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0 or not np.isfinite(n):
        return float("nan")
    w, x, y, z = w / n, x / n, y / n, z / n
    z_axis_z = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(z_axis_z, -1.0, 1.0))))
