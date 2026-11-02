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
    horizon: int | None = None,
) -> EvalBundle:
    _resolve_hf_endpoint()

    config, agent_state, metadata = load_checkpoint(checkpoint_path)
    OmegaConf.set_struct(config, False)
    config.experiment.env_params = config.experiment.env_params.copy()

    # Evaluation-time overrides that the official entrypoint performs.
    config.experiment.env_params["headless"] = True
    if horizon is not None:
        # The checkpoint was trained with horizon=1000 control steps, which is
        # exactly 10.0 s at dt=0.01. Raising it lets the acceptance criterion be
        # our own 10 s clock instead of the environment's training truncation.
        config.experiment.env_params["horizon"] = int(horizon)
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


_MUSCLE_GROUPS_CACHE: dict | None = None


def muscle_groups(model) -> dict:
    """Classify all actuators into anatomical groups using the installed
    ``musclemimic_models`` XML (source of truth for the model version).

    MyoFullBody (model v1.0.5, ``disable_fingers=True``) exposes 354 actuators:

    * 210 torso/spine muscles -- ``myotorso_assets.xml`` (105 ``_l`` + 105 ``_r``)
    *  80 leg muscles         -- ``myolegs_muscle.xml``  ( 40 ``_l`` +  40 ``_r``)
    *  64 arm muscles         -- ``myoarm_bimanual_assets.xml``:
        32 right = 24 ``<general>`` upper-arm + 8 ``<muscle>`` forearm, no suffix;
        32 left  = the same with a ``_left`` suffix.

    Note the asymmetric naming convention: legs/torso use ``_l``/``_r`` while the
    arms use ``_left`` for the left side and *no suffix* for the right side.  A
    name-prefix heuristic would therefore miss the 32 left-arm muscles.

    Returns ``{group: {"indices": [...], "names": [...]}}`` for the six groups
    ``left_upper``, ``right_upper``, ``left_lower``, ``right_lower``,
    ``left_trunk``, ``right_trunk``.
    """
    global _MUSCLE_GROUPS_CACHE
    if _MUSCLE_GROUPS_CACHE is not None:
        return _MUSCLE_GROUPS_CACHE

    import pathlib
    import re

    import mujoco

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]

    try:
        import musclemimic_models  # type: ignore

        model_dir = pathlib.Path(musclemimic_models.__file__).parent / "model"
    except Exception:  # pragma: no cover - fallback to name heuristics
        model_dir = None

    def _tag_names(path: pathlib.Path) -> set[str]:
        text = path.read_text(errors="ignore")
        found = set()
        for pat in (r"<general\b[^>]*>", r"<muscle\b[^>]*>"):
            for m in re.finditer(pat, text):
                nm = re.search(r'name="([^"]+)"', m.group(0))
                if nm:
                    found.add(nm.group(1))
        return found

    arm_tags = _tag_names(model_dir / "arm" / "assets" / "myoarm_bimanual_assets.xml")
    leg_tags = _tag_names(model_dir / "leg" / "assets" / "myolegs_muscle.xml")
    torso_tags = _tag_names(model_dir / "torso" / "assets" / "myotorso_assets.xml")

    groups = {
        key: {"indices": [], "names": []}
        for key in ("left_upper", "right_upper", "left_lower", "right_lower", "left_trunk", "right_trunk")
    }
    for i, name in enumerate(names):
        if name in arm_tags:
            key = "left_upper" if name.endswith("_left") else "right_upper"
        elif name in leg_tags:
            key = "left_lower" if name.endswith("_l") else "right_lower"
        elif name in torso_tags:
            key = "left_trunk" if name.endswith("_l") else "right_trunk"
        else:  # pragma: no cover - should not happen for a matched model version
            raise ValueError(f"unclassified actuator {i}: {name}")
        groups[key]["indices"].append(i)
        groups[key]["names"].append(name)

    _MUSCLE_GROUPS_CACHE = groups
    return groups


def scale_muscle_groups(model, **scales: float) -> dict:
    """Scale F0 of specific anatomical muscle groups (see :func:`muscle_groups`).

    ``scales`` maps a group name to a multiplier, e.g.
    ``scale_muscle_groups(model, left_lower=0.7, left_upper=0.7)``.  Groups not
    passed keep their original F0.  Each call works on a fresh model; applying a
    region scale and the legacy ``scale_muscle_f0`` to the same model would
    compose multiplicatively, so callers should use one mechanism per run.
    """
    import mujoco

    groups = muscle_groups(model)
    info: dict = {}
    for key, factor in scales.items():
        if factor == 1.0:
            continue
        g = groups[key]
        idx = np.asarray(g["indices"], dtype=int)
        original = model.actuator_gainprm[idx, 0].copy()
        model.actuator_gainprm[idx, 0] *= factor
        info[key] = {
            "scale": float(factor),
            "count": int(len(idx)),
            "names": list(g["names"]),
            "indices": [int(i) for i in idx],
            "f0_original": [float(v) for v in original],
            "f0_scaled": [float(v) for v in model.actuator_gainprm[idx, 0]],
        }
    return {"groups_scaled": info}


def scale_muscle_f0(model, left: float = 1.0, right: float = 1.0) -> dict:
    """Legacy whole-side F0 scaling.

    Scales every actuator whose name ends in ``_l`` / ``_r``.  Because of the
    asymmetric naming convention (see :func:`muscle_groups`) this touches the
    **torso + leg** muscles of a side (105 + 40 = 145) and does **not** touch the
    32 arm muscles of either side (left-arm ``_left`` and right-arm no-suffix).

    ``actuator_gainprm[i, 0]`` is the peak isometric force F0 of the MuJoCo
    muscle actuator (``gaintype="muscle"``), so scaling it is the standard way to
    model weakness.  Must be called after ``env_model`` and before the first
    ``env.reset()`` so every subsequent ``mj_forward`` uses the scaled gains.
    """
    import mujoco

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    counts = {"left": 0, "right": 0}
    for i, name in enumerate(names):
        if name.endswith("_l"):
            model.actuator_gainprm[i, 0] *= left
            counts["left"] += 1
        elif name.endswith("_r"):
            model.actuator_gainprm[i, 0] *= right
            counts["right"] += 1
    return {
        "f0_left": left,
        "f0_right": right,
        "muscles_scaled_left": counts["left"],
        "muscles_scaled_right": counts["right"],
        "note": "legacy: scales '_l'/'_r' = torso+leg only; arm muscles ('_left'/no-suffix) untouched",
    }


def quat_to_tilt_deg(quat_wxyz: np.ndarray) -> float:
    """Angle between the body's +z axis and world +z, in degrees."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=float)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0 or not np.isfinite(n):
        return float("nan")
    w, x, y, z = w / n, x / n, y / n, z / n
    z_axis_z = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(z_axis_z, -1.0, 1.0))))
