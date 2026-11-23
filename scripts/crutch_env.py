"""Mount crutch_robot.xml into the MyoFullBody MuJoCo scene.

One MjModel / MjData holds the human and the crutch.  The human part keeps its
original DOF, actuator and sensor layout; the crutch is appended after it, and
everything the policy sees is filtered back to the human items by name prefix.

Integration points (all monkeypatches, upstream tree untouched):

1. ``Mujoco.load_mujoco``      attach the crutch spec to the human spec before compile
2. ``MyoFullBody._get_observation_specification`` / ``_get_action_specification``
                               drop ``crutch/`` items so obs/action dims are unchanged
3. ``DefaultInitialStateHandler.set_qpos/set_qvel``
                               write only the human part; the reference has 89 DOF,
                               the merged model 92
4. ``compute_trajectory_kinematic_caches``
                               zero-pad the reference to the merged nq/nv so the
                               kinematic cache can still be built (the crutch columns
                               never influence human site positions)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

PREFIX = "crutch/"
"""Every name carried in from crutch_robot.xml gets this prefix by ``MjSpec.attach``."""


@dataclass
class CrutchMount:
    xml: str = "crutch_robot.xml"
    rod_length: float = 1.00
    hand_body: dict = None
    hand_site: dict = None

    def __post_init__(self):
        self.hand_body = self.hand_body or {"left": "radius_l", "right": "radius_r"}
        self.hand_site = self.hand_site or {"left": "left_hand_mimic", "right": "right_hand_mimic"}


GRIP_SOLREF = [0.02, 1.0]
GRIP_SOLIMP = [0.9, 0.95, 0.001, 0.5, 2.0]


def build_child(mount: CrutchMount) -> mujoco.MjSpec:
    """Crutch spec sized for the human, with the demo hand stubs removed.

    Deleting a stub body also deletes the connect equality that referenced it, so
    both grips are re-created against the real hands after attach (see ``install``).
    """
    child = mujoco.MjSpec.from_file(mount.xml)
    L = float(mount.rod_length)
    child.geom("rod_shaft").size = [0.016, L / 2.0, 0.0]
    child.geom("rod_shaft").pos = [0.0, 0.0, L / 2.0]
    child.body("handle").pos = [0.0, 0.0, L]
    for stub in ("hand_stub_l", "hand_stub_r"):
        if child.body(stub) is not None:
            child.delete(child.body(stub))
    # the human scene already has a floor; a second coplanar plane z-fights
    if child.geom("ground") is not None:
        child.delete(child.geom("ground"))
    return child


_ORIG: dict[str, Any] = {}
_RENDER: tuple[int, int] | None = None


def install(mount: CrutchMount | None = None, render_size: tuple[int, int] | None = None):
    """Install the env patches once. ``mount=None`` patches only the render size.

    Returns the crutch spec that will be attached, or None when no mount is given.
    """
    global _RENDER
    _RENDER = render_size
    child = build_child(mount) if mount is not None else None
    if _ORIG:
        _ORIG["child"] = child
        _ORIG["mount"] = mount
        return child

    from loco_mujoco.core.initial_state_handler.default import DefaultInitialStateHandler
    from loco_mujoco.core.mujoco_base import Mujoco
    from loco_mujoco.trajectory import handler as traj_handler
    from musclemimic.environments.humanoids.myofullbody import MyoFullBody

    _ORIG.update(
        mount=mount,
        load_mujoco=Mujoco.load_mujoco,
        obs=MyoFullBody._get_observation_specification,
        act=MyoFullBody._get_action_specification,
        set_qpos=DefaultInitialStateHandler.set_qpos,
        set_qvel=DefaultInitialStateHandler.set_qvel,
        caches=traj_handler.compute_trajectory_kinematic_caches,
        child=child,
    )

    def load_mujoco(self, xml_file, model_option_conf=None):
        if isinstance(xml_file, mujoco.MjSpec):
            spec = xml_file
        elif isinstance(xml_file, str):
            spec = mujoco.MjSpec.from_file(xml_file)
        else:
            return _ORIG["load_mujoco"](self, xml_file, model_option_conf)
        if _RENDER is not None:
            spec.visual.global_.offwidth = int(_RENDER[0])
            spec.visual.global_.offheight = int(_RENDER[1])
        if _ORIG.get("child") is not None and not any(b.name == f"{PREFIX}chassis" for b in spec.bodies):
            frame = spec.worldbody.add_frame(pos=[0, 0, 0], quat=[1, 0, 0, 0], name="crutch_mount")
            spec.attach(_ORIG["child"], frame=frame, prefix=PREFIX)
            for side in ("left", "right"):
                spec.add_equality(
                    name=f"{PREFIX}grip_{side[0]}",
                    type=mujoco.mjtEq.mjEQ_CONNECT,
                    active=False,
                    name1=f"{PREFIX}handle",
                    name2=_ORIG["mount"].hand_body[side],
                    objtype=mujoco.mjtObj.mjOBJ_BODY,
                    solref=list(GRIP_SOLREF),
                    solimp=list(GRIP_SOLIMP),
                )
        return _ORIG["load_mujoco"](self, spec, model_option_conf)

    def obs_spec(self, spec):
        out = []
        for o in _ORIG["obs"](self, spec):
            single = getattr(o, "_xml_name", None)
            if isinstance(single, str) and single.startswith(PREFIX):
                continue
            many = getattr(o, "_xml_names", None)
            if isinstance(many, list):
                o._xml_names = [n for n in many if not str(n).startswith(PREFIX)]
            out.append(o)
        return out

    def act_spec(self, spec):
        return [n for n in _ORIG["act"](self, spec) if not n.startswith(PREFIX)]

    def set_qpos(qpos, data, backend):
        n = len(qpos)
        if backend is np:
            data.qpos[:n] = qpos
            return data
        return data.replace(qpos=data.qpos.at[:n].set(qpos))

    def set_qvel(qvel, data, backend):
        n = len(qvel)
        if backend is np:
            data.qvel[:n] = qvel
            return data
        return data.replace(qvel=data.qvel.at[:n].set(qvel))

    def caches(traj_data, model, cache_type, site_names=None, backend=None):
        qpos = np.asarray(traj_data.qpos)
        if qpos.shape[1] < model.nq:
            pad = model.nq - qpos.shape[1]
            traj_data = traj_data.replace(
                qpos=np.concatenate([qpos, np.zeros((len(qpos), pad), qpos.dtype)], axis=1),
                qvel=np.concatenate(
                    [np.asarray(traj_data.qvel), np.zeros((len(qpos), model.nv - np.asarray(traj_data.qvel).shape[1]),
                                                          np.asarray(traj_data.qvel).dtype)], axis=1),
            )
        if backend is None:
            return _ORIG["caches"](traj_data, model, cache_type, site_names)
        return _ORIG["caches"](traj_data, model, cache_type, site_names, backend)

    Mujoco.load_mujoco = load_mujoco
    MyoFullBody._get_observation_specification = obs_spec
    MyoFullBody._get_action_specification = act_spec
    DefaultInitialStateHandler.set_qpos = staticmethod(set_qpos)
    DefaultInitialStateHandler.set_qvel = staticmethod(set_qvel)
    traj_handler.compute_trajectory_kinematic_caches = caches
    return child


def layout(model) -> dict:
    """Human/crutch partitioning of the merged model, by name."""
    crutch_joints = [j for j in range(model.njnt)
                     if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(PREFIX)]
    crutch_acts = [a for a in range(model.nu)
                   if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "").startswith(PREFIX)]
    human_q = [int(model.jnt_qposadr[j]) for j in range(model.njnt)
               if not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(PREFIX)]
    return {
        "nq": model.nq, "nv": model.nv, "nu": model.nu,
        "n_human_joints": model.njnt - len(crutch_joints),
        "n_human_actuators": model.nu - len(crutch_acts),
        "crutch_joints": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in crutch_joints],
        "crutch_actuators": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in crutch_acts],
        "crutch_act_ids": crutch_acts,
        "crutch_joint_ids": crutch_joints,
        "first_crutch_qpos": min(int(model.jnt_qposadr[j]) for j in crutch_joints),
        "grip_l": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{PREFIX}grip_l"),
        "grip_r": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{PREFIX}grip_r"),
    }


def reach(model, data, side: str) -> dict:
    """Arm posture at the current state, using joint anchors for shoulder/elbow."""
    def anchor(name):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return None if j < 0 else np.array(data.xanchor[j], copy=True)

    sh = anchor(f"shoulder_elv_{side[0]}")
    el = anchor(f"elbow_flex_{side[0]}")
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_hand_mimic")
    hand = np.array(data.site_xpos[site], copy=True) if site >= 0 else None
    out = {"shoulder": sh, "elbow": el, "hand_site": hand}
    if sh is not None and el is not None and hand is not None:
        upper = float(np.linalg.norm(el - sh))
        fore = float(np.linalg.norm(hand - el))
        span = upper + fore
        d = float(np.linalg.norm(hand - sh))
        cos_elbow = (upper ** 2 + fore ** 2 - d ** 2) / (2.0 * upper * fore) if upper * fore > 0 else 1.0
        out.update(
            upper_arm=round(upper, 4),
            forearm_to_hand=round(fore, 4),
            arm_span=round(span, 4),
            hand_to_shoulder=round(d, 4),
            reach_ratio=round(d / span if span > 0 else float("nan"), 4),
            elbow_flexion_deg=round(float(np.degrees(np.pi - np.arccos(np.clip(cos_elbow, -1, 1)))), 2),
        )
    return out
