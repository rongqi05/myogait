"""Rigid cane attached to one hand through an ideal fixed grasp.

Model
-----
The cane is a rigid body **welded** to the hand body (``radius_l`` / ``radius_r``,
which is where ``disable_fingers=True`` ends the arm). "Ideal fixed grasp" means
zero relative motion between hand and shaft, so a jointless child body of the
hand realises that assumption exactly: the cane inherits the full 3D motion of
the floating base + arm chain, and the ground reaction at its tip enters the
*same* articulated dynamics as the feet -- no imposed forces, no kinematic rail,
nothing world-anchored.

Trade-off versus a free joint + weld equality constraint: the jointless body has
no extra DOF, so ``nq``/``nv`` stay 89/88 and the reference trajectory, the
reset path (``data.qpos[:] = traj_qpos``) and every existing baseline stay valid.
A free-joint variant would add 7 DOF and require padding every reference; the
rigid grip is also stiffer than any constraint, and the tip contact force is the
same either way. So the jointless attach is used here.

Placement
---------
The cane is oriented **world-vertically at the model's default pose**
(``m.qpos0``) with its grip point at the hand site. Alignment uses model
constants only, never the motion reference, so the very same cane geometry is
used for every reference, seed and condition -- which is what makes the
ipsilateral/contralateral comparison a paired one.

Collision
---------
The cane gets its own contact bit (``contype = conaffinity = 1 << 29``) and the
floor's ``conaffinity`` gets that bit. The cane therefore collides with the floor
and nothing else: no permanent self-contact with the arm it is welded to, and no
cane-leg contacts that would confound the support measurement.

Disabling cane-leg collision is a simplification -- a real cane can strike the
shin. It is off here so that the measured support force contains no self-contact
terms; turn it back on deliberately if cane-body collision becomes the subject.

Length
------
The default 0.90 m is measured, not assumed: at 0.86 m the tip stays ~16 mm above
the floor through a whole 10 s episode on the CMU 136 reference, so the cane
carries nothing; at 0.90 m the tip plants during the arm's downswing and carries
a plausible 20-29 %BW while planted, with penetration under 9 mm.

Usage (set knobs, then patch before any env is built)::

    import _cane
    _cane.configure(side="right", length=0.90)
    _cane.patch()
"""

from __future__ import annotations

import numpy as np

CANE_CONTYPE_BIT = 1 << 29

CANE = {
    "side": "right",           # "left" | "right" -- which hand holds it
    "length": 0.90,            # grip point -> lowest tip point, metres
    "shaft_radius": 0.011,
    "tip_radius": 0.018,
    "grip_offset_z": 0.02,     # shaft starts this far below the grip point
    "shaft_mass": 0.35,
    "tip_mass": 0.10,
    "tip_friction": (1.0, 0.02, 0.001),
}
_PATCHED = False


def configure(**kwargs) -> None:
    unknown = set(kwargs) - set(CANE)
    if unknown:
        raise KeyError(f"unknown cane knob(s): {sorted(unknown)}")
    CANE.update(kwargs)


def names() -> tuple[str, str, str]:
    """(cane body, hand body, hand site) for the configured side."""
    side = CANE["side"]
    suffix = "l" if side == "left" else "r"
    return f"cane_{side}", f"radius_{suffix}", f"{side}_hand_mimic"


def _quat_conj_normalized(q: np.ndarray) -> np.ndarray:
    out = np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)
    n = np.linalg.norm(out)
    return out / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def add_cane_to_spec(spec) -> bool:
    """Insert the cane into ``spec``. Returns True when it was added."""
    import mujoco

    body_name, hand_body_name, hand_site_name = names()
    if spec.body(body_name) is not None:
        return False

    hand_body = spec.body(hand_body_name)
    if hand_body is None:
        raise RuntimeError(f"hand body {hand_body_name!r} not found in the spec")

    # Compile a *copy* to read the hand pose at the default state; the caller's
    # spec is left untouched by this probe.
    bare = spec.copy().compile()
    data = mujoco.MjData(bare)
    mujoco.mj_resetData(bare, data)
    mujoco.mj_forward(bare, data)

    bid = mujoco.mj_name2id(bare, mujoco.mjtObj.mjOBJ_BODY, hand_body_name)
    sid = mujoco.mj_name2id(bare, mujoco.mjtObj.mjOBJ_SITE, hand_site_name)
    if bid < 0 or sid < 0:
        raise RuntimeError(f"cannot resolve {hand_body_name!r}/{hand_site_name!r} in the compiled model")

    body_w = np.array(data.xpos[bid], dtype=float)
    body_q = np.array(data.xquat[bid], dtype=float)
    site_w = np.array(data.site_xpos[sid], dtype=float)

    rot = _quat_to_mat(body_q)
    local_pos = rot.T @ (site_w - body_w)      # grip point in the hand frame
    local_quat = _quat_conj_normalized(body_q)  # cancels hand rotation -> world-aligned

    length = float(CANE["length"])
    tip_r = float(CANE["tip_radius"])
    shaft_r = float(CANE["shaft_radius"])
    grip = float(CANE["grip_offset_z"])

    # Shaft runs from just below the grip down to the *centre* of the tip sphere,
    # so the sphere alone defines the lowest point of the cane.
    shaft_top_z = -grip
    tip_center_z = -(length - tip_r)
    if shaft_top_z <= tip_center_z:
        raise ValueError(
            f"cane length {length} m is too short for grip_offset_z={grip}: "
            "the shaft would collapse"
        )
    shaft_half = (shaft_top_z - tip_center_z) / 2.0
    shaft_center_z = (shaft_top_z + tip_center_z) / 2.0

    cane_body = hand_body.add_body(name=body_name, pos=local_pos, quat=local_quat)
    cane_body.add_geom(
        name=f"{body_name}_shaft",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[shaft_r, shaft_half, 0.0],
        pos=[0.0, 0.0, shaft_center_z],
        mass=float(CANE["shaft_mass"]),
        contype=CANE_CONTYPE_BIT,
        conaffinity=CANE_CONTYPE_BIT,
        rgba=[0.55, 0.55, 0.60, 1.0],
    )
    cane_body.add_geom(
        name=f"{body_name}_tip",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[tip_r, 0.0, 0.0],
        pos=[0.0, 0.0, tip_center_z],
        mass=float(CANE["tip_mass"]),
        contype=CANE_CONTYPE_BIT,
        conaffinity=CANE_CONTYPE_BIT,
        condim=3,
        priority=1,
        rgba=[0.15, 0.15, 0.15, 1.0],
    )
    cane_body.geoms[-1].friction = [float(v) for v in CANE["tip_friction"]]

    floor = spec.geom("floor")
    if floor is None:
        raise RuntimeError("no 'floor' geom in the spec; the cane needs a ground plane")
    floor.conaffinity = int(floor.conaffinity) | CANE_CONTYPE_BIT
    return True


def patch() -> None:
    """Patch ``Mujoco.load_mujoco`` so every compiled env model carries the cane."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    import sys

    import mujoco
    from loco_mujoco.core.mujoco_base import Mujoco

    orig = Mujoco.load_mujoco

    def _patched(self, xml_file, model_option_conf=None):
        if isinstance(xml_file, mujoco.MjSpec):
            spec = xml_file
        elif isinstance(xml_file, str):
            spec = mujoco.MjSpec.from_file(xml_file)
        else:
            return orig(self, xml_file, model_option_conf)
        if add_cane_to_spec(spec):
            body_name, hand_body_name, _ = names()
            print(
                f"[cane] attached {body_name} to {hand_body_name} (length={CANE['length']} m)",
                file=sys.stderr,
                flush=True,
            )
            xml_file = spec
        return orig(self, xml_file, model_option_conf)

    Mujoco.load_mujoco = _patched


def cane_body_id(model) -> int:
    import mujoco

    body_name, _, _ = names()
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)


def cane_geom_ids(model) -> set[int]:
    import mujoco

    out = set()
    prefix = f"{names()[0]}_"
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if name.startswith(prefix):
            out.add(i)
    return out


def cane_tip_geom_ids(model) -> list[int]:
    import mujoco

    out = []
    prefix = f"{names()[0]}_"
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if name.startswith(prefix) and name.endswith("_tip"):
            out.append(i)
    return out


def tip_lowest_z(model, data, tip_ids=None) -> float:
    """World z of the lowest point of the cane tip (negative = penetration)."""
    if tip_ids is None:
        tip_ids = cane_tip_geom_ids(model)
    if not tip_ids:
        return float("nan")
    zs = [float(data.geom_xpos[g][2]) - float(model.geom_size[g][0]) for g in tip_ids]
    return min(zs)


def cane_grf(model, data, geom_ids=None) -> dict:
    """Ground reaction at the cane tip, resolved in world coordinates.

    ``normal``   summed contact normal force magnitude (N)
    ``vertical`` its world-z component (N); the floor pushes the cane up
    ``contact``  whether any contact exists at all
    """
    import mujoco

    if geom_ids is None:
        geom_ids = cane_geom_ids(model)
    fbuf = np.zeros(6, dtype=float)
    normal = 0.0
    vertical = 0.0
    ncon = 0
    pts = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 not in geom_ids and g2 not in geom_ids:
            continue
        mujoco.mj_contactForce(model, data, i, fbuf)
        n_world = np.asarray(c.frame, dtype=float).reshape(3, 3)[0]
        f_n = abs(float(fbuf[0]))
        normal += f_n
        vertical += f_n * abs(float(n_world[2]))
        ncon += 1
        pts.append(np.array(c.pos, dtype=float))
    return {
        "normal": normal,
        "vertical": vertical,
        "ncon": ncon,
        "contact": ncon > 0,
        "point": (np.mean(pts, axis=0) if pts else np.zeros(3)),
        "tip_lowest_z": tip_lowest_z(model, data),
    }
