#!/usr/bin/env python3
"""Build a >=10 s straight walking reference from short retargeted clips.

The retargeted dataset available locally contains no continuous straight walk of
10 s or more (see reports/motion_straightness.json). This script is the
documented, reviewable workaround: it extracts declared straight windows from
several clips, rotates each window so its walking direction is +x, joins them
with a short crossfade, and recomputes every derived MuJoCo array by forward
kinematics so the artefact is a valid trajectory cache.

What the stitching does NOT fix, and what the reports must state:

  * the joint/velocity blend over the crossfade is an interpolation, not a
    physically consistent transition; foot contact can be momentarily wrong in
    those frames;
  * the seam introduces a small root-velocity discontinuity;
  * the segments still come from a few seconds of real motion, so the stitched
    reference contains less gait variety than a genuine 10 s recording.

Usage:
    PYTHONPATH=external/MuscleMimic .venv/bin/python scripts/prepare_straight_walk.py \
        --segment KIT/3/walk_6m_straight_line04_poses:0.30:7.00 \
        --segment KIT/3/walk_6m_straight_line06_poses:0.25:7.05 \
        --out-name straight_stitched_10s \
        --out ~/.musclemimic/caches/AMASS/MyoFullBody/gmr/KIT/3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def yaw_quat(deg: float):
    h = np.radians(deg) / 2.0
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)])


def rot_z(v, deg: float):
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def default_motion_root() -> Path:
    return Path.home() / ".musclemimic/caches/AMASS/MyoFullBody/gmr"


def load_clip(path: Path, t0: float, t1: float, pelvis_site: str):
    d = np.load(path, allow_pickle=True)
    freq = float(d["frequency"])
    names = [str(n) for n in d["site_names"]]
    i0 = int(round(t0 * freq))
    i1 = min(len(d["qpos"]), int(round(t1 * freq)))
    if i1 - i0 < 2:
        raise ValueError(f"empty window {t0}-{t1} in {path}")
    qpos = np.array(d["qpos"][i0:i1], dtype=np.float64)
    qvel = np.array(d["qvel"][i0:i1], dtype=np.float64)
    pel = np.array(d["site_xpos"][i0:i1, names.index(pelvis_site)], dtype=np.float64)

    # heading of this window, from the pelvis trajectory
    dxy = pel[-1, :2] - pel[0, :2]
    if np.linalg.norm(dxy) < 0.2:
        dxy = pel[len(pel) // 2, :2] - pel[0, :2]
    heading = float(np.degrees(np.arctan2(dxy[1], dxy[0])))
    # rotate so that heading becomes +x
    yaw = -heading

    for f in range(len(qpos)):
        qpos[f, 0:2] = rot_z(qpos[f, 0:2], yaw)
        qpos[f, 3:7] = quat_mul(yaw_quat(yaw), qpos[f, 3:7])
        qvel[f, 0:2] = rot_z(qvel[f, 0:2], yaw)
        qvel[f, 3:5] = rot_z(qvel[f, 3:5], yaw)
    qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    return {
        "qpos": qpos,
        "qvel": qvel,
        "freq": freq,
        "source": str(path),
        "window_s": [float(t0), float(t1)],
        "heading_before_deg": heading,
    }


def slerp_quat(q0: np.ndarray, q1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Spherical interpolation of unit quaternions, w in [0, 1].

    Quaternions are a double cover: q and -q are the same rotation, so two
    consecutive poses can sit at dot = -1 and still be adjacent. Interpolating
    that pair linearly drives the blend through magnitude ~0.08, and after
    renormalisation its *direction* swings through ~180 deg, i.e. a rotational
    snap of hundreds of degrees per frame. The hemisphere flip below is what
    makes the path short instead of long.
    """
    d = np.einsum("ij,ij->i", q0, q1)
    q1 = np.where(d[:, None] < 0.0, -q1, q1)
    d = np.abs(d)
    # near-parallel: lerp is numerically better conditioned and the error is tiny
    close = d > 0.9995
    out = np.empty_like(q0)
    if close.any():
        out[close] = (1.0 - w[close, None]) * q0[close] + w[close, None] * q1[close]
    far = ~close
    if far.any():
        theta = np.arccos(np.clip(d[far], -1.0, 1.0))
        s = np.sin(theta)
        out[far] = (
            np.sin((1.0 - w[far]) * theta)[:, None] / s[:, None] * q0[far]
            + np.sin(w[far] * theta)[:, None] / s[:, None] * q1[far]
        )
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def stitch(segments, crossfade_s: float, freq: float):
    """Join segments end to end, aligning root xy, crossfading the seam.

    The root quaternion is crossfaded with slerp: the two segments may arrive
    antipodal, and a linear blend would rotate the pelvis by ~290 deg inside
    0.2 s even though the two poses are only a few degrees apart.
    """
    n_fade = max(0, int(round(crossfade_s * freq)))
    qpos = segments[0]["qpos"].copy()
    qvel = segments[0]["qvel"].copy()
    seams = []

    for seg in segments[1:]:
        a_q, a_v = qpos, qvel
        b_q, b_v = seg["qpos"].copy(), seg["qvel"].copy()

        # place the new segment so its first root xy continues the previous end
        off = a_q[-1, 0:2] - b_q[0, 0:2]
        b_q[:, 0:2] += off

        if n_fade == 0:
            qpos = np.concatenate([a_q, b_q], axis=0)
            qvel = np.concatenate([a_v, b_v], axis=0)
            seams.append({"index": int(len(a_q)), "crossfade_frames": 0})
            continue

        tail = a_q[-n_fade:]
        head = b_q[:n_fade]
        w = np.linspace(0.0, 1.0, n_fade)
        blended = (1.0 - w[:, None]) * tail + w[:, None] * head
        # root xy: linear bridge between the two endpoints (position must stay smooth)
        blended[:, 0:2] = (1.0 - w[:, None]) * tail[:, 0:2] + w[:, None] * head[:, 0:2]
        # root orientation: slerp, hemisphere aligned
        blended[:, 3:7] = slerp_quat(tail[:, 3:7], head[:, 3:7], w)

        qpos = np.concatenate([a_q[:-n_fade], blended, b_q[n_fade:]], axis=0)
        qvel = np.concatenate(
            [a_v[:-n_fade], (1.0 - w[:, None]) * a_v[-n_fade:] + w[:, None] * b_v[:n_fade], b_v[n_fade:]], axis=0
        )
        dot = float(np.dot(tail[-1, 3:7], head[0, 3:7]) / (np.linalg.norm(tail[-1, 3:7]) * np.linalg.norm(head[0, 3:7])))
        seams.append(
            {
                "index": int(len(a_q) - n_fade),
                "crossfade_frames": n_fade,
                "root_xy_jump_m": float(np.linalg.norm(head[0, 0:2] - tail[-1, 0:2])),
                "root_quat_dot": dot,
                "root_quat_antipodal": bool(dot < 0.0),
                "root_quat_angle_deg": float(np.degrees(2 * np.arccos(min(1.0, abs(dot))))),
            }
        )
    return qpos, qvel, seams


def recompute_derived(model, qpos: np.ndarray, qvel: np.ndarray, site_names, body_names):
    data = mujoco.MjData(model)
    n = len(qpos)
    xpos = np.zeros((n, model.nbody, 3), dtype=np.float32)
    xquat = np.zeros((n, model.nbody, 4), dtype=np.float32)
    cvel = np.zeros((n, model.nbody, 6), dtype=np.float32)
    subtree_com = np.zeros((n, model.nbody, 3), dtype=np.float32)
    site_xpos = np.zeros((n, len(site_names), 3), dtype=np.float32)
    site_xmat = np.zeros((n, len(site_names), 9), dtype=np.float32)

    mid = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names}
    body_idx = [mid[name] for name in body_names]
    sid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in site_names]

    for i in range(n):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, data)
        xpos[i] = data.xpos[body_idx]
        xquat[i] = data.xquat[body_idx]
        cvel[i] = data.cvel[body_idx]
        subtree_com[i] = data.subtree_com[body_idx]
        site_xpos[i] = data.site_xpos[sid]
        site_xmat[i] = data.site_xmat[sid]
    return {
        "xpos": xpos,
        "xquat": xquat,
        "cvel": cvel,
        "subtree_com": subtree_com,
        "site_xpos": site_xpos,
        "site_xmat": site_xmat,
    }


def build_model(expect_nq: int):
    """Compile the same MyoFullBody variant the env uses (fingers disabled)."""
    from musclemimic.environments.humanoids.myofullbody import MyoFullBody

    env = MyoFullBody(disable_fingers=True, headless=True)
    model = env._model
    if model.nq != expect_nq:
        raise SystemExit(
            f"model nq={model.nq} does not match clip nq={expect_nq}; "
            "the retargeted clip was produced for a different model variant"
        )
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", action="append", required=True, help="CLIP:START_S:END_S (repeatable)")
    ap.add_argument("--motion_root", default=str(default_motion_root()))
    ap.add_argument("--out_name", required=True)
    ap.add_argument("--out", required=True, help="directory for the produced npz")
    ap.add_argument("--crossfade_s", type=float, default=0.20)
    ap.add_argument("--pelvis_site", default="pelvis_mimic")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    motion_root = Path(args.motion_root).expanduser()
    parsed = []
    for spec in args.segment:
        clip, t0, t1 = spec.rsplit(":", 2)
        parsed.append((motion_root / f"{clip}.npz", float(t0), float(t1)))
    missing = [str(p) for p, _, _ in parsed if not p.exists()]
    if missing:
        raise SystemExit(f"missing motion file(s): {missing}")

    segments = [load_clip(p, t0, t1, args.pelvis_site) for p, t0, t1 in parsed]
    freq = segments[0]["freq"]
    if any(abs(s["freq"] - freq) > 1e-6 for s in segments):
        raise SystemExit("segments must share the same sampling frequency")

    qpos, qvel, seams = stitch(segments, args.crossfade_s, freq)

    template = np.load(parsed[0][0], allow_pickle=True)
    site_names = [str(n) for n in template["site_names"]]
    body_names = [str(n) for n in template["body_names"]]

    model = build_model(expect_nq=qpos.shape[1])
    if model.nq != qpos.shape[1] or model.nv != qvel.shape[1]:
        raise SystemExit(f"model mismatch: model nq/nv={model.nq}/{model.nv} vs clip {qpos.shape[1]}/{qvel.shape[1]}")

    derived = recompute_derived(model, qpos, qvel, site_names, body_names)

    manifest = {
        "name": args.out_name,
        "created_by": "scripts/prepare_straight_walk.py",
        "frequency_hz": freq,
        "n_frames": int(len(qpos)),
        "duration_s": float((len(qpos) - 1) / freq),
        "crossfade_s": args.crossfade_s,
        "segments": [
            {"source": s["source"], "window_s": s["window_s"], "heading_before_deg": s["heading_before_deg"]}
            for s in segments
        ],
        "seams": seams,
        "caveats": [
            "crossfade frames blend qpos/qvel; not dynamically consistent",
            "root velocity has a small discontinuity at each seam",
            "gait variety is lower than a genuine 10 s recording",
            "derived arrays recomputed by MuJoCo forward kinematics",
        ],
    }

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}.npz"

    np.savez_compressed(
        out_path,
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        xpos=derived["xpos"],
        xquat=derived["xquat"],
        cvel=derived["cvel"],
        subtree_com=derived["subtree_com"],
        site_xpos=derived["site_xpos"],
        site_xmat=derived["site_xmat"],
        split_points=np.array([0, len(qpos)], dtype=np.int32),
        joint_names=template["joint_names"],
        frequency=np.float64(freq),
        body_names=template["body_names"],
        site_names=template["site_names"],
        metadata=np.array(json.dumps(manifest)),
        njnt=template["njnt"],
        jnt_type=template["jnt_type"],
        nbody=template["nbody"],
        body_rootid=template["body_rootid"],
        body_weldid=template["body_weldid"],
        body_mocapid=template["body_mocapid"],
        body_pos=template["body_pos"],
        body_quat=template["body_quat"],
        body_ipos=template["body_ipos"],
        body_iquat=template["body_iquat"],
        nsite=template["nsite"],
        site_bodyid=template["site_bodyid"],
        site_pos=template["site_pos"],
        site_quat=template["site_quat"],
    )

    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps(manifest, indent=2))

    pel = derived["site_xpos"][:, site_names.index(args.pelvis_site), :]
    xy = pel[:, :2]
    path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    print(f"wrote {out_path}")
    print(f"  frames={len(qpos)} duration={(len(qpos)-1)/freq:.2f}s freq={freq:.0f}Hz")
    print(f"  path={path:.2f} m  mean_speed={path/((len(qpos)-1)/freq):.3f} m/s")
    print(f"  lateral extent={np.ptp(xy[:,1]):.3f} m  height min={pel[:,2].min():.3f} m")
    print(f"  seams={seams}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
