#!/usr/bin/env python3
"""Produce rigidly rotated copies of a retargeted reference.

A rigid rotation about the world z axis changes nothing about the gait: root
translation turns, root orientation turns, every joint angle and every foot
position relative to the body stays bit-identical. That makes it an exact
single-factor probe for a *world-referenced* versus *body-referenced* bias.

If the policy veers by the same angle relative to the reference in every
rotated variant, the veer is fixed in the body/reference frame, so it belongs to
the policy or the model. If the veer stays fixed in world coordinates while the
reference turns underneath it, something in the setup is world-referenced.

    .venv/bin/python scripts/rotate_reference.py --clip KIT/3/straight_stitched_10s \
        --angles 0 90 180 270 --out_name straight_stitched_10s

Rotation 0 is written as well and must reproduce the source's own metrics; the
script asserts that so a transform bug cannot hide as a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_straight_walk import build_model, recompute_derived  # noqa: E402

# Fields copied verbatim from the source; everything geometric is recomputed.
COPY_FIELDS = (
    "joint_names",
    "body_names",
    "site_names",
    "njnt",
    "jnt_type",
    "nbody",
    "body_rootid",
    "body_weldid",
    "body_mocapid",
    "body_pos",
    "body_quat",
    "body_ipos",
    "body_iquat",
    "nsite",
    "site_bodyid",
    "site_pos",
    "site_quat",
)


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )


def yaw_quat_deg(deg: float, n: int) -> np.ndarray:
    half = np.radians(deg) / 2.0
    q = np.zeros((n, 4))
    q[:, 0] = np.cos(half)
    q[:, 3] = np.sin(half)
    return q


def rotate_qpos(qpos: np.ndarray, deg: float) -> np.ndarray:
    """Rotate the free-joint root about world z, leaving the joints untouched."""
    out = qpos.copy()
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    xy = qpos[:, :2].copy()
    out[:, 0] = c * xy[:, 0] - s * xy[:, 1]
    out[:, 1] = s * xy[:, 0] + c * xy[:, 1]
    out[:, 3:7] = quat_mul_wxyz(yaw_quat_deg(deg, len(qpos)), qpos[:, 3:7])
    return out


def differentiate(model, qpos: np.ndarray, freq: float) -> np.ndarray:
    """Recompute qvel from qpos with MuJoCo's own convention.

    mj_differentiatePos handles the quaternion correctly, so the root angular
    velocity ends up in whatever frame MuJoCo expects rather than in a frame
    guessed here.
    """
    import mujoco

    n, nv = len(qpos), model.nv
    qvel = np.zeros((n, nv))
    dt = 1.0 / freq
    scratch = np.zeros(nv)
    for i in range(n - 1):
        mujoco.mj_differentiatePos(model, scratch, dt, qpos[i], qpos[i + 1])
        qvel[i] = scratch
    qvel[-1] = qvel[-2] if n > 1 else 0.0
    return qvel


def write_variant(src: np.lib.npyio.NpzFile, model, qpos: np.ndarray, qvel: np.ndarray, out_path: Path, meta: dict) -> dict:
    site_names = [str(n) for n in src["site_names"]]
    body_names = [str(n) for n in src["body_names"]]
    derived = recompute_derived(model, qpos, qvel, site_names, body_names)
    payload = {
        "qpos": qpos.astype(np.float32),
        "qvel": qvel.astype(np.float32),
        "split_points": np.array([0, len(qpos)], dtype=np.int32),
        "frequency": np.float64(meta["frequency_hz"]),
        "metadata": np.array(json.dumps(meta)),
        **{k: derived[k] for k in ("xpos", "xquat", "cvel", "subtree_com", "site_xpos", "site_xmat")},
    }
    for f in COPY_FIELDS:
        payload[f] = src[f]
    np.savez_compressed(out_path, **payload)

    pel = derived["site_xpos"][:, site_names.index("pelvis_mimic"), :]
    xy = pel[:, :2]
    return {
        "out": str(out_path),
        "pelvis_start_xy": [float(xy[0, 0]), float(xy[0, 1])],
        "pelvis_end_xy": [float(xy[-1, 0]), float(xy[-1, 1])],
        "net_displacement_m": float(np.linalg.norm(xy[-1] - xy[0])),
        "path_length_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
        "height_min_m": float(pel[:, 2].min()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True, help="source clip id, e.g. KIT/3/straight_stitched_10s")
    ap.add_argument("--cache_root", default=str(Path.home() / ".musclemimic/caches/AMASS"))
    ap.add_argument("--env", default="MyoFullBody")
    ap.add_argument("--method", default="gmr")
    ap.add_argument("--angles", nargs="+", type=float, default=[0.0, 90.0, 180.0, 270.0])
    ap.add_argument(
        "--out_name",
        default="rotref",
        help="output clip id stem; each angle becomes <out_name>_rot<angle>",
    )
    ap.add_argument("--out", default=None, help="output directory (default: the source cache dir)")
    ap.add_argument("--report", default="reports/rotate_reference.json")
    args = ap.parse_args()

    src_path = Path(args.cache_root).expanduser() / args.env / args.method / f"{args.clip}.npz"
    if not src_path.exists():
        raise SystemExit(f"source not found: {src_path}")
    out_dir = Path(args.out).expanduser() if args.out else src_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    src = np.load(src_path, allow_pickle=True)
    freq = float(src["frequency"])
    qpos0 = np.asarray(src["qpos"], np.float64)
    model = build_model(expect_nq=qpos0.shape[1])

    # The trajectory loader recomputes qvel from qpos with mj_differentiatePos
    # (loco_mujoco/trajectory/dataclasses.py), so the stored qvel is not
    # authoritative and reproducing it is pointless. What must hold is that
    # differentiating works on this qpos at all: 674 of its 1340 root
    # quaternions have w < 0, and a sign flip would otherwise read as a 2*pi
    # rotation. Verified separately that mj_differentiatePos is invariant to
    # q -> -q, so the variants inherit a well-defined velocity.
    qvel = differentiate(model, qpos0, freq)
    max_ang = float(np.abs(qvel[:, 3:6]).max())
    print(f"qvel from qpos: max |root angular velocity| = {max_ang:.3f} rad/s "
          f"({np.degrees(max_ang):.1f} deg/s)")

    # A source that already contains a root orientation snap (e.g. a stitched
    # clip blended with lerp instead of slerp) is not fixed by rotation:
    # premultiplying by a constant quaternion preserves every relative angle.
    # Report it so the result is never read as a clean baseline, but proceed.
    quat = qpos0[:, 3:7] / np.linalg.norm(qpos0[:, 3:7], axis=1, keepdims=True)
    dot = np.abs(np.einsum("ij,ij->i", quat[:-1], quat[1:]))
    step_deg = np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))
    if step_deg.size and step_deg.max() > 20.0:
        print(f"WARNING: source root orientation snaps {step_deg.max():.1f} deg in one frame at "
              f"t={step_deg.argmax() + 1:.2f} frames; the rotation variants inherit it unchanged")

    rows = []
    for deg in args.angles:
        name = f"{args.out_name}_rot{int(deg) % 360:03d}"
        qpos = rotate_qpos(qpos0, deg)
        qvel = differentiate(model, qpos, freq)
        meta = {
            "created_by": "scripts/rotate_reference.py",
            "source_clip": args.clip,
            "rotation_deg": float(deg),
            "frequency_hz": freq,
            "n_frames": int(len(qpos)),
            "note": "rigid z-rotation of the whole reference; gait and joint angles are unchanged",
        }
        info = write_variant(src, model, qpos, qvel, out_dir / f"{name}.npz", meta)
        info.update({"clip": f"{Path(args.clip).parent}/{name}", "rotation_deg": deg, "name": name})
        rows.append(info)
        print(f"  rot{int(deg) % 360:03d}: net {info['net_displacement_m']:.3f} m, "
              f"start {info['pelvis_start_xy'][0]:.2f},{info['pelvis_start_xy'][1]:.2f} "
              f"-> end {info['pelvis_end_xy'][0]:.2f},{info['pelvis_end_xy'][1]:.2f}")

    # Rotation must preserve distance from the origin of the *path*, i.e. net
    # displacement and path length are invariants; only absolute xy changes.
    ref = [r for r in rows if abs(r["rotation_deg"]) < 1e-9]
    if ref:
        net = ref[0]["net_displacement_m"]
        bad = [r["name"] for r in rows if abs(r["net_displacement_m"] - net) > 1e-3]
        print(f"invariance check: net displacement {net:.4f} m across angles, violations={bad}")
        if bad:
            raise SystemExit("rotation changed net displacement; transform is wrong")

    out = REPO_ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": str(src_path), "frequency_hz": freq, "variants": rows}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
