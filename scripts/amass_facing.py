#!/usr/bin/env python3
"""Check whether an AMASS clip walks forward or backward.

The straightness audit only reads `trans`, so it measures the path the root
travels and cannot tell which way the body faces. A backward-walking trial is
perfectly straight, steady and unremarkable by those numbers, yet it is useless
as a forward-walking reference: after retargeting, the MyoFullBody feet point
against the direction of travel and the policy (trained on forward walking)
cannot track it.

The test is convention free: compare the horizontal ankle -> foot joint
direction of the SMPL-H skeleton with the direction the root actually travels.
Both come out of the same body model, so no axis convention is assumed.

    .venv-gmr/bin/python scripts/amass_facing.py --clip CMU/136/136_25_poses CMU/120/120_19_poses

Exit status: 0 when every clip is FORWARD, 1 when any clip is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# SMPL-H body joint indices: ankle -> foot points along the body's forward axis.
ANKLE_FOOT = {"left": (7, 10), "right": (8, 11)}
FORWARD_DOT = 0.5
BACKWARD_DOT = -0.5


def _up_axis(trans: np.ndarray) -> tuple[int, list[int]]:
    rng = np.ptp(trans, axis=0)
    up = int(np.argmin(rng))
    return up, [i for i in range(3) if i != up]


def facing(clip_path: Path, smpl_root: Path, max_frames: int = 2000) -> dict:
    import smplx
    import torch

    d = np.load(clip_path, allow_pickle=True)
    poses = np.asarray(d["poses"], np.float32)
    trans = np.asarray(d["trans"], np.float32)
    betas = np.asarray(d["betas"], np.float32)[:16]
    n = len(poses)
    idx = np.arange(0, n, max(1, n // max_frames))[:max_frames]

    model = smplx.SMPLH(model_path=str(smpl_root), gender="neutral", use_pca=False, batch_size=len(idx))
    with torch.no_grad():
        out = model(
            betas=torch.tensor(betas)[None].repeat(len(idx), 1),
            global_orient=torch.tensor(poses[idx, :3]),
            body_pose=torch.tensor(poses[idx, 3:66]),
            left_hand_pose=torch.tensor(poses[idx, 66:111]),
            right_hand_pose=torch.tensor(poses[idx, 111:156]),
            transl=torch.tensor(trans[idx]),
        )
    joints = out.joints.numpy()

    up, horiz = _up_axis(trans)
    xy = trans[:, horiz]
    net = float(np.linalg.norm(xy[-1] - xy[0]))
    if net < 0.3:
        return {"clip": clip_path.stem, "verdict": "STATIONARY", "net_displacement_m": net}

    travel = (xy[-1] - xy[0]) / net
    dots = {}
    for side, (ankle, foot) in ANKLE_FOOT.items():
        d3 = (joints[:, foot] - joints[:, ankle]).mean(0)[horiz]
        dots[side] = float(np.dot(d3 / np.linalg.norm(d3), travel))

    if all(v > FORWARD_DOT for v in dots.values()):
        verdict = "FORWARD"
    elif all(v < BACKWARD_DOT for v in dots.values()):
        verdict = "BACKWARD"
    else:
        verdict = "AMBIGUOUS"

    return {
        "clip": str(clip_path.stem).replace("_poses", ""),
        "verdict": verdict,
        "toes_dot_travel": dots,
        "net_displacement_m": net,
        "up_axis": up,
        "frames": n,
        "frames_used": int(len(idx)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", nargs="+", required=True, help="clip ids relative to the AMASS root")
    ap.add_argument("--amass-root", default=None)
    ap.add_argument("--smpl-root", default=str(Path.home() / ".musclemimic" / "smpl"))
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from loco_mujoco.smpl import retargeting as R

    amass_root = Path(args.amass_root).expanduser() if args.amass_root else Path(R.get_amass_dataset_path())
    rows = []
    bad = 0
    for clip in args.clip:
        path = amass_root / f"{clip}.npz"
        if not path.exists():
            print(f"  MISSING  {clip}  ({path})")
            rows.append({"clip": clip, "verdict": "MISSING"})
            bad += 1
            continue
        row = facing(path, Path(args.smpl_root).expanduser())
        row["clip_id"] = clip
        rows.append(row)
        dots = row.get("toes_dot_travel", {})
        detail = " ".join(f"{k}={v:+.3f}" for k, v in dots.items())
        print(f"  {row['verdict']:9s} {clip:34s} net {row['net_displacement_m']:6.2f} m  {detail}")
        if row["verdict"] != "FORWARD":
            bad += 1

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {out}")

    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
