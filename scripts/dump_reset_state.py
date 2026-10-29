#!/usr/bin/env python3
"""Record the reset (pre-step) state of the evaluation environment.

Used to compare model-package versions: the reset state set by the trajectory
initial-state handler must be identical, so any later divergence is caused by
the model definition, not by the initial condition.

    PYTHONPATH=/tmp/mm_models_106:external/MuscleMimic:scripts \
        .venv/bin/python scripts/dump_reset_state.py --tag v106 --out reports/D_model_divergence/reset_v106.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from _env_builder import build_eval_bundle, env_data, env_model


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/checkpoints/checkpoint_12500")
    ap.add_argument("--motion_path", default="KIT/314/walking_medium09_poses")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import musclemimic_models
    from importlib.metadata import version

    b = build_eval_bundle(args.checkpoint, args.motion_path, start_step=0)
    env = b.env
    model = env_model(env)
    obs = np.asarray(env.reset())
    data = env_data(env)

    pkg = Path(musclemimic_models.__file__).parent
    xml_dir = pkg / "model"
    hashes = {}
    for f in sorted(xml_dir.rglob("*.xml")):
        hashes[str(f.relative_to(pkg))] = sha256_file(f)[:16]

    out = {
        "tag": args.tag,
        "module_file": str(Path(musclemimic_models.__file__)),
        "dist_version": version("musclemimic-models"),
        "xml_path": musclemimic_models.get_xml_path("myofullbody").as_posix(),
        "xml_hashes": hashes,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "obs_dim": int(obs.size),
        "obs_sha256": hashlib.sha256(np.ascontiguousarray(obs).tobytes()).hexdigest(),
        "obs_head": np.round(obs[:12], 6).tolist(),
        "obs": np.round(obs.astype(np.float64), 9).tolist(),
        "qpos": np.round(np.asarray(data.qpos), 9).tolist(),
        "qvel": np.round(np.asarray(data.qvel), 9).tolist(),
        "actuator_length_head": np.round(np.asarray(data.actuator_length[:12]), 6).tolist(),
        "site_xpos_pelvis": np.round(np.asarray(data.site_xpos[env.model.site("pelvis_mimic").id]), 6).tolist()
        if env.model.nsite
        else None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"{args.tag}: obs_sha={out['obs_sha256'][:12]} qpos_sha={hashlib.sha256(np.asarray(data.qpos).tobytes()).hexdigest()[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
