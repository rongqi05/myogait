#!/usr/bin/env python3
"""Retarget exactly one AMASS clip to MyoFullBody through MuscleMimic's GMR-Fit path.

``external/MuscleMimic/scripts/retarget_dataset.py`` only accepts named dataset
groups, so a single hand-picked clip cannot go through it. This script drives the
same public entry point that script uses internally
(``ImitationFactory.make`` + ``AMASSDatasetConf``) for one motion, without
touching upstream code.

Three gates run before any retargeting work starts, because a retargeted
reference is only comparable with the frozen acceptance baseline if all three
hold:

  1. ``musclemimic_models`` is 1.0.5 (1.0.6 negates the left-knee constraint
     polycoefs, see reports/D_model_version_compatibility.md);
  2. the raw clip is long enough and straight enough *before* retargeting, so a
     clip that is not a 10 s straight walk cannot become one by accident;
  3. the SMPL-H body model the GMR parser needs is actually present.

    .venv-gmr/bin/python scripts/retarget_one.py --clip CMU/136/136_25_poses

Add ``--dry-run`` to run only the gates and print the exact command that would
be executed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Frozen GMR parameters. Changing any of these changes the reference the policy
# is later evaluated on, so they are not exposed as flags.
GMR_PARAMS = {
    "src_human": "smplh",
    "target_fps": 30,
    "solver": "daqp",
    "damping": 0.5,
    "offset_to_ground": False,
    "use_velocity_limit": False,
    "verbose": False,
}

MIN_DURATION_S = 10.0
MIN_WINDOW_SPEED_MPS = 0.6


def fail(stage: str, detail: str) -> int:
    print(f"\nBLOCKED at stage '{stage}':\n  {detail}", file=sys.stderr)
    return 2


def gate_model() -> None:
    """Raise unless the environment model is the 1.0.5 MyoFullBody variant."""
    import importlib.metadata as md

    version = md.version("musclemimic_models")
    if version != "1.0.5":
        raise RuntimeError(f"musclemimic_models is {version}, need 1.0.5 (see configs/constraints.txt)")

    from musclemimic.environments.humanoids.myofullbody import MyoFullBody

    model = MyoFullBody()._model
    dims = (model.nq, model.nv, model.nu)
    if dims != (89, 88, 354):
        raise RuntimeError(f"MyoFullBody dims are {dims}, need (89, 88, 354)")

    mag = 0.369499
    pos = [i for i in range(model.neq) if abs(model.eq_data[i][1] - mag) < 1e-6]
    neg = [i for i in range(model.neq) if abs(model.eq_data[i][1] + mag) < 1e-6]
    if not (len(pos) == 1 and len(neg) == 1):
        raise RuntimeError(
            f"knee constraint sign pattern is {len(pos)} positive / {len(neg)} negative, "
            "need 1/1 (1.0.5 layout)"
        )
    print(f"  model: musclemimic_models {version}, nq/nv/nu={dims}, knee 1.0.5 layout")


def gate_raw_clip(clip: str, amass_root: Path) -> dict:
    """Measure the raw clip; retargeting must not be the thing that makes it usable."""
    import numpy as np

    from amass_kit_audit import analyse

    path = amass_root / f"{clip}.npz"
    if not path.exists():
        raise RuntimeError(f"raw clip not found: {path}")

    with np.load(path, allow_pickle=True) as d:
        trans = np.asarray(d["trans"], dtype=np.float64)
        fps = float(d["mocap_framerate"])

    row = analyse(f"{clip}.npz", trans, fps)
    window = row["longest_straight_window"]
    print(
        f"  raw: {row['duration_s']:.2f} s @ {fps:.0f} Hz, "
        f"straight window {window['start_s']:.2f}-{window['end_s']:.2f} s "
        f"({window['duration_s']:.2f} s, {window['mean_speed_mps']:.3f} m/s)"
    )
    if window["duration_s"] < MIN_DURATION_S:
        raise RuntimeError(
            f"raw straight window is {window['duration_s']:.2f} s < {MIN_DURATION_S} s; "
            "a retargeted clip cannot be longer/straighter than its source"
        )
    return row


def gate_facing(clip: str, amass_root: Path, smpl_root: Path) -> dict:
    """The straightness audit only sees the root path, not which way the body faces.

    A backward-walking trial passes every geometric check and then produces a
    reference whose feet point against the direction of travel, which the policy
    cannot track (measured: 18/20 falls). Checking facing before spending a
    retarget keeps that class of clip out of the pipeline.
    """
    from amass_facing import facing

    row = facing(amass_root / f"{clip}.npz", smpl_root)
    dots = row.get("toes_dot_travel", {})
    detail = " ".join(f"{k}={v:+.3f}" for k, v in dots.items())
    print(f"  toes vs travel: {detail} -> {row['verdict']}")
    if row["verdict"] != "FORWARD":
        raise RuntimeError(
            f"clip is {row['verdict']} (toes·travel {detail}); "
            "a backward-walking clip cannot serve as a forward-walking reference"
        )
    return row


def gate_body_model(smpl_root: Path) -> None:
    from loco_mujoco.smpl import retargeting as R

    resolved = Path(R.get_smpl_model_path())
    if resolved != smpl_root:
        print(f"  note: SMPL_MODEL_PATH resolves to {resolved} (configured: {smpl_root})")
    model = resolved / "SMPLH_NEUTRAL.pkl"
    if not model.exists():
        raise RuntimeError(
            f"SMPL-H body model missing: {model}\n"
            "  The GMR parser loads <SMPL_MODEL_PATH>/SMPLH_NEUTRAL.pkl via smplx.SMPLH(ext='pkl').\n"
            "  It is licence-restricted (Max Planck / MANO). Download the Extended SMPL+H model\n"
            "  and the MANO hand models from https://mano.is.tue.mpg.de/download.php with your\n"
            "  own account, then run external/MuscleMimic/loco_mujoco/smpl/install_smplh.sh.\n"
            "  See reports/I_gmr_toolchain.md section I.5.1."
        )
    print(f"  body model: {model} ({model.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True, help="AMASS clip id relative to the AMASS root, e.g. CMU/136/136_25_poses")
    ap.add_argument("--smpl-root", default=str(Path.home() / ".musclemimic" / "smpl"))
    ap.add_argument("--dry-run", action="store_true", help="run the gates only")
    ap.add_argument("--json-out", default=None, help="write a machine-readable summary")
    args = ap.parse_args()

    from loco_mujoco.smpl import retargeting as R

    amass_root = Path(R.get_amass_dataset_path())
    smpl_root = Path(args.smpl_root).expanduser()

    print(f"retarget {args.clip} -> MyoFullBody (GMR-Fit, frozen params)")
    print(f"  AMASS root: {amass_root}")

    summary = {"clip": args.clip, "amass_root": str(amass_root), "gmr": GMR_PARAMS, "gates": {}}

    print("\n[gate 1] environment model version")
    try:
        gate_model()
        summary["gates"]["model"] = "pass"
    except Exception as exc:
        summary["gates"]["model"] = f"fail: {exc}"
        return fail("model", str(exc))

    print("\n[gate 2] raw clip is already a >=10 s straight walk")
    try:
        summary["raw"] = gate_raw_clip(args.clip, amass_root)
        summary["gates"]["raw_clip"] = "pass"
    except Exception as exc:
        summary["gates"]["raw_clip"] = f"fail: {exc}"
        return fail("raw_clip", str(exc))

    print("\n[gate 3] clip walks forward")
    try:
        summary["facing"] = gate_facing(args.clip, amass_root, smpl_root)
        summary["gates"]["facing"] = "pass"
    except Exception as exc:
        summary["gates"]["facing"] = f"fail: {exc}"
        return fail("facing", str(exc))

    print("\n[gate 4] SMPL-H body model")
    try:
        gate_body_model(smpl_root)
        summary["gates"]["body_model"] = "pass"
    except Exception as exc:
        summary["gates"]["body_model"] = f"fail: {exc}"
        return fail("body_model", str(exc))

    if args.dry_run:
        print("\ndry run: all gates passed, nothing retargeted")
        _dump(summary, args.json_out)
        return 0

    print("\n[retarget] building the environment (this triggers GMR-Fit)")
    from loco_mujoco.task_factories import AMASSDatasetConf, ImitationFactory

    conf = AMASSDatasetConf([args.clip])
    conf.retargeting_method = "gmr"
    conf.gmr_config = dict(GMR_PARAMS)

    t0 = time.time()
    try:
        env = ImitationFactory.make("MyoFullBody", amass_dataset_conf=conf, headless=True)
        del env
    except Exception as exc:
        summary["gates"]["retarget"] = f"fail: {type(exc).__name__}: {exc}"
        _dump(summary, args.json_out)
        return fail("retarget", f"{type(exc).__name__}: {exc}")

    elapsed = time.time() - t0
    summary["gates"]["retarget"] = "pass"
    summary["retarget_seconds"] = elapsed
    print(f"\nretargeted in {elapsed:.1f} s")

    from pathlib import Path as _P

    cache = _P(R.get_converted_amass_dataset_path()) / "MyoFullBody" / "gmr" / f"{args.clip}.npz"
    summary["cache_path"] = str(cache)
    summary["cache_exists"] = cache.exists()
    print(f"  cache: {cache} (exists={cache.exists()})")
    print("\nnext: .venv/bin/python scripts/qc_retargeted.py --clip " + args.clip)

    _dump(summary, args.json_out)
    return 0


def _dump(summary: dict, out: str | None) -> None:
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2, default=str))
        print(f"wrote {p}")


if __name__ == "__main__":
    raise SystemExit(main())
