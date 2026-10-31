#!/usr/bin/env python3
"""Verify the GMR-Fit retargeting toolchain inside .venv-gmr.

Run with the retargeting interpreter:

    .venv-gmr/bin/python scripts/check_gmr_toolchain.py

The check is split into two tiers:

  * toolchain  -- python packages, GMR integration flags, MyoFullBody model
                  dimensions. All of these are installable from public sources.
  * assets     -- licensed body models and motion data. These cannot be
                  fetched automatically; the checker reports the exact paths
                  the pipeline expects instead of guessing.

Exit status is 0 when every ``blocking`` item passes, 1 otherwise. Items that
are only needed for a specific pipeline variant are marked non-blocking.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Values frozen in configs/eval_walk.json. The retargeted reference must be
# produced with the same environment model the acceptance run uses, otherwise
# the two are not comparable.
FROZEN = {
    "musclemimic_models": "1.0.5",
    "mujoco": "3.4.0",
    "numpy": "2.2.6",
    "jax": "0.7.2",
}

EXPECTED_MODEL = {"nq": 89, "nv": 88, "nu": 354, "neq": 51}

RESULTS: list[dict] = []


def record(section: str, name: str, status: str, detail: str, blocking: bool = True) -> None:
    RESULTS.append(
        {
            "section": section,
            "name": name,
            "status": status,  # ok | mismatch | missing | unknown
            "detail": detail,
            "blocking": blocking,
        }
    )


def version_of(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def check_runtime() -> None:
    record("runtime", "python", "ok", platform.python_version())
    record("runtime", "interpreter", "ok", str(Path(sys.executable).resolve()))
    in_venv = sys.prefix != sys.base_prefix
    record(
        "runtime",
        "virtualenv",
        "ok" if in_venv else "mismatch",
        sys.prefix,
        blocking=True,
    )


def check_packages() -> None:
    for dist, want in FROZEN.items():
        got = version_of(dist)
        if got is None:
            record("packages", dist, "missing", "not installed")
        elif got != want:
            record("packages", dist, "mismatch", f"{got} (frozen: {want})")
        else:
            record("packages", dist, "ok", got)

    # cpu-only torch is intended: retargeting is pure IK and leaves VRAM free.
    got = version_of("torch")
    if got is None:
        record("packages", "torch", "missing", "not installed")
    else:
        record("packages", "torch", "ok", f"{got} (cuda={torch_cuda_available()})")

    for dist in ("smplx", "general_motion_retargeting", "mink", "joblib", "qpsolvers", "daqp"):
        got = version_of(dist)
        record("packages", dist, "ok" if got else "missing", got or "not installed")

    # Direct URL installs have no version pin in metadata; report the commit.
    for dist, mod in (("smplx", "smplx"), ("general_motion_retargeting", "general_motion_retargeting")):
        direct = _direct_url_commit(dist)
        if direct:
            record("packages", f"{dist}@vcs", "ok", direct)


def _direct_url_commit(dist: str) -> str | None:
    try:
        text = md.distribution(dist).read_text("direct_url.json")
    except Exception:
        return None
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    url = data.get("url", "")
    commit = (data.get("vcs_info") or {}).get("commit_id", "")
    return f"{url}@{commit[:12]}" if commit else url


def torch_cuda_available() -> str:
    try:
        import torch
    except Exception:
        return "n/a"
    return str(torch.cuda.is_available())


def check_gmr_integration() -> None:
    try:
        from loco_mujoco.smpl import retargeting as R
    except Exception as exc:  # pragma: no cover - import failure is the finding
        record("gmr_integration", "loco_mujoco.smpl.retargeting", "missing", f"{type(exc).__name__}: {exc}")
        return

    record(
        "gmr_integration",
        "_GMR_INSTALLED",
        "ok" if R._GMR_INSTALLED else "missing",
        str(R._GMR_INSTALLED),
    )
    record(
        "gmr_integration",
        "_GMR_SHAPE_FITTING_INSTALLED",
        "ok" if R._GMR_SHAPE_FITTING_INSTALLED else "missing",
        str(R._GMR_SHAPE_FITTING_INSTALLED),
    )

    # qpsolvers must expose daqp, the solver the frozen GMR invocation uses.
    try:
        import qpsolvers

        have = qpsolvers.available_solvers
        record("gmr_integration", "qpsolvers solvers", "ok" if "daqp" in have else "missing", ", ".join(have))
    except Exception as exc:
        record("gmr_integration", "qpsolvers solvers", "missing", f"{type(exc).__name__}: {exc}")

    # The MyoFullBody IK config is what binds SMPL-H joints to MyoFullBody sites.
    ik_conf = Path(R.__file__).parent / "gmr_configs" / "smplh_to_myofullbody.json"
    if ik_conf.exists():
        conf = json.loads(ik_conf.read_text())
        n1 = len(conf.get("ik_match_table1", {}))
        n2 = len(conf.get("ik_match_table2", {}))
        record("gmr_integration", "smplh_to_myofullbody.json", "ok", f"{ik_conf.name}: {n1}+{n2} IK matches")
    else:
        record("gmr_integration", "smplh_to_myofullbody.json", "missing", str(ik_conf))


def check_model() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        from musclemimic.environments.humanoids.myofullbody import MyoFullBody

        env = MyoFullBody()
        model = env._model
    except Exception as exc:
        record("model", "MyoFullBody env", "missing", f"{type(exc).__name__}: {exc}")
        return

    got = {"nq": model.nq, "nv": model.nv, "nu": model.nu, "neq": model.neq}
    bad = {k: v for k, v in got.items() if EXPECTED_MODEL.get(k) not in (None, v)}
    status = "ok" if not bad else "mismatch"
    record(
        "model",
        "MyoFullBody dims",
        status,
        " ".join(f"{k}={v}" for k, v in got.items())
        + ("" if not bad else f" (frozen: {EXPECTED_MODEL})"),
    )

    # 1.0.6 negates both polycoefs of the left-knee equality constraint, which is
    # what makes mm-10m-2 fall over (reports/D_model_version_compatibility.md).
    # Both knee constraints share the same |coef[1]| = 0.369499; in 1.0.5 exactly
    # one of them is negative, in 1.0.6 both are positive. Checking the sign
    # pattern catches an accidental `uv sync` that re-pinned the package.
    try:
        record("model", "left-knee eq sign", *_knee_sign_status(model))
    except Exception as exc:
        record("model", "left-knee eq sign", "unknown", f"{type(exc).__name__}: {exc}")

    record("model", "sites", "ok", str(model.nsite), blocking=False)


KNEE_COEF1_MAG = 0.369499


def _knee_sign_status(model):
    """Classify the knee equality-constraint sign pattern as 1.0.5 or 1.0.6."""
    pos = [i for i in range(model.neq) if abs(model.eq_data[i][1] - KNEE_COEF1_MAG) < 1e-6]
    neg = [i for i in range(model.neq) if abs(model.eq_data[i][1] + KNEE_COEF1_MAG) < 1e-6]
    detail = f"{len(pos)} positive / {len(neg)} negative knee polycoefs"
    if len(pos) == 1 and len(neg) == 1:
        return "ok", detail + " (1.0.5 layout)"
    if len(pos) == 2 and len(neg) == 0:
        return "mismatch", detail + " (1.0.6 layout: left-knee polycoef negated)"
    return "unknown", detail + f" (expected 1 positive + 1 negative; rows +{pos} -{neg})"


def check_assets() -> None:
    from loco_mujoco.smpl import retargeting as R

    smpl_root = Path(R.get_smpl_model_path())
    record("assets", "SMPL_MODEL_PATH", "ok" if smpl_root.exists() else "missing", str(smpl_root), blocking=False)

    # Runtime asset: gmr_plus' SMPLH_Parser subclasses smplx.SMPLH with the fork's
    # default ext="pkl", so it loads <root>/SMPLH_NEUTRAL.pkl directly.
    neutral_pkl = smpl_root / "SMPLH_NEUTRAL.pkl"
    _record_file("SMPLH_NEUTRAL.pkl (runtime body model)", neutral_pkl)

    # Inputs for loco_mujoco/smpl/install_smplh.sh, which builds SMPLH_NEUTRAL.pkl.
    _record_file(
        "smplh/neutral/model.npz (Extended SMPL+H)",
        smpl_root / "smplh" / "neutral" / "model.npz",
        blocking=False,
        note="only needed to regenerate SMPLH_NEUTRAL.pkl",
    )
    for side in ("LEFT", "RIGHT"):
        _record_file(
            f"mano_v1_2/models/MANO_{side}.pkl",
            smpl_root / "mano_v1_2" / "models" / f"MANO_{side}.pkl",
            blocking=False,
            note="only needed to regenerate SMPLH_NEUTRAL.pkl",
        )

    amass_root = Path(R.get_amass_dataset_path())
    n_clips = sum(1 for _ in amass_root.glob("**/*_poses.npz")) if amass_root.exists() else 0
    record(
        "assets",
        "AMASS dataset root",
        "ok" if n_clips else "missing",
        f"{amass_root} ({n_clips} *_poses.npz)",
    )
    if not n_clips:
        # An archive in Downloads is not usable until it is extracted to the exact
        # layout the retargeter expects (<root>/<dataset>/<subject>/<clip>_poses.npz).
        staged = sorted(p.name for p in (Path.home() / "Downloads").glob("*.tar.bz2"))
        staged += sorted(p.name for p in (Path.home() / "Downloads").glob("*.crdownload"))
        record(
            "assets",
            "AMASS archive in ~/Downloads",
            "unknown",
            ", ".join(staged) if staged else "none; extract AMASS to " + str(amass_root),
            blocking=False,
        )

    conv = Path(R.get_converted_amass_dataset_path())
    n_cached = sum(1 for _ in conv.glob("**/*.npz")) if conv.exists() else 0
    record("assets", "converted cache root", "ok" if conv.exists() else "missing", f"{conv} ({n_cached} .npz)", blocking=False)


def _record_file(name: str, path: Path, blocking: bool = True, note: str = "") -> None:
    exists = path.exists()
    detail = str(path)
    if exists:
        detail = f"{path} ({path.stat().st_size / 1e6:.1f} MB)"
    elif note:
        detail = f"{path}  [{note}]"
    record("assets", name, "ok" if exists else "missing", detail, blocking=blocking)


def check_rendering() -> None:
    """Render one frame in a child process.

    Mujoco's EGL teardown raises inside ``__del__`` on this driver, which prints
    an "Exception ignored" traceback that has nothing to do with the result. A
    subprocess keeps that noise out of the report.
    """
    probe = (
        "import os; os.environ['MUJOCO_GL']='egl';"
        "import mujoco;"
        "m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type=\"sphere\" size=\".1\"/></worldbody></mujoco>');"
        "r=mujoco.Renderer(m,64,64); d=mujoco.MjData(m); mujoco.mj_forward(m,d);"
        "r.update_scene(d); f=r.render(); print('FRAME', f.shape, float(f.mean()))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False, timeout=180
    )
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("FRAME")), None)
    if line:
        record("rendering", "EGL offscreen", "ok", line.removeprefix("FRAME ").strip())
    else:
        err = (out.stderr.strip().splitlines() or ["no output"])[-1]
        record("rendering", "EGL offscreen", "missing", err, blocking=False)


def check_resources() -> None:
    usage = shutil.disk_usage(REPO_ROOT)
    record("resources", "disk free (repo volume)", "ok" if usage.free > 20e9 else "unknown", f"{usage.free / 1e9:.1f} GB")

    mem = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            mem[key] = rest.strip()
    except OSError:
        pass
    total = mem.get("MemTotal", "?")
    avail = mem.get("MemAvailable", "?")
    swap = mem.get("SwapTotal", "?")
    record("resources", "memory", "ok", f"total={total} available={avail} swap={swap}")

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if gpu.returncode == 0 and gpu.stdout.strip():
        record("resources", "gpu", "ok", gpu.stdout.strip().replace("\n", " | "), blocking=False)
    else:
        record("resources", "gpu", "unknown", "nvidia-smi unavailable", blocking=False)

    record("resources", "venv size", "ok", _du(REPO_ROOT / ".venv-gmr"), blocking=False)


def _du(path: Path) -> str:
    if not path.exists():
        return "n/a"
    out = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True, check=False)
    return out.stdout.split("\t")[0] if out.stdout else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=None, help="write the full result table as JSON")
    args = parser.parse_args()

    check_runtime()
    check_packages()
    check_gmr_integration()
    check_model()
    check_assets()
    check_rendering()
    check_resources()

    width = max(len(r["name"]) for r in RESULTS)
    section = None
    for r in RESULTS:
        if r["section"] != section:
            section = r["section"]
            print(f"\n[{section}]")
        flag = {"ok": "PASS", "missing": "MISS", "mismatch": "FAIL", "unknown": "  ??"}[r["status"]]
        print(f"  {flag}  {r['name']:<{width}}  {r['detail']}")

    blockers = [r for r in RESULTS if r["blocking"] and r["status"] != "ok"]
    print()
    if blockers:
        print(f"BLOCKED: {len(blockers)} blocking item(s)")
        for r in blockers:
            print(f"  - {r['name']}: {r['detail']}")
    else:
        print("toolchain OK")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"results": RESULTS, "blocked": bool(blockers)}, indent=2))
        print(f"wrote {args.json_out}")

    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
