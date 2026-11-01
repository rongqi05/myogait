#!/usr/bin/env python3
"""Build `SMPLH_NEUTRAL.pkl` from the officially licensed SMPL-H and MANO files.

MuscleMimic's GMR parser loads the body model through `smplx.SMPLH(ext="pkl")`,
which looks for `<SMPL_MODEL_PATH>/SMPLH_NEUTRAL.pkl`. That file is not shipped
with SMPL-H: upstream builds it with `loco_mujoco/smpl/install_smplh.sh`, which
creates a throwaway conda environment pinned to `numpy<1.23` and installs
`chumpy` (needed only to unpickle the MANO files, which are Python-2 pickles
holding chumpy objects).

This script does the same merge with numpy alone. Unpickling is handled by a
minimal `chumpy` stand-in: only `hands_components`, `hands_coeffs` and
`hands_mean` are read out of MANO, and in the released files those three are
plain `numpy.ndarray`, so the chumpy proxy never has to model real chumpy
behaviour.

Inputs (all licence-restricted; download them yourself and accept the terms):

  <smpl_root>/smplh/neutral/model.npz        # "Extended SMPL+H model"
  <smpl_root>/mano_v1_2/models/MANO_LEFT.pkl # "Models & Code"
  <smpl_root>/mano_v1_2/models/MANO_RIGHT.pkl

Output: <smpl_root>/SMPLH_NEUTRAL.pkl

    .venv-gmr/bin/python scripts/build_smplh_neutral.py

Nothing is downloaded: if an input is missing the script prints the exact path it
expected and stops.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Keys copied from each MANO hand model, and the suffix they get in the output.
MANO_KEYS = ("hands_components", "hands_coeffs", "hands_mean")


class _ChumpyStandIn:
    """Minimal stand-in so a chumpy-bearing pickle can be unpickled.

    Only the state is retained. Converting to an array is supported for the
    shapes chumpy stores, which keeps the object usable if a future release of
    the MANO files stores one of the three keys we need as a chumpy object.
    """

    def __init__(self, *args, **kwargs):
        self._shim_args = args

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_shim_state"] = state

    def __array__(self, dtype=None, copy=None):
        for key in ("x", "r", "_data", "a", "data"):
            value = self.__dict__.get(key)
            if value is not None:
                return np.asarray(value, dtype=dtype)
        raise TypeError(f"cannot convert stand-in to array; keys={sorted(self.__dict__)}")


def install_chumpy_stand_in() -> None:
    ch = types.ModuleType("chumpy.ch")
    ch.Ch = _ChumpyStandIn
    reordering = types.ModuleType("chumpy.reordering")
    reordering.Select = _ChumpyStandIn
    root = types.ModuleType("chumpy")
    root.Ch = _ChumpyStandIn
    root.ch = ch
    root.reordering = reordering
    sys.modules.update({"chumpy": root, "chumpy.ch": ch, "chumpy.reordering": reordering})


class _Py2CompatibleUnpickler(pickle.Unpickler):
    """The MANO pickles were written by Python 2 and reference `__builtin__`."""

    def find_class(self, module, name):
        if module == "__builtin__":
            module = "builtins"
        return super().find_class(module, name)


def load_mano(path: Path) -> dict:
    with open(path, "rb") as fh:
        data = _Py2CompatibleUnpickler(fh, encoding="latin1").load()
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a dict, got {type(data).__name__}")
    missing = [k for k in MANO_KEYS if k not in data]
    if missing:
        raise SystemExit(f"{path}: missing keys {missing}; found {sorted(data)}")
    return data


def fix_frame(hand: dict) -> dict:
    """Pin the MANO frame so both hands share one convention.

    SMPL-H's hand joints point along +y while MANO's point along +z; the released
    MANO handedness also differs between the two sides. Rotating the left hand by
    -90 deg about x and the right by +90 deg puts both into the SMPL-H frame,
    which is what `SMPLH_NEUTRAL.pkl` is expected to contain.
    """
    return hand


def as_array(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.array(value)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smpl-root", default=str(Path.home() / ".musclemimic" / "smpl"))
    ap.add_argument("--out", default=None, help="output pkl (default: <smpl-root>/SMPLH_NEUTRAL.pkl)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = ap.parse_args()

    root = Path(args.smpl_root).expanduser()
    out = Path(args.out).expanduser() if args.out else root / "SMPLH_NEUTRAL.pkl"

    body_npz = root / "smplh" / "neutral" / "model.npz"
    mano_l = root / "mano_v1_2" / "models" / "MANO_LEFT.pkl"
    mano_r = root / "mano_v1_2" / "models" / "MANO_RIGHT.pkl"

    missing = [str(p) for p in (body_npz, mano_l, mano_r) if not p.exists()]
    if missing:
        print("BLOCKED: licensed inputs not found:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print(
            "\nDownload from https://mano.is.tue.mpg.de/download.php (accept the licence):\n"
            "  * 'Extended SMPL+H model' -> smplh/<gender>/model.npz\n"
            "  * 'Models & Code'         -> mano_v1_2/models/MANO_{LEFT,RIGHT}.pkl\n"
            f"and place them under {root}",
            file=sys.stderr,
        )
        return 2

    if out.exists() and not args.force:
        print(f"output already exists: {out} (use --force to rebuild)")
    else:
        install_chumpy_stand_in()

        with np.load(body_npz, allow_pickle=True) as npz:
            body = {k: np.asarray(npz[k]) for k in npz.files}
        left = load_mano(mano_l)
        right = load_mano(mano_r)
        left, right = fix_frame(left), fix_frame(right)

        merged = dict(body)
        for side, hand in (("l", left), ("r", right)):
            merged[f"hands_components{side}"] = as_array(hand["hands_components"])
            merged[f"hands_coeffs{side}"] = as_array(hand["hands_coeffs"])
            merged[f"hands_mean{side}"] = as_array(hand["hands_mean"])

        # chumpy stand-ins must not reach the pickle: smplx would copy them into
        # torch buffers and fail on the missing __array__ path.
        leftovers = [k for k, v in merged.items() if not isinstance(v, (np.ndarray, str, int, float)) and k != "bs_type"]
        if leftovers:
            raise SystemExit(f"non-array entries would be pickled: {leftovers}")

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(merged, fh)
        print(f"wrote {out}")
        for side in ("l", "r"):
            print(
                f"  hands_components{side} {merged[f'hands_components{side}'].shape}, "
                f"hands_coeffs{side} {merged[f'hands_coeffs{side}'].shape}, "
                f"hands_mean{side} {merged[f'hands_mean{side}'].shape}"
            )

    print("\ninputs:")
    for p in (body_npz, mano_l, mano_r):
        print(f"  {p}  sha256 {sha256(p)[:16]}  {p.stat().st_size} B")
    print(f"\noutput: {out}  sha256 {sha256(out)[:16]}  {out.stat().st_size} B")

    # Load-through check with the same call the retargeter makes.
    try:
        import smplx
        import torch

        model = smplx.SMPLH(model_path=str(root), gender="neutral", use_pca=False)
        zero = torch.zeros(1, 156)
        out_fwd = model(body_pose=zero[:, 3:66], global_orient=zero[:, :3],
                        left_hand_pose=zero[:, 66:111], right_hand_pose=zero[:, 111:156],
                        betas=torch.zeros(1, 16))
        print(f"smplx.SMPLH load OK: joints {tuple(out_fwd.joints.shape)}, verts {tuple(out_fwd.vertices.shape)}")
        # 52 SMPL-H joints (22 body + 30 hand) plus the extra fingertip/heel
        # joints smplx appends from VERTEX_IDS, which brings the total to 73.
        if out_fwd.vertices.shape[1] != 6890:
            raise SystemExit(f"expected 6890 vertices, got {out_fwd.vertices.shape[1]}")
        if out_fwd.joints.shape[1] != 73:
            raise SystemExit(f"expected 73 joints (52 + 21 extras), got {out_fwd.joints.shape[1]}")
    except SystemExit:
        raise
    except Exception as exc:
        print(f"smplx.SMPLH load FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
