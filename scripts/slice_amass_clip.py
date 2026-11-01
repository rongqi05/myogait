#!/usr/bin/env python3
"""Write a continuous sub-window of an AMASS clip as a standalone clip.

Slicing takes whole frames with no interpolation and no blending, so the result
is still a single continuous recording -- unlike the stitched reference, which
joins two different walks. Used to cut a long AMASS session down to its straight
forward-walking window before retargeting.

    .venv/bin/python scripts/slice_amass_clip.py \
        --clip CMU/120/120_19_poses --start 48.3 --end 59.8 \
        --out_clip CMU/120/120_19_straight
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TIME_KEYS = ("trans", "dmpls")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True, help="source clip id relative to the AMASS root")
    ap.add_argument("--start", type=float, required=True, help="start time (s)")
    ap.add_argument("--end", type=float, required=True, help="end time (s)")
    ap.add_argument("--out_clip", required=True, help="output clip id relative to the AMASS root")
    ap.add_argument("--amass-root", default=str(Path.home() / ".musclemimic" / "AMASS"))
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    root = Path(args.amass_root).expanduser()
    src = root / f"{args.clip}.npz"
    dst = root / f"{args.out_clip}.npz"
    if not src.exists():
        raise SystemExit(f"source not found: {src}")

    with np.load(src, allow_pickle=True) as d:
        data = {k: np.asarray(d[k]) for k in d.files}
    fps = float(data["mocap_framerate"])
    n = len(data["trans"])
    i0 = max(0, int(round(args.start * fps)))
    i1 = min(n, int(round(args.end * fps)) + 1)
    if i1 - i0 < 10:
        raise SystemExit(f"window too short: {i1 - i0} frames")

    out = {}
    for k, v in data.items():
        if k in TIME_KEYS or k == "poses":
            out[k] = v[i0:i1]
        else:
            out[k] = v

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)

    info = {
        "source_clip": args.clip,
        "out_clip": args.out_clip,
        "fps": fps,
        "source_frames": int(n),
        "frame_range": [i0, i1],
        "window_s": [i0 / fps, (i1 - 1) / fps],
        "out_frames": int(i1 - i0),
        "duration_s": float((i1 - i0 - 1) / fps),
        "method": "contiguous frame slice, no interpolation, no blending",
    }
    print(f"wrote {dst}")
    for k, v in info.items():
        print(f"  {k}: {v}")
    if args.report:
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
