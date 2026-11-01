#!/usr/bin/env python3
"""Quality-check a retargeted MyoFullBody reference before the policy is evaluated on it.

A retargeted clip can be numerically fine on paper and still be unusable as a
walking reference: joints can snap, the root can teleport, feet can sink through
the floor or skate, and the straight window measured on the raw AMASS clip can
shrink during retargeting. All of those must be ruled out *before* spending
20 rollouts on it.

    .venv/bin/python scripts/qc_retargeted.py --clip CMU/136/136_25_poses
    .venv/bin/python scripts/qc_retargeted.py --clip KIT/3/straight_stitched_10s \
        --reference-for-comparison KIT/3/walk_6m_straight_line06_poses

Every threshold printed here is a *diagnostic* flag, not an acceptance criterion:
the acceptance thresholds live in configs/eval_walk.json and are not changed by
this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from motion_straightness import heading_series, longest_straight_window, wrap180  # noqa: E402

EXPECTED_NQ, EXPECTED_NV, EXPECTED_SITES = 89, 88, 17
FOOT_SITES = ("left_ankle_mimic", "left_toes_mimic", "right_ankle_mimic", "right_toes_mimic")
JOINT_JUMP_DEG_PER_FRAME = 20.0  # a hinge rotating this fast in one frame is a snap
ROOT_ROT_JUMP_DEG_PER_FRAME = 20.0  # a pelvis rotating this fast in one frame is a snap
CONTACT_BAND_M = 0.02  # foot counts as down within 2 cm of its own lowest point
START_XY_TOL_M = 0.10  # env.reset() zeroes root XY, so the reference must start near the origin
ROOT_TELEPORT_FACTOR = 3.0  # root step vs median step


def _fk_site_positions(qpos: np.ndarray, qvel: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Forward-kinematics pelvis/site positions from qpos.

    A cache written by the GMR path stores only qpos/qvel and leaves the derived
    arrays empty: the trajectory loader materialises them with MuJoCo at load
    time. Doing the same here means the QC numbers are computed from exactly the
    quantities the environment will see, rather than from a different file's
    cached copies.
    """
    import mujoco

    from prepare_straight_walk import build_model

    model = build_model(expect_nq=qpos.shape[1])
    env_sites = _mimic_site_names()
    sids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in env_sites]
    if any(i < 0 for i in sids):
        raise SystemExit(f"model is missing some mimic sites: {env_sites}")

    data = mujoco.MjData(model)
    out = np.zeros((len(qpos), len(sids), 3))
    for i in range(len(qpos)):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, data)
        out[i] = data.site_xpos[sids]
    return env_sites, out


def _mimic_site_names() -> list[str]:
    from musclemimic.environments.humanoids.myofullbody import MyoFullBody

    return list(MyoFullBody(disable_fingers=True, headless=True).sites_for_mimic)


def load_clip(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    freq = float(d["frequency"])
    qpos = np.asarray(d["qpos"], np.float64)
    qvel = np.asarray(d["qvel"], np.float64) if "qvel" in d.files else np.zeros((len(qpos), qpos.shape[1] - 1))

    have_derived = "site_xpos" in d.files and np.asarray(d["site_xpos"]).size > 0
    if have_derived and "site_names" in d.files and np.asarray(d["site_names"]).shape != ():
        names = [str(n) for n in d["site_names"]]
        site_xpos = np.asarray(d["site_xpos"], np.float64)
        source = "cached"
    else:
        names, site_xpos = _fk_site_positions(qpos, qvel)
        source = "recomputed (cache stores qpos/qvel only)"

    meta = d["metadata"] if "metadata" in d.files else None
    meta_str = ""
    if meta is not None and np.asarray(meta).shape == ():
        value = np.asarray(meta).item()
        meta_str = str(value) if value is not None else ""

    return {
        "path": path,
        "freq": freq,
        "t": np.arange(qpos.shape[0]) / freq,
        "qpos": qpos,
        "qvel": qvel,
        "site_xpos": site_xpos,
        "site_names": names,
        "joint_names": [str(n) for n in d["joint_names"]] if "joint_names" in d.files else [],
        "jnt_type": np.asarray(d["jnt_type"]) if "jnt_type" in d.files else np.zeros(len(qpos[0]), int),
        "split_points": np.asarray(d["split_points"]) if "split_points" in d.files else np.array([0, len(qpos)]),
        "metadata": meta_str,
        "derived_source": source,
        "raw_keys": d.files,
    }


def check_finite(c: dict) -> dict:
    bad = {}
    for key in ("qpos", "qvel", "site_xpos"):
        arr = c[key]
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        if n_bad:
            bad[key] = n_bad
    return {"ok": not bad, "non_finite_counts": bad}


def check_shape(c: dict) -> dict:
    nq, nv = c["qpos"].shape[1], c["qvel"].shape[1]
    nsite = c["site_xpos"].shape[1]
    ok = nq == EXPECTED_NQ and nv == EXPECTED_NV and nsite == EXPECTED_SITES
    return {"ok": ok, "nq": nq, "nv": nv, "nsite": nsite, "expected": [EXPECTED_NQ, EXPECTED_NV, EXPECTED_SITES]}


def pelvis_and_feet(c: dict) -> dict:
    names, sx = c["site_names"], c["site_xpos"]
    pelvis = sx[:, names.index("pelvis_mimic"), :]
    feet = {s: sx[:, names.index(s), :] for s in FOOT_SITES if s in names}
    return {"pelvis": pelvis, "feet": feet}


def window_metrics(c: dict) -> dict:
    """Straight-window duration/speed, measured the same way as the raw clips."""
    _, speed = heading_series(pelvis_and_feet(c)["pelvis"][:, :2], c["freq"])
    heading, _ = heading_series(pelvis_and_feet(c)["pelvis"][:, :2], c["freq"])
    window = longest_straight_window(heading, c["t"], 20.0)
    i = int(round(window["start_s"] * c["freq"]))
    j = min(len(c["t"]) - 1, int(round(window["end_s"] * c["freq"])))
    pel = pelvis_and_feet(c)["pelvis"]
    xy = pel[i : j + 1, :2]
    w_net = float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) > 1 else 0.0
    w_path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0
    dur = float(c["t"][j] - c["t"][i])
    return {
        **window,
        "net_displacement_m": w_net,
        "path_length_m": w_path,
        "mean_speed_mps": w_net / dur if dur else 0.0,
        "path_speed_mps": w_path / dur if dur else 0.0,
        "straightness": w_net / w_path if w_path else 0.0,
        "clip_mean_speed_mps": float(np.mean(speed)),
        "speed_profile_0p5s": _speed_profile(pel[:, :2], c["freq"], 0.5),
    }


def _speed_profile(xy: np.ndarray, freq: float, bin_s: float) -> list[dict]:
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1) * freq
    k = max(1, int(round(bin_s * freq)))
    out = []
    for b in range(len(step) // k):
        seg = step[b * k : (b + 1) * k]
        out.append({"t": round(b * bin_s, 2), "mean": round(float(seg.mean()), 3), "min": round(float(seg.min()), 3)})
    return out


def check_foot_height(c: dict) -> dict:
    feet = pelvis_and_feet(c)["feet"]
    out = {}
    for name, xyz in feet.items():
        z = xyz[:, 2]
        floor = float(z.min())
        out[name] = {
            "z_min_m": floor,
            "z_max_m": float(z.max()),
            "frames_below_floor_minus_5mm": int(np.count_nonzero(z < floor - 0.005)),
            "range_m": float(z.max() - floor),
        }
    # A reference whose ankle never rises cannot be producing swing phases.
    lifts = [v["range_m"] for k, v in out.items() if "ankle" in k]
    return {"ok": bool(lifts and max(lifts) > 0.05), "feet": out, "max_ankle_lift_m": max(lifts) if lifts else 0.0}


def check_foot_slip(c: dict) -> dict:
    """Horizontal travel of a foot site while it is planted.

    The reference has no contact forces, so contact has to be inferred from foot
    height. That inference is the weak point: a foot passing low during swing
    (heel-strike approach) is the same height as a planted foot but moves ~3 m/s.
    A wide height band therefore mixes swing pass-through into "slip" and
    overstates it, so several bands are reported and the strict 5 mm band is the
    headline number. Whatever the absolute value, it is measured identically for
    every reference, which is what makes references comparable.

    The policy's own slip is measured separately in eval_walk.py from contact
    normal forces, which is the physically grounded version.
    """
    feet = pelvis_and_feet(c)["feet"]
    margins_mm = (2, 5, 10, 20)
    out = {}
    for name, xyz in feet.items():
        z = xyz[:, 2]
        floor = float(z.min())
        dxy = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
        by_margin = {}
        for mm in margins_mm:
            planted = z < floor + mm / 1000.0
            pair = planted[1:] & planted[:-1]
            slip_d = float(dxy[pair].sum())
            span_s = float(planted.sum()) / c["freq"]
            by_margin[f"{mm}mm"] = {
                "planted_fraction": round(float(planted.mean()), 4),
                "slip_distance_m": round(slip_d, 4),
                "slip_speed_mps": round(slip_d / span_s, 4) if span_s else 0.0,
            }
        out[name] = {"floor_z_m": floor, "by_margin_mm": by_margin}
    return {"headline_margin_mm": 5, "definition": "z < z_min + margin, no speed gate", "feet": out}


def check_joint_jumps(c: dict) -> dict:
    """Per-frame joint change; hinge joints are reported in degrees.

    MuJoCo mjtJoint: 0=free (7 qpos), 1=ball (4 qpos), 2=slide (1 qpos),
    3=hinge (1 qpos). This model has 1 free + 72 hinge + 10 slide joints, so
    qpos is 7 + 72 + 10 = 89, matching nq.
    """
    qpos, jnt_type, names = c["qpos"], c["jnt_type"], c["joint_names"]
    dq = np.abs(np.diff(qpos, axis=0))
    offenders = []
    worst = 0.0
    col = 7  # 3 root position + 4 root quaternion
    for k, jtype in enumerate(jnt_type):
        if jtype == 0:
            continue  # free joint, handled by check_root
        if jtype == 1:
            col += 4  # ball: quaternion components do not have a scalar delta
            continue
        is_hinge = jtype == 3
        delta = float(np.degrees(dq[:, col].max())) if is_hinge else float(dq[:, col].max())
        at = int(dq[:, col].argmax()) + 1
        if delta > JOINT_JUMP_DEG_PER_FRAME:
            offenders.append(
                {
                    "joint": names[k] if k < len(names) else f"joint{k}",
                    "type": "hinge" if is_hinge else "slide",
                    "max_delta_deg_per_frame": round(delta, 2) if is_hinge else None,
                    "max_delta_m_per_frame": None if is_hinge else round(delta, 4),
                    "at_s": round(float(c["t"][at]), 2),
                }
            )
        worst = max(worst, delta)
        col += 1
    if col != qpos.shape[1]:
        raise RuntimeError(f"joint column walk ended at {col}, qpos has {qpos.shape[1]} columns")
    offenders.sort(key=lambda r: -(r["max_delta_deg_per_frame"] or 0))
    return {
        "ok": not offenders,
        "max_hinge_delta_deg_per_frame": round(worst, 2),
        "threshold_deg_per_frame": JOINT_JUMP_DEG_PER_FRAME,
        "n_offenders": len(offenders),
        "top_offenders": offenders[:8],
    }


def check_root(c: dict) -> dict:
    """Root translation must be smooth; a teleport means the fit jumped."""
    root = c["qpos"][:, :3]
    step = np.linalg.norm(np.diff(root, axis=0), axis=1)
    median = float(np.median(step))
    worst = float(step.max()) if step.size else 0.0
    worst_at = int(step.argmax()) + 1 if step.size else 0
    quat = c["qpos"][:, 3:7]
    qnorm = np.linalg.norm(quat, axis=1)

    # env.reset() zeroes the free-joint XY, so the policy always starts at the
    # world origin while the reference keeps its own absolute position. A
    # reference that does not start near the origin therefore hands the policy a
    # constant offset from its own targets for the whole episode.
    start_offset = float(np.hypot(root[0, 0], root[0, 1]))
    return {
        "ok": bool(worst <= max(median * ROOT_TELEPORT_FACTOR, 0.05)),
        "start_xy_offset_from_origin_m": round(start_offset, 4),
        "start_xy_ok": bool(start_offset <= START_XY_TOL_M),
        "root_z_min_m": float(root[:, 2].min()),
        "root_z_range_m": float(np.ptp(root[:, 2])),
        "max_step_m": worst,
        "median_step_m": median,
        "max_step_at_s": round(float(c["t"][worst_at]), 2),
        "quat_norm_min": float(qnorm.min()),
        "quat_norm_max": float(qnorm.max()),
    }


def check_root_rotation(c: dict) -> dict:
    """Per-frame pelvis orientation change, in degrees.

    Quaternions are a double cover, so consecutive poses are compared by the
    absolute dot product; that makes the measure blind to q vs -q, which is what
    we want. A crossfade that interpolates quaternions linearly instead of by
    slerp drives the blend through magnitude ~0 and renormalisation then snaps
    the direction through ~180 deg; on straight_stitched_10s that produced a
    165.8 deg/frame jump at the seam while every other frame stayed under 2 deg.
    """
    import numpy as np

    quat = c["qpos"][:, 3:7]
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    unit = quat / np.where(norm > 0, norm, 1.0)
    dot = np.abs(np.einsum("ij,ij->i", unit[:-1], unit[1:]))
    step_deg = np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))
    worst = float(step_deg.max()) if step_deg.size else 0.0
    at = int(step_deg.argmax()) + 1 if step_deg.size else 0
    return {
        "ok": worst <= ROOT_ROT_JUMP_DEG_PER_FRAME,
        "max_step_deg_per_frame": round(worst, 2),
        "median_step_deg_per_frame": round(float(np.median(step_deg)), 3),
        "p99_step_deg_per_frame": round(float(np.percentile(step_deg, 99)), 3),
        "max_step_at_s": round(float(c["t"][at]), 2),
        "threshold_deg_per_frame": ROOT_ROT_JUMP_DEG_PER_FRAME,
        "quat_norm_min": round(float(norm.min()), 6),
        "quat_norm_max": round(float(norm.max()), 6),
    }


def check_contact(c: dict) -> dict:
    """Gait events from foot-site height.

    The reference carries no contact forces, so contact is inferred the same way
    the straightness audit does it: a foot counts as down while its site sits
    within a band above its own lowest point. Reporting onsets and the support
    fraction is what separates "walking" from "sliding along in a fixed pose".
    """
    feet = pelvis_and_feet(c)["feet"]
    ankle = {k: v for k, v in feet.items() if "ankle" in k}
    down = {}
    for name, xyz in ankle.items():
        z = xyz[:, 2]
        down[name] = z < z.min() + CONTACT_BAND_M
    onsets = {k: int(np.count_nonzero(np.diff(v.astype(np.int8)) == 1)) for k, v in down.items()}
    any_down = np.zeros(len(c["t"]), bool)
    for v in down.values():
        any_down |= v
    both_down = np.ones(len(c["t"]), bool)
    for v in down.values():
        both_down &= v
    return {
        "ok": bool(min(onsets.values()) >= 2),
        "band_m": CONTACT_BAND_M,
        "foot_contact_onsets": onsets,
        "support_fraction": round(float(any_down.mean()), 4),
        "double_support_fraction": round(float(both_down.mean()), 4),
        "step_period_s": round(len(c["t"]) / c["freq"] / max(1, sum(onsets.values()) / max(1, len(onsets))), 4),
    }


def check_split_points(c: dict) -> dict:
    """split_points mark where several clips were concatenated into one cache."""
    sp = c["split_points"]
    n = c["qpos"].shape[0]
    segments = [(int(sp[i]), int(sp[i + 1])) for i in range(len(sp) - 1)] if len(sp) > 1 else [(0, n)]
    covered = segments[0][0] == 0 and segments[-1][1] == n
    return {
        "concatenated": len(segments) > 1,
        "n_segments": len(segments),
        "segment_lengths_frames": [b - a for a, b in segments],
        "covers_full_clip": bool(covered),
    }


def check_sites(c: dict) -> dict:
    names = c["site_names"]
    missing = [s for s in FOOT_SITES + ("pelvis_mimic",) if s not in names]
    return {"ok": not missing, "n_sites": len(names), "missing": missing}


def compare_with_raw(clip: str, amass_root: Path, retargeted_window: dict, c: dict) -> dict | None:
    """Did the retargeted clip keep the raw clip's straight window and speed?

    The raw clip is measured on the SMPL-H root translation; the retargeted clip
    on the pelvis site. They are different points on the body, so the absolute
    values are not expected to match exactly -- what matters is whether the long
    straight window and the walking speed survive the retargeting. A raw window
    of 11.6 s that comes out of the fit as 8 s means the reference is no longer
    usable, no matter how clean the raw data was.
    """
    raw_path = amass_root / f"{clip}.npz"
    if not raw_path.exists():
        return None
    from amass_kit_audit import analyse as analyse_raw

    with np.load(raw_path, allow_pickle=True) as d:
        trans = np.asarray(d["trans"], dtype=np.float64)
        fps = float(d["mocap_framerate"])
    raw = analyse_raw(f"{clip}.npz", trans, fps)
    rw = raw["longest_straight_window"]

    rows = [
        ("duration_s", raw["duration_s"], float(c["t"][-1])),
        ("straight_window_s", rw["duration_s"], retargeted_window["duration_s"]),
        ("window_net_m", rw["net_displacement_m"], retargeted_window["net_displacement_m"]),
        ("window_speed_mps", rw["mean_speed_mps"], retargeted_window["mean_speed_mps"]),
        ("window_straightness", rw["straightness"], retargeted_window["straightness"]),
        ("clip_mean_speed_mps", raw["mean_path_speed"], retargeted_window["clip_mean_speed_mps"]),
    ]
    table = [{"quantity": n, "raw": round(float(a), 4), "retargeted": round(float(b), 4),
              "delta": round(float(b) - float(a), 4)} for n, a, b in rows]
    return {
        "raw_path": str(raw_path),
        "raw_fps": fps,
        "raw_window_s": [rw["start_s"], rw["end_s"]],
        "retargeted_window_s": [retargeted_window["start_s"], retargeted_window["end_s"]],
        "table": table,
        "survived": {
            "straight_window_ge_10s_after": bool(retargeted_window["duration_s"] >= 10.0),
            "window_shrank_over_10pct": bool(
                retargeted_window["duration_s"] < 0.9 * rw["duration_s"]
            ),
            "speed_changed_over_10pct": bool(
                rw["mean_speed_mps"] > 0
                and abs(retargeted_window["mean_speed_mps"] - rw["mean_speed_mps"])
                > 0.1 * rw["mean_speed_mps"]
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True, help="cache-relative clip id, e.g. CMU/136/136_25_poses")
    ap.add_argument("--cache-root", default=str(Path.home() / ".musclemimic/caches/AMASS"))
    ap.add_argument("--env", default="MyoFullBody")
    ap.add_argument("--method", default="gmr")
    ap.add_argument("--out", default=None, help="JSON output path (default: reports/qc_<clip>.json)")
    ap.add_argument(
        "--compare-raw",
        nargs="?",
        const="__same__",
        default=None,
        help="also compare against the raw AMASS clip (default id: same as --clip)",
    )
    args = ap.parse_args()

    path = Path(args.cache_root).expanduser() / args.env / args.method / f"{args.clip}.npz"
    if not path.exists():
        print(f"error: retargeted cache not found: {path}", file=sys.stderr)
        return 2

    c = load_clip(path)
    dur = float(c["t"][-1])

    report = {
        "clip": args.clip,
        "cache_path": str(path),
        "frequency_hz": c["freq"],
        "n_frames": int(c["qpos"].shape[0]),
        "duration_s": dur,
        "metadata": c["metadata"][:400],
        "checks": {
            "finite": check_finite(c),
            "shape": check_shape(c),
            "sites": check_sites(c),
            "root": check_root(c),
            "root_rotation": check_root_rotation(c),
            "joint_jumps": check_joint_jumps(c),
            "foot_height": check_foot_height(c),
            "contact": check_contact(c),
            "split_points": check_split_points(c),
        },
        "window": window_metrics(c),
        "foot_slip": check_foot_slip(c),
    }
    report["checks"]["foot_slip"] = report["foot_slip"]

    w = report["window"]

    if args.compare_raw is not None:
        raw_clip = args.clip if args.compare_raw == "__same__" else args.compare_raw
        from loco_mujoco.smpl import retargeting as _R

        cmp_ = compare_with_raw(raw_clip, Path(_R.get_amass_dataset_path()), w, c)
        if cmp_ is None:
            report["raw_comparison"] = {"error": f"raw clip not found for {raw_clip}"}
        else:
            report["raw_comparison"] = cmp_
    report["verdict"] = {
        "duration_ge_10s": bool(dur >= 10.0),
        "straight_window_ge_10s": bool(w["duration_s"] >= 10.0),
        "window_speed_mps": round(w["mean_speed_mps"], 3),
        "window_speed_ge_0p6": bool(w["mean_speed_mps"] >= 0.6),
        "straightness_ge_0p99": bool(w["straightness"] >= 0.99),
        "no_numeric_anomaly": bool(
            report["checks"]["finite"]["ok"]
            and report["checks"]["shape"]["ok"]
            and report["checks"]["root"]["ok"]
            and report["checks"]["root_rotation"]["ok"]
            and report["checks"]["joint_jumps"]["ok"]
        ),
        "single_segment": bool(not report["checks"]["split_points"]["concatenated"]),
        "starts_near_origin": bool(report["checks"]["root"]["start_xy_ok"]),
    }

    _print_report(report)

    out = Path(args.out) if args.out else REPO_ROOT / "reports" / f"qc_{args.clip.replace('/', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")

    returns_ok = report["verdict"]["no_numeric_anomaly"]
    return 0 if returns_ok else 1


def _print_report(r: dict) -> None:
    print(f"clip      : {r['clip']}")
    print(f"cache     : {r['cache_path']}")
    print(f"frames    : {r['n_frames']} @ {r['frequency_hz']:.0f} Hz = {r['duration_s']:.2f} s")
    sp = r["checks"]["split_points"]
    print(f"segments  : {sp['n_segments']} {sp['segment_lengths_frames']} (concatenated={sp['concatenated']})")

    print("\n[window]")
    w = r["window"]
    print(
        f"  straight {w['start_s']:.2f}-{w['end_s']:.2f} s = {w['duration_s']:.2f} s, "
        f"net {w['net_displacement_m']:.2f} m, speed {w['mean_speed_mps']:.3f} m/s, "
        f"straightness {w['straightness']:.3f}"
    )
    prof = w["speed_profile_0p5s"]
    if prof:
        print("  per-0.5 s speed (m/s): " + " ".join(f"{p['mean']:.2f}" for p in prof))

    print("\n[checks]")
    for key in ("finite", "shape", "sites", "root", "root_rotation", "joint_jumps", "foot_height", "contact"):
        v = r["checks"][key]
        print(f"  {'PASS' if v.get('ok', True) else 'FAIL'}  {key}: {json.dumps({k: x for k, x in v.items() if k != 'ok'})}")

    print("\n[foot height]")
    for name, v in r["checks"]["foot_height"]["feet"].items():
        print(f"  {name:22s} z {v['z_min_m']:.4f}..{v['z_max_m']:.4f} m (lift {v['range_m']:.3f} m)")

    print(f"\n[foot slip while planted]  (headline band {r['foot_slip']['headline_margin_mm']} mm)")
    print(f"  {'site':22s} " + "  ".join(f"{m:>16s}" for m in ("2mm", "5mm", "10mm", "20mm")))
    for name, v in r["foot_slip"]["feet"].items():
        cells = []
        for m in ("2mm", "5mm", "10mm", "20mm"):
            e = v["by_margin_mm"][m]
            cells.append(f"{e['slip_speed_mps']:.3f}@{e['planted_fraction'] * 100:4.1f}%")
        print(f"  {name:22s} " + "  ".join(f"{c:>16s}" for c in cells))
    print("  (value@fraction: slip speed in m/s @ planted fraction; wide bands mix in swing pass-through)")

    rc = r.get("raw_comparison")
    if rc is not None:
        print("\n[raw -> retargeted]")
        if "error" in rc:
            print(f"  {rc['error']}")
        else:
            print(f"  raw window {rc['raw_window_s'][0]:.2f}-{rc['raw_window_s'][1]:.2f} s, "
                  f"retargeted window {rc['retargeted_window_s'][0]:.2f}-{rc['retargeted_window_s'][1]:.2f} s")
            print(f"  {'quantity':24s} {'raw':>10s} {'retargeted':>11s} {'delta':>9s}")
            for row in rc["table"]:
                print(f"  {row['quantity']:24s} {row['raw']:10.4f} {row['retargeted']:11.4f} {row['delta']:+9.4f}")
            for k, v in rc["survived"].items():
                print(f"  {'FLAG' if v else 'OK  '}  {k}: {v}")

    if not r["checks"]["root"]["start_xy_ok"]:
        print(
            f"\nWARNING: reference starts {r['checks']['root']['start_xy_offset_from_origin_m']:.3f} m from the origin. "
            "env.reset() zeroes the policy root XY, so every rollout would begin with that constant offset "
            "against the reference targets. Translate the clip to the origin first "
            "(scripts/rotate_reference.py does this by default)."
        )

    print("\n[verdict]")
    for k, v in r["verdict"].items():
        print(f"  {k:26s} {v}")


if __name__ == "__main__":
    raise SystemExit(main())
