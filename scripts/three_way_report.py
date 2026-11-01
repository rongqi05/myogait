#!/usr/bin/env python3
"""Report the three questions about a walking baseline separately.

They are often conflated, and they genuinely have different answers:

  A. survived_10s   -- did the policy stay upright for the full 10 s window?
                       Physics only: duration, no fall, no numeric anomaly,
                       exactly one reset.
  B. follows_ref    -- did it actually track the reference trajectory?
                       Path adherence: cross-track and along-path progress.
  C. passes_gate    -- did it pass the frozen straight-walking acceptance
                       (10 s, 0.6 m/s, 0.35 m lateral, 45 deg tilt)?

A policy that walks stably in the wrong direction scores A but not B. A policy
that tracks a slow reference perfectly might still score B but not C, because
0.6 m/s is an absolute speed gate that a 0.5 m/s reference cannot satisfy.

Thresholds for B are marked ``new`` in the output: they were introduced by this
report and are *not* part of the frozen protocol in configs/eval_walk.json. The
numbers reused from the frozen protocol (0.35 m, 0.6 m/s) are marked ``frozen``.

    .venv/bin/python scripts/three_way_report.py \
        --group stitched=runs/TW_stitched/stochastic \
        --group unstitched6m=runs/TW_unstitched6m/stochastic
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from recompute_metrics import HEAD_W, MOTION_SPEED, windowed_heading, windowed_speed  # noqa: E402

# --- thresholds -------------------------------------------------------------
# B (follow the reference). Marked as introduced here, not frozen protocol.
NEW_MAX_CROSS_TRACK_M = 0.35  # reuses the frozen lateral number, same meaning
NEW_PROGRESS_TOL = 0.25  # policy covers 75-125 % of the reference distance
NEW_MAX_MEAN_HEADING_DEG = 20.0

ORIGIN_TOL_M = 0.05  # "starts at the same place" tolerance for the paired plot


def load_seed(json_path: Path) -> dict:
    meta = json.loads(json_path.read_text())
    if isinstance(meta, list):
        meta = meta[-1]
    npz = json_path.with_suffix(".npz")
    d = np.load(npz)
    return {"meta": meta, "npz": d, "seed": int(meta["seed"]), "npz_path": str(npz)}


def path_metrics(d, straight_window: tuple[float, float] | None = None) -> dict:
    """Cross-track and along-path adherence, measured against the reference path."""
    t = np.asarray(d["time"], float)
    dt = float(np.median(np.diff(t)))
    root = np.asarray(d["root_pos"], float)[:, :2]
    ref = np.asarray(d["ref_qpos"], float)[:, :2]

    if straight_window is not None:
        lo, hi = straight_window
        sel = (t >= lo) & (t <= hi)
        if sel.sum() < 10:
            sel = np.ones(len(t), bool)
        t, root, ref = t[sel], root[sel], ref[sel]

    # Per-step reference heading and normal.
    dref = np.diff(ref, axis=0)
    step_len = np.linalg.norm(dref, axis=1)
    valid = step_len > 1e-9
    heading = np.zeros_like(dref)
    heading[valid] = dref[valid] / step_len[valid, None]
    normal = np.stack([-heading[:, 1], heading[:, 0]], axis=1)

    # Re-anchor the policy at the reference start so along/cross-track are
    # measured as tracking error, not as an absolute world offset.
    policy = root - root[0] + ref[0]
    delta = policy[1:] - ref[1:]
    cross = np.abs(np.einsum("ij,ij->i", delta, normal))
    along = np.einsum("ij,ij->i", delta, heading)

    policy_prog = float(np.einsum("ij,ij->i", np.diff(policy, axis=0), heading).sum())
    ref_prog = float(step_len[valid].sum())
    drift_dir = float(np.einsum("ij,ij->i", np.diff(policy, axis=0), normal).sum())

    hp = windowed_heading(root)
    hr = windowed_heading(ref)
    vp = windowed_speed(root, dt)
    vr = windowed_speed(ref, dt)
    both = (vp > MOTION_SPEED) & (vr > MOTION_SPEED) & np.isfinite(hp) & np.isfinite(hr)
    herr = np.abs((hp[both] - hr[both] + 180.0) % 360.0 - 180.0) if both.any() else np.array([])

    return {
        "window_s": [float(t[0]), float(t[-1])],
        "max_cross_track_m": float(cross.max()) if cross.size else float("nan"),
        "mean_cross_track_m": float(cross.mean()) if cross.size else float("nan"),
        "max_along_track_m": float(np.abs(along).max()) if along.size else float("nan"),
        "final_along_track_m": float(along[-1]) if along.size else float("nan"),
        "policy_progress_m": policy_prog,
        "reference_progress_m": ref_prog,
        "progress_ratio": policy_prog / ref_prog if ref_prog > 0 else float("nan"),
        "net_cross_drift_m": drift_dir,
        "mean_heading_error_deg": float(herr.mean()) if herr.size else float("nan"),
        "max_heading_error_deg": float(herr.max()) if herr.size else float("nan"),
        "heading_frames_used": int(both.sum()),
        "policy_speed_mps": float(vp[both].mean()) if both.any() else float("nan"),
        "reference_speed_mps": float(vr[both].mean()) if both.any() else float("nan"),
    }


def straight_window_of(d) -> tuple[float, float]:
    """Longest straight window of the reference, same definition as the audits."""
    t = np.asarray(d["time"], float)
    ref = np.asarray(d["ref_qpos"], float)[:, :2]
    heading = windowed_heading(ref)
    tol = 20.0
    n = len(heading)
    best = (0.0, 0.0)
    i = 0
    while i < n:
        if not np.isfinite(heading[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and np.isfinite(heading[j + 1]):
            seg = heading[i : j + 2]
            rel = (seg - heading[i] + 180.0) % 360.0 - 180.0
            if np.max(np.abs(rel - rel.mean())) > tol:
                break
            j += 1
        if t[j] - t[i] > best[1] - best[0]:
            best = (float(t[i]), float(t[j]))
        i = j + 1
    return best


def verdict(meta: dict, pm: dict) -> dict:
    crit = meta.get("criteria", {})
    survived = bool(
        meta.get("duration_s", 0.0) >= 10.0 - 1e-6
        and meta.get("end_condition") != "physical_fall"
        and crit.get("no_numeric_anomaly", False)
        and crit.get("no_reset_or_state_override", False)
        and int(meta.get("reset_count", 0)) == 1
    )
    follows = bool(
        np.isfinite(pm["max_cross_track_m"])
        and pm["max_cross_track_m"] <= NEW_MAX_CROSS_TRACK_M
        and np.isfinite(pm["progress_ratio"])
        and abs(pm["progress_ratio"] - 1.0) <= NEW_PROGRESS_TOL
        and np.isfinite(pm["mean_heading_error_deg"])
        and pm["mean_heading_error_deg"] <= NEW_MAX_MEAN_HEADING_DEG
    )
    passes = bool(meta.get("success", False))
    return {"survived_10s": survived, "follows_ref": follows, "passes_gate": passes}


def run_group(name: str, directory: Path, make_figures: bool) -> dict:
    files = sorted(glob.glob(str(directory / "*_seed*.npz")), key=lambda f: int(re.search(r"seed(\d+)", f).group(1)))
    if not files:
        raise SystemExit(f"no seed npz files in {directory}")
    rows = []
    for f in files:
        s = load_seed(Path(f).with_suffix(".json"))
        sw = straight_window_of(s["npz"])
        pm_full = path_metrics(s["npz"])
        pm_straight = path_metrics(s["npz"], straight_window=sw)
        v = verdict(s["meta"], pm_full)
        v_straight = verdict(s["meta"], pm_straight)
        rows.append(
            {
                "seed": s["seed"],
                "end_condition": s["meta"].get("end_condition"),
                "duration_s": s["meta"].get("duration_s"),
                "reset_count": s["meta"].get("reset_count"),
                "forward_distance_m": s["meta"].get("forward_distance"),
                "avg_forward_speed_mps": s["meta"].get("avg_forward_speed"),
                "max_lateral_deviation_m": s["meta"].get("max_lateral_deviation"),
                "max_tilt_pelvis_deg": s["meta"].get("max_tilt_pelvis_deg"),
                "min_root_height_m": s["meta"].get("min_root_height"),
                "foot_slip_m": s["meta"].get("foot_slip_distance"),
                "reference_straight_window_s": [round(sw[0], 2), round(sw[1], 2)],
                "full_window": pm_full,
                "straight_window": pm_straight,
                **v,
                "follows_ref_straight_window": v_straight["follows_ref"],
            }
        )

    def count(key: str) -> int:
        return sum(1 for r in rows if r[key])

    summary = {
        "group": name,
        "dir": str(directory),
        "n": len(rows),
        "survived_10s": count("survived_10s"),
        "follows_ref": count("follows_ref"),
        "follows_ref_straight_window": count("follows_ref_straight_window"),
        "passes_gate": count("passes_gate"),
        "reference_speed_mps": float(np.nanmean([r["full_window"]["reference_speed_mps"] for r in rows])),
        "policy_speed_mps": float(np.nanmean([r["full_window"]["policy_speed_mps"] for r in rows])),
        "mean_progress_ratio": float(np.nanmean([r["full_window"]["progress_ratio"] for r in rows])),
        "mean_max_cross_track_m": float(np.nanmean([r["full_window"]["max_cross_track_m"] for r in rows])),
        "mean_heading_error_deg": float(np.nanmean([r["full_window"]["mean_heading_error_deg"] for r in rows])),
        "reference_straight_window_s": rows[0]["reference_straight_window_s"],
    }

    if make_figures:
        summary["figure_overlay"] = str(plot_overlay(name, directory, rows))
        summary["figure_grid"] = str(plot_grid(name, directory, rows))
    return {"summary": summary, "per_seed": rows}


def _episode_paths(npz) -> tuple[np.ndarray, np.ndarray]:
    root = np.asarray(npz["root_pos"], float)[:, :2]
    ref = np.asarray(npz["ref_qpos"], float)[:, :2]
    return root, ref


def plot_overlay(name: str, directory: Path, rows: list[dict]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    first = True
    for r in rows:
        d = np.load(Path(r_seed_npz(directory, r["seed"])))
        root, ref = _episode_paths(d)
        if first:
            ax.plot(ref[:, 0], ref[:, 1], "k--", lw=2.0, label="reference")
            first = False
        ok = r["passes_gate"]
        ax.plot(root[:, 0], root[:, 1], color="tab:green" if ok else "tab:red", lw=0.9, alpha=0.75)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{name}: policy pelvis (top-down), {len(rows)} seeds\ngreen = passes frozen gate, red = fails")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    out = REPO_ROOT / "reports" / "figures" / f"threeway_{name}_overlay.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_grid(name: str, directory: Path, rows: list[dict]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, rows):
        d = np.load(Path(r_seed_npz(directory, r["seed"])))
        root, ref = _episode_paths(d)
        ax.plot(ref[:, 0], ref[:, 1], "k--", lw=1.2)
        ax.plot(root[:, 0], root[:, 1], "tab:blue", lw=1.0)
        ax.set_aspect("equal", adjustable="datalim")
        flags = "".join(k[0].upper() if r[k] else "-" for k in ("survived_10s", "follows_ref", "passes_gate"))
        ax.set_title(f"seed {r['seed']} [{flags}]\nxtrk {r['full_window']['max_cross_track_m']:.2f} m", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{name}: per-episode top-down (dashed = reference)  [S=survived, F=follows, G=gate]", fontsize=10)
    out = REPO_ROOT / "reports" / "figures" / f"threeway_{name}_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def r_seed_npz(directory: Path, seed: int) -> str:
    hits = glob.glob(str(directory / f"*seed{seed}.npz"))
    if not hits:
        raise FileNotFoundError(f"seed {seed} npz not found in {directory}")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append", required=True, help="NAME=DIR (repeatable)")
    ap.add_argument("--out", default="reports/three_way.json")
    ap.add_argument("--md", default="reports/three_way.md")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    result = {
        "definitions": {
            "survived_10s": "duration >= 10 s, end_condition != physical_fall, no numeric anomaly, exactly one reset",
            "follows_ref": {
                "max_cross_track_m": f"<= {NEW_MAX_CROSS_TRACK_M} (frozen number, reused)",
                "progress_ratio": f"within 1 +- {NEW_PROGRESS_TOL} (new)",
                "mean_heading_error_deg": f"<= {NEW_MAX_MEAN_HEADING_DEG} (new)",
            },
            "passes_gate": "the frozen acceptance criteria in configs/eval_walk.json",
        },
        "groups": {},
    }

    for spec in args.group:
        name, d = spec.split("=", 1)
        print(f"\n=== {name} ===")
        result["groups"][name] = run_group(name, Path(d), not args.no_figures)

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float))

    lines = _render_markdown(result)
    md = REPO_ROOT / args.md
    md.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}\nwrote {md}")
    return 0


def _render_markdown(result: dict) -> list[str]:
    out = ["# Three separate answers: survived 10 s / follows reference / passes frozen gate", ""]
    out.append("| group | n | A survived 10 s | B follows ref | B' follows ref on straight window | C passes gate |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for name, g in result["groups"].items():
        s = g["summary"]
        out.append(
            f"| {name} | {s['n']} | {s['survived_10s']}/{s['n']} | {s['follows_ref']}/{s['n']} | "
            f"{s['follows_ref_straight_window']}/{s['n']} | {s['passes_gate']}/{s['n']} |"
        )
    out.append("")
    for name, g in result["groups"].items():
        s = g["summary"]
        out.append(f"## {name}")
        out.append("")
        out.append(
            f"reference {s['reference_speed_mps']:.3f} m/s, policy {s['policy_speed_mps']:.3f} m/s, "
            f"progress ratio {s['mean_progress_ratio']:.3f}, "
            f"mean max cross-track {s['mean_max_cross_track_m']:.3f} m, "
            f"mean heading error {s['mean_heading_error_deg']:.1f} deg, "
            f"reference straight window {s['reference_straight_window_s']}"
        )
        out.append("")
        out.append(
            "| seed | end | dur s | fwd m | v m/s | lat m | cross m | progress | hdg deg | tilt deg | A | B | C |"
        )
        out.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|")
        for r in g["per_seed"]:
            fw = r["full_window"]
            out.append(
                f"| {r['seed']} | {r['end_condition']} | {r['duration_s']:.2f} | {r['forward_distance_m']:.2f} | "
                f"{r['avg_forward_speed_mps']:.3f} | {r['max_lateral_deviation_m']:.3f} | "
                f"{fw['max_cross_track_m']:.3f} | {fw['progress_ratio']:.3f} | "
                f"{fw['mean_heading_error_deg']:.1f} | {r['max_tilt_pelvis_deg']:.1f} | "
                f"{'Y' if r['survived_10s'] else '.'} | {'Y' if r['follows_ref'] else '.'} | "
                f"{'Y' if r['passes_gate'] else '.'} |"
            )
        out.append("")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
