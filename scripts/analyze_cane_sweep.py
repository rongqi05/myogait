#!/usr/bin/env python3
"""Aggregate the cane sweep into per-condition and paired tables.

Reads ``runs/cane_sweep/<cond>/stochastic/stochastic_seed*.json`` written by
``scripts/run_cane_sweep.sh`` and prints:

1. a per-condition summary (completion, speed, lateral drift, trunk tilt,
   cane vertical support, foot slip);
2. the ipsilateral vs contralateral pair on matched seeds;
3. every level against the no-cane control on matched seeds.

Nothing is recomputed from video: every number comes from the per-step arrays
saved by ``eval_walk.py`` (contact forces, root pose, cane tip force).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

FAIL_REASONS = {"physical_fall", "numeric_anomaly", "reset_or_state_override"}

COLS = [
    ("n", "采样数"),
    ("survived", "未跌倒"),
    ("success", "全部判据通过"),
    ("dur_med", "存活中位 s"),
    ("v_med", "前进速度中位 m/s"),
    ("lat_med", "最大侧偏中位 m"),
    ("tilt_med", "最大躯干倾角中位 °"),
    ("slip_med", "每米足滑中位 m"),
    ("cane_sup_pct", "竖直支撑 均值 %BW"),
    ("cane_loaded_pct", "着地时竖直支撑 %BW"),
    ("cane_contact", "着地时间占比"),
    ("cane_peak_N", "峰值竖直力 N"),
    ("cane_pen_frac", "杖端穿透时间占比"),
]


def load(root: Path, cond: str) -> list[dict]:
    d = root / cond / "stochastic"
    out = []
    for f in sorted(d.glob("stochastic_seed*.json")):
        if f.name.startswith("summary"):
            continue
        out.append(json.loads(f.read_text()))
    return out


def _med(xs: list[float]) -> float:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def slip_per_m(r: dict) -> float:
    s = r.get("slip_per_meter") or {}
    vals = [v for v in (s.get("left"), s.get("right")) if v is not None and v == v]
    return max(vals) if vals else float("nan")


def summarise(rows: list[dict]) -> dict:
    cm = [r.get("cane_metrics") or {} for r in rows]
    has_cane = bool(rows and (rows[0].get("cane") or {}).get("enabled"))
    return {
        "n": len(rows),
        "survived": sum(1 for r in rows if r.get("reason") not in FAIL_REASONS),
        "success": sum(1 for r in rows if r.get("success")),
        "dur_med": _med([r.get("duration_s") for r in rows]),
        "v_med": _med([r.get("avg_forward_speed") for r in rows]),
        "lat_med": _med([r.get("max_lateral_deviation") for r in rows]),
        "tilt_med": _med([max(r.get("max_tilt_pelvis_deg") or 0.0, r.get("max_tilt_torso_deg") or 0.0) for r in rows]),
        "slip_med": _med([slip_per_m(r) for r in rows]),
        "cane_sup_pct": _mean([m.get("cane_support_mean_pct_bw") for m in cm]) if has_cane else float("nan"),
        "cane_loaded_pct": _mean([m.get("cane_support_mean_loaded_pct_bw") for m in cm]) if has_cane else float("nan"),
        "cane_contact": _mean([m.get("cane_contact_fraction") for m in cm]) if has_cane else float("nan"),
        "cane_peak_N": _med([m.get("cane_support_peak_N") for m in cm]) if has_cane else float("nan"),
        "cane_pen_frac": _mean([m.get("cane_tip_penetration_fraction") for m in cm]) if has_cane else float("nan"),
    }


def fmt(v) -> str:
    if isinstance(v, float):
        if v != v:
            return "—"
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"
    return str(v)


def print_table(title: str, table: dict[str, dict]) -> None:
    print(f"\n### {title}\n")
    head = "| 条件 | " + " | ".join(lbl for _, lbl in COLS) + " |"
    sep = "|" + "---|" * (len(COLS) + 1)
    print(head)
    print(sep)
    for cond, row in table.items():
        print(f"| {cond} | " + " | ".join(fmt(row[k]) for k, _ in COLS) + " |")


def paired_delta(a: list[dict], b: list[dict], key: str) -> tuple[float, float]:
    """mean and median of a[key] over seeds present in both, computed as a-b."""
    ia = {r["seed"]: r for r in a}
    ib = {r["seed"]: r for r in b}
    common = sorted(set(ia) & set(ib))

    def val(r, k):
        if k == "support":
            return ((r.get("cane_metrics") or {}).get("cane_support_mean_pct_bw"))
        if k == "lat":
            return r.get("max_lateral_deviation")
        if k == "tilt":
            return max(r.get("max_tilt_pelvis_deg") or 0.0, r.get("max_tilt_torso_deg") or 0.0)
        if k == "dur":
            return r.get("duration_s")
        return r.get(k)

    d = []
    for s in common:
        x, y = val(ia[s], key), val(ib[s], key)
        if x is None or y is None or x != x or y != y:
            continue
        d.append(x - y)
    return (_mean(d), _med(d))


def mcnemar(a: list[dict], b: list[dict]) -> tuple[int, int, float]:
    """Exact McNemar test on paired survival of a vs b.

    Returns (b_only, a_only, two-sided p). ``b_only`` counts seeds where b
    survived and a did not; ``a_only`` the reverse.
    """
    from scipy.stats import binomtest

    ia = {r["seed"]: r.get("reason") not in FAIL_REASONS for r in a}
    ib = {r["seed"]: r.get("reason") not in FAIL_REASONS for r in b}
    common = sorted(set(ia) & set(ib))
    b_only = sum(1 for s in common if ib[s] and not ia[s])
    a_only = sum(1 for s in common if ia[s] and not ib[s])
    n = b_only + a_only
    p = 1.0 if n == 0 else binomtest(b_only, n, 0.5).pvalue
    return b_only, a_only, float(p)


def boot_ci(a: list[dict], b: list[dict], key: str, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap CI for mean(a[key] - b[key]) over matched seeds."""
    rng = np.random.default_rng(seed)

    def val(r, k):
        if k == "support":
            return (r.get("cane_metrics") or {}).get("cane_support_mean_pct_bw")
        if k == "lat":
            return r.get("max_lateral_deviation")
        if k == "tilt":
            return max(r.get("max_tilt_pelvis_deg") or 0.0, r.get("max_tilt_torso_deg") or 0.0)
        return r.get(k)

    ia = {r["seed"]: r for r in a}
    ib = {r["seed"]: r for r in b}
    d = []
    for s in sorted(set(ia) & set(ib)):
        x, y = val(ia[s], key), val(ib[s], key)
        if x is None or y is None or x != x or y != y:
            continue
        d.append(x - y)
    if not d:
        return float("nan"), float("nan"), float("nan")
    d = np.asarray(d)
    m = float(d.mean())
    boots = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)])
    return m, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/cane_sweep")
    ap.add_argument("--levels", default="1.0,0.9,0.8,0.7")
    ap.add_argument("--out_prefix", default="runs/cane_sweep_results")
    args = ap.parse_args()

    root = Path(args.root)
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]

    conds: list[str] = []
    for u in levels:
        conds += [f"nocane_U{u}", f"ipsi_U{u}", f"contra_U{u}"]

    data = {c: load(root, c) for c in conds}
    missing = [c for c in conds if not data[c]]
    if missing:
        print(f"[warn] no results for: {', '.join(missing)}")

    table = {c: summarise(data[c]) for c in conds if data[c]}
    print_table("各条件汇总（拐杖固定左下肢 F0，左上肢 F0 为行内 U 值）", table)

    # ---- paired: ipsi vs contra ------------------------------------------
    print("\n### 健侧持杖 vs 患侧持杖（配对种子，contra − ipsi）\n")
    print("| U | 竖直支撑 Δ%BW | 存活 Δ | 速度 Δm/s | 侧偏 Δm | 倾角 Δ° |")
    print("|---|---|---|---|---|---|")
    for u in levels:
        a, b = data.get(f"ipsi_U{u}"), data.get(f"contra_U{u}")
        if not a or not b:
            continue
        sup, _ = paired_delta(b, a, "support")
        lat, _ = paired_delta(b, a, "lat")
        tilt, _ = paired_delta(b, a, "tilt")
        v, _ = paired_delta(b, a, "avg_forward_speed")
        ds = summarise(b)["survived"] - summarise(a)["survived"]
        print(f"| {u} | {sup:+.2f} | {ds:+d} | {v:+.3f} | {lat:+.3f} | {tilt:+.2f} |")

    print("\n### 配对统计：健侧（contra）相对患侧（ipsi）\n")
    print("| U | 存活 contra vs ipsi | McNemar 仅 contra 存活 / 仅 ipsi 存活 | 精确 p | 躯干倾角 Δ° [95%CI] | 速度 Δm/s |")
    print("|---|---|---|---|---|---|")
    for u in levels:
        a, b = data.get(f"ipsi_U{u}"), data.get(f"contra_U{u}")
        if not a or not b:
            continue
        sa, sb = summarise(a)["survived"], summarise(b)["survived"]
        b_only, a_only, pv = mcnemar(a, b)
        tm, tlo, thi = boot_ci(b, a, "tilt")
        vm, vlo, vhi = boot_ci(b, a, "avg_forward_speed")
        print(f"| {u} | {sb}/{len(b)} vs {sa}/{len(a)} | {b_only} / {a_only} | {pv:.3f} | "
              f"{tm:+.1f} [{tlo:+.1f}, {thi:+.1f}] | {vm:+.3f} [{vlo:+.3f}, {vhi:+.3f}] |")

    # ---- paired: cane vs no cane -----------------------------------------
    print("\n### 持杖 vs 无拐杖（配对种子，cane − nocane）\n")
    print("| 条件 | 速度 Δm/s | 侧偏 Δm | 倾角 Δ° | 存活 Δ |")
    print("|---|---|---|---|---|")
    for u in levels:
        base = data.get(f"nocane_U{u}")
        if not base:
            continue
        for side in ("ipsi", "contra"):
            c = data.get(f"{side}_U{u}")
            if not c:
                continue
            v, _ = paired_delta(c, base, "avg_forward_speed")
            lat, _ = paired_delta(c, base, "lat")
            tilt, _ = paired_delta(c, base, "tilt")
            ds = summarise(c)["survived"] - summarise(base)["survived"]
            print(f"| {side}_U{u} | {v:+.3f} | {lat:+.3f} | {tilt:+.2f} | {ds:+d} |")

    # ---- persist ---------------------------------------------------------
    out_csv = Path(f"{args.out_prefix}_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["condition"] + [k for k, _ in COLS])
        w.writeheader()
        for c, row in table.items():
            w.writerow({"condition": c, **row})

    per_seed = Path(f"{args.out_prefix}_per_seed.csv")
    with per_seed.open("w", newline="") as fh:
        w = None
        for c in conds:
            for r in data[c]:
                cm = r.get("cane_metrics") or {}
                rec = {
                    "condition": c,
                    "seed": r.get("seed"),
                    "reason": r.get("reason"),
                    "success": r.get("success"),
                    "duration_s": r.get("duration_s"),
                    "avg_forward_speed": r.get("avg_forward_speed"),
                    "max_lateral_deviation": r.get("max_lateral_deviation"),
                    "max_tilt_pelvis_deg": r.get("max_tilt_pelvis_deg"),
                    "max_tilt_torso_deg": r.get("max_tilt_torso_deg"),
                    "min_root_height": r.get("min_root_height"),
                    "slip_per_meter": slip_per_m(r),
                    "cane_support_mean_pct_bw": cm.get("cane_support_mean_pct_bw"),
                    "cane_support_mean_loaded_pct_bw": cm.get("cane_support_mean_loaded_pct_bw"),
                    "cane_contact_fraction": cm.get("cane_contact_fraction"),
                    "cane_support_peak_N": cm.get("cane_support_peak_N"),
                    "cane_tip_penetration_fraction": cm.get("cane_tip_penetration_fraction"),
                }
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=list(rec))
                    w.writeheader()
                w.writerow(rec)
    print(f"\nwrote {out_csv}\nwrote {per_seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
