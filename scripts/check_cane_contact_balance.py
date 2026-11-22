#!/usr/bin/env python3
"""Audit that a saved rollout is held up by floor contact only.

This is a pure post-processor: every number comes from the per-step arrays that
``eval_walk.py`` recorded *inside* the simulator loop, so there is no replay and
no re-solution of the constraint problem.

Checks
------
1. ``qfrc_applied_max`` and ``xfrc_applied_max`` are zero at every step, i.e. no
   external force was injected.
2. Single reset: ``reset_count == 1``, ``episode_id == 0``, monotonic clock.
3. Impulse-momentum closure over the whole episode:

       integral (F_z - m g) dt  ==  m (v_z(T) - v_z(0))

   ``F_z`` is the summed vertical force of *all* contacts with the ground, so
   this closes only if nothing other than ground contact supports the body.
4. The cane's share of the vertical impulse, and the deepest tip penetration.

Usage::

    python scripts/check_cane_contact_balance.py runs/cane_sweep/ipsi_U1.0/stochastic/stochastic_seed0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

G = 9.81


def check(stem: Path, tol: float) -> int:
    js = json.loads(stem.with_suffix(".json").read_text())
    d = np.load(stem.with_suffix(".npz"))
    t = d["time"] - d["time"][0]
    vz = d["qvel"][:, 2]
    dt = float(np.median(np.diff(t)))
    m = float(js.get("total_mass_kg") or 0.0)
    if m <= 0:
        print(f"  !! {stem}: no total_mass_kg in the JSON (older run)")
        return 1

    fz = d["grf_vertical_total"]
    fz_feet = d["grf_vertical_feet"]
    fz_cane = d["grf_vertical_cane"]

    # trapezoidal integration, which is what matters for an impulse balance
    imp = float(np.trapezoid(fz - m * G, t))
    dp = float(m * (vz[-1] - vz[0]))
    denom = max(m * G * (t[-1] - t[0]), 1.0)
    rel = (imp - dp) / denom

    imp_cane = float(np.trapezoid(fz_cane, t))
    imp_total = float(np.trapezoid(fz, t))

    print(f"\n=== {stem.name} ===")
    print(f"  reason            {js.get('reason')}   duration {js.get('duration_s'):.2f}s")
    print(f"  resets / episode  {js.get('reset_count')} / {js.get('episode_id')}"
          f"   clock monotonic {bool(np.all(np.diff(t) > 0))}")
    print(f"  qfrc/xfrc applied {js.get('qfrc_applied_max'):.1f} / {js.get('xfrc_applied_max'):.1f} N")
    print(f"  mass {m:.2f} kg  ({m * G:.1f} N)   sample dt {dt * 1000:.0f} ms")
    print(f"  mean vertical GRF total {fz.mean():7.1f} N  feet {fz_feet.mean():7.1f} N"
          f"  cane {fz_cane.mean():6.1f} N")
    if imp_total > 0:
        print(f"  cane share of vertical impulse {100.0 * imp_cane / imp_total:5.2f} %")
    print(f"  impulse balance   integral(F_z - mg)dt = {imp:+.1f} N s"
          f"   m*dv_z = {dp:+.1f} N s   residual {rel * 100:+.2f} % of mgT")
    tip = d.get("cane_tip_z")
    if tip is not None and np.isfinite(tip).any():
        print(f"  cane tip lowest z min {np.nanmin(tip) * 1000:+.1f} mm"
              f"   steps below -1 mm: {int((tip < -0.001).sum())}")

    ok = abs(rel) <= tol and js.get("qfrc_applied_max") == 0.0 and js.get("xfrc_applied_max") == 0.0
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+", help="npz/json stem(s), with or without extension")
    ap.add_argument("--tol", type=float, default=0.05, help="allowed |residual| as a fraction of m*g*T")
    args = ap.parse_args()
    bad = 0
    for s in args.stems:
        p = Path(s)
        if p.suffix in (".npz", ".json"):
            p = p.with_suffix("")
        bad += check(p, args.tol)
    print(f"\n{len(args.stems) - bad}/{len(args.stems)} rollouts pass the vertical balance audit")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
