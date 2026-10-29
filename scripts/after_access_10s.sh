#!/usr/bin/env bash
# After the user has accepted the AMASS terms for amathislab/musclemimic-retargeted:
# download the 14.9 s straight-walk clip, verify its duration/speed, then run the
# frozen 20-seed acceptance protocol on it.
#
#   bash scripts/after_access_10s.sh
#
# Deliberately refuses to run the acceptance if the reference itself cannot satisfy
# the 10 s / 0.6 m/s preconditions; in that case report the numbers instead of
# relabelling a shorter clip as success.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="external/MuscleMimic:scripts:${PYTHONPATH:-}"
PY=.venv/bin/python
REL="MyoFullBody/gmr/KIT/3/walk_6m_straight_line06_poses.npz"
MOTION="KIT/3/walk_6m_straight_line06_poses"
CACHE="$HOME/.musclemimic/caches/AMASS/MyoFullBody/gmr/KIT/3"

echo "[1/4] download $REL"
$PY - <<PYEOF
from huggingface_hub import snapshot_download
p = snapshot_download("amathislab/musclemimic-retargeted", repo_type="dataset",
                      local_dir="artifacts/motions/retargeted",
                      allow_patterns=["$REL"])
print(p)
PYEOF

echo "[2/4] install into the gmr cache"
mkdir -p "$CACHE"
cp "artifacts/motions/retargeted/$REL" "$CACHE/"

echo "[3/4] reference statistics"
$PY scripts/motion_stats.py --out reports/motion_stats_after_access.json

echo "[4/4] precondition check + 20-seed acceptance"
$PY - "$MOTION" <<'PYEOF'
import json, sys
target = sys.argv[1]
stats = json.load(open("reports/motion_stats_after_access.json"))
rec = next(r for r in stats if target in r["file"])
print(json.dumps(rec, indent=2))
ok = rec["duration_s"] >= 10.0 and rec["mean_speed_whole_clip"] >= 0.6
print(f"duration_s={rec['duration_s']:.2f}  whole-clip speed={rec['mean_speed_whole_clip']:.3f}")
if not ok:
    print("REFERENCE DOES NOT MEET THE 10 s / 0.6 m/s PRECONDITION.")
    print("Do NOT run the acceptance as a success test; keep the decision as 'insufficient data'")
    print("and report these numbers. Use walking_window_forward_speed only as a secondary metric.")
    sys.exit(2)
PYEOF

$PY scripts/eval_walk.py \
  --checkpoint artifacts/checkpoints/checkpoint_12500 \
  --motion_path "$MOTION" \
  --mode stochastic --seeds $(seq 0 19) --out_dir runs/C_acceptance_long

$PY scripts/make_table.py runs/C_acceptance_long/stochastic --out reports/table_long.md
echo "done. summary: runs/C_acceptance_long/stochastic/summary.csv"
