#!/usr/bin/env bash
# Frozen acceptance for the straight walking baseline.
#   usage: CKPT=<checkpoint_dir> bash scripts/acceptance_straight.sh [SECONDS]
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="external/MuscleMimic:scripts"
CKPT="${CKPT:?set CKPT to a checkpoint directory}"
SECONDS_="${1:-10}"
OUT="${OUT:-runs/A_straight_${SECONDS_}s}"
.venv/bin/python scripts/eval_walk.py \
  --checkpoint "$CKPT" \
  --motion_path KIT/3/straight_stitched_10s \
  --mode stochastic --seeds $(seq 0 19) --seconds "$SECONDS_" \
  --out_dir "$OUT"
.venv/bin/python scripts/make_table.py "$OUT/stochastic" --out "reports/table_accept_${SECONDS_}s.md"
