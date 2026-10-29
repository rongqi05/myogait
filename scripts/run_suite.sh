#!/usr/bin/env bash
# Re-run the complete acceptance suite (65 rollouts, single process each, sequential).
# Usage: bash scripts/run_suite.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="external/MuscleMimic:scripts:${PYTHONPATH:-}"
PY=.venv/bin/python
CKPT=artifacts/checkpoints/checkpoint_12500
WALK=KIT/314/walking_medium09_poses
CIRCLE=KIT/4/WalkInCounterClockwiseCircle04_poses

echo "[1/6] pilot stochastic (3 seeds)"
$PY scripts/eval_walk.py --checkpoint "$CKPT" --motion_path "$WALK" \
  --mode stochastic --seeds 0 1 2 --out_dir runs/C_pilot

echo "[2/6] pilot deterministic (3 seeds)"
$PY scripts/eval_walk.py --checkpoint "$CKPT" --motion_path "$WALK" \
  --mode deterministic --seeds 0 1 2 --out_dir runs/C_pilot

echo "[3/6] acceptance stochastic (20 seeds)"
$PY scripts/eval_walk.py --checkpoint "$CKPT" --motion_path "$WALK" \
  --mode stochastic --seeds $(seq 0 19) --out_dir runs/C_acceptance

echo "[4/6] acceptance deterministic (20 seeds)"
$PY scripts/eval_walk.py --checkpoint "$CKPT" --motion_path "$WALK" \
  --mode deterministic --seeds $(seq 0 19) --out_dir runs/C_acceptance_det

echo "[5/6] acceptance circle clip (20 seeds)"
$PY scripts/eval_walk.py --checkpoint "$CKPT" --motion_path "$CIRCLE" \
  --mode stochastic --seeds $(seq 0 19) --out_dir runs/C_acceptance_circle

echo "[6/6] video (1 seed)"
MUJOCO_GL=egl $PY scripts/eval_walk.py --checkpoint "$CKPT" --motion_path "$WALK" \
  --mode stochastic --seeds 0 --video --video_every 5 --out_dir runs/C_acceptance_video

echo "done"
