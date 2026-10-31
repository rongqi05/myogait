#!/usr/bin/env bash
# Time a fixed number of PPO updates and record peak GPU/system memory.
# usage: NENV=256 UPDATES=4 bash scripts/bench_train_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="external/MuscleMimic:${PYTHONPATH:-}"
NENV="${NENV:-256}"; UPDATES="${UPDATES:-4}"; STEPS="${STEPS:-20}"
TOTAL=$(( NENV * STEPS * UPDATES ))
OUT="/tmp/bench_${NENV}"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > /tmp/bench_vram_${NENV}.txt &
MON=$!
/usr/bin/time -v .venv-train/bin/python external/MuscleMimic/fullbody/experiment.py --config-name=conf_fullbody \
  wandb.mode=disabled \
  experiment.env_params.num_envs="$NENV" experiment.num_envs="$NENV" \
  experiment.ppo_config.num_steps="$STEPS" experiment.ppo_config.num_minibatches=8 \
  experiment.total_timesteps="$TOTAL" \
  +experiment.task_factory.params.amass_dataset_conf.rel_dataset_path="[KIT/3/straight_stitched_10s]" \
  experiment.task_factory.params.amass_dataset_conf.dataset_group=null \
  +experiment.task_factory.params.amass_dataset_conf.retargeting_method=gmr \
  experiment.validation.active=false experiment.validation.num=1 \
  "hydra.run.dir=$OUT" > "${OUT}.log" 2>&1 || true
kill $MON 2>/dev/null || true
echo "=== peak VRAM MiB ==="; sort -n /tmp/bench_vram_${NENV}.txt | tail -1
echo "=== peak RSS ==="; grep "Maximum resident" "${OUT}.log" || true
echo "=== wall time ==="; grep "Elapsed (wall clock)" "${OUT}.log" || true
