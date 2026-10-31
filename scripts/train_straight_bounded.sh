#!/usr/bin/env bash
# Bounded straight-walking training run fitted to an 8 GB laptop GPU.
# Measured: 512 envs -> 6.99 GB peak VRAM, ~2.5k env-steps/s, ~5.0 GB RSS.
#   NENV=512 STEPS=20 MINUTES=25 bash scripts/train_straight_bounded.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="external/MuscleMimic:${PYTHONPATH:-}"
# Do NOT raise XLA's preallocation fraction: warp allocates outside the XLA
# pool, and 0.90 starved it (measured OOM at 512 envs). Default 0.75 fits.

NENV="${NENV:-512}"; STEPS="${STEPS:-20}"; MINUTES="${MINUTES:-25}"
UPDATES="${UPDATES:-$(( MINUTES * 60 / 4 ))}"          # ~4 s/update measured
TOTAL=$(( NENV * STEPS * UPDATES ))
RUN="runs/T_straight_${NENV}x${STEPS}_${UPDATES}u"
mkdir -p "$RUN"

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 2000 > "$RUN/vram.txt" &
MON=$!
/usr/bin/time -v .venv-train/bin/python external/MuscleMimic/fullbody/experiment.py --config-name=conf_fullbody \
  wandb.mode=disabled \
  experiment.env_params.num_envs="$NENV" experiment.num_envs="$NENV" \
  experiment.ppo_config.num_steps="$STEPS" experiment.ppo_config.num_minibatches=8 \
  experiment.actor_hidden_layers="[1024,1024,1024,1024,1024]" \
  experiment.critic_hidden_layers="[1024,1024,1024,1024,1024]" \
  experiment.total_timesteps="$TOTAL" experiment.validation.num=2 \
  +experiment.task_factory.params.amass_dataset_conf.rel_dataset_path="[KIT/3/straight_stitched_10s]" \
  experiment.task_factory.params.amass_dataset_conf.dataset_group=null \
  +experiment.task_factory.params.amass_dataset_conf.retargeting_method=gmr \
  experiment.validation.active=false \
  experiment.checkpoint_root="$PWD/$RUN/checkpoints" \
  "hydra.run.dir=$PWD/$RUN/hydra" > "$RUN/train.log" 2>&1 || true
kill $MON 2>/dev/null || true
echo "run dir: $RUN"
echo "peak VRAM MiB: $(sort -n "$RUN/vram.txt" | tail -1)"
grep -E "Maximum resident|Elapsed \(wall" "$RUN/train.log" || true
