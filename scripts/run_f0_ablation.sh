#!/usr/bin/env bash
# F0 单侧减力定位实验：上肢 vs 下肢（20 种子 × 5 条件，顺序执行，单进程控制内存）。
#
# 固定配置（与 O_muscle_force_scaling.md 相同）：
#   checkpoint  = artifacts/checkpoints/checkpoint_12500 (官方 mm-10m-2, update 12500)
#   参考运动    = CMU/120/cmu120_19_straight_h000, --start_step 53, --seconds 10
#   模式        = stochastic, seeds 0..19
#
# 条件：
#   A  左上肢 1.0, 左下肢 1.0 (正常基线, 右侧全 1.0)
#   B  左上肢 0.7, 左下肢 1.0 (仅上肢减力)
#   C  左上肢 1.0, 左下肢 0.7 (仅下肢减力)
#   D  左上肢 0.7, 左下肢 0.7 (上下肢均减力)
#   E  --f0_left 0.7 旧语义精确复现 (左躯干+左下肢 ×0.7, 左上肢不变)
#
# 每组只缩放左侧；右侧肌肉与躯干保持原值（E 除外，E 含左躯干，按旧语义）。
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=external/MuscleMimic:scripts
PY=".venv/bin/python"
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
BASE="--motion_path CMU/120/cmu120_19_straight_h000 --start_step 53 --seconds 10 --mode stochastic"

run_cond () {
    local cond="$1"; shift
    echo "########## CONDITION $cond ##########"
    $PY scripts/eval_walk.py $BASE --seeds $SEEDS --out_dir "runs/f0_ablation/$cond" "$@" 2>&1 | tail -25
}

run_cond A
run_cond B --f0_left_upper 0.7
run_cond C --f0_left_lower 0.7
run_cond D --f0_left_upper 0.7 --f0_left_lower 0.7
run_cond E --f0_left 0.7

echo "ALL DONE"
