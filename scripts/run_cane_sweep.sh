#!/usr/bin/env bash
# 拐杖条件扫描：固定左下肢损伤程度，扫描左上肢（患侧上肢）肌力，
# 比较健侧持杖（右手 = 对侧）与患侧持杖（左手 = 同侧）的竖直支撑、
# 行走完成率与躯干稳定性。
#
# 固定配置（与 10 s 冻结协议一致）：
#   checkpoint    = artifacts/checkpoints/checkpoint_12500 (官方 mm-10m-2)
#   参考运动      = CMU/136/cmu136_stitched_20s_v2_h000_rot000 (28.37 s, 不提前结束)
#   协议          = --protocol 10s, --mode stochastic
#   左下肢 F0     = 0.90（由 F0 扫描确定的“有损伤效应但仍有可行轨迹”强度）
#   右下肢/躯干   = 1.0
#
# 条件（U = 左上肢 F0 系数）：
#   nocane_U<x>   无拐杖对照（同一减力条件）
#   ipsi_U<x>     患侧（左）手持杖，刚性拐杖 + 理想固定握持
#   contra_U<x>   健侧（右）手持杖
#
# 单进程顺序执行：本机 30 GB 内存无 swap，禁止并发。
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=external/MuscleMimic:scripts
PY=".venv/bin/python"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
LEVELS="${LEVELS:-1.0 0.9 0.8 0.7}"
LOWER="${LOWER:-0.90}"
OUT="${OUT:-runs/cane_sweep}"
BASE="--motion_path CMU/136/cmu136_stitched_20s_v2_h000_rot000 --protocol 10s --mode stochastic"

run_cond () {
    local cond="$1"; shift
    echo "########## CONDITION $cond ($(date +%H:%M:%S)) ##########"
    $PY scripts/eval_walk.py $BASE --seeds $SEEDS --out_dir "$OUT/$cond" "$@" 2>&1 | tail -20
}

for U in $LEVELS; do
    run_cond "nocane_U$U" --f0_left_upper "$U" --f0_left_lower "$LOWER"
    run_cond "ipsi_U$U"   --f0_left_upper "$U" --f0_left_lower "$LOWER" --cane-side left
    run_cond "contra_U$U" --f0_left_upper "$U" --f0_left_lower "$LOWER" --cane-side right
done

echo "ALL DONE ($(date +%H:%M:%S))"
