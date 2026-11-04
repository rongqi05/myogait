# MuscleMimic 肌肉驱动行走基线（MyoFullBody + mm-10m-2）

独立复现/验收工作区：回答

> 官方 MuscleMimic（MyoFullBody + `amathislab/mm-10m-2`）能否作为“正常人肌肉驱动、
> 根节点自由”的直线行走基线？以及单侧 F0（肌肉最大等长力）缩放如何影响行走？

为毕业设计“降低患侧肌力 → 重新适应 → 比较拐杖辅助侧别”提供可复现的正常行走策略与
患侧肌力旋钮。**不修改** `muscle-rl` / `muscle` 或任何已有 checkpoint；上游源码只读放在
`external/`（不入 git）。

## 当前结论（摘要）

| 问题 | 结论 |
|---|---|
| 官方 checkpoint 能否走满 10 s 不跌倒 | **是**。干净参考 `CMU/120/cmu120_19_straight_h000` 上 **18/20** 走满 10 s（0.457 m/s） |
| 能否通过冻结的直线验收（0.6 m/s、侧偏 ≤0.35 m） | **否**。参考本身 0.457 m/s 达不到 0.6 m/s 门槛；横向漂移另见报告 |
| 单侧下肢 F0×0.7 | **20/20 早期跌倒**（~3.2 s，向弱侧偏转） |
| 单侧上肢 F0×0.7 | **14/20** 走满 10 s；可行上肢损伤区间 **\[0.7, 0.9\]** |

详细判定与逐种子数据见主要报告：

- [`reports/decision.md`](./reports/decision.md) —— 10 s 行走验收（通过/未通过/数据不足）
- [`reports/decision_straight_walking.md`](./reports/decision_straight_walking.md) —— 直线行走基线结论
- [`reports/O_muscle_force_scaling.md`](./reports/O_muscle_force_scaling.md) —— 单侧 F0 缩放实验
- [`reports/P_f0_upper_lower_ablation.md`](./reports/P_f0_upper_lower_ablation.md) —— 上肢/下肢减力定位
- [`reports/Q_upper_limb_feasible_range.md`](./reports/Q_upper_limb_feasible_range.md) —— 可行的上肢损伤区间

## 核心文件

```
scripts/
  _env_builder.py          按官方 fullbody/eval.py 的方式构建评估环境（含 F0 缩放核心）
  eval_walk.py             单 episode、无中途 reset 的验收；--f0_left/--f0_right 与
                           上肢/下肢分区缩放；逐帧日志 + 可选视频
  check_f0_groups.py       模型级检查：354 条肌肉按 XML 解剖分组、F0 缩放只命中目标肌肉
  run_f0_ablation.sh       上肢/下肢 × 单侧 F0 减力定位实验（A–E 五条件）
  train_straight_bounded.sh  8 GB 显存档位的受限训练（含显存监控）
  bench_train_gpu.sh         训练吞吐 / 显存基准
configs/
  eval_walk.json           冻结的验收协议、阈值、哈希、指标定义（v1.1）
  constraints.txt          固定 musclemimic_models==1.0.5（1.0.6 会跌倒）
reports/                   主要 Markdown 报告（原始 JSON/图/视频证据不入 git）
```

## 环境（已冻结）

| 组件 | 版本 |
|---|---|
| Python | 3.11.15 |
| MuscleMimic（上游源码） | `47b653db2bce25de834360d2ee6b2dd88142bd78` |
| **musclemimic_models** | **1.0.5**（`ba69358c…`）——必须固定，1.0.6 左膝约束符号翻转致跌倒 |
| MuJoCo | 3.4.0（CPU 验收） |
| numpy / jax / flax | 2.2.6 / 0.7.2 / 0.12.0 |
| checkpoint | `amathislab/mm-10m-2`，update 12500，digest `c8c7a964…` |
| GPU | RTX 5060 Laptop 8 GB（仅训练用） |

### 安装

```bash
cd ~/Documents/musclemimic

# 1) 上游源码（只读，不入 git）
mkdir -p external && cd external
git clone https://github.com/amathislab/MuscleMimic.git
git -C MuscleMimic checkout 47b653db2bce25de834360d2ee6b2dd88142bd78
git clone https://github.com/amathislab/musclemimic_models.git
git -C musclemimic_models checkout v1.0.5
cd ..

# 2) 虚拟环境
uv venv .venv --python 3.11

# 3) 依赖
uv pip install --python .venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://pypi.org/simple \
  numpy==2.2.6 "scipy>=1.14.0" mujoco==3.4.0 mujoco-mjx==3.4.0 warp-lang==1.10.0 \
  jax==0.7.2 jaxlib==0.7.2 flax==0.12.0 "optax>=0.2.4" orbax-checkpoint==0.11.37 \
  distrax==0.1.7 hydra-core pyyaml "ml-collections>=1.1.0" opencv-python gitpython \
  "lxml>=6.0.0" tqdm huggingface_hub "imageio>=2.37.0" "imageio-ffmpeg>=0.6.0" \
  "glfw>=2.9.0" "seaborn>=0.13.2" metrx "wandb>=0.19.10" "viser>=1.0.10" \
  "trimesh>=4.0.0" "Pillow>=10.0.0"

# 4) 模型包，严格 1.0.5（--no-deps 避免升级 mujoco）
uv pip install --python .venv/bin/python --no-deps ./external/musclemimic_models
```

> 不要用 `uv sync`：上游 `pyproject.toml` 写的是 `musclemimic_models>=1.0.2`，会解析到 1.0.6。

## 数据与 checkpoint

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 本机直连 huggingface.co 超时

mkdir -p artifacts/checkpoints
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("amathislab/mm-10m-2", local_dir="artifacts/checkpoints/mm-10m-2")
PY
# 官方 loader 需要 checkpoint_<step> 目录名；官方会自己建相对符号链接（会变坏链），
# 用绝对路径符号链接代替：
ln -sfn "$PWD/artifacts/checkpoints/mm-10m-2" artifacts/checkpoints/checkpoint_12500
```

参考运动（重定向后的 MyoFullBody `.npz`）放在
`~/.musclemimic/caches/AMASS/MyoFullBody/gmr/`；本仓库实验固定使用
`CMU/120/cmu120_19_straight_h000`（未拼接、正向、10.41 s 直线窗、0.457 m/s）。

## 复现命令

所有命令在仓库根目录执行，并 `export PYTHONPATH=external/MuscleMimic:scripts`。

```bash
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=external/MuscleMimic:scripts

# 单回合验收（3 seed 冒烟）
.venv/bin/python scripts/eval_walk.py \
  --checkpoint artifacts/checkpoints/checkpoint_12500 \
  --motion_path CMU/120/cmu120_19_straight_h000 --start_step 53 \
  --mode stochastic --seeds 0 1 2 --out_dir runs/pilot

# 20 seed 验收（可加 --video 导出 mp4）
.venv/bin/python scripts/eval_walk.py \
  --motion_path CMU/120/cmu120_19_straight_h000 --start_step 53 --seconds 10 \
  --mode stochastic --seeds $(seq 0 19) --out_dir runs/acceptance

# 单侧 F0 缩放（患侧肌力旋钮）
.venv/bin/python scripts/eval_walk.py \
  --motion_path CMU/120/cmu120_19_straight_h000 --start_step 53 --seconds 10 \
  --mode stochastic --seeds $(seq 0 19) \
  --f0_left_lower 0.7 --out_dir runs/left_lower_070

# 上肢/下肢减力定位（A–E 五条件 × 20 种子）
bash scripts/run_f0_ablation.sh

# 8 GB 显存下的受限训练 / 基准
NENV=512 STEPS=20 MINUTES=25 bash scripts/train_straight_bounded.sh
NENV=256 UPDATES=4 bash scripts/bench_train_gpu.sh
```

### F0 缩放语义（重要）

- 模型 354 条肌肉 = 左上肢 32 + 右上肢 32 + 左下肢 40 + 右下肢 40 + 左躯干 105 + 右躯干 105。
- 命名约定不对称：**腿/躯干**用 `_l`/`_r`，**手臂**左臂用 `_left`、右臂无后缀。
- `--f0_left/--f0_right`（旧语义）：缩放全部 `_l`/`_r` = **躯干+腿**（145 条），**不含手臂**。
- `--f0_left_upper/--f0_left_lower/--f0_right_upper/--f0_right_lower`：只缩放对应解剖分组。
- F0 只缩放主动力（Hill 模型），被动张力不缩放；主动力变化是**亚比例**的
  （F0×0.7 → 行走中下肢主动力约 ×0.94、上肢约 ×0.87）。

## 冻结验收协议（configs/eval_walk.json, v1.1）

| 阈值 | 值 |
|---|---|
| 最短时长 | 10.0 s（`data.time - reset_time`，单次 reset） |
| 平均前进速度 | ≥ 0.6 m/s |
| 最大侧偏 | ≤ 0.35 m |
| 骨盆/躯干倾角 | ≤ 45° |
| 跌倒 | 根高 < 0.6 m 或倾角 > 60° |

结束原因：`time_limit_success` / `physical_fall` / `reference_deviation` /
`clip_end` / `numeric_anomaly` / `reset_or_state_override` / `other`。

## 已知限制

- 未使用 MyoSuite 的 MuscleMimic 集成路径；一律以官方 `fullbody/eval.py` 为基线，未核实其
  “缺少策略文件时回退参考轨迹回放”的分支。
- checkpoint 未记录训练时的 `musclemimic_models` 版本；“1.0.5 是配对版本”依据是 MyoSuite 文档
  与本仓库的配对回放行为差异。
- 策略**朝向锁定**：物理上等价的 z 旋转会改变侧走角，不同世界朝向的组不可直接比较。
- F0 缩放比例 ≠ 临床肌力等级；跌倒回合是“冻结正常策略对损伤的反应”，不是稳定偏瘫步态。
