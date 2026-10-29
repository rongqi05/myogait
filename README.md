# MuscleMimic walking baseline（可审计的 MyoFullBody 10 s 行走基线）

本仓库是一个**独立**的复现/验收工作区，用来回答一个问题：

> 官方 MuscleMimic（MyoFullBody + `amathislab/mm-10m-2`）能否作为正常人连续行走的动力学基线？

它不修改 `~/Documents/muscle-rl`、`~/Documents/muscle` 或任何已有 checkpoint。
上游代码只以只读方式放在 `external/`（不入 git）。

## 当前结论（2026-10-29）

**因数据不足无法判定 10 s 验收。** 在本地可获得的全部参考片段上：

| 片段 | 时长 | 20 seed（stochastic） | 除 10 s/速度外的判据 | 速度 |
|---|---:|---|---|---:|
| `KIT/314/walking_medium09` | 7.72 s | 20/20 走完整段，无跌倒 | 20/20 满足 | 0.328 m/s（参考 0.335） |
| `KIT/4/WalkInCounterClockwiseCircle04` | 8.39 s | 20/20 走完整段，无跌倒 | 18/20 满足（2 个 seed 侧偏 >0.35 m） | 0.514 m/s |
| walking_medium09（deterministic） | 7.72 s | 20/20 走完整段，无跌倒 | **0/20**（横向漂移 0.61 m） | 0.311 m/s |

- 没有 ≥10 s 的可用片段，且参考本身平均速度 <0.6 m/s → 10 s / 0.6 m/s 判据**无法检验**。
- 之前“官方 checkpoint 回放会倒”在本机可复现，原因是 `musclemimic_models` **1.0.6**：
  3/3 seed 在 2.5–2.8 s 跌倒；固定到 **1.0.5** 后 3/3 正常，单因子回退腿 XML 即可恢复。
- 详细决策与下一步：`reports/decision.md`。

## 环境（已冻结）

| 组件 | 版本 |
|---|---|
| Python | 3.11.15 |
| MuscleMimic | `47b653db2bce25de834360d2ee6b2dd88142bd78` |
| **musclemimic_models** | **1.0.5** (`ba69358c6333692d547836b903f051fe6a8e9cc9`) — 必须固定，1.0.6 会跌倒 |
| MuJoCo | 3.4.0（CPU） |
| numpy / jax / flax | 2.2.6 / 0.7.2 / 0.12.0 |
| 后端 | 纯 MuJoCo CPU（本机无 GPU） |
| checkpoint | `amathislab/mm-10m-2`, update 12500, digest `c8c7a964…` |

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

# 2) 虚拟环境（用已有的 3.11 解释器，避免 uv 下载）
uv venv .venv --python /home/zrq/miniconda3/envs/hemi_msk/bin/python

# 3) 依赖（PyPI 直连不稳定，用清华镜像 + PyPI 兜底；distrax 依赖只在 PyPI 上的 tfp-nightly）
uv pip install --python .venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://pypi.org/simple \
  numpy==2.2.6 "scipy>=1.14.0" mujoco==3.4.0 mujoco-mjx==3.4.0 warp-lang==1.10.0 \
  jax==0.7.2 jaxlib==0.7.2 flax==0.12.0 "optax>=0.2.4" orbax-checkpoint==0.11.37 \
  distrax==0.1.7 hydra-core pyyaml "ml-collections>=1.1.0" opencv-python gitpython \
  "lxml>=6.0.0" tqdm huggingface_hub "imageio>=2.37.0" "imageio-ffmpeg>=0.6.0" \
  "glfw>=2.9.0" "seaborn>=0.13.2" metrx "wandb>=0.19.10" "viser>=1.0.10" \
  "trimesh>=4.0.0" "Pillow>=10.0.0"

# 4) 模型包，严格 1.0.5（--no-deps 避免把 mujoco 升级）
uv pip install --python .venv/bin/python --no-deps ./external/musclemimic_models
.venv/bin/python -c "import musclemimic_models; print(musclemimic_models.__file__)"
```

> 不要用 `uv sync`：`external/MuscleMimic/pyproject.toml` 写的是 `musclemimic_models>=1.0.2`，
> 今天会解析到 1.0.6，直接复现跌倒。

### 网络

本机直连 `huggingface.co` 会超时，所有 HF 下载都要走镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 数据与 checkpoint

```bash
mkdir -p artifacts/checkpoints artifacts/motions

# checkpoint（公开，无需 gate）
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("amathislab/mm-10m-2", local_dir="artifacts/checkpoints/mm-10m-2")
PY

# 官方 loader 需要一个 checkpoint_<step> 目录名；用绝对路径符号链接
ln -sfn "$PWD/artifacts/checkpoints/mm-10m-2" artifacts/checkpoints/checkpoint_12500
```

> 官方 `_canonicalize_resume_path()` 会自己建一个**相对**符号链接，在 `artifacts/checkpoints/` 下
> 解析会变成坏链。上面这步是必要的（未改上游代码）。

Demo 动作（3 条，gated-auto，需要 token 已同意 demo 条款）：

```bash
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -c \
  "from huggingface_hub import hf_hub_download as d; \
   [d('amathislab/demo_dataset', f'MyoFullBody/gmr/{p}.npz', repo_type='dataset', \
      local_dir='artifacts/motions/demo') for p in [ \
      'KIT/314/walking_medium09_poses','KIT/348/turn_right03_poses', \
      'KIT/4/WalkInCounterClockwiseCircle04_poses']]"
mkdir -p ~/.musclemimic/caches/AMASS/MyoFullBody/gmr
cp -r artifacts/motions/demo/MyoFullBody/gmr/* ~/.musclemimic/caches/AMASS/MyoFullBody/gmr/
```

完整重定向数据集 `amathislab/musclemimic-retargeted` 需要网页接受 AMASS 条款（403 否则），
其中才有 ≥10 s 的 walking 片段。见 `reports/decision.md`。

已验证的哈希：

| 对象 | sha256 |
|---|---|
| `mm-10m-2` 目录 digest（16 文件） | `c8c7a964b88b7163bc9989785063cbb751dae88cd4358eaced25bf3475a73b2a` |
| `walking_medium09_poses.npz` | `8f320295504ffc0a7759ba76ee454dd2b1e3f1feb7244d476b2373100235cc9b` |
| `WalkInCounterClockwiseCircle04_poses.npz` | `8e249a712f95a7d2cfcc3653db2012892b948bbcc94da21d2348078cce904ddf` |
| `turn_right03_poses.npz` | `99255a4e15866ce81b3a5fe6d7b9a5de9138905697170241bebccdb1bb84399d` |

## 复现命令

全部命令在仓库根目录执行，并设置 `PYTHONPATH=external/MuscleMimic:scripts`。

```bash
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=external/MuscleMimic:scripts

# A) 环境/模型/接口审计 → reports/env_audit.json
.venv/bin/python scripts/env_audit.py

# A) 参考片段时长与速度上限 → reports/motion_stats.json
.venv/bin/python scripts/motion_stats.py

# B) 官方入口原样回放（注意：n_steps 是跨 episode 总预算，会自动 reset）
.venv/bin/python external/MuscleMimic/fullbody/eval.py \
  --path artifacts/checkpoints/checkpoint_12500 \
  --motion_path KIT/314/walking_medium09_poses \
  --use_mujoco --no_render --stochastic --n_steps 1000 \
  --export_trajectory --trajectory_dir runs/B_official/stochastic

# C) 单回合验收（先 3 seed）
.venv/bin/python scripts/eval_walk.py \
  --checkpoint artifacts/checkpoints/checkpoint_12500 \
  --motion_path KIT/314/walking_medium09_poses \
  --mode stochastic --seeds 0 1 2 --out_dir runs/C_pilot

# C) 20 seed 验收 + 视频
.venv/bin/python scripts/eval_walk.py \
  --checkpoint artifacts/checkpoints/checkpoint_12500 \
  --motion_path KIT/314/walking_medium09_poses \
  --mode stochastic --seeds $(seq 0 19) --out_dir runs/C_acceptance

MUJOCO_GL=egl .venv/bin/python scripts/eval_walk.py \
  --mode stochastic --seeds 0 --video --video_every 5 \
  --out_dir runs/C_acceptance_video

# D) 版本配对回放（1.0.6 / 1.0.6+1.0.5 腿）
git -C external/musclemimic_models archive v1.0.6 | tar -x -C /tmp/mm_models_106
PYTHONPATH=/tmp/mm_models_106:external/MuscleMimic:scripts \
  .venv/bin/python scripts/eval_walk.py --mode stochastic --seeds 0 1 2 \
  --out_dir runs/D_model_version/106
PYTHONPATH=/tmp/mm_models_106:external/MuscleMimic:scripts \
  .venv/bin/python scripts/dump_reset_state.py --tag 106 \
  --out reports/D_model_divergence/reset_106.json
```

## 目录

```
scripts/
  _env_builder.py        按官方 eval.py 的方式构建环境（不修改上游）
  env_audit.py           A：模型/接口/checkpoint/motion 审计
  motion_stats.py        参考片段的时长、速度、连续行走段
  eval_walk.py           C：单 episode、无中途 reset 的验收 + 逐帧日志 + 视频
  dump_reset_state.py    D：记录 reset 后的 qpos/qvel/obs（版本对比用）
configs/eval_walk.json   冻结的协议、阈值、哈希、指标定义
reports/
  A_environment_audit.md
  B_reproduction.md
  C_acceptance.md
  D_model_version_compatibility.md
  decision.md            ← 通过 / 未通过 / 因数据不足无法判定
  env_audit.json, motion_stats.json, obs_layout.json, joint_map.json
  table_walk.md, table_walk_det.md, table_circle.md  逐 seed 表
  D_model_divergence/            版本对比的 reset 状态与 XML 哈希
  logs/                 过滤后的官方回放/套件/审计日志（证据）
  video/walking_medium09_stochastic_seed0.mp4
runs/                    每次运行的 JSON/CSV/NPZ/日志（npz/mp4/log 不入 git）
external/                只读上游源码（不入 git）
artifacts/               checkpoint 与动作缓存（不入 git）
```

## 资源

- 单进程顺序评估，峰值 RSS 2.5–4.6 GB；无 swap，无并行训练。
- 每 seed 约 12 s（含环境构建），20 seed 约 4 分钟。
- 视频渲染用 `MUJOCO_GL=egl`（软件 EGL，可渲染但会打印 MESA 警告）。

## 已知限制

- 未使用 MyoSuite 的 MuscleMimic 集成路径，未核实其“缺少策略文件时回退到参考轨迹回放”的分支；
  本仓库一律以官方 `fullbody/eval.py` 为基线。
- 无 GPU，未测试 MJX / warp 后端。
- checkpoint 未记录训练时的 `musclemimic_models` 版本；“1.0.5 是配对版本”依据是 MyoSuite 文档
  + 本仓库的配对回放行为差异。
