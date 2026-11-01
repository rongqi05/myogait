# MuscleMimic walking baseline（可审计的 MyoFullBody 10 s 行走基线）

本仓库是一个**独立**的复现/验收工作区，用来回答一个问题：

> 官方 MuscleMimic（MyoFullBody + `amathislab/mm-10m-2`）能否作为正常人连续行走的动力学基线？

它不修改 `~/Documents/muscle-rl`、`~/Documents/muscle` 或任何已有 checkpoint。
上游代码只以只读方式放在 `external/`（不入 git）。

## 当前结论（2026-10-29，已完成 10 s 验收）

> **未通过 10 s 验收：3/20（目标 18/20）。**
> 20/20 跑满 10.00 s、无跌倒、速度/倾角/根高全达标；**17/20 最大侧偏 >0.35 m**（最大 1.19 m）。
> 失败模式是**180° 转身时转弯半径偏大，转完后沿参考平行线行走**，不是跌倒或走不动。

| 片段 | 组 | 时长 | 20 seed（stochastic） | v_fwd | 侧偏 mean [min,max] |
|---|---|---:|---|---:|---|
| `KIT/3/walk_6m_straight_line06` | **TRAIN** | **14.90 s** | **3/20 通过**（唯一满足 10 s 前置条件的片段） | 0.668 | 0.582 [0.271, 1.192] |
| `KIT/314/walking_medium09` | TRAIN | 7.72 s | 20/20 通过侧偏等判据（时长/速度不达标） | 0.327 | 0.151 [0.065, 0.275] |
| `KIT/3/walk_6m_straight_line04` | TRAIN | 7.35 s | 13/20 满足侧偏 | 0.782 | 0.314 [0.172, 0.446] |
| `KIT/4/WalkInCounterClockwiseCircle04` | TEST | 8.39 s | 14/20 满足侧偏 | 0.514 | 0.295 [0.178, 0.568] |
| `KIT/314/walking_medium09`（deterministic） | TRAIN | 7.72 s | 0/20（横向漂移 0.61 m） | 0.311 | 0.606 |

- 整个数据集 1190 条动作里**只有 1 条 ≥10 s**，且属于**训练组** → 10 s 结果 **in-sample**。
  官方 testing 组最长 8.53 s，物理上无法满足 10 s 判据。
- 之前“官方 checkpoint 回放会倒”在本机可复现，原因是 `musclemimic_models` **1.0.6**：
  3/3 seed 在 2.5–2.8 s 跌倒；固定到 **1.0.5** 后 3/3 正常，单因子回退腿 XML 即可恢复。
- 详细判定：`reports/decision.md`；10 s 结果：`reports/C_acceptance.md` C8 节；
  逐 seed 表：`reports/table_10s_long.md`。

10 s 验收视频（20 fps，叠加真实时间轴与倾角）：

- `reports/video/10s_walk_6m_straight_line06_seed0_SUCCESS.mp4`（seed 0，侧偏 0.333 m，通过）
- `reports/video/10s_walk_6m_straight_line06_seed10_LATERAL_FAIL.mp4`（seed 10，侧偏 1.192 m）

## 直线行走训练基线（2026-10-31）

任务：实现并验证"正常人体肌肉驱动的直线行走基线"。结论：**未达到 10 秒直线稳定行走（0/20）**，
详见 [`reports/decision_straight_walking.md`](./reports/decision_straight_walking.md) 与
[`reports/E_audit_training.md`](./reports/E_audit_training.md)。

| 事实 | 值 |
|---|---|
| 本仓库算法 | **MuscleMimic PPOJax**（`conf_fullbody.yaml: algorithm: PPOJax`），不是 Muscle-SuperMimic（全盘无该代码） |
| 本地 ≥10 s 直线片段 | **不存在**（最长直线窗口 6.84 s） |
| 数据准备 | `straight_stitched_10s`，13.39 s / 0.874 m/s / 侧向 0.176 m，含 0.2 s 交叉淡入（非动力学一致） |
| 8 GB GPU 可行档位 | `num_envs=512 × 20 steps`，峰值显存 7043 MiB，≈2.8k env-step/s，RSS 3.9 GB |
| 本次训练 | 300 updates / 3.07M steps / 18 分钟 |
| 10 s 验收（本次策略） | **0/20**，1.24 s 跌倒 |
| 10 s 验收（官方 mm-10m-2） | **0/20**，19/20 走满 10 s 不跌倒，但侧偏 0.54–2.10 m |
| 航向误差（已修正） | 平均 **8.6°**（旧报告"86°"是指标错误，见 `reports/F_single_factor_reference.md`） |
| 未拼接的真实 ≥10 s 直线参考 | **不可得**。KIT 原始数据（4232 条）**最长直线窗口 7.09 s，无一条 ≥8 s** → 数据门槛未通过（`F`、`G`、`H` 三份审计） |

新增脚本：

```
scripts/motion_straightness.py     片段直线性/时长/接触质量审计
scripts/prepare_straight_walk.py   直线窗口抽取 + 对齐 + 拼接 + 重算派生量
scripts/train_straight_bounded.sh  8 GB 显存档位的受限训练（含显存监控）
scripts/acceptance_straight.sh     冻结协议的 20 种子验收
scripts/check_muscle_chain.py      肌肉力链路证据
scripts/bench_train_gpu.sh         吞吐/显存基准
```

## 三个问题必须分开回答（2026-11-01）

"走满 10 s"、"跟住参考"、"通过冻结验收"是三个不同的问题，答案确实不同。
详见 [`reports/K_retarget_blocker_and_three_way.md`](./reports/K_retarget_blocker_and_three_way.md)。

| 参考 | A 走满 10 s | B 跟随参考 | C 通过冻结验收 |
|---|---:|---:|---:|
| `straight_stitched_10s`（拼接） | 19/20 | 0/20 | 0/20 |
| `walk_6m_straight_line06`（未拼接，直行仅 6.84 s） | 20/20 | 4/20 | 3/20 |

失效机理已定位：**不是跌倒、不是航向错误（平均 8.6°）、不是速度不足**，而是
**持续性单侧侧移**——20/20 种子漂移方向相同，等效约 6.5° "蟹行"，10 s 累积 0.79 m，
且 `corr(侧向漂移, 最大侧偏) = −0.975`。

```bash
scripts/retarget_one.py      单条 AMASS → MyoFullBody 重定向（3 道前置关卡，冻结 GMR 参数）
scripts/qc_retargeted.py     重定向产物 QC（数值/根节点/关节突变/足高/足滑/拼接段）
scripts/three_way_report.py  A/B/C 三问分开报告 + 逐回合俯视图
scripts/check_gmr_toolchain.py / setup_gmr_env.sh   .venv-gmr 工具链安装与 14 项自检
```

> 单条 `CMU/136/136_25_poses`（原始 14.90 s、直行窗 11.64 s、0.507 m/s）的重定向
> **尚未执行**：前置关卡 1（模型 1.0.5）与 2（原始片段够长够直）已通过，
> 卡在 `~/.musclemimic/smpl/SMPLH_NEUTRAL.pkl`——SMPL-H 人体模型许可受限，
> 需在 <https://mano.is.tue.mpg.de/download.php> 用本人账号接受协议后合成。

## 环境（已冻结）

| 组件 | 版本 |
|---|---|
| Python | 3.11.15 |
| MuscleMimic | `47b653db2bce25de834360d2ee6b2dd88142bd78` |
| **musclemimic_models** | **1.0.5** (`ba69358c6333692d547836b903f051fe6a8e9cc9`) — 必须固定，1.0.6 会跌倒 |
| MuJoCo | 3.4.0（CPU） |
| numpy / jax / flax | 2.2.6 / 0.7.2 / 0.12.0 |
| 后端 | 纯 MuJoCo CPU（验收）；warp/CUDA 仅用于训练（`.venv-train`） |
| checkpoint | `amathislab/mm-10m-2`, update 12500, digest `c8c7a964…` |

三个环境，职责固定互不污染（`.gitignore` 已忽略后两个）：

| 环境 | 用途 | jax | torch |
|---|---|---|---|
| `.venv` | **冻结的验收环境**（`eval_walk.py`、A/B/C 三问报告） | CPU | 无 |
| `.venv-train` | PPO + warp 训练 | CUDA12 | 无 |
| `.venv-gmr` | GMR-Fit 重定向（纯 IK，不占显存） | CPU | **CPU-only** |

`.venv-gmr` 由 `bash scripts/setup_gmr_env.sh` 重建；它会强制把 `musclemimic_models`
回退到 1.0.5，并用 `scripts/check_gmr_toolchain.py` 校验膝约束符号模式。
GPU：RTX 5060 Laptop 8 GB（sm_120），驱动已加载，EGL 离屏渲染可用。

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

# 2) 虚拟环境（需要 Python 3.11.x；uv 会自动取 standalone 构建）
uv venv .venv --python 3.11

#    注意：本仓库当前使用的 .venv 是**自包含**的（python 二进制 + stdlib + .so
#    都复制进 .venv），不依赖任何 conda 环境，因此 `pyvenv.cfg` 已改名为
#    `pyvenv.cfg.hemi_msk_orig`。见 reports/env_snapshots/ 与下方“环境维护”。

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

# 10 s 验收（已取得数据集访问）：下载 14.9 s 片段 → 前置校验 → 20 seed 验收 → 视频
bash scripts/after_access_10s.sh
MUJOCO_GL=egl .venv/bin/python scripts/eval_walk.py \
  --motion_path KIT/3/walk_6m_straight_line06_poses \
  --mode stochastic --seeds 0 10 --video --video_every 5 --out_dir runs/C_video_10s

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
  make_table.py          从 run 目录生成逐 seed markdown 表
  retarget_one.py        单条 AMASS → MyoFullBody（GMR-Fit），3 道前置关卡
  qc_retargeted.py       重定向产物 QC：数值/根节点/关节突变/足高/足滑/拼接段
  three_way_report.py    A 走满 / B 跟随参考 / C 通过验收，逐回合报告 + 俯视图
  check_gmr_toolchain.py .venv-gmr 的 14 项自检（含 1.0.5 膝约束符号模式）
  setup_gmr_env.sh       重建 .venv-gmr 并回退 musclemimic_models 到 1.0.5
  amass_kit_audit.py     流式审计 AMASS .tar.bz2（不解压）：时长/直行窗/速度
  after_access_10s.sh    取得 gated 数据后的一键 10 s 验收（含前置校验）
  run_suite.sh           本次全部 6 组、67 个 rollout 的顺序复现
  dump_reset_state.py    D：记录 reset 后的 qpos/qvel/obs（版本对比用）
configs/eval_walk.json   冻结的协议、阈值、哈希、指标定义
reports/
  A_environment_audit.md
  B_reproduction.md
  C_acceptance.md
  D_model_version_compatibility.md
  decision.md            ← 通过 / 未通过 / 因数据不足无法判定
  env_audit.json, motion_stats.json, obs_layout.json, joint_map.json
  table_walk.md, table_walk_det.md, table_circle.md, table_10s_long.md,
  table_diag_straight.md         逐 seed 表
  D_model_divergence/            版本对比的 reset 状态与 XML 哈希
  logs/                 过滤后的官方回放/套件/审计日志（证据）
  video/10s_*_seed0_SUCCESS.mp4, 10s_*_seed10_LATERAL_FAIL.mp4   ← 10 s 验收视频
  video/walking_medium09_stochastic_seed0.mp4
runs/                    每次运行的 JSON/CSV/NPZ/日志（npz/mp4/log 不入 git）
external/                只读上游源码（不入 git）
artifacts/               checkpoint 与动作缓存（不入 git）
```

## 资源

- 单进程顺序评估，峰值 RSS 2.5–4.6 GB（实测 30 GiB / 0 swap 下两组 × 20 种子无压力）；无并行训练。
- 每 seed 约 15–21 s（含环境构建），20 seed 约 5 分钟；两组顺序跑约 10 分钟。
- 视频渲染用 `MUJOCO_GL=egl`（GPU 可用时为硬件 EGL）。
- `.venv` 1.8 GB · `.venv-train` 6.6 GB · `.venv-gmr` 2.5 GB。

## 环境维护（2026-10-31）

- `.venv` 已改为**自包含**：python 二进制、stdlib、依赖的 `.so` 都复制进 `.venv`，
  `pyvenv.cfg` 改名为 `pyvenv.cfg.hemi_msk_orig`（原件备份）。`sys.base_prefix == sys.prefix`。
  改动后已用 seed 0 回放验证与清理前的结果 **逐位一致**。
- 已删除 conda 环境 `hemi_msk`（旧 mojoco，可编辑安装已失效）与 `hemirl`（muscle-rl）。
  依赖快照保存在 `reports/env_snapshots/`。
- 清理了 pip / uv / conda 缓存。磁盘可用空间 85 G → 101 G。
- 已删除 conda 环境 `env_isaaclab`（Isaac Lab，11 G）。依赖快照见
  `reports/env_snapshots/env_isaaclab_freeze_20261031.txt`。
- **现在除了 conda `base` 没有其它 conda 环境**，`.venv` 是唯一可用的 Python 环境；
  重建 Isaac Lab 见 `reports/env_snapshots/README.md`。

## 已知限制

- 未使用 MyoSuite 的 MuscleMimic 集成路径，未核实其“缺少策略文件时回退到参考轨迹回放”的分支；
  本仓库一律以官方 `fullbody/eval.py` 为基线。
- checkpoint 未记录训练时的 `musclemimic_models` 版本；“1.0.5 是配对版本”依据是 MyoSuite 文档
  + 本仓库的配对回放行为差异。
- **`SMPLH_NEUTRAL.pkl` 缺失**，因此单条 `CMU/136/136_25_poses` 的 GMR 重定向从未跑通。
  已验证的只有前置关卡（1.0.5 变体、原始片段 11.64 s 直行）与工具链本身；
  `fit_gmr_motion()` 之后的 SMPL 前向、500 次 shape fitting、mink IK 全流程**尚未执行过**。
- `qc_retargeted.py` 的足滑指标是**高度阈值代理**（参考轨迹没有接触力），
  5 mm 带下参考自身滑步 0.06–0.11 m/s；宽阈值会混入摆动掠过帧，因此报告敏感性而非单一值。
- 策略的足滑另由 `eval_walk.py` 用接触法向力 >20 N 判定，与参考的口径不同，不可直接相减。
- 未拼接参考的改善**不能**归因于“去掉拼接”：两条参考的速度（0.883 vs 0.834 m/s）
  与几何（后者 10 s 窗口内直行仅 3.66–10.00 s）都不同。
