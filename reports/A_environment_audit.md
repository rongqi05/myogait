# A. 环境与兼容性审计

审计日期：2026-10-29。所有数字来自 `reports/env_audit.json`（可用 `scripts/env_audit.py` 复现）。
本文件只写**已核实的事实**；推测性内容集中在最后一节。

## A.1 结论摘要

| 项目 | 事实 |
|---|---|
| MuscleMimic 源码 | `github.com/amathislab/MuscleMimic` @ `47b653db2bce25de834360d2ee6b2dd88142bd78`（main, 2026-09-23） |
| 模型包 | `musclemimic_models` **1.0.5** @ `ba69358c6333692d547836b903f051fe6a8e9cc9` |
| MuJoCo | 3.4.0（仓库 `pyproject.toml` 固定） |
| Python / numpy / jax | 3.11.15 / 2.2.6 / 0.7.2（CPU） |
| 硬件 | 无可用 GPU（`nvidia-smi` 无法通信），20 vCPU，30 GiB RAM，0 swap |
| checkpoint | `amathislab/mm-10m-2`，`update_number=12500`，`step=1.6e6`，134 MB，digest `c8c7a964…` |
| 模型 | MyoFullBody：`nq=89, nv=88, nu=354`（354 个 Hill 型肌肉执行器） |
| 观测 / 动作 | obs 2418 维 / action 354 维 ∈ [-1, 1] |
| 控制 / 物理步长 | 控制 0.01 s（100 Hz）；物理 0.002 s × 5 substeps |
| 地形 / 外力 | `StaticTerrain`；`qfrc_applied`、`xfrc_applied` 恒为 0 |
| 可用 walking 片段 | 只有 3 条 demo，**最长 8.39 s，没有 ≥10 s 的片段** |
| 版本兼容性 | **musclemimic_models 1.0.6 会让 mm-10m-2 在 2.5–2.8 s 内跌倒；1.0.5 正常走完**（见 `D_model_version_compatibility.md`） |

## A.2 目录与依赖

新仓库 `~/Documents/musclemimic` 为独立 git 仓库，上游源码放在 `external/`（不入 git）：

```
external/MuscleMimic            47b653d  (read-only, 未修改)
external/musclemimic_models     v1.0.5  ba69358
.venv                           Python 3.11.15
```

安装时 `pypi.org` 不稳定，使用清华镜像 + PyPI 作为补充索引（`distrax==0.1.7` 依赖只在 PyPI 上的
`tfp-nightly`）。`mujoco==3.4.0`、`jax==0.7.2`、`numpy==2.2.6` 按仓库 `pyproject.toml` 固定。
**没有安装 CUDA 版 jax**：本次全部用 MuJoCo CPU 后端。

> `musclemimic_models` 的依赖声明是 `>=1.0.2`（未固定上限）。今天直接 `uv sync` 会装到 1.0.6，
> 这正是复现跌倒的版本。本仓库显式固定到 1.0.5。

## A.3 checkpoint

- 下载：`huggingface_hub.snapshot_download("amathislab/mm-10m-2")`，通过 `HF_ENDPOINT=https://hf-mirror.com`
  （本机直连 `huggingface.co` 超时；镜像 200）。
- 仓库未设 gate，16 个文件，共 134 MB。
- 目录本身没有 `checkpoint_<step>` 名字，官方 `_canonicalize_resume_path()` 会创建**相对路径**符号链接
  `checkpoint_12500 -> artifacts/checkpoints/mm-10m-2`（在 `artifacts/checkpoints/` 下解析会变成
  `artifacts/checkpoints/artifacts/checkpoints/mm-10m-2`，即坏链），随后 orbax 报
  `Checkpoint at .../checkpoint_12500 not found`。
  处理：手工建立**绝对路径**符号链接 `artifacts/checkpoints/checkpoint_12500 -> .../mm-10m-2`，未改上游代码。
- checkpoint 内部记录：`step=1600000, update_number=12500, global_timestep=2048000000,
  num_envs=8192, num_steps=20, num_minibatches=128, backend=warp`。
- 逐文件 sha256 与整体 digest 见 `reports/env_audit.json`。

## A.4 模型与接口

由 `scripts/env_audit.py` 在真实构建的环境上读取：

| 量 | 值 |
|---|---|
| `nq / nv / nu` | 89 / 88 / 354 |
| `njnt / nbody / ngeom / nsite` | 83 / 102 / 416 / 2038 |
| 执行器类型 | 354 个 `mjDYN_MUSCLE`（Hill 型），`ctrlrange = [-1, 1]` |
| `opt.timestep` | 0.002 s |
| `n_substeps` | 5 |
| `env.dt`（控制周期） | 0.01 s → 100 Hz |
| `horizon` | 1000 控制步（= 10 s） |
| 根自由关节 | `root`（free joint，qpos[0:7]） |
| `root_height_healthy_range` | (0.6, 1.5) m |
| 观测 | 2418 维（关节位置/速度 + 肌肉长度/速度/力/兴奋/激活 + 4 个触觉 + 469 维 goal） |
| 动作 | 354 维，[-1, 1]，`DefaultControl` 线性映射到肌肉 ctrl |
| 地形 | `StaticTerrain`，非动态 |
| reset 时外力 | `qfrc_applied` 0，`xfrc_applied` 0 |

观测布局（`reports/obs_layout.json`）：`touch_r_foot=1945, touch_r_toes=1946, touch_l_foot=1947,
touch_l_toes=1948`，`GoalTrajMimic=1949..2417`。

reset 后 `data.qpos` 与参考轨迹第 0 帧**逐位相同**（round 到 1e-9），说明初始状态确实来自
`TrajInitialStateHandler` 的轨迹注入——这是合法的“初始化”，不是逐步写 qpos。

## A.5 motion 真实性

本机只有官方 demo 数据集里的 3 条 MyoFullBody 动作（`~/.musclemimic/caches/AMASS/MyoFullBody/gmr/`）：

| 片段 | 帧数 | 频率 | 片段时长 | 全片段净速度 | 最长连续行走段 | 该段速度 |
|---|---:|---:|---:|---:|---:|---:|
| `KIT/314/walking_medium09` | 773 | 100 Hz | **7.72 s** | 0.335 m/s | 3.00 s | 0.846 m/s |
| `KIT/348/turn_right03` | 618 | 100 Hz | 6.17 s | 0.033 m/s | 1.70 s | 0.567 m/s |
| `KIT/4/WalkInCounterClockwiseCircle04` | 840 | 100 Hz | **8.39 s** | 0.575 m/s（路程） | 6.16 s | 0.757 m/s |

频率核验：`qvel[:, 0:3]` 与 `diff(qpos[:, 0:3]) * 100` 的比值为 0.998（相关系数 0.9995），
30 Hz 假设会差 3.3 倍，因此 **100 Hz 是真实频率**，7.72 s / 8.39 s 的时长可信。

`walking_medium09` 的结构：0–3.2 s 静止站立，3.2–6.1 s 行走 2.5 m（4 步），6.1–7.7 s 再次静止。
所以它本身也不满足“10 s 连续行走”，更不满足“平均速度 ≥0.6 m/s”（整段平均只有 0.335 m/s）。

**长片段不可得（事实）**：官方完整重定向数据集 `amathislab/musclemimic-retargeted` 是 gated
（`extra_gated_*`，需在网页接受 AMASS 条款），本账号 token 访问返回 403。其中确实有更长的片段，
例如 `MyoFullBody/gmr/KIT/3/walk_6m_straight_line06_poses.npz`（12,039,580 B ≈ 1487 帧 ≈ 14.9 s），
但当前无法下载。demo 数据集里没有其它 walking 片段。

## A.6 官方 `fullbody/eval.py` 调用链（逐段核对）

1. `load_checkpoint()` → `_canonicalize_resume_path()` → `UnifiedCheckpointManager.load_checkpoint()`
   （orbax `Composite(train_state, config, metadata)`）→ `create_agent_state_from_orbax()`。
2. 恢复训练配置并强制 `apply_temporal_params()`：timestep=0.002、n_substeps=5，**不接受配置文件之外的
   隐式频率改动**；`verify_env_dt()` 会打印实际 `env.dt`。
3. `apply_eval_terminal_defaults()`：默认把 **validation** 的终止器装进 eval 环境，即
   `MeanRelativeSiteDeviationTerminalStateHandler(mean_site_deviation_threshold=0.5)`。
   **注意：该终止器只有“平均相对 site 偏差”，没有任何根高度/跌倒检查**。训练用的
   `MeanRelativeSiteDeviationWithRootTerminalStateHandler`（阈值 1.0）只在 `--strict_termination` 时使用。
4. `TaskFactory.make()` → `MyoFullBody`（把 `MjxMyoFullBody` 的 `Mjx` 前缀去掉）→
   `TrajectoryHandler` 加载 gmr 缓存 → `TrajInitialStateHandler` 在 reset 时用轨迹帧初始化。
5. `PPOJax.init_agent_conf()` + `align_agent_state()` 绑定网络 `apply_fn`。
6. 动作：`sample_actions()` 中 `pi.sample(seed=rng)`；deterministic 模式是把 `log_std` 置为 `-inf`
   （即方差为 0 的高斯采样 = 均值）。没有动作缩放/裁剪以外的隐藏处理，`DefaultControl` 把 [-1,1]
   线性映射到执行器 ctrl 区间。
7. 步进：`mujoco.mj_step(model, data, n_substeps=5)`，每个控制步推进 0.01 s。
8. 终止：`_is_done = absorbing or step>=horizon`；`_is_absorbing` 来自终止器。轨迹结束
   （`reached_trajectory_end`）也会置 done。
9. reset：**`play_policy()` / `run_with_trajectory_export()` 在 `done` 后立即 `env.reset()` 并继续跑**，
   直到总步数达到 `n_steps`。

### 影响验收语义的三点（已核实）

- **`--n_steps 1000` 不是“连续 10 s”**。它是跨 episode 的总步数预算；跌倒/片段结束后会自动 reset
  并重新计时。官方 demo 用 `n_steps=1000` 完全可能出现“3 个 episode 拼起来 10 s”。
- **`--eval_seed` 在回放路径上不起作用**。`play_policy()` / `run_with_trajectory_export()` 内部固定
  `rng = jax.random.key(0)`（`play_policy` 的 `rng=None` 分支），`args.eval_seed` 只传给 metrics 路径。
  要换种子必须自己传 rng。
- **默认 eval 终止器不检测跌倒**。只有 site 偏差 >0.5 m 才终止；物理跌倒可能一直跑到片段结束。

## A.7 版本兼容性（详见 D 报告）

MyoSuite 集成文档记录 `mm-10m-2` 基于 `musclemimic_models 1.0.5`。本机对照实验：

| 模型包 | seed 0/1/2 结果 |
|---|---|
| 1.0.5（官方文档配对版本） | 3/3 走完 7.72 s 片段，无跌倒 |
| 1.0.6（当前 main / `>=1.0.2` 会装到的版本） | 3/3 在 **2.47–2.83 s** 跌倒（根高 0.28 m，骨盆倾角 53°） |
| 1.0.6 但仅把 `leg/assets/*` 回退到 1.0.5 | 走完片段，无跌倒 |

1.0.6 相对 1.0.5 的改动只有 4 个 XML：
`arm/assets/myoarm_bimanual_body.xml`、`leg/assets/myolegs_assets.xml`、`leg/assets/myolegs_chain.xml`、
`torso/assets/myotorso_bimanual_chain.xml`。其中 `myolegs_assets.xml` 把**左膝**两个等式约束
`knee_angle_translation2_constraint_l`、`knee_angle_rotation3_constraint_l` 的 `polycoef` **整体取反**。
单因子回退实验表明跌倒由这一改动导致。

## A.8 未验证 / 待验证

- **没有**测试 MyoSuite 的 MuscleMimic 集成路径（本机未安装 myosuite；本次以官方
  `fullbody/eval.py` 为唯一基线）。因此“MyoSuite 集成缺少策略文件时回退到参考轨迹回放”这条路径
  未核实，也未在本报告中作为证据。
- 未核实 checkpoint 训练时实际使用的 `musclemimic_models` commit（checkpoint 未记录依赖版本）。
  “1.0.5 是配对版本”的依据是 MyoSuite 文档 + 本次配对回放的行为差异，而不是训练日志。
- 未测试 MJX/warp 后端（无 GPU）。
- 未取得 `musclemimic-retargeted` 访问权限，无法核对更长片段的真实速度。
