# E. 直线行走基线：审计、数据、训练与验收

日期：2026-10-31 ｜ 仓库 commit（审计时）：`07059b2`
硬件：RTX 5060 Laptop（8 GB，sm_120，26 SM）｜30 GiB RAM，**0 swap**｜20 vCPU｜112 GB 空闲盘

---

## E.1 先回答"我跑的是哪个流程"

**本仓库运行的是 MuscleMimic 的 PPO（JAX）流程，不是 Muscle-SuperMimic。**

| 证据 | 结论 |
|---|---|
| `fullbody/experiment.py` → `musclemimic.runner.engine.run_experiment` | 训练入口 |
| `conf_fullbody.yaml: algorithm: PPOJax` | 算法 = PPOJax |
| `optimizer_type: muon`，`ppo_config.num_steps/num_minibatches/...` | PPO 超参 |
| checkpoint `mm-10m-2` 元数据 `algo_version: PPOJax_v1` | 权重与算法一致 |
| 全盘 `grep -ri "supermimic"` 在 `~/Documents/musclemimic`、`~/Documents/muscle`、`~/Documents/muscle-rl` | **无任何匹配** |

因此：本仓库没有 Muscle-SuperMimic 的代码或权重，也不存在"混用两者配置"的风险——因为
SuperMimic 从未出现在本机。若你确实有另一份 SuperMimic 仓库，它不在当前工作区，本报告不对其作任何声明。

## E.2 版本与接口（冻结）

| 项目 | 值 | 来源 |
|---|---|---|
| MuscleMimic 源码 | `47b653db2bce25de834360d2ee6b2dd88142bd78` | `external/MuscleMimic` |
| musclemimic_models | **1.0.5**（`ba69358`） | 必须；1.0.6 会翻转左膝约束符号并导致跌倒 |
| MuJoCo / MJX | 3.4.0 | `pyproject.toml` |
| 训练后端 | `mjx_backend: warp`（mujoco-warp 内置在 mujoco-mjx 3.4.0） | 需 CUDA；CPU 上不可训练 |
| 模型 | MyoFullBody，`nq=89 nv=88 nu=354`（354 个 Hill 型肌肉） | `reports/env_audit.json` |
| 观测 / 动作 | 2418 维 / 354 维 ∈ [-1,1] | 同上 |
| 控制 / 物理步长 | 0.01 s（100 Hz）/ 0.002 s × 5 substeps | 同上 |
| 奖励 | `MimicReward`：qpos 0.1 + qvel 0.1 + root_pos 0.1 + root_vel 0.1 + rpos 0.6 + rquat 0.01 + rvel 0.1 | `conf_fullbody.yaml` |
| 终止（训练） | `MeanRelativeSiteDeviationWithRootTerminalStateHandler`，site/root/root_orient 阈值均 1.0 | 同上 |
| 已有 checkpoint | `amathislab/mm-10m-2`，update 12500，1.6e9 步，8192 envs | `reports/env_audit.json` |

运动数据格式（`*.npz`，重定向缓存）：

```
qpos (T,89) qvel (T,88) xpos (T,102,3) xquat (T,102,4) cvel (T,102,6)
subtree_com (T,102,3) site_xpos (T,17,3) site_xmat (T,17,9)
split_points, joint_names(83), body_names(102), site_names(17),
frequency, njnt/nbody/nsite, jnt_type, body_*/site_* 静态字段, metadata
```

## E.3 数据审计：有没有 10 秒直线步行？

`scripts/motion_straightness.py`（新增）对**本地全部** 7 条片段测量时长、路径速度、
侧向范围、2 s 内最大航向变化、最长直线窗口、支撑相比例、骨盆最低高度。
完整结果：`reports/motion_straightness.json`。

| 片段 | 时长 | 路径速度 | 侧向范围 | 2s 内最大转向 | 最长直线窗口 | ≥10 s 且直线 |
|---|---:|---:|---:|---:|---:|---|
| walk_6m_straight_line06 | 14.90 | 0.649 | **2.00 m** | **86.4°** | 6.84 s | ✗ |
| walk_6m_straight_line04 | 7.35 | 0.803 | 2.01 m | 24.9° | 6.76 s | ✗ |
| walking_medium09 | 7.72 | 0.343 | 0.09 m | 52.4° | 2.68 s | ✗ |
| turn_right03 | 6.17 | 0.369 | 0.23 m | 0.0° | 1.69 s | ✗ |
| walking_slow08 | 8.51 | 0.234 | 0.15 m | 50.9° | 1.26 s | ✗ |
| WalkInCounterClockwiseCircle04 | 8.39 | 0.575 | 1.68 m | 154.4° | 1.11 s | ✗ |
| WalkInCounterClockwiseCircle08 | 8.53 | 0.606 | 1.92 m | 141.2° | 0.87 s | ✗ |

> **结论：本地没有 ≥10 s 的直线步行数据。**
> 唯一 ≥10 s 的片段（`walk_6m_straight_line06`）含 180° 往返转身，侧向范围 2.0 m。
> 其"6 m 直走"段只有 6.84 s。
> 更早的全区扫描（1190 条）显示 testing 组最长仅 8.53 s，且只有这一条 ≥10 s。
> **没有用循环播放或延长评估时长冒充长数据。**

补充：当前 Hugging Face token 是 OAuth 类型，已过期（`401`），无法再下载其它片段；
本次审计因此限于本地已有的 7 条。

### E.3.1 数据准备方案（可复核，有明确误差声明）

`scripts/prepare_straight_walk.py`（新增）按**声明的直线窗口**抽取、旋转对齐到 +x、
拼接并重算全部 MuJoCo 派生量：

```
segment 1: KIT/3/walk_6m_straight_line04_poses  0.13–6.89 s  (6.76 s)
segment 2: KIT/3/walk_6m_straight_line06_poses  3.75–10.59 s (6.84 s)
crossfade: 0.20 s
```

产物：`~/.musclemimic/caches/AMASS/MyoFullBody/gmr/KIT/3/straight_stitched_10s.npz`
清单：`reports/data_prep_straight_stitched.json`

| 指标 | 值 |
|---|---|
| 时长 | **13.39 s** |
| 帧数 / 频率 | 1340 / 100 Hz |
| 路径长度 / 平均速度 | 11.70 m / **0.874 m/s** |
| 侧向范围 | **0.176 m** |
| 净转向 | 7.5° |
| 最长直线窗口（±20° 容差） | **13.27 s** |
| 2 s 内最大航向变化 | 20.98°（略超 20° 容差，出现在拼接缝附近） |
| 骨盆最低高度 | 0.917 m |
| 移动时支撑相比例 | 0.997 |
| 拼接前后的接缝位移 | 8.9e-16 m（数值零） |

**必须声明的误差**（也写进了 npz 的 `metadata`）：

1. 0.2 s 交叉淡入区的 qpos/qvel 是插值，**不是动力学一致的过渡**，该区间足地接触可能失真；
2. 接缝处根节点速度有小间断；
3. 数据仍来自几秒钟真实动作，步态多样性低于真正的 10 s 录制；
4. 该片段同时用作训练与验收参考 → **验收是 in-sample 的**，报告中已注明。

## E.4 独立评估脚本

沿用并扩展 `scripts/eval_walk.py`（新增航向误差指标）：

- 单回合、**只 reset 一次**、不拼接 episode；
- 每控制步记录 `data.time`、`qpos/qvel`、动作、肌肉 `ctrl`/`act`/`actuator_force`、
  逐侧接触法向力与**接触点**滑移、参考帧与相位、终止标记；
- 独立检查 `qfrc_applied`/`xfrc_applied` 恒 0、`data.time` 单调、参考帧单调、
  qpos 不跳回 reset 值（无隐式 reset / 无状态覆盖）；
- 结束原因：`time_limit_success / physical_fall / reference_deviation / clip_end /
  numeric_anomaly / reset_or_state_override / other`；
- 新增 `max/mean_heading_error_deg`（策略行进方向 vs 参考行进方向）。

## E.5 GPU 预算（实测，不是估算）

训练环境：**独立** `.venv-train`（`.venv` 保持冻结，审计结论不受影响）。
jax 0.7.2 + `jax-cuda12-plugin==0.7.2[with-cuda]`，实测 `jax.devices() = [CudaDevice(id=0)]`。

| envs | 峰值显存 | 峰值 RSS | 稳态吞吐 | 备注 |
|---:|---:|---:|---:|---|
| 8（冒烟） | — | — | — | 8 次更新 + checkpoint 保存成功 |
| 256 | 6736 MiB | 5.0 GB | ~1820 env-step/s | 4.0 s / 更新 |
| **512** | **6986 MiB** | **5.1 GB** | **~2540 env-step/s** | **4.0 s / 更新，留 ~1.1 GB 余量** |
| 512 + `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` | OOM | — | — | warp 在 XLA 池外分配，**不要调高该比例** |

- 8 GB 下可用配置：`num_envs=512, num_steps=20, num_minibatches=8`，网络 `1024×5`。
- 官方默认 `num_envs=8192` 在本机**不可运行**（需 ~8× 显存）。
- 吞吐换算：约 **9.1M env-step/小时**。官方 checkpoint 训练量 2.048e9 步
  → 约 **225 小时**；即使只做 1e8 步也要 ~11 小时。
- 系统内存：5 GB/进程，30 GiB 无 swap 下可安全单进程；**不要并行多进程**。

---

## E.6 训练运行（实测资源）

命令（可复现）：

```bash
NENV=512 STEPS=20 MINUTES=20 bash scripts/train_straight_bounded.sh
```

| 项目 | 值 |
|---|---|
| run 目录 | `runs/T_straight_512x20_300u/` |
| 配置 | 512 envs × 20 steps，300 次更新 = **3,072,000 env-steps** |
| 网络 | actor/critic 各 `1024×5`（与 checkpoint 同构） |
| 优化器 | Muon，lr 4e-4，linear anneal |
| 参考 | `KIT/3/straight_stitched_10s`（13.39 s，0.874 m/s） |
| 墙钟 | **18:03** |
| 吞吐 | **≈2837 env-steps/s**（含 JIT 编译） |
| 峰值显存 | **7043 MiB / 8151 MiB**（86%） |
| 峰值系统内存 | **3.9 GB / 30 GiB** |
| 检查点 | `checkpoint_150`、`checkpoint_300`（可续训，orbax） |

冒烟测试已确认（在 8 envs 下）：观测 2418 / 动作 354 维度正确、肌肉激活 ∈ [0,1]、
`actuator_force` 非零、足地接触发生、PPO 更新发生、checkpoint 保存与恢复可用。

**肌肉驱动链路证据**（`scripts/check_muscle_chain.py`，MuJoCo 回放）：

| 策略 | \|ctrl\| 均值 | ctrl 逐个元素饱和率 | act 范围 | \|force\| 均值 / 峰值 | 根节点速度 |
|---|---:|---:|---|---:|---:|
| 发布 checkpoint（2e9 步） | 0.843 | 69.3% | [0, 1] | 131.9 N / 3270 N | 0.772 m/s |
| 本次 3M 步策略 | 0.870 | 74.4% | [0, 0.999] | 168.7 N / 3373 N | 1.043 m/s |

`qfrc_applied` / `xfrc_applied` 全程为 0；`data.time` 单调；参考帧单调；无隐式 reset。
→ 运动确实由肌肉力经 MuJoCo 动力学产生。

## E.7 验收结果：**未通过**

冻结协议：`--mode stochastic --seconds 10 --seeds 0..19`，参考 `straight_stitched_10s`。

### E.7.1 本次训练的策略（3.07M 步）：0/20

| 指标 | mean | max |
|---|---:|---:|
| 时长 | **1.24 s** | 1.35 s |
| 结束原因 | **physical_fall 20/20** | — |
| 前进速度 | **−0.689 m/s（倒退）** | −0.579 |
| 航向误差 | **173.7°** | 180.0° |
| 骨盆倾角 | 83.0° | 91.1° |
| 根高最低 | 0.505 m | 0.554 m |

逐种子：`reports/table_A_trained_10s.md`。
**这是典型的"训练量严重不足"**：3M 步只占官方 2.048e9 步的 0.15%，策略还没学会站立，
表现为向后倒（航向几乎 180°）。

### E.7.2 对照：官方发布 checkpoint（2.048e9 步）在同一参考上：0/20

| 指标 | mean | max |
|---|---:|---:|
| 时长 | **9.93 s** | 10.00 s |
| 结束原因 | `time_limit_success` 19/20、`physical_fall` 1/20 | — |
| 前进速度 | 0.727 m/s（参考 0.874） | 0.775 |
| **最大侧偏** | **1.056 m** | **2.102 m** |
| 航向误差（**已修正**，见 F.4） | 平均 8.6°、最大 33.1° | 平均 8.6°、最大 33.1° |
| 骨盆倾角 | 17.6° | 64.9° |
| 根高最低 | 0.875 m | 0.902 m |
| 足滑 / 前进距离 | 0.087 m/m | — |
| 触地事件 | 31.5 / 回合 | — |

逐种子：`reports/table_A_released_10s.md`。

> 更正（2026-10-31）：本表原写的"平均航向误差 86.3°"是**指标实现错误**（取的是逐回合最大值的均值，
> 且用单步位移方向计算）。修正为 0.5 s 窗口 PCA 后，官方策略平均航向误差为 **8.6°**。
> 详见 `reports/F_single_factor_reference.md` F.4 与 `reports/metrics_recomputed.json`。

**结论**：即使是最强的官方权重，在这条拼接直线参考上也只能"不跌倒"（19/20 走满 10 s），
但**侧偏 0.54–2.10 m，远超 0.35 m 阈值**，且速度系统性低于参考 17%。0/20 通过。

### E.7.3 三类证据

| 证据 | 结果 | 文件 |
|---|---|---|
| 参考轨迹运动学 | 13.39 s，11.70 m，0.874 m/s，侧向 0.176 m，支撑相 0.997 | `reports/motion_straightness.json`、`reports/data_prep_straight_stitched.json` |
| 策略在 MuJoCo 中的动力学回放 | 19/20 走满 10 s，无隐藏外力，肌肉产力 132 N 均值 | `runs/A_ref_mm10m2/`、`reports/muscle_chain_evidence.json` |
| 轻微初始扰动恢复 | σ=0.05 rad 关节噪声、3 seeds：全部**未跌倒**，走满 10 s（侧偏 0.41/0.79/0.56 m） | `runs/A_perturb/` |

扰动实验说明：该策略对 0.05 rad 的初始扰动**不会倒**，但跟踪误差本来就已经很大，
所以"恢复"在这里只意味着"没有立刻失败"，不能作为跟踪质量证据。

## E.8 没达到目标时的失效环节定位

按证据排序（不是猜测）：

1. **训练规模（决定性）**：3.07M 步 = 官方预算的 0.15%。策略连站立都没学会（1.24 s 就倒、
   航向 174°）。本机 8 GB 只能开到 512 envs（官方 8192），吞吐 2.8k steps/s，
   复现官方预算需 ~225 小时。
2. **数据时长与质量（结构性）**：本地无 ≥10 s 直线片段。拼接片段的 0.2 s 交叉淡入区不是
   动力学一致的过渡；两段步速不同（0.803 vs 更高），造成参考速度在接缝处跳变。
   证据：官方权重在**原始** `walk_6m_straight_line06` 上侧偏 0.582 m，在**拼接**片段上
   恶化到 1.056 m——拼接本身增加了难度。
3. **横向误差积累**：官方权重的失败模式与 10 s 验收一致（转身/漂移后沿平行线走），
   不是跌倒，也不是速度不足。这是策略层面的横向跟踪缺陷，不是本次训练引入的。
4. 未发现模型/版本不一致：`musclemimic_models 1.0.5`、`mujoco 3.4.0`、checkpoint
   `mm-10m-2` 与 `configs/eval_walk.json` 一致；肌肉链路与接触都正常。
5. 未发现数值异常：20+20 个回合中 `numeric_anomaly` 与 `reset_or_state_override` 均为 0。

## E.9 下一项最小可检验实验

**先解决数据，再谈训练**（单因子）：

> 用 GMR 重新重定向一段 **≥10 s、单一直线、恒定速度**的 KIT/AMASS 行走（而非拼接），
> 在同一冻结协议下重跑官方 `mm-10m-2` 的 20 种子。

判据：若侧偏从 1.06 m 降到 <0.35 m，则说明主因是拼接/参考质量，训练规模不是当前瓶颈；
否则说明横向跟踪是策略本身缺陷，下一步才需要训练或奖励改动。

**次选（不改数据）**：把侧偏阈值与"平行偏移"解耦——用"相对参考路径的横向误差"作为唯一
横向判据，并单独报告"是否存在平行偏移"，以区分"跟踪滞后"与"几何错误"。

**不建议**直接启动大规模训练：在参考尚不合格时，训练只会拟合一个含接缝的参考。

## E.10 视频与图（人工查看用）

| 文件 | 内容 | 规格 |
|---|---|---|
| `reports/video/straight10s_mm10m2_seed0_drift.mp4` | 官方 checkpoint 在拼接直线参考上的 10 s 回放（seed 0） | 640×480，20 fps，200 帧，10.0 s |
| `reports/video/straight10s_trained3M_seed0_fall.mp4` | 本次 3.07M 步策略（seed 0），1.22 s 跌倒 | 640×480，50 fps，61 帧 |
| `reports/figures/straight10s_path_and_height.png` | 俯视轨迹（策略 vs 参考）+ 根高/骨盆倾角时序 | 130 dpi |

每帧叠加真实 `step`、`t = data.time - reset_time`、瞬时骨盆/躯干倾角，与同名 `.npz` 的 `time` 对齐。

**看图/看视频时要注意的一点**（避免误判）：seed 0 的策略末位置 `(7.48, -0.55)`，参考 `(8.82, 0.05)`。
两者欧氏距离 1.46 m，但其中约 1.34 m 是**前进滞后**（速度 0.751 vs 参考 0.874），
真正横向（cross-track）分量是 0.672 m —— 仍然超过 0.35 m 阈值，但"偏了 1.5 米"的说法会夸大横向问题。
