# B. 原版策略复现（未训练、未改环境）

全部使用：MuJoCo CPU 后端、`amathislab/mm-10m-2`（update 12500）、`musclemimic_models 1.0.5`、
`KIT/314/walking_medium09_poses`（773 帧 / 7.72 s）。stochastic 与 deterministic 分开记录，不混合汇总。

## B.1 官方入口原样运行

```bash
# stochastic
PYTHONPATH=external/MuscleMimic .venv/bin/python external/MuscleMimic/fullbody/eval.py \
  --path artifacts/checkpoints/checkpoint_12500 \
  --motion_path KIT/314/walking_medium09_poses \
  --use_mujoco --no_render --stochastic --eval_seed 0 --n_steps 1000 \
  --export_trajectory --trajectory_dir runs/B_official/stochastic_seed0

# deterministic（不加 --stochastic）
... --n_steps 1000 --export_trajectory --trajectory_dir runs/B_official/deterministic_seed0
```

| 运行 | episode | 起始参考帧 | 步数 | 结束原因（据轨迹判定） | 根高 min | 骨盆倾角 max |
|---|---|---:|---:|---|---:|---:|
| stochastic（随机起点） | 0 | 70 | 702 | 片段结束（70→772） | 0.893 | 7.8° |
| | 1 | 763 | 9 | 片段结束 | 0.944 | 1.5° |
| | 2 | 129 | 289 | **参考偏差提前终止**（129→418，未到 772） | 0.892 | 2.7° |
| deterministic（随机起点） | 0 | 365 | 407 | 片段结束（365→772） | 0.887 | 7.9° |
| | 1 | 24 | 593 | **参考偏差提前终止**（24→617） | 0.878 | 7.5° |
| stochastic + `--traj_index 0 --traj_start_step 0` | 0 | 0 | **772** | 片段结束 | 0.890 | 7.9° |
| | 1 | 0 | 228 | （自动 reset 后继续，仅用于说明） | — | — |

**没有任何一个 episode 发生物理跌倒**。官方播放里出现的“提前结束”是默认 eval 终止器
（`MeanRelativeSiteDeviationTerminalStateHandler`, 0.5 m）判定的参考偏差，此时人体仍然直立
（根高 0.89 m、骨盆倾角 <3°）。这与“看到人体倾倒”不是一回事。

固定从第 0 帧开始时，stochastic 策略完整走完 772 步（7.72 s），根节点始终自由、未触发任何终止器。

## B.2 两种模式

- **stochastic**：`pi.sample(seed=rng)`。官方 `validation.deterministic=false`，这是官方推荐的评估模式。
- **deterministic**：官方实现是把 `log_std` 置为 `-inf` 后仍然 `sample`，等价于取均值。
  本仓库脚本用 `pi.mode()` 实现同一语义（对角高斯下逐元素相同）。
- deterministic 对 seed 不敏感：seed 0/1/2 的轨迹逐位一致（`max |Δqpos| = 0`），因此后续 20 seed
  验收只在 stochastic 模式下做。

## B.3 单回合（固定起点）结果

`scripts/eval_walk.py`，每个 seed 一个独立 episode、只 reset 一次、不中途重置：

| seed | 模式 | 结束原因 | 时长 | v_forward | 侧偏 | 骨盆/躯干倾角 | 根高 min | 片段进度 |
|---:|---|---|---:|---:|---:|---|---:|---|
| 0 | stochastic | clip_end | 7.72 s | 0.321 | 0.139 m | 7.9° / 30.0° | 0.890 | 772/772 |
| 1 | stochastic | clip_end | 7.72 s | 0.322 | 0.119 m | 8.4° / 30.0° | 0.888 | 772/772 |
| 2 | stochastic | clip_end | 7.72 s | 0.332 | 0.051 m | 8.2° / 29.5° | 0.887 | 772/772 |
| 0 | deterministic | clip_end | 7.72 s | 0.311 | **0.606 m** | 8.2° / 29.3° | 0.889 | 772/772 |
| 1 | deterministic | clip_end | 7.72 s | 0.311 | **0.606 m** | 8.2° / 29.3° | 0.889 | 772/772 |
| 2 | deterministic | clip_end | 7.72 s | 0.311 | **0.606 m** | 8.2° / 29.3° | 0.889 | 772/772 |

（侧偏列是 `max_lateral_deviation = max(侧滑积分, 到参考轨迹的垂直距离)`。deterministic 的均值动作
有约 0.5 m 的固定横向偏置：骨盆 y 走到 +0.52 m，而参考是 −0.05 m。它不跌倒，但侧偏超过 0.35 m。）

参考片段本身只有 7.72 s，所以 `clip_end` 是数据上限，不是策略失败。参考片段平均速度 0.335 m/s，
stochastic 策略 0.321–0.332 m/s，跟踪误差很小（骨盆 XY 误差均值 0.07 m，最大 0.21 m）。
deterministic 速度 0.311 m/s、骨盆 XY 误差均值 0.28 m，并在横向上持续偏置。

## B.4 “动作经肌肉动力学产生运动”的证据

对 seed 0 的单回合逐控制步记录（`runs/C_acceptance/stochastic/stochastic_seed0.npz`）：

- 策略动作 354 维，`|ctrl|` 均值 **0.84**，饱和比例 0（从不贴 ±1）；
- 肌肉激活 `data.act` 均值 0.47，范围 [0, 1]，逐帧变化；
- `data.actuator_force` 非零且随相位变化；
- `data.qfrc_applied` 与 `data.xfrc_applied` **恒为 0**（没有隐藏外力）；
- 步进期间从未调用 `env.reset()`（`reset_count=1`）、`data.time` 严格按 0.01 s 递增、
  `(traj_no, subtraj_step)` 单调、qpos 未出现“跳回 reset 值”；
- 地形为 `StaticTerrain`，`_reset_on_edge()`（会把根 XY 清零的代码）不会被调用；
- 因此 qpos/qvel 完全由 `mj_step` 从肌肉 ctrl 积分得到。参考轨迹只进入观测/奖励/终止器，未写入 `data`。

唯一的 qpos 写入发生在 reset 时的 `TrajInitialStateHandler`（官方设计），之后没有任何状态注入。

## B.5 判断

- 原版策略在 1.0.5 + 官方 checkpoint 下**能够稳定走完官方 demo 片段**，不是“一走就倒”。
- stochastic 与 deterministic 都能走完整段且不跌倒，但 deterministic 横向漂移 0.61 m，
  不满足侧偏判据；官方推荐的 stochastic 满足（20/20，最大 0.26 m）。
- 之前观察到的倾倒，在本机可复现的机制是 `musclemimic_models` 版本不匹配（见 D 报告），
  而不是 checkpoint 本身不能走。
- 由于片段只有 7.72 s，B 阶段**无法验证 10 s 连续行走**。
