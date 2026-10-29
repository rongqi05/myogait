# D. 故障定位：`musclemimic_models` 版本不匹配

## D.1 问题

用户此前用官方 checkpoint 回放时看到“人体倾倒”。本报告回答两个问题：

1. 1.0.6 与 1.0.5 是否会产生不同的回放结果？（配对实验）
2. 如果是，最早从哪个状态量开始偏离、由哪个单因子导致？（单因子回退实验）

## D.2 配对实验设置

保持完全相同的：checkpoint（mm-10m-2, update 12500）、motion（walking_medium09）、起始帧（0）、
seed（0/1/2）、模式（stochastic）、终止器、评估脚本。只替换 `musclemimic_models`：

| 实验组 | 模型来源 | 关键 XML 版本 |
|---|---|---|
| A | `git archive v1.0.5` → `/tmp/mm_models_105`（与正式安装内容相同） | 1.0.5 |
| B | `git archive v1.0.6` → `/tmp/mm_models_106` | 1.0.6 (main) |
| C | B 的副本，仅把 `leg/assets/myolegs_assets.xml`、`leg/assets/myolegs_chain.xml` 换回 1.0.5 | 1.0.6 + 1.0.5 腿 |

用 `PYTHONPATH=/tmp/mm_models_XXX:external/MuscleMimic:scripts` 遮蔽已安装包，`dump_reset_state.py`
记录实际 `musclemimic_models.__file__` 与全部 XML 的 sha256（见 `reports/D_model_divergence/reset_*.json`）。

## D.3 结果

| 组 | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| A (1.0.5) | clip_end, 7.72 s, 根高 min 0.890 | clip_end, 7.72 s, 0.888 | clip_end, 7.72 s, 0.887 |
| B (1.0.6) | **physical_fall, 2.83 s**, 根高 0.280, 骨盆倾角 52.8°, 侧偏 0.83 m | **physical_fall, 2.47 s**, 0.289, 54.0°, 0.86 m | **physical_fall, 2.72 s**, 0.293, 52.9°, 0.84 m |
| C (混合) | clip_end, 7.72 s, 根高 0.890 | — | — |

A 组的 1.0.5 阴影安装与正式 1.0.5 安装结果逐位一致（排除“路径/屏蔽”本身的副作用）。

## D.4 初始条件相同，动力学第一步就分开

`scripts/dump_reset_state.py`（reset 后、第一步之前）：

| 量 | A (1.0.5) | B (1.0.6) | 是否相同 |
|---|---|---|---|
| `data.qpos` sha256 | `2176b11614a6…` | `2176b11614a6…` | **相同** |
| `data.qvel` | 相同 | 相同 | 相同 |
| 观测 sha256 | `df792a1a…` | `b4a8710a…` | 不同 |
| 不同的观测维度 | — | 仅 2 维：`touch_r_foot`(1945)、`touch_l_foot`(1947) | — |
| 最大差异 | — | `touch_l_foot` 差 **4.17**（归一化触觉值） | — |

也就是说：初始关节状态完全一样，但**左脚触地力**在 reset 时就已经不同，说明左腿/左脚的几何与约束
被改动了。由此产生的 354 维动作在第一步就相差 `max|Δctrl| = 1.2e-3`。

## D.5 最早偏离的状态量与时间（seed 0）

以 A 组为基准，逐步比较 `qpos`（89 维）：

| 时间 | 现象 |
|---|---|
| 0.01 s（第 1 个控制步后） | `‖Δqpos‖=0.020`；最大单项 `knee_angle_l` **3.7e-3 rad**，其次 `ankle_angle_l` 2.3e-3、`hip_flexion_l` 2.3e-3 |
| 0.02 s | `knee_angle_l` **1.47e-2 rad**，`hip_flexion_l` 9.2e-3，`ankle_angle_l` 8.5e-3，`knee_angle_rotation3_l` 3.9e-3 |
| 0.04 s | `‖Δqpos‖ > 0.05` |
| 0.05 s | `|Δ knee_angle_l| > 0.05 rad` |

B 组的跌倒时间线（seed 0）：

| 事件 | 时间 |
|---|---|
| 骨盆相对参考 XY 误差 > 0.2 m | 2.07 s |
| 根高 < 0.85 m | 2.21 s |
| 根高 < 0.80 m | 2.42 s |
| 骨盆 XY 误差 > 0.5 m | 2.50 s |
| 骨盆倾角 > 20° | 2.53 s |
| 根高 < 0.60 m（环境“健康范围”下限） | 2.67 s |
| 骨盆倾角 > 45° | 2.77 s |
| episode 结束（终止器） | 2.83 s |

最早的偏离量是**左膝角度**，时间 **0.01 s**；不是“最后画面里看到的躯干倾倒”。倾倒发生在 2.2 s 之后，
是关节层面误差持续放大的结果。

## D.6 单因子定位：左膝等式约束符号翻转

1.0.6 相对 1.0.5 只改了 4 个 XML：

- `model/arm/assets/myoarm_bimanual_body.xml`（肩部/肘部 wrap geom 位置）
- `model/leg/assets/myolegs_assets.xml`
- `model/leg/assets/myolegs_chain.xml`
- `model/torso/assets/myotorso_bimanual_chain.xml`（背部 wrap 圆柱半径 0.170→0.166）

腿部的实质改动是**左膝**两个等式约束的多项式系数整体取反：

```
knee_angle_translation2_constraint_l:
  v1.0.5  polycoef="-7.69254e-11 -0.00587971 0.00125622 2.61846e-06 -6.24355e-07"
  v1.0.6  polycoef=" 7.69254e-11  0.00587971 -0.00125622 -2.61846e-06 6.24355e-07"
knee_angle_rotation3_constraint_l:
  v1.0.5  polycoef="-1.08939e-08 -0.369499 0.169478 -0.0251643 -3.50498e-07"
  v1.0.6  polycoef=" 1.08939e-08  0.369499 -0.169478 0.0251643 3.50498e-07"
```

（右膝同名约束未改；此外 `knee_angle_rotation3_{r,l}` 的 joint range 从 ±0.262788 改为
`-0.262788..0.263`，属数值噪声级改动。）

**C 组（只回退腿文件）完全恢复行走**：7.72 s、clip_end、根高 0.890、骨盆倾角 7.9°。
因此“1.0.6 导致跌倒”的单一因子是腿文件中的左膝约束改动，而不是手臂/背部 wrap 改动。

## D.7 与用户此前观察的关系

- 仓库 `pyproject.toml` 写的是 `musclemimic_models>=1.0.2`，**没有上限**。今天按 README 全新安装会拿到
  1.0.6，从而 100% 复现跌倒（3/3 seed）。
- 这与“MyoSuite 集成文档说 mm-10m-2 基于 1.0.5”一致：checkpoint 与 1.0.5 配对，与 1.0.6 不配对。
- **不能**据此断言用户当时的跌倒一定是这个原因：我们没有用户当时的安装记录，也没有测试 MyoSuite
  集成路径（其缺少策略文件时可能回退到参考轨迹回放）。但这是目前唯一可复现、可单因子定位的机制。

## D.8 处理建议

1. 固定 `musclemimic_models==1.0.5`（本仓库已固定，并在 `reports/env_audit.json` 记录 XML 路径与哈希）。
2. 如果要用 1.0.6，应当用基于 1.0.6 重新训练的 policy，不能拿 mm-10m-2 直接回放。
3. 上游若要修，应把 `pyproject.toml` 的 `>=1.0.2` 改成与 checkpoint 匹配的精确版本，或在 checkpoint
   元数据里记录模型版本。
