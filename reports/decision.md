# 决策：MyoFullBody + MuscleMimic 能否作为正常人行走基线

日期：2026-10-29（更新：取得完整数据集访问后完成 10 s 验收）
冻结配置：`configs/eval_walk.json`（协议 v1.1）
证据：`reports/A_environment_audit.md`、`B_reproduction.md`、`C_acceptance.md`、
`D_model_version_compatibility.md`、`reports/*.json|csv|md`、`reports/video/`

## 结论

> **未通过 10 s 验收（3/20，目标 18/20）。**
>
> 冻结的模型、配置、checkpoint 与评价代码下，在数据集中**唯一**满足"时长 ≥10 s 且前 10 s 参考
> 速度 ≥0.6 m/s"的行走片段（`KIT/3/walk_6m_straight_line06`，14.90 s）上：
> **20/20 跑满 10.00 s、20/20 无物理跌倒、速度/倾角/根高全部达标**，
> 但 **17/20 的最大侧偏 >0.35 m**（最大 1.19 m），只有 3 个 seed 满足全部判据。
>
> 失败模式**不是跌倒或走不动**，而是**180° 转身时转弯半径偏大，转完后沿参考平行线行走**，
> 横向偏移被保留。因此该策略目前不能直接作为"正常人 10 s 连续行走的动态基线"，
> 但可以作为**短距离直行基线**（7.72 s 直行片段：20/20 侧偏 ≤0.275 m）。

| 子问题 | 结论 | 依据 |
|---|---|---|
| 10 s 连续行走（全部判据） | **未通过 3/20** | `reports/table_10s_long.md` |
| 10 s 内不跌倒 | **通过 20/20** | 根高 min 0.890 m，骨盆倾角 ≤17.3°，无 numeric anomaly |
| 10 s 平均前进速度 ≥0.6 m/s | **通过 20/20** | 0.634–0.700 m/s |
| 10 s 最大侧偏 ≤0.35 m | **未通过 17/20** | 0.271–1.192 m |
| 倾斜/根高判据 | **通过 20/20** | ≤37.7° / ≥0.890 m |
| 短距离直行（7.72 s） | 通过 20/20 | 侧偏 ≤0.275 m，速度 0.328（参考 0.335） |
| 之前"回放会倒" | 已定位为 `musclemimic_models` 1.0.6 | 见 D 报告 |

### 为什么这次是"未通过"而不是"数据不足"

取得访问后，数据集里有 1190 条动作，但：
- **只有 1 条** ≥10 s（14.90 s），且属于**训练组**——因此 10 s 结果是 **in-sample**，偏乐观；
- 官方 testing 组最长 8.53 s，物理上无法满足 10 s。

前置条件已经满足，验收**已经执行**，所以结论是"未通过"，不是"无法判定"。
但同时必须承认这不是 out-of-sample 证据：策略在训练集片段上仍未通过侧偏判据，说明该缺陷是真实的。

### 失败根因（已定位到状态量，不依赖最终画面）

- 转身段（0–4.2 s）到参考路径的最大距离平均 0.347 m，14/20 仍在 0.35 m 内；
- 直行段（4.2–10 s）平均 0.495 m，仅 5/20 在 0.35 m 内 → 偏移在转身时产生并被保留；
- seed 10：t=9 s 时 root=(−4.10, 0.48)，ref=(−3.45, −0.41)，到路径真值 1.10 m；
- 指标与"到参考折线的最近距离"一致（1.192 vs 1.220 m），排除度量伪影；
- 三个通过 seed（0/7/18）的共同点：转身更贴参考。

### 与官方评测口径的差异（已核实）

- 官方 `--n_steps 1000` 是**跨 episode 总预算**，跌倒/片段结束后自动 reset 并继续计时，
  不能当作"连续 10 s"。
- 官方 eval 默认终止器（`MeanRelativeSiteDeviationTerminalStateHandler`, 0.5 m）**不含跌倒检查**。
- `--eval_seed` 在回放路径上**不生效**（内部固定 `key(0)`）。本脚本自己传 RNG 种子。
- checkpoint 的 `horizon=1000` 步 = 正好 10.0 s；已用 `--horizon 3000` 复跑，确认 10 s 由本脚本
  的时钟判定而非环境截断，结果不变。

## 下一步

### 立即可做（不需要重新训练）

1. **把基线定位为"短距离直行"**：用 `walking_medium09`（7.72 s）或
   `WalkInCounterClockwiseCircle04`（8.39 s）作为可用的健康人参照，明确标注时长上限。
2. 申请/生成**≥10 s 的直线行走参考**（这是数据侧问题）：官方 testing 组没有；可考虑用 GMR 自行
   重定向一段 ≥10 s 的 AMASS/KIT 直线行走，再重跑同一协议。
3. 若坚持用现有 14.9 s 片段做偏瘫对比，可把"侧偏"判据放宽或改为
   "相对自身的横向前进占比"，但**必须在报告中同时给出 0.35 m 口径的结果**，不能只报放宽后的。

### 需要重新训练才能做的（下一阶段）

见下文"患侧 F0 缩放与拐杖接触"，本次未实现。

## 附：取得访问后的 10 s 验收（已执行）

一键脚本：`bash scripts/after_access_10s.sh`（下载 → 前置校验 → 20 seed 验收）。
本次实际执行的步骤：

1. 从 `amathislab/musclemimic-retargeted` 下载 `KIT/3/walk_6m_straight_line06_poses.npz`
   （12,039,580 B，1491 帧，14.90 s），复制到 `~/.musclemimic/caches/AMASS/MyoFullBody/gmr/KIT/3/`。
2. `scripts/motion_stats.py` 做前置校验：
   `acceptance_precondition_ok = 时长 ≥10 s 且前 10 s 路径速度 ≥0.6 m/s`
   （用路径长度而非净位移，因为片段含转身；两种速度都在 `reports/motion_stats.json` 里）。
   1190 条动作里只有这一条通过（0.752 m/s）。
3. `scripts/eval_walk.py` 跑 20 个 stochastic seed（默认 horizon），再用 `--horizon 3000` 复跑一次。
4. `--video --video_every 5` 导出 seed 0（通过）与 seed 10（侧偏失败）的 10 s 视频。

全部命令与参数见 `reports/C_acceptance.md` C8 节和 README。

## 下一阶段接入患侧 F0 缩放与拐杖接触所需改动（本次未实现）

### 1. 患侧 F0 缩放

- **先验证槽位语义**：`muscle-rl` 在 MS-Human-700 上得出 F0 位于 `gainprm[2]`（主动）与
  `biasprm[2]`（被动），前 9 槽为 `range[0], range[1], force(F0), scale, lmin, lmax, vmax, fpmax, fvmax`。
  **MyoFullBody 是不同模型，必须用同样的扰动–响应实验重新确认**（逐个缩放槽位，观察
  `data.actuator_force` 响应），不得直接照搬。
- **实现位置**：在 `_apply_spec_changes()` 之后、`MjModel` 编译前，对 `MjSpec` 的 actuator
  `gainprm/biasmpr` 逐肌肉缩放；必须**从不可变基准计算**（保存原始 F0 数组，每次按比例重算，禁止累乘）。
  左右腿肌肉按名称后缀 `_l/_r` 分组，患侧全身/上肢/下肢分别给缩放系数。
- **影响面**：F0 改变会改变动力学 → 官方 checkpoint 的观测分布与最优动作失效，
  **必须重新训练**；本次的 10 s 基线只能作为“健康人参考”，不能直接用于偏瘫评估。
- **最小单因子实验**：健康受试先做 F0=1.0 的复现（应复现本次 20/20 clip_end），
  再做患侧 0.9/0.7/0.5 三档，每档 5 seed，记录能走完片段的最大下降比例；预期判据：
  步长下降、支撑相比例上升、患侧触地力下降、骨盆横移增加。

### 2. 拐杖接触

- 模型层：新增 `cane` body（杖身 + 触地 geom + 手部 weld/ball 关节），会增加 `nq/nv/ngeom`；
  需要给 `MyoFullBody` 增加对应的 `site`（杖端）与接触参数（`solref/solimp/friction`）。
- 观测/动作层：杖端接触力、杖的位姿、健侧/患侧手部力需要作为新观测；
  若要让策略主动压杖，还需新增 1–2 维动作。**这一定会改变接口 → 必须重新训练**。
- 奖励层：可加入杖端竖直力/滑动惩罚、患侧负重比例项；这些都属于“重新训练”的范畴，
  本次按要求没有改动。
- 评估层：在 `eval_walk.py` 的日志中加入 `cane_tip_force`、`cane_slip`、`left/right foot force share`；
  验收仍需同时满足 10 s、无跌倒、速度/侧偏/倾角阈值，且拐杖不得替代患侧承重（需额外报告
  “患侧足最大法向力占体重比例”）。

### 3. 临界点搜索（后续）

- 固定 F0 缩放（如 0.9→0.3 共 7 档）× 辅助模式（无拐杖 / 健侧手拐杖 / 患侧手拐杖）× 5 seed；
- 指标：10 s 完成率、患侧承重比例、步态对称性、能量（`Σ|activation|`）；
- 定义“辅助效果转换点”为：**无辅助失败、有辅助成功**的最小 F0 档位，用 logistic/分段回归给出
  置信区间；不要在奖励或展示里预设拐杖一定更好。

## 复现清单

```bash
# 0) 环境（Python 3.11.15，见 README 的安装步骤）
uv venv .venv --python /home/zrq/miniconda3/envs/env_isaaclab/bin/python
uv pip install --python .venv/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://pypi.org/simple <见 README>
uv pip install --python .venv/bin/python --no-deps ./external/musclemimic_models   # v1.0.5
#    当前 .venv 已自包含（不依赖任何 conda 环境）；完整依赖快照见
#    reports/env_snapshots/musclemimic_venv_freeze.txt

# 1) 审计
PYTHONPATH=external/MuscleMimic:scripts .venv/bin/python scripts/env_audit.py

# 2) 参考片段统计
.venv/bin/python scripts/motion_stats.py

# 3) 单回合 / 20 seed 验收
PYTHONPATH=external/MuscleMimic:scripts .venv/bin/python scripts/eval_walk.py ...
```
