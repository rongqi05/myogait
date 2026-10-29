# 决策：MyoFullBody + MuscleMimic 能否作为正常人行走基线

日期：2026-10-29
冻结配置：`configs/eval_walk.json`
证据：`reports/A_environment_audit.md`、`reports/B_reproduction.md`、
`reports/C_acceptance.md`、`reports/D_model_version_compatibility.md`、`reports/*.json|csv|md`

## 结论

> **因数据不足无法判定（10 s / 0.6 m/s 验收）。**
>
> 在**本地可获得的全部**参考片段上，官方 checkpoin 与 1.0.5 模型包可以稳定、无跌倒地走完整条
> 参考（walking_medium09：20/20；circle：20/20），但没有任何一条可用片段达到 10 s，且没有一条
> 片段的平均速度达到 0.6 m/s。因此“10 s 连续正常行走基线”这一门槛**尚未被验证**，既不能判通过，
> 也不是策略失败。

| 子问题 | 结论 | 依据 |
|---|---|---|
| 官方策略能否在官方 demo 上正常走（不跌倒） | **通过（stochastic）** | 20/20 走完 7.72 s，根高 ≥0.883 m，骨盆倾角 ≤9.5° |
| 除 10 s/速度外的其余判据 | **通过（stochastic 20/20；circle 18/20）** | 侧偏 ≤0.258 m（walk），≤0.426 m（circle 有 2 个 seed 超 0.35） |
| deterministic 模式 | **未通过侧偏判据** | 20/20 不跌倒，但横向漂移 0.606 m |
| 最长可验证的连续行走 | **8.39 s**（circle 片段） | 20/20 clip_end，速度 0.514 m/s |
| 10 s 连续行走 | **未验证** | 无 ≥10 s 片段；最长 8.39 s |
| 平均速度 ≥0.6 m/s | **未验证（参考本身就不达标）** | walking_medium09 参考 0.335 m/s；circle 0.575 m/s |
| “官方 checkpoint 回放会跌倒” | **可复现，原因是模型包版本不匹配** | 1.0.6：3/3 在 2.5–2.8 s 跌倒；1.0.5：3/3 正常；单因子回退腿文件后恢复 |
| 是否修改了奖励/模型/观测/肌肉力量 | **没有** | 只固定了 `musclemimic_models==1.0.5` |

### 为什么不是“未通过”

判据里“未通过”的含义应是“在满足数据前提的协议下，策略失败”。当前：

- stochastic 20/20 个独立 episode 都**没有** physical_fall / numeric_anomaly / reset_or_state_override；
- 20/20 都在参考结束时才停下（`clip_end`），倾角、侧偏（≤0.258 m）、根高全部满足阈值；
- 阻塞项只有 `duration_ge_10s`、`reference_not_ended`、`avg_forward_speed_ge_0.6`，
  这三项都直接由参考片段长度/速度决定，策略跟踪误差只有 0.007 m/s（0.328 vs 0.335）；
- deterministic 模式另有一个真实缺陷（0.61 m 横向漂移），但它不影响 stochastic 的结论，
  也不能用“数据不足”掩盖——已单独记录为 `C_acceptance.md` C.4 的未通过项。

所以是**数据前提不满足**，而不是策略不达标。

## 下一步（需要用户动作）：取得 ≥10 s 的 walking 参考

唯一缺口是参考片段。官方完整重定向数据集中已有满足长度的动作，例如
`MyoFullBody/gmr/KIT/3/walk_6m_straight_line06_poses.npz`（12,039,580 B ≈ 1487 帧 ≈ 14.9 s），
但 `amathislab/musclemimic-retargeted` 是 gated 数据集，本机 token 访问返回 403。

用户需要在该数据集页面点击 “I agree and request access”（auto-approve，只需一次），然后：

```bash
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("amathislab/musclemimic-retargeted", repo_type="dataset",
                  local_dir="artifacts/motions/retargeted",
                  allow_patterns=["MyoFullBody/gmr/KIT/3/walk_6m_straight_line06_poses.npz"])
PY
# 放到 gmr 缓存，供官方 loader 使用
mkdir -p ~/.musclemimic/caches/AMASS/MyoFullBody/gmr/KIT/3
cp artifacts/motions/retargeted/MyoFullBody/gmr/KIT/3/walk_6m_straight_line06_poses.npz \
   ~/.musclemimic/caches/AMASS/MyoFullBody/gmr/KIT/3/
# 重新做 20 seed 验收（协议/阈值/代码都不变）
PYTHONPATH=external/MuscleMimic:scripts .venv/bin/python scripts/eval_walk.py \
  --mode stochastic --seeds $(seq 0 19) \
  --motion_path KIT/3/walk_6m_straight_line06_poses --out_dir runs/C_acceptance_long
```

下载后先跑 `scripts/motion_stats.py`，确认该片段 (a) ≥10 s、(b) 全片段平均速度 ≥0.6 m/s。
如果片段仍含较长站立段导致平均速度不足，则把“平均速度”改为**行走窗口内**的速度（脚本已同时输出
`walking_window_forward_speed`），并在报告中明确口径，不要修改 10 s 时长判据。

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
# 0) 环境
uv venv .venv --python /home/zrq/miniconda3/envs/hemi_msk/bin/python
uv pip install --python .venv/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://pypi.org/simple <见 README>
uv pip install --python .venv/bin/python --no-deps ./external/musclemimic_models   # v1.0.5

# 1) 审计
PYTHONPATH=external/MuscleMimic:scripts .venv/bin/python scripts/env_audit.py

# 2) 参考片段统计
.venv/bin/python scripts/motion_stats.py

# 3) 单回合 / 20 seed 验收
PYTHONPATH=external/MuscleMimic:scripts .venv/bin/python scripts/eval_walk.py ...
```
