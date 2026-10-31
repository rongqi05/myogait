# F. 单因子实验：用"未拼接的真实连续 ≥10 s 直线行走"替换拼接参考

日期：2026-10-31 ｜ 冻结基线：`ccea919` ｜ 清单：`reports/freeze_straight_reference.json`

## 结论（先说）

> **数据门槛未通过，实验未执行。**
>
> 本机没有任何"未拼接、连续 ≥10 s 的真实直线步行"原始运动，也没有运行 GMR-Fit 所需的
> SMPL-H 人体模型与依赖。按你给定的规则（"若找不到合格数据，停止在数据门槛未通过"），
> 我在这一步停止，**没有**用不合格片段训练或宣称完成实验。
>
> 但本次仍完成了两件必须做的事：**(1) 冻结基线；(2) 发现并修正了旧报告中一个指标错误**
> （"平均航向误差 86°"实际应为 **8.6°**），并用同一版本重算了新旧可比的两组结果。

---

## F.1 冻结基线

| 项目 | 值 |
|---|---|
| 仓库 commit | `ccea9193646f992f548ab5afbfcc7cb9b94853a6` |
| MuscleMimic 源码 | `47b653db2bce25de834360d2ee6b2dd88142bd78` |
| musclemimic_models | **1.0.5** |
| MuJoCo / numpy | 3.4.0 / 2.2.6 |
| checkpoint | `artifacts/checkpoints/checkpoint_12500` → `amathislab/mm-10m-2`，update 12500，digest `c8c7a964…` |
| 模型变体 | MyoFullBody，`disable_fingers=True`，**nq=89 / nv=88 / nu=354**，root free joint `root` |
| 评估 | `--mode stochastic --seconds 10 --seeds 0..19`，每回合只 reset 一次，MuJoCo CPU |
| 阈值（未改动） | 10.0 s、0.6 m/s、**侧偏 0.35 m**、倾角 45°、跌倒判据 0.6 m / 60° |
| 旧参考 20 种子结果 | `runs/A_ref_mm10m2/stochastic`（官方 checkpoint，0/20） |
| 保留确认集 | **种子 101–120**，本实验**未运行** |

`nq/nv/nu` 与参考缓存字段已核对：现有 `.npz` 提供 `qpos(89)/qvel(88)/xpos(102)/xquat/cvel/
subtree_com/site_xpos(17)/site_xmat(17)` 与静态字段，与 `musclemimic_models 1.0.5` 编译出的
MyoFullBody 一致（`reports/env_audit.json`）。

**关于格式的硬性提醒**：GMR-Fit 的中间产物是 `.pkl`（拟合形状 / 姿态对象），MuscleMimic 训练
缓存是上表字段的 `.npz`。两者**不能**靠改名互换；必须经仓库的 `ImaginedFactory` 路径
（`scripts/retarget_dataset.py` → `AMASSDatasetConf([motion])`）生成。本次未做任何改名。

## F.2 数据门槛审计

### F.2.1 本机已有数据

| 类别 | 状态 |
|---|---|
| 原始 AMASS / KIT（SMPL-H `poses/trans/betas`） | **无**。全盘搜索 `*amass* / *smpl*` 只命中重定向缓存目录与仓库源码，没有任何原始 mocap 文件 |
| SMPL-H 人体模型（`SMPL_NEUTRAL.pkl` / `SMPLH_*.pkl`） | **无**（需在 smpl.is.tue.mpg.de 注册后人工下载） |
| 预重定向缓存 | 8 个 `.npz`（7 条真实片段 + 1 条我上次拼接的 `straight_stitched_10s`） |
| `~/Documents/muscle/data` | 233 个 CMU BVH + 716 个患者 BVH + 552 个 BIO `.npy` + `walking_bio.pt`；**已单独审计，见 `G_muscle_data_audit.md`**：92 条 ≥10 s，但满足"长+直+正常"的为 **0**（仅 ZombieWalk / Frankenstein 通过直线窗口） |

### F.2.2 候选运动逐一筛查（重定向之前）

对本地 7 条**真实**片段逐条测量（`reports/motion_straightness.json`，100 Hz，直线容差 ±20°）：

| 候选 | 帧数 | 时长 | 路径速度 | 侧向范围 | 2s 内最大转向 | 最长直线窗口 | 不合格原因 |
|---|---:|---:|---:|---:|---:|---:|---|
| KIT/3/walk_6m_straight_line06 | 1491 | 14.90 s | 0.649 | 2.00 m | **86.4°** | 6.84 s | 含 180° 往返转身，非连续直线 |
| KIT/3/walk_6m_straight_line04 | 736 | 7.35 s | 0.803 | 2.01 m | 24.9° | 6.76 s | **时长 < 10 s** |
| KIT/314/walking_medium09 | 773 | 7.72 s | 0.343 | 0.09 m | 52.4° | 2.68 s | 时长不足 + 含停步 |
| KIT/348/turn_right03 | 618 | 6.17 s | 0.369 | 0.23 m | 0.0° | 1.69 s | 时长不足 + 转向 |
| KIT/424/walking_slow08 | 852 | 8.51 s | 0.234 | 0.15 m | 50.9° | 1.26 s | 时长不足 + 速度过低 |
| KIT/4/WalkInCounterClockwiseCircle04 | 840 | 8.39 s | 0.575 | 1.68 m | 154.4° | 1.11 s | 绕圈 |
| KIT/4/WalkInCounterClockwiseCircle08 | 854 | 8.53 s | 0.606 | 1.92 m | 141.2° | 0.87 s | 绕圈 + 时长不足 |

**没有任何一条满足"未拼接 + 连续 ≥10 s + 直线"。** 最长直线窗口 6.84 s。

### F.2.3 全数据集筛查（此前用有效 token 完成，记录在案）

`reports/E_audit_training.md` 记录：完整重定向数据集共 1190 条 MyoFullBody 动作，
按字节数（实测 ≈8095 B/帧，7 条本地片段交叉验证误差 <0.1%）换算：
- ≥10 s 的片段极少，且几乎都在 `Transitions_mocap`（跳跃/奔跑/蹲行）或唯一那条含 180° 转身的行走；
- **testing 组最长 8.53 s**，物理上不可能提供 10 s 直线参考。

### F.2.4 需要下载/安装才能补上的缺口

| 缺口 | 现状 | 需要什么 |
|---|---|---|
| Hugging Face 重定向缓存 | **401**（token 为 `hf_oauth_…` 类型，已过期） | `hf auth login` 重新授权；但即便恢复，数据集内也没有合格片段（见 F.2.3） |
| 原始 AMASS/KIT 动作 | 无 | AMASS 账号 + 许可（非商业研究） |
| SMPL-H 人体模型 | 无 | smpl.is.tue.mpg.de 注册后人工下载（**不可脚本化**） |
| GMR-Fit 依赖 | 全部缺失 | `torch`、`smplx`(amathislab fork)、`mink`、`general-motion-retargeting`(gmr_plus) |

依赖实测：`import general_motion_retargeting / mink / smplx / torch / joblib` → **全部 ModuleNotFoundError**。

## F.3 GMR-Fit 流程（已核对 CLI，未执行）

仓库内的单条运动重定向入口（只读核对，未修改）：

```bash
# 官方脚本：按 dataset group 批量；单条运动通过 AMASSDatasetConf([motion]) 驱动
python external/MuscleMimic/scripts/retarget_dataset.py \
  --model MyoFullBody \
  --dataset <GROUP_NAME> \
  --retargeting-method gmr \
  --gmr-src-human smplh --gmr-target-fps 30 --gmr-solver daqp --gmr-damping 0.5
```

`conf` 里与本次实验一致的 GMR 参数（与 checkpoint 训练时相同）：
`src_human=smplh, target_fps=30, solver=daqp, damping=0.5, offset_to_ground=false,
use_velocity_limit=false, use_fitted_shape=true, shape_fitting_iterations=500`。

**未执行原因**：缺原始 SMPL-H 动作 + 缺 SMPL-H 模型 + 缺 GMR 依赖 + token 过期（F.2.4）。
因此没有任何"新参考"产物，也没有可对比的重定向质量报告。

## F.4 指标核查（本次实际发现的错误，已修正并统一重算）

### F.4.1 旧报告的"平均航向误差约 86°"是错的

旧汇总表打印的是**逐回合 `max_heading_error_deg` 的均值**，却被写成"平均航向误差"。
根因是原指标用**每步位移方向**（100 Hz 下单步仅约 8 mm）计算航向：
- 单步位移被左右摆动主导，方向接近随机；
- 接近静止的帧方向无定义，会给出接近 180° 的伪值。

**修正**：改为 0.5 s 窗口的 PCA 主方向，并要求策略与参考的窗口速度都 > 0.2 m/s。
新版本对**所有组**统一重算（`scripts/recompute_metrics.py`，不重跑物理）。

### F.4.2 修正后的结果（两组同一版本）

| 组 | n | 平均航向误差 | 最大航向误差 | 净航向误差 | 策略速度 | 参考速度 | 进度比 | 末端总偏差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧拼接参考（官方 checkpoint） | 20 | **8.6°** | 33.1° | 6.6° | 0.758 | 0.883 | **0.823** | 1.772 m |
| 本次 3M 步策略 | 20 | 161.8° | 170.2° | 160.5° | 0.802 | 0.702 | −0.992 | 1.776 m |

**结论修正**：官方策略的航向**基本正确（8.6°）**，与"沿平行线前进"的描述**一致**，
与旧报告"平均 86°"**不一致**——旧数字是度量错误，不是策略问题。
3M 步策略的 162° 才是真的反向行走（向后倒）。

### F.4.3 侧向失败的真实构成（官方 checkpoint，20 种子，末端偏差分解）

| 分量 | 均值 | 范围 |
|---|---:|---|
| 沿参考方向的滞后（负 = 落后） | **−1.477 m** | −3.131 … −1.029 |
| 横向（cross-track） | **−0.949 m** | 绝对最大 2.069 |
| 末端总偏差 | 1.772 m | — |
| \|cross-track\| ≤ 0.35 m 的种子 | **0 / 20** | — |

即：**失败是"落后 + 横移"并存**，而横移单独就已经 0/20 不达标。
速度 0.758 m/s 高于 0.6 的阈值，所以不是"走太慢"单一原因；是跟踪 10 s 后累积的横向几何偏差。

配套图：`reports/figures/straight10s_path_and_height.png`。

## F.5 判定

| 项目 | 结果 |
|---|---|
| 新参考（未拼接、连续 ≥10 s 直线）能否构建 | **否**（数据门槛未通过） |
| 10 s 直线验收（新参考） | **未执行**（无合格参考） |
| 旧参考结果（对照） | 20/20 走满 10 s 不跌倒、**0/20** 通过直线判据 |
| 是否可用本次结果证明"拼接是唯一原因" | **不能**，且本实验未产生新参考，连"更换参考后是否改善"也未能检验 |
| 确认集 101–120 | **未使用**，保持预留 |

## F.6 下一项最小实验（按阻塞顺序）

1. **补数据（唯一阻塞项）**：
   - 注册 AMASS 与 SMPL-H，下载一段 ≥12 s 的**单一直线、近似恒速**行走
     （KIT 的 `walking_*` 或 CMU 直线走；先在原始数据上做同样的直线性筛查，见
     `scripts/motion_straightness.py` 口径：±20° 容差、速度 >0.2 m/s）；
   - `conda create -n gmr python=3.11` 后安装 `torch`、`smplx`(amathislab fork)、
     `mink`、`gmr_plus`，设置 `musclemimic-set-smpl-model-path` / `musclemimic-set-amass-path`；
   - 运行 F.3 的命令重定向为 MyoFullBody，再用 `scripts/motion_straightness.py` 复核
     **重定向后**是否仍 ≥10 s 直线，并对照速度是否接近旧参考 0.874 m/s。
2. **然后**用本次冻结的 `reports/freeze_straight_reference.json` 与 `scripts/acceptance_straight.sh`
   跑种子 0–19；只有在确认集（101–120）上一次性复现，才算通过。
3. 若新参考速度无法接近 0.874 m/s，必须同时报告速度差，不能把结果变化全部归因于"去掉拼接"。

在数据门槛通过之前**不启动训练**。
