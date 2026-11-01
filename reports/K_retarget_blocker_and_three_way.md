# K. `136_25` 重定向门槛、QC 工具链，与三个问题的分开回答

日期：2026-11-01 ｜ 环境：`.venv`（验收，jax CPU）· `.venv-gmr`（重定向，torch CPU）
新增：`scripts/retarget_one.py`、`scripts/qc_retargeted.py`、`scripts/three_way_report.py`

---

## K.0 结论摘要

| # | 事项 | 结果 |
|---|---|---|
| 1 | `SMPLH_NEUTRAL.pkl` 获取 | **未能获取**（许可受限，需你本人注册接受协议）。已定位到唯一运行时路径与全部替代来源 |
| 2 | `136_25` 走 GMR | **未执行**——卡在上述单一缺失文件；前置两关都已实测通过 |
| 3 | 重定向后 QC | **脚本已写完并在现有参考上验证通过**；对 `136_25` 尚未能运行（无产物） |
| 4 | 三个问题的分开回答 | **已交付**（用现有两条参考实测，dev 种子 0–19） |

**三个答案确实不同**，而且差异很大：

| 参考 | A 走满 10 s | B 跟随参考 | B′ 跟随（限直行窗） | C 通过冻结验收 |
|---|---:|---:|---:|---:|
| `straight_stitched_10s`（旧，拼接） | **19/20** | **0/20** | 0/20 | **0/20** |
| `walk_6m_straight_line06`（未拼接，但非 10 s 直行） | **20/20** | **4/20** | 5/20 | **3/20** |

> 换成**未拼接**参考后侧偏显著改善（cross-track 1.038 → 0.515 m，配对 19/20 变好），
> 但仍是 **3/20，属于"改善但未通过"**；而且这条参考本身只有 6.84 s 直行，
> **不能**用它宣称达成 10 s 直线基线。

---

## K.1 为什么 `SMPLH_NEUTRAL.pkl` 拿不到（可复核的排除过程）

### K.1.1 运行时到底要哪个文件

```
fit_gmr_motion()
 └─ load_smplh_file(..., smpl_model_path, ...)
     └─ GMR_SMPLH_Parser(model_path=smpl_model_path, gender="neutral", use_pca=False)
         └─ smplx.body_models.SMPLH(...)   # amathislab fork，默认 ext="pkl"
             └─ <SMPL_MODEL_PATH>/SMPLH_NEUTRAL.pkl
```

`SMPL_MODEL_PATH` 解析为 `~/.musclemimic/smpl`（`loco_mujoco/smpl/retargeting.py:get_smpl_model_path`）。
注意 amathislab fork 找的是 `<dir>/SMPLH_<GENDER>.pkl` **直接位于目录下**，
不是上游 smplx 的 `smplh/neutral/model.npz` 相对布局。

### K.1.2 已排除的获取途径

| 途径 | 结果 | 证据 |
|---|---|---|
| 本机任意位置已有模型 | 无 | `find /home/zrq -maxdepth 9` 匹配 `SMPLH_*.pkl`/`SMPLH_*.npz`/`MANO_*.pkl`/`neutral/model.npz`/`basicModel*.pkl` → 0 命中 |
| `~/Documents/muscle`、`muscle-rl`、`poselib`、`archive` | 无 | 只有 BVH / BIO `.npy` / replay buffer `.pkl` |
| gmr_plus 自带 `assets/`（1.2 GB） | 只有机器人 MJCF/URDF | 无 `body_models` |
| 上游 `install_smplh.sh` 的输入 | 缺 | 需要 `smplh/neutral/model.npz` + `mano_v1_2/models/MANO_{LEFT,RIGHT}.pkl` |
| MANO 官网 `mano.is.tue.mpg.de` | 可访问（HTTP 200），但**必须先登录并接受 MPG 许可** | 我不能代你接受许可 |
| HuggingFace 上的 `Tevior/smplh`、`RoninZYR/smplh` 等 | **不使用** | 均为非授权镜像（`gated=None`），下载即绕过 SMPL 许可。按合规要求不采用 |
| HF token | 有效（`krzz05`，2026-11-12 到期） | 但官方 `amathislab/musclemimic-retargeted` 只含 KIT（1080）+ Transitions（110），**无 CMU** |

### K.1.3 你要做的三步（约 5 分钟）

1. 登录 <https://mano.is.tue.mpg.de/download.php>，接受许可，下载：
   - **Extended SMPL+H model**（`smplh/neutral/model.npz`）
   - **Models & Code**（`mano_v1_2/models/MANO_LEFT.pkl`、`MANO_RIGHT.pkl`）
2. 放到下面这个布局（目录名必须一致，`generate_smplh_model.py` 是硬编码的）：

```
~/.musclemimic/smpl/
├── smplh/neutral/model.npz
└── mano_v1_2/models/MANO_LEFT.pkl
                    MANO_RIGHT.pkl
```

3. 合成运行时文件：

```bash
cd external/MuscleMimic && bash loco_mujoco/smpl/install_smplh.sh
# 产物：~/.musclemimic/smpl/SMPLH_NEUTRAL.pkl
```

（该脚本会临时建 conda 环境装 `chumpy` 并打 numpy 兼容补丁；若不想用 conda，
等价逻辑在 `loco_mujoco/smpl/generate_smplh_model.py`，就是把 SMPL-H 的 npz
加上 MANO 双手的 `hands_meanl/r`、`hands_componentsl/r`、`hands_coeffsl/r` 合并后 pickle 出来。）

**SMPL-H ≠ SMPL**：需要带手的 SMPL-H，不能拿 SMPL 顶替（`poses` 是 156 维 = 3+63+45+45）。

---

## K.2 `136_25` 的前置关卡（已实测，只有第三关没过）

`scripts/retarget_one.py` 设了三道门，任何一道不过就拒绝重定向——因为重定向后的参考
只有在"模型变体一致 + 原始数据本身就够长够直"的前提下才可与冻结基线比较。

```bash
.venv-gmr/bin/python scripts/retarget_one.py --clip CMU/136/136_25_poses --dry-run
```

实跑输出：

```
[gate 1] environment model version
  model: musclemimic_models 1.0.5, nq/nv/nu=(89, 88, 354), knee 1.0.5 layout
[gate 2] raw clip is already a >=10 s straight walk
  raw: 14.90 s @ 120 Hz, straight window 1.85-13.49 s (11.64 s, 0.507 m/s)
[gate 3] SMPL-H body model
  BLOCKED: SMPLH_NEUTRAL.pkl missing
```

- **Gate 1 通过**：`musclemimic_models==1.0.5`，`nq/nv/nu=(89,88,354)`，
  膝约束符号模式为"1 正 1 负"（1.0.5 布局）。这是你要求的"确认模型仍为 1.0.5"。
- **Gate 2 通过**：`136_25` 原始 14.90 s、120 Hz，最长连续直线窗 **1.85–13.49 s = 11.64 s**，
  0.507 m/s。原始数据本身已经是 ≥10 s 直行，重定向只是换骨架、不该改变这件事。
- **Gate 3 未通过**：见 K.1。

AMASS 侧已就绪：`~/.musclemimic/AMASS` 已放好**仅这一条**片段（没有解压 2082 条）：

```
~/.musclemimic/AMASS/CMU/136/136_25_poses.npz   2.3 MB
```

`retarget_dataset.py` 不能用：它只接受预定义 dataset group 名（`AMASS_LOCOMOTION_DATASETS`
等），传单条路径会报 `Dataset 'CMU/136/136_25_poses' not found`。所以单条走
`retarget_one.py`，内部调用的是同一个 `ImitationFactory.make` + `AMASSDatasetConf`，
**没有改上游任何代码**。

冻结的 GMR 参数（写死在脚本里，不是命令行开关）：

```python
{"src_human": "smplh", "target_fps": 30, "solver": "daqp", "damping": 0.5,
 "offset_to_ground": False, "use_velocity_limit": False, "verbose": False}
```

> 与 `reports/I_gmr_toolchain.md` 的 I.7 相比有一处更正：`--gmr-offset-to-ground`
> 的 argparse 默认值是 **False**（`store_true` + `default=False`），所以实际默认不做落地下移。

---

## K.3 重定向后 QC 脚本（已写完，已在现有参考上验证）

`scripts/qc_retargeted.py` 逐项检查重定向产物，覆盖你点名的四项（骨盆速度、足部高度、
足滑、关节突变）加数值检查：

| 检查 | 判据 | 目的 |
|---|---|---|
| `finite` | `qpos/qvel/site_xpos` 无 NaN/Inf | 数值异常 |
| `shape` | `nq=89 nv=88 nsite=17` | 模型变体一致 |
| `sites` | 4 个足部 site + `pelvis_mimic` 存在 | 评价 site 齐全 |
| `root` | 每帧位移 ≤ max(3×中位数, 5 cm)；四元数模长 ≈1 | 根节点瞬移 |
| `joint_jumps` | 铰链每帧转角 ≤ 20° | 关节角突变 |
| `foot_height` | 足部 site z 抬升 > 5 cm | 有摆动相、无穿地 |
| `split_points` | 段数、是否覆盖全长 | 是否被拼接 |
| `foot_slip` | 见下 | 足滑 |

### K.3.1 在旧拼接参考上的实测（这一步证明脚本可用）

```bash
.venv/bin/python scripts/qc_retargeted.py --clip KIT/3/straight_stitched_10s
```

```
frames    : 1340 @ 100 Hz = 13.39 s
segments  : 1 [1340] (concatenated=False)
[window] straight 0.07-13.34 s = 13.27 s, net 11.61 m, speed 0.875 m/s, straightness 0.995
[checks] finite PASS · shape PASS · sites PASS · root PASS · joint_jumps PASS · foot_height PASS
  root: max_step 12.3 mm (median 9.0 mm), quat_norm 0.99999996..1.00000004
  joint_jumps: max hinge delta 10.47 deg/frame (threshold 20)
  foot_height: ankle lift 0.140/0.149 m, toes lift 0.089/0.143 m, 0 frames below floor
```

### K.3.2 足滑指标必须带敏感性表（我改了做法）

一开始我用"z < z_min + 20 mm 即视为着地"算滑步，得到 0.45 m/s——**这是错的**。
逐帧检查一个步态周期发现：脚在摆动末期会**贴着地面高度水平掠过**（z≈0.066 m，横向 67 mm/20 ms ≈ 3.3 m/s），
高度阈值越宽，混进来的摆动帧越多。所以改为报告多个阈值的敏感性：

```
[foot slip while planted]  (headline band 5 mm)
  site                                2mm               5mm              10mm              20mm
  left_ankle_mimic            0.055@ 8.5%       0.086@34.5%       0.191@45.4%       0.446@62.7%
  left_toes_mimic             0.048@ 1.0%       0.059@ 4.2%       0.062@21.9%       0.200@49.2%
  right_ankle_mimic           0.065@ 3.6%       0.111@36.1%       0.337@56.0%       0.532@68.1%
  right_toes_mimic            0.058@ 1.0%       0.073@ 3.4%       0.089@ 8.4%       0.165@36.7%
```

严格 5 mm 带下参考自身滑步 **0.06–0.11 m/s**，而身体前进 0.875 m/s，即脚基本是钉住的（≤13%）。
绝对数值次于"每个参考都用同一套算法量"这一点——那才是让参考之间可比的东西。

策略的滑步另有算法：`eval_walk.py` 用**接触法向力 > 20 N** 判定着地，那是有物理依据的版本。

### K.3.3 `136_25` 的 QC 尚未运行

没有重定向产物，所以无对象可查。产物出现后一条命令即可：

```bash
.venv/bin/python scripts/qc_retargeted.py --clip CMU/136/136_25_poses
```

判据门（脚本 `verdict` 段）：`duration≥10 s`、`straight_window≥10 s`、
`window_speed`、`straightness≥0.99`、`no_numeric_anomaly`、`single_segment`。

---

## K.4 三个问题的分开回答（dev 种子 0–19，官方 `checkpoint_12500`）

### K.4.1 定义（关键：三者的门槛来源不同）

| 标签 | 含义 | 阈值来源 |
|---|---|---|
| **A `survived_10s`** | 跑满 10 s 且未跌倒 | 冻结（duration、`physical_fall`、数值异常、`reset_count==1`） |
| **B `follows_ref`** | 真的跟住了参考轨迹 | `max_cross_track ≤ 0.35 m`（**冻结数字**，同侧偏阈值）+ `progress_ratio ∈ [0.75,1.25]`（**本次新增**）+ `mean_heading_error ≤ 20°`（**本次新增**） |
| **C `passes_gate`** | 通过冻结的直线验收 | `configs/eval_walk.json` 的 blocking criteria（10 s、0.6 m/s、0.35 m、45°） |

B 的两个新阈值是**本报告引入的描述性判据**，不是冻结协议的一部分，改动它们不需要重跑 A/C。
`three_way.json` 的 `definitions` 段把这件事写死了。

### K.4.2 复现命令

```bash
export PYTHONPATH="external/MuscleMimic:scripts"
CKPT=artifacts/checkpoints/checkpoint_12500
for spec in "stitched:KIT/3/straight_stitched_10s" "unstitched6m:KIT/3/walk_6m_straight_line06_poses"; do
  name="${spec%%:*}"; motion="${spec#*:}"
  .venv/bin/python scripts/eval_walk.py --checkpoint "$CKPT" --motion_path "$motion" \
     --mode stochastic --seeds $(seq 0 19) --seconds 10 --out_dir "runs/TW_${name}"
done
.venv/bin/python scripts/three_way_report.py \
  --group stitched=runs/TW_stitched/stochastic \
  --group unstitched6m=runs/TW_unstitched6m/stochastic
```

两组同一天、同一份代码、同一 checkpoint、同一指标版本，**没有只修一组**。
种子 101–120 仍然完全未动。

### K.4.3 汇总

| 参考 | A 走满 10 s | B 跟随参考 | B′ 跟随（限参考直行窗） | C 通过冻结验收 |
|---|---:|---:|---:|---:|
| `straight_stitched_10s` | 19/20 | 0/20 | 0/20 | 0/20 |
| `walk_6m_straight_line06` | 20/20 | 4/20 | 5/20 | 3/20 |

| 量（10 s 窗口均值） | 拼接参考 | 未拼接 6m | 配对差 | 变好的种子 |
|---|---:|---:|---:|---:|
| 参考速度 (m/s) | 0.883 | 0.834 | — | — |
| 策略速度 (m/s) | 0.758 | 0.766 | — | — |
| `progress_ratio` | 0.823 | 0.887 | −0.064 | 19/20 |
| `max_cross_track` (m) | 1.038 | 0.515 | **+0.523** | **19/20** |
| 净侧向漂移 (m) | −0.788 | −0.375 | −0.413 | 17/20 |
| 平均航向误差 (°) | 8.62 | 8.45 | +0.17 | 11/20 |
| 最大侧偏 (m) | 1.056 | 0.582 | — | — |

逐种子表、俯视图：`reports/three_way.md`、`reports/figures/threeway_*_overlay.png`、
`reports/figures/threeway_*_grid.png`。

### K.4.4 最重要的物理发现：失效不是航向问题，是**持续性单侧侧移**

把跟踪误差分解成沿路径与横向两个分量后：

| 参考 | 沿路径滞后 | 侧向漂移 | 等效"横向走"角度 | 20/20 同号？ |
|---|---:|---:|---:|---|
| 拼接 | −1.43 m | −0.79 m | **−6.5° ± 4.3°** | **是（全部为负）** |
| 未拼接 6m | −1.13 m | −0.38 m | −3.2° ± 2.8° | 否 |

- 平均航向误差只有 **8.6°**——策略确实是"朝前走"的，之前的 86° 是旧指标的假象（已修）。
- 但 `corr(侧向漂移, max_cross_track) = −0.975`：**侧偏几乎完全等于侧向漂移的累积**。
- 20 个种子漂移方向**全部相同**，说明这是**策略的系统性偏向**，不是种子噪声。
- 换算成角度：边走边横向移动 ≈ **6.5° 的"蟹行"分量**。10 s 内累积到 ~0.8 m。

这解释了为什么"提高速度"或"换参考"都不能解决 0.35 m：**只要存在 1°/s 量级的定向侧移，10 s 一定超限**。
对后续"患侧减力 + 拐杖辅助"而言这条尤其重要——它给出了一个**健康基线本身就带单侧偏置**的先验，
做左右对比时必须先把它扣除或作为协变量，否则会把模型固有偏置误读成患侧效应。

### K.4.5 未拼接参考的改善**不能**归因于"去掉拼接"

必须写清楚的三条混淆：

1. **速度不同**：窗口内参考 0.883 → 0.834 m/s，`progress_ratio` 变好有一部分只是"追得上了"。
2. **几何不同**：`walk_6m_straight_line06` 在 10 s 窗口内的最长直行子窗只有 **3.66–10.00 s（6.34 s）**，
   前 3.66 s 参考在转弯。所以这不是"拼接直行 vs 未拼接直行"，而是"拼接直行 vs 未拼接但含转弯"。
3. **这条参考本身不合格**：整条 14.90 s 的最长直行窗 6.84 s（`motion_straightness.py` 复核），
   **够不到 10 s**。它能验证"换参考是否改善"，但**不能**用来交付 10 s 直线基线。

**措辞**：换成未拼接参考后侧偏减半、进度跟踪改善，但仍是 3/20，
属于 **"改善但未通过"**；单凭这一次更换，不能严格证明拼接接缝是唯一原因。

---

## K.5 判定

### 对"`136_25` 单因子实验"

> **因数据不足无法判定（确切说是：因缺少许可受限的 SMPL-H 人体模型，无法执行重定向）。**
> 前置关卡（模型版本 1.0.5、原始片段 11.64 s 直行）已通过；
> 唯一的阻塞是 `SMPLH_NEUTRAL.pkl`，它在你的账号接受 MANO 许可后即可解决。

### 对"是否达到 10 s 直线稳定行走"

> **未通过。** 最好的一组（未拼接 6m 参考）为 A 20/20、C 3/20；
> 而该参考本身只有 6.84 s 直行，不满足 10 s 数据前提。

### 对"官方 checkpoint 能否作为正常人肌肉驱动的 10 s 直线行走基线"

> **作为"能站住 10 s"的基线可以（19–20/20）；作为"直线"基线不可以（0–3/20）。**
> 失效机理已定位到持续性单侧侧移（≈6.5° 蟹行 / 10 s 累积 0.8 m），
> 不是跌倒、不是航向错误、不是速度不足、也不是数值问题。

---

## K.6 下一项最小可检验实验（按顺序，一次一个因子）

1. **补齐 `SMPLH_NEUTRAL.pkl`，只重定向 `136_25`**，跑 `qc_retargeted.py`。
   预期判据：直行窗 ≥10 s、`straightness ≥ 0.99`、`no_numeric_anomaly=True`、
   窗口速度 ≈0.507 m/s。**若直行窗掉到 10 s 以下，就地停止并报告**（说明 IK 破坏了直行性）。
2. **若 K.6.1 通过**：官方 checkpoint × `136_25` × dev 种子 0–19，用 K.4 的同一套三问报告。
   预期：A 高（能站住）、B 取决于能否接受 0.507 m/s 的慢速参考、C **必然为 0**（0.507 < 0.6 m/s 是代数事实，
   不是策略缺陷）。这一步只回答"侧偏是否改善"，**不宣称达成 10 s 直线验收**。
3. **侧向偏置的独立验证**（与参考无关，最便宜）：把策略放到一条**完全笔直、速度 0.76 m/s**的
   人工参考上（用 `prepare_straight_walk.py` 生成恒定速度直线），看 20 个种子的侧向漂移
   是否仍然全部同号。**判据**：若仍 20/20 同号且 ≈ −0.08 m/s，则侧偏是策略/模型固有偏置，
   与参考数据的拼接/速度都无关；此时下一步应查动作对称性（左右肌肉执行器增益、
   观测里左右字段是否等价偏置），而不是继续换参考。
4. 只有在 1–3 都完成后，才讨论"改奖励/改阈值/再训练"。

---

## K.7 变更清单

| 文件 | 说明 |
|---|---|
| `scripts/retarget_one.py` | 新增：单条片段重定向驱动（三道前置关卡 + 冻结 GMR 参数） |
| `scripts/qc_retargeted.py` | 新增：重定向产物 QC（数值/根节点/关节突变/足高/足滑/拼接段） |
| `scripts/three_way_report.py` | 新增：A 走满 / B 跟随参考 / C 通过验收 三问分开报告 + 俯视图 |
| `reports/three_way.json`、`reports/three_way.md` | 新增：逐种子结果与定义 |
| `reports/figures/threeway_*.png` | 新增：两组各 1 张 20 种子叠加图 + 1 张 4×5 逐回合图 |
| `reports/qc_KIT_3_straight_stitched_10s.json` | 新增：QC 在旧参考上的验证结果 |
| `runs/TW_stitched/`、`runs/TW_unstitched6m/` | 新增：两组 × 20 种子原始 npz/json |
| `~/.musclemimic/AMASS/CMU/136/136_25_poses.npz` | 新增：只解压了这一条片段 |
