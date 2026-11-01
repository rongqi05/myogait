# M. 取得官方 SMPL-H、完成 `136_25` 重定向——但该片段是倒退行走

日期：2026-11-01 ｜ `.venv-gmr`（重定向）· `.venv`（验收）
新增：`scripts/build_smplh_neutral.py`、`scripts/amass_facing.py`、`scripts/slice_amass_clip.py`
修改：`scripts/retarget_one.py`（新增第 3 道关卡：朝向）、`scripts/qc_retargeted.py`（新增起点/根朝向/接触检查 + 原始对照）

---

## M.0 结论摘要

| # | 事项 | 结果 |
|---|---|---|
| 1 | 取得官方授权 SMPL-H | **完成**。用你下载的两个官方包，**不依赖 conda/chumpy** 合成了 `SMPLH_NEUTRAL.pkl` |
| 2 | `136_25` 单条 GMR | **重定向成功**（447 帧 @30 fps，17.6 s，150.7 FPS），产物数值全部合格 |
| 3 | 重定向后直线/速度/根朝向/足部/接触 | **全部通过**：直线窗 11.6417 → **11.6500 s**，速度 0.5067 → 0.5056 m/s |
| 4 | **但该片段不可用** | **`136_25` 是倒退行走**：脚尖朝向与前进方向点积 **−0.941/−0.920**（KIT 参考为 +0.97） |
| 5 | 上游审计漏洞 | 我的 `amass_kit_audit.py` 只读 `trans`，**无法判断人物朝向**；已补 `scripts/amass_facing.py` 并接入第 3 道关卡 |
| 6 | 另一个候选 | `120_19` 是**正向**行走（+0.904/+0.815）。已切出 10.41 s 连续窗口、重定向、QC 全过 |
| 7 | 合格参考的回放 | `cmu120_19_straight_h000`：**11/20 走满、9 次跌倒**；对照 `kit_stitched_slerp` 20/20、0 摔倒 |
| 8 | ~~新发现的真瓶颈~~ | ~~参考步速~~ → **已收回**，见 `reports/N_speed_vs_dataset_test.md`：真正约束是侧偏累积 |

---

## M.1 授权模型文件：从官方包到 `SMPLH_NEUTRAL.pkl`

你放在 `~/Downloads` 的两个官方包：

```
mano_v1_2.zip    175 MB   2026-11-01 18:07   # MANO "Models & Code"
smplh.tar.xz     113 MB   2026-11-01 18:06   # "Extended SMPL+H model"
```

内容已核对：

```
smplh.tar.xz   ->  neutral/model.npz, male/, female/, info.txt, LICENSE.txt
mano_v1_2.zip  ->  mano_v1_2/models/MANO_LEFT.pkl, MANO_RIGHT.pkl, LICENSE.txt
```

`smplh/info.txt` 明确写着 *"extended shape parameters (16 instead of original 10)"*，
与 AMASS 的 `betas (16,)` 一致——这正是 AMASS / GMR 需要的变体。

### M.1.1 为什么不照上游的 conda + chumpy 流程

上游 `loco_mujoco/smpl/install_smplh.sh` 会临时建一个 `python=3.10` 的 conda 环境，
装 `numpy<1.23` 和 `chumpy`，再用 `sed` 打补丁。原因是 **MANO 的 `.pkl` 是 Python 2 的
pickle，里面含 `chumpy.ch.Ch` 对象**，而 chumpy 无法在 numpy 2.x 下工作。

我们只需要 MANO 里的三个键。实测这三个键在发布文件里就是普通 `numpy.ndarray`：

```
hands_components  ndarray (45, 45)      hands_coeffs  ndarray (1554, 45)      hands_mean  ndarray (45,)
```

所以完全不需要 chumpy：只要能让 pickle **反序列化**通过即可。`scripts/build_smplh_neutral.py`
用一个最小的 `chumpy` 替身模块 + 一个把 `__builtin__` 映射到 `builtins` 的 `Unpickler` 子类完成这件事。

### M.1.2 产物与校验

```bash
# 布局（generate_smplh_model.py 的路径是硬编码的）
~/.musclemimic/smpl/smplh/neutral/model.npz
~/.musclemimic/smpl/mano_v1_2/models/MANO_{LEFT,RIGHT}.pkl

.venv-gmr/bin/python scripts/build_smplh_neutral.py
```

```
inputs:
  smplh/neutral/model.npz              sha256 a60a7e29d33f09ef   87259106 B
  mano_v1_2/models/MANO_LEFT.pkl       sha256 c4022f7083f2ca7c    3821391 B
  mano_v1_2/models/MANO_RIGHT.pkl      sha256 45d60aa3b27ef910    3821356 B
output:
  SMPLH_NEUTRAL.pkl                    sha256 f3fc0d3746e3f332   88409602 B

smplx.SMPLH load OK: joints (1, 73, 3), verts (1, 6890, 3)
```

校验不是"能反序列化"就算——脚本用**重定向器自己的调用方式**（`smplx.SMPLH(model_path=..., gender="neutral", use_pca=False)`）
做了一次前向，并要求 6890 顶点、73 关节（52 个 SMPL-H 关节 + smplx 追加的 21 个指尖/脚跟关节）。

`scripts/check_gmr_toolchain.py` 现在全绿：

```
[assets]
  PASS  SMPL_MODEL_PATH        /home/zrq/.musclemimic/smpl
  PASS  SMPLH_NEUTRAL.pkl      (88.4 MB)
  PASS  smplh/neutral/model.npz
  PASS  MANO_LEFT.pkl / MANO_RIGHT.pkl
  PASS  AMASS dataset root
PASS  left-knee eq sign       1 positive / 1 negative (1.0.5 layout)
```

---

## M.2 `136_25` 重定向：技术上是成功的

```bash
PYTHONPATH=external/MuscleMimic:scripts .venv-gmr/bin/python \
  scripts/retarget_one.py --clip CMU/136/136_25_poses --json-out reports/retarget_136_25.json
```

```
[gate 1] model: musclemimic_models 1.0.5, nq/nv/nu=(89,88,354), knee 1.0.5 layout
[gate 2] raw: 14.90 s @ 120 Hz, straight window 1.85-13.49 s (11.64 s, 0.507 m/s)
GMR:  447 frames, 150.71 FPS, 6.64 ms/frame
30 Hz -> 100 Hz:  445 -> 1484 samples
retargeted in 17.6 s
```

shape fitting 与 IK 质量（`136_25_poses_analysis.npz`）：

```
shape fitting: initial loss 0.004604 -> final 0.003341 (converged)
               height_scale 0.9980, offset_z -0.0171 m
IK 位置误差（17 个 site）：
  overall  mean 7.8 cm   p95 46.9 cm   max 49.0 cm
  腿部链   mean 1.08 cm  max 5.62 cm        <- 决定步态的部分很准
  ankle/toes  mean 0.3-0.4 cm  max <1.0 cm
  left_hand  47.70 cm  (std 0.63 cm)
  left_elbow 33.78 cm  (std 0.48 cm)
  left_shoulder 24.92 cm (std 0.19 cm)
  right_shoulder 9.34 cm
```

左臂误差高达 48 cm 但 **std 只有 0.6 cm**，说明它不是跟踪失败，而是一个**恒定偏移**：
源片段的左臂姿态在 MyoFullBody 上够不到，求解器稳定地停在极限位置。
腿链误差 1 cm 量级，说明**决定行走的部分拟合得很好**。

---

## M.3 重定向后的质量检查（你点名的五项）

`.venv/bin/python scripts/qc_retargeted.py --clip CMU/136/136_25_poses --compare-raw`

### M.3.1 直线窗口与速度：原始 11.64 s **完整保留**

| 量 | 原始（SMPL-H 根节点） | 重定向（pelvis site） | 差 |
|---|---:|---:|---:|
| 时长 | 14.9000 s | 14.8300 s | −0.0700 |
| **最长直线窗** | **11.6417 s** | **11.6500 s** | **+0.0083** |
| 窗口净位移 | 5.8992 m | 5.8902 m | −0.0090 |
| **窗口速度** | **0.5067 m/s** | **0.5056 m/s** | **−0.0011** |
| 窗口直线度 | 0.9872 | 0.9872 | 0.0000 |
| 片段平均速度 | 0.4286 m/s | 0.4276 m/s | −0.0010 |

窗口起止：原始 1.85–13.49 s，重定向 1.80–13.45 s。**11.64 s 的直行窗口没有被 IK 破坏。**

### M.3.2 根朝向：干净

```
root_rotation: max 0.51 deg/frame (median 0.158, p99 0.357)   PASS
```
对比：上一轮拼接参考的 bug 是 **165.77 deg/frame**。这个量级说明 GMR 输出的姿态是连续的。

### M.3.3 根节点平移与数值

```
root:  max_step 7.6 mm (median 4.6 mm), max_step_at 6.32 s
       quat_norm 0.99999997 .. 1.00000003            PASS
finite: 无 NaN/Inf                                     PASS
shape: nq=89 nv=88 nsite=17（与冻结基线一致）           PASS
joint_jumps: 铰链最大 3.03 deg/frame（阈值 20）         PASS
```

### M.3.4 足部轨迹与接触

```
foot_height: 左踝 0.0655..0.2183 m（抬升 0.153）
             左趾 0.0310..0.1066 m（抬升 0.076）
             右踝 0.0674..0.2593 m（抬升 0.192）
             右趾 0.0421..0.1455 m（抬升 0.103）
             穿地帧（低于自身最低点 5 mm）: 0            PASS

contact:     触地次数 左 7 / 右 8；支撑相占比 1.000
             双支撑 0.2918；步周期 1.98 s              PASS
```

足滑（严格 5 mm 带）：**0.028–0.068 m/s**，而身体前进 0.506 m/s，即 6–13%。
对比旧 KIT 参考是 0.06–0.11 m/s @ 0.875 m/s（7–13%），比例相当。

### M.3.5 唯一不满足的是我自己设的便利阈值

```
straightness_ge_0p99 = False   (实测 0.9872)
```
这是我在 `qc_retargeted.py` 里随手设的诊断阈值，不是验收判据。
**原始片段本身也是 0.9872**，即 1.3% 的路径多余量来自源数据，不是重定向引入的。
如实记录，不改阈值。

---

## M.4 但它不可用：`136_25` 是倒退行走

### M.4.1 发现过程

把参考转到 0° 朝向、起点归零后跑官方 checkpoint，结果异常：

| 参考 | A 走满 10 s | 跌倒 | 平均时长 | progress ratio | 航向误差 |
|---|---:|---:|---:|---:|---:|
| `cmu136_25_head000` | **2/20** | **18** | 3.75 s | **−2.13** | **93.6°** |
| `cmu136_25_head270` | 0/20 | 20 | 3.33 s | 0.24 | 64.0° |
| `kit_stitched_slerp`（对照） | 20/20 | 0 | 10.00 s | 0.885 | 6.4° |

`progress_ratio = −2.13` 意味着策略相对参考方向**后退**了 2 倍参考距离，而航向误差 93.6°
意味着它几乎走在垂直方向。18/20 在 1.5–7.6 s 跌倒。

### M.4.2 与坐标约定无关的判据

用**从踝到趾**的连线方向（这个人身上的一根真实骨骼）与根节点实际前进方向做点积：

| 参考 | heading | toes·travel（L / R） | 判定 |
|---|---:|---|---|
| `KIT/3/straight_stitched_10s_slerp` | 0.00° | **+0.975 / +0.946** | 正向 |
| `KIT/3/walk_6m_straight_line06` | 170.67° | **+0.992 / +0.952** | 正向 |
| `CMU/136/cmu136_25_head000` | 0.00° | **−0.978 / −0.965** | **倒退** |

### M.4.3 源数据本身就是倒退的（不是重定向造成的）

用 SMPL-H 模型对**原始 AMASS 文件**做前向运动学，取同样的踝→趾方向：

```
CMU/136/136_25    frames=1789  net=6.23 m   toes·travel L=-0.941 R=-0.920  -> BACKWARD
CMU/120/120_19    frames=12792 net=4.89 m   toes·travel L=+0.904 R=+0.815  -> FORWARD
```

`136_25` 在原始 SMPL-H 空间里同样是倒退的（−0.94），所以**问题在源数据，不在 GMR**。
这也解释了为何它的步态"看起来正常"：倒退行走同样笔直、匀速、足部轨迹干净。
0.506 m/s 的速度对倒退行走也非常合理。

### M.4.4 我的审计为什么漏了它

`amass_kit_audit.py` 只读 `trans`（根节点轨迹）与 `mocap_framerate`。它测时长、直线性、
速度、航向变化——倒退行走**全部满足**。它从不读 `poses`，因此**无法判断人物朝向**。

这是一处真实的方法学漏洞，已修复。

---

## M.5 补上朝向关卡

### M.5.1 `scripts/amass_facing.py`

对给定片段用 SMPL-H 做前向运动学，比较踝→脚关节的水平方向与根节点实际前进方向：

- 判据与坐标约定无关（两个量都出自同一个 body model）；
- 双侧都要 > +0.5 才判 `FORWARD`，都 < −0.5 判 `BACKWARD`，否则 `AMBIGUOUS`；
- 位移 < 0.3 m 判 `STATIONARY`。

### M.5.2 接入 `retarget_one.py` 的第 3 道关卡

现在四道关卡，任一不过就拒绝重定向：

```
[gate 1] environment model version        musclemimic_models 1.0.5 + 膝约束符号
[gate 2] raw clip is already a >=10 s straight walk
[gate 3] clip walks forward               <- 新增
[gate 4] SMPL-H body model
```

对 `136_25` 的实际输出：

```
[gate 3] clip walks forward
  toes vs travel: left=-0.941 right=-0.920 -> BACKWARD
BLOCKED at stage 'facing':
  clip is BACKWARD (...); a backward-walking clip cannot serve as a forward-walking reference
```

---

## M.6 顺带修掉的一个环境陷阱：参考必须从原点起步

重定向时发现 `136_25` 的 pelvis 起点在 `(−2.787, −0.278)`，而策略每次都在 `(0, 0)` 起步。
直接探测：

```
cache qpos[0][:3]       : [-2.7865 -0.2782  0.9558]
after reset data.qpos[:3]: [ 0.       0.       0.9558]
```

**`env.reset()` 把自由关节的 XY 归零，只保留 z。** 所以策略永远在世界原点起步，
而参考保留自己的绝对位置。之前所有 KIT 参考恰好都从原点附近开始
（`walk_6m_straight_line06` 偏移 0.000 m，`straight_stitched_10s` 偏移 0.028 m），
这个假设一直没暴露；CMU 这条偏移 2.8 m，直接让策略与自己的目标错开 2.8 m。

`qc_retargeted.py` 现在会检查并大声警告：

```
PASS  root: {"start_xy_offset_from_origin_m": 2.6944, "start_xy_ok": false, ...}
WARNING: reference starts 2.694 m from the origin. env.reset() zeroes the policy root XY,
         so every rollout would begin with that constant offset against the reference targets.
```

`scripts/rotate_reference.py` 默认把参考平移到原点（`--no-zero-start-xy` 可关闭）。
纯 XY 平移对重力/地面/模型是严格对称操作，不改变步态。

> 注意：这个修正把 `136_25` 的侧偏从 2.92 m 降到 1.26 m，但**跌倒仍是 18/20**
> ——所以起点偏移不是跌倒主因，倒退行走才是。

---

## M.7 替代参考：`120_19`（正向，已重定向并 QC）

另一个合格候选是 `CMU/120/120_19_poses`：106.59 s 长，其中 48.83–59.27 s 是直线，
**正向行走**。按帧切出 48.30–59.80 s（11.5 s，无插值、无混合）后重定向：

```bash
.venv/bin/python scripts/slice_amass_clip.py --clip CMU/120/120_19_poses \
  --start 48.30 --end 59.80 --out_clip CMU/120/120_19_straight
.venv-gmr/bin/python scripts/retarget_one.py --clip CMU/120/120_19_straight
.venv-gmr/bin/python scripts/rotate_reference.py --clip CMU/120/120_19_straight --angles -179.09
```

QC 结果：

| 量 | 原始 | 重定向 | 差 |
|---|---:|---:|---:|
| 时长 | 11.5000 s | 11.4300 s | −0.0700 |
| **最长直线窗** | **10.4417 s** | **10.4100 s** | **−0.0317** |
| 窗口净位移 | 4.7772 m | 4.7617 m | −0.0155 |
| **窗口速度** | **0.4575 m/s** | **0.4574 m/s** | **−0.0001** |
| 直线度 | 0.9876 | 0.9869 | −0.0007 |

```
segments: 1（未拼接）
root_rotation: max 0.36 deg/frame        PASS
joint_jumps:  max 2.10 deg/frame         PASS
foot_height:  踝抬升 0.161/0.164 m，0 帧穿地   PASS
contact:      触地 5/6 次，双支撑 0.2247，步周期 2.08 s   PASS
足滑（5 mm 带）：0.016–0.027 m/s（身体 0.457 m/s，即 4–6%）
朝向：heading −0.00°，起点 (0,0)，toes·travel +0.966/+0.954 -> FORWARD
```

**这是一条真正可用的、未拼接的、正向的、≥10 s 连续直线行走参考。**
唯一硬约束：速度 0.457 m/s 低于冻结的 0.6 m/s 门槛，所以 C（原定直线验收）**必然为 0**，
这是代数事实而非策略缺陷。

---

## M.8 判定

### 对"取得官方授权 SMPL-H 并完成 `136_25` 单条 GMR"

> **技术上完成**：模型从官方包合成、双向前向校验通过；`136_25` 重定向成功，
> 直线窗 11.6417 → 11.6500 s、速度 0.5067 → 0.5056 m/s、根朝向最大 0.51°/帧、
> 0 帧穿地、触地 7/8 次。

> **但该片段作为基线参考不合格**：它是**倒退行走**（原始 SMPL-H 空间 toes·travel = −0.94）。
> 用它评估得到 2/20 走满、18/20 跌倒、progress ratio −2.13、航向误差 93.6°。
> **结论：拒绝 `136_25`，不是策略的问题。**

### 对"原始 11.64 s 直线窗在新模型中是否仍合格"

> **几何上合格**（11.65 s，速度与直线度逐位保持）。
> 但"合格"必须包含**朝向**这一维，而它在朝向这一维上不合格。
> 这正说明只看 `trans` 的审计不够。

### 10 s 直线基线

> **仍未通过**（历史最好 3/20，见 `reports/L_*.md`）。
> 本轮提供了第一条真正合格的候选参考 `cmu120_19_straight_h000`，
> 其回放结果见 M.9。

---

## M.8bis 合格参考的回放：11/20，9 次跌倒——~~瓶颈是参考速度~~

> **【本节结论已收回】** 见 [`reports/N_speed_vs_dataset_test.md`](./N_speed_vs_dataset_test.md) N.0。
> 这里的归因建立在**一次 n=1 的对照**上。随后的受控实验推翻它：
> 策略在 0.457 m/s 参考上走出 **0.460 m/s（误差 0.7%）**，6 s 内侧偏仅 **0.222 m**，
> 优于 KIT 参考（0.685 m）。真正的约束是**侧偏随时间累积**，
> 与参考步速、数据集都不构成因果关系。本节保留作审计留痕。

`cmu120_19_straight_h000` 在几何与朝向上都合格，但官方 checkpoint 仍然表现不好：

| 参考 | A 走满 10 s | B 跟随参考 | C 通过验收 | 跌倒 | 平均时长 | 参考速度 | 策略速度 | progress | 航向误差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmu120_19_straight_h000`（0.457 m/s） | **11/20** | 3/20 | 0/20 | **9** | 8.01 s | 0.464 | 0.556 | 0.576 | **38.3°** |
| `straight_stitched_10s_slerp`（0.884 m/s） | **20/20** | 1/20 | 0/20 | **0** | 10.00 s | 0.884 | 0.792 | 0.885 | 6.4° |

两个参考的差别只剩下**速度**（0.464 vs 0.884 m/s，即 52%）。而：

- 几何：两者都 ≥10 s 连续直线、未拼接、0 帧穿地、根朝向连续；
- 朝向：都是 0°、都从原点起步、都是正向行走；
- checkpoint、模型版本、评估协议、种子完全相同。

结果 11/20 + 9 次跌倒 vs 20/20 + 0 次跌倒，航向误差 6.4° vs 38.3°。

~~**因此当前的瓶颈是参考步速**~~（已收回，见本节顶部说明），不是数据直线度、不是模型版本、不是拼接缝、不是朝向：
策略被训练成以 ~0.79 m/s 行走，当参考只有 0.46 m/s 时它无法维持平衡，
表现为前倾跌倒（tilt 最高 66.8/79.5°）。

这解释了为什么"找一个 ≥10 s 的直线片段"这件事本身解决不了问题：
AMASS 里所有满足 ≥10 s 的候选（0.457、0.507 m/s）都明显慢于策略的训练步速。

### 下一步最小实验（数据前提已具备，不再是"缺数据"）

1. **速度匹配的单因子实验（最直接）**：把 `cmu120_19_straight_h000` 的**时间轴按比例压缩**
   到参考速度 ≈0.87 m/s（30 fps → 约 57 fps 等效，即重采样），保持几何与朝向不变。
   注意这不是"循环或拼接"，是**播放速率缩放**，必须在报告中明确标注为变速参考。
   **判据**：若 A 从 11/20 升到 ≥18/20 且跌倒消失，则"速度"被确认为唯一瓶颈；
   若仍不足，则问题在慢速步态的动力学本身（跌倒而非跟踪误差）。
2. 若 1 成立，则真正需要的是**一段 ≥10 s、≥0.8 m/s 的连续直线真实行走**。
   按本轮的教训，筛选必须同时满足：`trans` 直线窗 ≥10 s、窗口速度 ≥0.75 m/s、
   **朝向 FORWARD**（`scripts/amass_facing.py`）、起点在原点附近。
   KIT 与 CMU 都不满足，需要新的数据源（跑步机/走廊连续行走）。

---

## M.9 变更清单

| 文件 | 说明 |
|---|---|
| `scripts/build_smplh_neutral.py` | 新增：无 conda/chumpy 合成 `SMPLH_NEUTRAL.pkl`，含 smplx 前向校验与输入/输出哈希 |
| `scripts/amass_facing.py` | 新增：正向/倒退行走判定（SMPL-H FK，与坐标约定无关） |
| `scripts/slice_amass_clip.py` | 新增：连续帧切片（无插值/无混合） |
| `scripts/retarget_one.py` | 新增第 3 道关卡「朝向」；四道关卡任一不过即拒绝 |
| `scripts/qc_retargeted.py` | 新增 `check_root_rotation`、`check_contact`、起点原点检查、`--compare-raw` 原始对照；`load_clip` 支持 GMR 稀疏缓存（FK 重算） |
| `scripts/rotate_reference.py` | 默认把参考平移到原点；元数据字段缺失时从模型补齐 |
| `reports/amass_facing.json` | 新增：两个候选的朝向判定 |
| `reports/qc_CMU_136_136_25_poses.json`、`qc_CMU_120_cmu120_19_straight_h000.json` | 新增 |
| `reports/data_prep_120_19_straight.json` | 新增：切片清单 |
| `~/.musclemimic/smpl/SMPLH_NEUTRAL.pkl` | 新增（88.4 MB） |
