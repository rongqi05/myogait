# L. 拼接缝的四元数 bug、以及走偏究竟是不是"固有偏置"

日期：2026-11-01 ｜ 环境：`.venv`（验收）· `.venv-gmr`（重定向）
新增：`scripts/rotate_reference.py`、`scripts/rotation_bias_report.py`；
修改：`scripts/prepare_straight_walk.py`（slerp 修复）、`scripts/qc_retargeted.py`（新增根朝向连续性检查）

---

## L.0 结论摘要

| # | 事项 | 结果 |
|---|---|---|
| 1 | `SMPLH_NEUTRAL.pkl` | **仍未获得**，`136_25` 重定向依然卡在这一关（与上一轮相同） |
| 2 | 发现并修复**我自己的 bug** | `straight_stitched_10s` 拼接缝处根四元数**反极性**（dot = −0.9926），线性插值穿过零点 → 骨盆 **165.8°/帧** 的姿态跳变 |
| 3 | 修复效果（单因子，配对 20 种子） | 进度 0.823→**0.885（20/20 变好）**，跌倒 1→**0**，cross-track 1.038→0.751，侧偏 1.056→**0.769**；**仍然 0/20 通过** |
| 4 | 走偏是"固有偏置"吗 | **不是简单的固有偏置。** 刚性旋转参考 0/90/180/270° 后表现差异巨大（veers −3.3°/−22.9°/−4.8°/−23.1°）→ 策略**对世界朝向不敏感/受训练分布限制** |
| 5 | 执行器左右不对称 | **排除**：145 对左右肌肉，gain 与 ctrlrange 全部逐位相同 |
| 6 | 10 s 直线基线 | **仍未通过**。本策略在干净直线参考上的最好侧偏均值 0.651 m（最小 0.366 m），门限 0.35 m |

---

## L.1 上一轮的结论需要收回一半

上一轮我写的是"失效是**持续性单侧侧移**，等效 6.5° 蟹行，20/20 同号，是策略/模型的系统性偏置"。

其中"20/20 同号""侧偏≈侧向漂移累积"是对的，但**"固有偏置"这个归因过头了**。
本轮做了两件事后，图景变了：

1. 修掉拼接缝的四元数 bug 后，等效侧走角从 **−6.48° 降到 −4.17°**（sd 4.30 → 1.99）；
2. 把参考刚性地转到别的世界朝向后，这个角**在 −3.3° 到 −23.1° 之间大幅变化**。

所以正确表述是：

> 走偏由**两部分**组成：(a) 拼接缝引入的姿态跳变（已修复，贡献约 1/3）；
> (b) 一个**依赖参考世界朝向**的残差，在沿 ±x 走时约 −3°~−5°（10 s 累积约 0.5 m），
> 沿 ±y 走时放大到 −23°。由于绕 z 旋转是重力+地面+模型的严格对称操作，
> 这一朝向依赖只能来自**策略的输入编码与训练分布**，不能来自物理。

---

## L.2 拼接缝 bug：三行代码，165.8°/帧

### L.2.1 现象

一次常规 QC 里我给 `qc_retargeted.py` 加了"根朝向连续性"检查，结果旧拼接参考直接失败：

```
straight_stitched_10s        max  165.77 deg/frame at frame  665 (t=6.65s)  p99 1.28
straight_stitched_10s_slerp  max    1.14 deg/frame at frame  656 (t=6.56s)  p99 0.89
```

除拼接缝外全程 p99 只有 1.28°/帧，而缝上单帧转了 **165.8°**——0.05 s 内骨盆姿态甩了约 290°。

### L.2.2 根因（可复核）

`prepare_straight_walk.py` 的 `stitch()` 原来这样混合根四元数：

```python
blended = (1.0 - w) * tail + w * head
blended[:, 3:7] /= np.linalg.norm(blended[:, 3:7], axis=1, keepdims=True)
```

四元数是**双覆盖**：`q` 与 `−q` 是同一个姿态。实测两个片段在缝处的根四元数：

```
A tail quat [ 0.7555 -0.0337 -0.0357  0.6533]
B head quat [-0.8208  0.0015  0.0781 -0.5658]
dot = -0.9926      -> 两者其实只差 13.9°
lerp magnitude min 0.0803   -> 归一化把方向翻转"藏"了起来
```

`dot < 0` 表示这两个姿态在四元数球面上是**对径点**：线性插值的路径穿过接近原点处，
模长掉到 0.08，再归一化就把方向甩过约 180°。而 `dot` 的符号完全取决于四元数表示，
与物理无关——所以这是一个纯粹的实现错误。

同一个缝的 `root_xy_jump_m` 当时是 `8.9e-16`（平移完全连续），
**清单里没有检查朝向**，所以长时间没被发现。

### L.2.3 修复

`stitch()` 改为**半球对齐 + slerp**（`slerp_quat()`）：

```python
d = dot(q0, q1); q1 = where(d < 0, -q1, q1)   # 半球对齐
theta = arccos(|d|);  out = sin((1-w)θ)/sinθ * q0 + sin(wθ)/sinθ * q1
```

`slerp_quat()` 对 `|dot| > 0.9995` 的近似平行情形退回 lerp（数值更稳，误差可忽略）。
同时在 seam 记录里补上 `root_quat_dot` / `root_quat_antipodal` / `root_quat_angle_deg`。

新的清单（`reports/data_prep_straight_stitched_slerp.json`）现在会明说：

```json
{"index": 656, "crossfade_frames": 20, "root_xy_jump_m": 8.9e-16,
 "root_quat_dot": -0.9926, "root_quat_antipodal": true, "root_quat_angle_deg": 13.93}
```

**注意**：旧产物 `straight_stitched_10s.npz` 我没有覆盖，保留作审计留痕。
新产物另存为 `straight_stitched_10s_slerp.npz`。

### L.2.4 这个 bug 污染了哪些历史结果

所有在 `KIT/3/straight_stitched_10s` 上跑的验收都被污染了，包括：

- `runs/A_ref_mm10m2`（上一轮的三问基线，19/20 走满）
- `runs/TW_stitched`（本轮第一组）
- `reports/three_way.md` 里的 `stitched` 一列
- `reports/decision_straight_walking.md` 中与拼接参考相关的结论

**不**受影响的是所有未拼接片段上的结果：`walk_6m_straight_line06`、`walking_medium09`、
`walk_6m_straight_line04`、`circle` 那些（`runs/C_acceptance*`、`runs/TW_unstitched6m`）。

---

## L.3 QC 工具补上了这个缺口

`qc_retargeted.py` 新增 `check_root_rotation()`：逐帧比较相邻根四元数的**绝对点积**
（对 `q`/`−q` 免疫），报告每帧转角、中位数、p99 与最大值，阈值 20°/帧。
并且它已经被并入 `no_numeric_anomaly` 判据。

```
===== KIT/3/straight_stitched_10s =====
  FAIL  root_rotation: max 165.77 deg/frame @ 6.66 s (median 0.378, p99 1.282)
  no_numeric_anomaly  False
===== KIT/3/straight_stitched_10s_slerp =====
  PASS  root_rotation: max 1.14 deg/frame @ 6.57 s (median 0.378, p99 0.89)
  no_numeric_anomaly  True
```

其余 QC（有限值、89/88/17、滚转、关节突变、足高、足滑）两者都过。
所以这个 bug 只有这一项检查能抓到——之前六项全部 PASS。

---

## L.4 单因子实验：lerp → slerp（配对，dev 种子 0–19）

参考、checkpoint、`--mode stochastic --seconds 10`、阈值、指标版本全部不变，**只有混合方式变了**。

```bash
export PYTHONPATH="external/MuscleMimic:scripts"
.venv/bin/python scripts/eval_walk.py --checkpoint artifacts/checkpoints/checkpoint_12500 \
  --motion_path KIT/3/straight_stitched_10s_slerp --mode stochastic \
  --seeds $(seq 0 19) --seconds 10 --out_dir runs/TW_stitched_slerp
```

| 指标 | lerp（旧） | slerp（新） | 配对差 | 新版本变好的种子 |
|---|---:|---:|---:|---:|
| A 走满 10 s | 19/20 | **20/20** | — | 跌倒的那个 seed 7 恢复 |
| B 跟随参考 | 0/20 | 1/20 | — | — |
| C 通过冻结验收 | 0/20 | 0/20 | — | — |
| `progress_ratio` | 0.823 | **0.885** | −0.062 | **20/20** |
| 沿路径误差 (m) | 1.522 | **0.997** | −0.525 | **20/20** |
| cross-track (m) | 1.038 | 0.751 | −0.287 | 18/20 |
| 侧向漂移 (m) | −0.788 | −0.571 | −0.216 | 18/20 |
| 最大侧偏 (m) | 1.056 | 0.769 | −0.287 | — |
| 平均航向误差 (°) | 8.62 | 6.36 | −2.26 | 16/20 |
| 等效侧走角 (°) | −6.48 ± 4.30 | **−4.17 ± 1.99** | — | — |

**读法**：

- 缝的 bug 确实在拖累策略，而且是**系统性**的——沿路径跟踪 20/20 个种子都变好，
  不是随机波动。一个 5 帧的 290° 姿态跳变，就能让策略在 10 s 里少走 0.5 m。
- 但**修好之后仍然 0/20 通过侧偏**（均值 0.769 m ≫ 0.35 m），
  所以缝不是主因。

---

## L.5 走偏是固有偏置吗？刚性旋转实验

### L.5.1 设计

把**干净版**参考（`straight_stitched_10s_slerp`）绕世界 z 轴刚性旋转 0°/90°/180°/270°：

- 根平移旋转、根朝向左乘偏航四元数，**关节角一字不改**；
- 因此步态、足相对骨盆的位置、速度全部逐位不变；
- 唯一变化是参考在世界里的朝向。

绕 z 旋转是重力、地面、模型（肌肉 moment arm、wrap 体）的严格对称操作，
所以**物理上四个变体应当等价**。若不是等价，差异只能来自策略的输入或训练分布。

变换正确性已校验：

```
qvel from qpos: max |root angular velocity| = 1.799 rad/s (103.1 deg/s)   # 合理
invariance check: net displacement 11.6340 m across angles, violations=[]  # 保持
rot000 vs source site_xpos max abs diff: 9.5e-07 m                         # 逐位复现
```

（顺带说明：`mj_differentiatePos` 对 `q → −q` 不敏感，实测差 1.45e-14，
所以源文件里 674/1340 个 `w<0` 的四元数不会造成速度爆炸；
真正的爆炸来自 L.2 的 lerp。）

### L.5.2 结果（每个朝向 10 个种子）

| 参考朝向 | A 走满 10 s | 侧偏均值 (m) | 侧偏最小 (m) | cross-track | `progress_ratio` | 等效侧走角 |
|---|---:|---:|---:|---:|---:|---:|
| 0°（沿 +x） | 10/10 | **0.651** | **0.366** | 0.633 | 0.883 | **−3.25° ± 1.55** |
| 90°（沿 +y） | 9/10 | 2.620 | 1.139 | 2.603 | 0.662 | **−22.93° ± 12.53** |
| 180°（沿 −x） | 10/10 | 0.785 | 0.591 | 0.767 | 0.922 | **−4.77° ± 1.19** |
| 270°（沿 −y） | 9/10 | 2.906 | 1.091 | 2.880 | 0.742 | **−23.14° ± 17.87** |

### L.5.3 解释

`rotation_bias_report.py` 的判定输出是 **`ambiguous`**（既非 body-fixed 也非 world-fixed），
这正是重点：

```
veer-vs-reference spread across rotations: 19.88 deg
world-frame crab spread across rotations : 19.79 deg
```

- **不是 body-fixed**（否则四个朝向的侧走角应当相同）——所以不是"模型天生左右不对称"。
- **不是 world-fixed**（否则侧走角会随旋转整体平移，`world_crab` 应当相同）——所以也不是
  "某个世界坐标轴上的外力/偏置"。
- **而是朝向依赖**：0°/180°（±x）明显好于 90°/270°（±y）。

物理上四个变体等价，所以差异必然来自观测或训练分布。查了观测定义：

```
myofullbody.py:344  ObservationType.FreeJointPosNoXY("q_free_joint", ...)   # n=5
myofullbody.py:351  ObservationType.FreeJointVel("dq_free_joint", ...)      # n=6
```

`FreeJointPosNoXY` 去掉了 XY 平移（所以策略有平移不变性），但**保留了完整的世界系根四元数**，
`dq_free_joint` 也带世界系线速度与角速度。因此观测对绕 z 旋转**不是不变的**：
参考沿 +y 走时，喂给策略的是一组它很少见到的输入。

而官方训练数据大体沿一个方向（缓存里 KIT 片段净位移朝向约 −160°~180°，
即大致 ±x），所以 ±y 属于分布外。这就解释了为什么 0°/180° 都还行、90°/270° 崩掉。

**同时，执行器对称性已被排除**：`MyoFullBody` 有 145 对 `_l`/`_r` 肌肉，
`actuator_gainprm[:,0]` 与 `actuator_ctrlrange` 在每一对上都**逐位相同**
（例如 `vasmed_r = vasmed_l = 1.14989`）。所以左右力不对称不是原因。

### L.5.4 对"基线"的影响

- 即使挑最好的朝向（0°），侧偏均值 **0.651 m**、最小 **0.366 m**，**10/10 全部不通过** 0.35 m。
  最接近的那个只超出 5%——这给"再多一点训练/再多一点参考质量就能过"留下了空间，
  但**现在**的结论必须是未通过。
- 更重要：**策略是朝向锁定的**。这对毕业设计有直接影响——将来加拐杖、改患侧肌力后，
  策略很可能只能在它熟悉的朝向上正常走。任何新实验都必须**固定参考朝向并记录在案**，
  否则左右对比会被朝向效应污染。

---

## L.6 判定（更新）

### `136_25` 单因子实验

> **因数据不足无法判定**（缺 `SMPLH_NEUTRAL.pkl`；前置关卡仍是通过状态，本轮无变化）。

### 10 s 直线稳定行走

> **未通过。** 本策略在干净直线参考上的最好成绩是侧偏均值 0.651 m / 最小 0.366 m，门限 0.35 m，0/10 通过。
> 拼接缝 bug 修复后成绩改善（进度 20/20 变好、跌倒消失），但仍不达标。

### 上一轮归因的修正

> "失效是策略/模型固有单侧偏置"这个说法**收回**。
> 正确表述是：约 1/3 来自拼接缝姿态跳变（已修复），其余是**依赖参考世界朝向**的残差，
> 源于观测含世界系根姿态 + 训练分布集中在少数朝向。

---

## L.7 下一项最小可检验实验

按顺序，一次一个因子，都不需要 SMPL-H：

1. **朝向消融收尾（最便宜，直接指导后续实验）**：把 90°/270° 的种子数从 10 扩到 20，
   确认"±y 崩坏"不是小样本假象。**判据**：若 20 种子下 `progress_ratio` 仍 ≤0.75 且侧走角 ≤−20°，
   则朝向效应成立，后续所有实验一律固定参考朝 0°。
2. **分离"朝向输入"与"训练分布"**：把参考朝向设为 0°，但把环境初始 yaw 与观测里的
   根四元数人为置零/规范化（不改奖励、不改动作），看侧走角是否变化。
   **判据**：若侧走角从 −3.3° 变到接近 0，说明是输入编码在起作用；不变则是策略本身。
3. **残余 −3.3° 的来源**：在 0° 朝向下做左右镜像（需要左右关节/肌肉重映射）实验。
   这是唯一能把"策略偏左"与"策略偏慢"分开的干净测试，也直接服务你后续的患侧对比。
4. 只有 1–3 定性后，才谈改奖励/扩训练数据/提高步速。**在此之前不动 checkpoint、不动阈值、不训练。**

---

## L.8 变更清单

| 文件 | 说明 |
|---|---|
| `scripts/prepare_straight_walk.py` | **修 bug**：根四元数改用半球对齐 + slerp 交叉淡化；seam 记录补 `root_quat_dot`/`antipodal`/`angle` |
| `scripts/qc_retargeted.py` | 新增 `check_root_rotation()`（逐帧绝对点积，对 `q/−q` 免疫），并入 `no_numeric_anomaly` |
| `scripts/rotate_reference.py` | 新增：参考的刚性 z 旋转（校验平移不变性与 rot000 逐位复现） |
| `scripts/rotation_bias_report.py` | 新增：判定 body-fixed / world-fixed / 朝向依赖 |
| `reports/data_prep_straight_stitched_slerp.json` | 新增：修复版数据准备清单（含反极性证据） |
| `reports/three_way_slerp.json/.md` | 新增：三组（旧拼接 / 修复拼接 / 未拼接）三问对比 |
| `reports/three_way_rot.json/.md` | 新增：四个朝向的三问指标 |
| `reports/rotation_bias.json` | 新增：朝向判定原始数据 |
| `runs/TW_stitched_slerp/`、`runs/ROT_*/` | 新增：20 + 40 个逐 seed 结果 |
| `~/.musclemimic/caches/.../KIT/3/straight_stitched_10s_slerp.npz` | 新增参考（旧文件保留） |
| `~/.musclemimic/caches/.../KIT/3/rotref_slerp_rot{000,090,180,270}.npz` | 新增：4 个旋转变体 |
