# G. `~/Documents/muscle/data` 候选片段审计

日期：2026-10-31 ｜ 目的：回答"muscle 文件夹里有没有可用的 ≥10 s 连续直线正常步行片段"

## 结论

> **没有可用于 MyoFullBody 基线实验的片段。**
>
> 目录里确实有大量长片段（92 条 BVH ≥10 s），但满足"≥10 s 且直线"的只有两条，
> 分别是 **ZombieWalk** 和 **Frankenstein male walk** —— 风格化步态，不是正常行走。
> 且这些数据是 **BVH**，不是 GMR-Fit 需要的 SMPL-H 格式，转换还需要缺失的 SMPL 人体模型。

## G.1 目录构成

| 路径 | 大小 | 内容 |
|---|---:|---|
| `cmu_mocap/` | 226 M | **233 个 BVH**（CMU mocap，6 个受试者：008/009/035/091/104/105），120 fps |
| `cmu_bio_npy/` | 47 M | 245 个 `.npy`（BIO 格式，MS-Human-700 模型） |
| `patient_bvh/` | 860 M | 716 个 BVH（偏瘫患者，**非正常**） |
| `patient_bio_npy_repaired/` | 112 M | 患者 BIO 数据 |
| `walking_bio.pt` | 154 M | torch zip（966 个成员），MS-Human-700 的 walking 数据集 |
| `bvh.zip` | 102 M | BVH 打包 |
| `cmu_walk.yaml` | 12 K | 项目自选的 walking 片段清单 |

`.npz` 数量：**0** —— 没有任何 AMASS/SMPL-H 格式文件。

## G.2 筛选方法与结果

`scripts/bvh_straightness.py`（新增）只解析 BVH 头部与根节点位移通道后测：
时长、路径长度、**0.5 s 窗口 PCA 航向**、任意 2 s 内最大航向变化、最长直线窗口
（容差 ±20°，速度 >0.2 单位/s）。完整结果：`reports/bvh_straightness_cmu.json`。

| 统计 | 数量 |
|---|---:|
| BVH 总数 | 233 |
| 解析成功 / 失败 | 233 / 0 |
| 时长 ≥10 s | **92** |
| 描述非风格化且 ≥10 s | 72 |
| **≥10 s 且直线窗口 ≥10 s** | **2** |
| 其中属于正常行走 | **0** |

### 最长的两条（也是唯一通过时长+直线窗口的）

| clip | 时长 | 直线窗口 | 描述（CMU 官方索引） | 判定 |
|---|---:|---:|---|---|
| `104_41` | 12.98 s | 11.90 s | **ZombieWalk** | 风格化，非正常 |
| `104_03` | 13.49 s | 10.92 s | **Frankenstein male walk, exact footfalls** | 风格化，非正常 |

### 最好的"正常"候选（但直线窗口不足）

| clip | 时长 | 直线窗口 | 描述 |
|---|---:|---:|---|
| `104_51` | 11.52 s | 9.40 s | Unknown |
| `104_47` | 14.12 s | 8.37 s | Unknown |
| `104_49` | 22.72 s | 8.18 s | Unknown |
| `104_22` | 15.32 s | 5.40 s | RegularWalk8 |
| `105_01` | 22.81 s | 3.53 s | WalkDigital8（数字 8 字路线 = 转弯） |
| `35_27` | 81.72 s | 3.98 s | navigate around obstacles（绕障） |

CMU 的 walking 片段多为**短程穿过房间**（2–7 s），长片段基本是绕圈、绕障、数字 8、
或风格化表演，因此"长 + 直 + 正常"三者无法同时满足。

## G.3 额外的数据质量问题（实测）

同一批 BVH 的**单位与坐标轴约定不一致**：

| clip | 根节点通道范围（x, y, z） | 由髋高推算的比例 |
|---|---|---|
| `104_41` | 77.7, 1.1, 6.4 | 0.055 m/unit（髋高 17.2 单位） |
| `105_01` | 37.7, 0.9, 77.5 | 0.062 m/unit（髋高 15.4 单位） |
| `35_01` | 1.3, 0.7, 67.9 | 0.053 m/unit（髋高 17.8 单位） |
| `08_01` | 0.6, 1.6, 65.2 | **0.129 m/unit（髋高 7.4 单位）** |

`08_01` 的髋高只有 7.4 单位，与其它受试者差 2 倍以上 —— 说明**跨受试者不能共用一个尺度**。
项目自带的 `data/scripts/inspect_bvh.py` 也会报：
`[警告] 骨架高度 23.6 (cm) 异常偏小, 疑似数据整体缩放, 建议人工确认 scale`。
这也解释了为什么这批数据在 `muscle` 项目里需要先转成 BIO `.npy` 再使用。

## G.4 为什么不能直接拿来重定向

1. **格式不匹配**：GMR-Fit 的入口是 `--gmr-src-human smplh`，输入必须是 SMPL-H 参数化动作
   （AMASS `.npz`：`poses/trans/betas`）。BVH 是关节旋转 + 根位移，**不是** SMPL-H。
2. **缺 SMPL 人体模型**：BVH→SMPL-H 需要 SMPL/SMPL-H/SMPL-X 的 `.pkl` 模型文件，
   本机没有，且需在 smpl.is.tue.mpg.de 注册后人工下载（不可脚本化）。
3. **缺 GMR 依赖**：`torch`、`smplx`(amathislab fork)、`mink`、`gmr_plus` 全部未安装。
4. **`cmu_bio_npy` / `walking_bio.pt` 不可用**：它们是 **MS-Human-700** 的 BIO 数据，
   与 MyoFullBody（`nq=89/nv=88/nu=354`）是不同模型，不能改名或简单转换当成 MyoFullBody 参考。
5. **患者 BVH 不适用**：本阶段要求"正常人体"。

## G.5 判定与下一步

| 问题 | 回答 |
|---|---|
| muscle 文件夹里有 ≥10 s 连续直线**正常**步行吗 | **没有** |
| 有 ≥10 s 且直线的片段吗 | 有 2 条，都是风格化步态（Zombie / Frankenstein） |
| 能直接重定向到 MyoFullBody 吗 | **不能**（BVH 非 SMPL-H，且缺 SMPL 模型与 GMR 依赖） |

**下一步最小实验（与 F 报告一致，路径更新）**：
优先获取一段**已经是 SMPL-H 格式**的 ≥10 s 直线正常步行（AMASS 原始数据），
这样可跳过 BVH→SMPL 拟合这一步；若只能用 BVH，则必须先补 SMPL 人体模型并做
BVH→SMPL-H 拟合，再用 GMR-Fit 重定向，且**跨受试者必须逐条标定单位尺度**。

在拿到合格参考前，不启动训练、不改阈值、不宣称完成实验。
