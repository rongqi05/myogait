# I. GMR-Fit 重定向工具链（`.venv-gmr`）

日期：2026-10-31 ｜ 脚本：`scripts/setup_gmr_env.sh`（建环境）、`scripts/check_gmr_toolchain.py`（自检）
结果：`reports/gmr_toolchain.json`

## I.0 一句话结论

> **工具链已装好并逐项自检通过：GMR-Fit、SMPL-H 解析、MyoFullBody IK 配置、
> qpsolvers/daqp、EGL 渲染全部可用，且模型与 1.0.5 变体对齐。**
>
> **仍卡在两个"资产"门槛上，二者都不是 pip 能装的：**
> 1. `SMPLH_NEUTRAL.pkl`（SMPL-H 人体模型，需 MANO 网站注册 + 接受许可）；
> 2. AMASS 解压目录（`~/Downloads/CMU.tar.bz2` 已下载，尚未解压到约定位置）。
>
> 在 I.5 的两项到位前，`retarget_dataset.py` 会在 `fit_gmr_motion()` 的第一行
> `FileNotFoundError: SMPL-H models not found at ...` 处失败。

## I.1 为什么新建一个环境，而不是装进 `.venv`

三个环境职责分离，互不污染：

| 环境 | 用途 | jax | torch | 状态 |
|---|---|---|---|---|
| `.venv` | **冻结的验收环境**（`eval_walk.py`、10 s/20 s 判据） | CPU | 无 | 已冻结，本次未改动 |
| `.venv-train` | PPO + warp 训练 | CUDA12 | 无 | 已冻结，本次未改动 |
| `.venv-gmr` | **本次新建**：GMR-Fit 重定向 | CPU | **CPU-only** | 新增 |

两条关键理由：

- `.venv` 是 20 种子验收的基准，往里装 `torch`/`smplx` 会改变 `numpy`/`scipy`
  的解析结果，使"冻结"失效。
- 重定向是**纯 IK 问题**：不动力学步进、不跑 JAX、不训练。所以 `torch` 用 CPU 版
  即可，**完全不占 8 GB 显存**（`torch==2.15.0+cpu`，`cuda=False` 已实测）。

## I.2 安装内容

创建命令（`scripts/setup_gmr_env.sh` 的可执行版本）：

```bash
cd external/MuscleMimic
UV_PROJECT_ENVIRONMENT=/home/zrq/Documents/musclemimic/.venv-gmr \
  uv sync --extra smpl --extra gmr --python 3.11
```

`smpl` extra = `torch` + `smplx`(amathislab fork) + `joblib`；`gmr` extra =
`general-motion-retargeting`(gmr_plus)，后者再带来 `mink` / `qpsolvers[daqp]` /
`redis` / `natsort` 等。

锁定的实际版本（`uv sync` 解析结果，非猜写）：

```
musclemimic==0.1.1            mujoco==3.4.0        numpy==2.2.6
jax==0.7.2                    mujoco-mjx==3.4.0    scipy==1.17.1
flax==0.12.0                  optax==0.2.8         orbax-checkpoint==0.11.37
warp-lang==1.10.0             distrax==0.1.7
torch==2.15.0+cpu             smplx==0.1.28        joblib==1.6.0
general_motion_retargeting==0.2.0                  mink==1.1.0
qpsolvers==4.13.0             daqp==0.10.3
```

两个 VCS 依赖的 commit（`direct_url.json` 读出，便于复现）：

| 包 | 来源 | commit |
|---|---|---|
| `smplx` | `github.com/amathislab/smplx` | `a5b8e4ac14f79f3f33fd2cf2a16e6f507146b813` |
| `general_motion_retargeting` | `github.com/amathislab/gmr_plus.git` | `1b4eb9864360…` |

上游工作树仍干净（`git status --short` 无输出），`uv.lock` 被上游 `.gitignore` 覆盖，
**没有污染 `external/MuscleMimic`**。Python 由 uv 管理，为 `3.11.16`。

## I.3 自检结果（`scripts/check_gmr_toolchain.py`）

```
[packages]               全部 PASS
  musclemimic_models 1.0.5 | mujoco 3.4.0 | numpy 2.2.6 | jax 0.7.2
  torch 2.15.0+cpu (cuda=False) | smplx 0.1.28 | gmr 0.2.0 | mink 1.1.0 | daqp 0.10.3

[gmr_integration]
  PASS  _GMR_INSTALLED                  True
  PASS  _GMR_SHAPE_FITTING_INSTALLED    True
  PASS  qpsolvers solvers               daqp, jaxopt_osqp
  PASS  smplh_to_myofullbody.json       17+17 IK matches

[model]
  PASS  MyoFullBody dims                nq=89 nv=88 nu=354 neq=51
  PASS  left-knee eq sign               1 positive / 1 negative knee polycoefs (1.0.5 layout)
  PASS  sites                           2038

[rendering]  PASS  EGL offscreen        (64, 64, 3)

[resources]  disk 101.7 GB free | RAM 30 GiB avail 24 GiB | swap 0
             GPU RTX 5060 Laptop 8151 MiB (7143 MiB free) | .venv-gmr 2.5 GB
```

逐项含义：

- **`nq=89 nv=88 nu=354`**：`MyoFullBody()` 环境模型的维度，与
  `configs/eval_walk.json` 冻结值完全一致。重定向会把 GMR 的 IK 模型替换成
  这个"手指已禁用"的版本（`fit_gmr_motion()` 里的 `retarget.model = env._model`），
  所以维度对不上就不能用。
- **`_GMR_INSTALLED` / `_GMR_SHAPE_FITTING_INSTALLED`**：`loco_mujoco.smpl.retargeting`
  在 import 期用 `try/except ImportError` 置位。两者都必须为 `True`，
  否则 `fit_gmr_motion()` 会直接抛 `ValueError: GMR ... required`。
- **`smplh_to_myofullbody.json` 17+17 IK matches**：把 SMPL-H 关节绑到 MyoFullBody
  网站的映射表；这是"重定向"真正发生的地方。
- **EGL**：`retarget_dataset.py` 顶部强制 `MUJOCO_GL=egl`，实测可离屏渲染。
- **CPU torch**：确认没有偷偷带上 CUDA，重定向期间显存完全空出来给训练用。

## I.4 一个必须记住的坑：`uv sync` 会把模型悄悄升到 1.0.6

`pyproject.toml` 里 `musclemimic_models>=1.0.2` 是范围约束，`uv sync` 解析到最新
**1.0.6**；而验收环境用 **1.0.5**。两者的差别是 D 阶段定位到的单因子：
1.0.6 把左膝两个等式约束的 polycoef **整体取反**，导致 `mm-10m-2` 在 2.5–2.8 s 跌倒
（`reports/D_model_version_compatibility.md`）。

重定向阶段虽然不跑动力学，但 IK 的目标模型必须是**同一个 MyoFullBody 变体**，
否则新旧参考不在同一模型上，配对比较失去意义。因此：

- `configs/constraints.txt` 写入 `musclemimic_models==1.0.5`；
- `setup_gmr_env.sh` 在每次 `uv sync` 之后强制回退到 1.0.5；
- 自检脚本用**约束符号模式**检测这件事——两条膝约束的 `coef[1]` 绝对值都是
  `0.369499`，1.0.5 是"一正一负"，1.0.6 是"两条都正"：

```
PASS  left-knee eq sign   1 positive / 1 negative knee polycoefs (1.0.5 layout)
```

如果哪天自检报 `2 positive / 0 negative (1.0.6 layout: left-knee polycoef negated)`，
说明有人重跑了 `uv sync` 而没回退——此时**不要**拿去生成参考。

## I.5 剩余的两项资产门槛

自检的 `BLOCKED` 输出（原样）：

```
BLOCKED: 2 blocking item(s)
  - SMPLH_NEUTRAL.pkl (runtime body model): /home/zrq/.musclemimic/smpl/SMPLH_NEUTRAL.pkl
  - AMASS dataset root: /home/zrq/.musclemimic/AMASS (0 *_poses.npz)
```

### I.5.1 `SMPLH_NEUTRAL.pkl` —— 需要人工注册下载

调用链决定了**唯一**的运行时文件名：

```
fit_gmr_motion()
  -> load_smplh_file(..., smpl_model_path, ...)
     -> SMPLH_Parser(model_path=smpl_model_path, gender="neutral", use_pca=False)
        -> smplx.body_models.SMPLH(...)   # amathislab fork, 默认 ext="pkl"
           -> <smpl_model_path>/SMPLH_NEUTRAL.pkl
```

注意 amathislab fork 找的是 `<dir>/SMPLH_<GENDER>.pkl` **直接位于目录下**，
不是上游 smplx 的 `smplh/neutral/model.npz` 布局。这个 `.pkl` 由
`loco_mujoco/smpl/install_smplh.sh` 合成，输入是三份**许可受限**文件：

```
~/.musclemimic/smpl/
├── smplh/neutral/model.npz              # MANO 网站 "Extended SMPL+H model"
├── mano_v1_2/models/MANO_LEFT.pkl       # MANO 网站 "Models & Code"
└── mano_v1_2/models/MANO_RIGHT.pkl
```

这三份需要以**你自己的账号**在 <https://mano.is.tue.mpg.de/download.php> 接受许可后下载。
我无法代下载，也不会绕过——这是有版权的模型文件。

放好之后生成运行时文件（该脚本会临时建 conda 环境装 `chumpy` 并打 numpy 兼容补丁）：

```bash
cd external/MuscleMimic && bash loco_mujoco/smpl/install_smplh.sh
```

（若 conda 不可用，等价做法是只做"合并 SMPL-H npz + MANO 双手参数 → `SMPLH_NEUTRAL.pkl`"，
逻辑见 `loco_mujoco/smpl/generate_smplh_model.py`，需要的键是
`hands_meanl/r`、`hands_componentsl/r`、`hands_coeffsl/r`。）

### I.5.2 AMASS 目录

`retarget_dataset.py` 按 `os.path.join(AMASS_root, f"{name}.npz")` 取文件，
`name` 形如 `KIT/314/walking_medium09_poses`。所以需要**解压后的树**：

```
~/.musclemimic/AMASS/CMU/38/38_03_poses.npz
```

`~/Downloads/CMU.tar.bz2`（1.9 GB，`bzip2 -t` 校验通过，**2082 个 `*_poses.npz`**）
恰好就是这个布局，只是还没解开。这一步**不需要** SMPL 模型，所以可以现在做：

```bash
mkdir -p ~/.musclemimic/AMASS
tar -xjf ~/Downloads/CMU.tar.bz2 -C ~/.musclemimic/AMASS
```

（本次未解压：数据门槛未过之前解压没有意义，见 `reports/J_cmu_source_audit.md`。）

## I.6 资源占用

| 项 | 实测 |
|---|---|
| `.venv-gmr` 磁盘 | **2.5 GB**（对比 `.venv` 1.8 GB、`.venv-train` 6.6 GB） |
| 剩余磁盘 | 101.7 GB |
| 系统内存 | 30 GiB，可用 24 GiB，**swap 0** |
| 显存 | 重定向**不占用**（CPU torch）；当前 7143 MiB 空闲 |

重定向单条运动是单进程、内存占用由单条 clip 的帧数决定；`retarget_dataset.py`
用 `maxtasksperchild=1` 的 `Pool`（`--workers`），默认 `min(cpu_count, 8)`。
在 0 swap 的机器上建议 `--workers 4` 起步并盯 RSS——**这是尚未实测的一项**。

## I.7 资产到位后的确切命令

```bash
# 1) 自检应当不再有 BLOCKED
.venv-gmr/bin/python scripts/check_gmr_toolchain.py

# 2) 重定向单条运动（参数与 reports/freeze_straight_reference.json 一致）
cd external/MuscleMimic
MUJOCO_GL=egl /home/zrq/Documents/musclemimic/.venv-gmr/bin/python scripts/retarget_dataset.py \
  --model MyoFullBody --retargeting-method gmr \
  --gmr-src-human smplh --gmr-target-fps 30 --gmr-solver daqp --gmr-damping 0.5 \
  --dataset CMU/<subject>/<clip>_poses --workers 1
```

注意 `--dataset` 的取值是**不带 `.npz` 的相对路径**，且 `retarget_dataset.py`
的默认值是 `MyoBimanualArm` + 一组预置 `dataset_group`——单条运动必须显式传
`--model MyoFullBody` 和上面的相对路径，否则会去跑双臂数据集。

## I.8 尚未验证的部分（诚实清单）

1. **完整重定向尚未跑通一次**：`fit_gmr_motion()` 在 body model 之前的部分
   （env 构建、IK 配置注入、qpsolvers 求解器可用性）已验证，但其后的
   SMPL 前向、`ensure_gmr_fitted_shape()` 的 500 次 shape fitting、
   `mink` IK 全流程**尚未执行**——缺 I.5.1 的模型文件。
2. **shape fitting 的收敛性与耗时未测**：首次运行会拟合一个
   `myofullbody_shape.pkl`（默认 `shape_fitting_iterations=500`，CPU），
   耗时与是否收敛未知。
3. **`--workers` 的内存上限未测**：0 swap 环境下安全并发数未定。
4. **重定向后的 `qpos/qvel` 质量未检验**：需要 `scripts/motion_straightness.py`
   + 视频人工核对，等 I.5 通过后做。

## I.9 变更清单

| 文件 | 说明 |
|---|---|
| `scripts/setup_gmr_env.sh` | 新增：建 `.venv-gmr` + 回退 1.0.5 + 自检 |
| `scripts/check_gmr_toolchain.py` | 新增：14 项包/集成/模型/资产/渲染/资源自检 |
| `configs/constraints.txt` | 新增：`musclemimic_models==1.0.5` 约束 |
| `scripts/amass_kit_audit.py` | 修 bug：短片段（<0.5 s 窗口）`np.convolve` 长度不匹配致崩溃；跳过 `shape.npz`；`walk` 关键字改为 `--keyword` 参数 |
| `.gitignore` | 增加 `.venv-gmr/` |
| `reports/gmr_toolchain.json` | 新增：自检原始结果 |
