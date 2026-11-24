# myogait

这个仓库记录 MyoFullBody 人体模型的行走回放、肌力干预和验收实验。毕业设计最终要比较拐杖机器人在健侧与患侧辅助时的效果，并观察这种差异是否随患侧上肢恢复程度发生改变。这里先把无拐杖时的正常肌力行走和单侧减力反应做实，供后续辅助实验使用。

目前使用 [MuscleMimic](https://github.com/amathislab/musclemimic) 的 `mm-10m-2` 预训练策略与 MyoFullBody 肌肉骨骼模型。策略读取当前状态和运动参考，输出肌肉控制量；MuJoCo 计算肌肉、接触与人体运动。每个回合只在开始时重置一次，之后不周期性覆盖根节点位置。参考轨迹仍是运动模仿任务的目标，实验结果会受参考的速度、长度和朝向影响。

## 目前做到哪里

| 实验 | 固定条件 | 结果 |
| --- | --- | --- |
| 连续正常肌力行走 | CMU 136 拼接参考；官方 checkpoint；20 个随机种子 | 10 s 和 20 s 均为 **20/20 走满、无跌倒**；20 s 全部未通过预设直线判据，中位最大侧偏 0.831 m |
| 患侧上肢单独减力 | CMU 120 单段参考；左臂 32 条肌肉的 \(F_0\) 乘 0.7；策略冻结 | 10 s **14/20 走满**（P 报告口径）；随强度扫描，侧向运动与足滑等指标有变化 |
| 患侧下肢单独减力 | 同一 CMU 120 参考；左腿 40 条肌肉的 \(F_0\) 乘 0.7；策略冻结 | **20/20 在约 3.1 s 跌倒** |
| 完整肌力持杖行走基线 | CMU 136 参考；F0 全为 1.0；拐杖机器人接入同一 MuJoCo 场景；3 个配对种子；对称几何变体 | 无拐杖 / 随行不握持 / 左手 / 右手 **5 s 与 10 s 各 12/12 走满**；握点闭合偏差 +0.18 / −1.35 mm，全程 \|Δz\| ≤ 5.2 mm；握持使速度下降 8 %（左）/ 2 %（右），左握侧偏 1.25 m、右握 0.19 m |

20 s 的“不跌倒”和“沿直线保持在阈值内”分别统计。当前 0.35 m 侧偏上限是本仓库预设的工程验收线，尚未用临床数据确定。上表的连续行走与减力实验采用不同参考，数值不能作为同一组配对比较。减力后尚未训练适应策略；跌倒结果反映冻结正常策略受到损伤参数扰动后的反应，不能直接称为偏瘫步态。

实验报告和逐种子表已从仓库移除。运行时产生的轨迹和汇总保存在本地 `runs/`；验收参数与指标定义见 [`configs/eval_walk.json`](configs/eval_walk.json)。

## 代码和数据

| 路径 | 用途 |
| --- | --- |
| `scripts/_env_builder.py` | 加载上游 checkpoint、建立 MuJoCo 环境，按解剖分组缩放肌力 |
| `scripts/eval_walk.py` | 单回合、单次 reset 回放；记录结束原因、速度、侧偏、足滑及肌肉/状态量 |
| `scripts/check_f0_groups.py` | 检查肌肉分组和 \(F_0\) 缩放对象 |
| `scripts/run_f0_ablation.sh` | 顺序运行上肢、下肢和旧语义减力对照 |
| `scripts/train_straight_bounded.sh` | 8 GB 显存配置的训练试验入口；仓库目前采用的主要结果来自官方 checkpoint |
| `crutch_robot.xml` | 拐杖机器人模型：全向底盘（X/Y slide + yaw hinge，仅阻尼）、1.00 m 支撑杆、Ø64 mm 球形把手（球心即握点，z = 1.30 m）、两条软 connect 握持约束、力/力矩与底盘位姿传感器 |
| `env_step_snippet.py` | 底盘随行控制、辅助模式切换、握点对齐、力传感与最小验收；`python env_step_snippet.py` 可跑自检 |
| `scripts/crutch_env.py` | 把 `crutch_robot.xml` 挂进人体 MuJoCo 场景（`mount` / `layout` / `reach`），并保证策略观测与动作维度不变 |
| `scripts/run_crutch_baseline.py` | 完整肌力持杖基线：标定握点、四组条件（无拐杖 / 随行不握持 / 左握 / 右握）、逐种子指标与三组并排视频 |

拐杖机器人可以通过 `scripts/crutch_env.py` 接入人体场景：拐杖以 `crutch/` 前缀挂到人体 spec 下，删除演示假手并按真实手部重建两条软连接，人体与拐杖共用同一 `MjModel` / `MjData`（合并后 `nq/nv/nu = 92/91/356`，人体仍在 `qpos[0:89]` / `ctrl[0:354]`）。观测与动作维度保持不变，同一参考帧下与纯人体模型的观测和策略动作逐元素相同。握持用 `connect`（球铰），只传力不传力矩。

`crutch_robot.xml` 自身的演示把手在 1.30 m（离地 0.05 + 底盘 0.25 + 杆 1.00），而基线按参考姿态的手高重算杆长（见下节），不沿用这个演示值。`CrutchInterface.validate()` 会报出当前模型的手柄高度与相对手部参考高度之差。

上游代码、checkpoint、重定向后的 `.npz` 与原始回放日志未随仓库提交。20 s 实验使用的 CMU 136 连续参考由五段同一受试者的运动拼接而成（28.37 s）；接缝做过姿态插值，尚需检查接缝附近动力学。构建该参考的数据处理脚本已从当前分支移除，可在历史提交 `bc8a8fe` 查阅。若本地没有相应 `.npz`，下面的 20 s 命令无法直接运行。

## 运行环境

记录实验时使用 Python 3.11.15、MuJoCo 3.4.0、`musclemimic_models==1.0.5`、MuscleMimic 源码提交 `47b653db2bce25de834360d2ee6b2dd88142bd78`，以及 Hugging Face 上 `amathislab/mm-10m-2` 的 update 12500 checkpoint。模型版本需要固定：本地对照中 1.0.6 的左膝约束行为与 1.0.5 不同。完整版本和哈希见 [`configs/eval_walk.json`](configs/eval_walk.json)。

在仓库根目录准备 `external/MuscleMimic`、Python 环境 `.venv`、`artifacts/checkpoints/checkpoint_12500` 和重定向后的参考运动；上游仓库与模型包按上述版本检出。评估依赖可按现有上游环境安装，然后以 `python -c 'import jax, mujoco, omegaconf, musclemimic_models'` 检查。模型包安装时固定 1.0.5；不要让依赖解析自动升级到 1.0.6。参考文件位于 `~/.musclemimic/caches/AMASS/MyoFullBody/gmr/`，其路径由 `--motion_path` 指定。

若从空目录开始，先准备上游源码和评估环境（以下命令在 `myogait` 根目录执行）：

```bash
mkdir -p external
git clone https://github.com/amathislab/MuscleMimic.git external/MuscleMimic
git -C external/MuscleMimic checkout 47b653db2bce25de834360d2ee6b2dd88142bd78
git clone https://github.com/amathislab/musclemimic_models.git external/musclemimic_models
git -C external/musclemimic_models checkout v1.0.5
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python \
  numpy==2.2.6 "scipy>=1.14.0" mujoco==3.4.0 mujoco-mjx==3.4.0 \
  warp-lang==1.10.0 jax==0.7.2 jaxlib==0.7.2 flax==0.12.0 \
  "optax>=0.2.4" orbax-checkpoint==0.11.37 distrax==0.1.7 \
  hydra-core pyyaml "ml-collections>=1.1.0" opencv-python gitpython \
  "lxml>=6.0.0" tqdm huggingface_hub "imageio>=2.37.0" \
  "imageio-ffmpeg>=0.6.0" "glfw>=2.9.0" "seaborn>=0.13.2" \
  metrx "wandb>=0.19.10" "viser>=1.0.10" "trimesh>=4.0.0" "Pillow>=10.0.0"
uv pip install --python .venv/bin/python --no-deps ./external/musclemimic_models
```

官方 checkpoint 从 Hugging Face 的 `amathislab/mm-10m-2` 下载到 `artifacts/checkpoints/mm-10m-2`，随后建立指向该目录的绝对路径符号链接：

```bash
mkdir -p artifacts/checkpoints
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("amathislab/mm-10m-2", local_dir="artifacts/checkpoints/mm-10m-2")
PY
ln -sfn "$PWD/artifacts/checkpoints/mm-10m-2" artifacts/checkpoints/checkpoint_12500
```

需要 Hugging Face 镜像的机器可在下载前设置 `HF_ENDPOINT`。上述 checkpoint 和数据文件都需要自行取得；仓库不提供重定向参考的下载链接。

```bash
export PYTHONPATH="$PWD/external/MuscleMimic:$PWD/scripts"

# 已备好 CMU 136 的重定向参考时，回放正常肌力 20 s
.venv/bin/python scripts/eval_walk.py \
  --checkpoint artifacts/checkpoints/checkpoint_12500 \
  --motion_path CMU/136/cmu136_stitched_20s_v2_h000_rot000 \
  --mode stochastic --seeds $(seq 0 19) --protocol 20s \
  --out_dir runs/normal_20s

# 已备好 CMU 120 参考时，只调整左上肢肌力
.venv/bin/python scripts/eval_walk.py \
  --checkpoint artifacts/checkpoints/checkpoint_12500 \
  --motion_path CMU/120/cmu120_19_straight_h000 --start_step 53 \
  --mode stochastic --seeds $(seq 0 19) --protocol 10s \
  --f0_left_upper 0.7 --out_dir runs/upper_070
```

结果写在 `runs/<实验名>/stochastic/` 的逐种子文件与 `summary.csv`。`time_limit_success` 表示回合走到指定时长；`success` 还要求满足预设速度、侧偏、倾角等判据。两个字段应分别报告。

`--f0_left_upper` 和 `--f0_left_lower` 分别对应 32 条左臂和 40 条左腿肌肉。旧参数 `--f0_left` 根据 `_l` 后缀选择 145 条左躯干和左腿肌肉，**不包含左臂**；新实验建议使用分区参数。缩放的是最大等长肌力 \(F_0\)，实际肌肉力还取决于激活和力长等状态，\(F_0\) 比例也不能直接换算为临床肌力等级。

## 持杖行走基线

四组条件共用 CMU 136 参考、官方 `checkpoint_12500`（stochastic，冻结）与配对的种子 0/1/2，
所有肌群 \(F_0\) = 1.0，不训练。

**握持点**：官方 `left_hand_mimic` / `right_hand_mimic` 落在 `lunate`（腕骨），不是掌心。
所以 `scripts/crutch_env.py` 在第三掌骨 `thirdmc_l/r`（腕骨的刚性子刚体）上新增掌心 site
`hand_grip_l/r`，约束用 **site-to-site connect** 直接连 `crutch/grip_point`（球心）与掌心 site，
不再重算任何锚点。约束、标定、距离检查与渲染都用同一个点。

**杆长**：由闭合瞬间的掌心高度决定，\(L = z_{palm} - 0.30\ \mathrm{m}\)（0.30 = 离地 0.05 + 底盘 0.25），
并且是用真实回放协议测的——从 `start_step` reset、走 30 个控制步、再读掌心 site，
不是读起始帧、也不是整段平均。

两个几何变体：

| 变体 | start_step / 起握帧 | 掌心高 L / R | 杆长 | 侧向目标 L / R | 闭合前 Δz |
| --- | --- | --- | --- | --- | --- |
| `paired` | 0 / 0.30 s | 0.8062 / 0.8411 m | 0.5236 m | +0.201 / −0.368 m | −17.4 / +17.5 mm |
| `symmetric` | 220 / 2.50 s | 0.8450 / 0.8438 m | 0.5444 m | +0.287 / −0.272 m | **+0.18 / −1.35 mm** |

`symmetric` 的起始帧在 0–3.0 s 内以 0.2 s 步长逐帧实测选出（比较闭合时刻的两手高度差与侧向镜像误差）。

**协议**：所有条件先走 30 步（0.30 s）不记录的热身，再记录固定 500 / 1000 步；`duration_s` 由记录窗口
步数给出。走满与"走满且握持全程有效"分开统计。

| 变体 | 条件 | 5 s 走满+有效 | 10 s 走满+有效 | 10 s 速度 m/s | 最大侧偏 m | 倾角 ° | 闭合前 Δz mm | 启动峰值 N | 持续 \|Fz\| N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| symmetric | A 无拐杖 | 3/3 | 3/3 | 0.803 | 0.22 | 31.9 | — | — | — |
| symmetric | B 随行不握持 | 3/3 | 3/3 | 0.803 | 0.22 | 31.9 | — | 0 | 0 |
| symmetric | C 左手（患侧） | 3/3 | 3/3 | 0.740 | 1.25 | 31.8 | −5.5 | 322 | 95 |
| symmetric | D 右手（健侧） | 3/3 | 3/3 | 0.787 | **0.19** | 32.5 | −4.4 | 282 | 73 |
| paired | C 左手 | 3/3 | 3/3 | 0.835 | 0.66 | 32.0 | −14.5 | 340 | 109 |
| paired | D 右手 | 2/3 | **1/3** | 0.377 | 1.38 | 44.7 | +14.2 | 217 | 95 |

读法：

- **几何问题已解决**：对称变体闭合前 Δz 为 +0.18 / −1.35 mm，整段 10 s 的 \|Δz\| 均值 −0.15 mm、
  p05/p95 = −3.5/+3.6 mm、最大 5.2 mm，约束始终把掌心按在球心上，不再出现"手和球分开"；
- `paired` 变体按设计追随参考自身的不对称姿态（两侧掌心差 3.5 cm），用一根公共杆必然一侧 −17 mm、
  另一侧 +17 mm，右手因此丢掉 2 个 10 s 回合——这是几何后果，不能当侧别结论；
- **对称变体下左右都 3/3 走满**，差别只在侧向漂移（C 1.25 m vs D 0.19 m）。上一版报告的
  "左手更好"是握点错位造成的假象，方向已经翻转；
- **残余偏差来自逐种子差异**：公共杆只按 seed 0 标定，seed 1/2 起握时刻差 4–12 mm，闭合时吸收一次
  冲量，启动峰值从 188 N 升到 508 N。要去掉得按种子单独标定，或给约束加软启动；
- 底盘没有竖直自由度、球高锁死，手臂摆动高于球时被向下拉，Fz 在 ±300 N 周期变号，这是模型选择的
  结果而非数值问题。

逐回合数据在本地 `runs/crutch_baseline/results_all.csv`；带三色握点标记的侧视闭合检查在
`reports/video/crutch_closure_seed0.mp4`，Δz/Fz 曲线在 `reports/figures/closure_traces_seed0.png`；
三格并排与分镜在 `runs/crutch_baseline/`（均未提交）。

```bash
export PYTHONPATH="$PWD/external/MuscleMimic:$PWD/scripts"

# 标定（真实回放协议 + 网格选帧 + 公共杆长）
.venv/bin/python scripts/run_crutch_baseline.py --mode calibrate --seconds 10

# 两个变体 × 五条件 × 三配对种子
for sec in 5 10; do
  for v in paired symmetric; do
    for c in A B_left B_right C D; do
      .venv/bin/python scripts/run_crutch_baseline.py --mode run --condition $c \
        --seeds 0 1 2 --seconds $sec --variant $v
    done
  done
done

# 带握点标记的侧视闭合检查
.venv/bin/python scripts/run_crutch_baseline.py --mode run --condition C --seeds 0 \
  --seconds 10 --variant symmetric --video --markers --camera side \
  --out_dir runs/crutch_closure --calib runs/crutch_baseline/calibration.json

# 无拐杖 / 左握 / 右握并排
for c in A C D; do
  .venv/bin/python scripts/run_crutch_baseline.py --mode run --condition $c \
    --seeds 0 --seconds 10 --variant symmetric --video
done
.venv/bin/python scripts/run_crutch_baseline.py --mode compose --seeds 0
```

## 接下来的实验

几何与计时已经对齐：握点锚在掌心、杆长按闭合瞬间的掌心高度定、闭合前偏差在毫米级，整段行走的
垂直偏差也保持在 ±5 mm。按预定判据，这排除了"靠加持杖侧手部目标救几何"这一支。仍然存在的问题是：

1. **逐种子闭合偏差**：公共杆只按 seed 0 标定，其余种子在起握时刻差 4–12 mm，闭合瞬间的冲量把
   启动峰值从 188 N 抬到 508 N。要么按种子单独标定杆长，要么给约束加软启动（渐变 `solref`）。
2. **持续侧向漂移**：对称几何下左握 1.25 m、右握 0.19 m、无拐杖 0.22 m。左握的漂移是真实的代价，
   需要判断它是"球高固定 + 左臂摆动"的冲突，还是底盘随行增益的问题，再决定是否进入微调。
3. **训练侧接入**：策略目前不知道拐杖存在（观测与奖励都未改），所有结果都是冻结正常步态策略被动
   承受握持的后果。当前训练环境（MJX/warp）不支持带拐杖的耦合模型，训练侧接入需要改 MJX 的场景
   构建，是独立的一轮工作。

顺序建议：先做 1（代价小，直接减少启动冲量），再用更多种子确认 2 是否稳定，最后才考虑微调。

**不是损伤后适应步态，也不是临床偏瘫步态**；F0 缩放不等于临床肌力分级。若两侧辅助效果的差值随
上肢肌力改变符号，再估计转换区间及其不确定性，并用临床采集的 IMU、三维力传感器和上肢肌力资料验证。
当前仓库已能接入刚性拐杖、完成完整肌力基线与对称侧别对照，但尚未得到侧别转换点。
