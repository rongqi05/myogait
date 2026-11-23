# myogait

这个仓库记录 MyoFullBody 人体模型的行走回放、肌力干预和验收实验。毕业设计最终要比较拐杖机器人在健侧与患侧辅助时的效果，并观察这种差异是否随患侧上肢恢复程度发生改变。这里先把无拐杖时的正常肌力行走和单侧减力反应做实，供后续辅助实验使用。

目前使用 [MuscleMimic](https://github.com/amathislab/musclemimic) 的 `mm-10m-2` 预训练策略与 MyoFullBody 肌肉骨骼模型。策略读取当前状态和运动参考，输出肌肉控制量；MuJoCo 计算肌肉、接触与人体运动。每个回合只在开始时重置一次，之后不周期性覆盖根节点位置。参考轨迹仍是运动模仿任务的目标，实验结果会受参考的速度、长度和朝向影响。

## 目前做到哪里

| 实验 | 固定条件 | 结果 |
| --- | --- | --- |
| 连续正常肌力行走 | CMU 136 拼接参考；官方 checkpoint；20 个随机种子 | 10 s 和 20 s 均为 **20/20 走满、无跌倒**；20 s 全部未通过预设直线判据，中位最大侧偏 0.831 m |
| 患侧上肢单独减力 | CMU 120 单段参考；左臂 32 条肌肉的 \(F_0\) 乘 0.7；策略冻结 | 10 s **14/20 走满**（P 报告口径）；随强度扫描，侧向运动与足滑等指标有变化 |
| 患侧下肢单独减力 | 同一 CMU 120 参考；左腿 40 条肌肉的 \(F_0\) 乘 0.7；策略冻结 | **20/20 在约 3.1 s 跌倒** |
| 完整肌力持杖行走基线 | CMU 136 参考；F0 全为 1.0；拐杖机器人接入同一 MuJoCo 场景；3 个配对种子 | 无拐杖 / 随行不握持 / 左手握持 / 右手握持，**5 s 与 10 s 各 15/15 走满、无跌倒**；握持使速度下降 10 %（左）～37 %（右），竖直握持力在 ±400 N 量级周期变化 |

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

四组条件共用 CMU 136 参考、`--start_step 0`、官方 `checkpoint_12500`（stochastic，冻结）与配对的
种子 0/1/2。所有肌群 \(F_0\) = 1.0，不训练。

几何由参考姿态实测确定，不沿用演示值：起始帧手到肩 0.531 m / 最大臂展 0.549 m（reach 0.968），
肘屈曲 29.3°（左）/ 26.6°（右），属于可承重的持杖姿态；握柄球心取起握时刻两手平均手高
**0.8691 m**，对应支撑杆 **0.5691 m**。底盘侧向目标取各手在起握帧相对骨盆的偏移
（左 0.2284 m，右 0.3829 m）。

| 条件 | 5 s 走满 | 10 s 走满 | 10 s 速度 m/s | 最大侧偏 m | 躯干倾角 ° | \|Fz\| 均值 N | 首步 Fz N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A 无拐杖 | 3/3 | 3/3 | 0.778 | 0.33 | 32.3 | — | — |
| B 随行不握持（左/右） | 3/3 | 3/3 | 0.757 | 0.32 | 32.3 | 0.0 | 0.0 |
| C 左手握持 | 3/3 | 3/3 | 0.696 | 1.10 | 31.9 | 131.6 | 155.7 |
| D 右手握持 | 3/3 | 3/3 | 0.487 | 1.66 | 33.9 | 97.2 | 237.8 |

读法：

- 纯随行几乎零代价（速度 −3 %，握持力恒 0），说明场景集成与动作映射没问题；
- 握持是唯一有明显代价的因素，且代价主要在**侧向漂移**（0.33 → 1.1/1.7 m），矢状面倾角仍与参考一致；
- 底盘没有竖直自由度，握柄高度锁死，手臂在摆动中高于握柄时被向下拉，所以 Fz 在 ±400 N 量级周期
  变化、均值可正可负，这是物理结果而非数值不稳；激活瞬间仍有最高 451 N 的一次冲量，
  因为 `set_assistance` 只检查握点位置、未检查相对速度；
- **C 与 D 的差异不能读作"健侧优势"**：参考运动左右臂不对称（右臂平均手高 0.923 m、侧偏 0.383 m，
  左臂 0.888 m、0.228 m），D 的竖直预载更大。要回答侧别问题必须用同一侧偏、同一握柄高度、
  对称初始臂姿重跑。

逐回合数据在本地 `runs/crutch_baseline/results_all.csv`，逐种子 JSON/NPZ 与
`baseline_comparison_seed0.mp4`、`baseline_keyframes_seed0.png` 在 `runs/crutch_baseline/`（未提交）。

```bash
export PYTHONPATH="$PWD/external/MuscleMimic:$PWD/scripts"

# 标定握点高度与左右偏移
.venv/bin/python scripts/run_crutch_baseline.py --mode calibrate --seconds 10

# 四组条件，各 3 个配对种子
for c in A B_left B_right C D; do
  .venv/bin/python scripts/run_crutch_baseline.py --mode run --condition $c --seeds 0 1 2 --seconds 10
done

# 同一 seed 的无拐杖 / 左握 / 右握并排视频
for c in A C D; do
  .venv/bin/python scripts/run_crutch_baseline.py --mode run --condition $c --seeds 0 --seconds 10 --video
done
.venv/bin/python scripts/run_crutch_baseline.py --mode compose --seeds 0
```

## 接下来的实验

先解决侧别比较的前提：让 C 与 D 使用同一侧向偏置、同一握柄高度与对称初始臂姿，否则两组的
竖直预载与拖拽基数不同，无法归因于左右侧。然后在 `set_assistance` 中加入激活前的相对速度门限，
消除首步冲量。

再分别运行无辅助、健侧辅助和患侧辅助，用相同初始状态及种子比较跌倒率、步态稳定性、足滑、
步速和拐杖受力。策略目前不知道拐杖存在（观测与奖励都未改），所有结果都是冻结正常步态策略
被动承受握持的后果，**不是损伤后适应步态，也不是临床偏瘫步态**。若两侧辅助效果的差值随上肢
肌力改变符号，再估计转换区间及其不确定性，并用临床采集的 IMU、三维力传感器和上肢肌力资料验证。
当前仓库已能接入刚性拐杖并完成完整肌力基线，但尚未得到侧别转换点。
