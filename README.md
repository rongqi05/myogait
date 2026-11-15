# myogait

这个仓库记录 MyoFullBody 人体模型的行走回放、肌力干预和验收实验。毕业设计最终要比较拐杖机器人在健侧与患侧辅助时的效果，并观察这种差异是否随患侧上肢恢复程度发生改变。这里先把无拐杖时的正常肌力行走和单侧减力反应做实，供后续辅助实验使用。

目前使用 [MuscleMimic](https://github.com/amathislab/musclemimic) 的 `mm-10m-2` 预训练策略与 MyoFullBody 肌肉骨骼模型。策略读取当前状态和运动参考，输出肌肉控制量；MuJoCo 计算肌肉、接触与人体运动。每个回合只在开始时重置一次，之后不周期性覆盖根节点位置。参考轨迹仍是运动模仿任务的目标，实验结果会受参考的速度、长度和朝向影响。

## 目前做到哪里

| 实验 | 固定条件 | 结果 |
| --- | --- | --- |
| 连续正常肌力行走 | CMU 136 拼接参考；官方 checkpoint；20 个随机种子 | 10 s 和 20 s 均为 **20/20 走满、无跌倒**；20 s 全部未通过预设直线判据，中位最大侧偏 0.831 m |
| 患侧上肢单独减力 | CMU 120 单段参考；左臂 32 条肌肉的 \(F_0\) 乘 0.7；策略冻结 | 10 s **14/20 走满**（P 报告口径）；随强度扫描，侧向运动与足滑等指标有变化 |
| 患侧下肢单独减力 | 同一 CMU 120 参考；左腿 40 条肌肉的 \(F_0\) 乘 0.7；策略冻结 | **20/20 在约 3.1 s 跌倒** |

20 s 的“不跌倒”和“沿直线保持在阈值内”分别统计。当前 0.35 m 侧偏上限是本仓库预设的工程验收线，尚未用临床数据确定。上表的连续行走与减力实验采用不同参考，数值不能作为同一组配对比较。减力后尚未训练适应策略；跌倒结果反映冻结正常策略受到损伤参数扰动后的反应，不能直接称为偏瘫步态。

实验细节和逐种子证据：

- [R：连续步态基线](reports/R_continuous_gait_baseline.md)；[逐种子表](reports/R_v2_per_seed.csv)
- [P：上肢、下肢分区减力](reports/P_f0_upper_lower_ablation.md)；[Q：上肢强度扫描](reports/Q_upper_limb_feasible_range.md)
- [O：单侧肌力缩放](reports/O_muscle_force_scaling.md)；[验收口径](configs/eval_walk.json)
- [开题研究逻辑图](docs/proposal_logic.md)

## 代码和数据

| 路径 | 用途 |
| --- | --- |
| `scripts/_env_builder.py` | 加载上游 checkpoint、建立 MuJoCo 环境，按解剖分组缩放肌力 |
| `scripts/eval_walk.py` | 单回合、单次 reset 回放；记录结束原因、速度、侧偏、足滑及肌肉/状态量 |
| `scripts/check_f0_groups.py` | 检查肌肉分组和 \(F_0\) 缩放对象 |
| `scripts/run_f0_ablation.sh` | 顺序运行上肢、下肢和旧语义减力对照 |
| `scripts/train_straight_bounded.sh` | 8 GB 显存配置的训练试验入口；仓库目前采用的主要结果来自官方 checkpoint |

上游代码、checkpoint、重定向后的 `.npz` 与原始回放日志未随仓库提交。R 报告中的 CMU 136 连续参考由五段同一受试者的运动拼接而成（28.37 s）；接缝做过姿态插值，尚需检查接缝附近动力学。构建该参考的数据处理脚本已从当前分支移除，可在历史提交 `bc8a8fe` 查阅。若本地没有相应 `.npz`，下面的 20 s 命令无法直接运行。

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

## 接下来的实验

先在同一参考和同一策略设置下复查正常肌力与分区减力的配对差异，再训练或调整损伤后的控制策略，使不同肌力条件都具有可比较的步行能力。之后加入拐杖机器人的接触与力反馈，分别运行无辅助、健侧辅助和患侧辅助，用相同初始状态及种子比较跌倒率、步态稳定性、足滑、步速和拐杖受力。若两侧辅助效果的差值随上肢肌力改变符号，再估计转换区间及其不确定性，并用临床采集的 IMU、三维力传感器和上肢肌力资料验证。当前仓库尚无拐杖交互模型，也没有得到侧别转换点。
