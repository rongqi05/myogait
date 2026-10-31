# 环境快照（2026-10-31 清理前）

本目录保存删除 conda 环境之前的依赖清单，用于将来重建。

| 文件 | 来源 | 说明 |
|---|---|---|
| `musclemimic_venv_freeze.txt` | `~/Documents/musclemimic/.venv` | **当前唯一在用**的环境（122 个包，自包含，Python 3.11.15）。权威快照。 |
| `env_isaaclab_freeze_20261031.txt` | conda `env_isaaclab`（已删除） | Isaac Lab / Isaac Sim 环境，409 个包。含 `isaacsim==5.1.0.0`、`isaaclab==0.54.2`、`torch==2.7.0+cu128`、`numpy==1.26.0`、`mujoco==3.11.0`、`warp-lang==1.16.0`。 |
| `hemirl_freeze_20261031.txt` | conda `hemirl`（已删除） | `muscle-rl`（MS-Human-700 + DynSyn-SAC）原环境，Python 3.12.13，170 个包。 |
| `hemi_msk_freeze_20261031.txt` | conda `hemi_msk`（已删除） | 旧 `mojoco` 项目环境，Python 3.11.15，157 个包。其可编辑安装指向已不存在的 `~/Documents/mojoco/src`。 |

## 当前状态

- conda 环境只剩 `base`（Python 3.14.6，miniconda 自身）。
- `~/Documents/musclemimic/.venv` 为自包含环境，不需要任何 conda 环境即可运行。

## 重建 Isaac Lab（`env_isaaclab`）

源码仍在 `~/Documents/muscle/IsaacLab`（`isaaclab.sh` 存在）。

```bash
# 1) 按 Isaac Lab 官方要求建 conda 环境（Python 3.11）
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab

# 2) 用 Isaac Lab 自带安装脚本（会装 isaacsim 与全部扩展，下载量大，预留 ≥30 GB）
cd ~/Documents/muscle/IsaacLab
./isaaclab.sh --install

# 3) 与本机配对的版本目标（见 env_isaaclab_freeze_20261031.txt）
#    isaacsim==5.1.0.0  isaaclab==0.54.2  torch==2.7.0+cu128
#    numpy==1.26.0      mujoco==3.11.0     warp-lang==1.16.0
```

注意：
- `env_isaaclab` 里的 `numpy==1.26.0` 与本仓库 `.venv` 的 `numpy==2.2.6` **不能混用**；
  两边是两个独立环境，不要互相 `pip install`。
- 该环境曾因误装本项目依赖把 torch 从 2.7.0 升到 2.14.0，导致 isaacsim/torchvision/torchaudio 损坏。
- 现在本机 GPU 可用（RTX 5060 Laptop, sm_120, 8 GB），Isaac Sim 5.1 对 Blackwell 需要较新驱动；
  本机驱动 595.99.02 / CUDA 13.2 已满足。

## 重建 `hemirl`（muscle-rl）

- 按 `~/Documents/muscle-rl/requirements-hemirl.txt` + 本快照重建；**必须 Python 3.12**
  （官方 checkpoint 用 3.12 保存，cloudpickle 字节码在 3.11 上不兼容）。
- 安装时务必 `PIP_CONFIG_FILE=/dev/null`，否则会走 `pypi.nvidia.com` 并升级 torch。

## 关键版本（musclemimic 验收环境）

```
jax==0.7.2           jaxlib==0.7.2        mujoco==3.4.0
numpy==2.2.6         flax==0.12.0         orbax-checkpoint==0.11.37
warp-lang==1.10.0    musclemimic-models==1.0.5
```

`musclemimic_models` **必须 1.0.5**：1.0.6 会把左膝等式约束符号翻转，导致 mm-10m-2 在 2.5–2.8 s 跌倒
（见 `../D_model_version_compatibility.md`）。
