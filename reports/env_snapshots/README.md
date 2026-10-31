# 环境快照（2026-10-31 清理前）

本目录保存删除 conda 环境之前的依赖清单，用于将来重建。

| 文件 | 来源 | 说明 |
|---|---|---|
| `musclemimic_venv_freeze.txt` | `~/Documents/musclemimic/.venv` | **当前仍在用**的冻结验收环境（122 个包）。这是权威快照。 |
| `hemirl_freeze_20261031.txt` | conda `hemirl`（已删除） | `muscle-rl`（MS-Human-700 + DynSyn-SAC）原环境，Python 3.12.13，170 个包。 |
| `hemi_msk_freeze_20261031.txt` | conda `hemi_msk`（已删除） | 旧 `mojoco` 项目环境，Python 3.11.15，157 个包。其可编辑安装指向已不存在的 `~/Documents/mojoco/src`。 |

## 重建说明

- `hemirl`：按 `~/Documents/muscle-rl/requirements-hemirl.txt` + 本快照重建；需要 Python **3.12**
  （官方 checkpoint 用 3.12 保存，cloudpickle 字节码在 3.11 上不兼容）。
- 安装时务必 `PIP_CONFIG_FILE=/dev/null`，否则会走 `pypi.nvidia.com` 并升级 torch。
- **禁止**向 `env_isaaclab` 安装本项目依赖（会升级 torch，破坏 isaacsim/torchvision/torchaudio）。

## 关键版本（musclemimic 验收环境）

```
jax==0.7.2           jaxlib==0.7.2        mujoco==3.4.0
numpy==2.2.6         flax==0.12.0         orbax-checkpoint==0.11.37
warp-lang==1.10.0    musclemimic-models==1.0.5
```

`musclemimic_models` **必须 1.0.5**：1.0.6 会把左膝等式约束符号翻转，导致 mm-10m-2 在 2.5–2.8 s 跌倒
（见 `../D_model_version_compatibility.md`）。
