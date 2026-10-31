#!/usr/bin/env bash
# Create .venv-gmr: the retargeting-only toolchain (GMR-Fit + SMPL-H parsing).
#
# Kept separate from .venv (frozen acceptance) on purpose:
#   .venv       -> jax CPU, musclemimic_models 1.0.5, used by scripts/eval_walk.py
#   .venv-train -> jax CUDA, used for PPO training
#   .venv-gmr   -> torch CPU + smplx + general-motion-retargeting, used for retargeting
#
# Retargeting is a pure IK problem (no MuJoCo stepping, no JAX), so CPU torch is
# enough and keeps the 8 GB of VRAM free for training.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${REPO_ROOT}/external/MuscleMimic"
VENV="${REPO_ROOT}/.venv-gmr"

if [ ! -d "${UPSTREAM}" ]; then
    echo "error: ${UPSTREAM} not found; clone MuscleMimic there first" >&2
    exit 1
fi

echo "==> syncing ${VENV} (extras: smpl, gmr)"
cd "${UPSTREAM}"
UV_PROJECT_ENVIRONMENT="${VENV}" uv sync --extra smpl --extra gmr --python 3.11

# `uv sync` resolves musclemimic_models from the `>=1.0.2` loose pin and picks the
# newest release. 1.0.6 flips the sign of both left-knee equality-constraint
# polycoef coefficients (reports/D_model_version_compatibility.md), so the
# retargeting model would stop being the same MyoFullBody variant that the frozen
# acceptance env uses. Re-pin after every sync.
echo "==> pinning musclemimic_models==1.0.5"
uv pip install --python "${VENV}/bin/python" -r "${REPO_ROOT}/configs/constraints.txt"

echo "==> verifying toolchain"
"${VENV}/bin/python" "${REPO_ROOT}/scripts/check_gmr_toolchain.py" --json-out "${REPO_ROOT}/reports/gmr_toolchain.json"
