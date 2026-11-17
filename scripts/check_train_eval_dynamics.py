#!/usr/bin/env python3
"""Check that the *training* (MJX) model and the *evaluation* (MuJoCo CPU) model
integrate the same F0-scaled dynamics.

The adaptation run optimises against ``MjxMyoFullBody.sys`` (a ``mjx.put_model``
of the scaled MuJoCo model) while acceptance replays against a plain
``MyoFullBody`` model. If those two disagree on the impaired dynamics, no
amount of tuning fixes the gap.

Both models are driven open loop from the same reference state with the same
constant activation, and their qpos trajectories are compared.

Usage:
    PYTHONPATH=external/MuscleMimic:scripts MUJOCO_GL=egl \
      .venv-train/bin/python scripts/check_train_eval_dynamics.py --f0-left-lower 0.90
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "external" / "MuscleMimic"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import _train_f0  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f0-left-lower", type=float, default=0.90)
    ap.add_argument("--reference", default="KIT/3/straight_stitched_10s")
    ap.add_argument("--steps", type=int, default=50, help="control steps to integrate")
    ap.add_argument("--activation", type=float, default=0.3, help="constant muscle activation")
    ap.add_argument("--out", default="runs/dynamics_check.json")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    # ---- training model: build through the public factory, F0 patched in ----
    _train_f0.F0_LEFT_LOWER = args.f0_left_lower
    _train_f0.patch()

    from loco_mujoco.task_factories import TaskFactory
    from musclemimic.environments.humanoids.myofullbody import MyoFullBody

    factory = TaskFactory.get_factory_cls("ImitationFactory")
    train_env = factory.make(
        env_name="MjxMyoFullBody",
        headless=True,
        disable_fingers=True,
        num_envs=1,
        amass_dataset_conf=dict(
            rel_dataset_path=[args.reference],
            dataset_group=None,
            retargeting_method="gmr",
        ),
    )
    m_train = train_env._model
    sys_train = train_env.sys

    # ---- evaluation model: plain MuJoCo CPU, same F0 via _env_builder ------
    from _env_builder import scale_muscle_groups

    eval_env = MyoFullBody(disable_fingers=True, headless=True)
    m_eval = eval_env._model
    if args.f0_left_lower != 1.0:
        scale_muscle_groups(m_eval, left_lower=args.f0_left_lower)

    # ---- identical initial state, taken from the reference trajectory ------
    from _env_builder import build_eval_bundle, env_data, env_model

    bundle = build_eval_bundle(
        checkpoint_path="artifacts/checkpoints/checkpoint_12500",
        motion_path=args.reference,
        start_step=0,
        traj_index=0,
        strict_termination=False,
        horizon=None,
    )
    ref_model = env_model(bundle.env)
    ref_data = env_data(bundle.env)
    bundle.env.reset()
    qpos0 = np.asarray(ref_data.qpos, dtype=np.float64).copy()
    qvel0 = np.asarray(ref_data.qvel, dtype=np.float64).copy()

    # ---- gains must agree ------------------------------------------------
    act_names = [mujoco.mj_id2name(m_train, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m_train.nu)]
    from _env_builder import muscle_groups

    idx = np.asarray(muscle_groups(m_train)["left_lower"]["indices"], dtype=int)
    g_train = np.asarray(m_train.actuator_gainprm)[idx, 0]
    g_eval = np.asarray(m_eval.actuator_gainprm)[idx, 0]
    g_sys = np.asarray(sys_train.actuator_gainprm)[idx, 0]
    gain_dev = float(np.max(np.abs(g_train - g_eval)))
    sys_dev = float(np.max(np.abs(g_sys - g_eval)))

    # ---- open-loop integration ------------------------------------------
    d_mj = mujoco.MjData(m_eval)
    d_mj.qpos[:] = qpos0
    d_mj.qvel[:] = qvel0
    # Muscle actuators: ctrl drives the activation state, so drive with ctrl and
    # let BOTH integrators evolve `act` themselves (holding act constant on one
    # side only would manufacture a difference that is not a model difference).
    d_mj.act[:] = 0.0
    d_mj.ctrl[:] = args.activation
    mujoco.mj_forward(m_eval, d_mj)

    d_mjx = mjx.put_data(m_train, d_mj)
    sys_j = sys_train
    d_mjx = d_mjx.replace(
        act=jnp.zeros_like(d_mjx.act),
        ctrl=jnp.full_like(d_mjx.ctrl, args.activation),
    )

    n_sub = int(round(float(train_env.dt) / float(m_train.opt.timestep)))

    @jax.jit
    def step_mjx(sys_, d_):
        for _ in range(n_sub):  # n_sub is a Python int -> unrolled at trace time
            d_ = mjx.step(sys_, d_)
        return d_

    traj_mj, traj_mjx = [], []
    for k in range(args.steps):
        for _ in range(n_sub):
            mujoco.mj_step(m_eval, d_mj)
        traj_mj.append(np.asarray(d_mj.qpos).copy())
        d_mjx = step_mjx(sys_j, d_mjx)
        traj_mjx.append(np.asarray(d_mjx.qpos).copy())

    traj_mj = np.asarray(traj_mj)
    traj_mjx = np.asarray(traj_mjx)
    dq = np.abs(traj_mj - traj_mjx)
    per_step = [float(v) for v in dq.max(axis=1)]
    horizon = {}
    for k in (1, 2, 3, 5, 10, 20, 50):
        if k <= len(per_step):
            horizon[f"step_{k}"] = per_step[k - 1]
    def _cmp(name, a, b):
        import numpy as _np
        a = _np.asarray(a, dtype=float).ravel()
        b = _np.asarray(b, dtype=float).ravel()
        if a.shape != b.shape:
            return {f"{name}_shape_mismatch": [list(a.shape), list(b.shape)]}
        return {f"{name}_max_abs_diff": float(_np.abs(a - b).max()) if a.size else 0.0}

    consts = {}
    consts.update(_cmp("body_mass", m_train.body_mass, m_eval.body_mass))
    consts.update(_cmp("dof_damping", m_train.dof_damping, m_eval.dof_damping))
    consts.update(_cmp("dof_armature", m_train.dof_armature, m_eval.dof_armature))
    consts.update(_cmp("actuator_gainprm", m_train.actuator_gainprm, m_eval.actuator_gainprm))
    consts.update(_cmp("actuator_biasprm", m_train.actuator_biasprm, m_eval.actuator_biasprm))
    consts.update(_cmp("actuator_dynprm", m_train.actuator_dynprm, m_eval.actuator_dynprm))
    consts.update(_cmp("actuator_forcerange", m_train.actuator_forcerange, m_eval.actuator_forcerange))
    consts.update(_cmp("eq_data", m_train.eq_data, m_eval.eq_data))
    consts.update(_cmp("geom_friction", m_train.geom_friction, m_eval.geom_friction))
    consts.update(_cmp("jnt_stiffness", m_train.jnt_stiffness, m_eval.jnt_stiffness))
    consts["timestep_train"] = float(m_train.opt.timestep)
    consts["timestep_eval"] = float(m_eval.opt.timestep)
    consts["integrator_train"] = int(m_train.opt.integrator)
    consts["integrator_eval"] = int(m_eval.opt.integrator)
    consts["solver_iterations_train"] = int(m_train.opt.iterations)
    consts["solver_iterations_eval"] = int(m_eval.opt.iterations)
    consts["neq_train"] = int(m_train.neq)
    consts["neq_eval"] = int(m_eval.neq)

    res = {
        "f0_left_lower": args.f0_left_lower,
        "model_constants": consts,
        "divergence_by_step_m": horizon,
        "reference": args.reference,
        "activation": args.activation,
        "steps": args.steps,
        "substeps_per_control": n_sub,
        "control_dt_train": float(train_env.dt),
        "control_dt_eval": float(m_eval.opt.timestep * 5),
        "nq_nv_nu_train": [int(m_train.nq), int(m_train.nv), int(m_train.nu)],
        "nq_nv_nu_eval": [int(m_eval.nq), int(m_eval.nv), int(m_eval.nu)],
        "left_lower_actuators": int(len(idx)),
        "gain_max_abs_diff_train_vs_eval": gain_dev,
        "gain_max_abs_diff_sys_vs_eval": sys_dev,
        "qpos_max_abs_diff_m": float(dq.max()),
        "qpos_final_abs_diff_m": float(np.abs(traj_mj[-1] - traj_mjx[-1]).max()),
        "root_xy_final_diff_m": float(np.linalg.norm(traj_mj[-1][:2] - traj_mjx[-1][:2])),
        "root_z_final_diff_m": float(abs(traj_mj[-1][2] - traj_mjx[-1][2])),
        "nan_train": bool(not np.all(np.isfinite(traj_mjx))),
        "nan_eval": bool(not np.all(np.isfinite(traj_mj))),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
