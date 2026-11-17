#!/usr/bin/env python3
"""Adaptation-training entry: fine-tune the official mm-10m-2 policy under a
left-lower-limb F0 deficit, from an 8 GB laptop GPU.

This is the *only* training entry for the impaired-adaptation experiment. It:

  1. applies left-lower-limb F0 scaling to the MJX training model
     (``scripts/_train_f0.py``) BEFORE the model is compiled for MJX;
  2. builds the hydra config with the exact architecture the checkpoint was
     trained with (5x1024, gated residual, LayerNorm, muon), so the weights
     load instead of hitting ScopeParamNotFoundError;
  3. resumes from ``experiment.resume_from`` into a fresh, unique checkpoint
     directory (``auto_resume=false``) so the official weights are never
     overwritten;
  4. keeps validation and checkpoint saving ON, and propagates training
     failures (no ``|| true``).

Usage:

    export PYTHONPATH=external/MuscleMimic:scripts HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl
    .venv-train/bin/python scripts/train_adapt.py \
        --f0-left-lower 0.9 --n-env 128 --updates 20 --steps 20 \
        --reference KIT/3/straight_stitched_10s --out-dir runs/T_adapt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "external" / "MuscleMimic"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import _train_f0  # noqa: E402

# The upstream LR schedule is linear in the *optimizer step counter*:
#   frac = 1 - (step // (num_minibatches * update_epochs)) / num_updates
# Resuming restores `step` from the checkpoint (~1.6e6) while `num_updates`
# is only the fine-tuning budget (20/100), so frac goes sharply negative and
# the run trains with a NEGATIVE learning rate. That is what destroyed the
# first adaptation round; `reset_lr_schedule_on_resume=true` zeroes the
# counter (optimizer momentum is kept) and restores a positive LR.
LR_RESET_ON_RESUME = True


def resume_step(checkpoint_path: str) -> int:
    """Optimizer step stored in the checkpoint (0 if it cannot be read)."""
    from musclemimic.runner.checkpointing import _canonicalize_resume_path

    try:
        resolved = Path(_canonicalize_resume_path(checkpoint_path))
        meta = resolved / "metadata" / "metadata"
        if meta.exists():
            import json

            return int(json.loads(meta.read_text()).get("step", 0) or 0)
    except Exception:
        pass
    return 0


def effective_start_lr(
    *,
    resumed_step: int,
    num_minibatches: int,
    update_epochs: int,
    base_lr: float,
    num_updates: int,
    reset_lr_schedule: bool,
) -> float:
    """The LR the first PPO update will actually use (mirrors the upstream call)."""
    from musclemimic.algorithms.common.optimizer import linear_lr_schedule

    step = 0 if reset_lr_schedule else resumed_step
    return float(linear_lr_schedule(step, num_minibatches, update_epochs, base_lr, num_updates))


def _jsonable(v):
    """Best-effort conversion of a logged value to plain JSON."""
    try:
        import numpy as np

        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.ndarray,)):
            return v.tolist()
    except Exception:
        pass
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


class _MetricsJSONLHooks:
    """Wraps UnifiedHooks and appends every log_dict to a JSONL file.

    With ``wandb.mode=disabled`` the upstream callback drops all per-update
    metrics (lr / KL / clip fraction / policy_std). Writing them to JSONL keeps
    them inspectable without enabling wandb.
    """

    @staticmethod
    def build(metrics_path: Path):
        from musclemimic.runner.logging import UnifiedHooks as _UH

        class _Hooks(_UH):
            def enrich_log(self, log_dict, metrics_dict, env):
                try:
                    payload = {
                        k: _jsonable(v) for k, v in log_dict.items() if not k.startswith("_")
                    }
                    with metrics_path.open("a") as fh:
                        fh.write(__import__("json").dumps(payload) + "\n")
                except Exception as exc:  # never break training for logging
                    print(f"warning: metrics jsonl write failed: {exc}")
                return super().enrich_log(log_dict, metrics_dict, env)

        return _Hooks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f0-left-lower", type=float, default=1.0, help="left-leg F0 multiplier for training")
    ap.add_argument("--n-env", type=int, default=128)
    ap.add_argument("--updates", type=int, default=20, help="number of PPO updates (smoke <= 20)")
    ap.add_argument("--steps", type=int, default=20, help="PPO rollout steps per update")
    ap.add_argument("--checkpoint", default="artifacts/checkpoints/checkpoint_12500")
    ap.add_argument("--reference", default="KIT/3/straight_stitched_10s")
    ap.add_argument("--out-dir", default="runs/T_adapt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-num", type=int, default=1, help="number of validations")
    ap.add_argument("--val-video-length", type=int, default=500, help="validation video length in steps")
    ap.add_argument(
        "--no-reset-lr-schedule",
        dest="reset_lr_schedule",
        action="store_false",
        help="keep the checkpoint's optimizer-step counter (yields a negative LR for short budgets)",
    )
    ap.add_argument(
        "--check-lr-only",
        action="store_true",
        help="print the effective first-update learning rate and exit (pre-flight check)",
    )
    ap.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="save a checkpoint every N updates (0 = only on validation / at the end)",
    )
    ap.add_argument(
        "--metrics-jsonl",
        default=None,
        help="per-update metrics JSONL (default: <out-dir>/metrics.jsonl)",
    )
    args = ap.parse_args()

    # --- pre-flight: the resumed run must start with a positive LR ----------
    num_minibatches = 128
    update_epochs = 1
    base_lr = 4e-4
    step0 = resume_step(args.checkpoint)
    lr0 = effective_start_lr(
        resumed_step=step0,
        num_minibatches=num_minibatches,
        update_epochs=update_epochs,
        base_lr=base_lr,
        num_updates=args.updates,
        reset_lr_schedule=args.reset_lr_schedule,
    )
    print(
        f"[train_adapt] resume step={step0} updates={args.updates} "
        f"reset_lr_schedule={args.reset_lr_schedule} -> first-update LR={lr0:+.6f}",
        flush=True,
    )
    if lr0 <= 0.0:
        raise SystemExit(
            "REFUSING TO TRAIN: the first PPO update would use a non-positive learning rate "
            f"({lr0:+.6f}). The checkpoint's optimizer step ({step0}) exceeds the fine-tuning "
            "budget (num_updates="f"{args.updates}), so the linear LR schedule goes negative. "
            "Run with the default --reset-lr-schedule (step counter zeroed, momentum kept)."
        )
    if args.check_lr_only:
        return 0

    # --- apply F0 scaling to the MJX model before it is compiled -------------
    _train_f0.F0_LEFT_LOWER = args.f0_left_lower
    _train_f0.patch()

    n_env = args.n_env
    updates = args.updates
    steps = args.steps
    total = n_env * steps * updates

    out_dir = Path(args.out_dir).expanduser()
    hydra_dir = out_dir / "hydra"
    ckpt_root = (out_dir / "checkpoints").as_posix()
    metrics_path = Path(args.metrics_jsonl).expanduser() if args.metrics_jsonl else (out_dir / "metrics.jsonl")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("")  # fresh file per run

    # mm-10m-2 architecture (from the checkpoint's own metadata): 5x1024,
    # silu, LayerNorm, gated residual, muon. The stock conf_fullbody.yaml
    # defaults to a 16-layer network WITHOUT residual, so these overrides are
    # what make the official weights load at all.
    overrides = [
        "--config-name=conf_fullbody",
        "wandb.mode=disabled",
        f"experiment.env_params.num_envs={n_env}",
        f"experiment.num_envs={n_env}",
        "experiment.env_params.horizon=1000",
        "experiment.actor_hidden_layers=[1024,1024,1024,1024,1024]",
        "experiment.critic_hidden_layers=[1024,1024,1024,1024,1024]",
        "experiment.use_layernorm=true",
        "+experiment.use_residual=true",
        "+experiment.residual_type=gated",
        "+experiment.residual_gate_init=-2.0",
        "experiment.use_moe=false",
        "experiment.optimizer_type=muon",
        f"experiment.ppo_config.num_steps={steps}",
        "experiment.ppo_config.num_minibatches=128",
        f"experiment.total_timesteps={total}",
        f"experiment.seeds=[{args.seed}]",
        f"experiment.resume_from={args.checkpoint}",
        "experiment.auto_resume=false",
        f"experiment.reset_lr_schedule_on_resume={'true' if args.reset_lr_schedule else 'false'}",
        f"experiment.checkpoint_root={ckpt_root}",
        # training reference (single local clip; no HF dataset-group download)
        f"+experiment.task_factory.params.amass_dataset_conf.rel_dataset_path=[{args.reference}]",
        "experiment.task_factory.params.amass_dataset_conf.dataset_group=null",
        "+experiment.task_factory.params.amass_dataset_conf.retargeting_method=gmr",
        # validation ON, pointing at the same local reference (no testing-group download)
        "experiment.validation.active=true",
        f"experiment.validation.num={args.val_num}",
        "experiment.validation.num_envs=32",
        "experiment.validation.num_steps=200",
        f"experiment.validation.video_length={args.val_video_length}",
        f"+experiment.validation.amass_dataset_conf.rel_dataset_path=[{args.reference}]",
        "experiment.validation.amass_dataset_conf.dataset_group=null",
        "+experiment.validation.amass_dataset_conf.retargeting_method=gmr",
        f"+experiment.checkpoint_interval={args.checkpoint_interval}",
        f"hydra.run.dir={hydra_dir}",
    ]

    # Route the upstream UnifiedHooks through a JSONL sink so per-update
    # lr / KL / clip fraction / policy_std survive wandb being disabled.
    import musclemimic.runner.logging as _logging

    _logging.UnifiedHooks = _MetricsJSONLHooks.build(metrics_path)
    print(f"[train_adapt] metrics -> {metrics_path}", flush=True)

    print("[train_adapt] overrides:", flush=True)
    for o in overrides:
        print(f"  {o}", flush=True)
    print(
        f"[train_adapt] f0_left_lower={args.f0_left_lower} n_env={n_env} "
        f"updates={updates} steps={steps} total={total}", flush=True
    )

    # Run the upstream experiment entry via runpy so hydra resolves its
    # config_path relative to fullbody/experiment.py (importing the decorated
    # function from here makes hydra look for a 'fullbody' package instead).
    import runpy  # noqa: E402

    sys.argv = ["experiment.py"] + overrides
    runpy.run_path(
        str(REPO_ROOT / "external" / "MuscleMimic" / "fullbody" / "experiment.py"),
        run_name="__main__",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
