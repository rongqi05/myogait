"""Apply left-lower-limb F0 scaling to the *training* (MJX) environment.

``scripts/_env_builder.py`` scales the MuJoCo ``actuator_gainprm`` of an
already-built evaluation model. The training environment is an ``Mjx`` env, so
its MuJoCo model (``self._model``) is compiled into ``self.sys`` via
``mjx.put_model`` *during* ``Mjx.__init__``. Scaling must therefore happen
before that compilation, otherwise the network would train against the
unscaled dynamics while the evaluation model used the scaled ones.

This module monkeypatches ``musclemimic.core.mujoco_mjx.Mjx.__init__`` to scale
``left_lower`` right after the MuJoCo model exists and then rebuild ``self.sys``
with the scaled gains. It is imported only by the training entry
(``scripts/train_adapt.py``); the evaluation entry keeps its own scaling path.

Usage (module-global knob, set before any env is built):

    import _train_f0
    _train_f0.F0_LEFT_LOWER = 0.9
    _train_f0.patch()
"""

from __future__ import annotations

F0_LEFT_LOWER: float = 1.0
_PATCHED = False


def patch() -> None:
    """Patch ``Mjx.__init__`` to scale left-lower F0 before ``mjx.put_model``."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    import mujoco
    from mujoco import mjx

    from musclemimic.core.mujoco_mjx import Mjx

    orig_init = Mjx.__init__

    def _patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        f0 = F0_LEFT_LOWER
        if f0 != 1.0:
            from _env_builder import scale_muscle_groups

            info = scale_muscle_groups(self._model, left_lower=f0)
            # Recompile the MJX model with the scaled gains. The MuJoCo model
            # size/shape is unchanged, so the data budgets stored by the
            # original __init__ stay valid; resets rebuild data from _model.
            mujoco.mj_resetData(self._model, self._data)
            mujoco.mj_forward(self._model, self._data)
            self.sys = mjx.put_model(self._model, impl=self._backend_impl)
            import sys

            print(
                f"[train_f0] left_lower F0 scaled by {f0}: "
                f"{info['groups_scaled']['left_lower']['count']} actuators "
                f"(first F0 {info['groups_scaled']['left_lower']['f0_original'][0]:.4f} -> "
                f"{info['groups_scaled']['left_lower']['f0_scaled'][0]:.4f})",
                file=sys.stderr,
                flush=True,
            )

    Mjx.__init__ = _patched_init
