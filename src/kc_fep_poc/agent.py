from __future__ import annotations

from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .cem_planner import plan


class Agent:
    """Simple agent wrapping a Haiku model and CEM planner."""

    def __init__(
        self,
        forward_fn: Callable[[jnp.ndarray], dict[str, jnp.ndarray]],
        obs_shape: tuple[int, ...],
        *,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self._net = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(seed)
        init_rng, self._rng = jax.random.split(rng)
        dummy = jnp.zeros((1,) + obs_shape)
        self.params = self._net.init(init_rng, dummy)
        self.obs_shape = obs_shape
        self.opt = optax.adam(lr)
        self.opt_state = self.opt.init(self.params)
        self._plan_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Planner interface ------------------------------------------------
    def rollout(self, obs: np.ndarray, actions: np.ndarray) -> float:
        """Return predicted free energy cost for ``actions`` from ``obs``."""
        # TODO: incorporate actions once a dynamics model is available
        # This toy rollout ignores actions and just evaluates the current obs.
        data = jnp.asarray(obs)
        if data.shape == self.obs_shape:
            data = data[None, ...]
        self._rng, apply_rng = jax.random.split(self._rng)
        outputs = self._net.apply(self.params, apply_rng, data)
        return float(outputs["kl"] + outputs["nll"])

    # ------------------------------------------------------------------
    def step(self, obs: np.ndarray) -> int:
        """Select an action for ``obs`` using the CEM planner."""
        return plan(obs, self, rng=self._plan_rng)

    # ------------------------------------------------------------------
    def train(self, batch: np.ndarray) -> float:
        """Update the model with one gradient step using ``batch``."""

        def loss_fn(params, rng, data):
            out = self._net.apply(params, rng, data)
            return out["kl"] + out["nll"]

        self._rng, apply_rng = jax.random.split(self._rng)
        loss, grads = jax.value_and_grad(loss_fn)(self.params, apply_rng, batch)
        updates, self.opt_state = self.opt.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return float(loss)
