import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

hk = pytest.importorskip("haiku")  # noqa: E402
jax = pytest.importorskip("jax")  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from kc_fep_poc.agent import Agent  # noqa: E402


def make_agent() -> Agent:
    def forward(x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        y = hk.Linear(1)(x)
        nll = jnp.mean((y - 1.0) ** 2)
        return {"kl": jnp.array(0.0), "nll": nll}

    return Agent(forward, obs_shape=(1,), lr=0.1, seed=0)


def test_step_returns_binary_action() -> None:
    agent = make_agent()
    obs = np.zeros((1,), dtype=np.float32)
    action = agent.step(obs)
    assert isinstance(action, int)
    assert action in (0, 1)


def test_train_decreases_loss() -> None:
    agent = make_agent()
    batch = np.zeros((4, 1), dtype=np.float32)
    loss1 = agent.train(batch)
    loss2 = agent.train(batch)
    assert np.isfinite(loss1) and np.isfinite(loss2)
    assert loss2 <= loss1 + 1e-5
