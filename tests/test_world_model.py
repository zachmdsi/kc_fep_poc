import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import haiku as hk  # noqa: E402

from kc_fep_poc.world_model import WorldModel  # noqa: E402


def test_world_model_losses_finite():
    rng = jax.random.PRNGKey(0)
    batch = jnp.zeros((2, 16, 16, 1))

    def forward(x):
        model = WorldModel()
        return model(x)

    net = hk.transform(forward)
    params = net.init(rng, batch)
    outputs = net.apply(params, rng, batch)

    assert np.isfinite(float(outputs["kl"]))
    assert np.isfinite(float(outputs["nll"]))
    assert outputs["recon"].shape == batch.shape
