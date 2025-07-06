import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import haiku as hk  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from kc_fep_poc.world_model import WorldModel  # noqa: E402


def test_world_model_losses_finite():
    rng = jax.random.PRNGKey(0)
    x = np.random.rand(2, 16, 16, 1).astype(np.float32)

    def forward(data):
        model = WorldModel()
        return model(data)

    net = hk.transform(forward)
    init_rng, apply_rng = jax.random.split(rng)
    params = net.init(init_rng, x)
    outputs = net.apply(params, apply_rng, x)

    assert np.isfinite(outputs["kl"])  # KL should be finite
    assert np.isfinite(outputs["nll"])  # NLL should be finite
    assert outputs["recon"].shape == x.shape
