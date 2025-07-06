from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp


class WorldModel(hk.Module):
    """Simple convolutional VAE used as world model."""

    def __init__(self, latent_dim: int = 32, name: str | None = None):
        super().__init__(name=name)
        self.latent_dim = latent_dim

    def _encode(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        h = hk.Conv2D(16, kernel_shape=3, stride=2, padding="SAME")(x)
        h = jax.nn.relu(h)
        h = hk.Conv2D(32, kernel_shape=3, stride=2, padding="SAME")(h)
        h = jax.nn.relu(h)
        h = hk.Flatten()(h)
        params = hk.Linear(2 * self.latent_dim)(h)
        mean, logvar = jnp.split(params, 2, axis=-1)
        return mean, logvar

    def _decode(self, z: jnp.ndarray, output_shape: tuple[int, int, int]) -> jnp.ndarray:
        h, w, c = output_shape
        h_enc, w_enc = h // 4, w // 4
        x = hk.Linear(h_enc * w_enc * 32)(z)
        x = jax.nn.relu(x)
        x = jnp.reshape(x, (-1, h_enc, w_enc, 32))
        x = hk.Conv2DTranspose(16, kernel_shape=3, stride=2, padding="SAME")(x)
        x = jax.nn.relu(x)
        x = hk.Conv2DTranspose(c, kernel_shape=3, stride=2, padding="SAME")(x)
        return x

    def __call__(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Forward pass returning ``dict(kl, nll, recon)``."""
        mean, logvar = self._encode(x)
        eps = jax.random.normal(hk.next_rng_key(), mean.shape)
        z = mean + jnp.exp(0.5 * logvar) * eps
        logits = self._decode(z, x.shape[1:])
        recon = jax.nn.sigmoid(logits)

        kl = 0.5 * (jnp.square(mean) + jnp.exp(logvar) - 1.0 - logvar)
        kl = jnp.mean(jnp.sum(kl, axis=-1))

        bce = jnp.maximum(logits, 0) - logits * x + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        nll = jnp.mean(jnp.sum(bce, axis=[1, 2, 3]))

        return {"kl": kl, "nll": nll, "recon": recon}
