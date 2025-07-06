import haiku as hk
import jax
import jax.numpy as jnp


class WorldModel(hk.Module):
    """Simple convolutional VAE-style world model."""

    def __init__(self, latent_dim: int = 32, name: str | None = None) -> None:
        super().__init__(name=name)
        self.latent_dim = latent_dim

    def __call__(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        # Encoder: two convolutional layers
        h = hk.Conv2D(32, kernel_shape=3, stride=2, padding="SAME")(x)
        h = jax.nn.relu(h)
        h = hk.Conv2D(64, kernel_shape=3, stride=2, padding="SAME")(h)
        h = jax.nn.relu(h)
        enc_shape = h.shape[1:]
        h = hk.Flatten()(h)
        stats = hk.Linear(self.latent_dim * 2)(h)
        mean, logvar = jnp.split(stats, 2, axis=-1)

        # Reparameterisation trick
        eps = jax.random.normal(hk.next_rng_key(), mean.shape)
        z = mean + jnp.exp(0.5 * logvar) * eps

        # KL divergence to N(0,1)
        kl = 0.5 * jnp.sum(
            jnp.square(mean) + jnp.exp(logvar) - 1.0 - logvar, axis=-1
        )
        kl = jnp.mean(kl)

        # Decoder
        h = hk.Linear(int(jnp.prod(jnp.array(enc_shape))))(z)
        h = jax.nn.relu(h)
        h = jnp.reshape(h, (-1,) + tuple(enc_shape))
        h = hk.Conv2DTranspose(32, kernel_shape=3, stride=2, padding="SAME")(h)
        h = jax.nn.relu(h)
        logits = hk.Conv2DTranspose(x.shape[-1], kernel_shape=3, stride=2, padding="SAME")(h)
        recon = jax.nn.sigmoid(logits)

        # Bernoulli negative log-likelihood
        eps2 = 1e-6
        nll = -jnp.sum(
            x * jnp.log(recon + eps2) + (1 - x) * jnp.log(1 - recon + eps2),
            axis=(1, 2, 3),
        )
        nll = jnp.mean(nll)

        return {"kl": kl, "nll": nll, "recon": recon}
