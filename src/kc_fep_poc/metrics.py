import lzma
from dataclasses import dataclass

import numpy as np


def generate_observations(
    num_steps: int, p: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Simulate binary observations from Bernoulli(p).

    Parameters
    ----------
    num_steps : int
        Number of observations to generate.
    p : float
        Bernoulli success probability.
    rng : numpy.random.Generator, optional
        Random number generator to use. If omitted, ``numpy.random.default_rng``
        is used.
    """

    generator = np.random.default_rng() if rng is None else rng
    return generator.binomial(1, p, size=num_steps).astype(np.uint8)


def compute_nll(obs: np.ndarray, p: float) -> float:
    """Return negative log likelihood in bits for Bernoulli model."""
    eps = 1e-8
    probs = np.where(obs == 1, p, 1 - p)
    return float(-np.sum(np.log2(probs + eps)))


def compression_bound(nll_bits: float, theta: float) -> float:
    """Two-part code bound from README eq. (1)."""
    return theta**2 * np.log(2) + nll_bits


def lzma_size_bits(obs: np.ndarray) -> int:
    """Compress using LZMA and return size in bits."""
    data = bytes(obs)
    return len(lzma.compress(data)) * 8


@dataclass
class Metrics:
    g_t: float
    rho_t: float
    k_hat: float
    k_lzma: int
    free_energy: float


def compute_metrics(obs: np.ndarray, model_p: float) -> Metrics:
    """Compute compression and free-energy metrics."""
    nll = compute_nll(obs, model_p)
    k_hat = compression_bound(nll, model_p)
    k_lzma = lzma_size_bits(obs)
    g_t = k_lzma - k_hat
    # For this simple model free energy equals nll (accuracy term) as we ignore KL
    free_energy = nll
    rho_t = free_energy / g_t if g_t != 0 else float("inf")
    return Metrics(g_t, rho_t, k_hat, k_lzma, free_energy)
