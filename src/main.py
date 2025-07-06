import lzma
import numpy as np
from dataclasses import dataclass


def generate_observations(num_steps: int, p: float) -> np.ndarray:
    """Simulate binary observations from Bernoulli(p)."""
    return np.random.binomial(1, p, size=num_steps).astype(np.uint8)


def compute_nll(obs: np.ndarray, p: float) -> float:
    """Return negative log likelihood in bits for Bernoulli model."""
    eps = 1e-8
    probs = np.where(obs == 1, p, 1 - p)
    return float(-np.sum(np.log2(probs + eps)))


def compression_bound(nll_bits: float, theta: float) -> float:
    """Two-part code bound from README eq. (1)."""
    return theta ** 2 * np.log(2) + nll_bits


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
    nll = compute_nll(obs, model_p)
    k_hat = compression_bound(nll, model_p)
    k_lzma = lzma_size_bits(obs)
    g_t = k_lzma - k_hat
    # For this simple model free energy equals nll (accuracy term) as we ignore KL
    free_energy = nll
    rho_t = free_energy / g_t if g_t != 0 else float('inf')
    return Metrics(g_t, rho_t, k_hat, k_lzma, free_energy)


def main():
    num_steps = 1000
    true_p = 0.3
    obs = generate_observations(num_steps, true_p)
    # MLE estimate
    mle_p = float(obs.mean())
    metrics = compute_metrics(obs, mle_p)
    print("Observations:", num_steps)
    print("True p:", true_p)
    print("MLE p:", mle_p)
    print("K_lzma (bits):", metrics.k_lzma)
    print("K_hat (bits):", metrics.k_hat)
    print("Compression gap G_T:", metrics.g_t)
    print("Free energy F_T:", metrics.free_energy)
    print("Bit-elasticity rho_T:", metrics.rho_t)


if __name__ == "__main__":
    main()
