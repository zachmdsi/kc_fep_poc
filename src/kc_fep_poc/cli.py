"""Command-line interface for the toy compression demo."""
from .metrics import generate_observations, compute_metrics


def main() -> None:
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
