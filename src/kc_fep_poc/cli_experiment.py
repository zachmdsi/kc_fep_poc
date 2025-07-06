"""Command-line experiment runner using :mod:`kc_fep_poc.orchestrator`."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .metrics import compute_metrics, generate_observations
from .orchestrator import Settings, run


class BernoulliEnvWrapper:
    """Toy environment producing Bernoulli observations."""

    def __init__(self, p: float, steps: int, seed: int | None = None) -> None:
        import numpy as np

        self.p = p
        self.steps = steps
        self.rng = np.random.default_rng(seed)

    def run_episode(self, agent: object, binary_logger: object) -> object:
        obs = generate_observations(self.steps, self.p, self.rng)
        m = compute_metrics(obs, float(obs.mean()))
        return m


def _print_table(metrics: list) -> None:
    header = ("episode", "g_t", "rho_t", "k_hat", "k_lzma", "free_energy")
    print(
        f"{header[0]:>7} {header[1]:>10} {header[2]:>10} "
        f"{header[3]:>10} {header[4]:>10} {header[5]:>12}"
    )
    for i, m in enumerate(metrics):
        print(
            f"{i:7d} {m.g_t:10.2f} {m.rho_t:10.2f} "
            f"{m.k_hat:10.2f} {m.k_lzma:10d} {m.free_energy:12.2f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run KC-FEP experiment")
    parser.add_argument("--config", default="config.yml", help="path to config file")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    episodes = int(cfg.get("episodes", 1))
    csv_path = cfg.get("csv_path", "metrics.csv")
    p = float(cfg.get("p", 0.5))
    steps = int(cfg.get("steps", 128))
    seed = cfg.get("seed")

    env = BernoulliEnvWrapper(p=p, steps=steps, seed=seed)
    agent = object()
    logger = object()
    settings = Settings(episodes=episodes, csv_path=csv_path)

    metrics = run(env, logger, agent, settings)
    _print_table(metrics)


if __name__ == "__main__":
    main()
