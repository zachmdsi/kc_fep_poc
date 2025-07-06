from __future__ import annotations

"""Training orchestrator tying together environment, logger and agent."""

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Iterable, List

from .metrics import Metrics


@dataclass
class Settings:
    """Execution settings for the orchestrator."""

    episodes: int
    csv_path: str | Path = "metrics.csv"


def run(
    env_wrapper: object,
    binary_logger: object,
    agent: object,
    settings: Settings,
) -> list[Metrics]:
    """Run ``agent`` in ``env_wrapper`` for a number of episodes.

    Parameters
    ----------
    env_wrapper : object
        Environment wrapper providing a ``run_episode`` method.
    binary_logger : object
        Logger object passed through to ``env_wrapper``.
    agent : object
        Agent controlling the environment.
    settings : Settings
        Configuration containing the number of episodes and CSV log path.

    Returns
    -------
    list[Metrics]
        Metrics for each executed episode.
    """

    results: list[Metrics] = []
    csv_file = Path(settings.csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    with csv_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "g_t",
                "rho_t",
                "k_hat",
                "k_lzma",
                "free_energy",
            ]
        )

        for episode in range(settings.episodes):
            metrics: Metrics = env_wrapper.run_episode(agent, binary_logger)
            results.append(metrics)
            writer.writerow(
                [
                    episode,
                    metrics.g_t,
                    metrics.rho_t,
                    metrics.k_hat,
                    metrics.k_lzma,
                    metrics.free_energy,
                ]
            )

    return results
