from __future__ import annotations

"""Training orchestrator tying together environment, logger and agent."""

import csv
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .metrics import Metrics, bits_lzma


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

    with csv_file.open("w", newline="") as f, ThreadPoolExecutor() as pool:
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

        pending: list[tuple[int, Metrics, object]] = []

        for episode in range(settings.episodes):
            result = env_wrapper.run_episode(agent, binary_logger)
            if isinstance(result, tuple):
                metrics, obs_file = result
                future = pool.submit(bits_lzma, obs_file)
            else:
                metrics = result
                future = None

            pending.append((episode, metrics, future))

            i = 0
            while i < len(pending):
                ep, m, fut = pending[i]
                if fut is None or fut.done():
                    if fut is not None:
                        m.k_lzma = fut.result()
                    results.append(m)
                    writer.writerow(
                        [ep, m.g_t, m.rho_t, m.k_hat, m.k_lzma, m.free_energy]
                    )
                    pending.pop(i)
                else:
                    i += 1

        for ep, m, fut in pending:
            if fut is not None:
                m.k_lzma = fut.result()
            results.append(m)
            writer.writerow([ep, m.g_t, m.rho_t, m.k_hat, m.k_lzma, m.free_energy])

    return results
