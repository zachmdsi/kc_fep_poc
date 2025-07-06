import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from kc_fep_poc.metrics import compute_metrics, generate_observations  # noqa: E402
from kc_fep_poc.validator import validate  # noqa: E402


def test_validator_passes(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    csv_path = tmp_path / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "g_t",
            "rho_t",
            "k_hat",
            "k_lzma",
            "free_energy",
            "file",
        ])
        for ep, p in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
            obs = generate_observations(1000, p, rng)
            file = tmp_path / f"ep_{ep}.bin"
            obs.tofile(file)
            m = compute_metrics(obs, p)
            writer.writerow(
                [ep, m.g_t, m.rho_t, m.k_hat, m.k_lzma, m.free_energy, file.name]
            )

    assert validate(csv_path, base_dir=tmp_path)
