import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from kc_fep_poc.metrics import (  # noqa: E402
    bits_lzma,
    compute_metrics,
    generate_observations,
    lzma_size_bits,
)


def test_metrics_runs():
    obs = generate_observations(10, 0.5)
    metrics = compute_metrics(obs, 0.5)
    assert metrics.k_lzma > 0
    assert metrics.k_hat > 0


def test_generate_observations_values_and_mean():
    rng = np.random.default_rng(123)
    p = 0.3
    obs = generate_observations(1000, p, rng)
    assert set(np.unique(obs)).issubset({0, 1})
    assert float(obs.mean()) == pytest.approx(p, abs=0.02)


def test_generate_observations_deterministic_seed():
    seed = 42
    p = 0.7
    obs1 = generate_observations(50, p, np.random.default_rng(seed))
    obs2 = generate_observations(50, p, np.random.default_rng(seed))
    assert np.array_equal(obs1, obs2)


def test_bits_lzma_file(tmp_path):
    obs = generate_observations(32, 0.5)
    f = tmp_path / "obs.bin"
    obs.tofile(f)
    assert bits_lzma(f) == lzma_size_bits(obs)
