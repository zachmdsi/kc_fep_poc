import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kc_fep_poc.metrics import compute_metrics, generate_observations


def test_metrics_runs():
    obs = generate_observations(10, 0.5)
    metrics = compute_metrics(obs, 0.5)
    assert metrics.k_lzma > 0
    assert metrics.k_hat > 0
