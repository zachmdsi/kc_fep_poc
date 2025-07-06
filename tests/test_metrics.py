import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kc_fep_poc.metrics import compute_metrics  # noqa: E402
from kc_fep_poc.metrics import generate_observations  # noqa: E402


def test_metrics_runs():
    obs = generate_observations(10, 0.5)
    metrics = compute_metrics(obs, 0.5)
    assert metrics.k_lzma > 0
    assert metrics.k_hat > 0
