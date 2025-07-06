import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kc_fep_poc.free_energy import free_energy_step  # noqa: E402


def test_free_energy_step_returns_sum():
    assert free_energy_step(0.5, 1.0) == 1.5
