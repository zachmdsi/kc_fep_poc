import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from kc_fep_poc.cem_planner import plan  # noqa: E402


class DummyModel:
    def rollout(self, obs: np.ndarray, actions: np.ndarray) -> float:
        # Simple cost proportional to the sum of actions
        return float(np.sum(actions))


def test_plan_returns_int():
    obs = np.array([0])
    action = plan(obs, DummyModel(), seed=123)
    assert isinstance(action, int)
    assert action in (0, 1)


def test_plan_deterministic_with_seed():
    obs = np.array([0])
    a1 = plan(obs, DummyModel(), seed=42)
    a2 = plan(obs, DummyModel(), seed=42)
    assert a1 == a2
