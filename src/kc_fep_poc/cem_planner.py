"""Cross-entropy method planner."""

from __future__ import annotations

import numpy as np


# Number of sampled trajectories
_NUM_SAMPLES = 64
# Planning horizon
_HORIZON = 8
# Fraction of elites to keep when updating the distribution
_ELITE_FRAC = 0.25
# Number of refinement iterations
_NUM_ITERS = 3


def plan(obs: np.ndarray, model) -> int:
    """Return action planned via cross-entropy method.

    Parameters
    ----------
    obs : numpy.ndarray
        Current observation passed to the model.
    model : object
        Model must implement ``rollout(obs, actions)`` returning the expected
        free energy of executing ``actions`` starting from ``obs``.

    Returns
    -------
    int
        The chosen action (0 or 1).
    """
    # Initial probability of sampling action 1 at each time step
    probs = np.full(_HORIZON, 0.5)
    rng = np.random.default_rng()

    elite_count = max(1, int(_NUM_SAMPLES * _ELITE_FRAC))

    for _ in range(_NUM_ITERS):
        # Sample action sequences according to current probabilities
        samples = rng.random((_NUM_SAMPLES, _HORIZON)) < probs
        actions = samples.astype(int)

        # Evaluate sequences via model rollout
        costs = np.array([float(model.rollout(obs, seq)) for seq in actions])

        elite_idx = np.argsort(costs)[:elite_count]
        elite_actions = actions[elite_idx]

        # Update sampling distribution using elite set
        probs = elite_actions.mean(axis=0)

    # Choose the action with highest probability after final update
    return int(probs[0] >= 0.5)
