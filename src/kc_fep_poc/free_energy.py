"""Free energy computation helpers."""


def free_energy_step(kl: float, nll: float) -> float:
    """Return variational free energy for a single time step.

    Parameters
    ----------
    kl : float
        Divergence term measuring complexity.
    nll : float
        Negative log-likelihood or accuracy term.

    Returns
    -------
    float
        The sum ``kl + nll`` as a floating point number.
    """
    return float(kl + nll)
