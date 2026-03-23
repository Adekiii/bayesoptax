import jax
import jax.numpy as jnp
import jax.random as jr
from pyDOE import lhs


def sample_candidates(key: jax.Array, bounds: jax.Array, n_candidates: int):
    """Sample candidate points within given bounds using LHS."""

    D = bounds.shape[0]
    lb, ub = bounds[:, 0], bounds[:, 1]
    candidates = lb + lhs(D, n_candidates) * (ub - lb)
    return candidates


def sample_initial(key: jax.Array, bounds: jax.Array, n_init: int):
    """Sample initial points for BO."""

    return sample_candidates(key, bounds, n_init)
