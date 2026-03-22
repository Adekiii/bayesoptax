import jax
import jax.numpy as jnp
import jax.random as jr


def sample_candidates(key: jax.Array, bounds: jax.Array, n_candidates: int):
    """Sample candidate points within given bounds."""

    D = bounds.shape[0]
    X_unit = jr.uniform(key, shape=(n_candidates, D))
    return bounds[:, 0] + X_unit * (bounds[:, 1] - bounds[:, 0])


def sample_initial(key: jax.Array, bounds: jax.Array, n_init: int):
    """Sample initial points for BO."""

    return sample_candidates(key, bounds, n_init)
