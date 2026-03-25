import jax
import jax.random as jr


def sample_candidates(key: jax.Array, bounds: jax.Array, n_candidates: int):
    """Sample candidate points uniformly within given bounds."""

    lb, ub = bounds[:, 0], bounds[:, 1]
    return jr.uniform(key, shape=(n_candidates, lb.shape[0]), minval=lb, maxval=ub)


def sample_initial(key: jax.Array, bounds: jax.Array, n_init: int):
    """Sample initial points for BO."""

    return sample_candidates(key, bounds, n_init)
