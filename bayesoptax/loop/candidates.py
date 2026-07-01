import jax
import jax.numpy as jnp


def sample_candidates(key: jax.Array, bounds: jax.Array, n_candidates: int) -> jax.Array:
    """Sample candidate points uniformly at random within given bounds."""

    lb, ub = bounds[:, 0], bounds[:, 1]
    d = lb.shape[0]

    unit_samples = jax.random.uniform(key, (n_candidates, d))
    return lb + unit_samples * (ub - lb)


def sample_initial(key: jax.Array, bounds: jax.Array, n_init: int) -> jax.Array:
    """Sample initial points for BO."""

    return sample_candidates(key, bounds, n_init)
