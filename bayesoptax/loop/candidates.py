import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats.qmc import Sobol


def sample_candidates(key: jax.Array, bounds: jax.Array, n_candidates: int) -> jax.Array:
    """Sample candidate points using a Sobol sequence within given bounds."""

    lb, ub = np.array(bounds[:, 0]), np.array(bounds[:, 1])
    d = lb.shape[0]

    n_sobol = 1 << (n_candidates - 1).bit_length()
    seed = int(jax.random.bits(key, dtype=jnp.uint32))
    sampler = Sobol(d=d, scramble=True, seed=seed)
    unit_samples = sampler.random(n=n_sobol)[:n_candidates]

    scaled = lb + unit_samples * (ub - lb)
    return jnp.array(scaled)


def sample_initial(key: jax.Array, bounds: jax.Array, n_init: int) -> jax.Array:
    """Sample initial points for BO."""

    return sample_candidates(key, bounds, n_init)
