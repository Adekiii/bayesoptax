import jax
import jax.numpy as jnp
from typing import Callable

from .candidates import sample_candidates
from ..utils import Bounds


def run_random_search(
        objective: Callable,
        bounds: jax.Array,
        n_eval: int,
        key: jax.Array,
) -> jax.Array:
    """Run random search for n_eval evaluations and return best-so-far history.

    Samples n_eval points uniformly at random and tracks the running best,
    matching the total budget of a BO run (n_init + n_iter).
    """

    if isinstance(bounds, Bounds):
        bounds = bounds.to_array()

    X = sample_candidates(key, bounds, n_eval)
    y = jax.vmap(objective)(X)

    history = jnp.minimum.accumulate(y)
    return history
