import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Callable

from .candidates import sample_candidates
from ..utils import Bounds
from .result import MultiBOResult


def run_random_search(
        objective: Callable,
        bounds: jax.Array,
        n_eval: int,
        key: jax.Array,
) -> jax.Array:
    """Run random search for n_eval evaluations and return best-so-far history."""

    if isinstance(bounds, Bounds):
        bounds = bounds.to_array()

    X = sample_candidates(key, bounds, n_eval)
    y = jax.vmap(objective)(X)

    history = jnp.minimum.accumulate(y)
    return history


def run_random_seeds(
        objective: Callable,
        bounds: jax.Array,
        n_eval: int,
        n_seeds: int = 10,
        base_key: jax.Array | None = None,
) -> MultiBOResult:
    """Run random search across n_seeds random seeds."""

    if base_key is None:
        base_key = jr.PRNGKey(0)

    seed_keys = jr.split(base_key, n_seeds)
    histories = []

    for i, key in enumerate(seed_keys):
        print(f"Seed {i+1}/{n_seeds}...", end=" ", flush=True)
        history = run_random_search(objective, bounds, n_eval, key)
        histories.append(history)
        print(f"best y = {float(history[-1]):.4f}")

    histories_arr = jnp.stack(histories, axis=0)
    return MultiBOResult(
        histories=histories_arr,
        best_xs=jnp.zeros((n_seeds, 1)),
        best_ys=histories_arr[:, -1],
        n_seeds=n_seeds,
        n_iter=n_eval,
    )
