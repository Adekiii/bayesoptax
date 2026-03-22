from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr

from .loop import run
from .result import MultiBOResult


def run_seeds(
    objective: Callable,
    bounds: jax.Array,
    n_seeds: int = 10,
    base_key: jax.Array | None = None,
    verbose: bool = False,
    **run_kwargs,
) -> MultiBOResult:
    """Run the BO loop across n_seeds random seeds.

    TO DO: add documentation
    """

    if base_key is None:
        base_key = jr.PRNGKey(0)

    seed_keys = jr.split(base_key, n_seeds)
    n_iter = run_kwargs.get("n_iter", 50)

    histories = []
    best_ys = []
    all_results = []

    for i, key in enumerate(seed_keys):
        print(f"Seed {i+1}/{n_seeds}...", end=" ", flush=True)

        result = run(
            objective = objective,
            bounds = bounds,
            key = key,
            **run_kwargs,
        )

        histories.append(result.history)
        best_ys.append(result.best_y)
        all_results.append(result)

        print(f"best y = {result.best_y:.4f}")

    return MultiBOResult(
        histories = jnp.stack(histories, axis=0),
        best_ys = jnp.array(best_ys),
        results = all_results,
        n_seeds = n_seeds,
        n_iter = n_iter,
    )