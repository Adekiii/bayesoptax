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
    **run_kwargs,
) -> MultiBOResult:
    """Run the BO loop across n_seeds random seeds.

    TO DO: add documentation
    """

    if base_key is None:
        base_key = jr.PRNGKey(0)

    seed_keys = jr.split(base_key, n_seeds)

    histories = []
    random_histories = []
    best_xs = []
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
        if result.random_history is not None:
            random_histories.append(result.random_history)
        best_xs.append(result.best_x)
        best_ys.append(result.best_y)
        all_results.append(result)

        print(f"best y = {result.best_y:.4f}")

    indexed_results = list(enumerate(best_ys, start=1))
    sorted_results = sorted(indexed_results, key=lambda x: x[1])
    formatted_results = ([f"{i}: {y:.4f}" for i, y in sorted_results])
    print("best seeds: ", ", ".join(formatted_results))

    return MultiBOResult(
        histories = jnp.stack(histories, axis=0),
        best_xs = jnp.array(best_xs),
        best_ys = jnp.array(best_ys),
        results = all_results,
        n_seeds = n_seeds,
        n_iter = len(all_results[0].history),
        random_histories = jnp.stack(random_histories, axis=0) if random_histories else None,
    )