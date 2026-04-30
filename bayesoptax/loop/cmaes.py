import math
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
from evosax.algorithms import CMA_ES

from ..utils import Bounds
from .result import MultiBOResult


def run_cmaes(
        objective: Callable,
        bounds: jax.Array,
        n_eval: int,
        popsize: int,
        key: jax.Array,
) -> tuple[jax.Array, jax.Array, float]:
    """Run CMA-ES for n_eval total evaluations."""

    if isinstance(bounds, Bounds):
        bounds = bounds.to_array()

    low, high = bounds[:, 0], bounds[:, 1]
    center = 0.5 * (low + high)
    std_init = float(jnp.mean(high - low) / 3.0)

    n_generations = n_eval // popsize

    es = CMA_ES(population_size=popsize, solution=center)
    params = es.default_params.replace(std_init=std_init)

    key, init_key = jr.split(key)
    state = es.init(init_key, center, params)

    best_y = jnp.inf
    best_x = center
    history_per_gen = []

    for _ in range(n_generations):
        key, ask_key, tell_key = jr.split(key, 3)

        population, state = es.ask(ask_key, state, params)
        population = jnp.clip(population, low, high)
        fitness = jax.vmap(objective)(population)

        state, metrics = es.tell(tell_key, population, fitness, state, params)

        gen_best_idx = jnp.argmin(fitness)
        gen_best_y = fitness[gen_best_idx]
        if gen_best_y < best_y:
            best_y = gen_best_y
            best_x = population[gen_best_idx]

        history_per_gen.append(float(best_y))

    history = jnp.repeat(jnp.array(history_per_gen), popsize)[:n_eval]

    if len(history) < n_eval:
        pad = jnp.full(n_eval - len(history), history[-1])
        history = jnp.concatenate([history, pad])

    return history, best_x, float(best_y)


def run_cmaes_seeds(
        objective: Callable,
        bounds: jax.Array,
        n_eval: int,
        n_seeds: int = 10,
        popsize: int | None = None,
        base_key: jax.Array | None = None,
) -> MultiBOResult:
    """Run CMA-ES across n_seeds random seeds."""

    if isinstance(bounds, Bounds):
        bounds_arr = bounds.to_array()
    else:
        bounds_arr = bounds

    dim = bounds_arr.shape[0]
    if popsize is None:
        popsize = 4 + int(3 * math.log(dim)) # recommended pop size: 4 + floor(3 * ln(dim))

    if base_key is None:
        base_key = jr.PRNGKey(0)

    seed_keys = jr.split(base_key, n_seeds)
    histories = []
    best_xs = []
    best_ys = []

    for i, key in enumerate(seed_keys):
        print(f"Seed {i+1}/{n_seeds}...", end=" ", flush=True)
        history, best_x, best_y = run_cmaes(objective, bounds, n_eval, popsize, key)
        histories.append(history)
        best_xs.append(best_x)
        best_ys.append(best_y)
        print(f"best y = {best_y:.4f}")

    return MultiBOResult(
        histories=jnp.stack(histories, axis=0),
        best_xs=jnp.stack(best_xs, axis=0),
        best_ys=jnp.array(best_ys),
        n_seeds=n_seeds,
        n_iter=n_eval,
    )
