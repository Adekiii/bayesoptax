import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol
from typing import Callable


def optimize_acquisition(
        batch_score_fn: Callable,
        single_score_fn: Callable,
        D: int,
        n_restarts: int = 5,
        n_candidates: int = 2048,
        key: jax.Array | None = None,
) -> jax.Array:
    """Optimizes acquisition using multi-start L-BFGS-B."""

    n_sobol = 1 << (n_candidates - 1).bit_length()
    seed = int(jax.random.bits(key, dtype=jnp.uint32)) if key is not None else 0
    sampler = Sobol(d=D, scramble=True, seed=seed)
    X_cands_norm = jnp.array(sampler.random(n=n_sobol)[:n_candidates])

    scores = batch_score_fn(X_cands_norm)
    top_idx = jnp.argsort(scores)[-n_restarts:]
    X_starts = np.array(X_cands_norm[top_idx])

    val_and_grad_fn = jax.jit(jax.value_and_grad(lambda x: -single_score_fn(x)))
    unit_bounds = [(0.0, 1.0)] * D

    best_x_norm = X_starts[-1]
    best_val = -np.inf

    for x0 in X_starts:
        def obj(x):
            val, grad = val_and_grad_fn(jnp.array(x))
            return float(val), np.array(grad, dtype=np.float64)

        result = minimize(obj, x0, method="L-BFGS-B", jac=True, bounds=unit_bounds)
        if -result.fun > best_val:
            best_val = -result.fun
            best_x_norm = result.x

    return jnp.array(best_x_norm)
