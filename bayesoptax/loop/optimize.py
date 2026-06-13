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
        n_restarts: int = 3,
        n_candidates: int = 512,
        key: jax.Array | None = None,
        region_bounds: tuple[np.ndarray, np.ndarray] | None = None,
        tr_center: np.ndarray | None = None,
) -> jax.Array:
    """Optimizes acquisition using multi-start L-BFGS-B."""

    n_sobol = 1 << (n_candidates - 1).bit_length()
    seed = int(jax.random.bits(key, dtype=jnp.uint32)) if key is not None else 0
    sampler = Sobol(d=D, scramble=True, seed=seed)
    raw = np.array(sampler.random(n=n_sobol)[:n_candidates])

    if region_bounds is not None:
        lb_r, ub_r = region_bounds
        pert = lb_r + raw * (ub_r - lb_r)

        if tr_center is not None:
            prob_perturb = min(20.0 / D, 1.0)
            rng = np.random.default_rng(seed + 1)
            mask = rng.random((len(pert), D)) <= prob_perturb
            # ensure every candidate perturbs at least one dimension
            zero_rows = np.where(mask.sum(axis=1) == 0)[0]
            if len(zero_rows) > 0:
                mask[zero_rows, rng.integers(0, D, size=len(zero_rows))] = True
            X_cands_norm = np.tile(tr_center, (len(pert), 1))
            X_cands_norm[mask] = pert[mask]
        else:
            X_cands_norm = pert

        X_cands_norm = jnp.array(X_cands_norm)
        lbfgsb_bounds = [(float(lb_r[i]), float(ub_r[i])) for i in range(D)]
    else:
        X_cands_norm = jnp.array(raw)
        lbfgsb_bounds = [(0.0, 1.0)] * D

    scores = batch_score_fn(X_cands_norm)
    top_idx = jnp.argsort(scores)[-n_restarts:]
    X_starts = np.array(X_cands_norm[top_idx])

    val_and_grad_fn = jax.jit(jax.value_and_grad(lambda x: -single_score_fn(x)))

    best_x_norm = X_starts[-1]
    best_val = -np.inf

    for x0 in X_starts:
        def obj(x):
            val, grad = val_and_grad_fn(jnp.array(x))
            return float(val), np.array(grad, dtype=np.float64)

        result = minimize(obj, x0, method="L-BFGS-B", jac=True, bounds=lbfgsb_bounds)
        if -result.fun > best_val:
            best_val = -result.fun
            best_x_norm = result.x

    return jnp.array(best_x_norm)
