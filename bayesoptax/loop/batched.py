import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Callable

from ..surrogates.gp import precompute, predict_precomputed, init_params as gp_init_params
from ..surrogates.inference import fit
from ..acquisitions import get_acquisition
from .candidates import sample_initial
from .optimize import optimize_acquisition
from ..utils import Bounds
from .result import MultiBOResult


def _bo_step_one_seed(
        X_obs, y_obs, count, prev_params,
        fit_key, acq_key, opt_key,
        lb, ub, D, kernel_name, acquisition_fn,
        n_restarts, fit_max_iter, n_candidates, acq_restarts,
):
    """One BO iteration for a single seed. Pure function of arrays - vmappable across seeds."""

    max_n = X_obs.shape[0]
    mask = jnp.arange(max_n) < count

    y_valid_sum = jnp.sum(jnp.where(mask, y_obs, 0.0))
    y_mean = y_valid_sum / count
    y_var = jnp.sum(jnp.where(mask, (y_obs - y_mean) ** 2, 0.0)) / count
    y_std = jnp.sqrt(y_var) + 1e-8
    y_norm = (y_obs - y_mean) / y_std

    X_obs_norm = (X_obs - lb) / (ub - lb)

    fitted_params = fit(
        X_obs_norm, y_norm, kernel_name=kernel_name,
        n_restarts=n_restarts, max_iter=fit_max_iter,
        init_params_override=prev_params, key=fit_key, mask=mask,
    )

    L_chol, alpha = precompute(fitted_params, X_obs_norm, y_norm, kernel_name, mask=mask)
    best_y_norm = jnp.min(jnp.where(mask, y_norm, jnp.inf))

    def batch_score_fn(X_norm):
        mean, var = predict_precomputed(fitted_params, X_obs_norm, L_chol, alpha, X_norm, kernel_name, mask=mask)
        return acquisition_fn(mean, var, best_y=best_y_norm, key=acq_key)

    def single_score_fn(x_norm):
        mean, var = predict_precomputed(fitted_params, X_obs_norm, L_chol, alpha, x_norm[None], kernel_name, mask=mask)
        return acquisition_fn(mean, var, best_y=best_y_norm, key=acq_key).squeeze()

    x_next_norm = optimize_acquisition(
        batch_score_fn, single_score_fn, D,
        n_restarts=acq_restarts, n_candidates=n_candidates, key=opt_key,
    )
    X_next = lb + x_next_norm * (ub - lb)
    return X_next, fitted_params


def run_batched(
        objective: Callable,
        bounds: jax.Array,
        n_seeds: int = 10,
        n_init: int = 10,
        n_iter: int = 50,
        n_candidates: int = 512,
        n_restarts: int = 3,
        fit_max_iter: int = 200,
        acq_restarts: int = 3,
        kernel_name: str = "rbf",
        acquisition_name: str = "ei",
        base_key: jax.Array | None = None,
) -> MultiBOResult:
    """Run standard BO across n_seeds independent trajectories simultaneously."""

    if base_key is None:
        base_key = jr.PRNGKey(0)
    if isinstance(bounds, Bounds):
        bounds = bounds.to_array()

    acquisition_fn = get_acquisition(acquisition_name)
    lb, ub = bounds[:, 0], bounds[:, 1]
    D = bounds.shape[0]

    seed_keys = jr.split(base_key, n_seeds)
    init_sample_keys, loop_keys = jax.vmap(lambda k: tuple(jr.split(k)))(seed_keys)

    X_init = jax.vmap(lambda k: sample_initial(k, bounds, n_init))(init_sample_keys)
    y_init = jax.vmap(jax.vmap(objective))(X_init)

    print(f"Initializing {n_init} points x {n_seeds} seeds. Best y per seed = {jnp.min(y_init, axis=1)}")

    max_n = n_init + n_iter
    X_obs = jnp.zeros((n_seeds, max_n, D)).at[:, :n_init].set(X_init)
    y_obs = jnp.zeros((n_seeds, max_n)).at[:, :n_init].set(y_init)
    count = jnp.array(n_init)

    fresh_params = gp_init_params(kernel_name, D)
    prev_params = jax.tree.map(lambda x: jnp.broadcast_to(x, (n_seeds,) + x.shape), fresh_params)

    def scan_step(carry, _):
        X_obs, y_obs, count, prev_params, keys = carry

        def per_seed(key_s, X_obs_s, y_obs_s, prev_params_s):
            fit_key, acq_key, opt_key = jr.split(key_s, 3)
            return _bo_step_one_seed(
                X_obs_s, y_obs_s, count, prev_params_s,
                fit_key, acq_key, opt_key,
                lb, ub, D, kernel_name, acquisition_fn,
                n_restarts, fit_max_iter, n_candidates, acq_restarts,
            )

        X_next, fitted_params = jax.vmap(per_seed)(keys, X_obs, y_obs, prev_params)
        y_next = jax.vmap(objective)(X_next)

        X_obs = X_obs.at[:, count].set(X_next)
        y_obs = y_obs.at[:, count].set(y_next)
        new_count = count + 1

        mask_now = jnp.arange(X_obs.shape[1])[None, :] < new_count
        best_so_far = jnp.min(jnp.where(mask_now, y_obs, jnp.inf), axis=1)

        new_keys = jax.vmap(lambda k: jr.split(k, 1)[0])(keys)
        return (X_obs, y_obs, new_count, fitted_params, new_keys), best_so_far

    init_carry = (X_obs, y_obs, count, prev_params, loop_keys)
    (X_obs_final, y_obs_final, _, _, _), history_per_iter = jax.lax.scan(
        scan_step, init_carry, xs=None, length=n_iter
    )

    init_history = jax.vmap(jnp.minimum.accumulate)(y_init)
    history = jnp.concatenate([init_history, history_per_iter.T], axis=1)

    best_idx = jnp.argmin(y_obs_final, axis=1)
    best_xs = X_obs_final[jnp.arange(n_seeds), best_idx]
    best_ys = y_obs_final[jnp.arange(n_seeds), best_idx]

    print("Done. Best y per seed:", best_ys)

    return MultiBOResult(
        histories=history,
        best_xs=best_xs,
        best_ys=best_ys,
        n_seeds=n_seeds,
        n_iter=max_n,
    )


def _turbo_step_one_seed(
        X_obs, y_obs, count, prev_params, L,
        fit_key, acq_key, opt_key,
        lb, ub, D, kernel_name, acquisition_fn,
        n_restarts, fit_max_iter, n_candidates, acq_restarts,
):
    """One TuRBO iteration for a single seed."""

    max_n = X_obs.shape[0]
    mask = jnp.arange(max_n) < count

    y_valid_sum = jnp.sum(jnp.where(mask, y_obs, 0.0))
    y_mean = y_valid_sum / count
    y_var = jnp.sum(jnp.where(mask, (y_obs - y_mean) ** 2, 0.0)) / count
    y_std = jnp.sqrt(y_var) + 1e-8
    y_norm = (y_obs - y_mean) / y_std

    X_obs_norm = (X_obs - lb) / (ub - lb)

    best_idx = jnp.argmin(jnp.where(mask, y_obs, jnp.inf))
    center = X_obs_norm[best_idx]

    half_L = L / 2.0
    tr_lb = jnp.clip(center - half_L, 0.0, 1.0)
    tr_ub = jnp.clip(center + half_L, 0.0, 1.0)

    fitted_params = fit(
        X_obs_norm, y_norm, kernel_name=kernel_name,
        n_restarts=n_restarts, max_iter=fit_max_iter,
        init_params_override=prev_params, key=fit_key, mask=mask,
    )

    L_chol, alpha = precompute(fitted_params, X_obs_norm, y_norm, kernel_name, mask=mask)
    best_y_norm = jnp.min(jnp.where(mask, y_norm, jnp.inf))

    def batch_score_fn(X_norm):
        mean, var = predict_precomputed(fitted_params, X_obs_norm, L_chol, alpha, X_norm, kernel_name, mask=mask)
        return acquisition_fn(mean, var, best_y=best_y_norm, key=acq_key)

    def single_score_fn(x_norm):
        mean, var = predict_precomputed(fitted_params, X_obs_norm, L_chol, alpha, x_norm[None], kernel_name, mask=mask)
        return acquisition_fn(mean, var, best_y=best_y_norm, key=acq_key).squeeze()

    x_next_norm = optimize_acquisition(
        batch_score_fn, single_score_fn, D,
        n_restarts=acq_restarts, n_candidates=n_candidates, key=opt_key,
        region_bounds=(tr_lb, tr_ub), tr_center=center,
    )
    X_next = lb + x_next_norm * (ub - lb)
    return X_next, fitted_params


def _update_trust_region(
        y_next, prev_best, L, success_counter, failure_counter,
        length_init, length_min, length_max, success_tolerance, failure_tolerance,
):
    """Vectorized version of TuRBO."""

    improved = y_next < prev_best - 1e-3 * jnp.abs(prev_best)

    success_counter = jnp.where(improved, success_counter + 1, 0)
    failure_counter = jnp.where(improved, 0, failure_counter + 1)

    expand = success_counter >= success_tolerance
    shrink = (~expand) & (failure_counter >= failure_tolerance)

    L = jnp.where(expand, jnp.minimum(2.0 * L, length_max), L)
    L = jnp.where(shrink, L / 2.0, L)
    success_counter = jnp.where(expand, 0, success_counter)
    failure_counter = jnp.where(shrink, 0, failure_counter)

    restart = L < length_min
    L = jnp.where(restart, length_init, L)
    success_counter = jnp.where(restart, 0, success_counter)
    failure_counter = jnp.where(restart, 0, failure_counter)

    return L, success_counter, failure_counter, restart


def _select_per_seed(restart, fresh_leaf, new_leaf):
    r = restart.reshape((-1,) + (1,) * (new_leaf.ndim - 1))
    return jnp.where(r, fresh_leaf, new_leaf)


def run_turbo_batched(
        objective: Callable,
        bounds: jax.Array,
        n_seeds: int = 10,
        n_init: int = 10,
        n_iter: int = 50,
        n_candidates: int = 512,
        n_restarts: int = 3,
        fit_max_iter: int = 200,
        acq_restarts: int = 3,
        kernel_name: str = "rbf",
        acquisition_name: str = "ei",
        length: float = 0.8,
        length_min: float = 0.5**7,
        length_max: float = 1.6,
        success_tolerance: int = 3,
        failure_tolerance: int | None = None,
        base_key: jax.Array | None = None,
) -> MultiBOResult:
    """Run TuRBO across n_seeds independent trajectories simultaneously."""

    if base_key is None:
        base_key = jr.PRNGKey(0)
    if isinstance(bounds, Bounds):
        bounds = bounds.to_array()

    acquisition_fn = get_acquisition(acquisition_name)
    lb, ub = bounds[:, 0], bounds[:, 1]
    D = bounds.shape[0]

    if failure_tolerance is None:
        failure_tolerance = max(D, 5)

    seed_keys = jr.split(base_key, n_seeds)
    init_sample_keys, loop_keys = jax.vmap(lambda k: tuple(jr.split(k)))(seed_keys)

    X_init = jax.vmap(lambda k: sample_initial(k, bounds, n_init))(init_sample_keys)
    y_init = jax.vmap(jax.vmap(objective))(X_init)

    print(f"Initializing {n_init} points x {n_seeds} seeds. Best y per seed = {jnp.min(y_init, axis=1)}")

    max_n = n_init + n_iter
    X_obs = jnp.zeros((n_seeds, max_n, D)).at[:, :n_init].set(X_init)
    y_obs = jnp.zeros((n_seeds, max_n)).at[:, :n_init].set(y_init)
    count = jnp.array(n_init)

    fresh_params = gp_init_params(kernel_name, D)
    prev_params = jax.tree.map(lambda x: jnp.broadcast_to(x, (n_seeds,) + x.shape), fresh_params)

    L = jnp.full((n_seeds,), length)
    success_counter = jnp.zeros((n_seeds,), dtype=jnp.int32)
    failure_counter = jnp.zeros((n_seeds,), dtype=jnp.int32)

    def scan_step(carry, _):
        X_obs, y_obs, count, prev_params, L, success_counter, failure_counter, keys = carry

        def per_seed(key_s, X_obs_s, y_obs_s, prev_params_s, L_s):
            fit_key, acq_key, opt_key = jr.split(key_s, 3)
            return _turbo_step_one_seed(
                X_obs_s, y_obs_s, count, prev_params_s, L_s,
                fit_key, acq_key, opt_key,
                lb, ub, D, kernel_name, acquisition_fn,
                n_restarts, fit_max_iter, n_candidates, acq_restarts,
            )

        X_next, fitted_params = jax.vmap(per_seed)(keys, X_obs, y_obs, prev_params, L)
        y_next = jax.vmap(objective)(X_next)

        mask_before = jnp.arange(X_obs.shape[1]) < count
        prev_best = jnp.min(jnp.where(mask_before[None, :], y_obs, jnp.inf), axis=1)

        X_obs = X_obs.at[:, count].set(X_next)
        y_obs = y_obs.at[:, count].set(y_next)
        new_count = count + 1

        new_L, new_success_counter, new_failure_counter, restart = jax.vmap(
            lambda yn, pb, l, sc, fc: _update_trust_region(
                yn, pb, l, sc, fc, length, length_min, length_max,
                success_tolerance, failure_tolerance,
            )
        )(y_next, prev_best, L, success_counter, failure_counter)

        fresh_params_batched = jax.tree.map(lambda x: jnp.broadcast_to(x, (n_seeds,) + x.shape), fresh_params)
        new_prev_params = jax.tree.map(
            lambda fresh, new: _select_per_seed(restart, fresh, new),
            fresh_params_batched, fitted_params,
        )

        mask_now = jnp.arange(X_obs.shape[1])[None, :] < new_count
        best_so_far = jnp.min(jnp.where(mask_now, y_obs, jnp.inf), axis=1)

        new_keys = jax.vmap(lambda k: jr.split(k, 1)[0])(keys)
        new_carry = (
            X_obs, y_obs, new_count, new_prev_params,
            new_L, new_success_counter, new_failure_counter, new_keys,
        )
        return new_carry, best_so_far

    init_carry = (X_obs, y_obs, count, prev_params, L, success_counter, failure_counter, loop_keys)
    (X_obs_final, y_obs_final, *_), history_per_iter = jax.lax.scan(
        scan_step, init_carry, xs=None, length=n_iter
    )

    init_history = jax.vmap(jnp.minimum.accumulate)(y_init)
    history = jnp.concatenate([init_history, history_per_iter.T], axis=1)

    best_idx = jnp.argmin(y_obs_final, axis=1)
    best_xs = X_obs_final[jnp.arange(n_seeds), best_idx]
    best_ys = y_obs_final[jnp.arange(n_seeds), best_idx]

    print("Done. Best y per seed:", best_ys)

    return MultiBOResult(
        histories=history,
        best_xs=best_xs,
        best_ys=best_ys,
        n_seeds=n_seeds,
        n_iter=max_n,
    )
