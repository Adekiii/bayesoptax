import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Callable

from ..surrogates.gp import precompute, predict_precomputed
from ..surrogates.inference import fit
from ..acquisitions import get_acquisition
from .candidates import sample_initial
from .optimize import optimize_acquisition
from .random_search import run_random_search
from ..utils import Bounds
from .result import BOResult


def run(
        objective: Callable,
        bounds: jax.Array,
        n_init: int = 10,
        n_iter: int = 50,
        n_candidates: int = 512,
        n_restarts: int = 3,
        kernel_name: str = "rbf",
        acquisition_name: str = "ei",
        acquisition_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        X_init: jax.Array | None = None,
        y_init: jax.Array | None = None,
        key: jax.Array | None = None,
        max_points: int | None = None,
) -> BOResult:
    """Run BO."""

    if key is None:
        key = jr.PRNGKey(0)
    if acquisition_kwargs is None:
        acquisition_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}
    if isinstance(bounds, Bounds):
        bounds = bounds.to_array()

    acquisition_fn = get_acquisition(acquisition_name)
    lb, ub = bounds[:, 0], bounds[:, 1]
    D = bounds.shape[0]

    if X_init is not None and y_init is not None:
        X_init_pts, y_init_pts = X_init, y_init
    elif (X_init is not None) or (y_init is not None):
        raise ValueError("Provide both X_init and y_init (or neither).")
    else:
        key, subkey = jr.split(key)
        X_init_pts = sample_initial(subkey, bounds, n_init)
        y_init_pts = jax.vmap(objective)(X_init_pts)

    n_init_actual = X_init_pts.shape[0]
    print(f"Initializing {n_init_actual} points. Best y = {float(y_init_pts.min())}")

    max_n = n_init_actual + n_iter
    X_obs = jnp.zeros((max_n, D)).at[:n_init_actual].set(X_init_pts)
    y_obs = jnp.zeros((max_n,)).at[:n_init_actual].set(y_init_pts)
    count = n_init_actual

    history = list(jnp.minimum.accumulate(y_init_pts).tolist())
    fitted_params = None

    for t in range(n_iter):
        mask = jnp.arange(max_n) < count

        y_valid_sum = jnp.sum(jnp.where(mask, y_obs, 0.0))
        y_mean = y_valid_sum / count
        y_var = jnp.sum(jnp.where(mask, (y_obs - y_mean) ** 2, 0.0)) / count
        y_std = jnp.sqrt(y_var) + 1e-8
        y_norm = (y_obs - y_mean) / y_std

        X_obs_norm = (X_obs - lb) / (ub - lb)

        if max_points is not None and count > max_points:
            rank_key = jnp.where(mask, y_obs, jnp.inf)
            keep_idx = jnp.argsort(rank_key)[:max_points]
            X_gp = X_obs_norm[keep_idx]
            y_gp = y_norm[keep_idx]
            gp_mask = jnp.ones((max_points,), dtype=bool)
        else:
            X_gp = X_obs_norm
            y_gp = y_norm
            gp_mask = mask

        _fit_kw = dict(fit_kwargs)
        if t > 0:
            _fit_kw.setdefault("n_restarts", 1)
            _fit_kw.setdefault("max_iter", 50)

        key, subkey = jr.split(key)
        fitted_params = fit(
            X_gp, y_gp, kernel_name=kernel_name,
            init_params_override=fitted_params if t > 0 else None, key=subkey,
            mask=gp_mask, **_fit_kw
        )

        L_chol, alpha = precompute(fitted_params, X_gp, y_gp, kernel_name, mask=gp_mask)

        key, acq_key = jr.split(key)
        best_y_norm = jnp.min(jnp.where(gp_mask, y_gp, jnp.inf))
        dyn_kwargs = {"best_y": best_y_norm, "key": acq_key}
        merged_kwargs = {**acquisition_kwargs, **dyn_kwargs}

        def batch_score_fn(X_norm, _params=fitted_params, _X_obs=X_gp,
                           _L=L_chol, _alpha=alpha, _mask=gp_mask, _kw=merged_kwargs):
            mean, var = predict_precomputed(_params, _X_obs, _L, _alpha, X_norm, kernel_name, mask=_mask)
            return acquisition_fn(mean, var, **_kw)

        def single_score_fn(x_norm, _params=fitted_params, _X_obs=X_gp,
                            _L=L_chol, _alpha=alpha, _mask=gp_mask, _kw=merged_kwargs):
            mean, var = predict_precomputed(_params, _X_obs, _L, _alpha, x_norm[None], kernel_name, mask=_mask)
            return acquisition_fn(mean, var, **_kw).squeeze()

        key, opt_key = jr.split(key)
        x_next_norm = optimize_acquisition(
            batch_score_fn, single_score_fn, D,
            n_restarts=n_restarts, n_candidates=n_candidates, key=opt_key
        )
        X_next = lb + x_next_norm * (ub - lb)
        y_next = objective(X_next)

        X_obs = X_obs.at[count].set(X_next)
        y_obs = y_obs.at[count].set(y_next)
        count += 1

        best_y_so_far = float(jnp.min(jnp.where(jnp.arange(max_n) < count, y_obs, jnp.inf)))
        history.append(best_y_so_far)

        print(
            f"{t+1}/{n_iter} | "
            f"New y = {float(y_next)} | "
            f"Best y = {best_y_so_far}"
        )

    best_idx = int(jnp.argmin(y_obs))

    key, rs_key = jr.split(key)
    random_history = run_random_search(objective, bounds, n_init_actual + n_iter, rs_key)

    return BOResult(
        X_obs = X_obs,
        y_obs = y_obs,
        best_x = X_obs[best_idx],
        best_y = float(y_obs[best_idx]),
        history = jnp.array(history),
        random_history = random_history,
    )
