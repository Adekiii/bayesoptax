import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Callable

from ..surrogates.gp import predict
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
        n_candidates: int = 2048,
        n_restarts: int = 5,
        kernel_name: str = "rbf",
        acquisition_name: str = "ei",
        acquisition_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        X_init: jax.Array | None = None,
        y_init: jax.Array | None = None,
        key: jax.Array | None = None
) -> BOResult:
    """Run BO.

    TO-DO: add documentation
    """

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
        X_obs = X_init
        y_obs = y_init
    elif (X_init is not None) or (y_init is not None):
        raise ValueError("Provide both X_init and y_init (or neither).")
    else:
        key, subkey = jr.split(key)
        X_obs = sample_initial(subkey, bounds, n_init)
        y_obs = jax.vmap(objective)(X_obs)

    print(f"Initializing {n_init} points. Best y = {float(y_obs.min())}")

    init_history = jnp.minimum.accumulate(y_obs).tolist()
    history = list(init_history)
    fitted_params = None

    for t in range(n_iter):
        y_mean = y_obs.mean()
        y_std = y_obs.std() + 1e-8
        y_norm = (y_obs - y_mean) / y_std

        X_obs_norm = (X_obs - lb) / (ub - lb)

        key, subkey = jr.split(key)
        fitted_params = fit(
            X_obs_norm, y_norm, kernel_name=kernel_name,
            init_params_override=fitted_params if t>0 else None, key=subkey,
            **fit_kwargs
        )

        key, acq_key = jr.split(key)
        dyn_kwargs = {"best_y": float(y_norm.min()), "key": acq_key}
        merged_kwargs = {**acquisition_kwargs, **dyn_kwargs}

        def batch_score_fn(X_norm, _params=fitted_params, _X_obs=X_obs_norm,
                           _y=y_norm, _kw=merged_kwargs):
            mean, var = predict(_params, _X_obs, _y, X_norm, kernel_name)
            return acquisition_fn(mean, var, **_kw)

        def single_score_fn(x_norm, _params=fitted_params, _X_obs=X_obs_norm,
                            _y=y_norm, _kw=merged_kwargs):
            mean, var = predict(_params, _X_obs, _y, x_norm[None], kernel_name)
            return acquisition_fn(mean, var, **_kw).squeeze()

        key, opt_key = jr.split(key)
        x_next_norm = optimize_acquisition(
            batch_score_fn, single_score_fn, D,
            n_restarts=n_restarts, n_candidates=n_candidates, key=opt_key
        )
        X_next = lb + x_next_norm * (ub - lb)
        y_next = objective(X_next)

        X_obs = jnp.concatenate([X_obs, X_next[None]], axis=0)
        y_obs = jnp.concatenate([y_obs, jnp.array([y_next])], axis=0)

        history.append(float(y_obs.min()))

        print(
            f"{t+1}/{n_iter} | "
            f"New y = {float(y_next)} | "
            f"Best y = {float(y_obs.min())}"
        )

    best_idx = int(jnp.argmin(y_obs))

    key, rs_key = jr.split(key)
    random_history = run_random_search(objective, bounds, n_init + n_iter, rs_key)

    return BOResult(
        X_obs = X_obs,
        y_obs = y_obs,
        best_x = X_obs[best_idx],
        best_y = float(y_obs[best_idx]),
        history = jnp.array(history),
        random_history = random_history,
    )