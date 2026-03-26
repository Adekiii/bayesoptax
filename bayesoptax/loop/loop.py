import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Callable

from ..surrogates.gp import predict
from ..surrogates.inference import fit
from ..acquisitions import get_acquisition
from .candidates import sample_initial, sample_candidates
from ..utils import Bounds
from .result import BOResult


def run(
        objective: Callable,
        bounds: jax.Array,
        n_init: int = 10,
        n_iter: int = 50,
        n_candidates: int = 1000,
        kernel_name: str ="rbf",
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

    history = []
    fitted_params = None

    for t in range(n_iter):
        y_mean = y_obs.mean()
        y_std = y_obs.std() + 1e-8
        y_norm = (y_obs - y_mean) / y_std

        key, subkey = jr.split(key)
        fitted_params = fit(
            X_obs, y_norm, kernel_name=kernel_name,
            init_params_override=fitted_params if t>0 else None, key=subkey,
            **fit_kwargs
        )

        key, subkey = jr.split(key)

        X_cands = sample_candidates(subkey, bounds, n_candidates)
        mean, var = predict(fitted_params, X_obs, y_norm, X_cands, kernel_name)

        if acquisition_name == "ei":
            scores = acquisition_fn(mean, var, y_norm.min(), **acquisition_kwargs)
        elif acquisition_name == "ts":
            key, ts_key = jr.split(key)
            scores = acquisition_fn(mean, var, ts_key)
        else:
            scores = acquisition_fn(mean, var, **acquisition_kwargs)
        
        x_next = X_cands[jnp.argmax(scores)]
        y_next = objective(x_next)

        X_obs = jnp.concatenate([X_obs, x_next[None]], axis=0)
        y_obs = jnp.concatenate([y_obs, jnp.array([y_next])], axis=0)

        history.append(float(y_obs.min()))

        print(
            f"{t+1}/{n_iter} | "
            f"New y = {float(y_next)} | "
            f"Best y = {float(y_obs.min())}"
        )
    
    best_idx = int(jnp.argmin(y_obs))

    return BOResult(
        X_obs = X_obs,
        y_obs = y_obs,
        best_x = X_obs[best_idx],
        best_y = float(y_obs[best_idx]),
        history = jnp.array(history)
    )