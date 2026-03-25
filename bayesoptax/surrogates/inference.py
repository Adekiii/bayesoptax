import numpy as np
import scipy.optimize as scipy_opt

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from functools import partial

from .gp import log_marginal_likelihood, init_params


def _perturb_params(params: dict, key: jax.Array, scale: float = 0.5) -> dict:
    """Adds perturbation to params to ensure diverse starting points in multi-restart."""

    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jr.split(key, len(leaves))
    perturbed = [
        leaf + scale * jr.normal(k, leaf.shape)
        for leaf, k in zip(leaves, keys)
    ]
    return treedef.unflatten(perturbed)


@partial(jax.jit, static_argnames=("kernel_name", "unflatten"))
def _loss_and_grad(
    flat_params: jax.Array,
    unflatten,
    X: jax.Array,
    y: jax.Array,
    kernel_name: str
) -> tuple[jax.Array, jax.Array]:
    """Use negative LML and its gradient w.r.t. params."""

    params = unflatten(flat_params)
    loss, grad = jax.value_and_grad(
        lambda p: -log_marginal_likelihood(p, X, y, kernel_name)
    )(params)
    flat_grad, _ = ravel_pytree(grad)
    return loss, flat_grad


def _fit_single(
        init_p: dict,
        X: jax.Array,
        y: jax.Array,
        kernel_name: str,
        max_iter: int,
        tol: float
) -> tuple[dict, float]:
    """Run L-BFGS-B for a single initialization."""

    flat_init, unflatten = ravel_pytree(init_p)

    def loss_and_grad_np(flat_params_np: np.ndarray):
        flat_params = jnp.array(flat_params_np)
        loss, grad = _loss_and_grad(flat_params, unflatten, X, y, kernel_name)
        return np.array(loss, dtype=np.float64), np.array(grad, dtype=np.float64)
    
    result = scipy_opt.minimize(
        loss_and_grad_np,
        np.array(flat_init, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options = {
            "maxiter": max_iter,
            "ftol": 1e-9,
            "gtol": tol
        }
    )

    optimized_params = unflatten(jnp.array(result.x))
    lml = float(-result.fun)
    return optimized_params, lml


def fit(
        X: jax.Array,
        y: jax.Array,
        kernel_name: str,
        n_restarts: int = 5,
        max_iter: int = 500,
        tol: float = 1e-5,
        perturbation_scale: float = 0.5,
        init_params_override: dict | None = None,
        key: jax.Array | None = None
) -> dict:
    """Returns fitted GP hyperparameters.

    Args:
        X: training inputs of shape [N D].
        y: training targets of shape [N]. Normalize for better results.
        kernel_name: string of the kernel name to use.
        n_restarts: number of random restarts.
        max_iter: max L-BFGS-B iterations.
        tol: norm tolerance for convergence.
        perturbation_scale: scale of perturbation applied to params.
        init_params_override: enables 'warm-starting' from previously fitted params.
        key: jax PRNGKey for reproducibility.

    Returns:
        best_params: dict with the params with the highest LML
                     across all restarts.
    """

    if key is None:
        key = jr.PRNGKey(0)
 
    fresh_params = init_params(kernel_name)

    restart_keys = jr.split(key, n_restarts)
    init_params_list = [
        _perturb_params(fresh_params, k, scale=perturbation_scale)
        for k in restart_keys
    ]
    if init_params_override is not None:
        init_params_list[0] = init_params_override
 
    results = [
        _fit_single(p, X, y, kernel_name, max_iter, tol)
        for p in init_params_list
    ]
 
    all_params, all_lmls = zip(*results)
    best_idx = int(jnp.argmax(jnp.array(all_lmls)))
    best_params = all_params[best_idx]
    return best_params