import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
import optax

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


def _run_adam(loss_fn, flat_init, X, y, max_iter, lr):
    """Adam descent."""

    opt = optax.adam(lr)
    opt_state = opt.init(flat_init)

    def step(carry, _):
        x, opt_state = carry
        _, grad = jax.value_and_grad(loss_fn)(x, X, y)
        updates, opt_state = opt.update(grad, opt_state, x)
        x = optax.apply_updates(x, updates)
        return (x, opt_state), None

    (x_final, _), _ = jax.lax.scan(step, (flat_init, opt_state), None, length=max_iter)
    return x_final, loss_fn(x_final, X, y)


def fit(
        X: jax.Array,
        y: jax.Array,
        kernel_name: str,
        n_restarts: int = 3,
        max_iter: int = 200,
        lr: float = 0.05,
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
        max_iter: number of Adam steps.
        lr: Adam learning rate.
        perturbation_scale: scale of perturbation applied to params.
        init_params_override: enables 'warm-starting' from previously fitted params.
        key: jax PRNGKey for reproducibility.

    Returns:
        best_params: dict with the params with the highest LML
                     across all restarts.
    """

    if key is None:
        key = jr.PRNGKey(0)

    D = X.shape[1] if X.ndim > 1 else 1
    fresh_params = init_params(kernel_name, D)
    _, unflatten = ravel_pytree(fresh_params)

    restart_keys = jr.split(key, n_restarts)
    init_flats = jax.vmap(
        lambda k: ravel_pytree(_perturb_params(fresh_params, k, scale=perturbation_scale))[0]
    )(restart_keys)

    if init_params_override is not None:
        override_flat, _ = ravel_pytree(init_params_override)
        init_flats = init_flats.at[0].set(override_flat)

    def loss_fn(flat_params, X, y):
        params = unflatten(flat_params)
        return -log_marginal_likelihood(params, X, y, kernel_name)

    def run_one(flat_init):
        return _run_adam(loss_fn, flat_init, X, y, max_iter, lr)

    all_flat_params, all_losses = jax.vmap(run_one)(init_flats)
    best_idx = jnp.nanargmin(all_losses)
    best_params = unflatten(all_flat_params[best_idx])
    return best_params
