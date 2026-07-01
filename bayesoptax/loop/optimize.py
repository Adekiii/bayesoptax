import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from typing import Callable


def optimize_acquisition(
        batch_score_fn: Callable,
        single_score_fn: Callable,
        D: int,
        n_restarts: int = 3,
        n_candidates: int = 512,
        max_iter: int = 100,
        lr: float = 0.05,
        key: jax.Array | None = None,
        region_bounds: tuple[jax.Array, jax.Array] | None = None,
        tr_center: jax.Array | None = None,
) -> jax.Array:
    """Optimizes acquisition using multi-start gradient ascent."""

    cand_key, mask_key, forced_key = jr.split(key if key is not None else jr.PRNGKey(0), 3)
    raw = jr.uniform(cand_key, (n_candidates, D))

    if region_bounds is not None:
        lb_r, ub_r = region_bounds
        pert = lb_r + raw * (ub_r - lb_r)

        if tr_center is not None:
            prob_perturb = min(20.0 / D, 1.0)
            mask = jr.bernoulli(mask_key, prob_perturb, (n_candidates, D))
            # ensure every candidate perturbs at least one (random) dimension
            forced_dims = jr.randint(forced_key, (n_candidates,), 0, D)
            mask = mask | jax.nn.one_hot(forced_dims, D, dtype=bool)
            X_cands_norm = jnp.where(mask, pert, tr_center[None, :])
        else:
            X_cands_norm = pert

        lb_bounds, ub_bounds = lb_r, ub_r
    else:
        X_cands_norm = raw
        lb_bounds = jnp.zeros(D)
        ub_bounds = jnp.ones(D)

    scores = batch_score_fn(X_cands_norm)
    top_idx = jnp.argsort(scores)[-n_restarts:]
    X_starts = X_cands_norm[top_idx]

    def neg_score(x):
        return -single_score_fn(x)

    opt = optax.adam(lr)

    def run_one(x0):
        opt_state = opt.init(x0)

        def step(carry, _):
            x, opt_state = carry
            _, grad = jax.value_and_grad(neg_score)(x)
            updates, opt_state = opt.update(grad, opt_state, x)
            x = jnp.clip(optax.apply_updates(x, updates), lb_bounds, ub_bounds)
            return (x, opt_state), None

        (x_final, _), _ = jax.lax.scan(step, (x0, opt_state), None, length=max_iter)
        return x_final, neg_score(x_final)

    candidates, losses = jax.vmap(run_one)(X_starts)
    best_idx = jnp.argmin(losses)
    return candidates[best_idx]
