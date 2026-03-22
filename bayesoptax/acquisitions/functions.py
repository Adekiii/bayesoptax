import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import norm


def ei(
    mean: jax.Array, var: jax.Array, best_y: jax.Array, xi: float=0.01) -> jax.Array:
    """Expected Improvement (EI) acquisition function."""

    sigma = jnp.sqrt(jnp.clip(var, min=1e-9))
    Z = (best_y - mean - xi) / sigma
    return (best_y - mean - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)


def lcb(
    mean: jax.Array, var: jax.Array, beta: float=2.0) -> jax.Array:
    """Lower Confidence Bound (LCB) acquisition function."""

    return -(mean - jnp.sqrt(beta) * jnp.sqrt(jnp.clip(var, min=1e-9)))


def ts(
    mean: jax.Array, var: jax.Array, key: jax.Array) -> jax.Array:
    """Thompson Sampling (TS) acquisition function."""

    std = jnp.sqrt(jnp.clip(var, min=1e-9))
    return -(mean + std * jr.normal(key, shape=mean.shape))