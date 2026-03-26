import jax.numpy as jnp
from jax.nn import softplus

from .utils import scaled_euclidean_dist


def matern32(x1, x2, params):
    """Matérn 3/2 kernel."""

    l = softplus(params["log_lengthscale"])
    var = softplus(params["log_variance"])
    r = scaled_euclidean_dist(x1, x2, l)
    sr = jnp.sqrt(3.0) * r
    return var * (1.0 + sr) * jnp.exp(-sr)


def matern52(x1, x2, params):
    """Matérn 5/2 kernel."""

    l = softplus(params["log_lengthscale"])
    var = softplus(params["log_variance"])
    r = scaled_euclidean_dist(x1, x2, l)
    sr = jnp.sqrt(5.0) * r
    return var * (1.0 + sr + (5.0 / 3.0) * r**2) * jnp.exp(-sr)


def matern_default_params(D: int = 1):
    """Returns default initial parameters for the Matern kernels in log scale."""

    return {
        "log_lengthscale": jnp.zeros(D),
        "log_variance": jnp.zeros(())
    }