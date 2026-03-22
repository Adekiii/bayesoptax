import jax.numpy as jnp
from jax.nn import softplus

from .utils import sq_dist


def rbf(x1, x2, params):
    """Radial Basis Function (RBF) kernel."""

    l = softplus(params["log_lengthscale"])
    var = softplus(params["log_variance"])
    return var * jnp.exp(-0.5 * sq_dist(x1 / l, x2 / l))


def rbf_default_params():
    """Returns default initial parameters for RBF in log scale."""
    
    return {
        "log_lengthscale": jnp.zeros(()),
        "log_variance": jnp.zeros(())
    }