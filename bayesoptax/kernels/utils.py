import jax
import jax.numpy as jnp


def sq_dist(x1: jax.Array, x2: jax.Array) -> jax.Array:
    """Squared Euclidean distance: ||x1 - x2||^2"""

    return jnp.dot(x1 - x2, x1 - x2)


def euclidean_dist(x1: jax.Array, x2: jax.Array) -> jax.Array:
    """Euclidean distance: ||x1 - x2||"""

    return jnp.sqrt(sq_dist(x1, x2) + 1e-12)


def scaled_euclidean_dist(x1: jax.Array, x2: jax.Array, l: jax.Array) -> jax.Array:
    """Euclidean distance divided by lengthscale: ||x1 - x2|| / l"""

    return euclidean_dist(x1 / l, x2 / l)


def kernel_matrix(kernel_fn, X1: jax.Array, X2: jax.Array, params: dict) -> jax.Array:
    """Computes the [N M] kernel matrix from X1 [N D] and X2 [M D]."""
    
    return jax.vmap(
        lambda x1: jax.vmap(
            lambda x2: kernel_fn(x1, x2, params)
        )(X2)
    )(X1)