import jax
import jax.numpy as jnp
from jax.nn import softplus
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.numpy.linalg import cholesky
from functools import partial

from ..kernels.utils import kernel_matrix
from ..kernels import get_kernel


JITTER = 1e-6

def init_params(kernel_name: str) -> dict:
    """Get initial default parameters for corresponding kernel."""

    _, default_params_fn = get_kernel(kernel_name)
    kernel_params = default_params_fn()

    return {
        "kernel": kernel_params,
        "log_noise": jnp.zeros(())
    }


@partial(jax.jit, static_argnames=("kernel_name",))
def log_marginal_likelihood(
    params: dict,
    X: jax.Array,
    y: jax.Array,
    kernel_name: str
) -> jax.Array:
    """Returns the log marginal likelihood of the GP given training data.

    Calculated by:
    log p(y | X, params) =
        - 0.5 * y^T (K + sigma^2 I)^-1 y
        - 0.5 * log |K + sigma^2 I|
        - n/2 * log(2 pi)

    Args:
        params: dict with keys "kernel" and "log_noise", corresponding to
                    kernel hyperparams and noise parameter.
        X: training inputs of shape [N D].
        y: training targets of shape [N].
        kernel_name: string of the kernel name to use.

    Returns:
        lml: resulting log marginal likelihood as a scalar value.
    """

    kernel_fn, _ = get_kernel(kernel_name)
    N = X.shape[0]

    noise = softplus(params["log_noise"]) + JITTER

    K = kernel_matrix(kernel_fn, X, X, params["kernel"])
    K_noisy = K + noise * jnp.eye(N)
    L = cholesky(K_noisy)
    alpha = cho_solve((L, True), y)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

    lml = (
        - 0.5 * jnp.dot(y, alpha)
        - 0.5 * log_det
        - 0.5 * N * jnp.log(2.0 * jnp.pi)
    )
    return lml


@partial(jax.jit, static_argnames=("kernel_name",))
def predict(
    params: dict,
    X_train: jax.Array,
    y_train: jax.Array,
    X_test: jax.Array,
    kernel_name: str
) -> tuple[jax.Array, jax.Array]:
    """Returns a prediction at test points given training data.
    
    Computes the posterior mean and variance analytically.

    Args:
        params: dict with fitted params.
        X_train: training inputs of shape [N D].
        y_train: training targets of shape [N].
        X_test: test inputs of shape [M D]
        kernel_name: string of the kernel name to use.

    Returns:
        mean: predictive mean of shape [M]
        var: predictive variance of shape [M]
    """

    kernel_fn, _ = get_kernel(kernel_name)
    N = X_train.shape[0]

    noise = softplus(params["log_noise"]) + JITTER

    K = kernel_matrix(kernel_fn, X_train, X_train, params["kernel"])
    K_s = kernel_matrix(kernel_fn, X_train, X_test, params["kernel"])
    K_ss_diag = jax.vmap(
        lambda x: kernel_fn(x, x, params["kernel"])
    )(X_test)

    K_noisy = K + noise * jnp.eye(N)
    L = cholesky(K_noisy)
    alpha = cho_solve((L, True), y_train)
    v = solve_triangular(L, K_s, lower=True)

    mean = K_s.T @ alpha
    var = K_ss_diag - jnp.sum(v ** 2, axis=0)
    var = jnp.clip(var, a_min=0.0)

    return mean, var