import jax
import jax.numpy as jnp
from jax.nn import softplus
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.numpy.linalg import cholesky
from functools import partial

from ..kernels.utils import kernel_matrix
from ..kernels import get_kernel


JITTER = 1e-6

def init_params(kernel_name: str, D: int = 1) -> dict:
    """Get initial default parameters for corresponding kernel."""

    _, default_params_fn = get_kernel(kernel_name)
    kernel_params = default_params_fn(D)

    return {
        "kernel": kernel_params,
        "log_noise": jnp.zeros(())
    }


def _masked_kernel_matrix(kernel_fn, X, params_kernel, noise, mask):
    """Kernel matrix with rows/cols for masked points."""

    N = X.shape[0]
    K = kernel_matrix(kernel_fn, X, X, params_kernel)

    if mask is None:
        return K + noise * jnp.eye(N)

    mask_2d = mask[:, None] & mask[None, :]
    K = jnp.where(mask_2d, K, 0.0)
    diag = jnp.where(mask, noise, 1.0)
    return K + jnp.diag(diag)


@partial(jax.jit, static_argnames=("kernel_name",))
def log_marginal_likelihood(
    params: dict,
    X: jax.Array,
    y: jax.Array,
    kernel_name: str,
    mask: jax.Array | None = None,
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
        mask: optional boolean array of shape [N], True for real observations
            and False for padding rows in a fixed-size buffer. If None, all
            N rows are treated as real observations.

    Returns:
        lml: resulting log marginal likelihood as a scalar value.
    """

    kernel_fn, _ = get_kernel(kernel_name)
    N = X.shape[0]

    noise = softplus(params["log_noise"]) + JITTER
    K_noisy = _masked_kernel_matrix(kernel_fn, X, params["kernel"], noise, mask)

    y_fit = y if mask is None else jnp.where(mask, y, 0.0)
    n_eff = N if mask is None else jnp.sum(mask)

    L = cholesky(K_noisy)
    alpha = cho_solve((L, True), y_fit)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

    lml = (
        - 0.5 * jnp.dot(y_fit, alpha)
        - 0.5 * log_det
        - 0.5 * n_eff * jnp.log(2.0 * jnp.pi)
    )
    return lml


@partial(jax.jit, static_argnames=("kernel_name",))
def precompute(
    params: dict,
    X_train: jax.Array,
    y_train: jax.Array,
    kernel_name: str,
    mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Factor the training kernel matrix once for reuse during acquisition optimization."""

    kernel_fn, _ = get_kernel(kernel_name)
    noise = softplus(params["log_noise"]) + JITTER
    K_noisy = _masked_kernel_matrix(kernel_fn, X_train, params["kernel"], noise, mask)

    y_fit = y_train if mask is None else jnp.where(mask, y_train, 0.0)

    L = cholesky(K_noisy)
    alpha = cho_solve((L, True), y_fit)
    return L, alpha


@partial(jax.jit, static_argnames=("kernel_name",))
def predict_precomputed(
    params: dict,
    X_train: jax.Array,
    L: jax.Array,
    alpha: jax.Array,
    X_test: jax.Array,
    kernel_name: str,
    mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Predict using pre-factored Cholesky"""

    kernel_fn, _ = get_kernel(kernel_name)
    K_s = kernel_matrix(kernel_fn, X_train, X_test, params["kernel"])
    if mask is not None:
        K_s = jnp.where(mask[:, None], K_s, 0.0)
    K_ss_diag = jax.vmap(lambda x: kernel_fn(x, x, params["kernel"]))(X_test)
    v = solve_triangular(L, K_s, lower=True)
    mean = K_s.T @ alpha
    var = jnp.clip(K_ss_diag - jnp.sum(v**2, axis=0), min=0.0)
    return mean, var


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
    var = jnp.clip(var, min=0.0)

    return mean, var