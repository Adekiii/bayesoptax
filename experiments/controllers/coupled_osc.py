# Adapted from:
# Generative Modeling of Neural Dynamics via Latent Stochastic Differential Equations, ElGazzar. 2024
# https://arxiv.org/abs/2412.12112
import jax.numpy as jnp
import jax.random as jr


def init_params(key, n_osc, n_in=2, n_out=2):
    k1, k2, k3 = jr.split(key, 3)
    return {
        "alpha": jnp.ones(n_osc) * 0.1,
        "omega": jr.normal(k1, (n_osc,)),
        "W_kappa": jr.normal(k2, (n_osc, n_in)) * 0.1,
        "b_kappa": jnp.zeros(n_osc),
        "V": jr.normal(k3, (n_out, 2 * n_osc)) * 0.1,
    }


def init_state(n_osc):
    # unit amplitude, zero phase
    return jnp.concatenate([jnp.ones(n_osc), jnp.zeros(n_osc)])


def step(t, osc_state, plant_state, params):
    n = len(osc_state) // 2
    a, b = osc_state[:n], osc_state[n:]

    alpha = params["alpha"]
    omega = params["omega"]
    kappa = jnp.tanh(params["W_kappa"] @ plant_state + params["b_kappa"])

    r_sq = a**2 + b**2

    da = alpha * a - omega * b - r_sq * a + kappa * a
    db = omega * a + alpha * b - r_sq * b + kappa * b

    state_dot = jnp.concatenate([da, db])
    u = params["V"] @ osc_state
    return state_dot, u
