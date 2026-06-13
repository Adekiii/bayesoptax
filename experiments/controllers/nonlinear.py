import jax.numpy as jnp
import jax.random as jr


def init_params(key, n_in, n_out, n_hidden=8, scale=0.1):
    k1, k2, k3 = jr.split(key, 3)
    return {
        "W1": jr.normal(k1, (n_hidden, n_in)) * scale,
        "b1": jnp.zeros(n_hidden),
        "W2": jr.normal(k2, (n_hidden, n_hidden)) * scale,
        "b2": jnp.zeros(n_hidden),
        "W_out": jr.normal(k3, (n_out, n_hidden)) * scale,
        "b_out": jnp.zeros(n_out),
    }


def init_state():
    return jnp.zeros(0)


def step(t, controller_state, plant_state, params):
    x = jnp.tanh(params["W1"] @ plant_state + params["b1"])
    x = jnp.tanh(params["W2"] @ x + params["b2"])
    u = params["W_out"] @ x + params["b_out"]
    return jnp.zeros(0), u
