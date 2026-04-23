import jax.numpy as jnp
import jax.random as jr


def init_params(key, n_in, n_out, scale=0.01):
    k1, k2 = jr.split(key)
    return {
        "W": jr.normal(k1, (n_out, n_in)) * scale,
        "b": jr.normal(k2, (n_out,)) * scale,
    }


def init_state():
    return jnp.zeros(0)


def step(t, controller_state, plant_state, params):
    u = params["W"] @ plant_state + params["b"]
    return jnp.zeros(0), u
