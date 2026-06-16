import jax
import jax.numpy as jnp
import jax.random as jr


def init_params(key, num_neurons, n_in=2, n_out=2):
    k1, k2 = jr.split(key)
    return {
        "W": jnp.identity(num_neurons),
        "U": jr.normal(k1, (num_neurons, n_in)) * 0.1,
        "V": jr.normal(k2, (n_out, num_neurons)) * 0.1,
        "bias": jnp.zeros(num_neurons),
        "tau": jnp.ones(num_neurons),
    }


def init_state(num_neurons):
    return jnp.zeros(num_neurons)


def step(t, ctrnn_state, plant_state, params):
    h = ctrnn_state
    tau = jax.nn.softplus(params["tau"]) + 0.1
    h_dot = (-h + params["W"] @ jnp.tanh(h + params["bias"])
             + params["U"] @ plant_state) / tau
    u = jnp.tanh(params["V"] @ jnp.tanh(h))
    return h_dot, u
