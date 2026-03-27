# Adapted from:
# Analytical Characterization of Epileptic Dynamics in a Bistable System, Qin et al. 2024
# https://arxiv.org/abs/2404.03409
# Task: drive the dynamics towards the origin (normal brain activity)
import jax
import jax.numpy as jnp


PLANT_PARAMS = {
    "sigma": -0.5,
    "omega": 2.,
    "a": 1.,
    "b": 1.
}

a, b, sigma = PLANT_PARAMS["a"], PLANT_PARAMS["b"], PLANT_PARAMS["sigma"]
gamma0 = jnp.sqrt(a**2 + sigma / b)
R_SEP = jnp.sqrt(a - gamma0) # separatrix radius
R_CYCLE = jnp.sqrt(a + gamma0) # limit cycle radius


def dynamics(t, state, u, params):
    x, y = state[0], state[1]
    a, b, sigma, omega = params["a"], params["b"], params["sigma"], params["omega"]

    # forced bistable dynamics
    dxdt = -omega*y + x*(sigma + 2*a*b*(x**2 + y**2) - b*(x**2 + y**2)**2) + u[0]
    dydt = omega*x + y*(sigma + 2*a*b*(x**2 + y**2) - b*(x**2 + y**2)**2) + u[1]

    return jnp.array([dxdt, dydt])
