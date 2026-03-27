# Run from main directory using command:
# python -m experiments.bistable_ctrnn.run
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
import matplotlib.pyplot as plt
import diffrax

from .epileptic_bistable import dynamics, PLANT_PARAMS
from . import ctrnn, config as cfg
from bayesoptax.loop import run_seeds
from bayesoptax.utils import Bounds


def joint_system(t, state, args):
    plant_state = state[:2]
    ctrl_state  = state[2:]

    ctrl_state_dot, u = ctrnn.step(t, ctrl_state, plant_state, args["controller"])
    plant_state_dot = dynamics(t, plant_state, u, args["plant"])

    return jnp.concatenate([plant_state_dot, ctrl_state_dot])


def rollout(ctrnn_params):
    h0 = jnp.zeros(cfg.NUM_NEURONS)
    z0 = jnp.concatenate([jnp.array([cfg.X0, cfg.Y0]), h0])
    args = {"plant": PLANT_PARAMS, "controller": ctrnn_params}

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(joint_system),
        diffrax.Tsit5(),
        t0=cfg.T0,
        t1=cfg.T1,
        dt0=cfg.DT,
        y0=z0,
        args=args,
        saveat=diffrax.SaveAt(ts=cfg.TS),
        max_steps=16*cfg.N_STEPS
    )
    return solution.ys


def make_objective(unflatten):
    @jax.jit
    def objective(flat_params):
        ctrnn_params = unflatten(flat_params)
        traj = rollout(ctrnn_params)

        xs, ys = traj[:, 0], traj[:, 1]
        hs = traj[:, 2:]
        us = jax.vmap(lambda h: jnp.tanh(ctrnn_params["V"] @ jnp.tanh(h)))(hs)

        state_cost = jnp.mean(xs**2 + ys**2)
        control_cost = jnp.mean(jnp.sum(us**2, axis=1))
    
        return state_cost + cfg.CONTROL_WEIGHT * control_cost
    return objective


def main():
    key = jr.PRNGKey(0)
    key, init_key, bo_key = jr.split(key, 3)

    init_ctrl_params = ctrnn.init_params(init_key, cfg.NUM_NEURONS)
    flattened_params, unflatten = ravel_pytree(init_ctrl_params)
    dim = len(flattened_params)

    objective = make_objective(unflatten)
    bounds = Bounds.uniform(*cfg.BOUNDS, dim)

    results = run_seeds(
        objective=objective,
        bounds=bounds,
        n_seeds=cfg.N_SEEDS,
        n_init=2*dim,
        n_iter=cfg.N_ITER,
        kernel_name=cfg.KERNEL,
        acquisition_name=cfg.ACQUISITION,
        base_key=bo_key,
    )

    results.plot()


if __name__ == "__main__":
    main()
