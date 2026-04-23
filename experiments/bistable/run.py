# Run from main directory using command:
# python -m experiments.bistable.run --controller [linear|ctrnn|coupled_osc]
import argparse
import importlib

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
import diffrax

from .dynamics import dynamics, PLANT_PARAMS
from . import config as cfg
from bayesoptax.loop import run_seeds
from bayesoptax.utils import Bounds

PLANT_DIM = 2
N_IN = 2
N_OUT = 2

def make_controller_configs(num_neurons, n_osc):
    return {
        "linear": dict(init_kwargs=dict(n_in=N_IN, n_out=N_OUT), state_kwargs=dict()),
        "ctrnn": dict(init_kwargs=dict(num_neurons=num_neurons, n_in=N_IN, n_out=N_OUT), state_kwargs=dict(num_neurons=num_neurons)),
        "coupled_osc": dict(init_kwargs=dict(n_osc=n_osc, n_in=N_IN, n_out=N_OUT), state_kwargs=dict(n_osc=n_osc)),
    }


def load_controller(name):
    return importlib.import_module(f"experiments.controllers.{name}")


def make_rollout(controller, ctrl_params, state_kwargs):
    init_state = controller.init_state(**state_kwargs)
    z0 = jnp.concatenate([jnp.array([cfg.X0, cfg.Y0]), init_state])

    def joint_system(t, state, args):
        plant_state = state[:PLANT_DIM]
        ctrl_state = state[PLANT_DIM:]
        ctrl_state_dot, u = controller.step(t, ctrl_state, plant_state, args["controller"])
        plant_state_dot = dynamics(t, plant_state, u, args["plant"])
        return jnp.concatenate([plant_state_dot, ctrl_state_dot])

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(joint_system),
        diffrax.Tsit5(),
        t0=cfg.T0,
        t1=cfg.T1,
        dt0=cfg.DT,
        y0=z0,
        args={"plant": PLANT_PARAMS, "controller": ctrl_params},
        saveat=diffrax.SaveAt(ts=cfg.TS),
        max_steps=16 * cfg.N_STEPS,
    )
    return solution.ys


def make_objective(controller, unflatten, state_kwargs):
    @jax.jit
    def objective(flat_params):
        ctrl_params = unflatten(flat_params)
        traj = make_rollout(controller, ctrl_params, state_kwargs)

        plant_traj = traj[:, :PLANT_DIM]
        ctrl_traj = traj[:, PLANT_DIM:]

        us = jax.vmap(
            lambda cs, ps: controller.step(0., cs, ps, ctrl_params)[1]
        )(ctrl_traj, plant_traj)

        state_cost = jnp.mean(plant_traj[:, 0]**2 + plant_traj[:, 1]**2)
        control_cost = jnp.mean(jnp.sum(us**2, axis=1))
        cost = state_cost + cfg.CONTROL_WEIGHT * control_cost
        return jnp.where(jnp.isnan(cost), 1e6, cost)

    return objective


CONTROLLER_NAMES = ["linear", "ctrnn", "coupled_osc"]


def main(controller_name, num_neurons=2, n_osc=2):
    controller = load_controller(controller_name)

    key = jr.PRNGKey(0)
    key, init_key, bo_key = jr.split(key, 3)

    cfg_ctrl = make_controller_configs(num_neurons, n_osc)[controller_name]
    ctrl_params = controller.init_params(init_key, **cfg_ctrl["init_kwargs"])
    flat_params, unflatten = ravel_pytree(ctrl_params)
    dim = len(flat_params)
    print(f"Controller: {controller_name} | parameters: {dim}")

    objective = make_objective(controller, unflatten, cfg_ctrl["state_kwargs"])
    bounds = Bounds.uniform(*cfg.BOUNDS, dim)

    results = run_seeds(
        objective=objective,
        bounds=bounds,
        n_seeds=cfg.N_SEEDS,
        n_init=2 * dim,
        n_iter=cfg.N_ITER,
        kernel_name=cfg.KERNEL,
        acquisition_name=cfg.ACQUISITION,
        base_key=bo_key,
    )

    results.plot(title=f"Epileptic dynamics - controller: {controller_name} - dim: {dim}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=CONTROLLER_NAMES, required=True)
    parser.add_argument("--num-neurons", type=int, default=2)
    parser.add_argument("--n-osc", type=int, default=2)
    args = parser.parse_args()
    main(args.controller, num_neurons=args.num_neurons, n_osc=args.n_osc)
