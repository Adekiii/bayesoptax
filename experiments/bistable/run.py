# Run from main directory using command:
# python -m experiments.bistable.run --controller [linear|ctrnn|coupled_osc]
import argparse
import importlib
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
import diffrax

from .dynamics import dynamics, PLANT_PARAMS
from . import config as cfg
from bayesoptax.loop import run_batched, run_turbo_batched, run_cmaes_seeds, run_random_seeds, plot_comparison, save_run
from bayesoptax.utils import Bounds

PLANT_DIM = 2
N_IN = 2
N_OUT = 2

_noise_key = jr.PRNGKey(99)
NOISE = jr.normal(_noise_key, (cfg.N_ROLLOUTS, cfg.N_STEPS, 2)) * cfg.NOISE_STD

def make_controller_configs(num_neurons, n_osc, n_hidden):
    return {
        "linear": dict(init_kwargs=dict(n_in=N_IN, n_out=N_OUT), state_kwargs=dict()),
        "ctrnn": dict(init_kwargs=dict(num_neurons=num_neurons, n_in=N_IN, n_out=N_OUT), state_kwargs=dict(num_neurons=num_neurons)),
        "coupled_osc":dict(init_kwargs=dict(n_osc=n_osc, n_in=N_IN, n_out=N_OUT), state_kwargs=dict(n_osc=n_osc)),
        "nonlinear": dict(init_kwargs=dict(n_in=N_IN, n_out=N_OUT, n_hidden=n_hidden), state_kwargs=dict()),
    }


def load_controller(name):
    return importlib.import_module(f"experiments.controllers.{name}")


def make_rollout(controller, ctrl_params, state_kwargs, noise):
    init_state = controller.init_state(**state_kwargs)
    z0 = jnp.concatenate([jnp.array([cfg.X0, cfg.Y0]), init_state])

    def joint_system(t, state, args):
        plant_state = state[:PLANT_DIM]
        ctrl_state = state[PLANT_DIM:]
        ctrl_state_dot, u = controller.step(t, ctrl_state, plant_state, args["controller"])
        idx = jnp.clip(jnp.searchsorted(cfg.TS, t, side='right') - 1, 0, cfg.N_STEPS - 1)
        plant_state_dot = dynamics(t, plant_state, u + args["noise"][idx], args["plant"])
        return jnp.concatenate([plant_state_dot, ctrl_state_dot])

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(joint_system),
        diffrax.Tsit5(),
        t0=cfg.T0,
        t1=cfg.T1,
        dt0=cfg.DT,
        y0=z0,
        args={"plant": PLANT_PARAMS, "controller": ctrl_params, "noise": noise},
        saveat=diffrax.SaveAt(ts=cfg.TS),
        max_steps=16 * cfg.N_STEPS,
    )
    return solution.ys


def make_objective(controller, unflatten, state_kwargs):
    @jax.jit
    def objective(flat_params):
        ctrl_params = unflatten(flat_params)

        def single_rollout(noise):
            traj = make_rollout(controller, ctrl_params, state_kwargs, noise)
            plant_traj = traj[:, :PLANT_DIM]
            ctrl_traj = traj[:, PLANT_DIM:]
            us = jax.vmap(
                lambda cs, ps: controller.step(0., cs, ps, ctrl_params)[1]
            )(ctrl_traj, plant_traj)
            state_cost = jnp.mean(plant_traj[:, 0]**2 + plant_traj[:, 1]**2)
            control_cost = jnp.mean(jnp.sum(us**2, axis=1))
            return state_cost + cfg.CONTROL_WEIGHT * control_cost

        costs = jax.vmap(single_rollout)(NOISE)
        cost = jnp.mean(jnp.where(jnp.isnan(costs), 1e6, costs))
        return cost

    return objective


CONTROLLER_NAMES = ["linear", "ctrnn", "coupled_osc", "nonlinear"]


def main(controller_name, num_neurons=2, n_osc=2, n_hidden=8, save_dir="results"):
    controller = load_controller(controller_name)

    key = jr.PRNGKey(0)
    key, init_key, bo_key, turbo_key, cmaes_key, random_key = jr.split(key, 6)

    cfg_ctrl = make_controller_configs(num_neurons, n_osc, n_hidden)[controller_name]
    ctrl_params = controller.init_params(init_key, **cfg_ctrl["init_kwargs"])
    flat_params, unflatten = ravel_pytree(ctrl_params)
    dim = len(flat_params)
    n_init = 50
    n_eval = n_init + cfg.N_ITER
    print(f"Controller: {controller_name} | parameters: {dim} | total evaluations: {n_eval}")

    objective = make_objective(controller, unflatten, cfg_ctrl["state_kwargs"])
    bounds = Bounds.uniform(*cfg.BOUNDS, dim)
    title = f"Epileptic dynamics - {controller_name} (dim={dim})"

    print("\n--- Bayesian Optimisation ---")
    t0 = time.time()
    bo = run_batched(
        objective=objective, bounds=bounds, n_seeds=cfg.N_SEEDS,
        n_init=n_init, n_iter=cfg.N_ITER,
        kernel_name=cfg.KERNEL, acquisition_name=cfg.ACQUISITION,
        base_key=bo_key, max_points=n_init + 200,
    )
    bo_time = time.time() - t0

    print("\n--- TuRBO ---")
    t0 = time.time()
    turbo = run_turbo_batched(
        objective=objective, bounds=bounds, n_seeds=cfg.N_SEEDS,
        n_init=n_init, n_iter=cfg.N_ITER,
        kernel_name=cfg.KERNEL, acquisition_name=cfg.ACQUISITION,
        base_key=turbo_key, max_points=n_init + 200,
    )
    turbo_time = time.time() - t0

    print("\n--- CMA-ES ---")
    t0 = time.time()
    cmaes = run_cmaes_seeds(
        objective=objective, bounds=bounds,
        n_eval=n_eval, n_seeds=cfg.N_SEEDS, base_key=cmaes_key,
    )
    cmaes_time = time.time() - t0

    print("\n--- Random Search ---")
    t0 = time.time()
    random = run_random_seeds(
        objective=objective, bounds=bounds,
        n_eval=n_eval, n_seeds=cfg.N_SEEDS, base_key=random_key,
    )
    random_time = time.time() - t0

    results = {"BO": bo, "TuRBO": turbo, "CMA-ES": cmaes, "Random Search": random}
    plot_comparison(results, title=title)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{save_dir}/bistable_{controller_name}_{timestamp}"
    save_run(run_dir, results, meta={
        "experiment": "bistable",
        "controller": controller_name,
        "num_neurons": num_neurons,
        "n_osc": n_osc,
        "n_hidden": n_hidden,
        "dim": dim,
        "n_seeds": cfg.N_SEEDS,
        "n_init": n_init,
        "n_iter": cfg.N_ITER,
        "n_eval": n_eval,
        "kernel": cfg.KERNEL,
        "acquisition": cfg.ACQUISITION,
        "bounds": list(cfg.BOUNDS),
        "timing": {"bo": bo_time, "turbo": turbo_time, "cmaes": cmaes_time, "random": random_time},
        "timestamp": timestamp,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=CONTROLLER_NAMES, required=True)
    parser.add_argument("--num-neurons", type=int, default=2)
    parser.add_argument("--n-osc", type=int, default=2)
    parser.add_argument("--n-hidden", type=int, default=8)
    parser.add_argument("--save-dir", type=str, default="results")
    args = parser.parse_args()
    main(args.controller, num_neurons=args.num_neurons, n_osc=args.n_osc, n_hidden=args.n_hidden, save_dir=args.save_dir)
