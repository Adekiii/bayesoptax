# Run from main directory using command:
# python -m experiments.brax.run --env [environment_name] --controller [linear|ctrnn|coupled_osc]

# available Brax environments and corresponding (obs_dim, action_dim):
# inverted_pendulum        (4,  1)
# inverted_double_pendulum (8,  1)
# swimmer                  (8,  2)
# reacher                  (11, 2)
# hopper                   (11, 3)
# walker2d                 (17, 6)
# halfcheetah              (17, 6)
# ant                      (27, 8)
import argparse
import importlib
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from brax import envs

from . import config as cfg
from bayesoptax.loop import run_batched, run_turbo_batched, run_cmaes_seeds, run_random_seeds, plot_comparison, save_run
from bayesoptax.utils import Bounds

CONTROLLER_NAMES = ["linear", "ctrnn", "coupled_osc", "nonlinear"]


def make_controller_configs(num_neurons, n_osc, n_hidden, obs_dim, action_dim):
    return {
        "linear": dict(init_kwargs=dict(n_in=obs_dim, n_out=action_dim), state_kwargs=dict()),
        "ctrnn": dict(init_kwargs=dict(num_neurons=num_neurons, n_in=obs_dim, n_out=action_dim), state_kwargs=dict(num_neurons=num_neurons)),
        "coupled_osc":dict(init_kwargs=dict(n_osc=n_osc, n_in=obs_dim, n_out=action_dim), state_kwargs=dict(n_osc=n_osc)),
        "nonlinear": dict(init_kwargs=dict(n_in=obs_dim, n_out=action_dim, n_hidden=n_hidden), state_kwargs=dict()),
    }


def load_controller(name):
    return importlib.import_module(f"experiments.controllers.{name}")


def make_rollout(controller, ctrl_params, state_kwargs, env):
    @jax.jit
    def rollout(key):
        env_state = env.reset(key)
        ctrl_state = controller.init_state(**state_kwargs)

        def step_fn(carry, _):
            env_state, ctrl_state, done_acc = carry

            safe_obs = jnp.where(done_acc, jnp.zeros_like(env_state.obs), env_state.obs)
            ctrl_state_dot, u = controller.step(0., ctrl_state, safe_obs, ctrl_params)
            new_ctrl_state = ctrl_state + env.dt * ctrl_state_dot
            action = jnp.clip(u, -1.0, 1.0)

            next_state = env.step(env_state, action)
            done_acc = jnp.logical_or(done_acc, next_state.done.astype(bool))

            reward = jnp.where(done_acc, 0.0, next_state.reward)
            new_ctrl_state = jnp.where(done_acc, ctrl_state, new_ctrl_state)

            return (next_state, new_ctrl_state, done_acc), reward

        _, rewards = jax.lax.scan(
            step_fn, (env_state, ctrl_state, False), None, length=cfg.N_STEPS
        )
        return jnp.sum(rewards)

    return rollout


def make_objective(controller, unflatten, state_kwargs, env):
    @jax.jit
    def objective(flat_params):
        ctrl_params = unflatten(flat_params)
        rollout = make_rollout(controller, ctrl_params, state_kwargs, env)
        r = rollout(jr.PRNGKey(0))

        return -jnp.where(jnp.isnan(r), 0.0, r)

    return objective


def main(env_name, controller_name, num_neurons=4, n_osc=4, n_hidden=8, save_dir="results"):
    env = envs.get_environment(env_name)
    obs_dim = env.observation_size
    action_dim = env.action_size
    print(f"Env: {env_name} | obs_dim={obs_dim} | action_dim={action_dim}")

    controller = load_controller(controller_name)

    key = jr.PRNGKey(0)
    key, init_key, bo_key, turbo_key, cmaes_key, random_key = jr.split(key, 6)

    cfg_ctrl = make_controller_configs(num_neurons, n_osc, n_hidden, obs_dim, action_dim)[controller_name]
    ctrl_params = controller.init_params(init_key, **cfg_ctrl["init_kwargs"])
    flat_params, unflatten = ravel_pytree(ctrl_params)
    dim = len(flat_params)
    n_init = 50
    n_eval = n_init + cfg.N_ITER
    print(f"Controller: {controller_name} | parameters: {dim} | total evaluations: {n_eval}")

    objective = make_objective(controller, unflatten, cfg_ctrl["state_kwargs"], env)
    bounds = Bounds.uniform(*cfg.BOUNDS, dim)
    title = f"Brax {env_name} - {controller_name} (dim={dim})"

    print("\n--- Bayesian Optimisation ---")
    t0 = time.time()
    bo = run_batched(
        objective=objective, bounds=bounds, n_seeds=cfg.N_SEEDS,
        n_init=n_init, n_iter=cfg.N_ITER,
        kernel_name=cfg.KERNEL, acquisition_name=cfg.ACQUISITION,
        base_key=bo_key,
    )
    bo_time = time.time() - t0

    print("\n--- TuRBO ---")
    t0 = time.time()
    turbo = run_turbo_batched(
        objective=objective, bounds=bounds, n_seeds=cfg.N_SEEDS,
        n_init=n_init, n_iter=cfg.N_ITER,
        kernel_name=cfg.KERNEL, acquisition_name=cfg.ACQUISITION,
        base_key=turbo_key,
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
    run_dir = f"{save_dir}/brax_{env_name}_{controller_name}_{timestamp}"
    save_run(run_dir, results, meta={
        "experiment": "brax",
        "env_name": env_name,
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
    parser.add_argument("--env", type=str, default="swimmer")
    parser.add_argument("--controller", choices=CONTROLLER_NAMES, required=True)
    parser.add_argument("--num-neurons", type=int, default=4)
    parser.add_argument("--n-osc", type=int, default=4)
    parser.add_argument("--n-hidden", type=int, default=8)
    parser.add_argument("--save-dir", type=str, default="results")
    args = parser.parse_args()
    main(args.env, args.controller, num_neurons=args.num_neurons, n_osc=args.n_osc, n_hidden=args.n_hidden, save_dir=args.save_dir)
