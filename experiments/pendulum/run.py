# Run from main directory using command:
# python -m experiments.pendulum.run --controller [linear|ctrnn|coupled_osc]
import argparse
import importlib

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
import gymnax

from . import config as cfg
from bayesoptax.loop import run_seeds, run_cmaes_seeds, run_random_seeds, plot_comparison
from bayesoptax.utils import Bounds

env, env_params = gymnax.make("Pendulum-v1")
N_IN = 3 # obs: cos_theta, sin_theta, theta_dot
N_OUT = 1 # torque (continuous)

CONTROLLER_NAMES = ["linear", "ctrnn", "coupled_osc"]


def make_controller_configs(num_neurons, n_osc):
    return {
        "linear": dict(init_kwargs=dict(n_in=N_IN, n_out=N_OUT), state_kwargs=dict()),
        "ctrnn": dict(init_kwargs=dict(num_neurons=num_neurons, n_in=N_IN, n_out=N_OUT), state_kwargs=dict(num_neurons=num_neurons)),
        "coupled_osc": dict(init_kwargs=dict(n_osc=n_osc, n_in=N_IN, n_out=N_OUT), state_kwargs=dict(n_osc=n_osc)),
    }


def load_controller(name):
    return importlib.import_module(f"experiments.controllers.{name}")


def make_rollout(controller, ctrl_params, state_kwargs):
    @jax.jit
    def rollout(key):
        key_reset, key_ep = jr.split(key)
        obs, env_state = env.reset(key_reset, env_params)
        ctrl_state = controller.init_state(**state_kwargs)

        def policy_step(carry, _):
            obs, env_state, ctrl_state, rng = carry
            rng, step_key = jr.split(rng)

            ctrl_state_dot, u = controller.step(0., ctrl_state, obs, ctrl_params)
            new_ctrl_state = ctrl_state + env_params.dt * ctrl_state_dot
            action = jnp.clip(u, -env_params.max_torque, env_params.max_torque)

            next_obs, next_env_state, reward, _, _ = env.step(
                step_key, env_state, action, env_params
            )
            return (next_obs, next_env_state, new_ctrl_state, rng), reward

        _, rewards = jax.lax.scan(
            policy_step,
            (obs, env_state, ctrl_state, key_ep),
            (),
            length=env_params.max_steps_in_episode,
        )
        return jnp.sum(rewards)

    return rollout


def make_objective(controller, unflatten, state_kwargs):
    @jax.jit
    def objective(flat_params):
        ctrl_params = unflatten(flat_params)
        rollout = make_rollout(controller, ctrl_params, state_kwargs)
        return -rollout(jr.PRNGKey(0))

    return objective


def main(controller_name, num_neurons=2, n_osc=2):
    controller = load_controller(controller_name)

    key = jr.PRNGKey(0)
    key, init_key, bo_key, cmaes_key, random_key = jr.split(key, 5)

    cfg_ctrl = make_controller_configs(num_neurons, n_osc)[controller_name]
    ctrl_params = controller.init_params(init_key, **cfg_ctrl["init_kwargs"])
    flat_params, unflatten = ravel_pytree(ctrl_params)
    dim = len(flat_params)
    n_init = 2 * dim
    n_eval = n_init + cfg.N_ITER
    print(f"Controller: {controller_name} | parameters: {dim} | total evaluations: {n_eval}")

    objective = make_objective(controller, unflatten, cfg_ctrl["state_kwargs"])
    bounds = Bounds.uniform(*cfg.BOUNDS, dim)
    title = f"Pendulum - {controller_name} (dim={dim})"

    print("\n--- Bayesian Optimisation ---")
    bo = run_seeds(
        objective=objective, bounds=bounds, n_seeds=cfg.N_SEEDS,
        n_init=n_init, n_iter=cfg.N_ITER,
        kernel_name=cfg.KERNEL, acquisition_name=cfg.ACQUISITION,
        base_key=bo_key,
    )

    print("\n--- CMA-ES ---")
    cmaes = run_cmaes_seeds(
        objective=objective, bounds=bounds,
        n_eval=n_eval, n_seeds=cfg.N_SEEDS, base_key=cmaes_key,
    )

    print("\n--- Random Search ---")
    random = run_random_seeds(
        objective=objective, bounds=bounds,
        n_eval=n_eval, n_seeds=cfg.N_SEEDS, base_key=random_key,
    )

    plot_comparison({"BO": bo, "CMA-ES": cmaes, "Random Search": random}, title=title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=CONTROLLER_NAMES, required=True)
    parser.add_argument("--num-neurons", type=int, default=2)
    parser.add_argument("--n-osc", type=int, default=2)
    args = parser.parse_args()
    main(args.controller, num_neurons=args.num_neurons, n_osc=args.n_osc)
