# Run from main directory using command:
# python -m experiments.pendulum_co.run
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
import gymnax
import matplotlib.pyplot as plt

from experiments.controllers import coupled_osc
from . import config as cfg
from bayesoptax.loop import run_seeds
from bayesoptax.utils import Bounds

env, env_params = gymnax.make("Pendulum-v1")


def rollout(key, osc_params, env_params, steps_in_episode):
    key_reset, key_episode = jr.split(key)
    obs, state = env.reset(key_reset, env_params)
    osc_state = jnp.concatenate([jnp.ones(cfg.N_OSC), jnp.zeros(cfg.N_OSC)])

    def policy_step(carry, _):
        obs, env_state, osc_state, rng = carry
        rng, step_key = jr.split(rng)

        osc_dot, u = coupled_osc.step(0.0, osc_state, obs, osc_params)
        osc_state = osc_state + env_params.dt * osc_dot
        action = jnp.clip(u, -env_params.max_torque, env_params.max_torque)
        next_obs, next_state, reward, done, _ = env.step(step_key, env_state, action, env_params)

        carry = [next_obs, next_state, osc_state, rng]
        return carry, [obs, action, reward, next_obs, done]

    _, scan_out = jax.lax.scan(
        policy_step,
        [obs, state, osc_state, key_episode],
        (),
        steps_in_episode,
    )
    obs, action, reward, next_obs, done = scan_out
    return obs, action, reward, next_obs, done


jit_rollout = jax.jit(rollout, static_argnums=3)


def make_objective(unflatten):
    @jax.jit
    def objective(flat_params):
        osc_params = unflatten(flat_params)
        _, _, reward, _, _ = jit_rollout(jr.PRNGKey(0), osc_params, env_params, env_params.max_steps_in_episode)
        return -jnp.sum(reward)
    return objective


def visualize(osc_params, key):
    obs_traj, actions, rewards, _, _ = jit_rollout(key, osc_params, env_params, env_params.max_steps_in_episode)

    ts = jnp.arange(env_params.max_steps_in_episode) * env_params.dt
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(ts, jnp.arctan2(obs_traj[:, 1], obs_traj[:, 0]))
    axes[0].set_ylabel("angle (rad)")
    axes[1].plot(ts, actions[:, 0])
    axes[1].set_ylabel("torque")
    axes[2].plot(ts, rewards)
    axes[2].set_ylabel("reward")
    axes[2].set_xlabel("time (s)")
    plt.tight_layout()
    plt.show()


def main():
    key = jr.PRNGKey(0)
    key, init_key, bo_key, vis_key = jr.split(key, 4)

    init_params = coupled_osc.init_params(init_key, cfg.N_OSC, n_in=cfg.N_IN, n_out=cfg.N_OUT)
    flat_params, unflatten = ravel_pytree(init_params)
    dim = len(flat_params)
    print(f"Parameter count: {dim}")

    objective = make_objective(unflatten)

    bounds = Bounds.uniform(*cfg.BOUNDS, dim)
    results = run_seeds(
        objective=objective,
        bounds=bounds,
        n_seeds=cfg.N_SEEDS,
        n_init=5 * dim,
        n_iter=cfg.N_ITER,
        kernel_name=cfg.KERNEL,
        acquisition_name=cfg.ACQUISITION,
        base_key=bo_key,
    )

    results.plot()

    best_params = unflatten(results.best_xs[0])
    visualize(best_params, vis_key)


if __name__ == "__main__":
    main()
