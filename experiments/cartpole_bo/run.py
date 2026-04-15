# Run from main directory using command:
# python -m experiments.cartpole_bo.run
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import optax
import gymnax
import matplotlib.pyplot as plt
from jax.flatten_util import ravel_pytree

from . import config as cfg
from bayesoptax.loop import run_seeds
from bayesoptax.utils import Bounds

env, env_params = gymnax.make("CartPole-v1")
action_list = jnp.array([0, 1.0])
num_actions = len(action_list)


def init_params(key, obs_dim=4, scale=1e-2):
    k1, k2 = jr.split(key)
    W = jr.normal(k1, (obs_dim, num_actions)) * scale
    b = jr.normal(k2, (num_actions,)) * scale
    return (W, b)


def policy(params, obs):
    W, b = params
    logits = jnp.dot(obs, W) + b
    return jnn.softmax(logits)


def get_action(params, obs, key):
    probs = policy(params, obs)
    action_idx = jr.choice(key, jnp.arange(num_actions), p=probs)
    return action_list[action_idx], action_idx


def get_log_prob(params, obs, action_idx):
    probs = policy(params, obs)
    return jnp.log(probs[action_idx])


def rollout(params, env_params, key, steps_in_episode):
    key_reset, key_episode = jr.split(key)
    obs, state = env.reset(key_reset, env_params)

    def policy_step(carry, _):
        obs, state, rng, done = carry
        rng, rng_action, rng_step = jr.split(rng, 3)
        action, action_idx = get_action(params, obs, rng_action)
        next_obs, next_state, reward, next_done, _ = env.step(rng_step, state, action, env_params)

        done = jnp.logical_or(done, next_done)
        reward = reward * (1.0 - done)
        next_obs = jnp.where(done, obs, next_obs)
        next_state = jax.tree.map(lambda x, y: jnp.where(done, x, y), state, next_state)

        carry = [next_obs, next_state, rng, done]
        return carry, [obs, state, action, action_idx, reward, next_obs, done]

    _, scan_out = jax.lax.scan(
        policy_step,
        [obs, state, key_episode, False],
        (),
        length=steps_in_episode,
    )
    return scan_out


jit_rollout = jax.jit(rollout, static_argnums=3)


def make_objective(unflatten):
    @jax.jit
    def objective(flat_params):
        params = unflatten(flat_params)
        obs, _, _, _, reward, _, _ = jit_rollout(
            params, env_params, jr.PRNGKey(0), env_params.max_steps_in_episode
        )
        return -jnp.sum(reward)
    return objective


def visualize(params, key):
    obs, _, action, _, reward, _, done = rollout(
        params, env_params, key, env_params.max_steps_in_episode
    )
    ts = jnp.arange(env_params.max_steps_in_episode) * env_params.tau

    fig, axes = plt.subplots(5, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(ts, obs[:, 0])
    axes[0].set_ylabel("cart pos")
    axes[1].plot(ts, obs[:, 1])
    axes[1].set_ylabel("cart vel")
    axes[2].plot(ts, obs[:, 2])
    axes[2].set_ylabel("pole angle")
    axes[3].plot(ts, action, color="C1")
    axes[3].set_ylabel("action")
    axes[4].plot(ts, reward, color="C2")
    axes[4].set_ylabel("reward")
    axes[4].set_xlabel("time (s)")
    plt.tight_layout()
    plt.show()


def run_reinforce(key, params_init):
    optim = optax.adam(learning_rate=cfg.LR_PG)
    opt_state = optim.init(params_init)

    @jax.jit
    def update_delta(delta, grad_theta):
        updated_delta = jax.tree.map(lambda x, y: x + y, delta, grad_theta)
        return updated_delta, None

    @jax.jit
    def loss_REINFORCE(params, obs, action_idx, reward, baseline):
        def trajectory_gradients(reward, obs, action_idx, baseline, delta):
            G_init = 0

            def step(carry, variables):
                G, delta = carry
                r, o, a_idx, b = variables
                G = cfg.GAMMA_PG * G + r
                A_hat = G - b

                def loss_fn(params):
                    return -get_log_prob(params, o, a_idx).squeeze()

                grad_delta = jax.grad(loss_fn)(params)
                grad_delta = jax.tree.map(lambda gd: gd * A_hat, grad_delta)
                delta, _ = update_delta(delta, grad_delta)
                return (G, delta), G

            variables = (reward[::-1], obs[::-1], action_idx[::-1], baseline[::-1])
            (_, delta), Gt = jax.lax.scan(step, (G_init, delta), variables)
            return delta, Gt

        parallel_traj = jax.vmap(trajectory_gradients, in_axes=(0, 0, 0, None, None))
        delta = jax.tree.map(lambda t: jnp.zeros(t.shape), params)
        deltas, Gs = parallel_traj(reward, obs, action_idx, baseline, delta)
        delta, _ = jax.lax.scan(update_delta, delta, deltas)
        return delta, Gs

    def pg_step(carry, iter_key):
        params, opt_state = carry
        keys = jr.split(iter_key, cfg.N_BATCHES_PG)
        batch_rollout = jax.vmap(rollout, in_axes=(None, None, 0, None))
        obs_b, _, _, action_idx_b, reward_b, _, _ = batch_rollout(
            params, env_params, keys, env_params.max_steps_in_episode
        )
        baseline = jnp.zeros((env_params.max_steps_in_episode,))
        delta, _ = loss_REINFORCE(params, obs_b, action_idx_b, reward_b, baseline)
        updates, opt_state = optim.update(delta, opt_state, params)
        params = optax.apply_updates(params, updates)
        mean_return = jnp.mean(jnp.sum(reward_b, axis=-1))
        return (params, opt_state), mean_return

    iter_keys = jr.split(key, cfg.N_ITERS_PG)
    (params_final, _), history = jax.lax.scan(pg_step, (params_init, opt_state), iter_keys)
    return params_final, history


def plot_comparison(results, history_pg):
    import seaborn as sns
    sns.set_theme()

    n_total_bo = results.histories.shape[1]
    x_bo = jnp.arange(1, n_total_bo + 1)
    bo_returns = -results.histories
    bo_mean = bo_returns.mean(axis=0)
    bo_ci = 1.96 * bo_returns.std(axis=0) / jnp.sqrt(results.n_seeds)

    x_pg = jnp.arange(1, cfg.N_ITERS_PG + 1) * cfg.N_BATCHES_PG

    fig, ax = plt.subplots(figsize=(10, 5))

    bo_color = sns.color_palette()[0]
    pg_color = sns.color_palette()[1]

    for h in bo_returns:
        ax.plot(x_bo, h, alpha=0.1, color=bo_color, linewidth=0.8)
    ax.plot(x_bo, bo_mean, color=bo_color, label=f"BO (n={results.n_seeds} seeds)")
    ax.fill_between(x_bo, bo_mean - bo_ci, bo_mean + bo_ci, alpha=0.25, color=bo_color)

    ax.plot(x_pg, history_pg, color=pg_color, linewidth=0.8, label="REINFORCE")

    ax.axhline(y=499.0, linestyle=":", color="black", linewidth=1.2, label="optimum")

    ax.set_xscale("log")
    ax.set_xlabel("environment rollouts (log scale)")
    ax.set_ylabel("episode return")
    ax.set_title("BO vs REINFORCE: CartPole-v1")
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    key = jr.PRNGKey(0)
    key, init_key, bo_key, pg_key, vis_key_bo, vis_key_pg = jr.split(key, 6)

    init_params_ = init_params(init_key)
    flat_params, unflatten = ravel_pytree(init_params_)
    dim = len(flat_params)
    n_init = 5 * dim
    print(f"Parameter count: {dim}")

    # BayesOpt
    objective = make_objective(unflatten)
    bounds = Bounds.uniform(*cfg.BOUNDS, dim)
    results = run_seeds(
        objective=objective,
        bounds=bounds,
        n_seeds=cfg.N_SEEDS,
        n_init=n_init,
        n_iter=cfg.N_ITER,
        kernel_name=cfg.KERNEL,
        acquisition_name=cfg.ACQUISITION,
        base_key=bo_key,
    )

    # best_params_bo = unflatten(results.best_xs[0])
    # print("BO best trajectory:")
    # visualize(best_params_bo, vis_key_bo)

    # REINFORCE
    key, pg_init_key = jr.split(pg_key)
    params_pg_init = init_params(pg_init_key)
    params_pg, history_pg = run_reinforce(pg_key, params_pg_init)

    # print("\nREINFORCE best trajectory:")
    # visualize(params_pg, vis_key_pg)

    plot_comparison(results, history_pg)


if __name__ == "__main__":
    main()
