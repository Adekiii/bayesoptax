import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from etils import epath
from flax import struct


@struct.dataclass
class MjxState:
    pipeline_state: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array


class MjxEnv:
    """Base class: loads an MJCF model and steps it n_frames at a time per action."""

    def __init__(self, xml_path: str, n_frames: int):
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.model = mjx.put_model(self._mj_model)
        self.n_frames = n_frames

    @property
    def dt(self):
        return float(self._mj_model.opt.timestep) * self.n_frames

    @property
    def action_size(self):
        return self.model.nu

    def pipeline_init(self, qpos, qvel):
        data = mjx.make_data(self.model)
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.model, data)
        return data

    def pipeline_step(self, data, action):
        def substep(data, _):
            data = data.replace(ctrl=action)
            data = mjx.step(self.model, data)
            return data, None

        data, _ = jax.lax.scan(substep, data, None, length=self.n_frames)
        return data


class MjxSwimmer(MjxEnv):
    """Matches brax.envs.swimmer.Swimmer's reward/observation definitions."""

    def __init__(
            self, forward_reward_weight=1.0, ctrl_cost_weight=1e-4,
            reset_noise_scale=0.1,
    ):
        xml_path = str(epath.resource_path("brax") / "envs/assets/swimmer.xml")
        super().__init__(xml_path, n_frames=4)
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._observation_size = 8

    @property
    def observation_size(self):
        return self._observation_size

    def reset(self, key):
        k1, k2 = jax.random.split(key)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self._mj_model.qpos0 + jax.random.uniform(k1, (self.model.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(k2, (self.model.nv,), minval=low, maxval=hi)
        data = self.pipeline_init(qpos, qvel)
        return MjxState(pipeline_state=data, obs=self._get_obs(data), reward=jnp.zeros(()), done=jnp.zeros(()))

    def step(self, state, action):
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        x_velocity = (data.qpos[0] - data0.qpos[0]) / self.dt
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        reward = forward_reward - ctrl_cost

        return state.replace(pipeline_state=data, obs=self._get_obs(data), reward=reward)

    def _get_obs(self, data):
        # exclude x/y position (qpos[:2]), matching brax's default
        # exclude_current_positions_from_observation=True
        return jnp.concatenate([data.qpos[2:], data.qvel])


class MjxWalker2d(MjxEnv):
    """Matches brax.envs.walker2d.Walker2d's reward/observation definitions."""

    def __init__(
            self, forward_reward_weight=1.0, ctrl_cost_weight=1e-3,
            healthy_reward=1.0, healthy_z_range=(0.8, 2.0),
            healthy_angle_range=(-1.0, 1.0), reset_noise_scale=5e-3,
    ):
        xml_path = str(epath.resource_path("brax") / "envs/assets/walker2d.xml")
        super().__init__(xml_path, n_frames=4)
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._observation_size = 17
        self._torso_body_id = self._mj_model.body("torso").id

    @property
    def observation_size(self):
        return self._observation_size

    def reset(self, key):
        k1, k2 = jax.random.split(key)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self._mj_model.qpos0 + jax.random.uniform(k1, (self.model.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(k2, (self.model.nv,), minval=low, maxval=hi)
        data = self.pipeline_init(qpos, qvel)
        return MjxState(pipeline_state=data, obs=self._get_obs(data), reward=jnp.zeros(()), done=jnp.zeros(()))

    def step(self, state, action):
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        x_velocity = (data.qpos[0] - data0.qpos[0]) / self.dt
        forward_reward = self._forward_reward_weight * x_velocity

        z = data.xpos[self._torso_body_id, 2]
        angle = data.qpos[2]
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range
        is_healthy = jnp.logical_and(
            jnp.logical_and(z > min_z, z < max_z),
            jnp.logical_and(angle > min_angle, angle < max_angle),
        )
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        reward = forward_reward + self._healthy_reward - ctrl_cost
        done = jnp.where(is_healthy, 0.0, 1.0)

        return state.replace(pipeline_state=data, obs=self._get_obs(data), reward=reward, done=done)

    def _get_obs(self, data):
        position = data.qpos[1:]
        velocity = jnp.clip(data.qvel, -10, 10)
        return jnp.concatenate([position, velocity])


MJX_ENVS = {
    "swimmer": MjxSwimmer,
    "walker2d": MjxWalker2d,
}


def get_mjx_environment(env_name: str):
    if env_name not in MJX_ENVS:
        raise ValueError(f"No MJX environment for '{env_name}'. Available: {list(MJX_ENVS)}")
    return MJX_ENVS[env_name]()
