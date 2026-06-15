import jax.numpy as jnp
from .dynamics import R_CYCLE

T0, T1, DT = 0.0, 50.0, 0.05
N_STEPS = int((T1 - T0) / DT)
TS = jnp.linspace(T0, T1, N_STEPS)
X0 = float(1.5 * R_CYCLE)  # start outside the limit cycle
Y0 = 0

CONTROL_WEIGHT = 0.05

N_ROLLOUTS = 25
NOISE_STD = 0.1

N_SEEDS = 10
N_ITER = 200
KERNEL = "matern52"
ACQUISITION = "ts"
BOUNDS = (-3., 3.)
