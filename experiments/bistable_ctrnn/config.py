import jax.numpy as jnp
from epileptic_bistable import R_CYCLE


NUM_NEURONS = 2

T0, T1, DT = 0.0, 50.0, 0.05
N_STEPS = int((T1 - T0) / DT)
TS = jnp.linspace(T0, T1, N_STEPS)
X0 = float(R_CYCLE)
Y0 = float(R_CYCLE)

CONTROL_WEIGHT = 0.05

N_SEEDS = 10
N_ITER = 100
KERNEL = "matern52"
ACQUISITION = "ts"
BOUNDS = (-5., 5.)
