from .rbf import rbf, rbf_default_params
from .matern import matern32, matern52, matern_default_params


KERNELS = {
    "rbf": (rbf, rbf_default_params),
    "matern32": (matern32, matern_default_params),
    "matern52": (matern52, matern_default_params)
}

def get_kernel(name: str):
    if name not in KERNELS:
        raise ValueError(f"{name} is unknown. Available: {list(KERNELS)}")
    return KERNELS[name]