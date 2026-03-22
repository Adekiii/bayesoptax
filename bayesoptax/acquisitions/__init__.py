from .functions import ei, lcb, ts


FUNCTIONS = {
    "ei": ei,
    "lcb": lcb,
    "ts": ts
}

def get_acquisition(name: str):
    if name not in FUNCTIONS:
        raise ValueError(f"{name} is unknown. Available: {list(FUNCTIONS)}")
    return FUNCTIONS[name]