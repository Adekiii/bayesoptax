from dataclasses import dataclass, field
import jax
import jax.numpy as jnp


@dataclass
class Bounds:
    """Dataclass for BO bounds."""

    lb: jax.Array
    ub: jax.Array

    def __post_init__(self):
        self.lb = jnp.atleast_1d(jnp.array(self.lb, dtype=float))
        self.ub = jnp.atleast_1d(jnp.array(self.ub, dtype=float))
        assert self.lb.shape == self.ub.shape, "Lower and upper bounds must have the same shape"
        assert jnp.all(self.ub > self.lb), "Upper bounds must be greater than the lower bounds"

    @classmethod
    def uniform(cls, low: float, high: float, D: int):
        return cls(jnp.full(D, low), jnp.full(D, high))
    
    @property
    def D(self):
        return self.lb.shape[0]
    
    def to_array(self) -> jax.Array:
        return jnp.stack([self.lb, self.ub], axis=1)
