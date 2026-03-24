from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


@dataclass
class BOResult:
    """Dataclass for BO loop results."""

    X_obs: jax.Array
    y_obs: jax.Array
    best_x: jax.Array
    best_y: float
    history: jax.Array

    def plot(self, true_optimum: float | None=None):
        """Plot BO curve: best-so-far objective vs iteration."""

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.history, label="best y")

        if true_optimum is not None:
            ax.axhline(true_optimum, linestyle="--", label=f"optimum ({true_optimum})")
        
        ax.set_xlabel("iteration")
        ax.set_ylabel("best observed y")
        ax.set_title("BO result")
        ax.legend()
        plt.tight_layout()
        plt.show()


@dataclass
class MultiBOResult:
    """Dataclass for multi-seed BO loop results."""

    histories: jax.Array
    best_xs: jax.Array
    best_ys: jax.Array
    results: list[BOResult] = field(default_factory=list)
    n_seeds: int = 0
    n_iter: int = 0


    @property
    def mean_history(self) -> jax.Array:
        return self.histories.mean(axis=0)
    

    @property
    def std_history(self) -> jax.Array:
        return self.histories.std(axis=0)
    

    @property
    def ci95(self) -> jax.Array:
        return 1.96 * self.std_history / jnp.sqrt(self.n_seeds)


    def plot(self, true_optimum: float | None=None, show_individual: bool=True):
        """Plot mean BO curves with 95% CI bands."""

        fig, ax = plt.subplots(figsize=(8, 6))

        iters = jnp.arange(1, self.n_iter + 1)
        mean = jnp.array(self.mean_history)
        ci = jnp.array(self.ci95)

        if show_individual:
            for h in self.histories:
                ax.plot(iters, jnp.array(h), alpha=0.15)

        ax.plot(iters, mean, label=f"mean (n={self.n_seeds})")
        ax.fill_between(
            iters, mean - ci, mean + ci,
            alpha=0.25, label="95% CI"
        )

        if true_optimum is not None:
            ax.axhline(true_optimum, linestyle="--", label=f"optimum ({true_optimum})")
        
        ax.set_xlabel("iteration")
        ax.set_ylabel("best observed y")
        ax.set_title(f"BO result - {self.n_seeds} seeds")
        ax.legend()
        plt.tight_layout()
        plt.show()