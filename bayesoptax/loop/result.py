from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class BOResult:
    """Dataclass for BO loop results."""

    X_obs: jax.Array
    y_obs: jax.Array
    best_x: jax.Array
    best_y: float
    history: jax.Array
    random_history: jax.Array | None = None

    def plot(self, true_optimum: float | None=None, title: str="BO result"):
        """Plot BO curve: best-so-far objective vs iteration."""

        sns.set_theme()
        fig, ax = plt.subplots(figsize=(8, 6))
        iters = jnp.arange(1, len(self.history) + 1)
        sns.lineplot(x=iters, y=jnp.array(self.history), label="BO", ax=ax)

        if self.random_history is not None:
            rs_iters = jnp.arange(1, len(self.random_history) + 1)
            sns.lineplot(x=rs_iters, y=jnp.array(self.random_history),
                         label="random search", linestyle="--", ax=ax)

        if true_optimum is not None:
            ax.axhline(true_optimum, linestyle=":", label=f"optimum ({true_optimum})")

        ax.set_xlabel("iteration")
        ax.set_ylabel("best observed y")
        ax.set_title(title)
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
    random_histories: jax.Array | None = None


    @property
    def mean_history(self) -> jax.Array:
        return self.histories.mean(axis=0)


    @property
    def std_history(self) -> jax.Array:
        return self.histories.std(axis=0)


    @property
    def ci95(self) -> jax.Array:
        return 1.96 * self.std_history / jnp.sqrt(self.n_seeds)


    @property
    def mean_random_history(self) -> jax.Array:
        return self.random_histories.mean(axis=0)


    @property
    def ci95_random(self) -> jax.Array:
        std = self.random_histories.std(axis=0)
        return 1.96 * std / jnp.sqrt(self.n_seeds)


    def plot(self, true_optimum: float | None=None, show_individual: bool=True, title: str | None=None):
        """Plot mean BO curves with 95% CI bands, overlaid with random search."""

        sns.set_theme()
        fig, ax = plt.subplots(figsize=(8, 6))

        iters = jnp.arange(1, self.n_iter + 1)
        mean = jnp.array(self.mean_history)
        ci = jnp.array(self.ci95)

        if show_individual:
            for h in self.histories:
                sns.lineplot(x=iters, y=jnp.array(h), alpha=0.15, ax=ax, legend=False)

        bo_color = sns.color_palette()[0]
        sns.lineplot(x=iters, y=mean, label=f"BO (n={self.n_seeds})", ax=ax, color=bo_color)
        ax.fill_between(iters, mean - ci, mean + ci, alpha=0.25, color=bo_color)

        if self.random_histories is not None:
            rs_iters = jnp.arange(1, self.random_histories.shape[1] + 1)
            rs_mean = jnp.array(self.mean_random_history)
            rs_ci = jnp.array(self.ci95_random)
            rs_color = sns.color_palette()[1]
            sns.lineplot(x=rs_iters, y=rs_mean, label=f"random search (n={self.n_seeds})",
                         linestyle="--", ax=ax, color=rs_color)
            ax.fill_between(rs_iters, rs_mean - rs_ci, rs_mean + rs_ci,
                            alpha=0.25, color=rs_color)

        if true_optimum is not None:
            ax.axhline(true_optimum, linestyle=":", label=f"optimum ({true_optimum})")

        ax.set_xlabel("iteration")
        ax.set_ylabel("best observed y")
        ax.set_title(title if title is not None else f"BO result - {self.n_seeds} seeds")
        ax.legend()
        plt.tight_layout()
        plt.show()