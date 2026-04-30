import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from .result import MultiBOResult


def plot_comparison(
        results: dict[str, MultiBOResult],
        title: str = "",
        true_optimum: float | None = None,
) -> None:
    """Plot best-so-far curves for multiple optimizers."""

    sns.set_theme()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette()

    for i, (label, result) in enumerate(results.items()):
        x = jnp.arange(1, result.n_iter + 1)
        mean = jnp.array(result.mean_history)
        ci = jnp.array(result.ci95)
        color = colors[i % len(colors)]

        ax.plot(x, mean, label=f"{label} (n={result.n_seeds})", color=color)
        ax.fill_between(x, mean - ci, mean + ci, alpha=0.2, color=color)

    if true_optimum is not None:
        ax.axhline(true_optimum, linestyle=":", color="black", linewidth=1.2,
                   label=f"optimum ({true_optimum})")

    ax.set_xlabel("objective evaluations")
    ax.set_ylabel("best observed value")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
