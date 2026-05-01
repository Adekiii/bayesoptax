import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from .result import MultiBOResult


def _label_to_filename(label: str) -> str:
    return label.lower().replace(" ", "_").replace("-", "_")


def save_run(save_dir: str, results: dict[str, MultiBOResult], meta: dict) -> None:
    """Save optimization results and metadata to save_dir."""
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    for label, result in results.items():
        fname = _label_to_filename(label)
        np.savez(
            path / f"{fname}.npz",
            histories=np.array(result.histories),
            best_xs=np.array(result.best_xs),
            best_ys=np.array(result.best_ys),
            n_seeds=result.n_seeds,
            n_iter=result.n_iter,
        )

    meta_out = dict(meta)
    meta_out["labels"] = list(results.keys())
    with open(path / "meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"Saved to {path}")


def load_run(save_dir: str) -> tuple[dict[str, MultiBOResult], dict]:
    """Load results saved by save_run."""
    path = Path(save_dir)

    with open(path / "meta.json") as f:
        meta = json.load(f)

    results = {}
    for label in meta["labels"]:
        fname = _label_to_filename(label)
        data = np.load(path / f"{fname}.npz")
        results[label] = MultiBOResult(
            histories=jnp.array(data["histories"]),
            best_xs=jnp.array(data["best_xs"]),
            best_ys=jnp.array(data["best_ys"]),
            n_seeds=int(data["n_seeds"]),
            n_iter=int(data["n_iter"]),
        )

    return results, meta
