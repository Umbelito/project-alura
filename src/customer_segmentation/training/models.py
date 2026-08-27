"""Algoritmos compatíveis com o contrato operacional (predict em batch)."""

from __future__ import annotations

import itertools
from typing import Any, Iterator

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline

from customer_segmentation.training.preprocess import build_preprocessor

# Apenas estimadores com .predict() para novos clientes no scoring semanal.
SUPPORTED_ALGORITHMS = ("kmeans", "minibatch_kmeans", "gaussian_mixture")


def make_clusterer(algorithm: str, random_state: int = 42, **params: Any):
    algo = algorithm.lower()
    n_clusters = int(params.pop("n_clusters", params.pop("n_components", 5)))
    if algo == "kmeans":
        return KMeans(
            n_clusters=n_clusters,
            n_init=int(params.get("n_init", 10)),
            random_state=random_state,
        )
    if algo == "minibatch_kmeans":
        return MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=int(params.get("batch_size", 1024)),
            n_init=int(params.get("n_init", 10)),
            random_state=random_state,
        )
    if algo in {"gaussian_mixture", "gmm"}:
        return GaussianMixture(
            n_components=n_clusters,
            covariance_type=str(params.get("covariance_type", "diag")),
            n_init=int(params.get("n_init", 2)),
            random_state=random_state,
        )
    raise ValueError(
        f"Algoritmo '{algorithm}' incompatível com o contrato operacional. "
        f"Suportados: {SUPPORTED_ALGORITHMS}"
    )


def build_model_pipeline(
    algorithm: str,
    feature_columns: list[str],
    log_columns: list[str],
    random_state: int = 42,
    **cluster_params: Any,
) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(feature_columns, log_columns)),
            ("cluster", make_clusterer(algorithm, random_state=random_state, **cluster_params)),
        ]
    )


def iter_search_space(search: dict[str, dict[str, list]], random_state: int = 42) -> Iterator[dict[str, Any]]:
    """Expande a grade YAML em configurações planas."""
    for algorithm, grid in search.items():
        keys = list(grid.keys())
        values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo, strict=True))
            n_clusters = params.get("n_clusters", params.get("n_components"))
            extra = {k: v for k, v in params.items() if k not in {"n_clusters", "n_components"}}
            label_bits = [algorithm, f"k{n_clusters}"] + [f"{k}={v}" for k, v in extra.items()]
            yield {
                "algorithm": algorithm,
                "n_clusters": int(n_clusters),
                "params": params,
                "run_name": "_".join(str(b) for b in label_bits),
                "random_state": random_state,
            }
