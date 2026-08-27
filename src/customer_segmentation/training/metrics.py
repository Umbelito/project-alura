"""Métricas de qualidade de clustering alinhadas ao contrato operacional."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score


def cluster_distribution(labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels)
    values, counts = np.unique(labels, return_counts=True)
    total = int(counts.sum()) or 1
    sizes = {int(v): int(c) for v, c in zip(values, counts)}
    shares = {str(k): c / total for k, c in sizes.items()}
    return {
        "n_clusters": int(len(values)),
        "sizes": sizes,
        "shares": shares,
        "min_size": int(counts.min()) if len(counts) else 0,
        "max_size": int(counts.max()) if len(counts) else 0,
        "min_share": float(counts.min() / total) if len(counts) else 0.0,
        "empty_clusters": False,
    }


def clustering_metrics(
    X_transformed: np.ndarray,
    labels: np.ndarray,
    estimator=None,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    unique = np.unique(labels)
    dist = cluster_distribution(labels)

    silhouette = float("nan")
    davies_bouldin = float("nan")
    if len(unique) >= 2 and len(labels) > len(unique):
        sample_size = min(4000, len(labels)) if len(labels) > 4000 else None
        silhouette = float(
            silhouette_score(
                X_transformed,
                labels,
                metric="euclidean",
                sample_size=sample_size,
                random_state=42,
            )
        )
        davies_bouldin = float(davies_bouldin_score(X_transformed, labels))

    inertia = None
    bic = None
    aic = None
    clusterer = estimator
    if estimator is not None and hasattr(estimator, "named_steps"):
        clusterer = estimator.named_steps.get("cluster", estimator)
    if clusterer is not None:
        inertia = getattr(clusterer, "inertia_", None)
        if inertia is not None:
            inertia = float(inertia)
        if hasattr(clusterer, "bic") and hasattr(clusterer, "n_components"):
            try:
                bic = float(clusterer.bic(X_transformed))
                aic = float(clusterer.aic(X_transformed))
            except Exception:
                bic = None
                aic = None

    return {
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "inertia": inertia,
        "bic": bic,
        "aic": aic,
        **dist,
    }
