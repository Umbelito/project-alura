"""Gates técnicos/de negócio e ranking do candidato."""

from __future__ import annotations

import math
from typing import Any


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def evaluate_gates(metrics: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    k_min, k_max = gates.get("segment_count_between", [3, 10])
    checks = [
        {
            "name": "silhouette_min",
            "passed": _finite(metrics.get("silhouette"), -1) >= float(gates.get("silhouette_min", 0.20)),
            "value": metrics.get("silhouette"),
            "threshold": gates.get("silhouette_min", 0.20),
        },
        {
            "name": "segment_count_between",
            "passed": k_min <= int(metrics.get("n_clusters", 0)) <= k_max,
            "value": metrics.get("n_clusters"),
            "threshold": [k_min, k_max],
        },
        {
            "name": "no_empty_segment",
            "passed": int(metrics.get("min_size", 0)) >= int(gates.get("min_cluster_size", 1)),
            "value": metrics.get("min_size"),
            "threshold": gates.get("min_cluster_size", 1),
        },
        {
            "name": "min_cluster_share",
            "passed": _finite(metrics.get("min_share")) >= float(gates.get("min_cluster_share", 0.0)),
            "value": metrics.get("min_share"),
            "threshold": gates.get("min_cluster_share", 0.0),
        },
        {
            "name": "ari_subsample_min",
            "passed": _finite(metrics.get("stability_ari_subsample"), -1)
            >= float(gates.get("ari_subsample_min", 0.0)),
            "value": metrics.get("stability_ari_subsample"),
            "threshold": gates.get("ari_subsample_min", 0.0),
        },
    ]
    return {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
    }


def composite_score(metrics: dict[str, Any], weights: dict[str, float] | None = None) -> float:
    weights = weights or {
        "silhouette": 0.45,
        "davies_bouldin_inv": 0.20,
        "stability_ari": 0.35,
    }
    sil = _finite(metrics.get("silhouette"))
    db = _finite(metrics.get("davies_bouldin"), 1.0)
    ari = _finite(metrics.get("stability_ari_subsample"))
    db_inv = 1.0 / (1.0 + max(db, 0.0))
    return (
        weights["silhouette"] * sil
        + weights["davies_bouldin_inv"] * db_inv
        + weights["stability_ari"] * ari
    )


def select_candidate(
    results: list[dict[str, Any]],
    gates: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Prefere quem passou nos gates; desempate pelo score composto."""
    scored: list[dict[str, Any]] = []
    for row in results:
        gate_report = evaluate_gates(row, gates)
        item = {
            **row,
            "gates": gate_report,
            "composite_score": composite_score(row, weights),
        }
        scored.append(item)

    passing = [r for r in scored if r["gates"]["passed"]]
    pool = passing if passing else scored
    winner = max(pool, key=lambda r: r["composite_score"])
    return {
        "winner": winner,
        "results": scored,
        "n_passing_gates": len(passing),
        "selection_pool": "gates_passed" if passing else "best_effort",
    }
