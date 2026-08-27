"""Política de quality gates — aprovação e rejeição (Aula 4)."""

from __future__ import annotations

import numpy as np

from customer_segmentation.training.promotion import evaluate_promotion, tags_from_report


def _labels(n: int = 90, k: int = 3) -> np.ndarray:
    return np.array([i % k for i in range(n)])


def test_promotion_approved_when_all_gates_pass() -> None:
    report = evaluate_promotion(
        data_ok=True,
        labels=_labels(),
        candidate_metrics={
            "silhouette": 0.35,
            "davies_bouldin": 0.8,
            "stability_ari_subsample": 0.8,
            "n_clusters": 3,
        },
        champion_metrics={"silhouette": 0.33, "davies_bouldin": 0.85},
        signature_ok=True,
        integration_ok=True,
        smoke_ok=True,
    )
    assert report["approved"] is True
    assert report["decision"] == "approved"
    assert report["blocking_failed"] == []


def test_promotion_rejects_data_failure() -> None:
    report = evaluate_promotion(
        data_ok=False,
        data_detail="PK duplicada",
        labels=_labels(),
        candidate_metrics={"silhouette": 0.4, "stability_ari_subsample": 0.9},
        champion_metrics=None,
        signature_ok=True,
        integration_ok=True,
        smoke_ok=True,
    )
    assert report["approved"] is False
    assert "data_failure" in report["blocking_failed"]


def test_promotion_rejects_metric_regression() -> None:
    report = evaluate_promotion(
        data_ok=True,
        labels=_labels(),
        candidate_metrics={"silhouette": 0.22, "davies_bouldin": 1.5, "stability_ari_subsample": 0.8},
        champion_metrics={"silhouette": 0.40, "davies_bouldin": 0.7},
        signature_ok=True,
        integration_ok=True,
        smoke_ok=True,
    )
    assert report["approved"] is False
    assert "metric_regression" in report["blocking_failed"]


def test_promotion_rejects_empty_or_unstable_clusters() -> None:
    labels = np.array([0] * 89 + [1])
    report = evaluate_promotion(
        data_ok=True,
        labels=labels,
        candidate_metrics={"silhouette": 0.4, "stability_ari_subsample": 0.1},
        champion_metrics=None,
        signature_ok=True,
        integration_ok=True,
        smoke_ok=True,
    )
    assert report["approved"] is False
    assert "unstable_or_empty_clusters" in report["blocking_failed"]


def test_promotion_rejects_signature_and_integration() -> None:
    report = evaluate_promotion(
        data_ok=True,
        labels=_labels(),
        candidate_metrics={"silhouette": 0.4, "stability_ari_subsample": 0.9},
        champion_metrics=None,
        signature_ok=False,
        signature_detail="coluna ausente",
        integration_ok=False,
        integration_detail="parquet falhou",
        smoke_ok=True,
    )
    assert "signature_incompatible" in report["blocking_failed"]
    assert "integration_failure" in report["blocking_failed"]


def test_tags_from_report_include_decision() -> None:
    report = evaluate_promotion(
        data_ok=True,
        labels=_labels(),
        candidate_metrics={"silhouette": 0.4, "stability_ari_subsample": 0.9},
        champion_metrics=None,
        signature_ok=True,
        integration_ok=True,
        smoke_ok=True,
    )
    tags = tags_from_report(report)
    assert tags["quality_gate_decision"] == "approved"
    assert tags["quality_gate_data_failure"] == "pass"
    assert "quality_gate_signature_incompatible" in tags
