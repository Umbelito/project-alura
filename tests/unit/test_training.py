"""Testes do pipeline de treino, métricas, perfis e seleção (Aula 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from customer_segmentation.training.metrics import clustering_metrics
from customer_segmentation.training.models import build_model_pipeline
from customer_segmentation.training.preprocess import (
    DEFAULT_FEATURE_COLUMNS,
    build_preprocessor,
    select_model_frame,
)
from customer_segmentation.training.profiles import build_segment_profiles
from customer_segmentation.training.selection import evaluate_gates, select_candidate


def _toy_features(n: int = 90, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "customer_unique_id": [f"c{i}" for i in range(n)],
            "recency_days": np.r_[rng.integers(5, 20, n // 3), rng.integers(40, 80, n // 3), rng.integers(120, 200, n - 2 * (n // 3))],
            "frequency": np.r_[rng.integers(4, 8, n // 3), rng.integers(2, 4, n // 3), rng.integers(1, 2, n - 2 * (n // 3))],
            "monetary": np.r_[rng.uniform(400, 900, n // 3), rng.uniform(80, 200, n // 3), rng.uniform(20, 70, n - 2 * (n // 3))],
            "avg_ticket": np.r_[rng.uniform(80, 150, n // 3), rng.uniform(40, 80, n // 3), rng.uniform(15, 40, n - 2 * (n // 3))],
            "avg_items_per_order": rng.uniform(1, 3, n),
            "customer_age_days": rng.integers(0, 200, n),
        }
    )


def test_preprocessor_output_shape() -> None:
    df = _toy_features()
    X = select_model_frame(df)
    prep = build_preprocessor()
    transformed = prep.fit_transform(X)
    assert transformed.shape == (len(df), len(DEFAULT_FEATURE_COLUMNS))


def test_kmeans_pipeline_predict() -> None:
    X = select_model_frame(_toy_features())
    pipe = build_model_pipeline(
        "kmeans",
        list(X.columns),
        ["recency_days", "frequency", "monetary", "avg_ticket"],
        n_clusters=3,
    )
    labels = pipe.fit_predict(X)
    assert set(labels) <= {0, 1, 2}
    assert len(labels) == len(X)
    assert len(pipe.predict(X.head(3))) == 3


def test_clustering_metrics_and_profiles() -> None:
    features = _toy_features()
    X = select_model_frame(features)
    pipe = build_model_pipeline("kmeans", list(X.columns), ["monetary"], n_clusters=3)
    labels = pipe.fit_predict(X)
    X_t = pipe.named_steps["preprocess"].transform(X)
    metrics = clustering_metrics(X_t, labels, pipe)
    assert metrics["n_clusters"] == 3
    assert metrics["min_size"] > 0
    assert metrics["inertia"] is not None
    assert metrics["silhouette"] == metrics["silhouette"]  # not NaN
    profiles = build_segment_profiles(features, labels)
    assert len(profiles["label_map"]) == 3
    assert all(isinstance(v, str) for v in profiles["label_map"].values())


def test_select_candidate_prefers_gates() -> None:
    results = [
        {
            "run_name": "weak",
            "n_clusters": 4,
            "silhouette": 0.10,
            "davies_bouldin": 1.5,
            "min_size": 80,
            "min_share": 0.2,
            "stability_ari_subsample": 0.9,
        },
        {
            "run_name": "strong",
            "n_clusters": 4,
            "silhouette": 0.35,
            "davies_bouldin": 0.8,
            "min_size": 80,
            "min_share": 0.2,
            "stability_ari_subsample": 0.7,
        },
    ]
    gates = {
        "silhouette_min": 0.20,
        "segment_count_between": [3, 10],
        "min_cluster_size": 50,
        "min_cluster_share": 0.02,
        "ari_subsample_min": 0.50,
    }
    selection = select_candidate(results, gates)
    assert selection["winner"]["run_name"] == "strong"
    assert selection["n_passing_gates"] == 1
    assert evaluate_gates(results[0], gates)["passed"] is False


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("mlflow") is None,
    reason="mlflow não instalado",
)
def test_quick_experiment_registers_candidate(tmp_path, monkeypatch) -> None:
    from customer_segmentation.training import experiment as exp_mod

    features = _toy_features(n=120)
    feat_path = tmp_path / "customer_features.parquet"
    features.to_parquet(feat_path, index=False)
    monkeypatch.setattr(
        exp_mod,
        "load_features",
        lambda as_of, version, uri=None: (features, str(feat_path)),
    )
    monkeypatch.setattr(exp_mod, "data_dir", lambda *parts: tmp_path.joinpath(*parts))

    tracking = f"file:{(tmp_path / 'mlruns').as_posix()}"
    relaxed = {
        "silhouette_min": 0.05,
        "segment_count_between": [2, 10],
        "min_cluster_size": 5,
        "min_cluster_share": 0.01,
        "ari_subsample_min": 0.0,
    }
    payload = exp_mod.run_training_experiment(
        as_of_date="2018-08-31",
        feature_set_version="1.0.0",
        quick=True,
        tracking_uri=tracking,
        register=True,
        set_champion=True,
        gates=relaxed,
    )
    assert payload["winner"]["algorithm"] == "kmeans"
    assert payload["mlflow"]["model_version"]
    assert "candidate" in payload["mlflow"]["aliases"]
    assert payload["artifacts"]["comparison_uri"]
