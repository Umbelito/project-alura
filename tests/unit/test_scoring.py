"""Testes de inferência, assinatura e smoke do modelo (Aula 4)."""

from __future__ import annotations

import pandas as pd

from customer_segmentation.scoring.assign import assign_segments
from customer_segmentation.training.models import build_model_pipeline
from customer_segmentation.training.preprocess import DEFAULT_FEATURE_COLUMNS, select_model_frame
from customer_segmentation.training.profiles import build_segment_profiles
from customer_segmentation.training.signature import check_signature_compatibility


def _fit(toy_features: pd.DataFrame):
    X = select_model_frame(toy_features)
    pipe = build_model_pipeline(
        "kmeans",
        DEFAULT_FEATURE_COLUMNS,
        ["recency_days", "frequency", "monetary", "avg_ticket"],
        n_clusters=3,
    )
    pipe.fit(X)
    labels = pipe.predict(X)
    profiles = build_segment_profiles(toy_features, labels)
    return pipe, profiles


def test_assign_segments_matches_contract(toy_features: pd.DataFrame) -> None:
    pipe, profiles = _fit(toy_features)
    scored = assign_segments(
        toy_features,
        pipe,
        label_map=profiles["label_map"],
        model_version="test",
    )
    assert len(scored) == len(toy_features)
    assert scored["segment_id"].notna().all()
    assert scored["segment_label"].notna().all()
    assert scored["customer_unique_id"].is_unique
    for col in (
        "customer_unique_id",
        "as_of_date",
        "segment_id",
        "segment_label",
        "model_version",
        "feature_set_version",
        "scored_at",
    ):
        assert col in scored.columns


def test_signature_rejects_missing_column(toy_features: pd.DataFrame) -> None:
    pipe, _ = _fit(toy_features)
    report = check_signature_compatibility(pipe, toy_features, DEFAULT_FEATURE_COLUMNS)
    assert report["passed"]
    assert report["missing_column_raises"]


def test_smoke_predict_finite(toy_features: pd.DataFrame) -> None:
    from customer_segmentation.training.gates_run import smoke_predict

    pipe, _ = _fit(toy_features)
    smoke = smoke_predict(pipe, toy_features, min_rows=5)
    assert smoke["passed"]
    assert smoke["n_unique"] >= 1
