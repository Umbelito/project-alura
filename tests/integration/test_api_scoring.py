"""API + serving sobre a tabela histórica de segmentos (Aula 5)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from customer_segmentation.scoring.assign import assign_segments, persist_segments, publish_current_pointer
from customer_segmentation.serving.lookup import health_snapshot, lookup_segment
from customer_segmentation.training.models import build_model_pipeline
from customer_segmentation.training.preprocess import DEFAULT_FEATURE_COLUMNS, select_model_frame
from customer_segmentation.training.profiles import build_segment_profiles


def test_lookup_and_api_over_historical_table(
    tmp_path,
    toy_features: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    scored = assign_segments(
        toy_features,
        pipe,
        label_map=profiles["label_map"],
        model_name="customer_segmentation_kmeans",
        model_version="9",
    )
    scores_file = tmp_path / "customer_segments.parquet"
    monkeypatch.setattr(
        "customer_segmentation.scoring.assign.scores_table_path",
        lambda as_of: scores_file,
    )
    monkeypatch.setattr(
        "customer_segmentation.scoring.assign.scores_dir",
        lambda as_of: tmp_path,
    )
    monkeypatch.setattr(
        "customer_segmentation.scoring.assign.scores_current_pointer_path",
        lambda: tmp_path / "current.json",
    )
    monkeypatch.setattr(
        "customer_segmentation.serving.lookup.latest_scores_table_path",
        lambda: scores_file,
    )
    monkeypatch.setattr(
        "customer_segmentation.serving.lookup.scores_current_pointer_path",
        lambda: tmp_path / "current.json",
    )

    ref = persist_segments(scored, date(2018, 8, 31))
    publish_current_pointer(ref)

    cid = str(toy_features["customer_unique_id"].iloc[0])
    row = lookup_segment(cid, scores_file)
    assert row is not None
    assert row["model_version"] == "9"
    assert row["as_of_date"] == "2018-08-31"
    assert "updated_at" in row

    snap = health_snapshot(scores_file)
    assert snap["status"] == "ok"
    assert snap["scores_available"] is True
    assert snap["n_customers"] == len(toy_features)

    monkeypatch.setenv("SEGMENTS_PATH", str(scores_file))
    from api import main as api_main

    monkeypatch.setattr(api_main, "_scores_path", lambda: str(scores_file))
    client = TestClient(api_main.app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_version"] == "9"

    found = client.get(f"/segments/{cid}")
    assert found.status_code == 200
    body = found.json()
    assert body["customer_unique_id"] == cid
    assert body["model_version"] == "9"
    assert body["segment_label"]

    missing = client.get("/segments/does-not-exist")
    assert missing.status_code == 404


def test_batch_scoring_persists_history(
    tmp_path,
    toy_features: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from customer_segmentation.scoring.batch import run_batch_scoring
    from customer_segmentation.training import experiment as exp_mod

    feat_path = tmp_path / "features.parquet"
    toy_features.to_parquet(feat_path, index=False)
    monkeypatch.setattr(
        exp_mod,
        "load_features",
        lambda as_of, version, uri=None: (toy_features, str(feat_path)),
    )
    monkeypatch.setattr(
        "customer_segmentation.scoring.batch.load_features",
        lambda as_of, version, uri=None: (toy_features, str(feat_path)),
    )
    scores_file = tmp_path / "as_of" / "customer_segments.parquet"
    monkeypatch.setattr(
        "customer_segmentation.scoring.assign.scores_table_path",
        lambda as_of: scores_file,
    )
    monkeypatch.setattr(
        "customer_segmentation.scoring.assign.scores_dir",
        lambda as_of: scores_file.parent,
    )
    monkeypatch.setattr(
        "customer_segmentation.scoring.assign.scores_current_pointer_path",
        lambda: tmp_path / "current.json",
    )

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

    tracking = f"file:{(tmp_path / 'mlruns').as_posix()}"
    result = run_batch_scoring(
        as_of_date=date(2018, 8, 31),
        pipeline=pipe,
        label_map=profiles["label_map"],
        model_version="7",
        tracking_uri=tracking,
        log_mlflow=True,
    )
    assert result["model_version"] == "7"
    assert result["n_customers"] == len(toy_features)
    assert scores_file.is_file()
    assert (tmp_path / "current.json").is_file()
    hist = pd.read_parquet(scores_file)
    assert (hist["model_version"] == "7").all()
    assert hist["customer_unique_id"].nunique() == len(toy_features)
