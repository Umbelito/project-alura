"""Integração processamento + MLflow + armazenamento + smoke (Aula 4)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from customer_segmentation.storage import write_parquet_idempotent
from customer_segmentation.training.models import build_model_pipeline
from customer_segmentation.training.preprocess import DEFAULT_FEATURE_COLUMNS, select_model_frame
from customer_segmentation.training.profiles import build_segment_profiles


@pytest.mark.integration
def test_integration_storage_mlflow_and_gates(
    tmp_path,
    toy_features: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mlflow = pytest.importorskip("mlflow")

    from customer_segmentation.training import experiment as exp_mod
    from customer_segmentation.training import gates_run as gates_mod
    from customer_segmentation.training.gates_run import run_quality_gates

    feat_path = tmp_path / "customer_features.parquet"
    toy_features.to_parquet(feat_path, index=False)

    monkeypatch.setattr(
        exp_mod,
        "load_features",
        lambda as_of, version, uri=None: (toy_features, str(feat_path)),
    )
    monkeypatch.setattr(gates_mod, "load_features", lambda as_of, version, uri=None: (toy_features, str(feat_path)))
    monkeypatch.setattr(
        "customer_segmentation.storage.data_dir",
        lambda *parts: tmp_path.joinpath(*parts),
    )
    monkeypatch.setattr(
        "customer_segmentation.scoring.assign.scores_table_path",
        lambda as_of: tmp_path / "scores.parquet",
    )
    monkeypatch.setattr(
        "customer_segmentation.scoring.assign.scores_dir",
        lambda as_of: tmp_path,
    )
    monkeypatch.setattr(
        "customer_segmentation.training.gates_run.scores_table_path",
        lambda as_of: tmp_path / "scores.parquet",
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
    mlflow.set_tracking_uri(tracking)
    mlflow.set_experiment("quality_gates_ci")

    result = run_quality_gates(
        as_of_date=date(2018, 8, 31),
        pipeline=pipe,
        label_map=profiles["label_map"],
        model_version="ci",
        tracking_uri=tracking,
        promote=False,
        register_tags=True,
        candidate_metrics={"stability_ari_subsample": 0.85, "silhouette": 0.35},
    )
    assert "decision" in result["report"]
    assert result["smoke"]["passed"]
    assert result["integration"]["passed"]
    assert (tmp_path / "scores.parquet").is_file()
    scored = pd.read_parquet(tmp_path / "scores.parquet")
    assert len(scored) == len(toy_features)
    assert result["tags"]["quality_gate_decision"] in {"approved", "rejected"}


@pytest.mark.integration
def test_write_and_reload_parquet_roundtrip(tmp_path) -> None:
    path = tmp_path / "x.parquet"
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    ref = write_parquet_idempotent(df, path)
    loaded = pd.read_parquet(path)
    assert ref.row_count == 2
    assert list(loaded["a"]) == [1.0, 2.0]
