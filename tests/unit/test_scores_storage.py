"""Resolução de URIs da tabela histórica de scores (Aula 5)."""

from __future__ import annotations

import json

import pandas as pd

from customer_segmentation.storage import latest_scores_table_path, resolve_data_uri


def test_resolve_data_uri_prefers_existing_file(tmp_path) -> None:
    parquet = tmp_path / "customer_segments.parquet"
    pd.DataFrame({"customer_unique_id": ["a"]}).to_parquet(parquet, index=False)
    assert resolve_data_uri(str(parquet)) == parquet
    assert resolve_data_uri(None) is None
    assert resolve_data_uri(str(tmp_path / "missing.parquet")) is None


def test_latest_scores_follows_current_pointer(tmp_path, monkeypatch) -> None:
    partition = tmp_path / "scores" / "as_of_date=2018-08-31"
    partition.mkdir(parents=True)
    parquet = partition / "customer_segments.parquet"
    pd.DataFrame({"customer_unique_id": ["a"], "model_version": ["2"]}).to_parquet(
        parquet, index=False
    )
    pointer = tmp_path / "scores" / "current.json"
    pointer.write_text(
        json.dumps({"uri": "data/scores/as_of_date=2018-08-31/customer_segments.parquet"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("customer_segmentation.storage.data_root", lambda: tmp_path)
    assert latest_scores_table_path() == parquet
