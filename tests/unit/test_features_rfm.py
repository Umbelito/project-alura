"""Testes de features RFM e pipeline (Aula 2)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from customer_segmentation.features.rfm import compute_rfm_features
from customer_segmentation.ingestion.incremental import months_in_window
from customer_segmentation.storage import DatasetRef, relative_uri, write_parquet_idempotent


def test_months_in_window() -> None:
    months = months_in_window(date(2018, 8, 31), 60)
    assert (2018, 7) in months
    assert (2018, 8) in months
    assert months[0] <= months[-1]


def test_compute_rfm_features_basic() -> None:
    enriched = pd.DataFrame(
        {
            "customer_unique_id": ["u1", "u1", "u2"],
            "order_id": ["o1", "o2", "o3"],
            "order_item_id": [1, 1, 1],
            "price": [100.0, 50.0, 80.0],
            "freight_value": [10.0, 5.0, 8.0],
            "order_purchase_timestamp": [
                "2018-06-01",
                "2018-08-01",
                "2018-07-15",
            ],
            "payment_value_total": [110.0, 55.0, 88.0],
            "payment_installments_avg": [1.0, 2.0, 1.0],
            "payment_type_mode": ["credit_card", "boleto", "boleto"],
            "customer_state": ["SP", "SP", "RJ"],
        }
    )
    features = compute_rfm_features(enriched, date(2018, 8, 31))
    assert set(features["customer_unique_id"]) == {"u1", "u2"}
    u1 = features.set_index("customer_unique_id").loc["u1"]
    assert u1["frequency"] == 2
    assert u1["monetary"] == 150.0
    assert u1["recency_days"] == 30
    assert "r_score" in features.columns
    assert "preferred_payment_type" in features.columns


def test_write_parquet_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from customer_segmentation import storage

    monkeypatch.setattr(storage, "project_root", lambda: tmp_path)
    path = tmp_path / "data" / "features" / "t.parquet"
    df = pd.DataFrame({"a": [1, 2, 3]})
    ref1 = write_parquet_idempotent(df, path)
    ref2 = write_parquet_idempotent(df.assign(a=[1, 2, 4]), path)
    assert path.is_file()
    assert (path.parent / "_SUCCESS").is_file()
    assert ref1.row_count == 3
    assert ref2.row_count == 3
    assert pd.read_parquet(path)["a"].tolist() == [1, 2, 4]


def test_dataset_ref_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "x.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    ref = DatasetRef(uri=relative_uri(path, root=tmp_path), dataset="x", row_count=1)
    restored = DatasetRef.from_dict(ref.to_dict())
    assert restored.uri == ref.uri
    assert restored.absolute_path(root=tmp_path) == path
