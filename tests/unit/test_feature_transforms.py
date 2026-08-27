"""Testes unitários extras de transformações e engenharia de features (Aula 4)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from customer_segmentation.features.cleaning import clean_order_items, clean_orders
from customer_segmentation.features.rfm import _quintile_scores, compute_rfm_features
from customer_segmentation.training.preprocess import log1p_nonneg, select_model_frame


def test_log1p_nonneg_clips_negatives() -> None:
    out = log1p_nonneg([[-1.0, 0.0, np.e - 1]])
    assert out.shape == (1, 3)
    assert out[0, 0] == 0.0
    assert out[0, 1] == 0.0
    np.testing.assert_allclose(out[0, 2], 1.0, rtol=1e-6)


def test_select_model_frame_missing_column_raises(toy_features: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="ausentes"):
        select_model_frame(toy_features, ["recency_days", "not_a_column"])


def test_quintile_recent_gets_higher_r_score() -> None:
    series = pd.Series([1, 10, 20, 40, 80, 100, 120, 200, 300, 400])
    scores = _quintile_scores(series, ascending=False)
    assert int(scores.iloc[0]) > int(scores.iloc[-1])


def test_compute_rfm_empty_raises() -> None:
    with pytest.raises(ValueError, match="vazio"):
        compute_rfm_features(pd.DataFrame(), date(2018, 8, 31))


def test_avg_days_between_orders_nan_for_one_shot() -> None:
    enriched = pd.DataFrame(
        {
            "customer_unique_id": ["u1"],
            "order_id": ["o1"],
            "order_item_id": [1],
            "price": [10.0],
            "freight_value": [2.0],
            "order_purchase_timestamp": ["2018-08-01"],
            "payment_value_total": [12.0],
            "payment_installments_avg": [1.0],
            "payment_type_mode": ["boleto"],
            "customer_state": ["RJ"],
        }
    )
    features = compute_rfm_features(enriched, date(2018, 8, 31))
    assert features.loc[0, "frequency"] == 1
    assert pd.isna(features.loc[0, "avg_days_between_orders"])
    assert features.loc[0, "recency_days"] == 30


def test_clean_orders_drops_duplicates_and_nulls() -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2", None],
            "customer_id": ["c1", "c1", "c2", "c3"],
            "order_purchase_timestamp": ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"],
            "order_status": ["delivered", "delivered", "delivered", "canceled"],
        }
    )
    cleaned = clean_orders(orders)
    assert len(cleaned) == 2
    assert cleaned["order_id"].is_unique


def test_clean_order_items_drops_negative_price() -> None:
    items = pd.DataFrame(
        {
            "order_id": ["o1", "o1"],
            "order_item_id": [1, 2],
            "price": [10.0, -1.0],
            "freight_value": [1.0, 1.0],
        }
    )
    cleaned = clean_order_items(items)
    assert len(cleaned) == 1
    assert (cleaned["price"] >= 0).all()
