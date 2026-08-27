"""Testes unitários de validação de qualidade (Aula 2)."""

from __future__ import annotations

import pandas as pd
import pytest

from customer_segmentation.validation.quality import (
    assert_valid,
    validate_dataframe,
    validate_feature_table,
    validate_referential_integrity,
)


def test_validate_orders_happy_path() -> None:
    df = pd.DataFrame(
        {
            "order_id": ["o1", "o2"],
            "customer_id": ["c1", "c2"],
            "order_status": ["delivered", "shipped"],
            "order_purchase_timestamp": ["2018-01-01", "2018-01-02"],
            "order_approved_at": [None, "2018-01-02"],
            "order_delivered_carrier_date": [None, None],
            "order_delivered_customer_date": [None, None],
            "order_estimated_delivery_date": [None, None],
        }
    )
    report = validate_dataframe(df, "orders")
    assert report.passed
    assert_valid(report)


def test_validate_orders_duplicate_pk_fails() -> None:
    df = pd.DataFrame(
        {
            "order_id": ["o1", "o1"],
            "customer_id": ["c1", "c2"],
            "order_status": ["delivered", "delivered"],
            "order_purchase_timestamp": ["2018-01-01", "2018-01-02"],
            "order_approved_at": [None, None],
            "order_delivered_carrier_date": [None, None],
            "order_delivered_customer_date": [None, None],
            "order_estimated_delivery_date": [None, None],
        }
    )
    report = validate_dataframe(df, "orders")
    assert not report.passed
    with pytest.raises(ValueError, match="primary_key_unique"):
        assert_valid(report)


def test_referential_integrity_orphans() -> None:
    parent = pd.DataFrame({"order_id": ["o1"]})
    child = pd.DataFrame({"order_id": ["o1", "o2"]})
    check = validate_referential_integrity(
        child, parent, ["order_id"], ["order_id"], "fk_test"
    )
    assert not check.passed
    assert "1" in check.detail


def test_validate_feature_table_minimal() -> None:
    df = pd.DataFrame(
        {
            "customer_unique_id": ["u1", "u2"],
            "as_of_date": ["2018-08-31", "2018-08-31"],
            "recency_days": [10, 20],
            "frequency": [2, 1],
            "monetary": [100.0, 50.0],
            "r_score": [5, 4],
            "f_score": [3, 2],
            "m_score": [4, 3],
            "rfm_score": ["534", "423"],
            "avg_ticket": [50.0, 50.0],
            "avg_items_per_order": [1.0, 1.0],
            "avg_freight": [10.0, 8.0],
            "avg_installments": [1.0, 2.0],
            "preferred_payment_type": ["credit_card", "boleto"],
            "customer_state": ["SP", "RJ"],
            "customer_age_days": [30, 0],
            "avg_days_between_orders": [30.0, None],
            "first_purchase": ["2018-07-01", "2018-08-10"],
            "last_purchase": ["2018-07-31", "2018-08-10"],
            "feature_set_version": ["1.0.0", "1.0.0"],
            "built_at": ["2018-09-01T00:00:00Z", "2018-09-01T00:00:00Z"],
        }
    )
    report = validate_feature_table(df)
    assert report.passed
