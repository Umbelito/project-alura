"""Contratos de entrada e saída — schema, PK e consumidores (Aula 4)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from customer_segmentation.validation.contracts import load_contract
from customer_segmentation.validation.quality import validate_dataframe, validate_feature_table

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "contracts" / "input"
OUTPUT = ROOT / "contracts" / "output"


@pytest.mark.parametrize(
    "name,pk",
    [
        ("orders", ["order_id"]),
        ("customers", ["customer_id"]),
        ("order_items", ["order_id", "order_item_id"]),
        ("order_payments", ["order_id", "payment_sequential"]),
    ],
)
def test_input_contract_primary_key(name: str, pk: list[str]) -> None:
    contract = load_contract(INPUT / f"{name}.yaml")
    assert contract["primary_key"] == pk
    col_names = [c["name"] for c in contract["columns"]]
    for key in pk:
        assert key in col_names


def test_orders_foreign_key_to_customers() -> None:
    contract = load_contract(INPUT / "orders.yaml")
    fks = contract.get("foreign_keys") or []
    assert any(fk["references"] == "customers.customer_id" for fk in fks)


def test_output_contracts_have_consumers() -> None:
    for name in ("customer_features.yaml", "customer_segments.yaml"):
        contract = load_contract(OUTPUT / name)
        assert contract.get("consumers"), name
        assert contract.get("primary_key")


def test_validate_dataframe_honors_input_contract() -> None:
    df = pd.DataFrame(
        {
            "order_id": ["a"],
            "customer_id": ["c"],
            "order_status": ["delivered"],
            "order_purchase_timestamp": ["2018-01-01"],
            "order_approved_at": [None],
            "order_delivered_carrier_date": [None],
            "order_delivered_customer_date": [None],
            "order_estimated_delivery_date": [None],
        }
    )
    assert validate_dataframe(df, "orders").passed


def test_feature_table_contract_rejects_zero_frequency(toy_features: pd.DataFrame) -> None:
    bad = toy_features.copy()
    bad.loc[0, "frequency"] = 0
    report = validate_feature_table(bad)
    assert not report.passed
