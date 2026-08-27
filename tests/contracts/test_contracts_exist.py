"""Testes de presença e campos mínimos dos contratos (Aula 1)."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "contracts" / "input"
OUTPUT = ROOT / "contracts" / "output"

REQUIRED_INPUT = [
    "orders.yaml",
    "customers.yaml",
    "order_items.yaml",
    "order_payments.yaml",
]


@pytest.mark.parametrize("name", REQUIRED_INPUT)
def test_input_contract_exists(name: str) -> None:
    path = INPUT / name
    assert path.is_file(), f"Contrato ausente: {path}"
    text = path.read_text(encoding="utf-8")
    assert "version:" in text
    assert "columns:" in text
    assert "quality_checks:" in text


def test_output_customer_segments_contract() -> None:
    path = OUTPUT / "customer_segments.yaml"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    for field in (
        "customer_unique_id",
        "segment_id",
        "segment_label",
        "model_version",
        "feature_set_version",
        "as_of_date",
    ):
        assert field in text
