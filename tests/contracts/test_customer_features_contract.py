"""Contrato da feature table (Aula 2)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "contracts" / "output"


def test_customer_features_contract() -> None:
    path = OUTPUT / "customer_features.yaml"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    for field in (
        "customer_unique_id",
        "as_of_date",
        "recency_days",
        "frequency",
        "monetary",
        "r_score",
        "f_score",
        "m_score",
        "feature_set_version",
        "preferred_payment_type",
    ):
        assert field in text
