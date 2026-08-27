"""Engenharia de features RFM e variáveis comportamentais."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from customer_segmentation.storage import (
    DatasetRef,
    features_dir,
    features_table_path,
    processed_joined_path,
    write_manifest,
    write_parquet_idempotent,
)
from customer_segmentation.validation.quality import assert_valid, validate_feature_table

logger = logging.getLogger(__name__)

FEATURE_SET_VERSION = "1.0.0"


def _quintile_scores(series: pd.Series, ascending: bool) -> pd.Series:
    """Score 1–5 por quintis; empates usam rank promedio."""
    if series.nunique(dropna=True) < 2:
        return pd.Series(np.ones(len(series), dtype=int), index=series.index)
    ranked = series.rank(method="first", ascending=ascending)
    try:
        return pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5]).astype(int)
    except ValueError:
        # Poucos valores distintos: fallback por rank percentual
        pct = series.rank(pct=True, ascending=ascending)
        return pd.cut(pct, bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5]).astype(int)


def compute_rfm_features(enriched: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Agrega features RFM + comportamentais por customer_unique_id."""
    if enriched.empty:
        raise ValueError("Dataset enriched vazio — impossível gerar features RFM")

    df = enriched.copy()
    if "order_purchase_timestamp" not in df.columns:
        raise KeyError("order_purchase_timestamp ausente no enriched")

    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    as_of_ts = pd.Timestamp(as_of)

    order_level = (
        df.groupby(["customer_unique_id", "order_id"], as_index=False)
        .agg(
            order_purchase_timestamp=("order_purchase_timestamp", "min"),
            monetary_items=("price", "sum"),
            items_count=("order_item_id", "count"),
            freight_sum=("freight_value", "sum"),
            payment_value_total=("payment_value_total", "first"),
            payment_installments_avg=("payment_installments_avg", "first"),
            payment_type_mode=("payment_type_mode", "first"),
            customer_state=("customer_state", "first"),
        )
    )

    grouped = order_level.groupby("customer_unique_id")
    features = grouped.agg(
        recency_days=(
            "order_purchase_timestamp",
            lambda s: int((as_of_ts - s.max()).days),
        ),
        frequency=("order_id", "nunique"),
        monetary=("monetary_items", "sum"),
        avg_ticket=("monetary_items", "mean"),
        avg_items_per_order=("items_count", "mean"),
        avg_freight=("freight_sum", "mean"),
        avg_installments=("payment_installments_avg", "mean"),
        first_purchase=("order_purchase_timestamp", "min"),
        last_purchase=("order_purchase_timestamp", "max"),
        customer_state=("customer_state", "first"),
    ).reset_index()

    # Preferência de pagamento (moda entre pedidos)
    pay_mode = (
        order_level.dropna(subset=["payment_type_mode"])
        .groupby("customer_unique_id")["payment_type_mode"]
        .agg(lambda s: s.mode().iloc[0] if len(s) else None)
        .rename("preferred_payment_type")
    )
    features = features.merge(pay_mode, on="customer_unique_id", how="left")

    features["customer_age_days"] = (
        features["last_purchase"] - features["first_purchase"]
    ).dt.days.fillna(0).astype(int)
    features["avg_days_between_orders"] = np.where(
        features["frequency"] > 1,
        features["customer_age_days"] / (features["frequency"] - 1),
        np.nan,
    )

    # Recency: menor dias = melhor cliente → score 5 (ascending=False no rank).
    features["r_score"] = _quintile_scores(features["recency_days"], ascending=False)
    features["f_score"] = _quintile_scores(features["frequency"], ascending=True)
    features["m_score"] = _quintile_scores(features["monetary"], ascending=True)
    features["rfm_score"] = (
        features["r_score"].astype(str)
        + features["f_score"].astype(str)
        + features["m_score"].astype(str)
    )

    features["as_of_date"] = as_of.isoformat()
    features["feature_set_version"] = FEATURE_SET_VERSION
    features["built_at"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds").replace("+00:00", "Z")

    # Tipagem estável
    features["recency_days"] = features["recency_days"].astype(int)
    features["frequency"] = features["frequency"].astype(int)
    features["monetary"] = features["monetary"].astype(float)

    logger.info(
        "Features RFM geradas customers=%s as_of=%s version=%s",
        len(features),
        as_of,
        FEATURE_SET_VERSION,
    )
    return features


FEATURE_COLUMNS = [
    "customer_unique_id",
    "as_of_date",
    "recency_days",
    "frequency",
    "monetary",
    "r_score",
    "f_score",
    "m_score",
    "rfm_score",
    "avg_ticket",
    "avg_items_per_order",
    "avg_freight",
    "avg_installments",
    "preferred_payment_type",
    "customer_state",
    "customer_age_days",
    "avg_days_between_orders",
    "first_purchase",
    "last_purchase",
    "feature_set_version",
    "built_at",
]


def build_and_persist_features(
    as_of: date,
    feature_set_version: str = FEATURE_SET_VERSION,
    enriched_uri: str | None = None,
) -> dict[str, Any]:
    """Lê orders_enriched, calcula RFM, valida e persiste feature table versionada."""
    if enriched_uri:
        path = DatasetRef(uri=enriched_uri, dataset="orders_enriched").absolute_path()
    else:
        path = processed_joined_path(as_of)

    if not path.is_file():
        raise FileNotFoundError(f"orders_enriched ausente: {path}")

    enriched = pd.read_parquet(path)
    features = compute_rfm_features(enriched, as_of)
    features["feature_set_version"] = feature_set_version
    features = features[[c for c in FEATURE_COLUMNS if c in features.columns]]

    report = validate_feature_table(features)
    assert_valid(report)

    out = features_table_path(feature_set_version, as_of)
    ref = write_parquet_idempotent(features, out)
    ref.as_of_date = as_of.isoformat()
    ref.feature_set_version = feature_set_version
    ref.dataset = "customer_features"
    ref.partition = {"as_of_date": as_of.isoformat()}

    write_manifest(
        features_dir(feature_set_version, as_of),
        {
            "dataset": "customer_features",
            "feature_set_version": feature_set_version,
            "as_of_date": as_of.isoformat(),
            "uri": ref.uri,
            "row_count": ref.row_count,
            "validation": report.to_dict(),
            "source_enriched_uri": enriched_uri or str(path).replace("\\", "/"),
        },
    )
    logger.info("Feature table versionada uri=%s", ref.uri)
    return ref.to_dict()
