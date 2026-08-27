"""Pipeline orquestrável da feature table (chamável fora do Airflow)."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from customer_segmentation.features.cleaning import build_and_persist_joined
from customer_segmentation.features.rfm import FEATURE_SET_VERSION, build_and_persist_features
from customer_segmentation.ingestion.incremental import ingest_window

logger = logging.getLogger(__name__)


def parse_as_of_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def run_feature_table_pipeline(
    as_of_date: str | date,
    window_days: int = 365,
    feature_set_version: str = FEATURE_SET_VERSION,
    order_status_filter: list[str] | None = None,
) -> dict[str, Any]:
    """Executa ingestão → join → RFM de ponta a ponta (idempotente por as_of_date)."""
    as_of = parse_as_of_date(as_of_date)
    logger.info(
        "Início pipeline feature table as_of=%s window=%s version=%s",
        as_of,
        window_days,
        feature_set_version,
    )

    ingest_meta = ingest_window(as_of, window_days=window_days)
    joined_ref = build_and_persist_joined(
        as_of,
        window_days=window_days,
        order_status_filter=order_status_filter or ["delivered"],
    )
    features_ref = build_and_persist_features(
        as_of,
        feature_set_version=feature_set_version,
        enriched_uri=joined_ref["uri"],
    )

    result = {
        "as_of_date": as_of.isoformat(),
        "window_days": window_days,
        "feature_set_version": feature_set_version,
        "ingest": {
            "months": ingest_meta["months"],
            "missing_partitions": ingest_meta["missing_partitions"],
        },
        "orders_enriched": joined_ref,
        "customer_features": features_ref,
    }
    logger.info("Pipeline concluído features_uri=%s", features_ref["uri"])
    return result
