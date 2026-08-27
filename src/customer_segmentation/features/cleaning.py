"""Limpeza e junção das fontes Olist para engenharia de features."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd

from customer_segmentation.ingestion.incremental import (
    load_processed_partitions,
    months_in_window,
)
from customer_segmentation.storage import (
    processed_customers_path,
    processed_joined_path,
    write_parquet_idempotent,
)
from customer_segmentation.validation.quality import (
    assert_valid,
    validate_dataframe,
    validate_referential_integrity,
)

logger = logging.getLogger(__name__)


def _parse_timestamps(orders: pd.DataFrame) -> pd.DataFrame:
    out = orders.copy()
    for col in (
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    df = _parse_timestamps(orders)
    before = len(df)
    df = df.drop_duplicates(subset=["order_id"], keep="first")
    df = df.dropna(subset=["order_id", "customer_id", "order_purchase_timestamp"])
    logger.info("orders limpos: %s -> %s", before, len(df))
    return df


def clean_order_items(items: pd.DataFrame) -> pd.DataFrame:
    df = items.copy()
    before = len(df)
    df = df.drop_duplicates(subset=["order_id", "order_item_id"], keep="first")
    df = df.dropna(subset=["order_id", "order_item_id", "price"])
    df = df[df["price"] >= 0]
    if "freight_value" in df.columns:
        df = df[df["freight_value"] >= 0]
    logger.info("order_items limpos: %s -> %s", before, len(df))
    return df


def clean_order_payments(payments: pd.DataFrame) -> pd.DataFrame:
    df = payments.copy()
    before = len(df)
    df = df.drop_duplicates(subset=["order_id", "payment_sequential"], keep="first")
    df = df.dropna(subset=["order_id", "payment_sequential", "payment_value"])
    df = df[df["payment_value"] >= 0]
    logger.info("order_payments limpos: %s -> %s", before, len(df))
    return df


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    df = customers.copy()
    before = len(df)
    df = df.drop_duplicates(subset=["customer_id"], keep="first")
    df = df.dropna(subset=["customer_id", "customer_unique_id"])
    logger.info("customers limpos: %s -> %s", before, len(df))
    return df


def assert_referential_integrity(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    items: pd.DataFrame,
    payments: pd.DataFrame,
) -> None:
    """Valida FKs sobre o universo limpo (antes do filtro de negócio)."""
    checks = [
        validate_referential_integrity(
            orders,
            customers,
            ["customer_id"],
            ["customer_id"],
            "fk_orders_customers",
        ),
        validate_referential_integrity(
            items,
            orders,
            ["order_id"],
            ["order_id"],
            "fk_items_orders",
        ),
        validate_referential_integrity(
            payments,
            orders,
            ["order_id"],
            ["order_id"],
            "fk_payments_orders",
        ),
    ]
    for check in checks:
        level = logging.ERROR if not check.passed else logging.INFO
        logger.log(level, "%s passed=%s detail=%s", check.name, check.passed, check.detail)
        if not check.passed:
            raise ValueError(f"Integridade referencial falhou: {check.name} — {check.detail}")


def join_sources(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    items: pd.DataFrame,
    payments: pd.DataFrame,
) -> pd.DataFrame:
    """Junta dimensões e fatos já limpos/filtrados (granularidade de item)."""
    pay_agg = (
        payments.groupby("order_id", as_index=False)
        .agg(
            payment_value_total=("payment_value", "sum"),
            payment_installments_avg=("payment_installments", "mean"),
            payment_types_nunique=("payment_type", "nunique"),
            payment_type_mode=("payment_type", lambda s: s.mode().iloc[0] if len(s) else None),
        )
    )

    enriched = (
        items.merge(orders, on="order_id", how="inner")
        .merge(
            customers[["customer_id", "customer_unique_id", "customer_state"]],
            on="customer_id",
            how="inner",
        )
        .merge(pay_agg, on="order_id", how="left")
    )
    logger.info("orders_enriched rows=%s cols=%s", len(enriched), list(enriched.columns))
    return enriched


def build_and_persist_joined(
    as_of: date,
    window_days: int = 365,
    order_status_filter: list[str] | None = None,
) -> dict[str, Any]:
    """Carrega processed, valida, limpa, junta e persiste dataset intermediário."""
    status_filter = order_status_filter or ["delivered"]
    months = months_in_window(as_of, window_days)

    orders_raw = load_processed_partitions("orders", months)
    items_raw = load_processed_partitions("order_items", months)
    payments_raw = load_processed_partitions("order_payments", months)
    customers_raw = pd.read_parquet(processed_customers_path())

    for name, frame in (
        ("orders", orders_raw),
        ("order_items", items_raw),
        ("order_payments", payments_raw),
        ("customers", customers_raw),
    ):
        assert_valid(validate_dataframe(frame, name))

    orders = clean_orders(orders_raw)
    items = clean_order_items(items_raw)
    payments = clean_order_payments(payments_raw)
    customers = clean_customers(customers_raw)

    # FK no universo completo da janela (antes do filtro delivered)
    assert_referential_integrity(orders, customers, items, payments)

    as_of_ts = pd.Timestamp(as_of) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    start_ts = as_of_ts - pd.Timedelta(days=window_days)
    orders_f = orders[
        (orders["order_purchase_timestamp"] >= start_ts)
        & (orders["order_purchase_timestamp"] <= as_of_ts)
        & (orders["order_status"].isin(status_filter))
    ].copy()
    order_ids = set(orders_f["order_id"])
    items_f = items[items["order_id"].isin(order_ids)].copy()
    payments_f = payments[payments["order_id"].isin(order_ids)].copy()

    logger.info(
        "filtro negócio as_of=%s window=%s status=%s orders=%s items=%s payments=%s",
        as_of,
        window_days,
        status_filter,
        len(orders_f),
        len(items_f),
        len(payments_f),
    )

    enriched = join_sources(orders_f, customers, items_f, payments_f)
    path = processed_joined_path(as_of)
    ref = write_parquet_idempotent(enriched, path)
    ref.as_of_date = as_of.isoformat()
    ref.dataset = "orders_enriched"
    ref.partition = {"as_of_date": as_of.isoformat(), "window_days": window_days}
    return ref.to_dict()
