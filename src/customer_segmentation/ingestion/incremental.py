"""Ingestão incremental de partições mensais (landing → processed)."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd

from customer_segmentation.storage import (
    DatasetRef,
    landing_customers_path,
    landing_partition_path,
    processed_customers_path,
    processed_partition_path,
    relative_uri,
    write_parquet_idempotent,
)

logger = logging.getLogger(__name__)

INCREMENTAL_DATASETS = ("orders", "order_items", "order_payments")


def _read_landing_csv(path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Partição de landing ausente: {path}")
    df = pd.read_csv(path)
    logger.info("Lido landing uri=%s rows=%s", relative_uri(path), len(df))
    return df


def ingest_month_partition(dataset: str, year: int, month: int) -> dict[str, Any]:
    """Copia CSV da landing para parquet em processed (idempotente por partição)."""
    if dataset not in INCREMENTAL_DATASETS:
        raise ValueError(f"Dataset incremental inválido: {dataset}")

    src = landing_partition_path(dataset, year, month)
    dst = processed_partition_path(dataset, year, month)
    df = _read_landing_csv(src)
    ref = write_parquet_idempotent(df, dst)
    ref.dataset = dataset
    ref.partition = {"year": year, "month": month}
    logger.info(
        "Ingestão concluída dataset=%s partition=%04d-%02d rows=%s",
        dataset,
        year,
        month,
        ref.row_count,
    )
    return ref.to_dict()


def ingest_customers_snapshot() -> dict[str, Any]:
    """Persiste snapshot completo de customers (dimensão de referência)."""
    src = landing_customers_path()
    dst = processed_customers_path()
    df = _read_landing_csv(src)
    ref = write_parquet_idempotent(df, dst)
    ref.dataset = "customers"
    ref.partition = {"snapshot": "full"}
    return ref.to_dict()


def months_in_window(as_of: date, window_days: int) -> list[tuple[int, int]]:
    """Lista (year, month) cobertos pela janela [as_of - window_days, as_of]."""
    start = as_of - pd.Timedelta(days=window_days)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(as_of)
    periods = pd.period_range(start_ts.to_period("M"), end_ts.to_period("M"), freq="M")
    return [(int(p.year), int(p.month)) for p in periods]


def ingest_window(as_of: date, window_days: int = 365) -> dict[str, Any]:
    """Ingesta todas as partições mensais necessárias para a janela RFM."""
    months = months_in_window(as_of, window_days)
    refs: dict[str, list[dict[str, Any]]] = {ds: [] for ds in INCREMENTAL_DATASETS}
    missing: list[str] = []

    for year, month in months:
        for dataset in INCREMENTAL_DATASETS:
            src = landing_partition_path(dataset, year, month)
            if not src.is_file():
                missing.append(f"{dataset}/{year:04d}-{month:02d}")
                logger.warning("Partição ausente (pulada): %s", src)
                continue
            refs[dataset].append(ingest_month_partition(dataset, year, month))

    customers_ref = ingest_customers_snapshot()
    result = {
        "as_of_date": as_of.isoformat(),
        "window_days": window_days,
        "months": [{"year": y, "month": m} for y, m in months],
        "datasets": refs,
        "customers": customers_ref,
        "missing_partitions": missing,
    }
    logger.info(
        "Janela ingerida as_of=%s months=%s missing=%s",
        as_of,
        len(months),
        len(missing),
    )
    return result


def load_processed_partitions(
    dataset: str,
    months: list[tuple[int, int]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year, month in months:
        path = processed_partition_path(dataset, year, month)
        if not path.is_file():
            logger.warning("Processed ausente: %s", path)
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
