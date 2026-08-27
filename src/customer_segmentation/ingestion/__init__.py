"""Ingestão incremental a partir de partições mensais em data/landing."""

from customer_segmentation.ingestion.incremental import (
    ingest_customers_snapshot,
    ingest_month_partition,
    ingest_window,
    months_in_window,
)

__all__ = [
    "ingest_customers_snapshot",
    "ingest_month_partition",
    "ingest_window",
    "months_in_window",
]
