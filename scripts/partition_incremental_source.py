"""
Simula uma fonte incremental particionando os CSVs Olist por mês de compra.

- orders: partição por order_purchase_timestamp (year=YYYY/month=MM)
- order_items / order_payments: herdados do mês do pedido correspondente
- customers: snapshot estático em customers/snapshot=full (dimensão de referência)

Uso:
  python scripts/partition_incremental_source.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LANDING = ROOT / "data" / "landing"


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return fieldnames, rows


def _write_partition(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _month_key(timestamp: str) -> str | None:
    if not timestamp or len(timestamp) < 7:
        return None
    return timestamp[:7]  # YYYY-MM


def partition_orders() -> dict[str, str]:
    """Retorna mapa order_id -> YYYY-MM."""
    fieldnames, rows = _read_csv(RAW / "olist_orders_dataset.csv")
    by_month: dict[str, list[dict[str, str]]] = defaultdict(list)
    order_to_month: dict[str, str] = {}

    for row in rows:
        month = _month_key(row.get("order_purchase_timestamp", ""))
        if month is None:
            continue
        by_month[month].append(row)
        order_to_month[row["order_id"]] = month

    for month, month_rows in sorted(by_month.items()):
        year, mon = month.split("-")
        out = LANDING / "orders" / f"year={year}" / f"month={mon}" / "orders.csv"
        _write_partition(out, fieldnames, month_rows)
        print(f"[orders] {month}: {len(month_rows)} -> {out.relative_to(ROOT)}")

    return order_to_month


def partition_child_table(
    dataset: str,
    filename: str,
    order_months: dict[str, str],
) -> None:
    fieldnames, rows = _read_csv(RAW / filename)
    by_month: dict[str, list[dict[str, str]]] = defaultdict(list)
    orphan = 0

    for row in rows:
        month = order_months.get(row["order_id"])
        if month is None:
            orphan += 1
            continue
        by_month[month].append(row)

    for month, month_rows in sorted(by_month.items()):
        year, mon = month.split("-")
        out = LANDING / dataset / f"year={year}" / f"month={mon}" / f"{dataset}.csv"
        _write_partition(out, fieldnames, month_rows)
        print(f"[{dataset}] {month}: {len(month_rows)} -> {out.relative_to(ROOT)}")

    if orphan:
        print(f"[{dataset}] avisos: {orphan} linhas sem pedido correspondente", file=sys.stderr)


def snapshot_customers() -> None:
    fieldnames, rows = _read_csv(RAW / "olist_customers_dataset.csv")
    out = LANDING / "customers" / "snapshot=full" / "customers.csv"
    _write_partition(out, fieldnames, rows)
    print(f"[customers] snapshot=full: {len(rows)} -> {out.relative_to(ROOT)}")


def main() -> None:
    if not (RAW / "olist_orders_dataset.csv").exists():
        raise SystemExit(f"Dados brutos não encontrados em {RAW}")

    print(f"RAW: {RAW}")
    print(f"LANDING: {LANDING}")

    order_months = partition_orders()

    partition_child_table("order_items", "olist_order_items_dataset.csv", order_months)
    partition_child_table(
        "order_payments", "olist_order_payments_dataset.csv", order_months
    )
    snapshot_customers()
    print("Particionamento concluído.")


if __name__ == "__main__":
    main()
