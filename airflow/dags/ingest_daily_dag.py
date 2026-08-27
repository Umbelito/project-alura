"""
DAG: ingest_daily

Ingestão incremental diária — cadência independente do treino e do scoring.
Apenas metadados no XCom.
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.decorators import task
from airflow.operators.empty import EmptyOperator

_DAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (
        candidate
        for candidate in (_DAG_DIR.parents[1], _DAG_DIR.parents[2])
        if (candidate / "src" / "customer_segmentation").is_dir()
    ),
    _DAG_DIR.parents[2],
)
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "data-platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}


with DAG(
    dag_id="ingest_daily",
    description="Ingestão da partição do mês corrente (landing → processed).",
    default_args=DEFAULT_ARGS,
    schedule="0 3 * * *",
    start_date=datetime(2018, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["aula-05", "ingestion"],
    doc_md=__doc__,
) as dag:
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    @task(task_id="ingest_current_month")
    def ingest_current_month(**context) -> dict:
        from customer_segmentation.ingestion.incremental import ingest_customers_snapshot, ingest_month_partition

        logical = context.get("data_interval_end") or datetime.now(timezone.utc)
        if isinstance(logical, datetime):
            day = logical.date()
        elif isinstance(logical, date):
            day = logical
        else:
            day = date.today()
        year, month = day.year, day.month
        refs = []
        for dataset in ("orders", "order_items", "order_payments"):
            try:
                refs.append(ingest_month_partition(dataset, year, month))
            except FileNotFoundError:
                logger.warning("partição ausente %s %04d-%02d", dataset, year, month)
        customers = ingest_customers_snapshot()
        return {
            "year": year,
            "month": month,
            "n_partitions": len(refs),
            "customers_uri": customers.get("uri"),
        }

    start >> ingest_current_month() >> end
