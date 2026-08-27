"""
DAG: customer_feature_table

Orquestra ingestão incremental → validação → limpeza/junção → features RFM.
Entregável da Aula 2: DAG reprocessável que gera feature table validada e versionada.

Princípios:
- Tarefas idempotentes por as_of_date / partição mensal
- Entre tarefas circulam apenas DatasetRef (URI + metadados)
- Datasets ficam no armazenamento compartilhado (data/)
- Retries, timeouts e max_active_runs configurados
- catchup=True habilita backfill por data interval
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.decorators import task
from airflow.models.param import Param
from airflow.operators.empty import EmptyOperator

# Resolve raiz do projeto (local: .../Projeto 2 ; Docker: /opt/airflow)
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
    "owner": "ml-platform",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(minutes=45),
}


def _as_of_from_context(params: dict, data_interval_end: datetime | None) -> date:
    """Resolve as_of_date: param explícito > domingo do data_interval_end > hoje."""
    raw = params.get("as_of_date")
    if raw:
        return date.fromisoformat(str(raw)[:10])
    if data_interval_end is not None:
        # Features semanais: fim do intervalo (segunda 05:00) → domingo anterior
        end = data_interval_end.date() if isinstance(data_interval_end, datetime) else data_interval_end
        # Se cai numa segunda, as_of = domingo (end - 1 dia)
        return end - timedelta(days=1)
    return date.today()


with DAG(
    dag_id="customer_feature_table",
    description=(
        "Ingestão incremental + qualidade + RFM → feature table versionada "
        "(apenas metadados entre tasks)."
    ),
    default_args=DEFAULT_ARGS,
    schedule="0 5 * * 1",  # segunda 05:00 — configs/cadences.yaml
    start_date=datetime(2018, 1, 1),
    catchup=True,
    max_active_runs=1,
    max_active_tasks=4,
    tags=["aula-02", "features", "rfm", "quality"],
    params={
        "as_of_date": Param(
            default="",
            type=["null", "string"],
            description="Override YYYY-MM-DD (vazio = derivado do data_interval).",
        ),
        "window_days": Param(default=365, type="integer", minimum=30, maximum=730),
        "feature_set_version": Param(default="1.0.0", type="string"),
        "order_status_filter": Param(default=["delivered"], type="array"),
    },
    doc_md=__doc__,
) as dag:
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    @task(task_id="resolve_run_params", retries=1, execution_timeout=timedelta(minutes=5))
    def resolve_run_params(**context) -> dict:
        params = context["params"]
        data_interval_end = context.get("data_interval_end")
        as_of = _as_of_from_context(params, data_interval_end)
        meta = {
            "as_of_date": as_of.isoformat(),
            "window_days": int(params.get("window_days") or 365),
            "feature_set_version": str(params.get("feature_set_version") or "1.0.0"),
            "order_status_filter": list(params.get("order_status_filter") or ["delivered"]),
            "data_interval_start": str(context.get("data_interval_start")),
            "data_interval_end": str(data_interval_end),
            "run_id": context.get("run_id"),
        }
        logger.info("Parâmetros do run: %s", meta)
        return meta

    @task(task_id="ingest_incremental", execution_timeout=timedelta(minutes=30))
    def ingest_incremental(run_meta: dict) -> dict:
        from customer_segmentation.ingestion.incremental import ingest_window

        as_of = date.fromisoformat(run_meta["as_of_date"])
        result = ingest_window(as_of, window_days=run_meta["window_days"])
        # Retorna apenas metadados (sem dataframes)
        return {
            "as_of_date": result["as_of_date"],
            "window_days": result["window_days"],
            "months": result["months"],
            "missing_partitions": result["missing_partitions"],
            "customers_uri": result["customers"]["uri"],
            "orders_partitions": len(result["datasets"]["orders"]),
            "order_items_partitions": len(result["datasets"]["order_items"]),
            "order_payments_partitions": len(result["datasets"]["order_payments"]),
        }

    @task(task_id="validate_and_join", execution_timeout=timedelta(minutes=30))
    def validate_and_join(run_meta: dict, ingest_meta: dict) -> dict:
        from customer_segmentation.features.cleaning import build_and_persist_joined

        logger.info("Ingest meta recebido: %s", ingest_meta)
        as_of = date.fromisoformat(run_meta["as_of_date"])
        ref = build_and_persist_joined(
            as_of,
            window_days=run_meta["window_days"],
            order_status_filter=run_meta["order_status_filter"],
        )
        return ref  # DatasetRef serializado (uri + row_count + ...)

    @task(task_id="build_rfm_features", execution_timeout=timedelta(minutes=30))
    def build_rfm_features(run_meta: dict, joined_ref: dict) -> dict:
        from customer_segmentation.features.rfm import build_and_persist_features

        logger.info("Joined ref: uri=%s rows=%s", joined_ref.get("uri"), joined_ref.get("row_count"))
        as_of = date.fromisoformat(run_meta["as_of_date"])
        ref = build_and_persist_features(
            as_of,
            feature_set_version=run_meta["feature_set_version"],
            enriched_uri=joined_ref["uri"],
        )
        return ref

    @task(task_id="publish_feature_manifest", execution_timeout=timedelta(minutes=5))
    def publish_feature_manifest(run_meta: dict, features_ref: dict) -> dict:
        """Fecha o run com manifesto consolidado (diagnóstico / auditoria)."""
        from customer_segmentation.storage import features_dir, write_manifest

        as_of = date.fromisoformat(run_meta["as_of_date"])
        version = run_meta["feature_set_version"]
        manifest = {
            "run_id": run_meta.get("run_id"),
            "as_of_date": run_meta["as_of_date"],
            "window_days": run_meta["window_days"],
            "feature_set_version": version,
            "features": features_ref,
            "status": "success",
        }
        path = write_manifest(features_dir(version, as_of), {**manifest, "stage": "dag_complete"})
        logger.info("Manifesto publicado em %s", path)
        return {
            "manifest_uri": str(path).replace("\\", "/"),
            "features_uri": features_ref.get("uri"),
            "row_count": features_ref.get("row_count"),
            "feature_set_version": version,
            "as_of_date": run_meta["as_of_date"],
        }

    run_params = resolve_run_params()
    ingest_out = ingest_incremental(run_params)
    joined_out = validate_and_join(run_params, ingest_out)
    features_out = build_rfm_features(run_params, joined_out)
    published = publish_feature_manifest(run_params, features_out)

    start >> run_params
    published >> end
