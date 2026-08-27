"""
DAG: train_evaluate_register

Aula 3 — experimentação, estabilidade e registro no MLflow.
Tarefas trocam apenas metadados (URI da feature table, run_id, versão do modelo).
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
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=2),
}


with DAG(
    dag_id="train_evaluate_register",
    description="Busca de clustering RFM, avaliação de estabilidade e registro MLflow (candidate/champion).",
    default_args=DEFAULT_ARGS,
    schedule="0 4 1 * *",
    start_date=datetime(2018, 8, 1),
    catchup=False,
    max_active_runs=1,
    tags=["aula-03", "training", "mlflow"],
    params={
        "as_of_date": Param(default="2018-08-31", type="string"),
        "feature_set_version": Param(default="1.0.0", type="string"),
        "quick": Param(default=False, type="boolean"),
        "set_champion": Param(default=False, type="boolean"),
        "compare_as_of_date": Param(default="", type=["null", "string"]),
    },
    doc_md=__doc__,
) as dag:
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    @task(task_id="resolve_run_params", execution_timeout=timedelta(minutes=5))
    def resolve_run_params(**context) -> dict:
        params = context["params"]
        as_of = str(params.get("as_of_date") or "2018-08-31")[:10]
        compare = params.get("compare_as_of_date") or ""
        meta = {
            "as_of_date": as_of,
            "feature_set_version": str(params.get("feature_set_version") or "1.0.0"),
            "quick": bool(params.get("quick")),
            "set_champion": bool(params.get("set_champion")),
            "compare_as_of_date": str(compare)[:10] if compare else None,
            "run_id": context.get("run_id"),
        }
        logger.info("Parâmetros de treino: %s", meta)
        return meta

    @task(task_id="run_experiment", execution_timeout=timedelta(hours=2))
    def run_experiment(run_meta: dict) -> dict:
        from customer_segmentation.training.experiment import run_training_experiment

        payload = run_training_experiment(
            as_of_date=date.fromisoformat(run_meta["as_of_date"]),
            feature_set_version=run_meta["feature_set_version"],
            compare_as_of_date=run_meta.get("compare_as_of_date"),
            quick=run_meta["quick"],
            register=True,
            set_champion=run_meta["set_champion"],
        )
        mlflow_info = payload.get("mlflow") or {}
        # Apenas metadados no XCom
        return {
            "as_of_date": run_meta["as_of_date"],
            "features_uri": payload["data"]["features_uri"],
            "parent_run_id": mlflow_info.get("parent_run_id"),
            "model_name": mlflow_info.get("model_name"),
            "model_version": mlflow_info.get("model_version"),
            "aliases": mlflow_info.get("aliases"),
            "candidate_uri": mlflow_info.get("candidate_uri"),
            "winner_run_name": payload["winner"].get("run_name"),
            "gates_passed": payload["winner"].get("gates", {}).get("passed"),
            "comparison_uri": (payload.get("artifacts") or {}).get("comparison_uri"),
        }

    run_params = resolve_run_params()
    experiment_out = run_experiment(run_params)
    start >> run_params
    experiment_out >> end
