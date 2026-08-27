"""
DAG: quality_gates_promote

Aula 4 — suíte de quality gates e política de aprovação/rejeição.
Impede promoção a champion se dados, métricas, estabilidade, assinatura
ou integração falharem. XCom só com metadados da decisão.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
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
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=45),
}


with DAG(
    dag_id="quality_gates_promote",
    description="Quality gates: dados, regressão, estabilidade, assinatura, integração → champion ou rejeição.",
    default_args=DEFAULT_ARGS,
    schedule="30 4 1 * *",  # após o treino mensal (04:00)
    start_date=datetime(2018, 8, 1),
    catchup=False,
    max_active_runs=1,
    tags=["aula-04", "quality-gates", "mlflow"],
    params={
        "as_of_date": Param(default="2018-08-31", type="string"),
        "feature_set_version": Param(default="1.0.0", type="string"),
        "model_uri": Param(default="", type=["null", "string"]),
    },
    doc_md=__doc__,
) as dag:
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    @task(task_id="run_quality_gates")
    def run_gates(**context) -> dict:
        from customer_segmentation.training.gates_run import run_quality_gates

        params = context["params"]
        model_uri = params.get("model_uri") or None
        result = run_quality_gates(
            as_of_date=str(params.get("as_of_date") or "2018-08-31")[:10],
            feature_set_version=str(params.get("feature_set_version") or "1.0.0"),
            model_uri=str(model_uri) if model_uri else None,
            promote=True,
        )
        decision = result["report"]["decision"]
        logger.info("quality gates decision=%s failed=%s", decision, result["report"]["blocking_failed"])
        if not result["report"]["approved"]:
            raise ValueError(
                f"Promoção rejeitada: {result['report']['blocking_failed']}"
            )
        return {
            "decision": decision,
            "model_version": result["model_version"],
            "champion_promoted": (result.get("promotion") or {}).get("champion_promoted"),
            "scores_uri": (result.get("scores_ref") or {}).get("uri"),
        }

    gates = run_gates()
    start >> gates >> end
