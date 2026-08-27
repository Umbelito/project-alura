"""
DAG: batch_score_champion

Pipeline de scoring separado do treinamento (Aula 5).
Cadência semanal independente; carrega o alias champion e publica
customer_segments histórico + ponteiro current para a API.
XCom só com metadados (URI, versão do modelo, as_of_date).
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
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=1),
}


def _as_of(params: dict, data_interval_end: datetime | None) -> date:
    raw = params.get("as_of_date")
    if raw:
        return date.fromisoformat(str(raw)[:10])
    if data_interval_end is not None:
        end = data_interval_end.date() if isinstance(data_interval_end, datetime) else data_interval_end
        return end - timedelta(days=1)
    return date.today()


with DAG(
    dag_id="batch_score_champion",
    description="Scoring batch da base completa com models:/…@champion → customer_segments.",
    default_args=DEFAULT_ARGS,
    schedule="0 7 * * 1",
    start_date=datetime(2018, 8, 1),
    catchup=False,
    max_active_runs=1,
    tags=["aula-05", "scoring", "champion"],
    params={
        "as_of_date": Param(default="", type=["null", "string"]),
        "feature_set_version": Param(default="1.0.0", type="string"),
    },
    doc_md=__doc__,
) as dag:
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    @task(task_id="load_champion_meta")
    def load_champion_meta() -> dict:
        from customer_segmentation.scoring.batch import load_champion

        champ = load_champion()
        logger.info("champion v%s uri=%s", champ["model_version"], champ["model_uri"])
        return {
            "model_name": champ["model_name"],
            "model_version": champ["model_version"],
            "model_uri": champ["model_uri"],
            "run_id": champ["run_id"],
        }

    @task(task_id="score_full_base")
    def score_full_base(champ_meta: dict, **context) -> dict:
        from customer_segmentation.scoring.batch import run_batch_scoring

        as_of = _as_of(context["params"], context.get("data_interval_end"))
        result = run_batch_scoring(
            as_of_date=as_of,
            feature_set_version=str(context["params"].get("feature_set_version") or "1.0.0"),
        )
        logger.info(
            "scoring as_of=%s rows=%s model_version=%s",
            result["as_of_date"],
            result["n_customers"],
            result["model_version"],
        )
        return {
            "as_of_date": result["as_of_date"],
            "scores_uri": result["scores"]["uri"],
            "n_customers": result["n_customers"],
            "model_version": result["model_version"],
            "model_name": result["model_name"],
            "champion": champ_meta,
        }

    meta = load_champion_meta()
    scored = score_full_base(meta)
    start >> meta
    scored >> end
