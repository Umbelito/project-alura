"""Smoke test: DAG Airflow importa sem erro (se apache-airflow estiver instalado)."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest

DAG_FILE = Path(__file__).resolve().parents[2] / "airflow" / "dags" / "customer_feature_table_dag.py"

_AIRFLOW_READY = find_spec("airflow") is not None and find_spec("airflow.models") is not None


@pytest.mark.skipif(not _AIRFLOW_READY, reason="apache-airflow não instalado no ambiente de testes")
def test_dag_file_loads() -> None:
    from airflow.models import DagBag

    dagbag = DagBag(
        dag_folder=str(DAG_FILE.parent),
        include_examples=False,
        safe_mode=True,
    )
    assert "customer_feature_table" in dagbag.dags
    assert not dagbag.import_errors, dagbag.import_errors
    dag = dagbag.dags["customer_feature_table"]
    assert dag.catchup is True
    assert dag.max_active_runs == 1
    task_ids = {t.task_id for t in dag.tasks}
    assert "ingest_incremental" in task_ids
    assert "build_rfm_features" in task_ids
