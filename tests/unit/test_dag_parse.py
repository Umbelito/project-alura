"""Smoke test: DAGs Airflow importam sem erro (se apache-airflow estiver instalado)."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest

DAGS_DIR = Path(__file__).resolve().parents[2] / "airflow" / "dags"

_AIRFLOW_READY = find_spec("airflow") is not None and find_spec("airflow.models") is not None


@pytest.mark.skipif(not _AIRFLOW_READY, reason="apache-airflow não instalado no ambiente de testes")
def test_dag_files_load() -> None:
    from airflow.models import DagBag

    dagbag = DagBag(
        dag_folder=str(DAGS_DIR),
        include_examples=False,
        safe_mode=True,
    )
    assert not dagbag.import_errors, dagbag.import_errors
    assert "customer_feature_table" in dagbag.dags
    assert "train_evaluate_register" in dagbag.dags
    assert "quality_gates_promote" in dagbag.dags
    train = dagbag.dags["train_evaluate_register"]
    assert train.max_active_runs == 1
    assert "run_experiment" in {t.task_id for t in train.tasks}
