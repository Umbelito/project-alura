"""Estrutura das DAGs mesmo sem apache-airflow instalado (Aula 4)."""

from __future__ import annotations

import ast
from pathlib import Path

DAGS_DIR = Path(__file__).resolve().parents[2] / "airflow" / "dags"

EXPECTED = {
    "customer_feature_table_dag.py": {
        "dag_id": "customer_feature_table",
        "tasks": ["ingest_incremental", "build_rfm_features", "validate_and_join"],
    },
    "train_evaluate_dag.py": {
        "dag_id": "train_evaluate_register",
        "tasks": ["run_experiment"],
    },
    "quality_gates_dag.py": {
        "dag_id": "quality_gates_promote",
        "tasks": ["run_quality_gates"],
    },
}


def _string_constants(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)
        if isinstance(node, ast.keyword) and node.arg in {"dag_id", "task_id"}:
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                found.add(node.value.value)
    return found


def test_dag_files_exist() -> None:
    for name in EXPECTED:
        assert (DAGS_DIR / name).is_file()


def test_dag_ids_and_task_structure() -> None:
    for filename, spec in EXPECTED.items():
        tree = ast.parse((DAGS_DIR / filename).read_text(encoding="utf-8"))
        constants = _string_constants(tree)
        assert spec["dag_id"] in constants
        for task_id in spec["tasks"]:
            assert task_id in constants, f"{filename} sem task {task_id}"


def test_dags_do_not_pass_dataframes_in_docstring() -> None:
    """Documentação das DAGs reforça XCom só com metadados."""
    text = (DAGS_DIR / "customer_feature_table_dag.py").read_text(encoding="utf-8")
    assert "metadados" in text.lower() or "DatasetRef" in text
