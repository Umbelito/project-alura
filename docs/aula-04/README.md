# Aula 4 — Testes e quality gates do sistema de ML

## Entregável

Suíte automatizada (pytest + ruff + mypy + coverage) e **política de aprovação/rejeição** que só promove o alias `champion` se todos os gates blocking passarem.

## Suíte

| Camada | Onde |
|---|---|
| Transformações / RFM | `tests/unit/test_feature_transforms.py` |
| Contratos entrada/saída | `tests/contracts/` |
| DAGs (AST, sem Airflow) | `tests/unit/test_dag_structure.py` |
| DAGs (DagBag, se Airflow) | `tests/unit/test_dag_parse.py` |
| Inferência / clusters | `tests/unit/test_scoring.py` |
| Assinatura | `tests/unit/test_scoring.py` (`check_signature_compatibility`) |
| Política de promoção | `tests/unit/test_promotion.py` |
| Integração + smoke | `tests/integration/test_mlflow_storage.py` |

```bash
pytest tests/unit tests/contracts tests/integration -q
ruff check src tests scripts api
mypy src/customer_segmentation --ignore-missing-imports
pytest tests --cov=customer_segmentation --cov-fail-under=50
```

CI: [`.github/workflows/quality-gates.yml`](../../.github/workflows/quality-gates.yml).

## Política (`configs/quality_gates.yaml`)

| Gate | Impede promoção se |
|---|---|
| `data_failure` | Feature table inválida (schema/PK/nulos) |
| `metric_regression` | Silhouette < mínimo ou queda > 0,05 vs champion |
| `unstable_or_empty_clusters` | Cluster pequeno, k fora de 3–10 ou ARI < 0,50 |
| `signature_incompatible` | Colunas divergem ou `predict` quebra com schema errado |
| `integration_failure` | Falha ao persistir scores / recarregar parquet / smoke |

O candidato **sempre** permanece em `@candidate`. `@champion` só muda se `decision=approved`.

## Tags MLflow

Prefixo `quality_gate_`: `decision`, `approved`, cada check `pass|fail` + `_detail`, `blocking_failed`.

## Como executar

```bash
python scripts/run_quality_gates.py --as-of-date 2018-08-31 -v
# exit 0 aprovado; exit 2 rejeitado
```

DAG `quality_gates_promote` (`30 4 1 * *`) falha a task se a política rejeitar.

## Critérios de aceite

Ver [criterios-aceite.md](./criterios-aceite.md).
