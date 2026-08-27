# DAGs Airflow

## Aula 2

| DAG | Schedule | Arquivo |
|---|---|---|
| `customer_feature_table` | `0 5 * * 1` | [`customer_feature_table_dag.py`](./customer_feature_table_dag.py) |

```bash
python scripts/run_feature_table.py --as-of-date 2018-08-31
```

## Aula 3

| DAG | Schedule | Arquivo |
|---|---|---|
| `train_evaluate_register` | `0 4 1 * *` | [`train_evaluate_dag.py`](./train_evaluate_dag.py) |

```bash
python scripts/run_training_experiment.py --as-of-date 2018-08-31 --write-docs-report
```

## Aula 4

| DAG | Schedule | Arquivo |
|---|---|---|
| `quality_gates_promote` | `30 4 1 * *` | [`quality_gates_dag.py`](./quality_gates_dag.py) |

Política de aprovação/rejeição. Detalhes: [`docs/aula-04/README.md`](../docs/aula-04/README.md).

```bash
python scripts/run_quality_gates.py --as-of-date 2018-08-31
```

### Compose

```bash
docker compose --profile airflow up -d
docker compose up -d mlflow
```

## Próximas aulas

- Scoring semanal operacional com alias champion  
- Monitoramento de drift  

Cadências: `configs/cadences.yaml`.
