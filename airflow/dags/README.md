# DAGs Airflow

## Aula 2

| DAG | Schedule | Arquivo |
|---|---|---|
| `customer_feature_table` | `0 5 * * 1` | [`customer_feature_table_dag.py`](./customer_feature_table_dag.py) |

## Aula 3

| DAG | Schedule | Arquivo |
|---|---|---|
| `train_evaluate_register` | `0 4 1 * *` | [`train_evaluate_dag.py`](./train_evaluate_dag.py) |

## Aula 4

| DAG | Schedule | Arquivo |
|---|---|---|
| `quality_gates_promote` | `30 4 1 * *` | [`quality_gates_dag.py`](./quality_gates_dag.py) |

## Aula 5

| DAG | Schedule | Arquivo |
|---|---|---|
| `ingest_daily` | `0 3 * * *` | [`ingest_daily_dag.py`](./ingest_daily_dag.py) |
| `batch_score_champion` | `0 7 * * 1` | [`batch_score_dag.py`](./batch_score_dag.py) |

Scoring **não** treina: carrega `@champion` e publica `data/scores/`.

```bash
python scripts/run_batch_scoring.py --as-of-date 2018-08-31
docker compose up -d
```

Cadências: `configs/cadences.yaml`.
