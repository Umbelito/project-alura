# DAGs Airflow

## Aula 2

| DAG | Schedule | Arquivo |
|---|---|---|
| `customer_feature_table` | `0 5 * * 1` | [`customer_feature_table_dag.py`](./customer_feature_table_dag.py) |

Gera feature table RFM validada e versionada. Detalhes: [`docs/aula-02/README.md`](../docs/aula-02/README.md).

```bash
python scripts/run_feature_table.py --as-of-date 2018-08-31
```

## Aula 3

| DAG | Schedule | Arquivo |
|---|---|---|
| `train_evaluate_register` | `0 4 1 * *` | [`train_evaluate_dag.py`](./train_evaluate_dag.py) |

Busca de clustering, estabilidade e registro MLflow (`candidate` / `champion`).
Detalhes: [`docs/aula-03/README.md`](../docs/aula-03/README.md).

```bash
python scripts/run_training_experiment.py --as-of-date 2018-08-31 --write-docs-report
```

### Compose

```bash
docker compose --profile airflow up -d
docker compose up -d mlflow
```

## Próximas aulas

- `features_score_dag.py` — scoring semanal com alias champion  
- `monitor_dag.py` — freshness / drift / health  

Cadências: `configs/cadences.yaml`.
