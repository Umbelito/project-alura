# DAGs Airflow

## Aula 2

| DAG | Schedule | Arquivo |
|---|---|---|
| `customer_feature_table` | `0 5 * * 1` | [`customer_feature_table_dag.py`](./customer_feature_table_dag.py) |

Gera feature table RFM validada e versionada. Detalhes: [`docs/aula-02/README.md`](../docs/aula-02/README.md).

### Execução local sem scheduler

```bash
python scripts/run_feature_table.py --as-of-date 2018-08-31
```

### Compose

```bash
docker compose --profile airflow up -d
```

## Próximas aulas

- `features_score_dag.py` — scoring semanal com modelo Production  
- `train_promote_dag.py` — retreino mensal + gates  
- `monitor_dag.py` — freshness / drift / health  

Cadências: `configs/cadences.yaml`.
