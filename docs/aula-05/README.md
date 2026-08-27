# Aula 5 — Batch scoring e disponibilização da segmentação

## Entregável

Pipeline de scoring **separado do treino** e API que consulta a tabela histórica `customer_segments`.

## Cadências independentes

| Pipeline | DAG | Cron |
|---|---|---|
| Ingestão | `ingest_daily` | `0 3 * * *` |
| Features | `customer_feature_table` | `0 5 * * 1` |
| Treino | `train_evaluate_register` | `0 4 1 * *` |
| Quality gates | `quality_gates_promote` | `30 4 1 * *` |
| Scoring | `batch_score_champion` | `0 7 * * 1` |

O scoring **não treina**. Carrega `models:/customer_segmentation_kmeans@champion`.

## Histórico e versão

```
data/scores/as_of_date=YYYY-MM-DD/customer_segments.parquet
data/scores/as_of_date=YYYY-MM-DD/manifest.json
data/scores/current.json          # ponteiro da API
```

Cada linha traz `model_name`, `model_version`, `scored_at`, `as_of_date`.

## API

| Método | Caminho | Resposta |
|---|---|---|
| GET | `/health` | status, scores_available, as_of_date, model_version, n_customers |
| GET | `/segments/{customer_unique_id}` | segmento, updated_at, model_version (404 se ausente) |

```bash
python scripts/run_batch_scoring.py --as-of-date 2018-08-31
uvicorn api.main:app --reload --port 8000
```

## Docker Compose (stack completa)

Serviços: **API**, **MLflow**, **Postgres**, **Airflow** (web + scheduler), volume `./data` (armazenamento).

```bash
docker compose up -d
# API     http://localhost:8000/health
# MLflow  http://localhost:5000
# Airflow http://localhost:8080  (admin/admin)
```

## Validação ponta a ponta

1. Feature table existente (`scripts/run_feature_table.py --as-of-date 2018-08-31`).
2. Champion no registry (Aulas 3–4).
3. `python scripts/run_batch_scoring.py --as-of-date 2018-08-31` → `data/scores/as_of_date=2018-08-31/` + `current.json`.
4. `GET /health` com `scores_available: true` e `model_version`.
5. `GET /segments/{id}` com `segment_label`, `updated_at`, `model_version`.
6. `pytest tests/unit tests/contracts tests/integration -q`.

Treino e scoring **não** compartilham DAG: cadências e XComs são independentes.
