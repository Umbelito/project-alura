# Aula 2 — Orquestração e qualidade de dados com Airflow

## Entregável

DAG reprocessável `customer_feature_table` que gera uma **feature table RFM validada e versionada** em:

```
data/features/v{feature_set_version}/as_of_date={YYYY-MM-DD}/
  customer_features.parquet
  manifest.json
  _SUCCESS
```

## Fluxo da DAG

```
start
  → resolve_run_params          # data_interval + Params
  → ingest_incremental          # landing CSV → processed parquet (idempotente)
  → validate_and_join           # schema/PK/FK + limpeza + join
  → build_rfm_features          # RFM + comportamentais + validação
  → publish_feature_manifest
  → end
```

Entre tarefas circulam **apenas referências** (`DatasetRef`: URI, row_count, versão, partição). Os DataFrames permanecem no armazenamento compartilhado (`data/`).

## Qualidade

Checks executáveis em `src/customer_segmentation/validation/quality.py`:

| Check | Onde |
|---|---|
| Schema / colunas obrigatórias | contratos YAML |
| `not_null` | colunas `nullable: false` |
| Unicidade de PK | `primary_key` do contrato |
| `allowed_values` | ex. `order_status`, `payment_type` |
| Integridade referencial | orders↔customers, items/payments↔orders |
| Feature table | frequency>0, recency≥0, monetary≥0, PK |

## Features geradas

**RFM:** `recency_days`, `frequency`, `monetary`, scores R/F/M (quintis), `rfm_score`.

**Comportamentais:** `avg_ticket`, `avg_items_per_order`, `avg_freight`, `avg_installments`, `preferred_payment_type`, `customer_state`, `customer_age_days`, `avg_days_between_orders`.

Contrato: [`contracts/output/customer_features.yaml`](../../contracts/output/customer_features.yaml).

## Data intervals, parâmetros e backfill

| Config | Valor |
|---|---|
| `schedule` | `0 5 * * 1` (segunda 05:00) |
| `start_date` | `2018-01-01` |
| `catchup` | `True` (backfill) |
| `max_active_runs` | `1` |
| Params | `as_of_date`, `window_days`, `feature_set_version`, `order_status_filter` |

`as_of_date` default = `data_interval_end - 1 day` (domingo anterior ao run de segunda).

Backfill no Airflow UI: *Trigger DAG* com intervalo histórico, ou CLI:

```bash
airflow dags backfill customer_feature_table -s 2018-06-01 -e 2018-09-01
```

Sem Airflow (dev local):

```bash
python scripts/run_feature_table.py --as-of-date 2018-08-31 -v
```

## Idempotência e resiliência

- Escrita parquet via arquivo `.tmp` + replace atômico
- Marcador `_SUCCESS` por partição
- Reexecução sobrescreve a mesma partição `as_of_date` / `year=month`
- `retries=2`, backoff exponencial, `execution_timeout=45min`
- Logging estruturado por etapa (dataset, check, URI, rows)

## Por que só metadados entre tasks?

O Airflow **não garante** que tarefas consecutivas rodem no mesmo worker. XComs grandes degradam o metastore. A documentação recomenda persistência externa e troca de metadados leves — exatamente o padrão `DatasetRef` deste projeto.

## Como subir o Airflow (local)

```bash
docker compose up -d
# UI: http://localhost:8080  (admin / admin)
```

O volume monta `./airflow/dags`, `./src`, `./data`, `./contracts`, `./configs`.

## Critérios de aceite

- [x] DAG de ingestão/features incremental e reprocessável
- [x] Validação de schema, obrigatórios, duplicidades e FK
- [x] Limpeza e junção das fontes
- [x] Features RFM + comportamentais
- [x] Persistência versionada da feature table
- [x] Data intervals, params e backfill (`catchup`)
- [x] Tarefas idempotentes
- [x] Retries, timeouts, `max_active_runs`
- [x] Logging / falhas explícitas na validação
- [x] Passagem apenas de referências entre tarefas
