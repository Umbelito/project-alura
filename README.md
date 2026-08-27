# Segmentação de Clientes — MLOps (Projeto 2)

Sistema batch de machine learning para segmentar clientes com features RFM, orquestrado com Airflow, rastreado com MLflow e consumido via FastAPI.

**Dataset:** [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

---

## Objetivo

Transformar o experimento de Ciência de Dados em um pipeline reprodutível e operável: ingerir dados, validar qualidade, gerar features, treinar/avaliar modelos, registrar versões, segmentar clientes e disponibilizar o resultado para campanhas de marketing.

---

## Estrutura do repositório

```
api/                         # FastAPI — consulta de segmentos
airflow/dags/                # Ingestão, features, treino, gates, scoring
configs/                     # Cadências, paths, treino e quality gates
contracts/                   # Contratos de entrada e saída
data/
  raw/                       # CSVs Olist originais
  landing/                   # Partições mensais (fonte incremental)
  processed/                 # Parquets limpos / orders_enriched
  features/                  # Features RFM versionadas (v*/as_of_date=*)
  scores/                    # Histórico customer_segments + current.json
docs/aula-01/ … docs/aula-05/
scripts/                     # Execução local (features, treino, gates, scoring)
src/customer_segmentation/
  ingestion/
  validation/
  features/
  training/
  scoring/                   # Batch com alias champion
  serving/                   # Lookup da tabela histórica
tests/
```

---

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

Coloque os CSVs Olist em `data/raw/` (veja `data/raw/README.md`) e gere a fonte incremental:

```bash
python scripts/partition_incremental_source.py
```

### Feature table (Aula 2) — sem Airflow

```bash
python scripts/run_feature_table.py --as-of-date 2018-08-31 -v
```

Saída versionada em `data/features/v1.0.0/as_of_date=2018-08-31/`.

### Experimento MLflow (Aula 3)

```bash
python scripts/run_training_experiment.py --as-of-date 2018-08-31 --write-docs-report -v
```

### Quality gates (Aula 4)

```bash
python scripts/run_quality_gates.py --as-of-date 2018-08-31
pytest tests/unit tests/contracts tests/integration -q
ruff check src tests scripts api
```

### Scoring batch + API (Aula 5)

```bash
python scripts/run_batch_scoring.py --as-of-date 2018-08-31
uvicorn api.main:app --reload --port 8000
# GET http://localhost:8000/health
# GET http://localhost:8000/segments/{customer_unique_id}
```

### Stack local (Docker Compose)

Sobe API, MLflow, Postgres, Airflow e o volume `./data`:

```bash
docker compose up -d
# API     http://localhost:8000/health
# MLflow  http://localhost:5000
# Airflow http://localhost:8080  (admin / admin)
```

### Testes

```bash
pytest tests/unit tests/contracts tests/integration -q
```

---

## Cadências

| Pipeline | Frequência | Cron | DAG |
|---|---|---|---|
| Ingestão | Diária | `0 3 * * *` | `ingest_daily` |
| Features RFM | Semanal (segunda) | `0 5 * * 1` | `customer_feature_table` |
| Scoring (champion) | Semanal (segunda) | `0 7 * * 1` | `batch_score_champion` |
| Treinamento | Mensal | `0 4 1 * *` | `train_evaluate_register` |
| Quality gates | Mensal | `30 4 1 * *` | `quality_gates_promote` |
| Monitoramento | A cada 30 min | `*/30 * * * *` | health da API |

Detalhes em [`configs/cadences.yaml`](configs/cadences.yaml) e [`docs/aula-01/cadencias.md`](docs/aula-01/cadencias.md).

---

## Documentação

### Aula 1

| Documento | Arquivo |
|---|---|
| Arquitetura e reuso do Projeto 1 | [docs/aula-01/arquitetura.md](docs/aula-01/arquitetura.md) |
| Diagrama | [docs/aula-01/diagrama-arquitetura.md](docs/aula-01/diagrama-arquitetura.md) |
| Exploração dos dados Olist | [docs/aula-01/exploracao-dados.md](docs/aula-01/exploracao-dados.md) |
| Contratos de dados | [docs/aula-01/contratos.md](docs/aula-01/contratos.md) · pasta [`contracts/`](contracts/) |
| Cadências | [docs/aula-01/cadencias.md](docs/aula-01/cadencias.md) |
| Critérios de aceite | [docs/aula-01/criterios-aceite.md](docs/aula-01/criterios-aceite.md) |

### Aula 2

| Documento | Arquivo |
|---|---|
| Orquestração, qualidade e feature table | [docs/aula-02/README.md](docs/aula-02/README.md) |
| Critérios de aceite | [docs/aula-02/criterios-aceite.md](docs/aula-02/criterios-aceite.md) |
| DAG | [airflow/dags/customer_feature_table_dag.py](airflow/dags/customer_feature_table_dag.py) |
| Contrato features | [contracts/output/customer_features.yaml](contracts/output/customer_features.yaml) |

### Aula 3

| Documento | Arquivo |
|---|---|
| Experimentação e MLflow | [docs/aula-03/README.md](docs/aula-03/README.md) |
| Critérios de aceite | [docs/aula-03/criterios-aceite.md](docs/aula-03/criterios-aceite.md) |
| Relatório comparativo | [docs/aula-03/relatorio-comparativo.md](docs/aula-03/relatorio-comparativo.md) |
| DAG | [airflow/dags/train_evaluate_dag.py](airflow/dags/train_evaluate_dag.py) |

### Aula 4

| Documento | Arquivo |
|---|---|
| Testes e quality gates | [docs/aula-04/README.md](docs/aula-04/README.md) |
| Critérios de aceite | [docs/aula-04/criterios-aceite.md](docs/aula-04/criterios-aceite.md) |
| Política | [configs/quality_gates.yaml](configs/quality_gates.yaml) |
| CI | [.github/workflows/quality-gates.yml](.github/workflows/quality-gates.yml) |
| DAG | [airflow/dags/quality_gates_dag.py](airflow/dags/quality_gates_dag.py) |

### Aula 5

| Documento | Arquivo |
|---|---|
| Scoring batch e API | [docs/aula-05/README.md](docs/aula-05/README.md) |
| Critérios de aceite | [docs/aula-05/criterios-aceite.md](docs/aula-05/criterios-aceite.md) |
| DAG scoring | [airflow/dags/batch_score_dag.py](airflow/dags/batch_score_dag.py) |
| DAG ingestão | [airflow/dags/ingest_daily_dag.py](airflow/dags/ingest_daily_dag.py) |
| Compose | [docker-compose.yml](docker-compose.yml) |

---

## Stack

Git · Docker · FastAPI · Apache Airflow · MLflow · GitHub Actions · AWS (padrão do Projeto 1)
