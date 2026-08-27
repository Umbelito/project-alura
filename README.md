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
airflow/dags/                # DAG customer_feature_table (Aula 2)
configs/                     # Cadências, paths e features
contracts/                   # Contratos de entrada e saída
data/
  raw/                       # CSVs Olist originais
  landing/                   # Partições mensais (fonte incremental)
  processed/                 # Parquets limpos / orders_enriched
  features/                  # Features RFM versionadas (v*/as_of_date=*)
  scores/                    # Tabela customer_segments
docs/aula-01/                # Documentação da Aula 1
docs/aula-02/                # Orquestração Airflow + qualidade
scripts/                     # Particionamento e run local da feature table
src/customer_segmentation/
  ingestion/                 # Ingestão incremental
  validation/                # Contratos + quality checks
  features/                  # Limpeza, join, RFM
  training/
  scoring/
  serving/
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

### Airflow (opcional)

```bash
docker compose --profile airflow up -d
# UI http://localhost:8080 — usuário/senha: admin / admin
```

### API (esqueleto)

```bash
uvicorn api.main:app --reload --port 8000
```

### Testes

```bash
pytest tests/unit tests/contracts tests/integration -q
```

---

## Cadências

| Pipeline | Frequência | Cron |
|---|---|---|
| Ingestão + validação | Diária | `0 3 * * *` |
| Features RFM + scoring | Semanal (segunda) | `0 5` / `0 7 * * 1` |
| Treinamento + promoção | Mensal | `0 4 1 * *` |
| Monitoramento | A cada 30 min | `*/30 * * * *` |

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

---

## Stack

Git · Docker · FastAPI · Apache Airflow · MLflow · GitHub Actions · AWS (padrão do Projeto 1)

---

## Próximos passos

Scoring semanal operacional, CI/CD de deploy e monitoramento de drift.
