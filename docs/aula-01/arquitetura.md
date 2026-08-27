# Aula 1 — Arquitetura do sistema de MLOps

## Problema e consumidores

A empresa precisa manter **segmentação atualizada de clientes** com base em comportamento de compra (features RFM) para personalizar campanhas.

| Consumidor | Uso |
|---|---|
| Time de Marketing / CRM | Listas por `segment_label` para campanhas |
| API de consulta (FastAPI) | Lookup por `customer_unique_id` em tempo de requisição |
| Analytics / BI | Distribuição de segmentos, estabilidade ao longo do tempo |
| Model Registry (MLflow) | Auditoria de versões, promoção e rollback |

## Componentes reutilizados do Projeto 1

| Componente | Reuso |
|---|---|
| Git + GitHub | Versionamento, PRs, proteção de branch |
| Docker / Compose | Empacotamento local de API, MLflow, Airflow |
| FastAPI | Serviço consumidor dos scores |
| GitHub Actions | CI (lint/test) e CD (imagem/deploy) |
| AWS (padrão do Projeto 1) | Destino de deploy do serviço e artefatos |

**Novo foco deste projeto:** orquestração (Airflow), qualidade/contratos, feature store batch, experiment tracking (MLflow), continuous training e monitoramento.

## Camadas

```
Fonte Olist (CSV) 
  → Landing particionada (mensal)
    → Validação de contratos
      → Features RFM versionadas
        → Treino (mensal) / Scoring (semanal)
          → Model Registry + Tabela customer_segments
            → API FastAPI / export CRM
```

### Armazenamento

| Zona | Conteúdo | Path local (dev) |
|---|---|---|
| Raw | Dump original Olist | `data/raw/` |
| Landing | Partições mensais incrementais | `data/landing/{orders,order_items,order_payments}/year=*/month=*/` |
| Processed | Tabelas limpas / joins | `data/processed/` |
| Features | Datasets RFM versionados | `data/features/` |
| Scores | `customer_segments` | `data/scores/` |

Em produção, as mesmas zonas mapeiam para buckets S3/MinIO com o mesmo layout Hive-style.

### Orquestração — Apache Airflow

DAGs previstas (aulas seguintes):

1. `ingest_validate` — diária  
2. `build_features_score` — semanal  
3. `train_evaluate_promote` — mensal  
4. `monitor_health` — frequente  

### Experimentação — MLflow

- Tracking de parâmetros, métricas (silhouette, tamanho de clusters, ARI de estabilidade) e artefatos (modelo, scaler, mapeamento segment_id → label).
- Model Registry com stages `Staging` / `Production` e procedimento de rollback.

### Serviço de consulta — FastAPI

- `GET /health`
- `GET /segments/{customer_unique_id}` → lê a última partição de `customer_segments`

### Pipeline de entrega

- CI: testes unitários, contratos, DAG parse  
- CD: build da imagem da API + deploy (padrão AWS do Projeto 1)  
- CT (Continuous Training): DAG mensal + gates de promoção  

## Diagrama

Ver [diagrama-arquitetura.md](./diagrama-arquitetura.md).
