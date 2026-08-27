# Diagrama da arquitetura — Segmentação de clientes (MLOps)

```mermaid
flowchart TB
  subgraph Fonte
    RAW["data/raw<br/>CSVs Olist"]
  end

  subgraph Landing["Landing incremental"]
    ORD["orders<br/>year=/month="]
    ITM["order_items<br/>year=/month="]
    PAY["order_payments<br/>year=/month="]
    CUS["customers<br/>snapshot=full"]
  end

  subgraph Qualidade
    CTR["Contratos YAML<br/>contracts/input"]
    VAL["Validação<br/>Great Expectations / checks"]
  end

  subgraph Features
    RFM["Feature set RFM<br/>data/features/v*"]
  end

  subgraph Treino["Treinamento mensal"]
    TR["Clustering RFM"]
    MLF["MLflow Tracking + Registry"]
  end

  subgraph Scoring["Scoring semanal"]
    SC["Modelo Production"]
    SEG["customer_segments<br/>data/scores"]
  end

  subgraph Entrega
    API["FastAPI<br/>/segments/{id}"]
    CRM["CRM / Campanhas"]
  end

  subgraph Orquestracao["Apache Airflow"]
    D1["DAG ingest_validate<br/>diária"]
    D2["DAG features_score<br/>semanal"]
    D3["DAG train_promote<br/>mensal"]
    D4["DAG monitor<br/>30 min"]
  end

  RAW -->|partition script| ORD
  RAW --> ITM
  RAW --> PAY
  RAW --> CUS

  ORD --> VAL
  ITM --> VAL
  PAY --> VAL
  CUS --> VAL
  CTR --> VAL

  VAL --> RFM
  RFM --> TR
  TR --> MLF
  MLF -->|Production| SC
  RFM --> SC
  SC --> SEG
  SEG --> API
  SEG --> CRM

  D1 -.-> VAL
  D2 -.-> RFM
  D2 -.-> SC
  D3 -.-> TR
  D4 -.-> API
```

## Fluxo resumido

1. **Ingestão diária** lê a partição do mês corrente e valida contratos.  
2. **Features + scoring semanal** regeneram RFM e publicam segmentos.  
3. **Treino mensal** retreina, registra no MLflow e promove só se passar nos gates.  
4. **API / CRM** consomem `customer_segments` com rastreio de `model_version` e `feature_set_version`.
