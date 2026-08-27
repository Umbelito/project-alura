# Critérios de aceite — Aula 2

## Entregável

- [x] DAG `customer_feature_table` em `airflow/dags/customer_feature_table_dag.py`
- [x] Feature table validada e versionada em `data/features/v*/as_of_date=*/`
- [x] Contrato `contracts/output/customer_features.yaml`
- [x] Pipeline executável sem Airflow: `scripts/run_feature_table.py`
- [x] Documentação em `docs/aula-02/`

## Definition of Done

1. Ingestão lê partições Hive-style em `data/landing` e grava parquet em `data/processed`.
2. Validação cobre schema, nulos, PK e integridade referencial antes do join.
3. Join produz `orders_enriched` intermediário por `as_of_date`.
4. Features RFM + comportamentais são calculadas e validadas.
5. Persistência é idempotente (reexecução não duplica; sobrescreve a partição).
6. XCom / retorno de tasks contém só URI e metadados.
7. DAG declara retries, timeout, catchup e params para backfill.
8. Testes unitários de qualidade e RFM passam sem depender do scheduler.

## Fora de escopo (próximas aulas)

- ~~Treino / MLflow Registry~~ → Aula 3 (`docs/aula-03/`)
- Scoring com modelo champion publicando `customer_segments`
- Deploy da API com tabela de segmentos
- Monitoramento
