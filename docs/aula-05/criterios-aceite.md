# Critérios de aceite — Aula 5

## Entregável

- [x] Pipeline de scoring separado do treinamento (`scoring/batch.py`, DAG `batch_score_champion`)
- [x] Cadências independentes (ingest diário, features/scoring semanais, treino mensal)
- [x] Carga do modelo pelo alias `champion`
- [x] Segmentação batch da base completa
- [x] Tabela histórica por `as_of_date` + ponteiro `current.json`
- [x] `model_version` persistido em cada execução
- [x] API: segmento atual, data de atualização, versão do modelo, health
- [x] Docker Compose: API, MLflow, Postgres, Airflow, armazenamento `./data`
- [x] Teste ponta a ponta (score → parquet → lookup → HTTP)

## Definition of Done

1. `python scripts/run_batch_scoring.py --as-of-date 2018-08-31` gera partição em `data/scores/`.
2. `GET /segments/{id}` devolve label + `model_version` + `updated_at`.
3. `GET /health` indica se a tabela está disponível.
4. Treino mensal e scoring semanal são DAGs distintas.
5. `docker compose up -d` sobe API, MLflow, banco e Airflow.

## Fora de escopo (próximas aulas)

- Monitoramento contínuo de drift / freshness em produção
- Autenticação da API
