# Critérios de aceite e entregáveis — Aula 1

## Entregáveis desta aula

- [x] Diagrama da arquitetura (`docs/aula-01/diagrama-arquitetura.md`)
- [x] Contratos de dados de entrada e da tabela final (`contracts/`)
- [x] Repositório inicial organizado para MLOps (módulos, configs, API esqueleto)
- [x] Simulação de fonte incremental por partições mensais (`data/landing/`, script)
- [x] Definição de cadências (`configs/cadences.yaml`, `docs/aula-01/cadencias.md`)
- [x] Exploração documentada das tabelas (`docs/aula-01/exploracao-dados.md`)

## Critérios de aceite (Definition of Done — Aula 1)

1. O repositório separa claramente **ingestão, validação, features, treino, scoring e serving**.
2. Os CSVs Olist estão em `data/raw/` e as partições mensais em `data/landing/`.
3. Existem contratos versionados para `orders`, `customers`, `order_items`, `order_payments` e `customer_segments`.
4. Cadências de ingestão (diária), scoring (semanal) e treino (mensal) estão documentadas e versionadas em config.
5. O diagrama descreve armazenamento, Airflow, MLflow, API e pipeline de entrega.
6. A API FastAPI sobe com `/health` (esqueleto reutilizando o padrão do Projeto 1).
7. README explica como particionar dados e onde estão os artefatos da Aula 1.

## Fora de escopo (próximas aulas)

- ~~DAGs Airflow / feature table / quality checks~~ → Aula 2 (`docs/aula-02/`)
- Treino K-Means + MLflow real
- Scoring e publicação Parquet/Delta
- CI/CD completo e monitoramento em produção
