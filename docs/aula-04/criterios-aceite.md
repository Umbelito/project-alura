# Critérios de aceite — Aula 4

## Entregável

- [x] Testes unitários de transformações e RFM
- [x] Testes de contratos de entrada e saída
- [x] Testes de importação/estrutura das DAGs
- [x] Testes de integração (processamento, MLflow, storage)
- [x] Testes de inferência / atribuição de clusters
- [x] Testes de assinatura e compatibilidade
- [x] Smoke do modelo (predict finito)
- [x] Coverage, lint (Ruff) e verificação de tipos (mypy)
- [x] Quality gates blocking: dados, regressão, clusters, assinatura, integração
- [x] Resultado gravado como tags MLflow
- [x] Política de aprovação/rejeição (`configs/quality_gates.yaml`)

## Definition of Done

1. `pytest tests/unit tests/contracts tests/integration` passa.
2. Ruff e mypy fazem parte do workflow CI.
3. Candidato rejeitado **não** recebe alias `champion`.
4. Relatório/tags distinguem cada gate (`pass`/`fail`).
5. DAG `quality_gates_promote` falha quando a política rejeita.

## Fora de escopo (próximas aulas)

- Scoring semanal em produção com DAG dedicada além do probe de integração
- Monitoramento contínuo de drift
