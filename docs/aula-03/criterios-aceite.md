# Critérios de aceite — Aula 3

## Entregável

- [x] Pipeline sklearn de transformação + clustering
- [x] Comparação K-Means, MiniBatch K-Means e Gaussian Mixture
- [x] Busca de hiperparâmetros (`configs/training.yaml`)
- [x] MLflow: params, métricas, dataset, artefatos, tags
- [x] Silhouette, inércia, Davies-Bouldin, distribuição
- [x] Estabilidade por subamostra e (opcional) período
- [x] Perfis interpretáveis de segmentos
- [x] Gates técnicos e de negócio
- [x] Registro com signature e input example
- [x] Aliases `candidate` e `champion`
- [x] Associação modelo ↔ run ↔ versão dos dados
- [x] Relatório comparativo

## Definition of Done

1. Um parent run agrupa nested runs da grade.
2. O vencedor é registrado no Model Registry.
3. `models:/customer_segmentation_kmeans@candidate` resolve a versão desta execução.
4. Tags incluem `feature_set_version`, `as_of_date`, `features_uri`, `parent_run_id`.
5. Relatório lista métricas de todos os trials e o perfil do vencedor.
6. Testes unitários de preprocess/métricas/seleção passam sem scheduler.

## Fora de escopo (próximas aulas)

- ~~Quality gates de promoção / suíte CI~~ → Aula 4 (`docs/aula-04/`)
- Scoring semanal em produção com DAG dedicada
- Monitoramento de drift
