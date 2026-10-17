# Relatório comparativo — segmentação RFM (Aula 3)

## Dados

- `as_of_date`: `2018-08-31`
- `feature_set_version`: `1.0.0`
- URI: `data/features/v1.0.0/as_of_date=2018-08-31/customer_features.parquet`
- Clientes: **72247**
- Features: `recency_days, frequency, monetary, avg_ticket, avg_items_per_order, customer_age_days`

## Protocolo

- Pré-processamento: imputação mediana + `log1p` nas colunas assimétricas + `StandardScaler`.
- Algoritmos (contrato operacional, com `predict()`): K-Means, MiniBatchK-Means, Gaussian Mixture.
- Métricas: silhouette, inércia (quando houver), Davies-Bouldin, distribuição dos clusters.
- Estabilidade: ARI em subamostras (retreino 80%) e, se houver, ARI entre períodos.
- Seleção: gates técnicos/negócio e score composto (silhouette + DB invertido + ARI).

## Resultados

| run | algoritmo | k | silhouette | davies_bouldin | inertia | ARI subsample | score | gates |
|---|---|---:|---:|---:|---:|---:|---:|---|
| kmeans_k4 | kmeans | 4 | 0.345 | 0.960 | 208817.5 | 0.937 | 0.585 | pass |
| kmeans_k5 | kmeans | 5 | 0.353 | 0.844 | 173499.0 | 0.664 | 0.500 | pass |
| kmeans_k6 | kmeans | 6 | 0.375 | 0.791 | 136575.3 | 0.984 | 0.625 | pass |
| minibatch_kmeans_k4_batch_size=1024 | minibatch_kmeans | 4 | 0.324 | 0.992 | 218222.1 | 0.616 | 0.462 | pass |
| minibatch_kmeans_k5_batch_size=1024 | minibatch_kmeans | 5 | 0.354 | 1.063 | 183758.1 | 0.606 | 0.468 | pass |
| minibatch_kmeans_k6_batch_size=1024 | minibatch_kmeans | 6 | 0.311 | 0.901 | 156920.8 | 0.532 | 0.431 | pass |
| gaussian_mixture_k4_covariance_type=diag | gaussian_mixture | 4 | 0.289 | 1.202 | — | 0.657 | 0.451 | pass |
| gaussian_mixture_k5_covariance_type=diag | gaussian_mixture | 5 | 0.338 | 1.131 | — | 0.612 | 0.460 | pass |
| gaussian_mixture_k6_covariance_type=diag | gaussian_mixture | 6 | 0.289 | 1.113 | — | 0.726 | 0.479 | pass |

## Candidato selecionado

- Pool: `gates_passed` (9 passaram nos gates)
- Gates do vencedor: **pass**
- Run: `kmeans_k6`
- Algoritmo: **kmeans** (k=6)
- Silhouette: **0.375**
- Davies-Bouldin: 0.791
- ARI subsample: 0.984
- Score composto: 0.625

### Perfis dos segmentos

| segment_id | label | size | recency_mean | frequency_mean | monetary_mean |
|---:|---|---:|---:|---:|---:|
| 0 | new_customers | 11760 | 34.0 | 1.00 | 98.78 |
| 1 | champions | 392 | 89.9 | 2.22 | 265.20 |
| 2 | hibernating | 28165 | 204.9 | 1.00 | 44.63 |
| 3 | loyal_customers | 1391 | 177.4 | 2.06 | 253.84 |
| 4 | need_attention | 28926 | 199.3 | 1.00 | 235.40 |
| 5 | big_spenders | 1613 | 179.8 | 1.01 | 311.61 |

## MLflow

- Experiment: `customer_segmentation_rfm`
- Parent run: `f3fd55a6263f44c383831471493d33f9`
- Modelo: `customer_segmentation_kmeans` v`3`
- Aliases: `candidate`
- URI candidato: `models:/customer_segmentation_kmeans@candidate`

Associação modelo ↔ pipeline ↔ dados via tags: `feature_set_version`, `as_of_date`,
`features_uri`, `parent_run_id`.

