# Relatório comparativo — segmentação RFM (Aula 3)

## Dados

- `as_of_date`: `2018-08-31`
- `feature_set_version`: `1.0.0`
- URI: `data/features/v1.0.0/as_of_date=2018-08-31/customer_features.parquet`
- Clientes: **18073**
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
| kmeans_k4 | kmeans | 4 | 0.339 | 1.003 | 50849.8 | 0.914 | 0.572 | pass |
| kmeans_k5 | kmeans | 5 | 0.342 | 0.810 | 42299.4 | 0.668 | 0.498 | pass |
| kmeans_k6 | kmeans | 6 | 0.340 | 0.805 | 33938.8 | 0.903 | 0.580 | pass |
| minibatch_kmeans_k4_batch_size=1024 | minibatch_kmeans | 4 | 0.304 | 1.123 | 54352.8 | 0.536 | 0.418 | pass |
| minibatch_kmeans_k5_batch_size=1024 | minibatch_kmeans | 5 | 0.298 | 1.029 | 44937.2 | 0.434 | 0.385 | fail |
| minibatch_kmeans_k6_batch_size=1024 | minibatch_kmeans | 6 | 0.333 | 0.900 | 37611.3 | 0.548 | 0.447 | pass |
| gaussian_mixture_k4_covariance_type=diag | gaussian_mixture | 4 | 0.336 | 1.028 | — | 0.903 | 0.566 | pass |
| gaussian_mixture_k5_covariance_type=diag | gaussian_mixture | 5 | 0.226 | 1.209 | — | 0.517 | 0.373 | pass |
| gaussian_mixture_k6_covariance_type=diag | gaussian_mixture | 6 | 0.282 | 1.104 | — | 0.852 | 0.520 | pass |

## Candidato selecionado

- Pool: `gates_passed` (8 passaram nos gates)
- Gates do vencedor: **pass**
- Run: `kmeans_k6`
- Algoritmo: **kmeans** (k=6)
- Silhouette: **0.340**
- Davies-Bouldin: 0.805
- ARI subsample: 0.903
- Score composto: 0.580

### Perfis dos segmentos

| segment_id | label | size | recency_mean | frequency_mean | monetary_mean |
|---:|---|---:|---:|---:|---:|
| 0 | new_customers | 4227 | 16.1 | 1.00 | 85.63 |
| 1 | hibernating | 6321 | 54.4 | 1.00 | 42.82 |
| 2 | loyal_customers | 160 | 39.2 | 2.02 | 274.05 |
| 3 | big_spenders | 6939 | 52.8 | 1.00 | 246.61 |
| 4 | champions | 41 | 23.0 | 2.15 | 197.06 |
| 5 | big_spenders_2 | 385 | 47.5 | 1.00 | 283.98 |

## MLflow

- Experiment: `customer_segmentation_rfm`
- Parent run: `2e8782e8dd8c4aaab8fa54b0e346a70e`
- Modelo: `customer_segmentation_kmeans` v`2`
- Aliases: `candidate, champion`
- URI candidato: `models:/customer_segmentation_kmeans@candidate`

Associação modelo ↔ pipeline ↔ dados via tags: `feature_set_version`, `as_of_date`,
`features_uri`, `parent_run_id`.

