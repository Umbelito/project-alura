# Aula 3 — Experimentação, estabilidade e registro com MLflow

## Entregável

Experimento rastreável no MLflow, relatório comparativo e **versão candidata** registrada com:

- signature e input example
- aliases `candidate` e `champion` (champion só se ainda não existir, ou `--set-champion`)
- tags ligando modelo ↔ `parent_run_id` ↔ `feature_set_version` / `as_of_date` / `features_uri`

## Pipeline de transformação e modelagem

`Pipeline` sklearn:

1. `ColumnTransformer`: imputação mediana
2. `log1p` nas colunas assimétricas (`recency_days`, `frequency`, `monetary`, `avg_ticket`)
3. `StandardScaler` em todas as features de modelagem
4. Clusterer com `.predict()` — exigência do contrato operacional (scoring semanal em clientes novos)

Features: `configs/training.yaml` → `feature_columns`.

## Algoritmos comparados

| Algoritmo | Por que entra no contrato |
|---|---|
| K-Means | `predict()`, inércia, estável, interpretável via centróides |
| MiniBatch K-Means | mesmo contrato, melhor custo em volume |
| Gaussian Mixture | `predict()` + incerteza; BIC/AIC no lugar da inércia |

Agglomerative / DBSCAN ficam de fora: não atribuem cluster a um cliente novo sem retrabalho incompatível com o scoring batch.

## Busca de hiperparâmetros

Grade em `configs/training.yaml` (`n_clusters` 4–6; GMM ainda varia `covariance_type`).  
`--quick` usa só K-Means com k=3 e 4.

Cada ponto da grade vira um **nested run** MLflow.

## Métricas

- Silhouette (maior é melhor)
- Inércia (K-Means / MiniBatch)
- Davies-Bouldin (menor é melhor)
- Distribuição: `n_clusters`, tamanho min/max, share mínimo
- GMM: BIC / AIC quando disponível

## Estabilidade

- **Subamostras:** retreino em 80% dos clientes e ARI vs. labels do fit completo
- **Períodos:** `--compare-as-of-date` treina o mesmo spec em outra feature table e calcula ARI nos `customer_unique_id` comuns

## Perfis interpretáveis

Centróides RFM em z-score geram rótulos de negócio: `champions`, `loyal_customers`, `big_spenders`, `new_customers`, `at_risk`, `hibernating`, `need_attention`. Artefato `segment_profiles.json`.

## Gates de seleção (`configs/training.yaml` / `cadences.yaml`)

| Gate | Default |
|---|---|
| silhouette_min | 0.20 |
| segment_count_between | 3–10 |
| min_cluster_size | 20 |
| min_cluster_share | 0,2% |
| ari_subsample_min | 0.50 |

Quem passa nos gates compete por score composto (silhouette 0.45 + DB invertido 0.20 + ARI 0.35). Se ninguém passa, registra-se o melhor como candidato com `gates_passed=false` (não cria `champion` novo, salvo first-time).

## Como executar

```bash
# Feature table da Aula 2 (se ainda não existir)
python scripts/run_feature_table.py --as-of-date 2018-08-31

python scripts/run_training_experiment.py --as-of-date 2018-08-31 --write-docs-report -v

# UI MLflow (opcional)
docker compose up -d mlflow
# http://localhost:5000  +  --tracking-uri http://localhost:5000
```

Tracking default: `file:./mlruns` (sem depender do servidor).

### Airflow

DAG `train_evaluate_register` — `0 4 1 * *` — XCom só com versão/aliases/URIs.

## Artefatos

```
data/models/experiments/as_of_date=YYYY-MM-DD/{parent_run_id}/
  comparison.json
  relatorio-comparativo.md
  manifest.json
docs/aula-03/relatorio-comparativo.md   # com --write-docs-report
```

## Critérios de aceite

Ver [criterios-aceite.md](./criterios-aceite.md).
