# Cadências — ingestão, treinamento e scoring

Fonte de verdade: [`configs/cadencias.yaml`](../../configs/cadences.yaml).

| Pipeline | Cron | Frequência | Objetivo |
|---|---|---|---|
| Ingestão + validação | `0 3 * * *` | Diária | Trazer partição do mês corrente e validar contrato |
| Features RFM | `0 5 * * 1` | Semanal (segunda) | Recalcular R, F, M com janela de 365 dias |
| Scoring | `0 7 * * 1` | Semanal (segunda) | Publicar `customer_segments` com modelo Production |
| Treinamento + promoção | `0 4 1 * *` | Mensal | Retreinar, avaliar estabilidade e promover/reverter |
| Monitoramento | `*/30 * * * *` | 30 min | Heartbeat, freshness, drift de distribuição, API |

## Por que essas cadências?

- **Ingestão diária:** o mês corrente cresce ao longo do tempo; validar cedo reduz surpresas no scoring.
- **Scoring semanal:** campanhas de marketing não precisam de score intradaily; equilibra custo e frescor.
- **Treino mensal:** clustering RFM é relativamente estável; retreino frequente demais gera churn de rótulos sem ganho de negócio.
- **Gates de promoção:** silhouette, cardinalidade de segmentos, ARI vs. versão anterior e ausência de segmentos vazios.

## Janela RFM

- `observation_window_days: 365`
- `as_of_date`: domingo anterior ao run de features (23:59:59, timezone `America/Sao_Paulo`)
- Pedidos: apenas `delivered`
