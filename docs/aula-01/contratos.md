# Contratos de dados — Aula 1

Contratos versionados em YAML:

## Entrada

| Contrato | Arquivo |
|---|---|
| orders | [`contracts/input/orders.yaml`](../../contracts/input/orders.yaml) |
| customers | [`contracts/input/customers.yaml`](../../contracts/input/customers.yaml) |
| order_items | [`contracts/input/order_items.yaml`](../../contracts/input/order_items.yaml) |
| order_payments | [`contracts/input/order_payments.yaml`](../../contracts/input/order_payments.yaml) |

Cada contrato define: schema, PK/FK, particionamento, `allowed_values`, checks de qualidade e consumidores.

## Saída

| Contrato | Arquivo |
|---|---|
| customer_features | [`contracts/output/customer_features.yaml`](../../contracts/output/customer_features.yaml) (Aula 2) |
| customer_segments | [`contracts/output/customer_segments.yaml`](../../contracts/output/customer_segments.yaml) |

Campos essenciais da tabela final:

- `customer_unique_id`, `as_of_date`
- Features: `recency_days`, `frequency`, `monetary` (+ scores R/F/M opcionais)
- Segmento: `segment_id`, `segment_label`
- Rastreabilidade: `model_name`, `model_version`, `feature_set_version`, `scored_at`

## Evolução

- Versão semântica no campo `version` do YAML.
- Breaking change ⇒ bump major + migração do consumidor (API/CRM).
- Testes de contrato em `tests/contracts/` (aulas seguintes).
