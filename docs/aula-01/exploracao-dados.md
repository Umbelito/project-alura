# Exploração das tabelas Olist (Aula 1)

Dataset: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

## Tabelas usadas no pipeline de segmentação

| Tabela | Arquivo | Linhas (aprox.) | Papel |
|---|---|---:|---|
| customers | `olist_customers_dataset.csv` | 99.441 | Dimensão; chave de negócio = `customer_unique_id` |
| orders | `olist_orders_dataset.csv` | 99.441 | Fato temporal; base da partição mensal |
| order_items | `olist_order_items_dataset.csv` | 112.650 | Monetary (`price`) e composição do pedido |
| order_payments | `olist_order_payments_dataset.csv` | 103.886 | Features auxiliares de pagamento |

Tabelas disponíveis mas **fora do escopo inicial de RFM**: geolocation, products, sellers, reviews, category translation (podem enriquecer features em iterações futuras).

## Período observado

- **Mínimo:** 2016-09-04  
- **Máximo:** 2018-10-17  
- Meses esparsos em 2016-09/12 e 2018-09/10 (volume muito baixo) — tratar com cuidado em backfill e validação de volume.

## Status de pedidos

| Status | Qtd |
|---|---:|
| delivered | 96.478 |
| shipped | 1.107 |
| canceled | 625 |
| unavailable | 609 |
| invoiced | 314 |
| processing | 301 |
| created | 5 |
| approved | 2 |

**Regra de negócio inicial:** RFM considera apenas `order_status = delivered`.

## Relacionamentos

```
customers.customer_id 1──* orders.customer_id
orders.order_id       1──* order_items.order_id
orders.order_id       1──* order_payments.order_id
customers.customer_unique_id 1──* customers.customer_id  (mesmo cliente, vários ids de sessão)
```

A agregação RFM é feita em **`customer_unique_id`**, não em `customer_id`.

## Fonte incremental simulada

Script: `scripts/partition_incremental_source.py`

Layout Hive-style:

```
data/landing/orders/year=2018/month=08/orders.csv
data/landing/order_items/year=2018/month=08/order_items.csv
data/landing/order_payments/year=2018/month=08/order_payments.csv
data/landing/customers/snapshot=full/customers.csv
```

Isso permite DAGs Airflow com `execution_date` mensal e reprocessamento idempotente por partição.
