"""
API de consulta de segmentação (esqueleto — Aula 1).

Reutiliza o padrão FastAPI do Projeto 1. Endpoints reais serão
ligados à tabela customer_segments após o pipeline de scoring.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="Customer Segmentation API",
    description="Consulta segmentos de clientes produzidos pelo pipeline MLOps.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "customer-segmentation-api"}


@app.get("/segments/{customer_unique_id}")
def get_segment(customer_unique_id: str) -> dict:
    """Placeholder: retorna 501 até a tabela de scores existir."""
    return {
        "customer_unique_id": customer_unique_id,
        "segment_label": None,
        "detail": "Scoring ainda não disponível — implementar nas aulas seguintes.",
        "status": "not_implemented",
    }
