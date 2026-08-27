"""
API de consulta da segmentação produzida pelo scoring batch (Aula 5).
Lê a tabela histórica `customer_segments` (partição corrente).
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from customer_segmentation.serving.lookup import health_snapshot, lookup_segment

app = FastAPI(
    title="Customer Segmentation API",
    description="Consulta segmentos de clientes produzidos pelo pipeline batch (alias champion).",
    version="1.0.0",
)


class HealthResponse(BaseModel):
    status: str
    service: str
    scores_available: bool
    as_of_date: str | None = None
    model_version: str | None = None
    model_name: str | None = None
    n_customers: int | None = None
    n_segments: int | None = None
    scored_at: str | None = None
    scores_uri: str | None = None
    pointer_uri: str | None = None
    detail: str | None = None


class SegmentResponse(BaseModel):
    customer_unique_id: str
    segment_id: int
    segment_label: str
    as_of_date: str = Field(description="Data de referência da janela RFM")
    updated_at: str = Field(description="Timestamp da execução de scoring")
    model_name: str
    model_version: str
    feature_set_version: str


def _scores_path() -> str | None:
    return os.environ.get("SEGMENTS_PATH") or None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    snap = health_snapshot(_scores_path())
    return HealthResponse(**snap)


@app.get("/segments/{customer_unique_id}", response_model=SegmentResponse)
def get_segment(customer_unique_id: str) -> SegmentResponse:
    try:
        row = lookup_segment(customer_unique_id, _scores_path())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Cliente {customer_unique_id} não encontrado na segmentação vigente",
        )
    return SegmentResponse(**row)
