"""Camada de consulta consumida pela API FastAPI."""

from customer_segmentation.serving.lookup import health_snapshot, lookup_segment

__all__ = ["health_snapshot", "lookup_segment"]
