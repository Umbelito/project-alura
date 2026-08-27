"""Geração e versionamento de features RFM."""

from customer_segmentation.features.pipeline import run_feature_table_pipeline
from customer_segmentation.features.rfm import (
    FEATURE_SET_VERSION,
    build_and_persist_features,
    compute_rfm_features,
)

__all__ = [
    "FEATURE_SET_VERSION",
    "build_and_persist_features",
    "compute_rfm_features",
    "run_feature_table_pipeline",
]
