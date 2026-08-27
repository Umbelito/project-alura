"""Atribuição de clusters e persistência de customer_segments."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from customer_segmentation.features.pipeline import parse_as_of_date
from customer_segmentation.storage import (
    _utc_now_iso,
    scores_dir,
    scores_table_path,
    write_manifest,
    write_parquet_idempotent,
)
from customer_segmentation.training.preprocess import select_model_frame
from customer_segmentation.validation.quality import assert_valid, validate_feature_table

logger = logging.getLogger(__name__)


def load_label_map(payload: dict[str, Any] | None) -> dict[str, str]:
    if not payload:
        return {}
    raw = payload.get("label_map") if "label_map" in payload else payload
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def assign_segments(
    features: pd.DataFrame,
    pipeline,
    *,
    label_map: dict[str, str] | None = None,
    model_name: str = "customer_segmentation_kmeans",
    model_version: str = "candidate",
    feature_set_version: str = "1.0.0",
) -> pd.DataFrame:
    """Aplica o modelo e devolve o contrato `customer_segments`."""
    if features.empty:
        raise ValueError("Feature table vazia — inferência abortada")
    report = validate_feature_table(features)
    assert_valid(report)

    X = select_model_frame(features)
    labels = np.asarray(pipeline.predict(X))
    if labels.ndim > 1:
        labels = labels.ravel()

    mapping = load_label_map(label_map)
    as_of = features["as_of_date"].iloc[0]
    if hasattr(as_of, "isoformat"):
        as_of_str = as_of.isoformat()[:10]
    else:
        as_of_str = str(as_of)[:10]

    scored = pd.DataFrame(
        {
            "customer_unique_id": features["customer_unique_id"].astype(str).values,
            "as_of_date": as_of_str,
            "recency_days": features["recency_days"].astype(int).values,
            "frequency": features["frequency"].astype(int).values,
            "monetary": features["monetary"].astype(float).values,
            "r_score": features["r_score"].astype(int).values if "r_score" in features else None,
            "f_score": features["f_score"].astype(int).values if "f_score" in features else None,
            "m_score": features["m_score"].astype(int).values if "m_score" in features else None,
            "segment_id": labels.astype(int),
        }
    )
    scored["segment_label"] = scored["segment_id"].map(
        lambda i: mapping.get(str(int(i)), f"cluster_{int(i)}")
    )
    scored["model_name"] = model_name
    scored["model_version"] = str(model_version)
    scored["feature_set_version"] = feature_set_version
    scored["scored_at"] = _utc_now_iso()
    logger.info("Scoring concluído rows=%s clusters=%s", len(scored), scored["segment_id"].nunique())
    return scored


def persist_segments(scored: pd.DataFrame, as_of: str | date) -> dict[str, Any]:
    as_of_d = parse_as_of_date(as_of)
    path = scores_table_path(as_of_d)
    ref = write_parquet_idempotent(scored, path)
    ref.dataset = "customer_segments"
    ref.as_of_date = as_of_d.isoformat()
    write_manifest(
        scores_dir(as_of_d),
        {
            "dataset": "customer_segments",
            "as_of_date": as_of_d.isoformat(),
            "uri": ref.uri,
            "row_count": ref.row_count,
            "n_segments": int(scored["segment_id"].nunique()),
            "model_version": str(scored["model_version"].iloc[0]) if len(scored) else None,
        },
    )
    return ref.to_dict()
