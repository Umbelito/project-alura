"""Leitura da tabela histórica de segmentos para a API."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from customer_segmentation.storage import latest_scores_table_path, scores_current_pointer_path

logger = logging.getLogger(__name__)


def resolve_scores_path(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    return latest_scores_table_path()


def load_segments_table(path: Path | None = None) -> pd.DataFrame:
    target = path or resolve_scores_path()
    if target is None or not target.is_file():
        raise FileNotFoundError(
            "Tabela customer_segments ausente. Rode: python scripts/run_batch_scoring.py --as-of-date YYYY-MM-DD"
        )
    df = pd.read_parquet(target)
    if df.empty:
        raise ValueError("Tabela customer_segments vazia")
    return df


@lru_cache(maxsize=4)
def _cached_table(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path_str)


def get_segments_frame(explicit: str | Path | None = None) -> pd.DataFrame:
    path = resolve_scores_path(explicit)
    if path is None:
        raise FileNotFoundError("customer_segments não encontrado")
    stat = path.stat()
    return _cached_table(str(path), stat.st_mtime)


def lookup_segment(customer_unique_id: str, explicit: str | Path | None = None) -> dict[str, Any] | None:
    df = get_segments_frame(explicit)
    hit = df.loc[df["customer_unique_id"].astype(str) == str(customer_unique_id)]
    if hit.empty:
        return None
    row = hit.iloc[-1]
    return {
        "customer_unique_id": str(row["customer_unique_id"]),
        "segment_id": int(row["segment_id"]),
        "segment_label": str(row["segment_label"]),
        "as_of_date": str(row["as_of_date"])[:10],
        "updated_at": str(row["scored_at"]),
        "model_name": str(row["model_name"]),
        "model_version": str(row["model_version"]),
        "feature_set_version": str(row.get("feature_set_version", "")),
    }


def health_snapshot(explicit: str | Path | None = None) -> dict[str, Any]:
    path = resolve_scores_path(explicit)
    pointer = scores_current_pointer_path()
    base: dict[str, Any] = {
        "service": "customer-segmentation-api",
        "scores_available": False,
        "as_of_date": None,
        "model_version": None,
        "model_name": None,
        "n_customers": None,
        "n_segments": None,
        "scored_at": None,
        "scores_uri": None,
        "pointer_uri": str(pointer) if pointer.is_file() else None,
    }
    if path is None or not path.is_file():
        return {**base, "status": "degraded", "detail": "tabela de segmentos ausente"}
    try:
        df = get_segments_frame(explicit)
        row = df.iloc[0]
        return {
            **base,
            "status": "ok",
            "scores_available": True,
            "as_of_date": str(row["as_of_date"])[:10],
            "model_version": str(row["model_version"]),
            "model_name": str(row["model_name"]),
            "n_customers": int(len(df)),
            "scored_at": str(row["scored_at"]),
            "scores_uri": str(path).replace("\\", "/"),
            "n_segments": int(df["segment_id"].nunique()),
        }
    except Exception as exc:
        logger.exception("health snapshot falhou")
        return {**base, "status": "degraded", "detail": str(exc)}
