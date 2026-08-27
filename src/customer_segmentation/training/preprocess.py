"""Pré-processamento sklearn-nativo da feature table (contrato operacional)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

DEFAULT_FEATURE_COLUMNS = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_ticket",
    "avg_items_per_order",
    "customer_age_days",
]

DEFAULT_LOG_COLUMNS = ["recency_days", "frequency", "monetary", "avg_ticket"]


def log1p_nonneg(X):
    """log1p com clip em 0 — função de módulo para pickle/MLflow."""
    return np.log1p(np.clip(np.asarray(X, dtype=float), 0, None))


def select_model_frame(
    features: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    cols = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    missing = [c for c in cols if c not in features.columns]
    if missing:
        raise KeyError(f"Colunas de modelagem ausentes: {missing}")
    return features[cols].copy().astype("float64")


def build_preprocessor(
    feature_columns: list[str] | None = None,
    log_columns: list[str] | None = None,
) -> ColumnTransformer:
    """Imputação + log1p (colunas assimétricas) + StandardScaler."""
    cols = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    log_cols = [c for c in (log_columns or DEFAULT_LOG_COLUMNS) if c in cols]
    linear_cols = [c for c in cols if c not in log_cols]

    log_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "log",
                FunctionTransformer(
                    log1p_nonneg,
                    validate=True,
                    feature_names_out="one-to-one",
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    lin_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = []
    if log_cols:
        transformers.append(("log", log_pipe, log_cols))
    if linear_cols:
        transformers.append(("lin", lin_pipe, linear_cols))

    return ColumnTransformer(transformers, remainder="drop")
