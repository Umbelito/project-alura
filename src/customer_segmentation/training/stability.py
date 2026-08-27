"""Estabilidade de clusters entre subamostras e períodos."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

logger = logging.getLogger(__name__)


def subsample_stability_ari(
    pipeline_factory: Callable[[], Any],
    X: pd.DataFrame,
    labels_full: np.ndarray,
    n_subsamples: int = 3,
    sample_frac: float = 0.8,
    random_state: int = 42,
) -> dict[str, float]:
    """Retreina em subamostras e compara labels no conjunto completo (ARI)."""
    rng = np.random.RandomState(random_state)
    n = len(X)
    size = max(int(n * sample_frac), 2)
    scores: list[float] = []

    for i in range(n_subsamples):
        idx = rng.choice(n, size=min(size, n), replace=False)
        est = pipeline_factory()
        est.fit(X.iloc[idx])
        pred = np.asarray(est.predict(X))
        scores.append(float(adjusted_rand_score(labels_full, pred)))
        logger.info("stability subsample=%s ari=%.4f", i, scores[-1])

    return {
        "stability_ari_subsample": float(np.mean(scores)) if scores else float("nan"),
        "stability_ari_subsample_std": float(np.std(scores)) if scores else float("nan"),
        "stability_n_subsamples": float(len(scores)),
    }


def period_stability_ari(
    pipeline_a,
    pipeline_b,
    X_a: pd.DataFrame,
    X_b: pd.DataFrame,
    ids_a: pd.Series,
    ids_b: pd.Series,
    min_overlap: int = 50,
) -> dict[str, float | None]:
    """ARI nos clientes comuns entre dois períodos (dois modelos já fitted)."""
    frame_a = X_a.copy()
    frame_a["_cid"] = pd.Series(ids_a).astype(str).values
    frame_b = X_b.copy()
    frame_b["_cid"] = pd.Series(ids_b).astype(str).values
    common = sorted(set(frame_a["_cid"]) & set(frame_b["_cid"]))
    if len(common) < min_overlap:
        logger.warning(
            "overlap temporal insuficiente: %s < %s — ARI periódico omitido",
            len(common),
            min_overlap,
        )
        return {"stability_ari_period": None, "stability_period_overlap": float(len(common))}

    aligned_a = frame_a.drop_duplicates("_cid").set_index("_cid").loc[common]
    aligned_b = frame_b.drop_duplicates("_cid").set_index("_cid").loc[common]
    xa = aligned_a.drop(columns=["_cid"], errors="ignore")
    xb = aligned_b.drop(columns=["_cid"], errors="ignore")
    pred_a = np.asarray(pipeline_a.predict(xa))
    pred_b = np.asarray(pipeline_b.predict(xb))
    ari = float(adjusted_rand_score(pred_a, pred_b))
    logger.info("stability period overlap=%s ari=%.4f", len(common), ari)
    return {"stability_ari_period": ari, "stability_period_overlap": float(len(common))}
