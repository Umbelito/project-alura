"""Scoring batch da base completa usando o alias MLflow `champion`."""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from customer_segmentation.features.pipeline import parse_as_of_date
from customer_segmentation.scoring.assign import assign_segments, persist_segments, publish_current_pointer
from customer_segmentation.training.experiment import load_features, load_training_config
from customer_segmentation.training.tracking import CHAMPION_ALIAS, resolve_tracking_uri

logger = logging.getLogger(__name__)


def load_champion(
    model_name: str | None = None,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Carrega o sklearn pipeline e metadados do alias champion."""
    import mlflow
    from mlflow.tracking import MlflowClient

    cfg = load_training_config()
    name = model_name or cfg.get("model_name", "customer_segmentation_kmeans")
    uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    model_uri = f"models:/{name}@{CHAMPION_ALIAS}"
    pipeline = mlflow.sklearn.load_model(model_uri)
    client = MlflowClient()
    version = "champion"
    run_id = None
    try:
        mv = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
        version = str(mv.version)
        run_id = mv.run_id
    except Exception as exc:
        logger.warning("versão champion via alias indisponível: %s", exc)

    label_map = _load_label_map_from_run(client, run_id)
    logger.info("Champion carregado %s v%s run=%s", name, version, run_id)
    return {
        "pipeline": pipeline,
        "model_name": name,
        "model_version": version,
        "model_uri": model_uri,
        "run_id": run_id,
        "label_map": label_map,
        "tracking_uri": uri,
        "alias": CHAMPION_ALIAS,
    }


def _load_label_map_from_run(client, run_id: str | None) -> dict[str, str]:
    if not run_id:
        return {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = client.download_artifacts(run_id, "label_map.json", tmp)
            payload = json.loads(Path(local).read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {str(k): str(v) for k, v in payload.items()}
    except Exception as exc:
        logger.warning("label_map.json não encontrado no run %s: %s", run_id, exc)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = client.download_artifacts(run_id, "segment_profiles.json", tmp)
            payload = json.loads(Path(local).read_text(encoding="utf-8"))
            raw = payload.get("label_map") if isinstance(payload, dict) else None
            if isinstance(raw, dict):
                return {str(k): str(v) for k, v in raw.items()}
    except Exception:
        return {}
    return {}


def run_batch_scoring(
    as_of_date: str | date,
    *,
    feature_set_version: str = "1.0.0",
    features_uri: str | None = None,
    tracking_uri: str | None = None,
    model_name: str | None = None,
    pipeline=None,
    label_map: dict[str, str] | None = None,
    model_version: str | None = None,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    """Segmenta a base completa e persiste histórico + ponteiro current."""
    as_of = parse_as_of_date(as_of_date)
    features, source_uri = load_features(as_of, feature_set_version, features_uri)

    if pipeline is None:
        champ = load_champion(model_name=model_name, tracking_uri=tracking_uri)
        pipeline = champ["pipeline"]
        label_map = label_map or champ["label_map"]
        model_name = champ["model_name"]
        model_version = champ["model_version"]
        model_uri = champ["model_uri"]
        tracking = champ["tracking_uri"]
    else:
        cfg = load_training_config()
        model_name = model_name or cfg.get("model_name", "customer_segmentation_kmeans")
        model_version = str(model_version or "local")
        model_uri = None
        tracking = resolve_tracking_uri(tracking_uri)

    scored = assign_segments(
        features,
        pipeline,
        label_map=label_map or {},
        model_name=str(model_name),
        model_version=str(model_version),
        feature_set_version=feature_set_version,
    )
    ref = persist_segments(scored, as_of)
    ref["model_name"] = model_name
    ref["model_version"] = str(model_version)
    ref["feature_set_version"] = feature_set_version
    ref["n_segments"] = int(scored["segment_id"].nunique())
    pointer = publish_current_pointer(ref)

    if log_mlflow:
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking)
            with mlflow.start_run(run_name=f"batch_score_{as_of.isoformat()}"):
                mlflow.set_tags(
                    {
                        "pipeline": "batch_scoring",
                        "alias": CHAMPION_ALIAS,
                        "as_of_date": as_of.isoformat(),
                        "feature_set_version": feature_set_version,
                        "features_uri": source_uri,
                        "model_name": str(model_name),
                        "model_version": str(model_version),
                    }
                )
                mlflow.log_metrics(
                    {
                        "n_customers": float(len(scored)),
                        "n_segments": float(scored["segment_id"].nunique()),
                    }
                )
                mlflow.log_dict(ref, "scores_ref.json")
        except Exception as exc:
            logger.warning("log MLflow do scoring omitido: %s", exc)

    logger.info(
        "Batch scoring ok as_of=%s rows=%s model=%s@%s",
        as_of,
        len(scored),
        model_name,
        model_version,
    )
    return {
        "as_of_date": as_of.isoformat(),
        "features_uri": source_uri,
        "scores": ref,
        "current_pointer": str(pointer).replace("\\", "/"),
        "model_name": model_name,
        "model_version": str(model_version),
        "model_uri": model_uri,
        "n_customers": int(len(scored)),
        "n_segments": int(scored["segment_id"].nunique()),
    }
