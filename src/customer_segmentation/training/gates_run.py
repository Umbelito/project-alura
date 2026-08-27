"""Executa a suíte de quality gates e aplica a política de promoção."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import clone

from customer_segmentation.features.pipeline import parse_as_of_date
from customer_segmentation.scoring.assign import assign_segments, persist_segments
from customer_segmentation.storage import scores_table_path
from customer_segmentation.training.experiment import load_features, load_training_config
from customer_segmentation.training.metrics import clustering_metrics
from customer_segmentation.training.preprocess import select_model_frame
from customer_segmentation.training.promotion import (
    apply_promotion_decision,
    evaluate_promotion,
    load_quality_gate_config,
    persist_gate_tags,
    tags_from_report,
)
from customer_segmentation.training.signature import check_mlflow_signature, check_signature_compatibility
from customer_segmentation.training.stability import subsample_stability_ari
from customer_segmentation.training.tracking import CHAMPION_ALIAS, CANDIDATE_ALIAS, resolve_tracking_uri
from customer_segmentation.validation.quality import validate_feature_table

logger = logging.getLogger(__name__)


def smoke_predict(pipeline, features: pd.DataFrame, min_rows: int = 5) -> dict[str, Any]:
    X = select_model_frame(features)
    n = min(max(min_rows, 1), len(X))
    pred = np.asarray(pipeline.predict(X.head(n)))
    finite = np.isfinite(pred.astype(float)).all()
    passed = len(pred) == n and finite
    return {
        "passed": passed,
        "n": int(n),
        "n_unique": int(len(np.unique(pred))),
        "detail": "ok" if passed else "smoke: predição vazia ou não finita",
    }


def integration_probe(
    features: pd.DataFrame,
    pipeline,
    label_map: dict[str, str],
    as_of: date,
    model_version: str,
) -> dict[str, Any]:
    """Processamento → inferência → armazenamento compartilhado."""
    try:
        scored = assign_segments(
            features,
            pipeline,
            label_map=label_map,
            model_version=model_version,
            feature_set_version=str(features["feature_set_version"].iloc[0])
            if "feature_set_version" in features
            else "1.0.0",
        )
        ref = persist_segments(scored, as_of)
        reloaded = pd.read_parquet(scores_table_path(as_of))
        passed = len(reloaded) == len(scored) == len(features)
        return {
            "passed": passed,
            "detail": f"scores uri={ref.get('uri')} rows={ref.get('row_count')}",
            "scores_ref": ref,
        }
    except Exception as exc:
        logger.exception("integration probe falhou")
        return {"passed": False, "detail": str(exc), "scores_ref": None}


def _run_metrics(pipeline, features: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    X = select_model_frame(features)
    labels = np.asarray(pipeline.predict(X))
    X_t = pipeline.named_steps["preprocess"].transform(X)
    metrics = clustering_metrics(X_t, labels, pipeline)
    return labels, metrics


def _champion_metrics_from_registry(model_name: str, candidate_version: str) -> dict[str, Any] | None:
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        champ = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        if str(champ.version) == str(candidate_version):
            return None
        run = client.get_run(champ.run_id)
        return {
            "silhouette": run.data.metrics.get("winner_silhouette"),
            "davies_bouldin": run.data.metrics.get("winner_davies_bouldin"),
            "stability_ari_subsample": run.data.metrics.get("winner_stability_ari"),
            "version": str(champ.version),
        }
    except Exception as exc:
        logger.info("champion metrics indisponíveis: %s", exc)
        return None


def run_quality_gates(
    as_of_date: str | date,
    *,
    feature_set_version: str = "1.0.0",
    features_uri: str | None = None,
    model_uri: str | None = None,
    pipeline=None,
    label_map: dict[str, str] | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    candidate_metrics: dict[str, Any] | None = None,
    tracking_uri: str | None = None,
    promote: bool = True,
    register_tags: bool = True,
) -> dict[str, Any]:
    """Avalia o candidato e promove champion somente se a política aprovar."""
    import mlflow

    as_of = parse_as_of_date(as_of_date)
    qcfg = load_quality_gate_config()
    tcfg = load_training_config()
    model_name = model_name or tcfg.get("model_name", "customer_segmentation_kmeans")
    uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)

    features, source_uri = load_features(as_of, feature_set_version, features_uri)
    data_report = validate_feature_table(features)

    if pipeline is None:
        load_uri = model_uri or f"models:/{model_name}@{CANDIDATE_ALIAS}"
        pipeline = mlflow.sklearn.load_model(load_uri)
        if model_version is None:
            try:
                from mlflow.tracking import MlflowClient

                mv = MlflowClient().get_model_version_by_alias(model_name, CANDIDATE_ALIAS)
                model_version = str(mv.version)
                model_uri = load_uri
            except Exception:
                model_version = model_version or "candidate"
    model_version = str(model_version or "candidate")

    labels, computed = _run_metrics(pipeline, features)
    metrics = {**computed, **(candidate_metrics or {})}
    if metrics.get("stability_ari_subsample") is None:
        X = select_model_frame(features)
        metrics.update(
            subsample_stability_ari(
                lambda: clone(pipeline),
                X,
                labels,
                n_subsamples=1,
                sample_frac=0.8,
                random_state=42,
            )
        )

    sig = check_signature_compatibility(pipeline, features, tcfg["feature_columns"])
    if model_uri:
        try:
            ml_sig = check_mlflow_signature(model_uri, tcfg["feature_columns"])
            sig["passed"] = bool(sig["passed"] and ml_sig["passed"])
            sig["mlflow"] = ml_sig
        except Exception as exc:
            logger.warning("signature MLflow omitida: %s", exc)

    smoke = smoke_predict(pipeline, features, int(qcfg["thresholds"].get("smoke_min_rows", 5)))
    probe = integration_probe(features, pipeline, label_map or {}, as_of, model_version)
    champ = _champion_metrics_from_registry(model_name, model_version)

    report = evaluate_promotion(
        data_ok=data_report.passed,
        data_detail=f"rows={data_report.row_count}",
        labels=labels,
        candidate_metrics=metrics,
        champion_metrics=champ,
        signature_ok=bool(sig.get("passed")),
        signature_detail="ok" if sig.get("passed") else str(sig.get("predict_error") or sig),
        integration_ok=bool(probe.get("passed")),
        integration_detail=str(probe.get("detail") or ""),
        smoke_ok=bool(smoke.get("passed")),
        smoke_detail=str(smoke.get("detail") or ""),
        config=qcfg,
    )

    tags = tags_from_report(report, prefix=qcfg.get("tag_prefix", "quality_gate_"))
    tags["features_uri"] = source_uri
    tags["as_of_date"] = as_of.isoformat()
    tags["feature_set_version"] = feature_set_version

    promotion: dict[str, Any] = {}
    with mlflow.start_run(run_name=f"quality_gates_{as_of.isoformat()}"):
        if register_tags:
            persist_gate_tags(tags, model_name=model_name, model_version=model_version)
        mlflow.log_dict(report, "quality_gate_report.json")
        if promote:
            try:
                promotion = apply_promotion_decision(
                    report,
                    model_name=model_name,
                    model_version=model_version,
                    promote=True,
                )
            except Exception as exc:
                logger.warning("apply_promotion_decision falhou: %s", exc)
                promotion = {"error": str(exc), "champion_promoted": False}

    return {
        "as_of_date": as_of.isoformat(),
        "features_uri": source_uri,
        "model_name": model_name,
        "model_version": model_version,
        "report": report,
        "tags": tags,
        "promotion": promotion,
        "smoke": smoke,
        "integration": {k: v for k, v in probe.items() if k != "scores_ref"},
        "scores_ref": probe.get("scores_ref"),
    }
