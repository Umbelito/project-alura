"""Tracking MLflow: params, métricas, datasets, artefatos, signature e aliases."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from customer_segmentation.config import load_yaml, project_root
from customer_segmentation.storage import _utc_now_iso

logger = logging.getLogger(__name__)

CANDIDATE_ALIAS = "candidate"
CHAMPION_ALIAS = "champion"


def resolve_tracking_uri(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    paths = project_root() / "configs" / "paths.yaml"
    if paths.is_file():
        cfg = load_yaml(paths)
        uri = (cfg.get("mlflow") or {}).get("tracking_uri")
        if uri and not str(uri).startswith("http://localhost"):
            return str(uri)
        # localhost só se o caller quiser; default file store evita hang
    return f"file:{(project_root() / 'mlruns').as_posix()}"


def _safe_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value == value and abs(value) != float("inf"):
            out[key] = float(value)
    return out


def log_dataset_ref(features: pd.DataFrame, source_uri: str, name: str = "customer_features") -> None:
    import mlflow

    try:
        dataset = mlflow.data.from_pandas(features.head(100), source=source_uri, name=name)
        mlflow.log_input(dataset, context="training")
    except Exception as exc:  # pragma: no cover - API varia entre versões
        logger.warning("mlflow.log_input indisponível: %s", exc)
        mlflow.log_param("dataset_source_uri", source_uri)
        mlflow.log_param("dataset_name", name)
        mlflow.log_metric("dataset_logged_rows_sample", min(100, len(features)))


def log_trial(
    *,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    tags: dict[str, str],
    profiles: dict[str, Any] | None = None,
    extra_artifacts: dict[str, Path] | None = None,
) -> str:
    import mlflow

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.log_params({k: _stringify(v) for k, v in params.items()})
        mlflow.log_metrics(_safe_metrics(metrics))
        mlflow.set_tags(tags)
        if profiles is not None:
            mlflow.log_dict(profiles, "segment_profiles.json")
        if extra_artifacts:
            for artifact_name, artifact_path in extra_artifacts.items():
                if artifact_path.is_file():
                    mlflow.log_artifact(str(artifact_path), artifact_path=artifact_name)
        logger.info("trial logged run_id=%s name=%s", run.info.run_id, run_name)
        return run.info.run_id


def register_sklearn_model(
    *,
    pipeline,
    input_example: pd.DataFrame,
    model_name: str,
    tags: dict[str, str],
    artifact_path: str = "model",
    set_champion: bool = False,
    extra_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Registra o pipeline com signature + input example e aplica aliases."""
    import mlflow
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    predictions = pipeline.predict(input_example)
    signature = infer_signature(input_example, predictions)

    model_info = mlflow.sklearn.log_model(
        pipeline,
        artifact_path=artifact_path,
        signature=signature,
        input_example=input_example,
        registered_model_name=model_name,
    )
    if extra_artifacts:
        for name, payload in extra_artifacts.items():
            if isinstance(payload, dict):
                mlflow.log_dict(payload, name)
            elif isinstance(payload, (str, Path)) and Path(payload).is_file():
                mlflow.log_artifact(str(payload))

    mlflow.set_tags(tags)

    version = _resolve_model_version(model_name, model_info)
    client = MlflowClient()
    _set_alias(client, model_name, CANDIDATE_ALIAS, version)
    aliases = [CANDIDATE_ALIAS]
    if set_champion:
        _set_alias(client, model_name, CHAMPION_ALIAS, version)
        aliases.append(CHAMPION_ALIAS)

    for key, value in tags.items():
        try:
            client.set_model_version_tag(model_name, str(version), key, value)
        except Exception as exc:
            logger.warning("tag de versão não aplicada %s: %s", key, exc)

    result = {
        "model_name": model_name,
        "model_version": str(version),
        "model_uri": f"models:/{model_name}/{version}",
        "aliases": aliases,
        "candidate_uri": f"models:/{model_name}@{CANDIDATE_ALIAS}",
        "registered_at": _utc_now_iso(),
        "signature": True,
        "input_example_rows": int(len(input_example)),
    }
    logger.info("modelo registrado %s", result)
    return result


def _stringify(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _resolve_model_version(model_name: str, model_info) -> str:
    version = getattr(model_info, "registered_model_version", None)
    if version:
        return str(version)
    from mlflow.tracking import MlflowClient

    latest = MlflowClient().get_latest_versions(model_name)
    if not latest:
        raise RuntimeError(f"Nenhuma versão registrada para {model_name}")
    return str(sorted(latest, key=lambda v: int(v.version))[-1].version)


def _set_alias(client, model_name: str, alias: str, version: str) -> None:
    try:
        client.set_registered_model_alias(model_name, alias, str(version))
        logger.info("alias %s -> %s v%s", alias, model_name, version)
        return
    except Exception as exc:
        logger.warning("aliases indisponíveis (%s); fallback stages", exc)
    stage = "Production" if alias == CHAMPION_ALIAS else "Staging"
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage=stage,
            archive_existing_versions=False,
        )
    except Exception as exc:
        logger.warning("fallback de stage falhou: %s", exc)


def _alias_exists(client, model_name: str, alias: str) -> bool:
    try:
        client.get_model_version_by_alias(model_name, alias)
        return True
    except Exception:
        return False
