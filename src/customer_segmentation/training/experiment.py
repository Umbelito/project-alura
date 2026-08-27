"""Orquestra o experimento: busca, avaliação, seleção e registro MLflow."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from customer_segmentation.config import configs_dir, data_dir, load_yaml, project_root
from customer_segmentation.features.pipeline import parse_as_of_date
from customer_segmentation.storage import (
    DatasetRef,
    features_table_path,
    relative_uri,
    write_manifest,
)
from customer_segmentation.training.metrics import clustering_metrics
from customer_segmentation.training.models import build_model_pipeline, iter_search_space
from customer_segmentation.training.preprocess import select_model_frame
from customer_segmentation.training.profiles import build_segment_profiles
from customer_segmentation.training.selection import select_candidate
from customer_segmentation.training.stability import period_stability_ari, subsample_stability_ari
from customer_segmentation.training.tracking import (
    log_dataset_ref,
    log_trial,
    register_sklearn_model,
    resolve_tracking_uri,
)

logger = logging.getLogger(__name__)


def load_training_config() -> dict[str, Any]:
    return load_yaml(configs_dir("training.yaml"))


def load_features(as_of: date, feature_set_version: str, uri: str | None = None) -> tuple[pd.DataFrame, str]:
    if uri:
        path = DatasetRef(uri=uri, dataset="customer_features").absolute_path()
    else:
        path = features_table_path(feature_set_version, as_of)
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature table ausente: {path}. "
            "Gere com: python scripts/run_feature_table.py --as-of-date YYYY-MM-DD"
        )
    df = pd.read_parquet(path)
    return df, relative_uri(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if number != number else number
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def render_comparison_report(payload: dict[str, Any]) -> str:
    winner = payload["winner"]
    lines = [
        "# Relatório comparativo — segmentação RFM (Aula 3)",
        "",
        "## Dados",
        "",
        f"- `as_of_date`: `{payload['data']['as_of_date']}`",
        f"- `feature_set_version`: `{payload['data']['feature_set_version']}`",
        f"- URI: `{payload['data']['features_uri']}`",
        f"- Clientes: **{payload['data']['n_rows']}**",
        f"- Features: `{', '.join(payload['data']['feature_columns'])}`",
        "",
        "## Protocolo",
        "",
        "- Pré-processamento: imputação mediana + `log1p` nas colunas assimétricas + `StandardScaler`.",
        "- Algoritmos (contrato operacional, com `predict()`): K-Means, MiniBatchK-Means, Gaussian Mixture.",
        "- Métricas: silhouette, inércia (quando houver), Davies-Bouldin, distribuição dos clusters.",
        "- Estabilidade: ARI em subamostras (retreino 80%) e, se houver, ARI entre períodos.",
        "- Seleção: gates técnicos/negócio e score composto (silhouette + DB invertido + ARI).",
        "",
        "## Resultados",
        "",
        "| run | algoritmo | k | silhouette | davies_bouldin | inertia | ARI subsample | score | gates |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["results"]:
        inertia = row.get("inertia")
        inertia_s = f"{inertia:.1f}" if isinstance(inertia, (int, float)) and inertia is not None else "—"
        sil = row.get("silhouette")
        db = row.get("davies_bouldin")
        ari = row.get("stability_ari_subsample")
        lines.append(
            "| {run} | {algo} | {k} | {sil:.3f} | {db:.3f} | {inertia} | {ari:.3f} | {score:.3f} | {gates} |".format(
                run=row.get("run_name"),
                algo=row.get("algorithm"),
                k=row.get("n_clusters"),
                sil=float(sil or 0),
                db=float(db or 0),
                inertia=inertia_s,
                ari=float(ari or 0),
                score=float(row.get("composite_score") or 0),
                gates="pass" if row.get("gates", {}).get("passed") else "fail",
            )
        )
    lines += [
        "",
        "## Candidato selecionado",
        "",
        f"- Pool: `{payload['selection_pool']}` ({payload['n_passing_gates']} passaram nos gates)",
        f"- Gates do vencedor: **{'pass' if winner.get('gates', {}).get('passed') else 'fail'}**",
        f"- Run: `{winner.get('run_name')}`",
        f"- Algoritmo: **{winner.get('algorithm')}** (k={winner.get('n_clusters')})",
        f"- Silhouette: **{float(winner.get('silhouette') or 0):.3f}**",
        f"- Davies-Bouldin: {float(winner.get('davies_bouldin') or 0):.3f}",
        f"- ARI subsample: {float(winner.get('stability_ari_subsample') or 0):.3f}",
        f"- Score composto: {float(winner.get('composite_score') or 0):.3f}",
        "",
        "### Perfis dos segmentos",
        "",
        "| segment_id | label | size | recency_mean | frequency_mean | monetary_mean |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for profile in winner.get("profiles", {}).get("profiles", []):
        lines.append(
            "| {id} | {label} | {size} | {r:.1f} | {f:.2f} | {m:.2f} |".format(
                id=profile["segment_id"],
                label=profile["segment_label"],
                size=profile["size"],
                r=profile["recency_days_mean"],
                f=profile["frequency_mean"],
                m=profile["monetary_mean"],
            )
        )
    mlflow_info = payload.get("mlflow") or {}
    lines += [
        "",
        "## MLflow",
        "",
        f"- Experiment: `{mlflow_info.get('experiment_name')}`",
        f"- Parent run: `{mlflow_info.get('parent_run_id')}`",
        f"- Modelo: `{mlflow_info.get('model_name')}` v`{mlflow_info.get('model_version')}`",
        f"- Aliases: `{', '.join(mlflow_info.get('aliases') or [])}`",
        f"- URI candidato: `{mlflow_info.get('candidate_uri')}`",
        "",
        "Associação modelo ↔ pipeline ↔ dados via tags: `feature_set_version`, `as_of_date`,",
        "`features_uri`, `parent_run_id`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def run_training_experiment(
    as_of_date: str | date,
    feature_set_version: str = "1.0.0",
    features_uri: str | None = None,
    compare_as_of_date: str | date | None = None,
    quick: bool = False,
    tracking_uri: str | None = None,
    register: bool = True,
    set_champion: bool = False,
    write_docs_report: bool = False,
    gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Executa busca, avalia estabilidade, seleciona e registra o candidato."""
    import mlflow

    cfg = load_training_config()
    as_of = parse_as_of_date(as_of_date)
    features, source_uri = load_features(as_of, feature_set_version, features_uri)
    feature_columns = list(cfg["feature_columns"])
    log_columns = list(cfg["log_columns"])
    X = select_model_frame(features, feature_columns)
    search = cfg["search_quick"] if quick else cfg["search"]
    random_state = int(cfg.get("random_state", 42))
    stability_cfg = cfg.get("stability") or {}
    active_gates = gates or cfg["gates"]

    uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    experiment_name = cfg.get("experiment_name", "customer_segmentation_rfm")
    mlflow.set_experiment(experiment_name)

    parent_tags = {
        "pipeline": "train_evaluate",
        "feature_set_version": feature_set_version,
        "as_of_date": as_of.isoformat(),
        "features_uri": source_uri,
        "quick": str(quick).lower(),
    }

    results: list[dict[str, Any]] = []
    fitted_winner_pipeline = None
    winner_profiles = None

    with mlflow.start_run(run_name=f"experiment_{as_of.isoformat()}") as parent:
        parent_run_id = parent.info.run_id
        mlflow.set_tags(parent_tags)
        mlflow.log_params(
            {
                "n_rows": len(features),
                "n_features": len(feature_columns),
                "search_mode": "quick" if quick else "full",
                "feature_columns": ",".join(feature_columns),
            }
        )
        log_dataset_ref(features, source_uri)

        for spec in iter_search_space(search, random_state=random_state):

            def factory(spec=spec):
                return build_model_pipeline(
                    spec["algorithm"],
                    feature_columns,
                    log_columns,
                    random_state=random_state,
                    **spec["params"],
                )

            pipeline = factory()
            pipeline.fit(X)
            labels = np.asarray(pipeline.predict(X))
            X_trans = pipeline.named_steps["preprocess"].transform(X)
            metrics = clustering_metrics(X_trans, labels, pipeline)
            stability = subsample_stability_ari(
                factory,
                X,
                labels,
                n_subsamples=int(stability_cfg.get("n_subsamples", 2 if quick else 3)),
                sample_frac=float(stability_cfg.get("sample_frac", 0.8)),
                random_state=random_state,
            )
            metrics.update(stability)
            profiles = build_segment_profiles(features, labels)
            trial = {
                **spec,
                **metrics,
                "profiles": profiles,
                "params": spec["params"],
            }
            trial_run_id = log_trial(
                run_name=spec["run_name"],
                params={
                    "algorithm": spec["algorithm"],
                    "n_clusters": spec["n_clusters"],
                    **spec["params"],
                },
                metrics=metrics,
                tags={**parent_tags, "parent_run_id": parent_run_id},
                profiles=profiles,
            )
            trial["mlflow_run_id"] = trial_run_id
            results.append(trial)
            logger.info(
                "trial %s silhouette=%.3f ari=%.3f",
                spec["run_name"],
                metrics.get("silhouette") or 0,
                metrics.get("stability_ari_subsample") or 0,
            )

        selection = select_candidate(results, active_gates, cfg.get("score_weights"))
        winner = selection["winner"]
        winner_profiles = winner["profiles"]

        fitted_winner_pipeline = build_model_pipeline(
            winner["algorithm"],
            feature_columns,
            log_columns,
            random_state=random_state,
            **winner["params"],
        )
        fitted_winner_pipeline.fit(X)

        period_metrics: dict[str, Any] = {}
        if compare_as_of_date:
            other_as_of = parse_as_of_date(compare_as_of_date)
            try:
                other_df, other_uri = load_features(other_as_of, feature_set_version)
                X_other = select_model_frame(other_df, feature_columns)
                other_pipe = build_model_pipeline(
                    winner["algorithm"],
                    feature_columns,
                    log_columns,
                    random_state=random_state,
                    **winner["params"],
                )
                other_pipe.fit(X_other)
                period_metrics = period_stability_ari(
                    fitted_winner_pipeline,
                    other_pipe,
                    X,
                    X_other,
                    features["customer_unique_id"],
                    other_df["customer_unique_id"],
                    min_overlap=int(stability_cfg.get("min_overlap", 50)),
                )
                mlflow.log_metrics(
                    {k: float(v) for k, v in period_metrics.items() if isinstance(v, (int, float))}
                )
                mlflow.log_param("compare_as_of_date", other_as_of.isoformat())
                mlflow.log_param("compare_features_uri", other_uri)
            except FileNotFoundError as exc:
                logger.warning("comparação temporal omitida: %s", exc)

        registry: dict[str, Any] = {}
        if register:
            input_example = X.head(5)
            registry = register_sklearn_model(
                pipeline=fitted_winner_pipeline,
                input_example=input_example,
                model_name=cfg.get("model_name", "customer_segmentation_kmeans"),
                tags={
                    **parent_tags,
                    "parent_run_id": parent_run_id,
                    "algorithm": str(winner["algorithm"]),
                    "n_clusters": str(winner["n_clusters"]),
                    "gates_passed": str(winner["gates"]["passed"]).lower(),
                    "selection_pool": selection["selection_pool"],
                },
                set_champion=set_champion,
                extra_artifacts={
                    "segment_profiles.json": winner_profiles,
                    "label_map.json": winner_profiles.get("label_map", {}),
                },
            )
            mlflow.log_params(
                {
                    "winner_algorithm": winner["algorithm"],
                    "winner_n_clusters": winner["n_clusters"],
                    "winner_run_name": winner["run_name"],
                    "selection_pool": selection["selection_pool"],
                }
            )
            mlflow.log_metrics(
                {
                    "winner_silhouette": float(winner.get("silhouette") or 0),
                    "winner_composite_score": float(winner.get("composite_score") or 0),
                    "n_passing_gates": float(selection["n_passing_gates"]),
                }
            )

        payload = {
            "data": {
                "as_of_date": as_of.isoformat(),
                "feature_set_version": feature_set_version,
                "features_uri": source_uri,
                "n_rows": int(len(features)),
                "feature_columns": feature_columns,
            },
            "winner": winner,
            "results": selection["results"],
            "n_passing_gates": selection["n_passing_gates"],
            "selection_pool": selection["selection_pool"],
            "period_stability": period_metrics,
            "mlflow": {
                "tracking_uri": uri,
                "experiment_name": experiment_name,
                "parent_run_id": parent_run_id,
                **registry,
            },
            "gates": active_gates,
        }
        payload = _json_safe(payload)

        out_dir = data_dir("models", "experiments", f"as_of_date={as_of.isoformat()}", parent_run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = out_dir / "comparison.json"
        comparison_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        report_md = render_comparison_report(payload)
        report_path = out_dir / "relatorio-comparativo.md"
        report_path.write_text(report_md, encoding="utf-8")
        write_manifest(
            out_dir,
            {
                "parent_run_id": parent_run_id,
                "winner_run_name": winner.get("run_name"),
                "model_version": registry.get("model_version"),
                "aliases": registry.get("aliases"),
            },
        )
        mlflow.log_artifact(str(comparison_path))
        mlflow.log_artifact(str(report_path))

        if write_docs_report:
            docs = project_root() / "docs" / "aula-03"
            docs.mkdir(parents=True, exist_ok=True)
            (docs / "relatorio-comparativo.md").write_text(report_md, encoding="utf-8")

        logger.info(
            "Experimento concluído parent=%s winner=%s version=%s",
            parent_run_id,
            winner.get("run_name"),
            registry.get("model_version"),
        )
        payload["artifacts"] = {
            "comparison_uri": relative_uri(comparison_path),
            "report_uri": relative_uri(report_path),
        }
        return payload
