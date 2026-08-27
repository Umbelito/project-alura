"""Quality gates de promoção: aprova ou rejeita o alias champion."""

from __future__ import annotations

import json
import logging
from typing import Any

from customer_segmentation.config import configs_dir, load_yaml
from customer_segmentation.training.metrics import cluster_distribution
from customer_segmentation.training.selection import _finite
from customer_segmentation.training.tracking import CHAMPION_ALIAS, CANDIDATE_ALIAS, _set_alias

logger = logging.getLogger(__name__)


def load_quality_gate_config() -> dict[str, Any]:
    return load_yaml(configs_dir("quality_gates.yaml"))


def _check(name: str, passed: bool, detail: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"name": name, "passed": bool(passed), "detail": detail}
    if extra:
        payload.update(extra)
    return payload


def gate_data_failure(feature_validation_passed: bool, detail: str = "") -> dict[str, Any]:
    return _check(
        "data_failure",
        feature_validation_passed,
        detail or ("feature table válida" if feature_validation_passed else "falha de dados/contrato"),
    )


def gate_empty_or_unstable_clusters(
    labels,
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    dist = cluster_distribution(labels)
    k_min, k_max = thresholds.get("n_clusters_between", [3, 10])
    empty_ok = dist["min_size"] >= int(thresholds.get("min_cluster_size", 1))
    share_ok = dist["min_share"] >= float(thresholds.get("min_cluster_share", 0.0))
    k_ok = k_min <= dist["n_clusters"] <= k_max
    ari = _finite(metrics.get("stability_ari_subsample"), -1)
    stable_ok = ari >= float(thresholds.get("ari_subsample_min", 0.0))
    passed = empty_ok and share_ok and k_ok and stable_ok
    return _check(
        "unstable_or_empty_clusters",
        passed,
        (
            f"k={dist['n_clusters']} min_size={dist['min_size']} "
            f"min_share={dist['min_share']:.4f} ari={ari:.3f}"
        ),
        extra={"distribution": dist, "stability_ari_subsample": ari},
    )


def gate_metric_regression(
    candidate: dict[str, Any],
    champion: dict[str, Any] | None,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    sil_min = float(thresholds.get("silhouette_min", 0.20))
    sil = _finite(candidate.get("silhouette"), -1)
    abs_ok = sil >= sil_min
    if not champion:
        return _check(
            "metric_regression",
            abs_ok,
            f"sem champion prévio; silhouette={sil:.3f} (min {sil_min})",
            extra={"skipped_vs_champion": True, "candidate_silhouette": sil},
        )
    drop_max = float(thresholds.get("silhouette_max_drop_vs_champion", 0.05))
    champ_sil = _finite(champion.get("silhouette"), sil)
    drop = champ_sil - sil
    db_cand = _finite(candidate.get("davies_bouldin"), 0)
    db_champ = _finite(champion.get("davies_bouldin"), db_cand)
    db_increase = db_cand - db_champ
    db_ok = db_increase <= float(thresholds.get("davies_bouldin_max_increase_vs_champion", 0.40))
    passed = abs_ok and drop <= drop_max and db_ok
    return _check(
        "metric_regression",
        passed,
        f"sil {sil:.3f} vs champion {champ_sil:.3f} (drop={drop:.3f}); db Δ={db_increase:.3f}",
        extra={
            "candidate_silhouette": sil,
            "champion_silhouette": champ_sil,
            "silhouette_drop": drop,
            "skipped_vs_champion": False,
        },
    )


def evaluate_promotion(
    *,
    data_ok: bool,
    data_detail: str = "",
    labels,
    candidate_metrics: dict[str, Any],
    champion_metrics: dict[str, Any] | None,
    signature_ok: bool,
    signature_detail: str = "",
    integration_ok: bool,
    integration_detail: str = "",
    smoke_ok: bool = True,
    smoke_detail: str = "",
    thresholds: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or load_quality_gate_config()
    thr = thresholds or cfg.get("thresholds") or {}
    checks = [
        gate_data_failure(data_ok, data_detail),
        gate_metric_regression(candidate_metrics, champion_metrics, thr),
        gate_empty_or_unstable_clusters(labels, candidate_metrics, thr),
        _check("signature_incompatible", signature_ok, signature_detail or "assinatura ok"),
        _check("integration_failure", integration_ok, integration_detail or "integração ok"),
        _check("smoke", smoke_ok, smoke_detail or "smoke ok"),
    ]
    blocking = set(cfg.get("blocking_gates") or [])
    blocking_failed = [c["name"] for c in checks if c["name"] in blocking and not c["passed"]]
    # smoke alimenta integration_failure se a política listar só os 5 nomes da aula
    if not smoke_ok and "integration_failure" in blocking and "smoke" not in blocking:
        if "integration_failure" not in blocking_failed:
            blocking_failed.append("integration_failure")
            for check in checks:
                if check["name"] == "integration_failure":
                    check["passed"] = False
                    check["detail"] = f"{check['detail']}; smoke falhou"
    decision = cfg.get("decision_approved", "approved")
    if blocking_failed:
        decision = cfg.get("decision_rejected", "rejected")
    report = {
        "decision": decision,
        "approved": decision == cfg.get("decision_approved", "approved"),
        "blocking_failed": blocking_failed,
        "checks": checks,
    }
    logger.info("quality gates decision=%s failed=%s", decision, blocking_failed)
    return report


def tags_from_report(report: dict[str, Any], prefix: str = "quality_gate_") -> dict[str, str]:
    tags = {
        f"{prefix}decision": str(report["decision"]),
        f"{prefix}approved": str(report["approved"]).lower(),
        f"{prefix}blocking_failed": json.dumps(report.get("blocking_failed") or [], ensure_ascii=False),
    }
    for check in report.get("checks") or []:
        tags[f"{prefix}{check['name']}"] = "pass" if check["passed"] else "fail"
        tags[f"{prefix}{check['name']}_detail"] = str(check.get("detail") or "")[:500]
    return tags


def persist_gate_tags(
    tags: dict[str, str],
    *,
    model_name: str | None = None,
    model_version: str | None = None,
) -> None:
    """Grava tags no run ativo e na versão do registry, se houver contexto MLflow."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:  # pragma: no cover
        logger.warning("MLflow indisponível — tags não gravadas")
        return

    try:
        mlflow.set_tags(tags)
    except Exception as exc:
        logger.warning("tags de run não gravadas: %s", exc)

    if not model_name or not model_version:
        return
    client = MlflowClient()
    for key, value in tags.items():
        try:
            client.set_model_version_tag(model_name, str(model_version), key, value)
        except Exception as exc:
            logger.warning("tag de versão %s falhou: %s", key, exc)


def apply_promotion_decision(
    report: dict[str, Any],
    *,
    model_name: str,
    model_version: str,
    promote: bool = True,
) -> dict[str, Any]:
    """Promove champion só se aprovado; candidate permanece em qualquer caso."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    _set_alias(client, model_name, CANDIDATE_ALIAS, model_version)
    promoted = False
    if promote and report.get("approved"):
        _set_alias(client, model_name, CHAMPION_ALIAS, model_version)
        promoted = True
        logger.info("Promovido champion %s v%s", model_name, model_version)
    else:
        logger.info("Promoção recusada para %s v%s", model_name, model_version)
    return {
        "model_name": model_name,
        "model_version": str(model_version),
        "decision": report["decision"],
        "champion_promoted": promoted,
        "candidate_alias": CANDIDATE_ALIAS,
        "champion_alias": CHAMPION_ALIAS if promoted else None,
    }
