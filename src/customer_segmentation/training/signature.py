"""Verificação de assinatura e compatibilidade do modelo sklearn/MLflow."""

from __future__ import annotations

from typing import Any

import pandas as pd

from customer_segmentation.training.preprocess import DEFAULT_FEATURE_COLUMNS, select_model_frame


def pipeline_input_columns(pipeline) -> list[str]:
    prep = pipeline.named_steps.get("preprocess") if hasattr(pipeline, "named_steps") else None
    if prep is None or not hasattr(prep, "transformers_"):
        return list(DEFAULT_FEATURE_COLUMNS)
    cols: list[str] = []
    for _name, _trans, columns in prep.transformers_:
        cols.extend([str(c) for c in columns])
    return cols


def check_signature_compatibility(
    pipeline,
    sample: pd.DataFrame,
    expected_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Garante colunas de entrada, predict estável e falha se o schema quebrar."""
    expected = list(expected_columns or DEFAULT_FEATURE_COLUMNS)
    fitted_cols = pipeline_input_columns(pipeline)
    missing_fitted = [c for c in expected if c not in fitted_cols]
    X = select_model_frame(sample, expected)
    predict_ok = True
    predict_error = None
    n_pred = 0
    try:
        pred = pipeline.predict(X)
        n_pred = int(len(pred))
        if n_pred != len(X):
            predict_ok = False
            predict_error = f"tamanho da predição {n_pred} != {len(X)}"
    except Exception as exc:  # noqa: BLE001
        predict_ok = False
        predict_error = str(exc)

    missing_col_fails = False
    try:
        broken = X.drop(columns=[expected[0]])
        pipeline.predict(broken)
    except Exception:
        missing_col_fails = True

    passed = (
        not missing_fitted
        and predict_ok
        and n_pred == len(X)
        and missing_col_fails
    )
    return {
        "name": "signature_incompatible",
        "passed": passed,
        "expected_columns": expected,
        "fitted_columns": fitted_cols,
        "missing_fitted_columns": missing_fitted,
        "predict_ok": predict_ok,
        "predict_error": predict_error,
        "missing_column_raises": missing_col_fails,
        "n_predicted": n_pred,
    }


def check_mlflow_signature(model_uri: str, expected_columns: list[str]) -> dict[str, Any]:
    """Compara a signature serializada no MLflow com o contrato de features."""
    import mlflow

    pyfunc = mlflow.pyfunc.load_model(model_uri)
    signature = pyfunc.metadata.signature
    input_names: list[str] = []
    if signature is not None and signature.inputs is not None:
        input_names = [item.name for item in signature.inputs]
    missing = [c for c in expected_columns if c not in input_names]
    extra = [c for c in input_names if c not in expected_columns]
    passed = signature is not None and not missing
    return {
        "name": "mlflow_signature",
        "passed": passed,
        "has_signature": signature is not None,
        "input_names": input_names,
        "missing": missing,
        "extra": extra,
        "model_uri": model_uri,
    }
