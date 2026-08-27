"""Validação de schema, obrigatoriedade, duplicidades e integridade referencial."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from customer_segmentation.validation.contracts import load_contract
from customer_segmentation.config import contracts_dir

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: str
    detail: str


@dataclass
class ValidationReport:
    dataset: str
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    row_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "passed": self.passed,
            "row_count": self.row_count,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }


TYPE_MAP = {
    "string": "object",
    "integer": "int",
    "float": "float",
    "timestamp": "datetime",
    "date": "datetime",
}


def _is_nullish(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.isna() | series.astype(str).str.strip().isin(["", "nan", "None", "NaT"])
    return series.isna()


def validate_required_columns(df: pd.DataFrame, contract: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    expected = [c["name"] for c in contract["columns"]]
    missing = [c for c in expected if c not in df.columns]
    results.append(
        CheckResult(
            name="schema_columns",
            passed=len(missing) == 0,
            severity="error",
            detail="ok" if not missing else f"colunas ausentes: {missing}",
        )
    )
    return results


def validate_nullability(df: pd.DataFrame, contract: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    for col_def in contract["columns"]:
        if col_def.get("nullable", True):
            continue
        name = col_def["name"]
        if name not in df.columns:
            continue
        nulls = int(_is_nullish(df[name]).sum())
        results.append(
            CheckResult(
                name=f"not_null:{name}",
                passed=nulls == 0,
                severity="error",
                detail="ok" if nulls == 0 else f"{nulls} nulos em {name}",
            )
        )
    return results


def validate_primary_key(df: pd.DataFrame, contract: dict) -> list[CheckResult]:
    pk = contract.get("primary_key") or []
    if not pk:
        return []
    if any(c not in df.columns for c in pk):
        return [
            CheckResult(
                name="primary_key",
                passed=False,
                severity="error",
                detail=f"PK incompleta no dataframe: {pk}",
            )
        ]
    dupes = int(df.duplicated(subset=pk).sum())
    return [
        CheckResult(
            name="primary_key_unique",
            passed=dupes == 0,
            severity="error",
            detail="ok" if dupes == 0 else f"{dupes} duplicatas em PK {pk}",
        )
    ]


def validate_allowed_values(df: pd.DataFrame, contract: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    for col_def in contract["columns"]:
        allowed = col_def.get("allowed_values")
        name = col_def["name"]
        if not allowed or name not in df.columns:
            continue
        invalid = df.loc[~df[name].isna() & ~df[name].isin(allowed), name]
        n = int(invalid.shape[0])
        sample = invalid.drop_duplicates().head(5).tolist()
        results.append(
            CheckResult(
                name=f"allowed_values:{name}",
                passed=n == 0,
                severity="error",
                detail="ok" if n == 0 else f"{n} inválidos; amostra={sample}",
            )
        )
    return results


def validate_referential_integrity(
    child: pd.DataFrame,
    parent: pd.DataFrame,
    fk_columns: list[str],
    parent_columns: list[str],
    name: str,
) -> CheckResult:
    if any(c not in child.columns for c in fk_columns):
        return CheckResult(name, False, "error", f"FK ausente no filho: {fk_columns}")
    if any(c not in parent.columns for c in parent_columns):
        return CheckResult(name, False, "error", f"PK ausente no pai: {parent_columns}")

    parent_keys = parent[parent_columns].drop_duplicates()
    merged = child[fk_columns].merge(
        parent_keys,
        left_on=fk_columns,
        right_on=parent_columns,
        how="left",
        indicator=True,
    )
    orphans = int((merged["_merge"] == "left_only").sum())
    return CheckResult(
        name=name,
        passed=orphans == 0,
        severity="error",
        detail="ok" if orphans == 0 else f"{orphans} órfãos em {fk_columns}",
    )


def validate_dataframe(df: pd.DataFrame, contract_name: str) -> ValidationReport:
    path = contracts_dir("input") / f"{contract_name}.yaml"
    if not path.is_file():
        path = contracts_dir("output") / f"{contract_name}.yaml"
    contract = load_contract(path)

    checks: list[CheckResult] = []
    checks.extend(validate_required_columns(df, contract))
    checks.extend(validate_nullability(df, contract))
    checks.extend(validate_primary_key(df, contract))
    checks.extend(validate_allowed_values(df, contract))

    row_min = next(
        (q for q in contract.get("quality_checks", []) if q.get("name") == "row_count_min"),
        None,
    )
    if row_min is not None:
        ok = len(df) >= 1
        checks.append(
            CheckResult(
                name="row_count_min",
                passed=ok,
                severity=row_min.get("severity", "error"),
                detail=f"rows={len(df)}",
            )
        )

    errors = [c for c in checks if not c.passed and c.severity == "error"]
    report = ValidationReport(
        dataset=contract_name,
        passed=len(errors) == 0,
        checks=checks,
        row_count=len(df),
    )
    for check in checks:
        level = logging.ERROR if (not check.passed and check.severity == "error") else logging.INFO
        logger.log(
            level,
            "check dataset=%s name=%s passed=%s detail=%s",
            contract_name,
            check.name,
            check.passed,
            check.detail,
        )
    return report


def validate_feature_table(df: pd.DataFrame) -> ValidationReport:
    """Valida a feature table RFM (contrato de saída de features)."""
    path = contracts_dir("output") / "customer_features.yaml"
    contract = load_contract(path)
    checks: list[CheckResult] = []
    checks.extend(validate_required_columns(df, contract))
    checks.extend(validate_nullability(df, contract))
    checks.extend(validate_primary_key(df, contract))

    if "frequency" in df.columns:
        bad = int((df["frequency"] < 1).sum())
        checks.append(
            CheckResult(
                name="frequency_positive",
                passed=bad == 0,
                severity="error",
                detail="ok" if bad == 0 else f"{bad} clientes com frequency < 1",
            )
        )
    if "recency_days" in df.columns:
        bad = int((df["recency_days"] < 0).sum())
        checks.append(
            CheckResult(
                name="recency_non_negative",
                passed=bad == 0,
                severity="error",
                detail="ok" if bad == 0 else f"{bad} recency negativas",
            )
        )
    if "monetary" in df.columns:
        bad = int((df["monetary"] < 0).sum())
        checks.append(
            CheckResult(
                name="monetary_non_negative",
                passed=bad == 0,
                severity="error",
                detail="ok" if bad == 0 else f"{bad} monetary negativas",
            )
        )

    errors = [c for c in checks if not c.passed and c.severity == "error"]
    report = ValidationReport(
        dataset="customer_features",
        passed=len(errors) == 0,
        checks=checks,
        row_count=len(df),
    )
    if not report.passed:
        failed = [c.name for c in errors]
        logger.error("Feature table inválida: %s", failed)
    else:
        logger.info("Feature table válida rows=%s", report.row_count)
    return report


def assert_valid(report: ValidationReport) -> dict[str, Any]:
    payload = report.to_dict()
    if not report.passed:
        failed = [c for c in report.checks if not c.passed and c.severity == "error"]
        details = "; ".join(f"{c.name}: {c.detail}" for c in failed)
        raise ValueError(f"Validação falhou para {report.dataset}: {details}")
    return payload
