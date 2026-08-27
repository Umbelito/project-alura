"""Persistência compartilhada e metadados leves entre tarefas.

O Airflow não garante que tarefas consecutivas rodem no mesmo worker.
Datasets ficam em armazenamento compartilhado; entre tarefas circulam
apenas URIs e metadados (contagens, versões, partições).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from customer_segmentation.config import data_dir, data_root, project_root

logger = logging.getLogger(__name__)

SUCCESS_MARKER = "_SUCCESS"
MANIFEST_NAME = "manifest.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class DatasetRef:
    """Referência leve a um dataset no armazenamento compartilhado."""

    uri: str
    dataset: str
    as_of_date: str | None = None
    feature_set_version: str | None = None
    partition: dict[str, Any] = field(default_factory=dict)
    row_count: int | None = None
    checksum: str | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DatasetRef:
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in payload.items() if k in known})

    def absolute_path(self, root: Path | None = None) -> Path:
        base = root or project_root()
        path = Path(self.uri)
        return path if path.is_absolute() else base / path


def relative_uri(path: Path, root: Path | None = None) -> str:
    resolved = path.resolve()
    bases = [root] if root is not None else [project_root(), data_root().parent]
    for base in bases:
        try:
            return str(resolved.relative_to(base.resolve())).replace("\\", "/")
        except ValueError:
            continue
    try:
        return f"data/{resolved.relative_to(data_root().resolve()).as_posix()}"
    except ValueError:
        return str(path).replace("\\", "/")


def resolve_data_uri(uri: str | None) -> Path | None:
    """Resolve URI gravada no pointer (dev local ou DATA_ROOT no Docker)."""
    if not uri:
        return None
    raw = str(uri).replace("\\", "/")
    candidates: list[Path] = [Path(raw)]
    if not Path(raw).is_absolute():
        candidates.append(project_root() / raw)
        candidates.append(data_root() / raw)
    if raw.startswith("data/"):
        candidates.append(data_root() / raw[len("data/") :])
    parts = Path(raw).parts
    if "scores" in parts:
        idx = list(parts).index("scores")
        candidates.append(data_root().joinpath(*parts[idx:]))
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def landing_partition_path(dataset: str, year: int, month: int) -> Path:
    return data_dir(
        "landing",
        dataset,
        f"year={year:04d}",
        f"month={month:02d}",
        f"{dataset}.csv",
    )


def landing_customers_path() -> Path:
    return data_dir("landing", "customers", "snapshot=full", "customers.csv")


def processed_partition_path(dataset: str, year: int, month: int) -> Path:
    return data_dir(
        "processed",
        dataset,
        f"year={year:04d}",
        f"month={month:02d}",
        f"{dataset}.parquet",
    )


def processed_customers_path() -> Path:
    return data_dir("processed", "customers", "snapshot=full", "customers.parquet")


def processed_joined_path(as_of: date) -> Path:
    return data_dir(
        "processed",
        "orders_enriched",
        f"as_of_date={as_of.isoformat()}",
        "orders_enriched.parquet",
    )


def features_dir(version: str, as_of: date) -> Path:
    return data_dir("features", f"v{version}", f"as_of_date={as_of.isoformat()}")


def features_table_path(version: str, as_of: date) -> Path:
    return features_dir(version, as_of) / "customer_features.parquet"


def scores_dir(as_of: date) -> Path:
    return data_dir("scores", f"as_of_date={as_of.isoformat()}")


def scores_table_path(as_of: date) -> Path:
    return scores_dir(as_of) / "customer_segments.parquet"


def scores_current_pointer_path() -> Path:
    return data_root() / "scores" / "current.json"


def latest_scores_table_path() -> Path | None:
    """Resolve a partição corrente (pointer) ou a as_of_date mais recente."""
    pointer = scores_current_pointer_path()
    if pointer.is_file():
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        resolved = resolve_data_uri(payload.get("uri"))
        if resolved is not None:
            return resolved
    scores_root = data_root() / "scores"
    if not scores_root.is_dir():
        return None
    partitions = sorted(scores_root.glob("as_of_date=*/customer_segments.parquet"))
    return partitions[-1] if partitions else None


def write_manifest(directory: Path, payload: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    enriched = {
        **payload,
        "written_at": _utc_now_iso(),
    }
    path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def mark_success(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / SUCCESS_MARKER
    marker.write_text(_utc_now_iso(), encoding="utf-8")
    return marker


def is_partition_complete(directory: Path) -> bool:
    return (directory / SUCCESS_MARKER).is_file()


def write_parquet_idempotent(df, path: Path) -> DatasetRef:
    """Escreve parquet de forma idempotente (sobrescreve a partição)."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df deve ser um pandas.DataFrame")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    mark_success(path.parent)

    ref = DatasetRef(
        uri=relative_uri(path),
        dataset=path.stem,
        row_count=int(len(df)),
        created_at=_utc_now_iso(),
    )
    logger.info("Dataset persistido uri=%s rows=%s", ref.uri, ref.row_count)
    return ref
