"""Carrega configurações de caminhos e cadências."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return ROOT


def data_root() -> Path:
    """Raiz dos dados: DATA_ROOT (Docker) ou <repo>/data."""
    env = os.environ.get("DATA_ROOT")
    if env:
        return Path(env)
    return ROOT / "data"


def data_dir(*parts: str) -> Path:
    return data_root() / Path(*parts)


def contracts_dir(*parts: str) -> Path:
    return ROOT / "contracts" / Path(*parts)


def configs_dir(*parts: str) -> Path:
    return ROOT / "configs" / Path(*parts)


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML é necessário. Instale com: pip install pyyaml") from exc

    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
