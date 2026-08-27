"""Carrega configurações de caminhos e cadências."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return ROOT


def data_dir(*parts: str) -> Path:
    return ROOT / "data" / Path(*parts)


def contracts_dir(*parts: str) -> Path:
    return ROOT / "contracts" / Path(*parts)


def configs_dir(*parts: str) -> Path:
    return ROOT / "configs" / Path(*parts)
