"""Utilitários para carregar contratos YAML da Aula 1."""

from __future__ import annotations

from pathlib import Path

from customer_segmentation.config import contracts_dir


def list_input_contracts() -> list[Path]:
    return sorted((contracts_dir("input")).glob("*.yaml"))


def list_output_contracts() -> list[Path]:
    return sorted((contracts_dir("output")).glob("*.yaml"))


def load_contract(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyYAML é necessário para carregar contratos. "
            "Instale com: pip install pyyaml"
        ) from exc

    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)
