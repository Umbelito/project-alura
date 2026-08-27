"""Garante que o script de particionamento está presente e importável."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "partition_incremental_source.py"


def test_partition_script_exists() -> None:
    assert SCRIPT.is_file()


def test_partition_helpers_load() -> None:
    spec = importlib.util.spec_from_file_location("partition_incremental_source", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._month_key("2018-08-15 10:00:00") == "2018-08"
    assert module._month_key("") is None
