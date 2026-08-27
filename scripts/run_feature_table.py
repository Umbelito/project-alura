#!/usr/bin/env python3
"""Executa o pipeline da feature table sem Airflow (dev / backfill local).

Exemplos:
  python scripts/run_feature_table.py --as-of-date 2018-08-31
  python scripts/run_feature_table.py --as-of-date 2018-08-31 --window-days 180
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_segmentation.features.pipeline import run_feature_table_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline feature table RFM")
    parser.add_argument("--as-of-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--window-days", type=int, default=365)
    parser.add_argument("--feature-set-version", default="1.0.0")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    result = run_feature_table_pipeline(
        as_of_date=args.as_of_date,
        window_days=args.window_days,
        feature_set_version=args.feature_set_version,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
