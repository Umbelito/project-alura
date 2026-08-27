#!/usr/bin/env python3
"""Scoring batch da base completa com o alias champion.

  python scripts/run_batch_scoring.py --as-of-date 2018-08-31
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_segmentation.scoring.batch import run_batch_scoring  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch scoring — alias champion")
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--feature-set-version", default="1.0.0")
    parser.add_argument("--features-uri", default=None)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    result = run_batch_scoring(
        as_of_date=args.as_of_date,
        feature_set_version=args.feature_set_version,
        features_uri=args.features_uri,
        tracking_uri=args.tracking_uri,
        model_name=args.model_name,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "pipeline"}, indent=2, default=str))


if __name__ == "__main__":
    main()
