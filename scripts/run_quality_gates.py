#!/usr/bin/env python3
"""Aplica quality gates ao modelo @candidate e promove ou rejeita o champion.

  python scripts/run_quality_gates.py --as-of-date 2018-08-31
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_segmentation.training.gates_run import run_quality_gates  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality gates de promoção MLflow")
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--feature-set-version", default="1.0.0")
    parser.add_argument("--features-uri", default=None)
    parser.add_argument("--model-uri", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    result = run_quality_gates(
        as_of_date=args.as_of_date,
        feature_set_version=args.feature_set_version,
        features_uri=args.features_uri,
        model_uri=args.model_uri,
        model_name=args.model_name,
        model_version=args.model_version,
        tracking_uri=args.tracking_uri,
        promote=not args.no_promote,
    )
    summary = {
        "decision": result["report"]["decision"],
        "approved": result["report"]["approved"],
        "blocking_failed": result["report"]["blocking_failed"],
        "model_version": result["model_version"],
        "promotion": result.get("promotion"),
        "scores_uri": (result.get("scores_ref") or {}).get("uri"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not result["report"]["approved"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
