#!/usr/bin/env python3
"""Executa o experimento de clustering RFM e registra o candidato no MLflow.

Exemplos:
  python scripts/run_training_experiment.py --as-of-date 2018-08-31 --write-docs-report
  python scripts/run_training_experiment.py --as-of-date 2018-08-31 --quick
  python scripts/run_training_experiment.py --as-of-date 2018-08-31 --compare-as-of-date 2018-05-31
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_segmentation.training.experiment import run_training_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimento MLflow de segmentação RFM")
    parser.add_argument("--as-of-date", required=True, help="YYYY-MM-DD da feature table")
    parser.add_argument("--feature-set-version", default="1.0.0")
    parser.add_argument("--features-uri", default=None)
    parser.add_argument("--compare-as-of-date", default=None, help="Período extra para ARI temporal")
    parser.add_argument("--tracking-uri", default=None, help="Default: file:./mlruns")
    parser.add_argument("--quick", action="store_true", help="Grade reduzida (k-means 3 e 4)")
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--set-champion", action="store_true", help="Força alias champion neste candidato")
    parser.add_argument("--write-docs-report", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    result = run_training_experiment(
        as_of_date=args.as_of_date,
        feature_set_version=args.feature_set_version,
        features_uri=args.features_uri,
        compare_as_of_date=args.compare_as_of_date,
        quick=args.quick,
        tracking_uri=args.tracking_uri,
        register=not args.no_register,
        set_champion=args.set_champion,
        write_docs_report=args.write_docs_report,
    )
    summary = {
        "winner": result["winner"].get("run_name"),
        "algorithm": result["winner"].get("algorithm"),
        "n_clusters": result["winner"].get("n_clusters"),
        "silhouette": result["winner"].get("silhouette"),
        "gates_passed": result["winner"].get("gates", {}).get("passed"),
        "mlflow": result.get("mlflow"),
        "artifacts": result.get("artifacts"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
