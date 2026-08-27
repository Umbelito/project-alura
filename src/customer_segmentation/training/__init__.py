"""Treinamento, avaliação e registro de modelos no MLflow."""

from customer_segmentation.training.experiment import run_training_experiment
from customer_segmentation.training.gates_run import run_quality_gates
from customer_segmentation.training.preprocess import (
    DEFAULT_FEATURE_COLUMNS,
    build_preprocessor,
    select_model_frame,
)

__all__ = [
    "DEFAULT_FEATURE_COLUMNS",
    "build_preprocessor",
    "run_quality_gates",
    "run_training_experiment",
    "select_model_frame",
]
