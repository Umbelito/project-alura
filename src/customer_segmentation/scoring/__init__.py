"""Scoring batch e publicação da tabela customer_segments."""

from customer_segmentation.scoring.assign import assign_segments, persist_segments, publish_current_pointer
from customer_segmentation.scoring.batch import load_champion, run_batch_scoring

__all__ = [
    "assign_segments",
    "load_champion",
    "persist_segments",
    "publish_current_pointer",
    "run_batch_scoring",
]
