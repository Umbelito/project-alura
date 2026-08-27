"""Scoring batch e publicação da tabela customer_segments."""

from customer_segmentation.scoring.assign import assign_segments, persist_segments

__all__ = ["assign_segments", "persist_segments"]
