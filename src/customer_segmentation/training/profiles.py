"""Perfis interpretáveis de segmentos a partir dos centróides RFM."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Ordem de prioridade: o primeiro match vence.
_RULES: list[tuple[str, str]] = [
    ("champions", "Recentes, compram com frequência e alto valor"),
    ("loyal_customers", "Frequência alta e valor acima da média"),
    ("big_spenders", "Ticket/valor alto com pouca recorrência"),
    ("new_customers", "Compra recente, ainda pouca história"),
    ("at_risk", "Já foram ativos, recência ruim"),
    ("hibernating", "Inativos, baixa frequência e baixo valor"),
    ("need_attention", "Perfil intermediário — campanha de reativação leve"),
]


def _label_from_z(r_recent: float, f_z: float, m_z: float) -> str:
    """r_recent > 0 significa mais recente que a média (recency invertida)."""
    if r_recent > 0.25 and f_z > 0.25 and m_z > 0.25:
        return "champions"
    if f_z > 0.35 and m_z > 0:
        return "loyal_customers"
    if m_z > 0.45 and f_z <= 0.1:
        return "big_spenders"
    if r_recent > 0.35 and f_z < 0 and m_z < 0.35:
        return "new_customers"
    if r_recent < -0.25 and f_z > 0:
        return "at_risk"
    if r_recent < -0.25 and f_z <= 0 and m_z <= 0:
        return "hibernating"
    return "need_attention"


def build_segment_profiles(
    features: pd.DataFrame,
    labels: np.ndarray,
    rfm_columns: tuple[str, str, str] = ("recency_days", "frequency", "monetary"),
) -> dict[str, Any]:
    """Agrega estatísticas por cluster e atribui rótulos de negócio únicos."""
    recency_col, freq_col, monetary_col = rfm_columns
    frame = features.copy()
    frame["segment_id"] = np.asarray(labels)

    global_mean = frame[[recency_col, freq_col, monetary_col]].mean()
    global_std = frame[[recency_col, freq_col, monetary_col]].std(ddof=0).replace(0, 1)

    profiles: list[dict[str, Any]] = []
    used_labels: dict[str, int] = {}

    for segment_id, grp in frame.groupby("segment_id"):
        means = grp[[recency_col, freq_col, monetary_col]].mean()
        r_z = float((means[recency_col] - global_mean[recency_col]) / global_std[recency_col])
        f_z = float((means[freq_col] - global_mean[freq_col]) / global_std[freq_col])
        m_z = float((means[monetary_col] - global_mean[monetary_col]) / global_std[monetary_col])
        r_recent = -r_z  # menor recency_days = melhor
        base = _label_from_z(r_recent, f_z, m_z)
        used_labels[base] = used_labels.get(base, 0) + 1
        label = base if used_labels[base] == 1 else f"{base}_{used_labels[base]}"
        description = next(desc for name, desc in _RULES if name == base)

        profiles.append(
            {
                "segment_id": int(segment_id),
                "segment_label": label,
                "size": int(len(grp)),
                "share": float(len(grp) / len(frame)),
                "recency_days_mean": float(means[recency_col]),
                "frequency_mean": float(means[freq_col]),
                "monetary_mean": float(means[monetary_col]),
                "avg_ticket_mean": float(grp["avg_ticket"].mean()) if "avg_ticket" in grp else None,
                "z_recency_inverted": r_recent,
                "z_frequency": f_z,
                "z_monetary": m_z,
                "description": description,
            }
        )

    profiles.sort(key=lambda p: p["segment_id"])
    label_map = {str(p["segment_id"]): p["segment_label"] for p in profiles}
    return {"profiles": profiles, "label_map": label_map}
