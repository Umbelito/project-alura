"""Fixtures compartilhadas dos testes."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_features() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 90
    recency = np.r_[
        rng.integers(5, 20, 30),
        rng.integers(40, 80, 30),
        rng.integers(120, 200, 30),
    ]
    freq = np.r_[rng.integers(3, 6, 30), rng.integers(2, 3, 30), np.ones(30, dtype=int)]
    monetary = np.r_[rng.uniform(300, 800, 30), rng.uniform(80, 200, 30), rng.uniform(20, 60, 30)]
    return pd.DataFrame(
        {
            "customer_unique_id": [f"c{i}" for i in range(n)],
            "as_of_date": "2018-08-31",
            "recency_days": recency.astype(int),
            "frequency": freq.astype(int),
            "monetary": monetary.astype(float),
            "r_score": np.clip(6 - (recency // 40), 1, 5).astype(int),
            "f_score": np.clip(freq, 1, 5).astype(int),
            "m_score": np.clip((monetary // 150).astype(int) + 1, 1, 5),
            "rfm_score": ["111"] * n,
            "avg_ticket": (monetary / np.maximum(freq, 1)).astype(float),
            "avg_items_per_order": rng.uniform(1, 3, n),
            "avg_freight": rng.uniform(5, 20, n),
            "avg_installments": rng.uniform(1, 4, n),
            "preferred_payment_type": ["credit_card"] * n,
            "customer_state": ["SP"] * n,
            "customer_age_days": rng.integers(0, 200, n),
            "avg_days_between_orders": np.nan,
            "first_purchase": pd.Timestamp("2018-01-01"),
            "last_purchase": pd.Timestamp("2018-08-01"),
            "feature_set_version": "1.0.0",
            "built_at": "2018-09-01T00:00:00Z",
        }
    )


@pytest.fixture
def as_of() -> date:
    return date(2018, 8, 31)
