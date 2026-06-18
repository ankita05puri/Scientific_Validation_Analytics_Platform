"""Reusable Limit of Blank calculations."""

from __future__ import annotations

import pandas as pd


def calculate_lob(blank_results: pd.Series) -> dict[str, float | int]:
    """Calculate LoB as mean blank + 1.645 * SD blank."""

    clean_results = pd.to_numeric(blank_results, errors="coerce").dropna()
    mean_blank = float(clean_results.mean())
    sd_blank = float(clean_results.std(ddof=1))
    return {
        "N": int(clean_results.count()),
        "Mean Blank": mean_blank,
        "SD Blank": sd_blank,
        "LoB": mean_blank + (1.645 * sd_blank),
    }

