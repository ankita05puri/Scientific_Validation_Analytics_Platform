"""Reusable Limit of Detection calculations."""

from __future__ import annotations

import pandas as pd


def calculate_lod(lob: float, low_level_results: pd.Series) -> dict[str, float | int]:
    """Calculate LoD as LoB + 1.645 * SD low-concentration sample."""

    clean_results = pd.to_numeric(low_level_results, errors="coerce").dropna()
    mean_low = float(clean_results.mean())
    sd_low = float(clean_results.std(ddof=1))
    return {
        "N": int(clean_results.count()),
        "Mean Low Sample": mean_low,
        "SD Low Sample": sd_low,
        "LoD": float(lob) + (1.645 * sd_low),
    }

