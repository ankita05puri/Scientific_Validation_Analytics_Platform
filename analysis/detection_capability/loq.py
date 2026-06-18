"""Reusable Limit of Quantitation calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_loq_summary(loq_data: pd.DataFrame, target_cv: float) -> pd.DataFrame:
    """Calculate per-concentration LoQ precision, bias, recovery, and pass/fail."""

    summary = (
        loq_data.groupby("Concentration Level", dropna=False)["Observed Result"]
        .agg(N="count", Mean="mean", SD="std")
        .reset_index()
        .sort_values("Concentration Level")
        .reset_index(drop=True)
    )
    summary["CV%"] = np.where(
        summary["Mean"] != 0,
        (summary["SD"] / summary["Mean"]) * 100,
        np.nan,
    )
    summary["Bias %"] = np.where(
        summary["Concentration Level"] != 0,
        ((summary["Mean"] - summary["Concentration Level"]) / summary["Concentration Level"]) * 100,
        np.nan,
    )
    summary["Recovery %"] = np.where(
        summary["Concentration Level"] != 0,
        (summary["Mean"] / summary["Concentration Level"]) * 100,
        np.nan,
    )
    summary["Pass/Fail"] = np.where(summary["CV%"] <= target_cv, "PASS", "FAIL")
    return summary

