"""Reusable acceptance criteria helpers for validation modules."""

from __future__ import annotations

import pandas as pd


def normalize_status(status: object) -> str:
    """Normalize module-specific decision labels for consistent review displays."""

    normalized = str(status).strip().upper()
    if normalized in {"PASS", "PASSED"}:
        return "PASS"
    if normalized in {"BORDERLINE", "REVIEW", "PASS WITH CAUTION"}:
        return "BORDERLINE"
    return "FAIL"


def count_statuses(table: pd.DataFrame, column: str = "Pass/Fail Status") -> dict[str, int]:
    """Count PASS, BORDERLINE, and FAIL statuses in a criteria or level table."""

    if table.empty or column not in table.columns:
        return {"PASS": 0, "BORDERLINE": 0, "FAIL": 0}
    normalized = table[column].map(normalize_status)
    return {
        "PASS": int((normalized == "PASS").sum()),
        "BORDERLINE": int((normalized == "BORDERLINE").sum()),
        "FAIL": int((normalized == "FAIL").sum()),
    }

