"""Cross-study validation matrix generation."""

from __future__ import annotations

import pandas as pd

from .models import StudyReport


def normalize_decision(decision: str) -> str:
    """Normalize module-specific decisions to PASS, BORDERLINE, or FAIL."""

    normalized = str(decision).strip().upper()
    if normalized == "PASS":
        return "PASS"
    if normalized in {"BORDERLINE", "REVIEW", "PASS WITH CAUTION"}:
        return "BORDERLINE"
    return "FAIL"


def determine_overall_status(studies: list[StudyReport]) -> str:
    """Determine overall validation package decision."""

    decisions = [normalize_decision(study.decision) for study in studies]
    if any(decision == "FAIL" for decision in decisions):
        return "FAIL"
    if any(decision == "BORDERLINE" for decision in decisions):
        return "BORDERLINE"
    return "PASS"


def create_validation_matrix(studies: list[StudyReport]) -> pd.DataFrame:
    """Create consolidated cross-study validation matrix."""

    rows = []
    for study in studies:
        key_metric = "Primary Metric"
        result = "Not available"
        if not study.key_findings.empty:
            key_metric = str(study.key_findings.iloc[0]["Metric"])
            result = str(study.key_findings.iloc[0]["Result"])
        rows.append(
            {
                "Study": study.study_type,
                "Status": normalize_decision(study.decision),
                "Key Metric": key_metric,
                "Result": result,
                "Decision": normalize_decision(study.decision),
            }
        )
    return pd.DataFrame(rows)


def coverage_table(studies: list[StudyReport], supported_studies: tuple[str, ...]) -> pd.DataFrame:
    """Create validation coverage table with completion indicators."""

    included = {study.study_type for study in studies}
    return pd.DataFrame(
        [
            {
                "Study": study_name,
                "Included": "YES" if study_name in included else "NO",
                "Completion Indicator": "Complete" if study_name in included else "Missing",
            }
            for study_name in supported_studies
        ]
    )


def completeness_percent(studies: list[StudyReport], supported_studies: tuple[str, ...]) -> float:
    """Calculate validation coverage completeness percentage."""

    if not supported_studies:
        return 0.0
    included = {study.study_type for study in studies}
    return (len(included.intersection(set(supported_studies))) / len(supported_studies)) * 100
