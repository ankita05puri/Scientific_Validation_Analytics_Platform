"""Data models for consolidated validation reports."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class StudyReport:
    """Normalized report payload for one validation study."""

    study_type: str
    study_name: str
    status: str
    decision: str
    date: str
    objective: str
    design: str
    metadata: dict[str, object]
    acceptance_criteria: pd.DataFrame
    key_findings: pd.DataFrame
    interpretation: str
    visualizations: dict[str, str] = field(default_factory=dict)
    raw_outputs: dict[str, pd.DataFrame] = field(default_factory=dict)
    analysis_version: str = "v1.0.0"
    source_dataset: str = "Built-in sample dataset"
    analysis_timestamp: str = ""
    sample_count: int = 0
    excluded_samples: int = 0
    statistical_methods: str = ""
    conclusion: str = ""


@dataclass
class ValidationPackage:
    """Consolidated report payload for a validation project."""

    project_metadata: dict[str, object]
    studies: list[StudyReport]
    validation_matrix: pd.DataFrame
    overall_status: str
    final_conclusion: str
    coverage_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    completeness_percent: float = 0.0
    risk_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    qa_findings: pd.DataFrame = field(default_factory=pd.DataFrame)
    executive_narrative: str = ""
    decision_justification: str = ""
