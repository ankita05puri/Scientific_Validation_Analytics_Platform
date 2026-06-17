"""Study type registry for the validation analytics platform."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StudyTypeConfig:
    """Configuration describing a validation study module."""

    name: str
    description: str
    metrics: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    visualizations: tuple[str, ...]
    report_sections: tuple[str, ...]
    implemented: bool = False


STANDARD_REPORT_SECTIONS = (
    "Study Metadata",
    "Acceptance Criteria",
    "Summary Statistics",
    "Method Comparison Results",
    "Bland-Altman Results",
    "Visualizations",
    "Outlier Review",
    "Analyst Notes",
    "Preliminary Conclusion",
    "Signature Section",
)


PRECISION_REPORT_SECTIONS = (
    "Study Metadata",
    "Acceptance Criteria",
    "Precision Summary",
    "Day-Level Summary",
    "Run-Level Summary",
    "Visualizations",
    "Interpretation",
    "Analyst Notes",
    "Deviations",
    "Preliminary Conclusion",
    "Signature Section",
)


STUDY_TYPES: dict[str, StudyTypeConfig] = {
    "Method Comparison": StudyTypeConfig(
        name="Method Comparison",
        description="Paired reference-versus-candidate method comparison workflow.",
        metrics=(
            "N",
            "Mean difference",
            "Mean percent bias",
            "Correlation r",
            "R²",
            "Regression slope",
            "Regression intercept",
            "Bland-Altman limits of agreement",
        ),
        acceptance_criteria=(
            "Minimum R²",
            "Minimum correlation r",
            "Slope range",
            "Maximum absolute intercept",
            "Maximum absolute mean bias",
            "Maximum absolute mean difference",
            "Minimum percent of samples meeting agreement criteria",
        ),
        visualizations=(
            "Reference vs Candidate scatter plot",
            "Percent bias histogram",
            "Difference plot with limits of agreement",
        ),
        report_sections=STANDARD_REPORT_SECTIONS,
        implemented=True,
    ),
    "Accuracy": StudyTypeConfig(
        name="Accuracy",
        description="Placeholder module for accuracy studies.",
        metrics=("Bias", "Recovery", "Agreement with expected value"),
        acceptance_criteria=("Accuracy limits", "Allowable total error"),
        visualizations=("Accuracy trend plots",),
        report_sections=STANDARD_REPORT_SECTIONS,
    ),
    "Precision Study": StudyTypeConfig(
        name="Precision Study",
        description="Repeated-measurement precision workflow for assay validation.",
        metrics=("N", "Mean", "SD", "CV%", "Level/day/run summaries"),
        acceptance_criteria=("Maximum acceptable CV%",),
        visualizations=(
            "Precision run chart",
            "CV% summary bar chart",
            "Result distribution box plot",
        ),
        report_sections=PRECISION_REPORT_SECTIONS,
        implemented=True,
    ),
    "Linearity": StudyTypeConfig(
        name="Linearity",
        description="Placeholder module for linearity verification.",
        metrics=("Slope", "Intercept", "R²", "Recovery by level"),
        acceptance_criteria=("Minimum R²", "Recovery limits"),
        visualizations=("Linearity plots",),
        report_sections=STANDARD_REPORT_SECTIONS,
    ),
    "Stability": StudyTypeConfig(
        name="Stability",
        description="Placeholder module for specimen and assay stability studies.",
        metrics=("Timepoint bias", "Percent change", "Stability trend"),
        acceptance_criteria=("Maximum percent change",),
        visualizations=("Stability trend plots",),
        report_sections=STANDARD_REPORT_SECTIONS,
    ),
    "Reference Range Verification": StudyTypeConfig(
        name="Reference Range Verification",
        description="Placeholder module for reference interval verification.",
        metrics=("Sample count", "In-range count", "Outlier count"),
        acceptance_criteria=("Minimum in-range percentage",),
        visualizations=("Distribution plots",),
        report_sections=STANDARD_REPORT_SECTIONS,
    ),
    "DBS Validation": StudyTypeConfig(
        name="DBS Validation",
        description="Placeholder module for dried blood spot validation workflows.",
        metrics=("Correction factor", "Bias", "Recovery", "Agreement"),
        acceptance_criteria=("DBS bias limits", "Recovery limits"),
        visualizations=("DBS correction plots",),
        report_sections=STANDARD_REPORT_SECTIONS,
    ),
    "Microtainer Validation": StudyTypeConfig(
        name="Microtainer Validation",
        description="Placeholder module for microtainer specimen validation workflows.",
        metrics=("Bias", "Correlation", "Regression", "Agreement"),
        acceptance_criteria=("Specimen comparison limits",),
        visualizations=("Specimen comparison plots",),
        report_sections=STANDARD_REPORT_SECTIONS,
    ),
}


def get_study_type_names() -> list[str]:
    """Return study type names in registry order."""

    return list(STUDY_TYPES.keys())


def get_study_type_config(name: str) -> StudyTypeConfig:
    """Return configuration for a study type."""

    return STUDY_TYPES[name]
