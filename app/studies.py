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


LINEARITY_REPORT_SECTIONS = (
    "Study Metadata",
    "Study Objective",
    "Study Design",
    "Acceptance Criteria",
    "Linearity Summary Table",
    "Acceptance Criteria Results",
    "Regression Summary",
    "Visualizations",
    "Interpretation",
    "Notes / Deviations",
    "Preliminary Conclusion",
    "Signature Section",
)


STABILITY_REPORT_SECTIONS = (
    "Study Metadata",
    "Acceptance Criteria",
    "Stability Summary",
    "Timepoint Analysis",
    "Recovery Analysis",
    "Bias Analysis",
    "Storage Condition Comparison",
    "Potential Stability Outliers",
    "Risk Assessment",
    "Visualizations",
    "Interpretation",
    "Analyst Notes",
    "Preliminary Conclusion",
    "Signature Section",
)


ACCURACY_REPORT_SECTIONS = (
    "Study Metadata",
    "Acceptance Criteria",
    "Executive Summary",
    "Accuracy Results",
    "Level-Specific Decision Table",
    "Acceptance Criteria Results",
    "Bias Summary",
    "Recovery Summary",
    "Worst-Case Performance",
    "Visualizations",
    "Interpretation",
    "Analyst Notes",
    "Deviations",
    "Preliminary Conclusion",
    "Signature Section",
)


DETECTION_CAPABILITY_REPORT_SECTIONS = (
    "Study Metadata",
    "Executive Summary",
    "Acceptance Criteria",
    "Calculation Methodology",
    "Data Quality Assessment",
    "LoB Analysis",
    "LoD Analysis",
    "LoQ Analysis",
    "Acceptance Criteria Results",
    "Detection Capability Decision Matrix",
    "Interpretation",
    "Visualizations",
    "Analyzed Dataset",
    "Analyst Notes",
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
    "Accuracy Study": StudyTypeConfig(
        name="Accuracy Study",
        description="Observed-versus-expected accuracy workflow for assigned-value assay validation.",
        metrics=(
            "N",
            "Expected result",
            "Mean observed result",
            "SD",
            "Difference",
            "Absolute difference",
            "Percent bias",
            "Percent recovery",
            "95% confidence intervals",
            "Level-specific decisions",
            "Worst-case performance",
        ),
        acceptance_criteria=(
            "Maximum absolute bias",
            "Maximum absolute percent bias",
            "Recovery range",
            "Pass with caution borderline zone",
        ),
        visualizations=(
            "Expected vs observed plot",
            "Percent bias by level plot",
            "Recovery by level plot",
            "Replicate distribution plot",
            "Accuracy performance heatmap",
            "Individual sample bias plot",
        ),
        report_sections=ACCURACY_REPORT_SECTIONS,
        implemented=True,
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
    "Linearity Study": StudyTypeConfig(
        name="Linearity Study",
        description="Expected-versus-observed analytical range linearity workflow.",
        metrics=(
            "N",
            "Mean observed result",
            "Percent recovery",
            "Percent bias",
            "Slope",
            "Intercept",
            "Correlation r",
            "R²",
            "Analytical range tested",
        ),
        acceptance_criteria=(
            "Minimum R²",
            "Slope limits",
            "Maximum absolute percent bias by level",
            "Percent recovery range",
        ),
        visualizations=(
            "Linearity plot",
            "Residual plot",
            "Percent recovery plot",
        ),
        report_sections=LINEARITY_REPORT_SECTIONS,
        implemented=True,
    ),
    "Stability Study": StudyTypeConfig(
        name="Stability Study",
        description="Timepoint-based stability workflow for specimen and assay validation.",
        metrics=(
            "N",
            "Baseline mean",
            "Timepoint mean",
            "Percent change",
            "Percent recovery",
            "Bias",
            "Maximum observed change",
            "Storage condition differences",
            "Potential outliers",
        ),
        acceptance_criteria=(
            "Maximum percent change from baseline",
            "Minimum recovery",
            "Maximum absolute bias",
            "Pass with caution borderline zone",
        ),
        visualizations=(
            "Stability trend plot",
            "Percent change plot",
            "Recovery plot",
            "Bias plot",
            "Condition difference plot",
            "Individual sample stability plot",
        ),
        report_sections=STABILITY_REPORT_SECTIONS,
        implemented=True,
    ),
    "Detection Capability": StudyTypeConfig(
        name="Detection Capability",
        description="Limit of Blank, Limit of Detection, and Limit of Quantitation validation workflow.",
        metrics=(
            "Mean blank",
            "SD blank",
            "LoB",
            "Mean low-level sample",
            "SD low-level sample",
            "LoD",
            "LoQ CV%",
            "Bias %",
            "Recovery %",
            "Operational LoQ",
            "Data quality assessment",
        ),
        acceptance_criteria=(
            "Maximum acceptable LoB",
            "Maximum acceptable LoD",
            "Target LoQ CV%",
            "Maximum acceptable LoQ concentration",
            "Pass with caution borderline zone",
        ),
        visualizations=(
            "Blank distribution histogram",
            "Blank replicate boxplot",
            "Low-level replicate distribution",
            "LoB vs LoD visualization",
            "CV% vs concentration",
            "Recovery vs concentration",
            "LoQ decision plot",
            "Replicate distribution by concentration",
            "Replicate scatter plot",
            "LoQ precision curve",
            "Detection capability ladder",
            "Distribution density plot",
        ),
        report_sections=DETECTION_CAPABILITY_REPORT_SECTIONS,
        implemented=True,
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
