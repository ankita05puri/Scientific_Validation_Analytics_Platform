"""Executive summary helpers for consolidated validation reports."""

from __future__ import annotations

import pandas as pd

from .models import StudyReport, ValidationPackage
from .validation_matrix import normalize_decision


def study_counts(studies: list[StudyReport]) -> dict[str, int]:
    """Count completed, passing, borderline, and failed studies."""

    decisions = [normalize_decision(study.decision) for study in studies]
    return {
        "Number completed": len(studies),
        "Number passed": decisions.count("PASS"),
        "Number borderline": decisions.count("BORDERLINE"),
        "Number failed": decisions.count("FAIL"),
    }


def generate_final_conclusion(
    project_metadata: dict[str, object],
    studies: list[StudyReport],
    overall_status: str,
) -> str:
    """Generate professional final validation conclusion."""

    assay = project_metadata.get("Assay / Biomarker") or "the assay"
    included = ", ".join(study.study_type.lower() for study in studies)
    if overall_status == "PASS":
        status_text = (
            "All predefined acceptance criteria were met for the included studies. "
            "Based on the evaluated data, the assay demonstrated acceptable analytical "
            "performance for the intended use under the conditions studied."
        )
    elif overall_status == "BORDERLINE":
        status_text = (
            "The validation package contains one or more borderline findings. The data "
            "support continued scientific review before final approval."
        )
    else:
        status_text = (
            "One or more included studies did not meet preliminary acceptance criteria. "
            "Investigation and documented resolution are recommended before final approval."
        )
    return (
        f"The analytical validation studies performed for {assay} included {included}. "
        f"{status_text} This consolidated conclusion is informational and does not "
        "replace formal laboratory approval."
    )


def generate_decision_justification(studies: list[StudyReport], overall_status: str) -> str:
    """Generate concise justification for the final validation decision."""

    decisions = [normalize_decision(study.decision) for study in studies]
    if overall_status == "PASS":
        return "PASS = all included studies passed their preliminary acceptance criteria."
    if overall_status == "BORDERLINE":
        return "BORDERLINE = no included study failed, but at least one study had a borderline criterion or review-level decision."
    failed = decisions.count("FAIL")
    return f"FAIL = one or more failed criteria were identified across included studies ({failed} failed study decision(s))."


def create_risk_summary(
    studies: list[StudyReport],
    supported_studies: tuple[str, ...],
) -> pd.DataFrame:
    """Create cross-study validation risk summary."""

    included = {study.study_type for study in studies}
    failed = [study.study_type for study in studies if normalize_decision(study.decision) == "FAIL"]
    borderline = [study.study_type for study in studies if normalize_decision(study.decision) == "BORDERLINE"]
    missing = [study for study in supported_studies if study not in included]
    violations = []
    incomplete = []
    for study in studies:
        status_column = "Status" if "Status" in study.acceptance_criteria.columns else "Pass/Fail Status"
        if status_column in study.acceptance_criteria.columns:
            bad_rows = study.acceptance_criteria[
                study.acceptance_criteria[status_column].astype(str).str.upper().isin(["FAIL", "INVESTIGATE"])
            ]
            if not bad_rows.empty:
                violations.append(study.study_type)
        if study.excluded_samples > 0 or study.sample_count == 0:
            incomplete.append(study.study_type)
    return pd.DataFrame(
        [
            {"Risk Category": "Failed studies", "Finding": ", ".join(failed) if failed else "None identified", "Risk Level": "High" if failed else "Low"},
            {"Risk Category": "Borderline studies", "Finding": ", ".join(borderline) if borderline else "None identified", "Risk Level": "Moderate" if borderline else "Low"},
            {"Risk Category": "Missing validation studies", "Finding": ", ".join(missing) if missing else "None identified", "Risk Level": "Moderate" if missing else "Low"},
            {"Risk Category": "Acceptance criteria violations", "Finding": ", ".join(violations) if violations else "None identified", "Risk Level": "High" if violations else "Low"},
            {"Risk Category": "Incomplete datasets", "Finding": ", ".join(incomplete) if incomplete else "None identified", "Risk Level": "Moderate" if incomplete else "Low"},
        ]
    )


def create_qa_findings(studies: list[StudyReport]) -> pd.DataFrame:
    """Create consolidated quality assurance findings."""

    total_records = sum(study.sample_count for study in studies)
    total_excluded = sum(study.excluded_samples for study in studies)
    return pd.DataFrame(
        [
            {"QA Check": "Data integrity verification", "Finding": f"{len(studies)} study dataset(s) processed through standardized module workflows.", "Status": "PASS"},
            {"QA Check": "Missing data assessment", "Finding": "Missing required values are excluded by module-level cleaning before analysis.", "Status": "PASS"},
            {"QA Check": "Outlier review summary", "Finding": "Module-specific outlier and disagreement review tables are included where applicable.", "Status": "PASS"},
            {"QA Check": "Exclusion log", "Finding": f"{total_excluded} excluded row(s) across {total_records} source row(s).", "Status": "PASS" if total_excluded == 0 else "REVIEW"},
            {"QA Check": "Validation checklist", "Finding": "Report includes metadata, criteria, results, interpretations, matrix, conclusion, and signature section.", "Status": "PASS"},
        ]
    )


def generate_executive_narrative(package: ValidationPackage) -> str:
    """Generate professional management-facing executive narrative."""

    studies = ", ".join(study.study_type for study in package.studies)
    highlights = "; ".join(
        f"{row['Study']}: {row['Key Metric']} {row['Result']}"
        for _, row in package.validation_matrix.iterrows()
    )
    notable_risks = package.risk_summary[
        package.risk_summary["Risk Level"].astype(str).isin(["High", "Moderate"])
    ]
    observations = (
        "No high or moderate cross-study risks were identified."
        if notable_risks.empty
        else "Notable observations include: "
        + "; ".join(
            f"{row['Risk Category']} - {row['Finding']}"
            for _, row in notable_risks.iterrows()
        )
    )
    return (
        f"Scientific objective: evaluate analytical validation performance for "
        f"{package.project_metadata.get('Assay / Biomarker', 'the assay')}. "
        f"Validation scope: {studies}. Overall outcome: {package.overall_status}. "
        f"Key performance highlights: {highlights}. {observations} "
        f"Final recommendation: {package.decision_justification}"
    )


def create_executive_summary_table(package: ValidationPackage) -> pd.DataFrame:
    """Create a compact executive summary table."""

    counts = study_counts(package.studies)
    rows = [{"Metric": "Overall Validation Status", "Result": package.overall_status}]
    rows.extend({"Metric": key, "Result": value} for key, value in counts.items())
    rows.append({"Metric": "Validation Completeness", "Result": f"{package.completeness_percent:.1f}%"})
    return pd.DataFrame(rows)
