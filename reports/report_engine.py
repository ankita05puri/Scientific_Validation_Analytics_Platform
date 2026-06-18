"""Core report engine for generating consolidated validation packages."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
for path in (str(APP_DIR), str(PROJECT_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(APP_DIR))
sys.path.insert(1, str(PROJECT_ROOT))
loaded_analysis = sys.modules.get("analysis")
if loaded_analysis is not None:
    loaded_path = Path(getattr(loaded_analysis, "__file__", "")).resolve()
    if loaded_path != (APP_DIR / "analysis.py").resolve():
        del sys.modules["analysis"]

from analysis import (
    build_detection_decision_matrix,
    calculate_percent_samples_meeting_agreement,
    evaluate_acceptance_criteria,
    evaluate_accuracy_criteria,
    evaluate_detection_capability_criteria,
    evaluate_dbs_criteria,
    evaluate_linearity_criteria,
    evaluate_precision_criteria,
    evaluate_stability_criteria,
    run_accuracy_study,
    run_dbs_validation_study,
    run_detection_capability_study,
    run_linearity_study,
    run_method_comparison,
    run_precision_study,
    run_stability_study,
)
from plots import (
    create_accuracy_expected_observed_plot,
    create_blank_distribution_histogram,
    create_dbs_bland_altman_plot,
    create_dbs_scatter_plot,
    create_difference_plot,
    create_linearity_plot,
    create_loq_precision_curve,
    create_precision_cv_bar_chart,
    create_precision_run_chart,
    create_scatter_plot,
    create_stability_percent_change_plot,
    create_stability_trend_plot,
)
from report import (
    build_criteria_table,
    build_summary_table,
    format_accuracy_criteria_table,
    format_accuracy_table,
    format_detection_criteria_table,
    format_detection_overall_summary,
    format_detection_table,
    format_dbs_criteria_table,
    format_dbs_overall_summary,
    format_dbs_table,
    format_linearity_criteria_table,
    format_linearity_regression_summary,
    format_linearity_summary_table,
    format_precision_criteria_table,
    format_precision_summary_table,
    format_stability_criteria_table,
    format_stability_overall_summary,
    format_stability_table,
    generate_accuracy_interpretation,
    generate_dbs_interpretation,
    generate_detection_interpretation,
    generate_interpretation,
    generate_linearity_interpretation,
    generate_precision_interpretation,
    generate_stability_interpretation,
)

from .executive_summary import (
    create_qa_findings,
    create_risk_summary,
    generate_decision_justification,
    generate_executive_narrative,
    generate_final_conclusion,
)
from .models import StudyReport, ValidationPackage
from .validation_matrix import (
    completeness_percent,
    coverage_table,
    create_validation_matrix,
    determine_overall_status,
)


SUPPORTED_STUDIES = (
    "Method Comparison",
    "Precision",
    "Linearity",
    "Stability",
    "Accuracy",
    "Detection Capability",
    "DBS Validation",
)


def _sample_path(root_dir: Path, filename: str) -> Path:
    return root_dir / "data" / "sample_data" / filename


def _criteria_decision(criteria_result: dict[str, object]) -> str:
    decision = str(criteria_result["decision"]).upper()
    if decision in {"REVIEW", "PASS WITH CAUTION"}:
        return "BORDERLINE"
    if decision == "INVESTIGATE":
        return "FAIL"
    return decision


def _base_metadata(
    project_metadata: dict[str, object],
    study_type: str,
    objective: str,
    design: str,
) -> dict[str, object]:
    return {
        "Study Name": f"{project_metadata.get('Assay / Biomarker', 'Assay')} {study_type}",
        "Study Objective": objective,
        "Study Design": design,
        "Assay / Biomarker": project_metadata.get("Assay / Biomarker", "HbA1c"),
        "Specimen Type": project_metadata.get("Specimen Type", "Whole Blood"),
        "Analyst Name": project_metadata.get("Analyst", ""),
        "Study Date": str(project_metadata.get("Study Date", date.today())),
        "Instrument": project_metadata.get("Instrument", ""),
        "Protocol Number": project_metadata.get("Protocol Number", ""),
        "Laboratory Name": project_metadata.get("Laboratory Name", ""),
        "Notes": "",
        "Deviations": "",
        "Conclusions": "",
        "Units": "%",
    }


def _figure_html(figure, include_plotlyjs: bool = False) -> str:
    return figure.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def _traceability(
    data: pd.DataFrame,
    analyzed_data: pd.DataFrame,
    source_dataset: str,
    statistical_methods: str,
    decision: str,
) -> dict[str, object]:
    """Build common audit-trail fields for consolidated study reports."""

    return {
        "analysis_version": "v0.7.1",
        "source_dataset": source_dataset,
        "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sample_count": int(len(data)),
        "excluded_samples": max(int(len(data) - len(analyzed_data)), 0),
        "statistical_methods": statistical_methods,
        "conclusion": f"Preliminary study decision: {decision}.",
    }


def build_method_comparison_report(root_dir: Path, project_metadata: dict[str, object]) -> StudyReport:
    data = pd.read_csv(_sample_path(root_dir, "hba1c_method_comparison.csv"))
    result = run_method_comparison(data, "Reference HbA1c", "Candidate HbA1c", "Sample ID")
    agreement = calculate_percent_samples_meeting_agreement(result.analyzed_data, 5.0)
    criteria_result = evaluate_acceptance_criteria(
        result.summary,
        min_r_squared=0.95,
        max_abs_mean_percent_bias=5.0,
        percent_samples_meeting_agreement=agreement,
    )
    decision = _criteria_decision(criteria_result)
    metadata = _base_metadata(
        project_metadata,
        "Method Comparison",
        "Evaluate agreement between candidate and reference results.",
        "Paired specimen comparison using reference and candidate measurements.",
    )
    metadata.update({"Reference Method": "Reference HbA1c", "Candidate Method": "Candidate HbA1c"})
    interpretation = generate_interpretation(
        result.summary,
        metadata,
        {"Minimum R²": 0.95, "Maximum Absolute Mean Percent Bias": 5.0},
        decision,
    )
    key_findings = pd.DataFrame(
        [
            {"Metric": "R²", "Result": f"{result.summary['R²']:.4f}"},
            {"Metric": "Slope", "Result": f"{result.summary['Slope']:.4f}"},
            {"Metric": "Intercept", "Result": f"{result.summary['Intercept']:.4f}"},
            {"Metric": "Mean Bias", "Result": f"{result.summary['Mean Bias']:.3f}"},
        ]
    )
    return StudyReport(
        study_type="Method Comparison",
        study_name=str(metadata["Study Name"]),
        status="Completed",
        decision=decision,
        date=str(metadata["Study Date"]),
        objective=str(metadata["Study Objective"]),
        design=str(metadata["Study Design"]),
        metadata=metadata,
        acceptance_criteria=build_criteria_table(criteria_result),
        key_findings=key_findings,
        interpretation=interpretation,
        visualizations={
            "Reference vs Candidate Scatter Plot": _figure_html(create_scatter_plot(result.analyzed_data, result.summary), True),
            "Difference Plot": _figure_html(create_difference_plot(result.analyzed_data)),
        },
        raw_outputs={
            "Summary Statistics": build_summary_table(result.summary),
            "Analyzed Dataset": result.analyzed_data,
        },
        **_traceability(
            data,
            result.analyzed_data,
            "hba1c_method_comparison.csv",
            "Paired method comparison using ordinary least-squares regression, Pearson correlation, R-squared, bias statistics, and Bland-Altman difference analysis.",
            decision,
        ),
    )


def build_precision_report(root_dir: Path, project_metadata: dict[str, object]) -> StudyReport:
    data = pd.read_csv(_sample_path(root_dir, "precision_study_hba1c.csv"))
    result = run_precision_study(data, "Result", "Level", "Day", "Run", "Replicate", "Sample ID")
    criteria_result = evaluate_precision_criteria(result.level_summary, 5.0)
    decision = _criteria_decision(criteria_result)
    metadata = _base_metadata(
        project_metadata,
        "Precision",
        "Evaluate repeatability of repeated assay measurements.",
        "Repeated quality control measurements across days, runs, and replicates.",
    )
    interpretation = generate_precision_interpretation(result.level_summary, metadata, 5.0, decision)
    worst_cv = result.level_summary["CV%"].max()
    key_findings = pd.DataFrame(
        [
            {"Metric": "Worst CV%", "Result": f"{worst_cv:.2f}%"},
            {"Metric": "Levels Tested", "Result": str(len(result.level_summary))},
        ]
    )
    return StudyReport(
        study_type="Precision",
        study_name=str(metadata["Study Name"]),
        status="Completed",
        decision=decision,
        date=str(metadata["Study Date"]),
        objective=str(metadata["Study Objective"]),
        design=str(metadata["Study Design"]),
        metadata=metadata,
        acceptance_criteria=format_precision_criteria_table(build_criteria_table(criteria_result)),
        key_findings=key_findings,
        interpretation=interpretation,
        visualizations={
            "Precision Run Chart": _figure_html(create_precision_run_chart(result.analyzed_data), True),
            "CV% Summary Bar Chart": _figure_html(create_precision_cv_bar_chart(result.level_summary)),
        },
        raw_outputs={
            "Precision Summary": format_precision_summary_table(result.level_summary),
            "Day-Level Summary": format_precision_summary_table(result.day_summary),
            "Run-Level Summary": format_precision_summary_table(result.run_summary),
        },
        **_traceability(
            data,
            result.analyzed_data,
            "precision_study_hba1c.csv",
            "Repeated-measure precision analysis using mean, standard deviation, coefficient of variation, and day/run stratified summaries.",
            decision,
        ),
    )


def build_linearity_report(root_dir: Path, project_metadata: dict[str, object]) -> StudyReport:
    data = pd.read_csv(_sample_path(root_dir, "linearity_study_hba1c.csv"))
    result = run_linearity_study(data, "Expected Result", "Observed Result", "Level", "Replicate", "Units", "Include in Analysis")
    criteria = {
        "Minimum R²": 0.99,
        "Slope Lower Limit": 0.95,
        "Slope Upper Limit": 1.05,
        "Maximum Absolute Percent Bias": 10.0,
        "Recovery Lower Limit": 90.0,
        "Recovery Upper Limit": 110.0,
    }
    criteria_result = evaluate_linearity_criteria(
        result.level_summary,
        result.regression_summary,
        criteria["Minimum R²"],
        criteria["Slope Lower Limit"],
        criteria["Slope Upper Limit"],
        criteria["Maximum Absolute Percent Bias"],
        criteria["Recovery Lower Limit"],
        criteria["Recovery Upper Limit"],
    )
    decision = _criteria_decision(criteria_result)
    metadata = _base_metadata(
        project_metadata,
        "Linearity",
        "Evaluate proportionality of observed results across the analytical range.",
        "Expected levels measured in replicate across the reportable analytical range.",
    )
    interpretation = generate_linearity_interpretation(result.level_summary, result.regression_summary, criteria, decision, metadata)
    key_findings = pd.DataFrame(
        [
            {"Metric": "R²", "Result": f"{result.regression_summary['R²']:.4f}"},
            {"Metric": "Slope", "Result": f"{result.regression_summary['Slope']:.4f}"},
            {"Metric": "Recovery Range", "Result": f"{result.level_summary['Percent Recovery'].min():.1f}% to {result.level_summary['Percent Recovery'].max():.1f}%"},
        ]
    )
    return StudyReport(
        study_type="Linearity",
        study_name=str(metadata["Study Name"]),
        status="Completed",
        decision=decision,
        date=str(metadata["Study Date"]),
        objective=str(metadata["Study Objective"]),
        design=str(metadata["Study Design"]),
        metadata=metadata,
        acceptance_criteria=format_linearity_criteria_table(build_criteria_table(criteria_result)),
        key_findings=key_findings,
        interpretation=interpretation,
        visualizations={
            "Linearity Plot": _figure_html(create_linearity_plot(result.level_summary, result.regression_summary), True),
        },
        raw_outputs={
            "Linearity Summary": format_linearity_summary_table(result.level_summary),
            "Regression Summary": format_linearity_regression_summary(result.regression_summary),
        },
        **_traceability(
            data,
            result.analyzed_data,
            "linearity_study_hba1c.csv",
            "Linearity analysis using expected-versus-observed regression, R-squared, slope/intercept, percent recovery, percent bias, and residual review.",
            decision,
        ),
    )


def build_accuracy_report(root_dir: Path, project_metadata: dict[str, object]) -> StudyReport:
    data = pd.read_csv(_sample_path(root_dir, "accuracy_study_hba1c.csv"))
    criteria = {
        "Maximum Absolute Bias": 0.50,
        "Maximum Absolute Percent Bias": 10.0,
        "Minimum Recovery": 90.0,
        "Maximum Recovery": 110.0,
        "Borderline Zone": 2.0,
    }
    result = run_accuracy_study(data, "Sample ID", "Level", "Expected Result", "Observed Result", "Units", "Replicate", "Include in Analysis")
    criteria_result = evaluate_accuracy_criteria(
        result.overall_summary,
        criteria["Maximum Absolute Bias"],
        criteria["Maximum Absolute Percent Bias"],
        criteria["Minimum Recovery"],
        criteria["Maximum Recovery"],
        criteria["Borderline Zone"],
    )
    decision = _criteria_decision(criteria_result)
    metadata = _base_metadata(
        project_metadata,
        "Accuracy",
        "Evaluate agreement between observed results and expected target values.",
        "Replicate measurements of samples with assigned expected values.",
    )
    interpretation = generate_accuracy_interpretation(result.overall_summary, result.worst_case_summary, criteria, decision, metadata)
    key_findings = pd.DataFrame(
        [
            {"Metric": "Maximum Absolute Percent Bias", "Result": f"{result.overall_summary['Maximum Absolute Percent Bias']:.2f}%"},
            {"Metric": "Recovery Range", "Result": f"{result.overall_summary['Minimum Recovery']:.2f}% to {result.overall_summary['Maximum Recovery']:.2f}%"},
        ]
    )
    return StudyReport(
        study_type="Accuracy",
        study_name=str(metadata["Study Name"]),
        status="Completed",
        decision=decision,
        date=str(metadata["Study Date"]),
        objective=str(metadata["Study Objective"]),
        design=str(metadata["Study Design"]),
        metadata=metadata,
        acceptance_criteria=format_accuracy_criteria_table(build_criteria_table(criteria_result)),
        key_findings=key_findings,
        interpretation=interpretation,
        visualizations={
            "Expected vs Observed Plot": _figure_html(create_accuracy_expected_observed_plot(result.accuracy_summary), True),
        },
        raw_outputs={
            "Accuracy Summary": format_accuracy_table(result.accuracy_summary),
            "Bias Summary": format_accuracy_table(result.bias_summary),
            "Recovery Summary": format_accuracy_table(result.recovery_summary),
        },
        **_traceability(
            data,
            result.analyzed_data,
            "accuracy_study_hba1c.csv",
            "Accuracy analysis using observed-versus-expected bias, absolute bias, percent bias, percent recovery, and level-specific decision review.",
            decision,
        ),
    )


def build_stability_report(root_dir: Path, project_metadata: dict[str, object]) -> StudyReport:
    data = pd.read_csv(_sample_path(root_dir, "stability_study_hba1c.csv"))
    criteria = {"Maximum Percent Change": 10.0, "Minimum Recovery": 90.0, "Maximum Absolute Bias": 0.50, "Borderline Zone": 2.0}
    result = run_stability_study(data, "Sample ID", "Timepoint", "Result", "Storage Condition", "Units", "Replicate", "Include in Analysis")
    criteria_result = evaluate_stability_criteria(
        result.overall_summary,
        criteria["Maximum Percent Change"],
        criteria["Minimum Recovery"],
        criteria["Maximum Absolute Bias"],
        criteria["Borderline Zone"],
    )
    decision = _criteria_decision(criteria_result)
    metadata = _base_metadata(
        project_metadata,
        "Stability",
        "Evaluate assay stability across predefined storage conditions and timepoints.",
        "Repeated measurements collected at baseline and subsequent timepoints.",
    )
    interpretation = generate_stability_interpretation(result.overall_summary, result.timepoint_summary, criteria, decision, metadata)
    key_findings = pd.DataFrame(
        [
            {"Metric": "Maximum Percent Change", "Result": f"{result.overall_summary['Maximum Observed Change']:.2f}%"},
            {"Metric": "Minimum Recovery", "Result": f"{result.overall_summary['Minimum Recovery']:.2f}%"},
        ]
    )
    return StudyReport(
        study_type="Stability",
        study_name=str(metadata["Study Name"]),
        status="Completed",
        decision=decision,
        date=str(metadata["Study Date"]),
        objective=str(metadata["Study Objective"]),
        design=str(metadata["Study Design"]),
        metadata=metadata,
        acceptance_criteria=format_stability_criteria_table(build_criteria_table(criteria_result)),
        key_findings=key_findings,
        interpretation=interpretation,
        visualizations={
            "Stability Trend Plot": _figure_html(create_stability_trend_plot(result.timepoint_summary, result.overall_summary["Baseline Mean"]), True),
            "Percent Change Plot": _figure_html(create_stability_percent_change_plot(result.timepoint_summary, criteria["Maximum Percent Change"], criteria["Borderline Zone"])),
        },
        raw_outputs={
            "Overall Stability Summary": format_stability_overall_summary(result.overall_summary),
            "Timepoint Summary": format_stability_table(result.timepoint_summary),
        },
        **_traceability(
            data,
            result.analyzed_data,
            "stability_study_hba1c.csv",
            "Stability analysis using baseline-relative percent change, recovery, bias, timepoint trend summaries, and storage-condition review.",
            decision,
        ),
    )


def build_detection_report(root_dir: Path, project_metadata: dict[str, object]) -> StudyReport:
    data = pd.read_csv(_sample_path(root_dir, "detection_capability_hba1c.csv"))
    criteria = {"Maximum LoB": 0.15, "Maximum LoD": 0.30, "Target LoQ CV%": 20.0, "Maximum LoQ Concentration": 1.0, "Borderline Zone": 2.0}
    result = run_detection_capability_study(data, "Sample ID", "Sample Type", "Concentration Level", "Observed Result", "Replicate", "Units", "Include in Analysis", criteria["Target LoQ CV%"])
    criteria_result = evaluate_detection_capability_criteria(
        result.overall_summary,
        criteria["Maximum LoB"],
        criteria["Maximum LoD"],
        criteria["Target LoQ CV%"],
        criteria["Maximum LoQ Concentration"],
        criteria["Borderline Zone"],
    )
    decision = _criteria_decision(criteria_result)
    metadata = _base_metadata(
        project_metadata,
        "Detection Capability",
        "Evaluate assay detection capability through determination of LoB, LoD, and LoQ.",
        "Replicate blank, low-concentration, and quantitation-level samples analyzed to estimate detection capability.",
    )
    interpretation = generate_detection_interpretation(result.overall_summary, result.data_quality_summary, criteria, decision, metadata)
    key_findings = pd.DataFrame(
        [
            {"Metric": "LoB", "Result": f"{result.overall_summary['LoB']:.3f}"},
            {"Metric": "LoD", "Result": f"{result.overall_summary['LoD']:.3f}"},
            {"Metric": "Operational LoQ", "Result": f"{result.overall_summary['LoQ']:.3f}"},
            {"Metric": "LoQ CV%", "Result": f"{result.overall_summary['Operational LoQ CV%']:.2f}%"},
        ]
    )
    return StudyReport(
        study_type="Detection Capability",
        study_name=str(metadata["Study Name"]),
        status="Completed",
        decision=decision,
        date=str(metadata["Study Date"]),
        objective=str(metadata["Study Objective"]),
        design=str(metadata["Study Design"]),
        metadata=metadata,
        acceptance_criteria=format_detection_criteria_table(build_criteria_table(criteria_result)),
        key_findings=key_findings,
        interpretation=interpretation,
        visualizations={
            "Blank Distribution Histogram": _figure_html(create_blank_distribution_histogram(result.analyzed_data), True),
            "LoQ Precision Curve": _figure_html(create_loq_precision_curve(result.loq_summary, criteria["Target LoQ CV%"], result.overall_summary["LoQ"])),
        },
        raw_outputs={
            "Detection Summary": format_detection_overall_summary(result.overall_summary),
            "LoB Summary": format_detection_table(result.lob_summary),
            "LoD Summary": format_detection_table(result.lod_summary),
            "LoQ Summary": format_detection_table(result.loq_summary),
            "Decision Matrix": format_detection_table(build_detection_decision_matrix(criteria_result)),
        },
        **_traceability(
            data,
            result.analyzed_data,
            "detection_capability_hba1c.csv",
            "Detection capability analysis using LoB = mean blank + 1.645 x SD blank, LoD = LoB + 1.645 x SD low sample, and LoQ precision/recovery review.",
            decision,
        ),
    )


def build_dbs_report(root_dir: Path, project_metadata: dict[str, object]) -> StudyReport:
    """Build normalized DBS validation payload for consolidated reports."""

    data = pd.read_csv(_sample_path(root_dir, "dbs_validation_hba1c.csv"))
    criteria = {
        "Maximum Percent Bias": 10.0,
        "Minimum Recovery": 90.0,
        "Maximum Recovery": 110.0,
        "Minimum R²": 0.95,
        "Maximum Mean Difference": 0.50,
        "Borderline Zone": 2.0,
    }
    result = run_dbs_validation_study(
        data,
        "Sample ID",
        "Reference Result",
        "DBS Result",
        "Include in Analysis",
        criteria["Maximum Percent Bias"],
        criteria["Minimum Recovery"],
        criteria["Maximum Recovery"],
    )
    criteria_result = evaluate_dbs_criteria(
        result.overall_summary,
        criteria["Maximum Percent Bias"],
        criteria["Minimum Recovery"],
        criteria["Maximum Recovery"],
        criteria["Minimum R²"],
        criteria["Maximum Mean Difference"],
        criteria["Borderline Zone"],
    )
    decision = _criteria_decision(criteria_result)
    metadata = _base_metadata(
        project_metadata,
        "DBS Validation",
        "Evaluate whether DBS-derived results demonstrate acceptable analytical agreement with reference venous specimens.",
        "Paired DBS and venous whole blood specimens analyzed to evaluate recovery, bias, correlation, and agreement.",
    )
    metadata.update(
        {
            "Specimen Comparison": "DBS vs Venous Whole Blood",
            "DBS Punch Size": "6 mm",
            "Specimen Matrix": "DBS vs Venous Whole Blood",
        }
    )
    interpretation = generate_dbs_interpretation(
        result.overall_summary, criteria, decision, metadata
    )
    scientific_findings = (
        ["All predefined acceptance criteria were met."]
        if decision == "PASS"
        else ["One or more preliminary acceptance criteria require reviewer attention."]
        if decision in {"PASS WITH CAUTION", "BORDERLINE", "REVIEW"}
        else ["One or more preliminary acceptance criteria were not met."]
    ) + result.scientific_findings
    key_findings = pd.DataFrame(
        [
            {"Metric": "R²", "Result": f"{result.overall_summary['R²']:.4f}"},
            {"Metric": "Mean Recovery", "Result": f"{result.overall_summary['Mean Recovery']:.2f}%"},
            {"Metric": "Maximum Absolute Percent Bias", "Result": f"{result.overall_summary['Maximum Absolute Percent Bias']:.2f}%"},
            {"Metric": "Mean Difference", "Result": f"{result.overall_summary['Mean Difference']:.3f}"},
        ]
    )
    return StudyReport(
        study_type="DBS Validation",
        study_name=str(metadata["Study Name"]),
        status="Completed",
        decision=decision,
        date=str(metadata["Study Date"]),
        objective=str(metadata["Study Objective"]),
        design=str(metadata["Study Design"]),
        metadata=metadata,
        acceptance_criteria=format_dbs_criteria_table(build_criteria_table(criteria_result)),
        key_findings=key_findings,
        interpretation=interpretation,
        visualizations={
            "DBS vs Reference Scatter Plot": _figure_html(
                create_dbs_scatter_plot(result.analyzed_data, result.overall_summary),
                True,
            ),
            "Bland-Altman Plot": _figure_html(
                create_dbs_bland_altman_plot(result.analyzed_data, result.overall_summary)
            ),
        },
        raw_outputs={
            "DBS Overall Summary": format_dbs_overall_summary(result.overall_summary),
            "Bias Summary": format_dbs_table(result.bias_summary),
            "Recovery Summary": format_dbs_table(result.recovery_summary),
            "Correlation Summary": format_dbs_table(result.correlation_summary),
            "Agreement Summary": format_dbs_table(result.agreement_summary),
            "Hematocrit Assessment": format_dbs_table(result.hematocrit_summary),
            "Extraction Delay Assessment": format_dbs_table(result.delay_summary),
            "Instrument Assessment": format_dbs_table(result.instrument_summary),
            "Outlier Investigation": format_dbs_table(result.outlier_review),
            "Scientific Review Findings": pd.DataFrame({"Finding": scientific_findings}),
        },
        **_traceability(
            data,
            result.analyzed_data,
            "dbs_validation_hba1c.csv",
            "DBS specimen equivalency analysis using paired bias, percent recovery, Pearson correlation, ordinary least-squares regression, and Bland-Altman agreement statistics.",
            decision,
        ),
    )


STUDY_BUILDERS = {
    "Method Comparison": build_method_comparison_report,
    "Precision": build_precision_report,
    "Linearity": build_linearity_report,
    "Stability": build_stability_report,
    "Accuracy": build_accuracy_report,
    "Detection Capability": build_detection_report,
    "DBS Validation": build_dbs_report,
}


def collect_completed_studies(
    selected_studies: list[str],
    root_dir: Path,
    project_metadata: dict[str, object],
) -> list[StudyReport]:
    """Collect normalized study reports for selected completed modules."""

    reports: list[StudyReport] = []
    for study_name in selected_studies:
        builder = STUDY_BUILDERS.get(study_name)
        if builder is None:
            continue
        reports.append(builder(root_dir, project_metadata))
    return reports


def generate_validation_package(
    selected_studies: list[str],
    root_dir: Path,
    project_metadata: dict[str, object],
) -> ValidationPackage:
    """Generate complete consolidated validation package."""

    studies = collect_completed_studies(selected_studies, root_dir, project_metadata)
    if not studies:
        raise ValueError("Select at least one completed study to generate a report package.")
    validation_matrix = create_validation_matrix(studies)
    overall_status = determine_overall_status(studies)
    coverage = coverage_table(studies, SUPPORTED_STUDIES)
    completeness = completeness_percent(studies, SUPPORTED_STUDIES)
    risk_summary = create_risk_summary(studies, SUPPORTED_STUDIES)
    qa_findings = create_qa_findings(studies)
    decision_justification = generate_decision_justification(studies, overall_status)
    final_conclusion = generate_final_conclusion(project_metadata, studies, overall_status)
    package = ValidationPackage(
        project_metadata=project_metadata,
        studies=studies,
        validation_matrix=validation_matrix,
        overall_status=overall_status,
        final_conclusion=final_conclusion,
        coverage_matrix=coverage,
        completeness_percent=completeness,
        risk_summary=risk_summary,
        qa_findings=qa_findings,
        decision_justification=decision_justification,
    )
    package.executive_narrative = generate_executive_narrative(package)
    return package
