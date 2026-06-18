"""Streamlit application for the Scientific Validation Analytics Platform."""

from __future__ import annotations

from datetime import date, datetime
from html import escape
from io import StringIO
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
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
    assess_accuracy_data_quality,
    calculate_percent_samples_meeting_agreement,
    evaluate_acceptance_criteria,
    evaluate_accuracy_criteria,
    evaluate_linearity_criteria,
    evaluate_precision_criteria,
    evaluate_stability_criteria,
    get_top_outliers,
    run_accuracy_study,
    evaluate_dbs_criteria,
    evaluate_microtainer_criteria,
    evaluate_detection_capability_criteria,
    build_detection_decision_matrix,
    run_dbs_validation_study,
    run_microtainer_validation_study,
    run_detection_capability_study,
    run_linearity_study,
    run_precision_study,
    run_method_comparison,
    run_stability_study,
)
from plots import (
    create_accuracy_expected_observed_plot,
    create_accuracy_percent_bias_plot,
    create_accuracy_performance_heatmap,
    create_accuracy_recovery_plot,
    create_accuracy_replicate_distribution_plot,
    create_individual_accuracy_bias_plot,
    create_blank_distribution_histogram,
    create_blank_replicate_boxplot,
    create_condition_difference_plot,
    create_dbs_bland_altman_plot,
    create_dbs_delay_bias_plot,
    create_dbs_delay_category_bias_plot,
    create_dbs_delay_distribution_plot,
    create_dbs_distribution_comparison,
    create_dbs_hematocrit_bias_plot,
    create_dbs_hematocrit_percent_bias_plot,
    create_dbs_instrument_bias_plot,
    create_dbs_instrument_recovery_plot,
    create_dbs_percent_bias_plot,
    create_dbs_recovery_plot,
    create_dbs_scatter_plot,
    create_microtainer_bland_altman_plot,
    create_microtainer_distribution_comparison,
    create_microtainer_percent_bias_plot,
    create_microtainer_recovery_plot,
    create_microtainer_scatter_plot,
    create_detection_replicate_distribution_plot,
    create_detection_replicate_scatter_plot,
    create_detection_capability_ladder,
    create_detection_density_plot,
    create_difference_plot,
    create_individual_stability_plot,
    create_lob_lod_visualization,
    create_low_level_distribution_plot,
    create_loq_cv_plot,
    create_loq_decision_plot,
    create_loq_precision_curve,
    create_loq_recovery_plot,
    create_linearity_plot,
    create_linearity_residual_plot,
    create_percent_bias_histogram,
    create_percent_recovery_plot,
    create_precision_box_plot,
    create_precision_cv_bar_chart,
    create_precision_run_chart,
    create_scatter_plot,
    create_stability_percent_change_plot,
    create_stability_recovery_plot,
    create_stability_trend_plot,
    create_stability_bias_plot,
)
from report import (
    build_accuracy_executive_summary,
    build_accuracy_html_report,
    build_criteria_table,
    build_detection_executive_summary,
    build_detection_html_report,
    build_detection_pdf_report,
    build_dbs_executive_summary,
    build_dbs_html_report,
    build_dbs_pdf_report,
    build_microtainer_executive_summary,
    build_microtainer_html_report,
    build_microtainer_pdf_report,
    build_html_report,
    build_linearity_executive_summary,
    build_linearity_html_report,
    build_precision_html_report,
    build_stability_executive_summary,
    build_stability_html_report,
    build_stability_pdf_report,
    build_summary_table,
    criteria_table_to_badged_html,
    accuracy_criteria_dashboard_html,
    format_accuracy_criteria_table,
    format_accuracy_level_decision_table,
    format_accuracy_overall_summary,
    format_accuracy_table,
    format_accuracy_worst_case_summary,
    format_detection_criteria_table,
    format_detection_overall_summary,
    format_detection_table,
    format_dbs_criteria_table,
    format_dbs_overall_summary,
    format_dbs_table,
    format_microtainer_criteria_table,
    format_microtainer_overall_summary,
    format_microtainer_table,
    format_linearity_criteria_table,
    format_linearity_equation,
    format_linearity_regression_summary,
    format_linearity_summary_table,
    format_precision_criteria_table,
    format_precision_summary_table,
    format_stability_criteria_table,
    format_stability_overall_summary,
    format_stability_outlier_table,
    format_storage_condition_comparison_table,
    format_stability_table,
    generate_interpretation,
    generate_accuracy_interpretation,
    generate_detection_interpretation,
    generate_dbs_interpretation,
    generate_microtainer_interpretation,
    generate_linearity_interpretation,
    generate_precision_interpretation,
    generate_stability_risk_assessment,
    generate_storage_condition_comparison_interpretation,
    generate_stability_interpretation,
    get_linearity_worst_case,
    status_badge_html,
)
from studies import get_study_type_config, get_study_type_names
from reports import SUPPORTED_STUDIES, generate_validation_package
from reports.executive_summary import study_counts
from reports.html_export import export_full_html
from reports.pdf_export import build_executive_summary_pdf, build_full_pdf
from reports.report_builder import build_study_section_html


APP_TITLE = "Scientific Validation Analytics Platform"
APP_VERSION = "v1.0.0"
APP_STATUS = "Production Release"
RELEASE_NAME = "Scientific Validation Analytics Platform"
ROOT_DIR = PROJECT_ROOT
SAMPLE_DATA_PATH = ROOT_DIR / "data" / "sample_data" / "hba1c_method_comparison.csv"
PRECISION_SAMPLE_DATA_PATH = ROOT_DIR / "data" / "sample_data" / "precision_study_hba1c.csv"
LINEARITY_SAMPLE_DATA_PATH = ROOT_DIR / "data" / "sample_data" / "linearity_study_hba1c.csv"
STABILITY_SAMPLE_DATA_PATH = ROOT_DIR / "data" / "sample_data" / "stability_study_hba1c.csv"
ACCURACY_SAMPLE_DATA_PATH = ROOT_DIR / "data" / "sample_data" / "accuracy_study_hba1c.csv"
DETECTION_SAMPLE_DATA_PATH = ROOT_DIR / "data" / "sample_data" / "detection_capability_hba1c.csv"
DBS_SAMPLE_DATA_PATH = ROOT_DIR / "data" / "sample_data" / "dbs_validation_hba1c.csv"
MICROTAINER_SAMPLE_DATA_PATH = ROOT_DIR / "data" / "sample_data" / "microtainer_validation_hba1c.csv"
DASHBOARD_MODULES = (
    ("Method Comparison", "Paired reference and candidate comparison studies."),
    ("Accuracy Studies", "Bias, recovery, and agreement with expected values."),
    ("Precision Studies", "Intra-assay and inter-assay precision workflows."),
    ("Linearity Studies", "Analytical measurement range and recovery by level."),
    ("Stability Studies", "Specimen, reagent, and timepoint stability review."),
    ("Detection Capability", "Limit of Blank, Limit of Detection, and Limit of Quantitation validation workflows."),
    ("DBS Validation", "Dried blood spot correction and agreement workflows."),
    ("Microtainer Validation", "Small-volume specimen comparison workflows."),
    ("Validation Reports", "Scientific summary reports and review packages."),
)
CORE_VALIDATION_MODULES = (
    "Method Comparison",
    "Precision",
    "Accuracy",
    "Linearity",
    "Stability",
    "Detection Capability",
    "DBS Validation",
    "Microtainer Validation",
)
STUDY_LIFECYCLE_STATES = (
    "Draft",
    "Submitted for Review",
    "Under Review",
    "Approved",
    "Rejected",
    "Locked",
    "Archived",
)
REPORT_ELIGIBLE_STATES = {"Approved", "Locked"}
PLATFORM_CAPABILITIES = CORE_VALIDATION_MODULES + ("Validation Reports",)
SAMPLE_DATASETS = (
    {
        "Study Type": "Method Comparison",
        "Dataset": "HbA1c Method Comparison",
        "File": SAMPLE_DATA_PATH,
        "Description": "Paired reference and candidate HbA1c results.",
    },
    {
        "Study Type": "Precision",
        "Dataset": "HbA1c Precision Study",
        "File": PRECISION_SAMPLE_DATA_PATH,
        "Description": "Low and high QC repeated measurements across days, runs, and replicates.",
    },
    {
        "Study Type": "Accuracy",
        "Dataset": "HbA1c Accuracy Study",
        "File": ACCURACY_SAMPLE_DATA_PATH,
        "Description": "Observed HbA1c values compared with expected target levels.",
    },
    {
        "Study Type": "Linearity",
        "Dataset": "HbA1c Linearity Study",
        "File": LINEARITY_SAMPLE_DATA_PATH,
        "Description": "Expected and observed HbA1c values across the analytical range.",
    },
    {
        "Study Type": "Stability",
        "Dataset": "HbA1c Stability Study",
        "File": STABILITY_SAMPLE_DATA_PATH,
        "Description": "Baseline and timepoint results across storage conditions.",
    },
    {
        "Study Type": "Detection Capability",
        "Dataset": "HbA1c Detection Capability",
        "File": DETECTION_SAMPLE_DATA_PATH,
        "Description": "Blank, low-concentration, and quantitation-level replicates.",
    },
    {
        "Study Type": "DBS Validation",
        "Dataset": "HbA1c DBS Validation",
        "File": DBS_SAMPLE_DATA_PATH,
        "Description": "DBS results paired with reference venous whole blood values.",
    },
    {
        "Study Type": "Microtainer Validation",
        "Dataset": "HbA1c Microtainer Validation",
        "File": MICROTAINER_SAMPLE_DATA_PATH,
        "Description": "Microtainer results paired with reference venous specimen values.",
    },
)
DEFAULT_PROJECTS = [
    {
        "Project Name": "HbA1c Validation Program",
        "Program Name": "HbA1c Validation Program",
        "Assay": "HbA1c",
        "Assay / Biomarker": "HbA1c",
        "Program Owner": "Validation Team",
        "Study Status": "Validation Complete",
        "Status": "Completed",
        "Start Date": "2026-06-18",
        "Target Completion Date": "2026-06-18",
        "Reviewer": "",
        "Notes": "Demonstration validation program containing all core validation study types.",
        "Required Studies": list(CORE_VALIDATION_MODULES),
        "Completed Studies": list(CORE_VALIDATION_MODULES),
        "Last Updated": "2026-06-18",
        "Overall Status": "Completed",
        "Final Package Generated": True,
    },
    {
        "Project Name": "Ferritin Validation Program",
        "Program Name": "Ferritin Validation Program",
        "Assay": "Ferritin",
        "Assay / Biomarker": "Ferritin",
        "Program Owner": "Validation Team",
        "Study Status": "In Progress",
        "Status": "In Progress",
        "Start Date": "2026-06-18",
        "Target Completion Date": "2026-07-18",
        "Reviewer": "",
        "Notes": "",
        "Required Studies": list(CORE_VALIDATION_MODULES),
        "Completed Studies": ["Method Comparison", "Precision", "Accuracy"],
        "Last Updated": "2026-06-18",
        "Overall Status": "In Progress",
        "Final Package Generated": False,
    },
    {
        "Project Name": "Vitamin D Validation Program",
        "Program Name": "Vitamin D Validation Program",
        "Assay": "Vitamin D",
        "Assay / Biomarker": "Vitamin D",
        "Program Owner": "Validation Team",
        "Study Status": "Planned",
        "Status": "Not Started",
        "Start Date": "2026-06-18",
        "Target Completion Date": "2026-08-18",
        "Reviewer": "",
        "Notes": "",
        "Required Studies": list(CORE_VALIDATION_MODULES),
        "Completed Studies": [],
        "Last Updated": "2026-06-18",
        "Overall Status": "Not Started",
        "Final Package Generated": False,
    },
]
DEFAULT_PLATFORM_SETTINGS = {
    "Laboratory Name": "",
    "Department": "",
    "Address": "",
    "Analyst Name": "",
    "Reviewer Name": "",
    "Report Logo": "",
    "Report Footer": "Generated by the Scientific Validation Analytics Platform.",
    "Organization Branding": "",
    "PDF Settings": "Standard scientific report layout",
    "Default Report Format": "PDF and HTML",
}


def inject_validation_styles() -> None:
    """Add shared styling for validation result cards and status badges."""

    st.markdown(
        """
        <style>
          .svap-card {
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            background: #ffffff;
            padding: 14px 16px;
            min-height: 104px;
            box-shadow: 0 1px 2px rgba(16, 42, 67, 0.04);
          }
          .svap-card-label {
            color: #52606d;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 8px;
            text-transform: uppercase;
          }
          .svap-card-value {
            color: #102a43;
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.2;
            overflow-wrap: anywhere;
          }
          .svap-card-subtext {
            color: #52606d;
            font-size: 0.9rem;
            margin-top: 6px;
          }
          .svap-equation {
            border-left: 4px solid #2a6f97;
            background: #f7f9fb;
            border-radius: 8px;
            padding: 16px 18px;
            margin: 8px 0 12px;
          }
          .svap-equation-title {
            color: #102a43;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 8px;
          }
          .svap-equation-body {
            color: #102a43;
            font-size: 1.25rem;
            font-weight: 700;
          }
          .svap-status-table table {
            border-collapse: collapse;
            width: 100%;
          }
          .svap-status-table th, .svap-status-table td {
            border: 1px solid #d9e2ec;
            padding: 8px 10px;
          }
          .svap-status-table th {
            background: #f0f4f8;
            color: #102a43;
          }
          .svap-status-table td {
            color: #1f2933;
          }
          .svap-status-table td:not(:first-child), .svap-status-table th:not(:first-child) {
            text-align: right;
          }
          .svap-status-table td:nth-child(4), .svap-status-table th:nth-child(4),
          .svap-status-table td:nth-child(5), .svap-status-table th:nth-child(5) {
            text-align: center;
          }
          .status-badge {
            border-radius: 999px;
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 700;
            line-height: 1;
            padding: 6px 10px;
          }
          .status-pass {
            background: #e3f9e5;
            color: #1f7a1f;
          }
          .status-borderline {
            background: #fff8c5;
            color: #946200;
          }
          .status-fail {
            background: #ffe3e3;
            color: #c92a2a;
          }
          .svap-page-header {
            border-bottom: 1px solid #d9e2ec;
            margin: 4px 0 20px;
            padding-bottom: 14px;
          }
          .svap-page-kicker {
            color: #52606d;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
          }
          .svap-page-title {
            color: #102a43;
            font-size: 1.65rem;
            font-weight: 750;
            line-height: 1.2;
            margin-top: 4px;
          }
          .svap-page-subtitle {
            color: #52606d;
            font-size: 0.98rem;
            margin-top: 6px;
          }
          .svap-card-button {
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            background: #ffffff;
            min-height: 132px;
            padding: 14px 16px;
          }
          .svap-check {
            color: #1f7a1f;
            font-weight: 800;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_platform_state() -> None:
    """Initialize local v1.0 platform state."""

    if "platform_settings" not in st.session_state:
        st.session_state.platform_settings = DEFAULT_PLATFORM_SETTINGS.copy()
    if "validation_projects" not in st.session_state:
        st.session_state.validation_projects = [project.copy() for project in DEFAULT_PROJECTS]
    st.session_state.validation_projects = [
        normalize_program(project) for project in st.session_state.validation_projects
    ]
    if "reports_library" not in st.session_state:
        st.session_state.reports_library = [
            {
                "Report Name": "HbA1c Full Validation Package",
                "Project": "HbA1c Validation Program",
                "Study Type": "Validation Reports",
                "Date": "2026-06-18",
                "Analyst": st.session_state.platform_settings.get("Analyst Name", ""),
                "Format": "HTML / PDF",
            },
            {
                "Report Name": "HbA1c Microtainer Validation Report",
                "Project": "HbA1c Validation Program",
                "Study Type": "Microtainer Validation",
                "Date": "2026-06-18",
                "Analyst": st.session_state.platform_settings.get("Analyst Name", ""),
                "Format": "HTML / PDF / CSV",
            },
        ]
    if "study_lifecycle_records" not in st.session_state:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.study_lifecycle_records = [
            {
                "Study ID": f"hba1c-{study.lower().replace(' ', '-').replace('/', '-')}",
                "Project": "HbA1c Validation Program",
                "Study Name": f"HbA1c {study}",
                "Study Type": study,
                "Assay": "HbA1c",
                "Analyst": st.session_state.platform_settings.get("Analyst Name", ""),
                "Completion Date": "2026-06-18",
                "Status": "Locked",
                "Version": 1,
                "Reviewer Comments": "",
                "Last Updated": timestamp,
                "Locked": True,
            }
            for study in CORE_VALIDATION_MODULES
        ]
    if "approval_history" not in st.session_state:
        st.session_state.approval_history = []
    if "program_timeline" not in st.session_state:
        st.session_state.program_timeline = [
            {
                "Timestamp": "2026-06-18 00:00",
                "Program": "HbA1c Validation Program",
                "Study": "Validation Package",
                "Activity": "Package generated",
                "Details": "Initial completed demonstration validation package.",
            }
        ]


def platform_settings() -> dict[str, str]:
    """Return current platform settings."""

    initialize_platform_state()
    return st.session_state.platform_settings


def render_page_header(title: str, subtitle: str = "", kicker: str = APP_STATUS) -> None:
    """Render a standardized platform page header."""

    st.markdown(
        f"""
        <div class="svap-page-header">
          <div class="svap-page-kicker">{escape(kicker)} · {escape(APP_VERSION)}</div>
          <div class="svap-page-title">{escape(title)}</div>
          <div class="svap-page-subtitle">{escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, status: str | None = None) -> None:
    """Render a compact validation metric card."""

    value_html = status_badge_html(value) if status else escape(value)
    st.markdown(
        f"""
        <div class="svap-card">
          <div class="svap-card-label">{escape(label)}</div>
          <div class="svap-card-value">{value_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_module_card(title: str, description: str, status: str | None = None) -> None:
    """Render a reusable platform navigation card."""

    status_html = f'<div class="svap-card-subtext">{status_badge_html(status)}</div>' if status else ""
    st.markdown(
        f"""
        <div class="svap-card-button">
          <div class="svap-card-label">{escape(title)}</div>
          <div class="svap-card-subtext">{escape(description)}</div>
          {status_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def project_progress(project: dict[str, object]) -> tuple[int, int, float]:
    """Return completed count, total count, and percent complete for a project."""

    required = list(project.get("Required Studies") or CORE_VALIDATION_MODULES)
    completed = len(project.get("Completed Studies", []))
    total = len(required)
    percent = (completed / total) * 100 if total else 0.0
    return completed, total, percent


def normalize_program(program: dict[str, object]) -> dict[str, object]:
    """Backfill v1.1 validation program fields for existing session records."""

    program_name = str(program.get("Program Name") or program.get("Project Name") or "Validation Program")
    assay = str(program.get("Assay / Biomarker") or program.get("Assay") or "")
    program.setdefault("Program Name", program_name)
    program.setdefault("Project Name", program_name)
    program.setdefault("Assay / Biomarker", assay)
    program.setdefault("Assay", assay)
    program.setdefault("Program Owner", "")
    program.setdefault("Status", program.get("Study Status", "Not Started"))
    program.setdefault("Start Date", str(date.today()))
    program.setdefault("Target Completion Date", str(date.today()))
    program.setdefault("Reviewer", "")
    program.setdefault("Notes", "")
    program.setdefault("Required Studies", list(CORE_VALIDATION_MODULES))
    program.setdefault("Completed Studies", [])
    program.setdefault("Final Package Generated", False)
    program.setdefault("Last Updated", str(date.today()))
    return program


def program_required_studies(program: dict[str, object]) -> list[str]:
    """Return assigned studies for a validation program."""

    return list(normalize_program(program).get("Required Studies") or CORE_VALIDATION_MODULES)


def normalize_lifecycle_status(status: object) -> str:
    """Normalize legacy and current lifecycle status labels."""

    value = str(status or "Draft")
    return {
        "Analyzed": "Draft",
        "Pending Review": "Submitted for Review",
        "Review Pending": "Submitted for Review",
        "Returned for Revision": "Draft",
    }.get(value, value if value in STUDY_LIFECYCLE_STATES else "Draft")


def program_lifecycle_records(program_name: str) -> list[dict[str, object]]:
    """Return lifecycle records assigned to a validation program."""

    records = []
    for record in get_lifecycle_records():
        record["Status"] = normalize_lifecycle_status(record.get("Status"))
        if str(record.get("Project")) == program_name:
            records.append(record)
    return records


def program_study_status(program_name: str, study_type: str) -> str:
    """Return the latest lifecycle status for a program study."""

    matching = [
        record for record in program_lifecycle_records(program_name)
        if record.get("Study Type") == study_type
    ]
    return str(matching[-1]["Status"]) if matching else "Draft"


def program_metrics(program: dict[str, object]) -> dict[str, object]:
    """Calculate program-level validation management metrics."""

    program = normalize_program(program)
    program_name = str(program["Program Name"])
    required = program_required_studies(program)
    statuses = {study: program_study_status(program_name, study) for study in required}
    approved = sum(1 for status in statuses.values() if status in REPORT_ELIGIBLE_STATES)
    pending = sum(1 for status in statuses.values() if status == "Submitted for Review")
    under_review = sum(1 for status in statuses.values() if status == "Under Review")
    completed = sum(1 for status in statuses.values() if status in {"Submitted for Review", "Under Review", "Approved", "Locked", "Archived"})
    remaining = len(required) - completed
    percent = (approved / len(required)) * 100 if required else 0.0
    if bool(program.get("Final Package Generated")):
        readiness = "Completed"
    elif len(required) > 0 and approved == len(required):
        readiness = "Ready for Final Package"
    elif under_review > 0:
        readiness = "Ready for Approval"
    elif pending > 0:
        readiness = "Review Pending"
    elif completed > 0:
        readiness = "In Progress"
    else:
        readiness = "Not Started"
    return {
        "Required": len(required),
        "Completed": completed,
        "Submitted for Review": pending,
        "Under Review": under_review,
        "Approved": approved,
        "Remaining": remaining,
        "Completion %": percent,
        "Readiness": readiness,
        "Statuses": statuses,
    }


def lifecycle_badge_html(status: str) -> str:
    """Render a lifecycle status badge."""

    normalized = str(status)
    status_class = {
        "Draft": "status-borderline",
        "Submitted for Review": "status-borderline",
        "Under Review": "status-borderline",
        "Approved": "status-pass",
        "Locked": "status-pass",
        "Archived": "status-pass",
        "Rejected": "status-fail",
    }.get(normalized, "status-borderline")
    return f'<span class="status-badge {status_class}">{escape(normalized)}</span>'


def lifecycle_table_to_html(table: pd.DataFrame, status_column: str = "Lifecycle Status") -> str:
    """Render a table with lifecycle badges."""

    if table.empty:
        return "<p>No records available.</p>"
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in table.columns)
    body_rows = []
    for _, row in table.iterrows():
        cells = []
        for column in table.columns:
            value = row[column]
            if column == status_column:
                cells.append(f"<td>{lifecycle_badge_html(str(value))}</td>")
            else:
                cells.append(f"<td>{escape(str(value))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="svap-status-table"><table><thead><tr>{headers}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'


def get_lifecycle_records() -> list[dict[str, object]]:
    """Return lifecycle study records from session state."""

    initialize_platform_state()
    return st.session_state.study_lifecycle_records


def lifecycle_counts(project_name: str | None = None) -> dict[str, int]:
    """Return lifecycle status counts."""

    records = get_lifecycle_records()
    if project_name:
        records = [record for record in records if record.get("Project") == project_name]
    return {
        status: sum(1 for record in records if normalize_lifecycle_status(record.get("Status")) == status)
        for status in STUDY_LIFECYCLE_STATES
    }


def eligible_study_types(program_name: str | None = None) -> list[str]:
    """Return study types eligible for consolidated final reports."""

    records = get_lifecycle_records()
    if program_name is not None:
        records = [record for record in records if str(record.get("Project")) == program_name]
    return sorted(
        {
            str(record["Study Type"])
            for record in records
            if normalize_lifecycle_status(record.get("Status")) in REPORT_ELIGIBLE_STATES
        }
    )


def canonical_study_type(study_type: str) -> str:
    """Normalize UI study names to report-engine study names."""

    return {
        "Precision Study": "Precision",
        "Accuracy Study": "Accuracy",
        "Linearity Study": "Linearity",
        "Stability Study": "Stability",
    }.get(study_type, study_type)


def record_approval_history(
    study: dict[str, object],
    previous_status: str,
    new_status: str,
    decision: str,
    reviewer: str,
    comment: str,
) -> None:
    """Append a review event to the approval history."""

    st.session_state.approval_history.append(
        {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Study": study.get("Study Name", ""),
            "Reviewer": reviewer,
            "Previous Status": previous_status,
            "New Status": new_status,
            "Decision": decision,
            "Comment": comment,
        }
    )


def record_program_activity(program: str, study: str, activity: str, details: str = "") -> None:
    """Append a chronological program activity event."""

    initialize_platform_state()
    st.session_state.program_timeline.append(
        {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Program": program,
            "Study": study,
            "Activity": activity,
            "Details": details,
        }
    )


def submit_study_for_review(study_type: str, metadata: dict[str, object]) -> None:
    """Create or update a study lifecycle record as pending review."""

    initialize_platform_state()
    canonical_type = canonical_study_type(study_type)
    study_name = str(metadata.get("Study Name") or f"{metadata.get('Assay / Biomarker', 'Assay')} {study_type}")
    study_id = f"{study_name.lower().replace(' ', '-')}-{canonical_type.lower().replace(' ', '-')}"
    existing = next(
        (record for record in st.session_state.study_lifecycle_records if record.get("Study ID") == study_id),
        None,
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    if existing and existing.get("Status") == "Locked":
        version = int(existing.get("Version", 1)) + 1
        study_id = f"{study_id}-v{version}"
        existing = None
    elif existing:
        version = int(existing.get("Version", 1))
    else:
        version = 1

    record = {
        "Study ID": study_id,
        "Project": str(metadata.get("Validation Project Name") or "Unassigned Validation Project"),
        "Study Name": study_name,
        "Study Type": canonical_type,
        "Assay": str(metadata.get("Assay / Biomarker") or metadata.get("Assay") or ""),
        "Analyst": str(metadata.get("Analyst Name") or metadata.get("Analyst") or ""),
        "Completion Date": date.today().isoformat(),
        "Status": "Submitted for Review",
        "Version": version,
        "Reviewer Comments": str(metadata.get("Reviewer Comments") or ""),
        "Submitted Date": date.today().isoformat(),
        "Last Updated": timestamp,
        "Locked": False,
    }
    if existing:
        existing.update(record)
    else:
        st.session_state.study_lifecycle_records.append(record)
        record_program_activity(
            str(record["Project"]),
            str(record["Study Name"]),
            "Study created",
            f"{record['Study Type']} lifecycle record created.",
        )
    record_program_activity(
        str(record["Project"]),
        str(record["Study Name"]),
        "Submitted for review",
        f"{record['Study Type']} moved to Submitted for Review.",
    )


def record_study_analyzed(study_type: str, metadata: dict[str, object]) -> None:
    """Record a completed analysis before reviewer submission."""

    initialize_platform_state()
    canonical_type = canonical_study_type(study_type)
    study_name = str(metadata.get("Study Name") or f"{metadata.get('Assay / Biomarker', 'Assay')} {study_type}")
    study_id = f"{study_name.lower().replace(' ', '-')}-{canonical_type.lower().replace(' ', '-')}"
    existing = next(
        (record for record in st.session_state.study_lifecycle_records if record.get("Study ID") == study_id),
        None,
    )
    if existing and normalize_lifecycle_status(existing.get("Status")) in {"Submitted for Review", "Under Review", "Approved", "Locked", "Archived"}:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    record = {
        "Study ID": study_id,
        "Project": str(metadata.get("Validation Project Name") or "Unassigned Validation Project"),
        "Study Name": study_name,
        "Study Type": canonical_type,
        "Assay": str(metadata.get("Assay / Biomarker") or metadata.get("Assay") or ""),
        "Analyst": str(metadata.get("Analyst Name") or metadata.get("Analyst") or ""),
        "Completion Date": date.today().isoformat(),
        "Status": "Draft",
        "Version": int(existing.get("Version", 1)) if existing else 1,
        "Reviewer Comments": str(existing.get("Reviewer Comments", "")) if existing else "",
        "Submitted Date": str(existing.get("Submitted Date", "")) if existing else "",
        "Last Updated": timestamp,
        "Locked": False,
    }
    if existing:
        existing.update(record)
    else:
        st.session_state.study_lifecycle_records.append(record)
    record_program_activity(
        str(record["Project"]),
        str(record["Study Name"]),
        "Analysis completed",
        f"{record['Study Type']} analysis completed.",
    )


def render_submit_for_review(study_type: str, metadata: dict[str, object]) -> None:
    """Render the post-analysis submit-for-review action."""

    record_study_analyzed(study_type, metadata)
    st.subheader("Validation Lifecycle")
    st.caption("Completed analyses can be submitted to the Validation Review Center.")
    canonical_type = canonical_study_type(study_type)
    current_record = next(
        (
            record for record in get_lifecycle_records()
            if record.get("Study Name") == metadata.get("Study Name")
            and record.get("Study Type") == canonical_type
        ),
        None,
    )
    current_status = normalize_lifecycle_status(current_record.get("Status")) if current_record else "Draft"
    st.markdown(f"Current lifecycle status: {lifecycle_badge_html(current_status)}", unsafe_allow_html=True)
    if current_status in {"Approved", "Locked", "Archived"}:
        st.info("This study is read-only. Create a new study version to modify or resubmit it.")
        return
    if st.button("Submit for Review", key=f"submit_review_{study_type}_{metadata.get('Study Name', '')}"):
        submit_study_for_review(study_type, metadata)
        st.success("Study submitted for review. Status: Submitted for Review.")


def render_linearity_executive_summary(
    level_summary: pd.DataFrame,
    regression_summary: dict[str, float | str],
    decision: str,
) -> None:
    """Render executive summary cards for a completed linearity study."""

    summary_values = build_linearity_executive_summary(
        level_summary, regression_summary, decision
    )
    st.subheader("Executive Summary")
    first_row = st.columns(3)
    with first_row[0]:
        render_metric_card("Overall Decision", summary_values["Overall Decision"], status=decision)
    with first_row[1]:
        render_metric_card("R²", summary_values["R²"])
    with first_row[2]:
        render_metric_card("Regression Slope", summary_values["Regression Slope"])

    second_row = st.columns(3)
    with second_row[0]:
        render_metric_card("Maximum Absolute Bias", summary_values["Maximum Absolute Bias"])
    with second_row[1]:
        render_metric_card("Percent Recovery Range", summary_values["Percent Recovery Range"])
    with second_row[2]:
        render_metric_card("Analytical Range Tested", summary_values["Analytical Range Tested"])


def render_stability_executive_summary(
    overall_summary: dict[str, float | str],
    decision: str,
    criteria_table: pd.DataFrame | None = None,
) -> None:
    """Render executive summary cards for a completed stability study."""

    summary_values = build_stability_executive_summary(
        overall_summary, decision, criteria_table
    )
    st.subheader("Executive Summary")
    first_row = st.columns(4)
    with first_row[0]:
        render_metric_card("Overall Decision", summary_values["Overall Decision"], status=decision)
    with first_row[1]:
        render_metric_card("Maximum Observed Change", summary_values["Maximum Observed Change"])
    with first_row[2]:
        render_metric_card("Minimum Recovery", summary_values["Minimum Recovery"])
    with first_row[3]:
        render_metric_card("Maximum Absolute Bias", summary_values["Maximum Absolute Bias"])

    second_row = st.columns(4)
    with second_row[0]:
        render_metric_card("Worst Timepoint", summary_values["Worst Timepoint"])
    with second_row[1]:
        render_metric_card("Worst Storage Condition", summary_values["Worst Storage Condition"])
    with second_row[2]:
        render_metric_card("Borderline Criteria", summary_values["Borderline Criteria"])
    with second_row[3]:
        render_metric_card("Failed Criteria", summary_values["Failed Criteria"])


def render_accuracy_executive_summary(
    overall_summary: dict[str, float | str],
    worst_case_summary: dict[str, float | str],
    decision: str,
    criteria_table: pd.DataFrame,
    level_decision_table: pd.DataFrame,
) -> None:
    """Render executive summary cards for a completed accuracy study."""

    summary_values = build_accuracy_executive_summary(
        overall_summary, worst_case_summary, decision, criteria_table, level_decision_table
    )
    st.subheader("Executive Summary")
    first_row = st.columns(3)
    with first_row[0]:
        render_metric_card("Overall Decision", summary_values["Overall Decision"], status=decision)
    with first_row[1]:
        render_metric_card("Overall Mean Percent Bias", summary_values["Overall Mean Percent Bias"])
    with first_row[2]:
        render_metric_card("Maximum Absolute Percent Bias", summary_values["Maximum Absolute Percent Bias"])

    second_row = st.columns(3)
    with second_row[0]:
        render_metric_card("Lowest Recovery", summary_values["Lowest Recovery"])
    with second_row[1]:
        render_metric_card("Highest Recovery", summary_values["Highest Recovery"])
    with second_row[2]:
        render_metric_card("Worst Performing Level", summary_values["Worst Performing Level"])

    third_row = st.columns(3)
    with third_row[0]:
        render_metric_card("Levels Passing", summary_values["Levels Passing"])
    with third_row[1]:
        render_metric_card("Levels Failing", summary_values["Levels Failing"])
    with third_row[2]:
        render_metric_card("Failed Criteria", summary_values["Failed Criteria"])


def render_detection_executive_summary(
    overall_summary: dict[str, float | str],
    decision: str,
    criteria_table: pd.DataFrame,
) -> None:
    """Render executive summary cards for a completed detection capability study."""

    summary_values = build_detection_executive_summary(
        overall_summary, decision, criteria_table
    )
    st.subheader("Executive Summary")
    first_row = st.columns(4)
    with first_row[0]:
        render_metric_card("Overall Decision", summary_values["Overall Decision"], status=decision)
    with first_row[1]:
        render_metric_card("LoB", summary_values["LoB"])
    with first_row[2]:
        render_metric_card("LoD", summary_values["LoD"])
    with first_row[3]:
        render_metric_card("Operational LoQ", summary_values["Operational LoQ"])

    second_row = st.columns(4)
    with second_row[0]:
        render_metric_card("LoQ CV%", summary_values["LoQ CV%"])
    with second_row[1]:
        render_metric_card("Blank Replicates", summary_values["Blank Replicates"])
    with second_row[2]:
        render_metric_card("Low-Level Replicates", summary_values["Low-Level Replicates"])
    with second_row[3]:
        render_metric_card("Quantitation Levels Tested", summary_values["Quantitation Levels Tested"])


def render_linearity_equation_card(regression_summary: dict[str, float | str]) -> None:
    """Render the regression equation card for linearity results."""

    equation = format_linearity_equation(regression_summary)
    st.markdown(
        f"""
        <div class="svap-equation">
          <div class="svap-equation-title">Regression Equation</div>
          <div class="svap-equation-body">{escape(equation)}</div>
          <div class="svap-card-subtext">R² = {regression_summary['R²']:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_linearity_worst_case(level_summary: pd.DataFrame) -> None:
    """Render the worst-performing linearity level summary card."""

    worst_case = get_linearity_worst_case(level_summary)
    if worst_case.empty:
        return
    st.markdown(
        f"""
        <div class="svap-equation">
          <div class="svap-equation-title">Worst Performing Level</div>
          <div class="svap-equation-body">{escape(str(worst_case['Level']))}</div>
          <div class="svap-card-subtext">
            Absolute Bias: {abs(worst_case['Percent Bias']):.2f}% &nbsp;|&nbsp;
            Recovery: {worst_case['Percent Recovery']:.1f}% &nbsp;|&nbsp;
            Expected: {worst_case['Expected Result']:.2f} &nbsp;|&nbsp;
            Mean Observed: {worst_case['Mean Observed Result']:.2f}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_badged_criteria_table(criteria_table: pd.DataFrame) -> None:
    """Render an acceptance criteria table with status badges."""

    st.markdown(
        f'<div class="svap-status-table">{criteria_table_to_badged_html(criteria_table)}</div>',
        unsafe_allow_html=True,
    )


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Load a CSV or Excel file uploaded through Streamlit."""

    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


def get_numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return columns that contain at least one numeric value after coercion."""

    numeric_columns: list[str] = []
    for column in data.columns:
        converted = pd.to_numeric(data[column], errors="coerce")
        if converted.notna().any():
            numeric_columns.append(column)
    return numeric_columns


def detect_sample_id_column(data: pd.DataFrame) -> str | None:
    """Detect a likely sample identifier column."""

    possible_names = {"sample id", "sample_id", "sampleid", "id", "specimen id"}
    for column in data.columns:
        normalized = str(column).strip().lower()
        if normalized in possible_names:
            return column
    return None


def render_dashboard() -> None:
    """Render the v1.0 unified platform dashboard."""

    initialize_platform_state()
    render_page_header(
        "Unified Validation Dashboard",
        "A centralized workspace for analytical validation studies, specimen equivalency workflows, and validation-ready reporting.",
    )
    projects = st.session_state.validation_projects
    completed_studies = sum(len(project.get("Completed Studies", [])) for project in projects)
    reports_generated = len(st.session_state.reports_library)
    active_projects = sum(1 for project in projects if project.get("Study Status") != "Validation Complete")
    counts = lifecycle_counts()
    ready_for_review = counts["Submitted for Review"]
    ready_for_approval = counts["Under Review"]
    ready_for_package = sum(
        1 for project in projects
        if program_metrics(project)["Readiness"] == "Ready for Final Package"
    )

    overview = st.columns(5)
    with overview[0]:
        render_metric_card("Platform Version", APP_VERSION)
    with overview[1]:
        render_metric_card("Validation Modules", str(len(PLATFORM_CAPABILITIES)))
    with overview[2]:
        render_metric_card("Studies Completed", str(completed_studies))
    with overview[3]:
        render_metric_card("Reports Generated", str(reports_generated))
    with overview[4]:
        render_metric_card("Active Projects", str(active_projects))

    st.subheader("Lifecycle Metrics")
    lifecycle_cols = st.columns(4)
    with lifecycle_cols[0]:
        render_metric_card("Draft Studies", str(counts["Draft"]))
    with lifecycle_cols[1]:
        render_metric_card("Studies Under Review", str(counts["Under Review"]))
    with lifecycle_cols[2]:
        render_metric_card("Approved Studies", str(counts["Approved"]))
    with lifecycle_cols[3]:
        render_metric_card("Locked Studies", str(counts["Locked"]))

    st.subheader("Validation Readiness")
    readiness_cols = st.columns(3)
    with readiness_cols[0]:
        render_metric_card("Ready for Review", str(ready_for_review))
    with readiness_cols[1]:
        render_metric_card("Ready for Approval", str(ready_for_approval))
    with readiness_cols[2]:
        render_metric_card("Ready for Package Generation", str(ready_for_package))

    st.subheader("Validation Coverage")
    for row_start in range(0, len(PLATFORM_CAPABILITIES), 3):
        columns = st.columns(3)
        for column, capability in zip(columns, PLATFORM_CAPABILITIES[row_start : row_start + 3]):
            with column:
                st.markdown(
                    f"""
                    <div class="svap-card">
                      <div class="svap-card-value"><span class="svap-check">✓</span> {escape(capability)}</div>
                      <div class="svap-card-subtext">Available in {escape(APP_VERSION)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.subheader("Quick Actions")
    action_cols = st.columns(4)
    quick_actions = [
        ("Create Validation Project", "Projects"),
        ("Start New Study", "Validation Workspace"),
        ("Open Review Center", "Validation Review Center"),
        ("Generate New Report", "Reports Library"),
    ]
    for column, (label, target_page) in zip(action_cols, quick_actions):
        with column:
            if st.button(label, use_container_width=True):
                st.session_state.pending_page = target_page
                st.rerun()

    st.subheader("Core Validation Modules")
    for row_start in range(0, len(DASHBOARD_MODULES), 3):
        columns = st.columns(3)
        for column, (title, description) in zip(columns, DASHBOARD_MODULES[row_start : row_start + 3]):
            with column:
                render_module_card(title, description, "AVAILABLE")


def render_projects_workspace() -> None:
    """Render validation program management workspace."""

    initialize_platform_state()
    render_page_header(
        "Validation Programs",
        "Manage complete assay validation programs, assigned studies, review readiness, and final package status.",
    )
    settings = platform_settings()

    with st.expander("Create Validation Program", expanded=False):
        row = st.columns(4)
        with row[0]:
            program_name = st.text_input("Program Name", value="New Assay Validation Program")
        with row[1]:
            assay = st.text_input("Assay / Biomarker", value="Custom Assay")
        with row[2]:
            owner = st.text_input("Program Owner", value=settings.get("Analyst Name", ""))
        with row[3]:
            reviewer = st.text_input("Reviewer", value=settings.get("Reviewer Name", ""))
        date_row = st.columns(3)
        with date_row[0]:
            start_date = st.date_input("Start Date", value=date.today(), key="new_program_start")
        with date_row[1]:
            target_date = st.date_input("Target Completion Date", value=date.today(), key="new_program_target")
        with date_row[2]:
            status = st.selectbox("Status", ["Not Started", "In Progress", "Review Pending", "Ready for Final Package", "Completed"])
        required_studies = st.multiselect(
            "Assigned Studies",
            CORE_VALIDATION_MODULES,
            default=list(CORE_VALIDATION_MODULES),
        )
        notes = st.text_area("Notes", height=80, key="new_program_notes")
        if st.button("Add Validation Program", type="primary"):
            st.session_state.validation_projects.append(
                {
                    "Project Name": program_name,
                    "Program Name": program_name,
                    "Assay": assay,
                    "Assay / Biomarker": assay,
                    "Program Owner": owner,
                    "Status": status,
                    "Study Status": status,
                    "Start Date": start_date.isoformat(),
                    "Target Completion Date": target_date.isoformat(),
                    "Reviewer": reviewer,
                    "Notes": notes,
                    "Required Studies": required_studies,
                    "Completed Studies": [],
                    "Last Updated": date.today().isoformat(),
                    "Overall Status": status,
                    "Final Package Generated": False,
                }
            )
            record_program_activity(program_name, "Program", "Study created", "Validation program created.")
            st.success(f"Created validation program: {program_name}")

    program_rows = []
    for project in st.session_state.validation_projects:
        project = normalize_program(project)
        metrics = program_metrics(project)
        program_rows.append(
            {
                "Program Name": project["Program Name"],
                "Assay / Biomarker": project["Assay / Biomarker"],
                "Program Owner": project["Program Owner"],
                "Status": metrics["Readiness"],
                "Start Date": project["Start Date"],
                "Target Completion Date": project["Target Completion Date"],
                "Reviewer": project["Reviewer"],
                "Required Studies": metrics["Required"],
                "Approved Studies": metrics["Approved"],
                "Completion %": f"{metrics['Completion %']:.0f}%",
                "Last Updated": project["Last Updated"],
            }
        )
    st.subheader("Program Dashboard")
    st.dataframe(pd.DataFrame(program_rows), width="stretch")

    project_names = [str(normalize_program(project)["Program Name"]) for project in st.session_state.validation_projects]
    selected_project_name = st.selectbox("Open Validation Program", project_names)
    project = next(
        normalize_program(item)
        for item in st.session_state.validation_projects
        if normalize_program(item)["Program Name"] == selected_project_name
    )
    metrics = program_metrics(project)

    st.subheader(selected_project_name)
    detail_cols = st.columns(5)
    with detail_cols[0]:
        render_metric_card("Assay", str(project["Assay / Biomarker"]))
    with detail_cols[1]:
        render_metric_card("Studies Required", str(metrics["Required"]))
    with detail_cols[2]:
        render_metric_card("Studies Approved", str(metrics["Approved"]))
    with detail_cols[3]:
        render_metric_card("Completion", f"{metrics['Completion %']:.0f}%")
    with detail_cols[4]:
        readiness = str(metrics["Readiness"])
        render_metric_card("Readiness", readiness, status="PASS" if readiness in {"Ready for Final Package", "Completed"} else None)

    st.progress(
        float(metrics["Completion %"]) / 100,
        text=f"Program completion: {metrics['Approved']} approved, {metrics['Remaining']} remaining",
    )
    dashboard_cols = st.columns(4)
    with dashboard_cols[0]:
        render_metric_card("Completed Studies", str(metrics["Completed"]))
    with dashboard_cols[1]:
        render_metric_card("Submitted for Review", str(metrics["Submitted for Review"]))
    with dashboard_cols[2]:
        render_metric_card("Approved Studies", str(metrics["Approved"]))
    with dashboard_cols[3]:
        render_metric_card("Remaining Studies", str(metrics["Remaining"]))

    st.markdown("### Program Information")
    info_rows = [
        {"Field": "Program Name", "Value": project["Program Name"]},
        {"Field": "Assay / Biomarker", "Value": project["Assay / Biomarker"]},
        {"Field": "Program Owner", "Value": project["Program Owner"]},
        {"Field": "Reviewer", "Value": project["Reviewer"]},
        {"Field": "Start Date", "Value": project["Start Date"]},
        {"Field": "Target Completion Date", "Value": project["Target Completion Date"]},
        {"Field": "Notes", "Value": project["Notes"]},
    ]
    st.dataframe(pd.DataFrame(info_rows), width="stretch")

    st.markdown("### Study Assignment")
    assigned = st.multiselect(
        "Assigned Studies for Program",
        CORE_VALIDATION_MODULES,
        default=program_required_studies(project),
        key=f"assigned_studies_{selected_project_name}",
    )
    if st.button("Update Study Assignment"):
        project["Required Studies"] = assigned
        project["Last Updated"] = date.today().isoformat()
        record_program_activity(selected_project_name, "Program", "Study assignment updated", ", ".join(assigned))
        st.success("Study assignment updated.")

    st.markdown("### Validation Coverage Matrix")
    matrix_rows = []
    for module in program_required_studies(project):
        status = program_study_status(selected_project_name, module)
        included = status in REPORT_ELIGIBLE_STATES
        matrix_rows.append(
            {
                "Study Type": module,
                "Status": status,
                "Approval Status": "Approved" if included else "Not Approved",
                "Included in Final Package": "Yes" if included else "No",
            }
        )
    st.markdown(lifecycle_table_to_html(pd.DataFrame(matrix_rows), status_column="Status"), unsafe_allow_html=True)

    st.markdown("### Program Timeline")
    timeline = pd.DataFrame(
        [
            event for event in st.session_state.program_timeline
            if event.get("Program") == selected_project_name
        ]
    )
    if timeline.empty:
        st.info("No program activity has been recorded yet.")
    else:
        st.dataframe(timeline.sort_values("Timestamp", ascending=False), width="stretch")


def render_progress_tracker() -> None:
    """Render platform-wide validation progress tracking."""

    initialize_platform_state()
    render_page_header(
        "Validation Progress Tracker",
        "Track validation readiness across active assay programs.",
    )
    rows = []
    for project in st.session_state.validation_projects:
        completed, total, percent = project_progress(project)
        status = "Ready for Final Review" if completed == total else "In Progress" if completed else "Not Started"
        rows.append(
            {
                "Validation Project": project["Project Name"],
                "Completed Studies": completed,
                "Remaining Studies": total - completed,
                "Validation Progress %": f"{percent:.0f}%",
                "Validation Readiness Status": status,
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch")

    for project in st.session_state.validation_projects:
        completed, total, percent = project_progress(project)
        st.progress(percent / 100, text=f"{project['Project Name']}: {completed} / {total} studies")


def render_validation_review_center() -> None:
    """Render auditable study review and approval workflow."""

    initialize_platform_state()
    render_page_header(
        "Validation Review Center",
        "Review submitted studies, document reviewer decisions, and maintain validation governance traceability.",
    )
    settings = platform_settings()
    st.caption(
        f"{settings.get('Laboratory Name') or 'Laboratory not specified'}"
        f" · Reviewer: {settings.get('Reviewer Name') or 'Not specified'}"
    )
    pending_records = [
        record for record in get_lifecycle_records()
        if normalize_lifecycle_status(record.get("Status")) in {"Submitted for Review", "Under Review"}
    ]
    st.subheader("Studies Awaiting Review")
    if not pending_records:
        st.info("No studies are currently pending review.")
    else:
        pending_table = pd.DataFrame(
            [
                {
                    "Study Name": record["Study Name"],
                    "Program": record.get("Project", ""),
                    "Study Type": record["Study Type"],
                    "Assay": record["Assay"],
                    "Analyst": record["Analyst"],
                    "Submission Date": record.get("Submitted Date") or record.get("Completion Date", ""),
                    "Current Status": normalize_lifecycle_status(record["Status"]),
                    "Version": record["Version"],
                }
                for record in pending_records
            ]
        )
        st.markdown(
            lifecycle_table_to_html(pending_table, status_column="Current Status"),
            unsafe_allow_html=True,
        )

        selected_study_id = st.selectbox(
            "Select study for review",
            [str(record["Study ID"]) for record in pending_records],
            format_func=lambda value: next(
                str(record["Study Name"]) for record in pending_records if record["Study ID"] == value
            ),
        )
        study = next(record for record in pending_records if record["Study ID"] == selected_study_id)
        previous_status = normalize_lifecycle_status(study.get("Status"))
        reviewer = st.text_input("Reviewer", value=settings.get("Reviewer Name", ""))
        comments = st.text_area("Reviewer Comments", value=str(study.get("Reviewer Comments", "")), height=100)
        if previous_status == "Submitted for Review":
            if st.button("Begin Review", use_container_width=True):
                study["Status"] = "Under Review"
                study["Last Updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                record_approval_history(study, previous_status, "Under Review", "Under Review", reviewer, comments)
                record_program_activity(
                    str(study.get("Project", "")),
                    str(study.get("Study Name", "")),
                    "Under review",
                    comments,
                )
                st.rerun()
        action_cols = st.columns(3)
        with action_cols[0]:
            if st.button("Approve Study", type="primary", use_container_width=True):
                previous_status = normalize_lifecycle_status(study.get("Status"))
                study["Status"] = "Approved"
                study["Locked"] = False
                study["Reviewer Comments"] = comments
                study["Last Updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                record_approval_history(study, previous_status, "Approved", "Approved", reviewer, comments)
                record_program_activity(
                    str(study.get("Project", "")),
                    str(study.get("Study Name", "")),
                    "Approved",
                    comments,
                )
                st.success("Study approved.")
                st.rerun()
        with action_cols[1]:
            if st.button("Reject Study", use_container_width=True):
                if not comments.strip():
                    st.error("Reviewer comments are required when rejecting a study.")
                    st.stop()
                previous_status = normalize_lifecycle_status(study.get("Status"))
                study["Status"] = "Draft"
                study["Locked"] = False
                study["Reviewer Comments"] = comments
                study["Last Updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                record_approval_history(study, previous_status, "Rejected", "Rejected", reviewer, comments)
                record_program_activity(
                    str(study.get("Project", "")),
                    str(study.get("Study Name", "")),
                    "Rejected",
                    comments,
                )
                st.warning("Study rejected and returned to Draft.")
                st.rerun()
        with action_cols[2]:
            if st.button("Return for Revision", use_container_width=True):
                previous_status = normalize_lifecycle_status(study.get("Status"))
                study["Status"] = "Draft"
                study["Locked"] = False
                study["Reviewer Comments"] = comments
                study["Last Updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                record_approval_history(study, previous_status, "Draft", "Returned for Revision", reviewer, comments)
                record_program_activity(
                    str(study.get("Project", "")),
                    str(study.get("Study Name", "")),
                    "Returned for revision",
                    comments,
                )
                st.info("Study returned for revision.")
                st.rerun()

    st.subheader("Approval History")
    history = pd.DataFrame(st.session_state.approval_history)
    if history.empty:
        st.info("No approval history has been recorded yet.")
    else:
        st.dataframe(history.sort_values("Timestamp", ascending=False), width="stretch")


def render_reports_library() -> None:
    """Render the report library and report search tools."""

    initialize_platform_state()
    render_page_header(
        "Reports Library",
        "Generate validation packages, view existing reports, and download PDF or HTML outputs.",
    )
    generate_tab, library_tab = st.tabs(["Generate New Report", "View Existing Reports"])
    with generate_tab:
        render_validation_reports_workspace(embedded=True)

    with library_tab:
        render_existing_reports_library()


def render_existing_reports_library() -> None:
    """Render searchable existing report records."""

    reports = pd.DataFrame(st.session_state.reports_library)
    if reports.empty:
        st.info("No reports are currently registered in the report library.")
        return

    filters = st.columns(4)
    with filters[0]:
        search = st.text_input("Search reports")
    with filters[1]:
        assay_filter = st.selectbox("Filter by assay/project", ["All"] + sorted(reports["Project"].unique().tolist()))
    with filters[2]:
        study_filter = st.selectbox("Filter by study type", ["All"] + sorted(reports["Study Type"].unique().tolist()))
    with filters[3]:
        analyst_filter = st.selectbox("Filter by analyst", ["All"] + sorted(reports["Analyst"].fillna("").unique().tolist()))

    filtered = reports.copy()
    if search:
        mask = filtered.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        filtered = filtered[mask]
    if assay_filter != "All":
        filtered = filtered[filtered["Project"] == assay_filter]
    if study_filter != "All":
        filtered = filtered[filtered["Study Type"] == study_filter]
    if analyst_filter != "All":
        filtered = filtered[filtered["Analyst"] == analyst_filter]

    st.dataframe(filtered, width="stretch")
    selected_report = st.selectbox("Report Actions", filtered["Report Name"].tolist() if not filtered.empty else ["No reports available"])
    selected_report_row = (
        filtered[filtered["Report Name"] == selected_report].iloc[0].to_dict()
        if not filtered.empty
        else {}
    )
    action_cols = st.columns(4)
    disabled = filtered.empty
    settings = platform_settings()
    project_metadata = {
        "Validation Project Name": str(selected_report_row.get("Project") or selected_report),
        "Analyst": settings.get("Analyst Name", ""),
        "Study Date": date.today().isoformat(),
        "Instrument": "",
        "Assay / Biomarker": "HbA1c",
        "Specimen Type": "Whole Blood",
        "Protocol Number": "",
        "Reviewer": settings.get("Reviewer Name", ""),
        "Laboratory Name": settings.get("Laboratory Name", ""),
        "Report Version": APP_VERSION,
        "Department": settings.get("Department", ""),
        "Address": settings.get("Address", ""),
        "Report Logo": settings.get("Report Logo", ""),
        "Organization Branding": settings.get("Organization Branding", ""),
        "Report Footer": settings.get("Report Footer", ""),
        "Package Generated By": settings.get("Analyst Name", ""),
        "Package Generation Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Approval Status": "Generated from Approved/Locked studies",
    }
    eligible = eligible_study_types(str(selected_report_row.get("Project") or ""))
    package = generate_validation_package(
        selected_studies=eligible,
        root_dir=ROOT_DIR,
        project_metadata=project_metadata,
    ) if not disabled and eligible else None
    if package is not None:
        package_program = str(selected_report_row.get("Project") or "")
        for study in package.studies:
            study.status = program_study_status(package_program, study.study_type)
    with action_cols[0]:
        st.button("Open", disabled=disabled, use_container_width=True)
    with action_cols[1]:
        st.download_button(
            "Download PDF",
            data=build_full_pdf(package) if package else b"",
            file_name="validation_report.pdf",
            mime="application/pdf",
            disabled=disabled or package is None,
            use_container_width=True,
        )
    with action_cols[2]:
        st.download_button(
            "Download HTML",
            data=export_full_html(package, SUPPORTED_STUDIES).encode("utf-8") if package else b"",
            file_name="validation_report.html",
            mime="text/html",
            disabled=disabled or package is None,
            use_container_width=True,
        )
    with action_cols[3]:
        if st.button("Delete", disabled=disabled, use_container_width=True):
            st.session_state.reports_library = [
                report for report in st.session_state.reports_library if report["Report Name"] != selected_report
            ]
            st.rerun()


def render_sample_dataset_library() -> None:
    """Render the reusable sample dataset repository."""

    render_page_header(
        "Sample Dataset Library",
        "Browse sample datasets and load them into validation modules for demonstration workflows.",
    )
    rows = []
    for item in SAMPLE_DATASETS:
        path = Path(item["File"])
        rows.append(
            {
                "Study Type": item["Study Type"],
                "Dataset": item["Dataset"],
                "File": path.name,
                "Rows": len(pd.read_csv(path)) if path.exists() else 0,
                "Description": item["Description"],
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch")

    selected_dataset = st.selectbox("Dataset", [item["Dataset"] for item in SAMPLE_DATASETS])
    item = next(dataset for dataset in SAMPLE_DATASETS if dataset["Dataset"] == selected_dataset)
    data = pd.read_csv(item["File"])
    st.subheader("Dataset Preview")
    st.dataframe(data.head(25), width="stretch")
    cols = st.columns(2)
    with cols[0]:
        st.download_button(
            "Download sample dataset CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name=Path(item["File"]).name,
            mime="text/csv",
            use_container_width=True,
        )
    with cols[1]:
        if st.button("Load Dataset in Study Workspace", use_container_width=True):
            st.session_state.selected_sample_dataset = item["Dataset"]
            st.session_state.pending_page = "Validation Workspace"
            st.rerun()


def render_platform_settings() -> None:
    """Render centralized platform settings."""

    initialize_platform_state()
    render_page_header(
        "Platform Settings",
        "Configure laboratory, user, reporting, and export defaults used throughout validation reports.",
    )
    settings = platform_settings().copy()

    st.subheader("Laboratory Information")
    lab_cols = st.columns(3)
    with lab_cols[0]:
        settings["Laboratory Name"] = st.text_input("Laboratory Name", value=settings.get("Laboratory Name", ""))
    with lab_cols[1]:
        settings["Department"] = st.text_input("Department", value=settings.get("Department", ""))
    with lab_cols[2]:
        settings["Address"] = st.text_input("Address", value=settings.get("Address", ""))

    st.subheader("User Information")
    user_cols = st.columns(2)
    with user_cols[0]:
        settings["Analyst Name"] = st.text_input("Analyst Name", value=settings.get("Analyst Name", ""))
    with user_cols[1]:
        settings["Reviewer Name"] = st.text_input("Reviewer Name", value=settings.get("Reviewer Name", ""))

    st.subheader("Reporting Preferences")
    report_cols = st.columns(3)
    with report_cols[0]:
        settings["Report Logo"] = st.text_input("Report Logo", value=settings.get("Report Logo", ""))
    with report_cols[1]:
        settings["Report Footer"] = st.text_input("Report Footer", value=settings.get("Report Footer", ""))
    with report_cols[2]:
        settings["Organization Branding"] = st.text_input("Organization Branding", value=settings.get("Organization Branding", ""))

    st.subheader("Export Preferences")
    export_cols = st.columns(2)
    with export_cols[0]:
        settings["PDF Settings"] = st.text_input("PDF Settings", value=settings.get("PDF Settings", ""))
    with export_cols[1]:
        settings["Default Report Format"] = st.selectbox(
            "Default Report Format",
            ["PDF and HTML", "PDF", "HTML"],
            index=["PDF and HTML", "PDF", "HTML"].index(settings.get("Default Report Format", "PDF and HTML")),
        )

    if st.button("Save Platform Settings", type="primary"):
        st.session_state.platform_settings = settings
        st.success("Platform settings saved for this session.")


def render_about_page() -> None:
    """Render professional About page."""

    render_page_header(
        "About",
        "Scientific Validation Analytics Platform for analytical validation studies, specimen equivalency assessments, and validation reporting.",
    )
    cols = st.columns(3)
    with cols[0]:
        render_metric_card("Version", APP_VERSION)
    with cols[1]:
        render_metric_card("Status", APP_STATUS)
    with cols[2]:
        render_metric_card("Release Name", RELEASE_NAME)

    st.subheader("Purpose")
    st.write(
        "A software platform for analytical validation studies, specimen equivalency assessments, and validation reporting."
    )

    st.subheader("Technology Stack")
    st.write(", ".join(["Python", "Pandas", "NumPy", "SciPy", "Plotly", "Streamlit"]))

    st.subheader("Platform Capabilities")
    capability_rows = [
        {"Capability": "Validation Modules Available", "Value": ", ".join(CORE_VALIDATION_MODULES)},
        {"Capability": "Report Formats Supported", "Value": "HTML, PDF, CSV"},
        {"Capability": "Statistical Methods Implemented", "Value": "Regression, correlation, bias, recovery, precision, linearity, stability, LoB, LoD, LoQ, Bland-Altman agreement"},
    ]
    st.dataframe(pd.DataFrame(capability_rows), width="stretch")

    st.subheader("Author")
    st.write("Built by: Ankita Puri, PhD")
    st.caption("Scientific Software Portfolio Project")


def render_validation_reports_workspace(embedded: bool = False) -> None:
    """Render the consolidated Validation Reports Engine workflow."""

    settings = platform_settings()
    if not embedded:
        render_page_header(
            "Validation Reports Engine",
            "Generate consolidated validation-ready packages from completed validation modules.",
        )
    else:
        st.subheader("Generate New Report")
        st.caption("Create a consolidated validation-ready package from completed validation modules.")

    st.markdown("### Validation Report Information")
    program_names = [str(normalize_program(program)["Program Name"]) for program in st.session_state.validation_projects]
    selected_program = st.selectbox("Validation Program", program_names)
    program = next(
        normalize_program(item)
        for item in st.session_state.validation_projects
        if normalize_program(item)["Program Name"] == selected_program
    )
    row_1 = st.columns(3)
    with row_1[0]:
        project_name = st.text_input("Validation Project Name", value=selected_program)
    with row_1[1]:
        analyst = st.text_input("Analyst", value=program.get("Program Owner") or settings.get("Analyst Name", ""))
    with row_1[2]:
        study_date = st.date_input("Study Date", value=date.today())

    row_2 = st.columns(3)
    with row_2[0]:
        instrument = st.text_input("Instrument")
    with row_2[1]:
        assay = st.text_input("Assay / Biomarker", value=str(program.get("Assay / Biomarker") or "HbA1c"))
    with row_2[2]:
        specimen_type = st.text_input("Specimen Type", value="Whole Blood")

    row_3 = st.columns(3)
    with row_3[0]:
        protocol_number = st.text_input("Protocol Number")
    with row_3[1]:
        reviewer = st.text_input("Reviewer", value=program.get("Reviewer") or settings.get("Reviewer Name", ""))
    with row_3[2]:
        laboratory_name = st.text_input("Laboratory Name", value=settings.get("Laboratory Name", ""))

    row_4 = st.columns(3)
    with row_4[0]:
        report_version = st.text_input("Report Version", value=APP_VERSION)

    project_metadata = {
        "Validation Project Name": project_name,
        "Analyst": analyst,
        "Study Date": study_date.isoformat(),
        "Instrument": instrument,
        "Assay / Biomarker": assay,
        "Specimen Type": specimen_type,
        "Protocol Number": protocol_number,
        "Reviewer": reviewer,
        "Laboratory Name": laboratory_name,
        "Report Version": report_version,
        "Department": settings.get("Department", ""),
        "Address": settings.get("Address", ""),
        "Report Logo": settings.get("Report Logo", ""),
        "Report Footer": settings.get("Report Footer", ""),
        "Organization Branding": settings.get("Organization Branding", ""),
        "Package Generated By": analyst,
        "Package Generation Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Approval Status": "Generated from Approved/Locked studies",
    }

    eligible = eligible_study_types(selected_program)
    if not eligible:
        st.warning("No approved or locked studies are currently eligible for final validation packages.")
        return
    st.markdown("### Select Approved or Locked Studies")
    st.caption("Only studies with Approved or Locked lifecycle status can be included in final validation packages.")
    selected_studies: list[str] = []
    cols = st.columns(3)
    for index, study_name in enumerate(SUPPORTED_STUDIES):
        if study_name not in eligible:
            continue
        with cols[index % 3]:
            if st.checkbox(study_name, value=True):
                selected_studies.append(study_name)

    if st.button("Generate Consolidated Validation Package", type="primary"):
        if not selected_studies:
            st.warning("Select at least one approved or locked study.")
            st.stop()
        try:
            package = generate_validation_package(
                selected_studies=selected_studies,
                root_dir=ROOT_DIR,
                project_metadata=project_metadata,
            )
            for study in package.studies:
                study.status = program_study_status(selected_program, study.study_type)
        except Exception as exc:
            st.error(f"Validation package could not be generated: {exc}")
            st.stop()

        report_record = {
            "Report Name": project_name,
            "Project": selected_program,
            "Study Type": "Validation Reports",
            "Date": study_date.isoformat(),
            "Analyst": analyst,
            "Format": "HTML / PDF",
        }
        if report_record["Report Name"] not in [
            report["Report Name"] for report in st.session_state.reports_library
        ]:
            st.session_state.reports_library.append(report_record)
        for program in st.session_state.validation_projects:
            if str(program.get("Program Name") or program.get("Project Name")) == selected_program:
                program["Final Package Generated"] = True
                program["Status"] = "Completed"
                program["Overall Status"] = "Completed"
                program["Last Updated"] = date.today().isoformat()
        record_program_activity(
            selected_program,
            "Validation Package",
            "Package generated",
            "Consolidated final validation package generated.",
        )

        counts = study_counts(package.studies)
        st.markdown("### Executive Summary")
        st.info(package.executive_narrative)
        first_row = st.columns(4)
        with first_row[0]:
            render_metric_card("Overall Validation Status", package.overall_status, status=package.overall_status)
        with first_row[1]:
            render_metric_card("Studies Completed", str(counts["Number completed"]))
        with first_row[2]:
            render_metric_card("Studies Passed", str(counts["Number passed"]))
        with first_row[3]:
            render_metric_card("Studies Failed", str(counts["Number failed"]))

        st.markdown("### Validation Risk Summary")
        st.dataframe(package.risk_summary, width="stretch")

        st.markdown("### Included Studies")
        for row_start in range(0, len(package.studies), 3):
            card_columns = st.columns(3)
            for column, study in zip(card_columns, package.studies[row_start : row_start + 3]):
                with column:
                    lifecycle_status = program_study_status(selected_program, study.study_type)
                    st.markdown(
                        f"""
                        <div class="svap-card">
                          <div class="svap-card-label">{escape(study.study_type)}</div>
                          <div class="svap-card-value">{status_badge_html(study.decision)}</div>
                          <div class="svap-card-subtext">
                            {escape(study.study_name)}<br>{escape(study.date)}<br>
                            Lifecycle: {lifecycle_badge_html(lifecycle_status)}
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.markdown("### Validation Coverage Matrix")
        st.caption(f"Overall validation completeness: {package.completeness_percent:.1f}%")
        st.dataframe(package.coverage_matrix, width="stretch")

        st.markdown("### Cross-Study Validation Summary")
        st.dataframe(package.validation_matrix, width="stretch")

        st.markdown("### Quality Assurance Section")
        st.dataframe(package.qa_findings, width="stretch")

        st.markdown("### Individual Study Sections")
        for study in package.studies:
            with st.expander(f"{study.study_type} - {study.decision}", expanded=False):
                st.markdown("**Study Traceability**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Traceability Element": "Study Name", "Value": study.study_name},
                            {"Traceability Element": "Analysis Version", "Value": study.analysis_version},
                            {"Traceability Element": "Source Dataset", "Value": study.source_dataset},
                            {"Traceability Element": "Analysis Timestamp", "Value": study.analysis_timestamp},
                            {"Traceability Element": "Sample Count", "Value": study.sample_count},
                            {"Traceability Element": "Excluded Samples", "Value": study.excluded_samples},
                        ]
                    ),
                    width="stretch",
                )
                st.markdown("**Key Findings**")
                st.dataframe(study.key_findings, width="stretch")
                st.markdown("**Acceptance Criteria Assessment**")
                render_badged_criteria_table(study.acceptance_criteria)
                st.markdown("**Statistical Methods**")
                st.write(study.statistical_methods)
                st.markdown("**Scientific Interpretation**")
                st.info(study.interpretation)
                st.markdown("**Conclusion**")
                st.write(study.conclusion)
                st.download_button(
                    f"Download {study.study_type} Section HTML",
                    data=build_study_section_html(study).encode("utf-8"),
                    file_name=f"{study.study_type.lower().replace(' ', '_')}_study_section.html",
                    mime="text/html",
                )

        st.markdown("### Final Validation Conclusion")
        st.write(package.decision_justification)
        st.info(package.final_conclusion)

        st.markdown("### Reviewer Sign-Off")
        st.markdown(
            """
            Prepared By: ________________________________

            Reviewed By: ________________________________

            Approved By: ________________________________

            Approval Date: ______________________________
            """
        )

        full_html = export_full_html(package, SUPPORTED_STUDIES)
        full_pdf = build_full_pdf(package)
        executive_pdf = build_executive_summary_pdf(package)

        export_cols = st.columns(3)
        with export_cols[0]:
            st.download_button(
                "Download Full Validation Package HTML",
                data=full_html.encode("utf-8"),
                file_name="validation_package.html",
                mime="text/html",
            )
        with export_cols[1]:
            st.download_button(
                "Download Full Validation Package PDF",
                data=full_pdf,
                file_name="validation_package.pdf",
                mime="application/pdf",
            )
        with export_cols[2]:
            st.download_button(
                "Download Executive Summary PDF",
                data=executive_pdf,
                file_name="validation_executive_summary.pdf",
                mime="application/pdf",
            )


def render_study_type_selector() -> str:
    """Render study-type selection and module capabilities."""

    render_page_header(
        "Validation Workspace",
        "Launch validation studies from a common workspace with shared documentation, criteria, analysis, and export patterns.",
    )
    if "selected_sample_dataset" in st.session_state:
        st.info(f"Selected sample dataset: {st.session_state.selected_sample_dataset}")
    st.subheader("Study Type")
    study_type = st.selectbox("Select validation study type", get_study_type_names())
    config = get_study_type_config(study_type)
    st.caption(config.description)

    with st.expander("Study module framework", expanded=False):
        cols = st.columns(4)
        cols[0].markdown("**Metrics**")
        cols[0].write("\n".join(f"- {item}" for item in config.metrics))
        cols[1].markdown("**Acceptance Criteria**")
        cols[1].write("\n".join(f"- {item}" for item in config.acceptance_criteria))
        cols[2].markdown("**Visualizations**")
        cols[2].write("\n".join(f"- {item}" for item in config.visualizations))
        cols[3].markdown("**Report Sections**")
        cols[3].write("\n".join(f"- {item}" for item in config.report_sections))

    if not config.implemented:
        st.info(
            f"{study_type} is registered in the platform framework. "
            "Its custom metrics, criteria, visualizations, and report sections can be implemented in a future module."
        )
    return study_type


def render_study_documentation(study_type: str) -> dict[str, object]:
    """Render validation study documentation inputs and return collected values."""

    settings = platform_settings()
    is_precision = study_type == "Precision Study"
    is_linearity = study_type == "Linearity Study"
    is_stability = study_type == "Stability Study"
    is_accuracy = study_type == "Accuracy Study"
    is_detection = study_type == "Detection Capability"
    is_dbs = study_type == "DBS Validation"
    is_microtainer = study_type == "Microtainer Validation"
    default_study_name = (
        "HbA1c Precision Study"
        if is_precision
        else "HbA1c Linearity Study"
        if is_linearity
        else "HbA1c Stability Study"
        if is_stability
        else "HbA1c Accuracy Study"
        if is_accuracy
        else "HbA1c Detection Capability Study"
        if is_detection
        else "HbA1c DBS Validation Study"
        if is_dbs
        else "HbA1c Microtainer Validation Study"
        if is_microtainer
        else "HbA1c Method Comparison Study"
    )
    default_objective = (
        "Evaluate repeatability of repeated HbA1c measurements."
        if is_precision
        else "Evaluate proportionality of observed HbA1c results across the analytical range."
        if is_linearity
        else "Evaluate stability of HbA1c measurements over predefined storage conditions and timepoints."
        if is_stability
        else "Evaluate agreement between observed HbA1c results and expected target values."
        if is_accuracy
        else "Evaluate assay detection capability through determination of LoB, LoD, and LoQ using replicate measurements of blank and low-concentration samples."
        if is_detection
        else "Evaluate whether DBS-derived results demonstrate acceptable analytical agreement with reference venous specimens."
        if is_dbs
        else "Evaluate whether microtainer-derived results demonstrate acceptable analytical agreement with reference venous specimens."
        if is_microtainer
        else "Evaluate agreement between candidate and reference results."
    )
    default_design = (
        "Repeated low-level and high-level quality control measurements across multiple days, runs, and replicates."
        if is_precision
        else "Expected HbA1c levels measured in replicate across the reportable analytical range."
        if is_linearity
        else "Repeated measurements collected at baseline and subsequent timepoints under controlled storage conditions."
        if is_stability
        else "Replicate measurements of samples with known or assigned expected values across multiple concentration levels."
        if is_accuracy
        else "Replicate blank, low-concentration, and quantitation-level samples analyzed to estimate LoB, LoD, and LoQ."
        if is_detection
        else "Paired DBS and venous whole blood specimens analyzed to evaluate recovery, bias, correlation, and agreement."
        if is_dbs
        else "Paired capillary microtainer and venous specimens analyzed to assess analytical equivalency, bias, recovery, correlation, and agreement."
        if is_microtainer
        else "Paired specimen comparison using reference and candidate measurements."
    )

    st.subheader("Validation Study Documentation")
    project_names = [str(project["Project Name"]) for project in st.session_state.validation_projects]
    validation_project_name = st.selectbox(
        "Validation Project",
        project_names,
        index=0 if project_names else None,
    )
    first_row = st.columns(3)
    with first_row[0]:
        study_name = st.text_input(
            "Study Name",
            value=default_study_name,
        )
    with first_row[1]:
        analyst_name = st.text_input("Analyst Name", value=settings.get("Analyst Name", ""))
    with first_row[2]:
        study_date = st.date_input("Study Date", value=date.today())

    study_objective = st.text_area(
        "Study Objective",
        value=default_objective,
        height=80,
    )
    study_design = st.text_area(
        "Study Design",
        value=default_design,
        height=80,
    )

    second_row = st.columns(2)
    with second_row[0]:
        assay_name = st.text_input("Assay / Biomarker", value="HbA1c")
    with second_row[1]:
        specimen_type = st.text_input("Specimen Type", value="Whole blood")

    reference_method = ""
    candidate_method = ""
    if is_precision or is_linearity or is_stability or is_accuracy or is_detection or is_dbs or is_microtainer:
        units = st.text_input("Units", value="%")
        key_prefix = (
            "detection"
            if is_detection
            else "dbs"
            if is_dbs
            else "microtainer"
            if is_microtainer
            else "accuracy"
            if is_accuracy
            else
            "stability"
            if is_stability
            else "linearity"
            if is_linearity
            else "precision"
        )
        instrument_name = instrument_id = ""
        laboratory_site = settings.get("Laboratory Name", "")
        reagent_lot = calibrator_lot = qc_lot = operator_name = ""
        study_protocol_id = clsi_guideline_reference = ""
        dbs_collection_device = dbs_punch_size = extraction_method = specimen_matrix = ""
        microtainer_type = anticoagulant = collection_device = specimen_comparison = ""
        with st.expander("Laboratory Documentation", expanded=False):
            lab_row_1 = st.columns(3)
            with lab_row_1[0]:
                instrument_name = st.text_input("Instrument Name", key=f"{key_prefix}_instrument_name")
            with lab_row_1[1]:
                instrument_id = st.text_input("Instrument ID", key=f"{key_prefix}_instrument_id")
            with lab_row_1[2]:
                laboratory_site = st.text_input(
                    "Laboratory Site",
                    value=settings.get("Laboratory Name", ""),
                    key=f"{key_prefix}_laboratory_site",
                )

            lab_row_2 = st.columns(3)
            with lab_row_2[0]:
                reagent_lot = st.text_input("Reagent Lot Number", key=f"{key_prefix}_reagent_lot")
            with lab_row_2[1]:
                calibrator_lot = st.text_input("Calibrator Lot Number", key=f"{key_prefix}_calibrator_lot")
            with lab_row_2[2]:
                qc_lot = st.text_input("Quality Control Lot Number", key=f"{key_prefix}_qc_lot")

            operator_name = st.text_input("Operator Name", key=f"{key_prefix}_operator_name")
            if is_dbs:
                dbs_row = st.columns(4)
                with dbs_row[0]:
                    dbs_collection_device = st.text_input(
                        "DBS Collection Device",
                        key=f"{key_prefix}_collection_device",
                    )
                with dbs_row[1]:
                    dbs_punch_size = st.text_input(
                        "DBS Punch Size",
                        value="6 mm",
                        key=f"{key_prefix}_punch_size",
                    )
                with dbs_row[2]:
                    extraction_method = st.text_input(
                        "Extraction Method",
                        key=f"{key_prefix}_extraction_method",
                    )
                with dbs_row[3]:
                    specimen_matrix = st.text_input(
                        "Specimen Matrix",
                        value="DBS vs Venous Whole Blood",
                        key=f"{key_prefix}_specimen_matrix",
                    )
            if is_microtainer:
                micro_row = st.columns(4)
                with micro_row[0]:
                    collection_device = st.text_input(
                        "Microtainer Collection Device",
                        key=f"{key_prefix}_collection_device",
                    )
                with micro_row[1]:
                    microtainer_type = st.text_input(
                        "Microtainer Type",
                        value="Capillary whole blood microtainer",
                        key=f"{key_prefix}_type",
                    )
                with micro_row[2]:
                    anticoagulant = st.text_input(
                        "Anticoagulant / Additive",
                        key=f"{key_prefix}_anticoagulant",
                    )
                with micro_row[3]:
                    specimen_comparison = st.text_input(
                        "Specimen Comparison",
                        value="Microtainer whole blood vs venous whole blood",
                        key=f"{key_prefix}_specimen_comparison",
                    )
            if is_detection:
                detection_row = st.columns(2)
                with detection_row[0]:
                    study_protocol_id = st.text_input(
                        "Study Protocol ID",
                        key=f"{key_prefix}_study_protocol_id",
                    )
                with detection_row[1]:
                    clsi_guideline_reference = st.text_input(
                        "CLSI Guideline Reference",
                        value="CLSI EP17-A2",
                        key=f"{key_prefix}_clsi_guideline_reference",
                    )
    else:
        third_row = st.columns(3)
        with third_row[0]:
            reference_method = st.text_input("Reference Method", value="Reference HbA1c")
        with third_row[1]:
            candidate_method = st.text_input("Candidate Method", value="Candidate HbA1c")
        with third_row[2]:
            units = st.text_input("Units", value="%")

    notes = st.text_area("Notes", height=80)
    deviations = st.text_area("Deviations", height=80)
    conclusions = st.text_area("Conclusions", height=80)

    metadata = {
        "Study Name": study_name,
        "Validation Project Name": validation_project_name,
        "Study Objective": study_objective,
        "Study Design": study_design,
        "Assay / Biomarker": assay_name,
        "Specimen Type": specimen_type,
        "Analyst Name": analyst_name,
        "Study Date": study_date.isoformat(),
        "Sample Count": "Calculated after analysis",
        "Units": units,
        "Notes": notes,
        "Deviations": deviations,
        "Conclusions": conclusions,
        "Laboratory Name": settings.get("Laboratory Name", ""),
        "Department": settings.get("Department", ""),
        "Address": settings.get("Address", ""),
        "Reviewer": settings.get("Reviewer Name", ""),
        "Report Logo": settings.get("Report Logo", ""),
        "Report Footer": settings.get("Report Footer", ""),
        "Organization Branding": settings.get("Organization Branding", ""),
    }
    if not (is_precision or is_linearity or is_stability or is_accuracy or is_detection or is_dbs or is_microtainer):
        metadata["Reference Method"] = reference_method
        metadata["Candidate Method"] = candidate_method
    elif is_precision or is_linearity or is_stability or is_accuracy or is_detection or is_dbs or is_microtainer:
        metadata.update(
            {
                "Instrument Name": instrument_name,
                "Instrument ID": instrument_id,
                "Reagent Lot Number": reagent_lot,
                "Calibrator Lot Number": calibrator_lot,
                "Quality Control Lot Number": qc_lot,
                "Operator Name": operator_name,
                "Laboratory Site": laboratory_site,
            }
        )
        if is_detection:
            metadata.update(
                {
                    "Instrument": instrument_name,
                    "Reagent Lot": reagent_lot,
                    "Calibrator Lot": calibrator_lot,
                    "Operator": operator_name,
                    "Study Protocol ID": study_protocol_id,
                    "CLSI Guideline Reference": clsi_guideline_reference,
                }
            )
        if is_dbs:
            metadata.update(
                {
                    "Assay": assay_name,
                    "Biomarker": assay_name,
                    "Protocol Number": "",
                    "DBS Collection Device": dbs_collection_device,
                    "DBS Punch Size": dbs_punch_size,
                    "Extraction Method": extraction_method,
                    "Instrument": instrument_name,
                    "Specimen Matrix": specimen_matrix or "DBS vs Venous Whole Blood",
                    "Specimen Comparison": "DBS vs Venous Whole Blood",
                }
            )
        if is_microtainer:
            metadata.update(
                {
                    "Reviewer": settings.get("Reviewer Name", ""),
                    "Protocol Number": "",
                    "Laboratory Name": laboratory_site or settings.get("Laboratory Name", ""),
                    "Instrument": instrument_name,
                    "Microtainer Collection Device": collection_device,
                    "Microtainer Type": microtainer_type,
                    "Anticoagulant / Additive": anticoagulant,
                    "Specimen Comparison": specimen_comparison or "Microtainer whole blood vs venous whole blood",
                }
            )
    return metadata


def render_method_comparison_criteria() -> dict[str, float | None]:
    """Render user-defined preliminary acceptance criteria controls."""

    st.subheader("Acceptance Criteria")
    st.caption(
        "These criteria are preliminary screening thresholds selected by the user."
    )
    criteria_columns = st.columns(4)
    with criteria_columns[0]:
        min_r_squared = st.number_input(
            "Minimum acceptable R²",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.01,
            format="%.2f",
        )
        min_correlation = st.number_input(
            "Minimum acceptable correlation coefficient (r)",
            min_value=-1.0,
            max_value=1.0,
            value=0.95,
            step=0.01,
            format="%.2f",
        )
    with criteria_columns[1]:
        slope_lower = st.number_input(
            "Slope lower limit",
            value=0.90,
            step=0.01,
            format="%.2f",
        )
        slope_upper = st.number_input(
            "Slope upper limit",
            value=1.10,
            step=0.01,
            format="%.2f",
        )
    with criteria_columns[2]:
        max_abs_intercept = st.number_input(
            "Maximum acceptable intercept",
            min_value=0.0,
            value=0.50,
            step=0.05,
            format="%.3f",
        )
        max_abs_mean_bias = st.number_input(
            "Maximum acceptable absolute mean bias",
            min_value=0.0,
            value=0.50,
            step=0.05,
            format="%.3f",
        )
    with criteria_columns[3]:
        max_abs_percent_bias = st.number_input(
            "Maximum acceptable absolute mean percent bias",
            min_value=0.0,
            value=5.0,
            step=0.5,
            format="%.2f",
        )
        max_abs_mean_difference = st.number_input(
            "Maximum acceptable absolute mean difference",
            min_value=0.0,
            value=0.50,
            step=0.05,
            format="%.3f",
        )

    agreement_columns = st.columns(2)
    with agreement_columns[0]:
        sample_agreement_limit = st.number_input(
            "Sample agreement percent bias limit",
            min_value=0.0,
            value=5.0,
            step=0.5,
            format="%.2f",
        )
    with agreement_columns[1]:
        min_percent_samples_agree = st.number_input(
            "Minimum percent of samples meeting agreement criteria",
            min_value=0.0,
            max_value=100.0,
            value=95.0,
            step=1.0,
            format="%.1f",
        )

    return {
        "Minimum R²": min_r_squared,
        "Minimum Correlation r": min_correlation,
        "Slope Lower Limit": slope_lower,
        "Slope Upper Limit": slope_upper,
        "Maximum Absolute Intercept": max_abs_intercept,
        "Maximum Absolute Mean Bias": max_abs_mean_bias,
        "Maximum Absolute Mean Percent Bias": max_abs_percent_bias,
        "Maximum Absolute Mean Difference": max_abs_mean_difference,
        "Sample Agreement Percent Bias Limit": sample_agreement_limit,
        "Minimum Percent Samples Meeting Agreement": min_percent_samples_agree,
    }


def render_precision_criteria() -> dict[str, float]:
    """Render user-defined preliminary precision acceptance criteria."""

    st.subheader("Acceptance Criteria")
    st.caption(
        "These criteria are preliminary screening thresholds selected by the user, not regulatory approval."
    )
    max_cv = st.number_input(
        "Maximum acceptable CV%",
        min_value=0.0,
        value=5.0,
        step=0.5,
        format="%.2f",
    )
    return {"Maximum CV%": max_cv}


def render_linearity_criteria() -> dict[str, float]:
    """Render user-defined preliminary linearity acceptance criteria."""

    st.subheader("Acceptance Criteria")
    st.caption(
        "These criteria are user-defined preliminary screening thresholds, not regulatory approval."
    )
    row_1 = st.columns(3)
    with row_1[0]:
        min_r_squared = st.number_input(
            "Minimum acceptable R²",
            min_value=0.0,
            max_value=1.0,
            value=0.99,
            step=0.001,
            format="%.3f",
        )
    with row_1[1]:
        slope_lower = st.number_input(
            "Slope lower limit",
            value=0.95,
            step=0.01,
            format="%.2f",
        )
    with row_1[2]:
        slope_upper = st.number_input(
            "Slope upper limit",
            value=1.05,
            step=0.01,
            format="%.2f",
        )

    row_2 = st.columns(3)
    with row_2[0]:
        max_abs_bias = st.number_input(
            "Maximum acceptable absolute percent bias by level",
            min_value=0.0,
            value=10.0,
            step=0.5,
            format="%.2f",
        )
    with row_2[1]:
        recovery_lower = st.number_input(
            "Percent recovery lower limit",
            value=90.0,
            step=1.0,
            format="%.2f",
        )
    with row_2[2]:
        recovery_upper = st.number_input(
            "Percent recovery upper limit",
            value=110.0,
            step=1.0,
            format="%.2f",
        )

    return {
        "Minimum R²": min_r_squared,
        "Slope Lower Limit": slope_lower,
        "Slope Upper Limit": slope_upper,
        "Maximum Absolute Percent Bias": max_abs_bias,
        "Recovery Lower Limit": recovery_lower,
        "Recovery Upper Limit": recovery_upper,
    }


def render_stability_criteria() -> dict[str, float]:
    """Render user-defined preliminary stability acceptance criteria."""

    st.subheader("Acceptance Criteria")
    st.caption(
        "These criteria are user-defined preliminary screening thresholds, not regulatory approval."
    )
    row = st.columns(4)
    with row[0]:
        max_percent_change = st.number_input(
            "Maximum acceptable percent change from baseline",
            min_value=0.0,
            value=10.0,
            step=0.5,
            format="%.2f",
        )
    with row[1]:
        min_recovery = st.number_input(
            "Minimum acceptable recovery %",
            min_value=0.0,
            max_value=150.0,
            value=90.0,
            step=0.5,
            format="%.2f",
        )
    with row[2]:
        max_abs_bias = st.number_input(
            "Maximum acceptable absolute bias",
            min_value=0.0,
            value=0.50,
            step=0.05,
            format="%.2f",
        )
    with row[3]:
        borderline_zone = st.number_input(
            "Borderline zone (%)",
            min_value=0.0,
            value=2.0,
            step=0.5,
            format="%.2f",
        )

    return {
        "Maximum Percent Change": max_percent_change,
        "Minimum Recovery": min_recovery,
        "Maximum Absolute Bias": max_abs_bias,
        "Borderline Zone": borderline_zone,
    }


def render_accuracy_criteria() -> dict[str, float]:
    """Render user-defined preliminary accuracy acceptance criteria."""

    st.subheader("Acceptance Criteria")
    st.caption(
        "These criteria are user-defined preliminary screening thresholds, not regulatory approval."
    )
    row = st.columns(5)
    with row[0]:
        max_abs_bias = st.number_input(
            "Maximum acceptable absolute bias",
            min_value=0.0,
            value=0.50,
            step=0.05,
            format="%.2f",
        )
    with row[1]:
        max_abs_percent_bias = st.number_input(
            "Maximum acceptable absolute percent bias",
            min_value=0.0,
            value=10.0,
            step=0.5,
            format="%.2f",
        )
    with row[2]:
        min_recovery = st.number_input(
            "Minimum acceptable recovery %",
            min_value=0.0,
            max_value=150.0,
            value=90.0,
            step=0.5,
            format="%.2f",
        )
    with row[3]:
        max_recovery = st.number_input(
            "Maximum acceptable recovery %",
            min_value=0.0,
            max_value=200.0,
            value=110.0,
            step=0.5,
            format="%.2f",
        )
    with row[4]:
        borderline_zone = st.number_input(
            "Borderline zone (%)",
            min_value=0.0,
            value=2.0,
            step=0.5,
            format="%.2f",
        )

    return {
        "Maximum Absolute Bias": max_abs_bias,
        "Maximum Absolute Percent Bias": max_abs_percent_bias,
        "Minimum Recovery": min_recovery,
        "Maximum Recovery": max_recovery,
        "Borderline Zone": borderline_zone,
    }


def render_detection_criteria() -> dict[str, float]:
    """Render user-defined preliminary detection capability criteria."""

    st.subheader("Acceptance Criteria")
    st.caption(
        "These criteria are user-defined preliminary screening thresholds, not regulatory approval."
    )
    row = st.columns(5)
    with row[0]:
        max_lob = st.number_input(
            "Maximum acceptable LoB",
            min_value=0.0,
            value=0.15,
            step=0.01,
            format="%.3f",
        )
    with row[1]:
        max_lod = st.number_input(
            "Maximum acceptable LoD",
            min_value=0.0,
            value=0.30,
            step=0.01,
            format="%.3f",
        )
    with row[2]:
        target_loq_cv = st.number_input(
            "Target LoQ CV%",
            min_value=0.0,
            value=20.0,
            step=1.0,
            format="%.2f",
        )
    with row[3]:
        max_loq = st.number_input(
            "Maximum acceptable LoQ concentration",
            min_value=0.0,
            value=1.00,
            step=0.05,
            format="%.3f",
        )
    with row[4]:
        borderline_zone = st.number_input(
            "Borderline zone %",
            min_value=0.0,
            value=2.0,
            step=0.5,
            format="%.2f",
        )
    return {
        "Maximum LoB": max_lob,
        "Maximum LoD": max_lod,
        "Target LoQ CV%": target_loq_cv,
        "Maximum LoQ Concentration": max_loq,
        "Borderline Zone": borderline_zone,
    }


def render_dbs_criteria() -> dict[str, float]:
    """Render user-defined preliminary DBS validation criteria."""

    st.subheader("Acceptance Criteria")
    st.caption(
        "These criteria are user-defined preliminary screening thresholds, not regulatory approval."
    )
    row = st.columns(5)
    with row[0]:
        max_percent_bias = st.number_input(
            "Maximum percent bias",
            min_value=0.0,
            value=10.0,
            step=0.5,
            format="%.2f",
        )
    with row[1]:
        min_recovery = st.number_input(
            "Recovery lower limit (%)",
            min_value=0.0,
            max_value=150.0,
            value=90.0,
            step=0.5,
            format="%.2f",
        )
    with row[2]:
        max_recovery = st.number_input(
            "Recovery upper limit (%)",
            min_value=0.0,
            max_value=200.0,
            value=110.0,
            step=0.5,
            format="%.2f",
        )
    with row[3]:
        min_r_squared = st.number_input(
            "Minimum acceptable R²",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.01,
            format="%.2f",
        )
    with row[4]:
        borderline_zone = st.number_input(
            "Borderline zone (%)",
            min_value=0.0,
            value=2.0,
            step=0.5,
            format="%.2f",
        )
    max_mean_difference = st.number_input(
        "Maximum mean difference",
        min_value=0.0,
        value=0.50,
        step=0.05,
        format="%.2f",
    )
    return {
        "Maximum Percent Bias": max_percent_bias,
        "Minimum Recovery": min_recovery,
        "Maximum Recovery": max_recovery,
        "Minimum R²": min_r_squared,
        "Maximum Mean Difference": max_mean_difference,
        "Borderline Zone": borderline_zone,
    }


def render_microtainer_criteria() -> dict[str, float | str]:
    """Render preset-driven Microtainer preliminary acceptance criteria."""

    st.subheader("Acceptance Criteria")
    st.caption(
        "These criteria are user-defined preliminary screening thresholds, not regulatory approval."
    )
    preset = st.selectbox(
        "Criteria template",
        ["Internal Validation", "Research Use", "CLIA Laboratory Evaluation", "Custom Criteria"],
    )
    preset_values = {
        "Internal Validation": (10.0, 90.0, 110.0, 0.95, 0.50, 2.0),
        "Research Use": (15.0, 85.0, 115.0, 0.90, 0.75, 3.0),
        "CLIA Laboratory Evaluation": (8.0, 92.0, 108.0, 0.97, 0.40, 2.0),
        "Custom Criteria": (10.0, 90.0, 110.0, 0.95, 0.50, 2.0),
    }
    max_bias, recovery_lower, recovery_upper, min_r2, max_difference, borderline = preset_values[preset]
    row = st.columns(6)
    with row[0]:
        max_bias = st.number_input("Maximum absolute percent bias", min_value=0.0, value=max_bias, step=0.5, format="%.2f")
    with row[1]:
        recovery_lower = st.number_input("Recovery lower limit", min_value=0.0, max_value=150.0, value=recovery_lower, step=0.5, format="%.2f")
    with row[2]:
        recovery_upper = st.number_input("Recovery upper limit", min_value=0.0, max_value=200.0, value=recovery_upper, step=0.5, format="%.2f")
    with row[3]:
        min_r2 = st.number_input("Minimum R²", min_value=0.0, max_value=1.0, value=min_r2, step=0.01, format="%.2f")
    with row[4]:
        max_difference = st.number_input("Maximum mean difference", min_value=0.0, value=max_difference, step=0.05, format="%.2f")
    with row[5]:
        borderline = st.number_input("Borderline zone", min_value=0.0, value=borderline, step=0.5, format="%.2f")
    return {
        "Criteria Template": preset,
        "Maximum Absolute Percent Bias": max_bias,
        "Recovery Lower Limit": recovery_lower,
        "Recovery Upper Limit": recovery_upper,
        "Minimum R²": min_r2,
        "Maximum Mean Difference": max_difference,
        "Borderline Zone": borderline,
    }


def optional_column_selectbox(
    label: str, columns: list[str], default_column: str | None = None
) -> str | None:
    """Render a selectbox that can return no selected column."""

    options = ["None"] + columns
    default_index = options.index(default_column) if default_column in options else 0
    selected = st.selectbox(label, options, index=default_index)
    return None if selected == "None" else selected


def show_decision(decision: str) -> None:
    """Display the preliminary decision with simple visual emphasis."""

    if decision == "PASS":
        st.success("Overall preliminary decision: PASS")
    elif decision in {"BORDERLINE", "REVIEW", "PASS WITH CAUTION"}:
        st.warning(f"Overall preliminary decision: {decision}")
    else:
        label = "INVESTIGATE" if decision == "INVESTIGATE" else "FAIL"
        st.error(f"Overall preliminary decision: {label}")


def build_precision_summary_csv(
    metadata: dict[str, object],
    criteria_table: pd.DataFrame,
    level_summary: pd.DataFrame,
    day_summary: pd.DataFrame,
    run_summary: pd.DataFrame,
) -> str:
    """Build a sectioned CSV summary for precision study documentation."""

    buffer = StringIO()
    pd.DataFrame(
        [{"Field": key, "Value": value or "Not specified"} for key, value in metadata.items()]
    ).to_csv(buffer, index=False)
    buffer.write("\nAcceptance Criteria Results\n")
    criteria_table.to_csv(buffer, index=False)
    buffer.write("\nPrecision Summary\n")
    level_summary.to_csv(buffer, index=False)
    if not day_summary.empty:
        buffer.write("\nDay-Level Summary\n")
        day_summary.to_csv(buffer, index=False)
    if not run_summary.empty:
        buffer.write("\nRun-Level Summary\n")
        run_summary.to_csv(buffer, index=False)
    return buffer.getvalue()


def build_linearity_summary_csv(
    metadata: dict[str, object],
    criteria_table: pd.DataFrame,
    level_summary: pd.DataFrame,
    regression_summary: pd.DataFrame,
) -> str:
    """Build a sectioned CSV summary for linearity study documentation."""

    buffer = StringIO()
    pd.DataFrame(
        [{"Field": key, "Value": value or "Not specified"} for key, value in metadata.items()]
    ).to_csv(buffer, index=False)
    buffer.write("\nLinearity Summary\n")
    level_summary.to_csv(buffer, index=False)
    buffer.write("\nAcceptance Criteria Results\n")
    criteria_table.to_csv(buffer, index=False)
    buffer.write("\nRegression Summary\n")
    regression_summary.to_csv(buffer, index=False)
    return buffer.getvalue()


def build_stability_summary_csv(
    metadata: dict[str, object],
    criteria_table: pd.DataFrame,
    overall_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    timepoint_summary: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    bias_summary: pd.DataFrame,
    condition_comparison: pd.DataFrame,
    outlier_table: pd.DataFrame,
    risk_assessment: str,
) -> str:
    """Build a sectioned CSV summary for stability study documentation."""

    buffer = StringIO()
    pd.DataFrame(
        [{"Field": key, "Value": value or "Not specified"} for key, value in metadata.items()]
    ).to_csv(buffer, index=False)
    buffer.write("\nOverall Stability Summary\n")
    overall_summary.to_csv(buffer, index=False)
    buffer.write("\nAcceptance Criteria Results\n")
    criteria_table.to_csv(buffer, index=False)
    buffer.write("\nStability Summary\n")
    stability_summary.to_csv(buffer, index=False)
    buffer.write("\nTimepoint Analysis\n")
    timepoint_summary.to_csv(buffer, index=False)
    buffer.write("\nRecovery Analysis\n")
    recovery_summary.to_csv(buffer, index=False)
    buffer.write("\nBias Analysis\n")
    bias_summary.to_csv(buffer, index=False)
    if not condition_comparison.empty:
        buffer.write("\nStorage Condition Comparison\n")
        condition_comparison.to_csv(buffer, index=False)
    if not outlier_table.empty:
        buffer.write("\nPotential Stability Outliers\n")
        outlier_table.to_csv(buffer, index=False)
    buffer.write("\nRisk Assessment\n")
    pd.DataFrame([{"Risk Assessment": risk_assessment}]).to_csv(buffer, index=False)
    return buffer.getvalue()


def build_accuracy_summary_csv(
    metadata: dict[str, object],
    criteria_table: pd.DataFrame,
    overall_summary: pd.DataFrame,
    level_decisions: pd.DataFrame,
    accuracy_summary: pd.DataFrame,
    bias_summary: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    worst_case_summary: pd.DataFrame,
) -> str:
    """Build a sectioned CSV summary for accuracy study documentation."""

    buffer = StringIO()
    pd.DataFrame(
        [{"Field": key, "Value": value or "Not specified"} for key, value in metadata.items()]
    ).to_csv(buffer, index=False)
    buffer.write("\nOverall Accuracy Summary\n")
    overall_summary.to_csv(buffer, index=False)
    buffer.write("\nLevel-Specific Decision Table\n")
    level_decisions.to_csv(buffer, index=False)
    buffer.write("\nAcceptance Criteria Results\n")
    criteria_table.to_csv(buffer, index=False)
    buffer.write("\nAccuracy Summary\n")
    accuracy_summary.to_csv(buffer, index=False)
    buffer.write("\nBias Summary\n")
    bias_summary.to_csv(buffer, index=False)
    buffer.write("\nRecovery Summary\n")
    recovery_summary.to_csv(buffer, index=False)
    buffer.write("\nWorst-Case Performance\n")
    worst_case_summary.to_csv(buffer, index=False)
    return buffer.getvalue()


def build_detection_summary_csv(
    metadata: dict[str, object],
    criteria_table: pd.DataFrame,
    overall_summary: pd.DataFrame,
    methodology_table: pd.DataFrame,
    data_quality_summary: pd.DataFrame,
    outlier_table: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    lob_summary: pd.DataFrame,
    lod_summary: pd.DataFrame,
    loq_summary: pd.DataFrame,
    analyzed_data: pd.DataFrame,
) -> str:
    """Build a sectioned CSV summary for detection capability documentation."""

    buffer = StringIO()
    pd.DataFrame(
        [{"Field": key, "Value": value or "Not specified"} for key, value in metadata.items()]
    ).to_csv(buffer, index=False)
    buffer.write("\nExecutive Summary\n")
    overall_summary.to_csv(buffer, index=False)
    buffer.write("\nAcceptance Criteria Results\n")
    criteria_table.to_csv(buffer, index=False)
    buffer.write("\nCalculation Methodology\n")
    methodology_table.to_csv(buffer, index=False)
    buffer.write("\nData Quality Assessment\n")
    data_quality_summary.to_csv(buffer, index=False)
    if not outlier_table.empty:
        buffer.write("\nIQR Outlier Review\n")
        outlier_table.to_csv(buffer, index=False)
    buffer.write("\nDecision Matrix\n")
    decision_matrix.to_csv(buffer, index=False)
    buffer.write("\nLoB Summary\n")
    lob_summary.to_csv(buffer, index=False)
    buffer.write("\nLoD Summary\n")
    lod_summary.to_csv(buffer, index=False)
    buffer.write("\nLoQ Summary\n")
    loq_summary.to_csv(buffer, index=False)
    buffer.write("\nAnalyzed Dataset\n")
    analyzed_data.to_csv(buffer, index=False)
    return buffer.getvalue()


def build_dbs_summary_csv(
    metadata: dict[str, object],
    criteria_table: pd.DataFrame,
    overall_summary: pd.DataFrame,
    study_summary: pd.DataFrame,
    bias_summary: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    agreement_summary: pd.DataFrame,
    hematocrit_summary: pd.DataFrame,
    delay_summary: pd.DataFrame,
    instrument_summary: pd.DataFrame,
    outlier_review: pd.DataFrame,
    scientific_findings: list[str],
    sample_review: pd.DataFrame,
) -> str:
    """Build a sectioned CSV summary for DBS validation documentation."""

    buffer = StringIO()
    pd.DataFrame(
        [{"Field": key, "Value": value or "Not specified"} for key, value in metadata.items()]
    ).to_csv(buffer, index=False)
    buffer.write("\nExecutive Summary\n")
    overall_summary.to_csv(buffer, index=False)
    buffer.write("\nAcceptance Criteria Results\n")
    criteria_table.to_csv(buffer, index=False)
    buffer.write("\nStudy Summary\n")
    study_summary.to_csv(buffer, index=False)
    buffer.write("\nBias Summary\n")
    bias_summary.to_csv(buffer, index=False)
    buffer.write("\nRecovery Summary\n")
    recovery_summary.to_csv(buffer, index=False)
    buffer.write("\nCorrelation Summary\n")
    correlation_summary.to_csv(buffer, index=False)
    buffer.write("\nAgreement Summary\n")
    agreement_summary.to_csv(buffer, index=False)
    if not hematocrit_summary.empty:
        buffer.write("\nHematocrit Impact Assessment\n")
        hematocrit_summary.to_csv(buffer, index=False)
    if not delay_summary.empty:
        buffer.write("\nExtraction Delay Assessment\n")
        delay_summary.to_csv(buffer, index=False)
    if not instrument_summary.empty:
        buffer.write("\nInstrument Assessment\n")
        instrument_summary.to_csv(buffer, index=False)
    if not outlier_review.empty:
        buffer.write("\nOutlier Investigation\n")
        outlier_review.to_csv(buffer, index=False)
    buffer.write("\nScientific Review Findings\n")
    pd.DataFrame({"Finding": scientific_findings}).to_csv(buffer, index=False)
    buffer.write("\nSample-Level Review\n")
    sample_review.to_csv(buffer, index=False)
    return buffer.getvalue()


def build_microtainer_summary_csv(
    metadata: dict[str, object],
    criteria_table: pd.DataFrame,
    overall_summary: pd.DataFrame,
    bias_summary: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    agreement_summary: pd.DataFrame,
    volume_summary: pd.DataFrame,
    delay_summary: pd.DataFrame,
    instrument_summary: pd.DataFrame,
    collection_site_summary: pd.DataFrame,
    outlier_review: pd.DataFrame,
    scientific_findings: list[str],
    sample_review: pd.DataFrame,
) -> str:
    """Build machine-readable Microtainer validation appendix CSV."""

    buffer = StringIO()
    pd.DataFrame([{"Field": key, "Value": value or "Not specified"} for key, value in metadata.items()]).to_csv(buffer, index=False)
    for title, table in [
        ("Executive Summary", overall_summary),
        ("Acceptance Criteria Results", criteria_table),
        ("Bias Analysis", bias_summary),
        ("Recovery Analysis", recovery_summary),
        ("Correlation Analysis", correlation_summary),
        ("Agreement Analysis", agreement_summary),
        ("Samples Requiring Review", outlier_review),
        ("Sample-Level Review", sample_review),
    ]:
        if not table.empty:
            buffer.write(f"\n{title}\n")
            table.to_csv(buffer, index=False)
    return buffer.getvalue()


def render_detection_workspace(metadata: dict[str, object]) -> None:
    """Render the Detection Capability workflow."""

    criteria = render_detection_criteria()
    uploaded_file = st.file_uploader(
        "Upload detection capability data",
        type=["csv", "xlsx", "xls"],
        help="Upload blank, low-concentration, and quantitation-level replicate results.",
    )
    use_sample_data = st.checkbox(
        "Use included sample HbA1c detection capability dataset",
        value=uploaded_file is None,
    )

    data = None
    if uploaded_file is not None:
        try:
            data = load_uploaded_file(uploaded_file)
        except Exception as exc:
            st.error(f"Unable to load file: {exc}")
            st.stop()
    elif use_sample_data:
        data = pd.read_csv(DETECTION_SAMPLE_DATA_PATH)

    if data is None:
        st.info("Upload a CSV or Excel file to begin.")
        st.stop()

    metadata["Sample Count"] = len(data)
    st.subheader("Data Preview")
    st.dataframe(data.head(25), width="stretch")

    numeric_columns = get_numeric_columns(data)
    all_columns = list(data.columns)
    if len(numeric_columns) < 2:
        st.error("Concentration and observed result numeric columns are required.")
        st.stop()

    st.subheader("Column Selection")
    first_row = st.columns(4)
    with first_row[0]:
        sample_id_column = st.selectbox(
            "Sample ID column",
            all_columns,
            index=all_columns.index("Sample ID") if "Sample ID" in all_columns else 0,
        )
    with first_row[1]:
        sample_type_column = st.selectbox(
            "Sample Type column",
            all_columns,
            index=all_columns.index("Sample Type") if "Sample Type" in all_columns else 0,
        )
    with first_row[2]:
        concentration_column = st.selectbox(
            "Concentration Level column",
            numeric_columns,
            index=numeric_columns.index("Concentration Level")
            if "Concentration Level" in numeric_columns
            else 0,
        )
    with first_row[3]:
        result_column = st.selectbox(
            "Observed Result column",
            numeric_columns,
            index=numeric_columns.index("Observed Result")
            if "Observed Result" in numeric_columns
            else min(1, len(numeric_columns) - 1),
        )

    second_row = st.columns(3)
    with second_row[0]:
        replicate_column = optional_column_selectbox(
            "Replicate column",
            all_columns,
            "Replicate" if "Replicate" in all_columns else None,
        )
    with second_row[1]:
        units_column = optional_column_selectbox(
            "Units column",
            all_columns,
            "Units" if "Units" in all_columns else None,
        )
    with second_row[2]:
        include_column = optional_column_selectbox(
            "Include in Analysis column",
            all_columns,
            "Include in Analysis" if "Include in Analysis" in all_columns else None,
        )

    run_analysis = st.button("Run Detection Capability Analysis", type="primary")
    if run_analysis:
        try:
            result = run_detection_capability_study(
                data=data,
                sample_id_column=sample_id_column,
                sample_type_column=sample_type_column,
                concentration_column=concentration_column,
                result_column=result_column,
                replicate_column=replicate_column,
                units_column=units_column,
                include_column=include_column,
                target_loq_cv=criteria["Target LoQ CV%"],
            )
        except Exception as exc:
            st.error(f"Detection capability analysis could not be completed: {exc}")
            st.stop()

        criteria_result = evaluate_detection_capability_criteria(
            result.overall_summary,
            max_lob=criteria["Maximum LoB"],
            max_lod=criteria["Maximum LoD"],
            target_loq_cv=criteria["Target LoQ CV%"],
            max_loq_concentration=criteria["Maximum LoQ Concentration"],
            borderline_zone=criteria["Borderline Zone"],
        )
        decision = str(criteria_result["decision"])
        criteria_table = build_criteria_table(criteria_result)
        decision_matrix = build_detection_decision_matrix(criteria_result)
        display_criteria = format_detection_criteria_table(criteria_table)
        display_overall = format_detection_overall_summary(result.overall_summary)
        display_methodology = format_detection_table(result.methodology_table)
        display_quality = format_detection_table(result.data_quality_summary)
        display_outliers = format_detection_table(result.outlier_table)
        display_decision_matrix = format_detection_table(decision_matrix)
        display_lob = format_detection_table(result.lob_summary)
        display_lod = format_detection_table(result.lod_summary)
        display_loq = format_detection_table(result.loq_summary)
        display_analyzed = format_detection_table(result.analyzed_data)
        interpretation = generate_detection_interpretation(
            result.overall_summary,
            result.data_quality_summary,
            criteria,
            decision,
            metadata,
        )

        render_detection_executive_summary(
            result.overall_summary, decision, display_criteria
        )
        st.subheader("Detection Capability Summary")
        show_decision(decision)
        st.dataframe(display_overall, width="stretch")

        st.subheader("Acceptance Criteria Results")
        render_badged_criteria_table(display_criteria)

        st.subheader("Calculation Methodology")
        st.dataframe(display_methodology, width="stretch")

        st.subheader("Data Quality Assessment")
        st.dataframe(display_quality, width="stretch")
        for _, quality_row in result.data_quality_summary.iterrows():
            if str(quality_row["Status"]).upper() != "PASS":
                st.warning(
                    f"{quality_row['Check']}: observed {quality_row['Observed']}. "
                    f"{quality_row['Recommendation']}."
                )
        if result.outlier_table.empty:
            st.info("No IQR outliers were detected.")
        else:
            st.markdown("**IQR Outlier Review**")
            st.dataframe(display_outliers, width="stretch")

        st.subheader("Detection Capability Decision Matrix")
        st.dataframe(display_decision_matrix, width="stretch")

        lob_col, lod_col = st.columns(2)
        with lob_col:
            st.subheader("LoB Summary")
            st.dataframe(display_lob, width="stretch")
        with lod_col:
            st.subheader("LoD Summary")
            st.dataframe(display_lod, width="stretch")

        st.subheader("LoQ Summary")
        st.dataframe(display_loq, width="stretch")

        st.subheader("Interpretation")
        st.info(interpretation)

        st.subheader("Visualizations")
        blank_histogram = create_blank_distribution_histogram(result.analyzed_data)
        blank_boxplot = create_blank_replicate_boxplot(result.analyzed_data)
        low_distribution = create_low_level_distribution_plot(result.analyzed_data)
        lob_lod_plot = create_lob_lod_visualization(
            result.lob_summary, result.lod_summary
        )
        cv_plot = create_loq_cv_plot(result.loq_summary, criteria["Target LoQ CV%"])
        recovery_plot = create_loq_recovery_plot(result.loq_summary)
        decision_plot = create_loq_decision_plot(
            result.loq_summary, criteria["Target LoQ CV%"]
        )
        replicate_plot = create_detection_replicate_distribution_plot(
            result.analyzed_data
        )
        replicate_scatter = create_detection_replicate_scatter_plot(result.analyzed_data)
        precision_curve = create_loq_precision_curve(
            result.loq_summary,
            criteria["Target LoQ CV%"],
            result.overall_summary["LoQ"],
        )
        ladder_plot = create_detection_capability_ladder(result.overall_summary)
        density_plot = create_detection_density_plot(
            result.analyzed_data,
            result.overall_summary["LoB"],
            result.overall_summary["LoD"],
        )

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(blank_histogram, width="stretch")
        with chart_right:
            st.plotly_chart(blank_boxplot, width="stretch")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(low_distribution, width="stretch")
        with chart_right:
            st.plotly_chart(lob_lod_plot, width="stretch")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(cv_plot, width="stretch")
        with chart_right:
            st.plotly_chart(recovery_plot, width="stretch")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(decision_plot, width="stretch")
        with chart_right:
            st.plotly_chart(replicate_plot, width="stretch")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(replicate_scatter, width="stretch")
        with chart_right:
            st.plotly_chart(precision_curve, width="stretch")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(ladder_plot, width="stretch")
        with chart_right:
            st.plotly_chart(density_plot, width="stretch")

        st.subheader("Analyzed Dataset")
        st.dataframe(display_analyzed, width="stretch")
        render_submit_for_review("Detection Capability", metadata)

        html_report = build_detection_html_report(
            lob_summary=result.lob_summary,
            lod_summary=result.lod_summary,
            loq_summary=result.loq_summary,
            methodology_table=result.methodology_table,
            data_quality_summary=result.data_quality_summary,
            outlier_table=result.outlier_table,
            decision_matrix=decision_matrix,
            analyzed_data=result.analyzed_data,
            overall_summary=result.overall_summary,
            criteria_table=criteria_table,
            interpretation=interpretation,
            metadata=metadata,
            criteria=criteria,
            decision=decision,
            visualization_html={
                "Blank Distribution Histogram": blank_histogram.to_html(
                    full_html=False, include_plotlyjs="cdn"
                ),
                "Blank Replicate Boxplot": blank_boxplot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Low-Level Replicate Distribution": low_distribution.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "LoB vs LoD Visualization": lob_lod_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "CV% vs Concentration": cv_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Recovery vs Concentration": recovery_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "LoQ Decision Plot": decision_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Replicate Distribution by Concentration": replicate_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Replicate Scatter Plot": replicate_scatter.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "LoQ Precision Curve": precision_curve.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Detection Capability Ladder": ladder_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Distribution Density Plot": density_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
            },
        )

        pdf_report = build_detection_pdf_report(
            lob_summary=result.lob_summary,
            lod_summary=result.lod_summary,
            loq_summary=result.loq_summary,
            methodology_table=result.methodology_table,
            data_quality_summary=result.data_quality_summary,
            outlier_table=result.outlier_table,
            decision_matrix=decision_matrix,
            overall_summary=result.overall_summary,
            criteria_table=criteria_table,
            interpretation=interpretation,
            metadata=metadata,
            criteria=criteria,
            decision=decision,
        )

        export_left, export_middle, export_right = st.columns(3)
        with export_left:
            st.download_button(
                "Download detection capability summary CSV",
                data=build_detection_summary_csv(
                    metadata,
                    display_criteria,
                    display_overall,
                    display_methodology,
                    display_quality,
                    display_outliers,
                    display_decision_matrix,
                    display_lob,
                    display_lod,
                    display_loq,
                    display_analyzed,
                ).encode("utf-8"),
                file_name="detection_capability_summary.csv",
                mime="text/csv",
            )
        with export_middle:
            st.download_button(
                "Download detection capability report HTML",
                data=html_report.encode("utf-8"),
                file_name="detection_capability_report.html",
                mime="text/html",
            )
        with export_right:
            st.download_button(
                "Download detection capability report PDF",
                data=pdf_report,
                file_name="detection_capability_report.pdf",
                mime="application/pdf",
            )


def render_dbs_workspace(metadata: dict[str, object]) -> None:
    """Render the DBS Validation workflow."""

    criteria = render_dbs_criteria()
    uploaded_file = st.file_uploader(
        "Upload DBS validation data",
        type=["csv", "xlsx", "xls"],
        help="Upload paired DBS and reference specimen results.",
    )
    use_sample_data = st.checkbox(
        "Use included sample HbA1c DBS validation dataset",
        value=uploaded_file is None,
    )

    data = None
    if uploaded_file is not None:
        try:
            data = load_uploaded_file(uploaded_file)
        except Exception as exc:
            st.error(f"Unable to load file: {exc}")
            st.stop()
    elif use_sample_data:
        data = pd.read_csv(DBS_SAMPLE_DATA_PATH)

    if data is None:
        st.info("Upload a CSV or Excel file to begin.")
        st.stop()

    metadata["Sample Count"] = len(data)
    st.subheader("Data Preview")
    st.dataframe(data.head(25), width="stretch")

    numeric_columns = get_numeric_columns(data)
    all_columns = list(data.columns)
    if len(numeric_columns) < 2:
        st.error("Reference and DBS numeric result columns are required.")
        st.stop()

    st.subheader("Column Selection")
    first_row = st.columns(3)
    with first_row[0]:
        sample_id_column = st.selectbox(
            "Sample ID column",
            all_columns,
            index=all_columns.index("Sample ID") if "Sample ID" in all_columns else 0,
        )
    with first_row[1]:
        reference_column = st.selectbox(
            "Reference Result column",
            numeric_columns,
            index=numeric_columns.index("Reference Result")
            if "Reference Result" in numeric_columns
            else 0,
        )
    with first_row[2]:
        dbs_column = st.selectbox(
            "DBS Result column",
            numeric_columns,
            index=numeric_columns.index("DBS Result")
            if "DBS Result" in numeric_columns
            else min(1, len(numeric_columns) - 1),
        )

    second_row = st.columns(4)
    with second_row[0]:
        collection_column = optional_column_selectbox(
            "Collection Date column",
            all_columns,
            "Collection Date" if "Collection Date" in all_columns else None,
        )
    with second_row[1]:
        extraction_column = optional_column_selectbox(
            "Extraction Date column",
            all_columns,
            "Extraction Date" if "Extraction Date" in all_columns else None,
        )
    with second_row[2]:
        hematocrit_column = optional_column_selectbox(
            "Hematocrit column",
            all_columns,
            "Hematocrit" if "Hematocrit" in all_columns else None,
        )
    with second_row[3]:
        include_column = optional_column_selectbox(
            "Include in Analysis column",
            all_columns,
            "Include in Analysis" if "Include in Analysis" in all_columns else None,
        )

    third_row = st.columns(2)
    with third_row[0]:
        replicate_column = optional_column_selectbox(
            "Replicate column",
            all_columns,
            "Replicate" if "Replicate" in all_columns else None,
        )
    with third_row[1]:
        instrument_column = optional_column_selectbox(
            "Instrument column",
            all_columns,
            "Instrument" if "Instrument" in all_columns else None,
        )
    _ = (collection_column, extraction_column, hematocrit_column, replicate_column, instrument_column)

    if reference_column == dbs_column:
        st.warning("Select different reference and DBS result columns.")

    run_analysis = st.button("Run DBS Validation Analysis", type="primary")
    if run_analysis:
        if reference_column == dbs_column:
            st.stop()
        try:
            result = run_dbs_validation_study(
                data=data,
                sample_id_column=sample_id_column,
                reference_column=reference_column,
                dbs_column=dbs_column,
                include_column=include_column,
                max_percent_bias=criteria["Maximum Percent Bias"],
                min_recovery=criteria["Minimum Recovery"],
                max_recovery=criteria["Maximum Recovery"],
            )
        except Exception as exc:
            st.error(f"DBS validation analysis could not be completed: {exc}")
            st.stop()

        criteria_result = evaluate_dbs_criteria(
            result.overall_summary,
            max_percent_bias=criteria["Maximum Percent Bias"],
            min_recovery=criteria["Minimum Recovery"],
            max_recovery=criteria["Maximum Recovery"],
            min_r_squared=criteria["Minimum R²"],
            max_mean_difference=criteria["Maximum Mean Difference"],
            borderline_zone=criteria["Borderline Zone"],
        )
        decision = str(criteria_result["decision"])
        criteria_table = build_criteria_table(criteria_result)
        display_criteria = format_dbs_criteria_table(criteria_table)
        display_overall = format_dbs_overall_summary(result.overall_summary)
        display_study = format_dbs_table(result.study_summary)
        display_bias = format_dbs_table(result.bias_summary)
        display_recovery = format_dbs_table(result.recovery_summary)
        display_correlation = format_dbs_table(result.correlation_summary)
        display_agreement = format_dbs_table(result.agreement_summary)
        display_hematocrit = format_dbs_table(result.hematocrit_summary)
        display_delay = format_dbs_table(result.delay_summary)
        display_instrument = format_dbs_table(result.instrument_summary)
        display_outlier_review = format_dbs_table(result.outlier_review)
        display_sample_review = format_dbs_table(result.sample_review)
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

        summary_values = build_dbs_executive_summary(
            result.overall_summary, decision, display_criteria
        )
        st.subheader("DBS Executive Summary")
        for row_start in range(0, len(summary_values), 4):
            columns = st.columns(4)
            for column, (label, value) in zip(
                columns, list(summary_values.items())[row_start : row_start + 4]
            ):
                with column:
                    render_metric_card(
                        label,
                        value,
                        status=value if label == "Overall Decision" else None,
                    )

        st.subheader("DBS Validation Summary")
        show_decision(decision)
        st.dataframe(display_overall, width="stretch")
        st.dataframe(display_study, width="stretch")

        st.subheader("Acceptance Criteria Results")
        render_badged_criteria_table(display_criteria)

        summary_left, summary_right = st.columns(2)
        with summary_left:
            st.subheader("Bias Summary")
            st.dataframe(display_bias, width="stretch")
            st.subheader("Correlation Summary")
            st.dataframe(display_correlation, width="stretch")
        with summary_right:
            st.subheader("Recovery Summary")
            st.dataframe(display_recovery, width="stretch")
            st.subheader("Agreement Summary")
            st.dataframe(display_agreement, width="stretch")

        st.subheader("Sample-Level Review")
        st.dataframe(display_sample_review.head(10), width="stretch")

        st.subheader("Hematocrit Impact Assessment")
        if result.hematocrit_summary.empty:
            st.info("Hematocrit impact assessment was not available for this dataset.")
        else:
            hct_status = (
                "PASS"
                if not result.hematocrit_summary["Status"].isin(["REVIEW", "INVESTIGATE"]).any()
                else "REVIEW"
            )
            show_decision(hct_status)
            st.dataframe(display_hematocrit, width="stretch")

        st.subheader("Extraction Delay Assessment")
        if result.delay_summary.empty:
            st.info("Extraction delay assessment was not available for this dataset.")
        else:
            st.dataframe(display_delay, width="stretch")

        st.subheader("Instrument Comparison Assessment")
        if result.instrument_summary.empty:
            st.info("Instrument comparison was not available for this dataset.")
        else:
            st.dataframe(display_instrument, width="stretch")

        st.subheader("Outlier Investigation Dashboard")
        if result.outlier_review.empty:
            st.info("No outlier investigation findings were available.")
        else:
            st.dataframe(display_outlier_review, width="stretch")

        st.subheader("Scientific Review Findings")
        for finding in scientific_findings:
            st.info(finding)

        st.subheader("Scientific Interpretation")
        st.info(interpretation)

        st.subheader("Visualizations")
        scatter_plot = create_dbs_scatter_plot(
            result.analyzed_data, result.overall_summary
        )
        bland_altman_plot = create_dbs_bland_altman_plot(
            result.analyzed_data, result.overall_summary
        )
        recovery_plot = create_dbs_recovery_plot(
            result.analyzed_data,
            criteria["Minimum Recovery"],
            criteria["Maximum Recovery"],
        )
        percent_bias_plot = create_dbs_percent_bias_plot(
            result.analyzed_data, criteria["Maximum Percent Bias"]
        )
        distribution_plot = create_dbs_distribution_comparison(result.analyzed_data)
        hematocrit_bias_plot = (
            create_dbs_hematocrit_bias_plot(result.analyzed_data)
            if "Hematocrit" in result.analyzed_data.columns
            else None
        )
        hematocrit_percent_bias_plot = (
            create_dbs_hematocrit_percent_bias_plot(result.analyzed_data)
            if "Hematocrit" in result.analyzed_data.columns
            else None
        )
        delay_bias_plot = (
            create_dbs_delay_bias_plot(result.analyzed_data)
            if "Extraction Delay (days)" in result.analyzed_data.columns
            else None
        )
        delay_distribution_plot = (
            create_dbs_delay_distribution_plot(result.analyzed_data)
            if "Extraction Delay (days)" in result.analyzed_data.columns
            else None
        )
        delay_category_plot = (
            create_dbs_delay_category_bias_plot(result.analyzed_data)
            if "Delay Category" in result.analyzed_data.columns
            else None
        )
        instrument_bias_plot = (
            create_dbs_instrument_bias_plot(result.instrument_summary)
            if not result.instrument_summary.empty
            else None
        )
        instrument_recovery_plot = (
            create_dbs_instrument_recovery_plot(result.instrument_summary)
            if not result.instrument_summary.empty
            else None
        )
        st.plotly_chart(scatter_plot, width="stretch")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(bland_altman_plot, width="stretch")
        with chart_right:
            st.plotly_chart(distribution_plot, width="stretch")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(recovery_plot, width="stretch")
        with chart_right:
            st.plotly_chart(percent_bias_plot, width="stretch")
        if hematocrit_bias_plot is not None and hematocrit_percent_bias_plot is not None:
            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.plotly_chart(hematocrit_bias_plot, width="stretch")
            with chart_right:
                st.plotly_chart(hematocrit_percent_bias_plot, width="stretch")
        if delay_bias_plot is not None and delay_distribution_plot is not None:
            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.plotly_chart(delay_bias_plot, width="stretch")
            with chart_right:
                st.plotly_chart(delay_distribution_plot, width="stretch")
        if delay_category_plot is not None:
            st.plotly_chart(delay_category_plot, width="stretch")
        if instrument_bias_plot is not None and instrument_recovery_plot is not None:
            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.plotly_chart(instrument_bias_plot, width="stretch")
            with chart_right:
                st.plotly_chart(instrument_recovery_plot, width="stretch")

        st.subheader("Analyzed Dataset")
        st.dataframe(format_dbs_table(result.analyzed_data), width="stretch")
        render_submit_for_review("DBS Validation", metadata)

        html_report = build_dbs_html_report(
            study_summary=result.study_summary,
            bias_summary=result.bias_summary,
            recovery_summary=result.recovery_summary,
            correlation_summary=result.correlation_summary,
            agreement_summary=result.agreement_summary,
            hematocrit_summary=result.hematocrit_summary,
            delay_summary=result.delay_summary,
            instrument_summary=result.instrument_summary,
            outlier_review=result.outlier_review,
            scientific_findings=scientific_findings,
            sample_review=result.sample_review,
            overall_summary=result.overall_summary,
            criteria_table=criteria_table,
            interpretation=interpretation,
            metadata=metadata,
            criteria=criteria,
            decision=decision,
            visualization_html={
                "DBS vs Reference Scatter Plot": scatter_plot.to_html(
                    full_html=False, include_plotlyjs="cdn"
                ),
                "Bland-Altman Plot": bland_altman_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Recovery Plot": recovery_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Percent Bias Plot": percent_bias_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Distribution Comparison": distribution_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                **(
                    {
                        "Hematocrit vs Bias Plot": hematocrit_bias_plot.to_html(
                            full_html=False, include_plotlyjs=False
                        ),
                        "Hematocrit vs Percent Bias Plot": hematocrit_percent_bias_plot.to_html(
                            full_html=False, include_plotlyjs=False
                        ),
                    }
                    if hematocrit_bias_plot is not None and hematocrit_percent_bias_plot is not None
                    else {}
                ),
                **(
                    {
                        "Delay vs Bias Plot": delay_bias_plot.to_html(
                            full_html=False, include_plotlyjs=False
                        ),
                        "Delay Distribution Histogram": delay_distribution_plot.to_html(
                            full_html=False, include_plotlyjs=False
                        ),
                        "Average Bias by Delay Category": delay_category_plot.to_html(
                            full_html=False, include_plotlyjs=False
                        ),
                    }
                    if delay_bias_plot is not None
                    and delay_distribution_plot is not None
                    and delay_category_plot is not None
                    else {}
                ),
                **(
                    {
                        "Bias by Instrument Plot": instrument_bias_plot.to_html(
                            full_html=False, include_plotlyjs=False
                        ),
                        "Recovery by Instrument Plot": instrument_recovery_plot.to_html(
                            full_html=False, include_plotlyjs=False
                        ),
                    }
                    if instrument_bias_plot is not None and instrument_recovery_plot is not None
                    else {}
                ),
            },
        )
        pdf_report = build_dbs_pdf_report(
            study_summary=result.study_summary,
            bias_summary=result.bias_summary,
            recovery_summary=result.recovery_summary,
            correlation_summary=result.correlation_summary,
            agreement_summary=result.agreement_summary,
            hematocrit_summary=result.hematocrit_summary,
            delay_summary=result.delay_summary,
            instrument_summary=result.instrument_summary,
            outlier_review=result.outlier_review,
            scientific_findings=scientific_findings,
            sample_review=result.sample_review,
            overall_summary=result.overall_summary,
            criteria_table=criteria_table,
            interpretation=interpretation,
            metadata=metadata,
            criteria=criteria,
            decision=decision,
        )

        export_left, export_middle, export_right = st.columns(3)
        with export_left:
            st.download_button(
                "Download DBS summary CSV",
                data=build_dbs_summary_csv(
                    metadata,
                    display_criteria,
                    display_overall,
                    display_study,
                    display_bias,
                    display_recovery,
                    display_correlation,
                    display_agreement,
                    display_hematocrit,
                    display_delay,
                    display_instrument,
                    display_outlier_review,
                    scientific_findings,
                    display_sample_review,
                ).encode("utf-8"),
                file_name="dbs_validation_summary.csv",
                mime="text/csv",
            )
        with export_middle:
            st.download_button(
                "Generate DBS Validation Report HTML",
                data=html_report.encode("utf-8"),
                file_name="dbs_validation_report.html",
                mime="text/html",
            )
        with export_right:
            st.download_button(
                "Generate DBS Validation Report PDF",
                data=pdf_report,
                file_name="dbs_validation_report.pdf",
                mime="application/pdf",
            )


def render_microtainer_workspace(metadata: dict[str, object]) -> None:
    """Render the Microtainer Validation workflow."""

    criteria = render_microtainer_criteria()
    uploaded_file = st.file_uploader(
        "Upload microtainer validation data",
        type=["csv", "xlsx", "xls"],
        help="Upload paired microtainer and reference specimen results. Maximum intended file size: 200 MB.",
    )
    use_sample_data = st.checkbox(
        "Use included sample HbA1c microtainer validation dataset",
        value=uploaded_file is None,
    )
    data = None
    if uploaded_file is not None:
        try:
            data = load_uploaded_file(uploaded_file)
        except Exception as exc:
            st.error(f"Unable to load file: {exc}")
            st.stop()
    elif use_sample_data:
        data = pd.read_csv(MICROTAINER_SAMPLE_DATA_PATH)
    if data is None:
        st.info("Upload a CSV or Excel file to begin.")
        st.stop()

    metadata["Sample Count"] = len(data)
    st.subheader("Data Preview")
    st.dataframe(data.head(25), width="stretch")

    numeric_columns = get_numeric_columns(data)
    all_columns = list(data.columns)
    if len(numeric_columns) < 2:
        st.error("Reference and microtainer numeric result columns are required.")
        st.stop()

    st.subheader("Column Selection")
    first_row = st.columns(3)
    with first_row[0]:
        sample_id_column = st.selectbox("Sample ID column", all_columns, index=all_columns.index("Sample ID") if "Sample ID" in all_columns else 0)
    with first_row[1]:
        reference_column = st.selectbox("Reference Result column", numeric_columns, index=numeric_columns.index("Reference Result") if "Reference Result" in numeric_columns else 0)
    with first_row[2]:
        microtainer_column = st.selectbox("Microtainer Result column", numeric_columns, index=numeric_columns.index("Microtainer Result") if "Microtainer Result" in numeric_columns else min(1, len(numeric_columns) - 1))

    second_row = st.columns(4)
    with second_row[0]:
        collection_date_column = optional_column_selectbox("Collection Date column", all_columns, "Collection Date" if "Collection Date" in all_columns else None)
    with second_row[1]:
        processing_date_column = optional_column_selectbox("Processing Date column", all_columns, "Processing Date" if "Processing Date" in all_columns else None)
    with second_row[2]:
        instrument_column = optional_column_selectbox("Instrument column", all_columns, "Instrument" if "Instrument" in all_columns else None)
    with second_row[3]:
        include_column = optional_column_selectbox("Include in Analysis column", all_columns, "Include in Analysis" if "Include in Analysis" in all_columns else None)
    third_row = st.columns(4)
    with third_row[0]:
        replicate_column = optional_column_selectbox("Replicate column", all_columns, "Replicate" if "Replicate" in all_columns else None)
    with third_row[1]:
        operator_column = optional_column_selectbox("Operator column", all_columns, "Operator" if "Operator" in all_columns else None)
    with third_row[2]:
        collection_site_column = optional_column_selectbox("Collection Site column", all_columns, "Collection Site" if "Collection Site" in all_columns else None)
    with third_row[3]:
        notes_column = optional_column_selectbox("Notes column", all_columns, "Notes" if "Notes" in all_columns else None)
    _ = (collection_date_column, processing_date_column, instrument_column, replicate_column, operator_column, collection_site_column, notes_column)

    if reference_column == microtainer_column:
        st.warning("Select different reference and microtainer result columns.")

    run_analysis = st.button("Run Microtainer Validation Analysis", type="primary")
    if run_analysis:
        if reference_column == microtainer_column:
            st.stop()
        try:
            result = run_microtainer_validation_study(
                data=data,
                sample_id_column=sample_id_column,
                reference_column=reference_column,
                microtainer_column=microtainer_column,
                include_column=include_column,
                max_percent_bias=float(criteria["Maximum Absolute Percent Bias"]),
                min_recovery=float(criteria["Recovery Lower Limit"]),
                max_recovery=float(criteria["Recovery Upper Limit"]),
            )
        except Exception as exc:
            st.error(f"Microtainer validation analysis could not be completed: {exc}")
            st.stop()

        criteria_result = evaluate_microtainer_criteria(
            result.overall_summary,
            max_percent_bias=float(criteria["Maximum Absolute Percent Bias"]),
            min_recovery=float(criteria["Recovery Lower Limit"]),
            max_recovery=float(criteria["Recovery Upper Limit"]),
            min_r_squared=float(criteria["Minimum R²"]),
            max_mean_difference=float(criteria["Maximum Mean Difference"]),
            borderline_zone=float(criteria["Borderline Zone"]),
        )
        raw_decision = str(criteria_result["decision"])
        decision = "PASS" if raw_decision == "PASS" else "FAIL"
        criteria_table = build_criteria_table(criteria_result)
        display_criteria = format_microtainer_criteria_table(criteria_table)
        display_overall = format_microtainer_overall_summary(result.overall_summary)
        display_bias = format_microtainer_table(result.bias_summary)
        display_recovery = format_microtainer_table(result.recovery_summary)
        display_correlation = format_microtainer_table(result.correlation_summary)
        display_agreement = format_microtainer_table(result.agreement_summary)
        display_outliers = format_microtainer_table(result.outlier_review)
        display_sample_review = format_microtainer_table(result.sample_review)
        interpretation = generate_microtainer_interpretation(
            result.overall_summary, dict(criteria), decision, metadata
        )

        st.subheader("Microtainer Executive Summary")
        summary_values = build_microtainer_executive_summary(result.overall_summary, decision, display_criteria)
        for row_start in range(0, len(summary_values), 4):
            columns = st.columns(4)
            for column, (label, value) in zip(columns, list(summary_values.items())[row_start : row_start + 4]):
                with column:
                    render_metric_card(label, value, status=value if label == "Overall Decision" else None)

        st.subheader("Microtainer Validation Summary")
        show_decision(decision)
        st.dataframe(display_overall, width="stretch")

        st.subheader("Acceptance Criteria Results")
        render_badged_criteria_table(display_criteria)

        left, right = st.columns(2)
        with left:
            st.subheader("Bias Analysis")
            st.dataframe(display_bias, width="stretch")
            st.subheader("Correlation Analysis")
            st.dataframe(display_correlation, width="stretch")
        with right:
            st.subheader("Recovery Analysis")
            st.dataframe(display_recovery, width="stretch")
            st.subheader("Agreement Analysis")
            st.dataframe(display_agreement, width="stretch")

        st.subheader("Sample-Level Review")
        st.dataframe(display_sample_review.head(10), width="stretch")
        st.subheader("Samples Requiring Review")
        if result.outlier_review.empty:
            st.info("No samples required reviewer attention.")
        else:
            st.dataframe(display_outliers, width="stretch")

        scatter_plot = create_microtainer_scatter_plot(result.analyzed_data, result.overall_summary)
        bland_altman_plot = create_microtainer_bland_altman_plot(result.analyzed_data, result.overall_summary)
        recovery_plot = create_microtainer_recovery_plot(result.analyzed_data, float(criteria["Recovery Lower Limit"]), float(criteria["Recovery Upper Limit"]))
        percent_bias_plot = create_microtainer_percent_bias_plot(result.analyzed_data, float(criteria["Maximum Absolute Percent Bias"]))
        distribution_plot = create_microtainer_distribution_comparison(result.analyzed_data)

        st.subheader("Visualizations")
        st.plotly_chart(scatter_plot, width="stretch")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(bland_altman_plot, width="stretch")
        with chart_right:
            st.plotly_chart(distribution_plot, width="stretch")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(recovery_plot, width="stretch")
        with chart_right:
            st.plotly_chart(percent_bias_plot, width="stretch")

        st.subheader("Analyzed Dataset")
        st.dataframe(format_microtainer_table(result.analyzed_data), width="stretch")
        st.subheader("Final Validation Decision")
        st.info(interpretation)
        render_submit_for_review("Microtainer Validation", metadata)

        visualization_html = {
            "Microtainer vs Reference Scatter Plot": scatter_plot.to_html(full_html=False, include_plotlyjs="cdn"),
            "Bland-Altman Plot": bland_altman_plot.to_html(full_html=False, include_plotlyjs=False),
            "Recovery by Sample": recovery_plot.to_html(full_html=False, include_plotlyjs=False),
            "Percent Bias by Sample": percent_bias_plot.to_html(full_html=False, include_plotlyjs=False),
            "Distribution Comparison": distribution_plot.to_html(full_html=False, include_plotlyjs=False),
        }

        html_report = build_microtainer_html_report(
            result.study_summary,
            result.bias_summary,
            result.recovery_summary,
            result.correlation_summary,
            result.agreement_summary,
            result.volume_summary,
            result.delay_summary,
            result.instrument_summary,
            result.collection_site_summary,
            result.outlier_review,
            [],
            result.sample_review,
            result.overall_summary,
            criteria_table,
            interpretation,
            metadata,
            dict(criteria),
            decision,
            visualization_html,
        )
        pdf_report = build_microtainer_pdf_report(
            result.study_summary,
            result.bias_summary,
            result.recovery_summary,
            result.correlation_summary,
            result.agreement_summary,
            result.volume_summary,
            result.delay_summary,
            result.instrument_summary,
            result.collection_site_summary,
            result.outlier_review,
            [],
            result.sample_review,
            result.overall_summary,
            criteria_table,
            interpretation,
            metadata,
            dict(criteria),
            decision,
        )
        summary_csv = build_microtainer_summary_csv(
            metadata,
            display_criteria,
            display_overall,
            display_bias,
            display_recovery,
            display_correlation,
            display_agreement,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            display_outliers,
            [],
            display_sample_review,
        )

        export_left, export_middle, export_right = st.columns(3)
        with export_left:
            st.download_button("Download regulatory appendix CSV", data=summary_csv.encode("utf-8"), file_name="microtainer_validation_appendix.csv", mime="text/csv")
        with export_middle:
            st.download_button("Download Microtainer validation report HTML", data=html_report.encode("utf-8"), file_name="microtainer_validation_report.html", mime="text/html")
        with export_right:
            st.download_button("Download Microtainer validation report PDF", data=pdf_report, file_name="microtainer_validation_report.pdf", mime="application/pdf")


def render_accuracy_workspace(metadata: dict[str, object]) -> None:
    """Render the Accuracy Study workflow."""

    criteria = render_accuracy_criteria()

    uploaded_file = st.file_uploader(
        "Upload accuracy study data",
        type=["csv", "xlsx", "xls"],
        help="Upload observed and expected results across assigned-value levels.",
    )
    use_sample_data = st.checkbox(
        "Use included sample HbA1c accuracy study dataset",
        value=uploaded_file is None,
    )

    data = None
    if uploaded_file is not None:
        try:
            data = load_uploaded_file(uploaded_file)
        except Exception as exc:
            st.error(f"Unable to load file: {exc}")
            st.stop()
    elif use_sample_data:
        data = pd.read_csv(ACCURACY_SAMPLE_DATA_PATH)

    if data is None:
        st.info("Upload a CSV or Excel file to begin.")
        st.stop()

    metadata["Sample Count"] = len(data)

    st.subheader("Data Preview")
    st.dataframe(data.head(20), width="stretch")

    numeric_columns = get_numeric_columns(data)
    all_columns = list(data.columns)
    if len(numeric_columns) < 2:
        st.error("Expected and observed numeric result columns are required.")
        st.stop()

    st.subheader("Column Selection")
    first_row = st.columns(4)
    with first_row[0]:
        sample_id_column = st.selectbox(
            "Sample ID column",
            all_columns,
            index=all_columns.index("Sample ID") if "Sample ID" in all_columns else 0,
        )
    with first_row[1]:
        level_column = st.selectbox(
            "Level column",
            all_columns,
            index=all_columns.index("Level") if "Level" in all_columns else 0,
        )
    with first_row[2]:
        expected_column = st.selectbox(
            "Expected Result column",
            numeric_columns,
            index=numeric_columns.index("Expected Result")
            if "Expected Result" in numeric_columns
            else 0,
        )
    with first_row[3]:
        observed_column = st.selectbox(
            "Observed Result column",
            numeric_columns,
            index=numeric_columns.index("Observed Result")
            if "Observed Result" in numeric_columns
            else min(1, len(numeric_columns) - 1),
        )

    second_row = st.columns(3)
    with second_row[0]:
        units_column = optional_column_selectbox(
            "Units column",
            all_columns,
            "Units" if "Units" in all_columns else None,
        )
    with second_row[1]:
        replicate_column = optional_column_selectbox(
            "Replicate column",
            all_columns,
            "Replicate" if "Replicate" in all_columns else None,
        )
    with second_row[2]:
        include_column = optional_column_selectbox(
            "Include in Analysis column",
            all_columns,
            "Include in Analysis" if "Include in Analysis" in all_columns else None,
        )

    if expected_column == observed_column:
        st.warning("Select different expected and observed result columns.")

    data_quality_warnings = assess_accuracy_data_quality(
        data=data,
        sample_id_column=sample_id_column,
        level_column=level_column,
        expected_column=expected_column,
        observed_column=observed_column,
        include_column=include_column,
    )
    st.subheader("Data Quality Review")
    if data_quality_warnings:
        for warning in data_quality_warnings:
            st.warning(warning)
    else:
        st.success("No automatic data quality warnings were detected for the selected columns.")

    run_analysis = st.button("Run Accuracy Analysis", type="primary")
    if run_analysis:
        if expected_column == observed_column:
            st.stop()

        try:
            result = run_accuracy_study(
                data=data,
                sample_id_column=sample_id_column,
                level_column=level_column,
                expected_column=expected_column,
                observed_column=observed_column,
                units_column=units_column,
                replicate_column=replicate_column,
                include_column=include_column,
                max_abs_bias=criteria["Maximum Absolute Bias"],
                max_abs_percent_bias=criteria["Maximum Absolute Percent Bias"],
                min_recovery=criteria["Minimum Recovery"],
                max_recovery=criteria["Maximum Recovery"],
                borderline_zone=criteria["Borderline Zone"],
            )
        except Exception as exc:
            st.error(f"Accuracy analysis could not be completed: {exc}")
            st.stop()

        criteria_result = evaluate_accuracy_criteria(
            result.overall_summary,
            max_abs_bias=criteria["Maximum Absolute Bias"],
            max_abs_percent_bias=criteria["Maximum Absolute Percent Bias"],
            min_recovery=criteria["Minimum Recovery"],
            max_recovery=criteria["Maximum Recovery"],
            borderline_zone=criteria["Borderline Zone"],
        )
        decision = str(criteria_result["decision"])
        criteria_table = build_criteria_table(criteria_result)
        display_criteria = format_accuracy_criteria_table(criteria_table)
        display_overall = format_accuracy_overall_summary(result.overall_summary)
        display_accuracy = format_accuracy_table(result.accuracy_summary)
        display_level_decisions = format_accuracy_level_decision_table(
            result.level_decision_table
        )
        display_bias = format_accuracy_table(result.bias_summary)
        display_recovery = format_accuracy_table(result.recovery_summary)
        display_worst_case = format_accuracy_worst_case_summary(
            result.worst_case_summary
        )
        interpretation = generate_accuracy_interpretation(
            result.overall_summary,
            result.worst_case_summary,
            criteria,
            decision,
            metadata,
        )

        render_accuracy_executive_summary(
            result.overall_summary,
            result.worst_case_summary,
            decision,
            display_criteria,
            result.level_decision_table,
        )

        st.subheader("Accuracy Results")
        show_decision(decision)
        st.markdown("**Level-Specific Decision Table**")
        render_badged_criteria_table(display_level_decisions)
        st.markdown("**Overall Accuracy Summary**")
        st.dataframe(display_overall, width="stretch")
        st.markdown("**Accuracy Summary Table**")
        st.dataframe(display_accuracy, width="stretch")

        st.subheader("Acceptance Criteria Dashboard")
        st.markdown(
            accuracy_criteria_dashboard_html(display_criteria),
            unsafe_allow_html=True,
        )

        st.subheader("Acceptance Criteria Results")
        render_badged_criteria_table(display_criteria)

        summary_left, summary_right = st.columns(2)
        with summary_left:
            st.subheader("Bias Summary")
            st.dataframe(display_bias, width="stretch")
        with summary_right:
            st.subheader("Recovery Summary")
            st.dataframe(display_recovery, width="stretch")

        st.subheader("Worst-Case Performance")
        st.dataframe(display_worst_case, width="stretch")

        st.subheader("Interpretation")
        st.info(interpretation)

        st.subheader("Visualizations")
        expected_observed_plot = create_accuracy_expected_observed_plot(
            result.accuracy_summary
        )
        percent_bias_plot = create_accuracy_percent_bias_plot(
            result.accuracy_summary,
            criteria["Maximum Absolute Percent Bias"],
            criteria["Borderline Zone"],
        )
        recovery_plot = create_accuracy_recovery_plot(
            result.accuracy_summary,
            criteria["Minimum Recovery"],
            criteria["Maximum Recovery"],
            criteria["Borderline Zone"],
        )
        distribution_plot = create_accuracy_replicate_distribution_plot(
            result.analyzed_data
        )
        heatmap_plot = create_accuracy_performance_heatmap(result.accuracy_summary)
        individual_bias_plot = create_individual_accuracy_bias_plot(
            result.analyzed_data,
            criteria["Maximum Absolute Bias"],
        )
        st.plotly_chart(expected_observed_plot, width="stretch")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(percent_bias_plot, width="stretch")
        with chart_right:
            st.plotly_chart(recovery_plot, width="stretch")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(heatmap_plot, width="stretch")
        with chart_right:
            st.plotly_chart(individual_bias_plot, width="stretch")
        st.plotly_chart(distribution_plot, width="stretch")

        st.subheader("Analyzed Data")
        st.dataframe(result.analyzed_data, width="stretch")
        render_submit_for_review("Accuracy Study", metadata)

        html_report = build_accuracy_html_report(
            accuracy_summary=result.accuracy_summary,
            bias_summary=result.bias_summary,
            recovery_summary=result.recovery_summary,
            level_decision_table=result.level_decision_table,
            worst_case_summary=result.worst_case_summary,
            overall_summary=result.overall_summary,
            criteria_table=criteria_table,
            interpretation=interpretation,
            metadata=metadata,
            criteria=criteria,
            decision=decision,
            visualization_html={
                "Expected vs Observed Plot": expected_observed_plot.to_html(
                    full_html=False, include_plotlyjs="cdn"
                ),
                "Percent Bias by Level Plot": percent_bias_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Recovery by Level Plot": recovery_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Replicate Distribution Plot": distribution_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Accuracy Performance Heatmap": heatmap_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Individual Sample Bias Plot": individual_bias_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
            },
        )

        export_left, export_right = st.columns(2)
        with export_left:
            st.download_button(
                "Download accuracy summary CSV",
                data=build_accuracy_summary_csv(
                    metadata,
                    display_criteria,
                    display_overall,
                    display_level_decisions,
                    display_accuracy,
                    display_bias,
                    display_recovery,
                    display_worst_case,
                ).encode("utf-8"),
                file_name="accuracy_summary.csv",
                mime="text/csv",
            )
        with export_right:
            st.download_button(
                "Download accuracy report HTML",
                data=html_report.encode("utf-8"),
                file_name="accuracy_study_report.html",
                mime="text/html",
            )


def render_precision_workspace(metadata: dict[str, object]) -> None:
    """Render the Precision Study workflow."""

    criteria = render_precision_criteria()

    uploaded_file = st.file_uploader(
        "Upload repeated-measurement precision data",
        type=["csv", "xlsx", "xls"],
        help="Upload a file containing repeated results by level, day, run, and replicate.",
    )

    use_sample_data = st.checkbox(
        "Use included sample HbA1c precision study dataset",
        value=uploaded_file is None,
    )

    data = None
    if uploaded_file is not None:
        try:
            data = load_uploaded_file(uploaded_file)
        except Exception as exc:
            st.error(f"Unable to load file: {exc}")
            st.stop()
    elif use_sample_data:
        data = pd.read_csv(PRECISION_SAMPLE_DATA_PATH)

    if data is None:
        st.info("Upload a CSV or Excel file to begin.")
        st.stop()

    metadata["Sample Count"] = len(data)

    st.subheader("Data Preview")
    st.dataframe(data.head(20), width="stretch")

    numeric_columns = get_numeric_columns(data)
    if len(numeric_columns) < 1:
        st.error("At least one numeric result column is required for precision analysis.")
        st.stop()

    all_columns = list(data.columns)
    detected_sample_id = detect_sample_id_column(data)

    st.subheader("Column Selection")
    first_row = st.columns(3)
    with first_row[0]:
        result_column = st.selectbox(
            "Result column",
            numeric_columns,
            index=numeric_columns.index("Result") if "Result" in numeric_columns else 0,
        )
    with first_row[1]:
        level_column = st.selectbox(
            "Level column",
            all_columns,
            index=all_columns.index("Level") if "Level" in all_columns else 0,
        )
    with first_row[2]:
        sample_id_column = optional_column_selectbox(
            "Sample ID column (optional)",
            all_columns,
            detected_sample_id,
        )

    second_row = st.columns(3)
    with second_row[0]:
        day_column = optional_column_selectbox(
            "Day column",
            all_columns,
            "Day" if "Day" in all_columns else None,
        )
    with second_row[1]:
        run_column = optional_column_selectbox(
            "Run column",
            all_columns,
            "Run" if "Run" in all_columns else None,
        )
    with second_row[2]:
        replicate_column = optional_column_selectbox(
            "Replicate column",
            all_columns,
            "Replicate" if "Replicate" in all_columns else None,
        )

    run_analysis = st.button("Run Precision Analysis", type="primary")

    if run_analysis:
        try:
            result = run_precision_study(
                data=data,
                result_column=result_column,
                level_column=level_column,
                day_column=day_column,
                run_column=run_column,
                replicate_column=replicate_column,
                sample_id_column=sample_id_column,
            )
        except Exception as exc:
            st.error(f"Precision analysis could not be completed: {exc}")
            st.stop()

        criteria_result = evaluate_precision_criteria(
            result.level_summary,
            max_acceptable_cv=criteria["Maximum CV%"],
        )
        decision = str(criteria_result["decision"])
        criteria_table = build_criteria_table(criteria_result)
        display_level_summary = format_precision_summary_table(result.level_summary)
        display_day_summary = format_precision_summary_table(result.day_summary)
        display_run_summary = format_precision_summary_table(result.run_summary)
        display_criteria_table = format_precision_criteria_table(criteria_table)
        interpretation = generate_precision_interpretation(
            result.level_summary,
            metadata,
            criteria["Maximum CV%"],
            decision,
        )

        st.subheader("Precision Summary")
        show_decision(decision)
        st.dataframe(display_level_summary, width="stretch")

        st.subheader("Acceptance Criteria Results")
        render_badged_criteria_table(display_criteria_table)

        if not result.day_summary.empty:
            st.subheader("Day-Level Precision Summary")
            st.dataframe(display_day_summary, width="stretch")

        if not result.run_summary.empty:
            st.subheader("Run-Level Precision Summary")
            st.dataframe(display_run_summary, width="stretch")

        st.subheader("Interpretation")
        st.info(interpretation)

        st.subheader("Visualizations")
        run_chart = create_precision_run_chart(result.analyzed_data)
        cv_bar_chart = create_precision_cv_bar_chart(result.level_summary)
        box_plot = create_precision_box_plot(result.analyzed_data)
        st.plotly_chart(run_chart, width="stretch")

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(cv_bar_chart, width="stretch")
        with chart_right:
            st.plotly_chart(box_plot, width="stretch")

        st.subheader("Analyzed Data")
        st.dataframe(result.analyzed_data, width="stretch")
        render_submit_for_review("Precision Study", metadata)

        html_report = build_precision_html_report(
            level_summary=result.level_summary,
            day_summary=result.day_summary,
            run_summary=result.run_summary,
            analyzed_data=result.analyzed_data,
            criteria_table=criteria_table,
            interpretation=interpretation,
            metadata=metadata,
            max_acceptable_cv=criteria["Maximum CV%"],
            decision=decision,
            visualization_html={
                "Precision Run Chart": run_chart.to_html(
                    full_html=False, include_plotlyjs="cdn"
                ),
                "Precision CV% Bar Chart": cv_bar_chart.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Precision Box Plot": box_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
            },
        )

        export_left, export_right = st.columns(2)
        with export_left:
            st.download_button(
                "Download precision summary CSV",
                data=build_precision_summary_csv(
                    metadata,
                    display_criteria_table,
                    display_level_summary,
                    display_day_summary,
                    display_run_summary,
                ).encode("utf-8"),
                file_name="precision_summary.csv",
                mime="text/csv",
            )
        with export_right:
            st.download_button(
                "Download precision report HTML",
                data=html_report.encode("utf-8"),
                file_name="precision_study_report.html",
                mime="text/html",
            )


def render_linearity_workspace(metadata: dict[str, object]) -> None:
    """Render the Linearity Study workflow."""

    criteria = render_linearity_criteria()

    uploaded_file = st.file_uploader(
        "Upload linearity study data",
        type=["csv", "xlsx", "xls"],
        help="Upload expected and observed results across linearity levels.",
    )
    use_sample_data = st.checkbox(
        "Use included sample HbA1c linearity study dataset",
        value=uploaded_file is None,
    )

    data = None
    if uploaded_file is not None:
        try:
            data = load_uploaded_file(uploaded_file)
        except Exception as exc:
            st.error(f"Unable to load file: {exc}")
            st.stop()
    elif use_sample_data:
        data = pd.read_csv(LINEARITY_SAMPLE_DATA_PATH)

    if data is None:
        st.info("Upload a CSV or Excel file to begin.")
        st.stop()

    metadata["Sample Count"] = len(data)

    st.subheader("Data Preview")
    st.dataframe(data.head(20), width="stretch")

    numeric_columns = get_numeric_columns(data)
    all_columns = list(data.columns)
    if len(numeric_columns) < 2:
        st.error("Expected and observed numeric result columns are required.")
        st.stop()

    st.subheader("Column Selection")
    first_row = st.columns(3)
    with first_row[0]:
        expected_column = st.selectbox(
            "Expected Result column",
            numeric_columns,
            index=numeric_columns.index("Expected Result")
            if "Expected Result" in numeric_columns
            else 0,
        )
    with first_row[1]:
        observed_column = st.selectbox(
            "Observed Result column",
            numeric_columns,
            index=numeric_columns.index("Observed Result")
            if "Observed Result" in numeric_columns
            else min(1, len(numeric_columns) - 1),
        )
    with first_row[2]:
        level_column = st.selectbox(
            "Level column",
            all_columns,
            index=all_columns.index("Level") if "Level" in all_columns else 0,
        )

    second_row = st.columns(3)
    with second_row[0]:
        replicate_column = optional_column_selectbox(
            "Replicate column",
            all_columns,
            "Replicate" if "Replicate" in all_columns else None,
        )
    with second_row[1]:
        units_column = optional_column_selectbox(
            "Units column",
            all_columns,
            "Units" if "Units" in all_columns else None,
        )
    with second_row[2]:
        include_column = optional_column_selectbox(
            "Include in Analysis column",
            all_columns,
            "Include in Analysis" if "Include in Analysis" in all_columns else None,
        )

    if expected_column == observed_column:
        st.warning("Select different expected and observed result columns.")

    run_analysis = st.button("Run Linearity Analysis", type="primary")
    if run_analysis:
        if expected_column == observed_column:
            st.stop()

        try:
            result = run_linearity_study(
                data,
                expected_column,
                observed_column,
                level_column,
                replicate_column,
                units_column,
                include_column,
            )
        except Exception as exc:
            st.error(f"Linearity analysis could not be completed: {exc}")
            st.stop()

        criteria_result = evaluate_linearity_criteria(
            result.level_summary,
            result.regression_summary,
            min_r_squared=criteria["Minimum R²"],
            slope_lower_limit=criteria["Slope Lower Limit"],
            slope_upper_limit=criteria["Slope Upper Limit"],
            max_abs_percent_bias=criteria["Maximum Absolute Percent Bias"],
            recovery_lower_limit=criteria["Recovery Lower Limit"],
            recovery_upper_limit=criteria["Recovery Upper Limit"],
        )
        decision = str(criteria_result["decision"])
        criteria_table = build_criteria_table(criteria_result)
        display_summary = format_linearity_summary_table(result.level_summary)
        display_criteria = format_linearity_criteria_table(criteria_table)
        display_regression = format_linearity_regression_summary(
            result.regression_summary
        )
        interpretation = generate_linearity_interpretation(
            result.level_summary,
            result.regression_summary,
            criteria,
            decision,
            metadata,
        )

        render_linearity_executive_summary(
            result.level_summary, result.regression_summary, decision
        )

        st.subheader("Linearity Summary Table")
        st.dataframe(display_summary, width="stretch")
        render_linearity_worst_case(result.level_summary)

        st.subheader("Acceptance Criteria Results")
        render_badged_criteria_table(display_criteria)

        render_linearity_equation_card(result.regression_summary)
        st.subheader("Regression Summary")
        st.dataframe(display_regression, width="stretch")

        st.subheader("Interpretation")
        st.info(interpretation)

        st.subheader("Visualizations")
        linearity_plot = create_linearity_plot(
            result.level_summary, result.regression_summary
        )
        residual_plot = create_linearity_residual_plot(
            result.level_summary, result.regression_summary
        )
        recovery_plot = create_percent_recovery_plot(result.level_summary)
        st.plotly_chart(linearity_plot, width="stretch")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(residual_plot, width="stretch")
        with chart_right:
            st.plotly_chart(recovery_plot, width="stretch")

        st.subheader("Analyzed Data")
        st.dataframe(
            result.analyzed_data.drop(columns=["Timepoint Sort"], errors="ignore"),
            width="stretch",
        )
        render_submit_for_review("Linearity Study", metadata)

        html_report = build_linearity_html_report(
            result.level_summary,
            result.regression_summary,
            criteria_table,
            interpretation,
            metadata,
            criteria,
            decision,
            visualization_html={
                "Linearity Plot": linearity_plot.to_html(
                    full_html=False, include_plotlyjs="cdn"
                ),
                "Residual Plot": residual_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Percent Recovery Plot": recovery_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
            },
        )

        export_left, export_right = st.columns(2)
        with export_left:
            st.download_button(
                "Download linearity summary CSV",
                data=build_linearity_summary_csv(
                    metadata,
                    display_criteria,
                    display_summary,
                    display_regression,
                ).encode("utf-8"),
                file_name="linearity_summary.csv",
                mime="text/csv",
            )
        with export_right:
            st.download_button(
                "Download linearity report HTML",
                data=html_report.encode("utf-8"),
                file_name="linearity_study_report.html",
                mime="text/html",
            )


def render_stability_workspace(metadata: dict[str, object]) -> None:
    """Render the Stability Study workflow."""

    criteria = render_stability_criteria()

    uploaded_file = st.file_uploader(
        "Upload stability study data",
        type=["csv", "xlsx", "xls"],
        help="Upload baseline and timepoint results for sample stability analysis.",
    )
    use_sample_data = st.checkbox(
        "Use included sample HbA1c stability study dataset",
        value=uploaded_file is None,
    )

    data = None
    if uploaded_file is not None:
        try:
            data = load_uploaded_file(uploaded_file)
        except Exception as exc:
            st.error(f"Unable to load file: {exc}")
            st.stop()
    elif use_sample_data:
        data = pd.read_csv(STABILITY_SAMPLE_DATA_PATH)

    if data is None:
        st.info("Upload a CSV or Excel file to begin.")
        st.stop()

    metadata["Sample Count"] = len(data)

    st.subheader("Data Preview")
    st.dataframe(data.head(20), width="stretch")

    numeric_columns = get_numeric_columns(data)
    all_columns = list(data.columns)
    if len(numeric_columns) < 1:
        st.error("At least one numeric result column is required for stability analysis.")
        st.stop()

    st.subheader("Column Selection")
    first_row = st.columns(3)
    with first_row[0]:
        sample_id_column = st.selectbox(
            "Sample ID column",
            all_columns,
            index=all_columns.index("Sample ID") if "Sample ID" in all_columns else 0,
        )
    with first_row[1]:
        timepoint_column = st.selectbox(
            "Timepoint column",
            all_columns,
            index=all_columns.index("Timepoint") if "Timepoint" in all_columns else 0,
        )
    with first_row[2]:
        result_column = st.selectbox(
            "Result column",
            numeric_columns,
            index=numeric_columns.index("Result") if "Result" in numeric_columns else 0,
        )

    second_row = st.columns(4)
    with second_row[0]:
        storage_condition_column = optional_column_selectbox(
            "Storage Condition column",
            all_columns,
            "Storage Condition" if "Storage Condition" in all_columns else None,
        )
    with second_row[1]:
        units_column = optional_column_selectbox(
            "Units column",
            all_columns,
            "Units" if "Units" in all_columns else None,
        )
    with second_row[2]:
        replicate_column = optional_column_selectbox(
            "Replicate column",
            all_columns,
            "Replicate" if "Replicate" in all_columns else None,
        )
    with second_row[3]:
        include_column = optional_column_selectbox(
            "Include in Analysis column",
            all_columns,
            "Include in Analysis" if "Include in Analysis" in all_columns else None,
        )

    run_analysis = st.button("Run Stability Analysis", type="primary")
    if run_analysis:
        try:
            result = run_stability_study(
                data,
                sample_id_column,
                timepoint_column,
                result_column,
                storage_condition_column,
                units_column,
                replicate_column,
                include_column,
            )
        except Exception as exc:
            st.error(f"Stability analysis could not be completed: {exc}")
            st.stop()

        criteria_result = evaluate_stability_criteria(
            result.overall_summary,
            max_percent_change=criteria["Maximum Percent Change"],
            min_recovery=criteria["Minimum Recovery"],
            max_abs_bias=criteria["Maximum Absolute Bias"],
            borderline_zone=criteria["Borderline Zone"],
        )
        decision = str(criteria_result["decision"])
        criteria_table = build_criteria_table(criteria_result)
        display_criteria = format_stability_criteria_table(criteria_table)
        display_overall = format_stability_overall_summary(result.overall_summary)
        display_stability = format_stability_table(result.stability_summary)
        display_timepoint = format_stability_table(result.timepoint_summary)
        display_recovery = format_stability_table(result.recovery_summary)
        display_bias = format_stability_table(result.bias_summary)
        display_condition_comparison = format_storage_condition_comparison_table(
            result.condition_comparison
        )
        display_outliers = format_stability_outlier_table(result.outlier_table)
        condition_interpretation = generate_storage_condition_comparison_interpretation(
            result.condition_comparison
        )
        risk_assessment = generate_stability_risk_assessment(
            result.overall_summary,
            display_criteria,
            criteria,
        )
        interpretation = generate_stability_interpretation(
            result.overall_summary,
            result.timepoint_summary,
            criteria,
            decision,
            metadata,
        )

        render_stability_executive_summary(
            result.overall_summary, decision, display_criteria
        )

        st.subheader("Stability Summary Table")
        st.dataframe(display_overall, width="stretch")
        st.dataframe(display_stability, width="stretch")

        st.subheader("Acceptance Criteria Results")
        render_badged_criteria_table(display_criteria)

        st.subheader("Timepoint Summary Table")
        st.dataframe(display_timepoint, width="stretch")

        summary_left, summary_right = st.columns(2)
        with summary_left:
            st.subheader("Recovery Summary")
            st.dataframe(display_recovery, width="stretch")
        with summary_right:
            st.subheader("Bias Summary")
            st.dataframe(display_bias, width="stretch")

        st.subheader("Storage Condition Comparison")
        st.info(condition_interpretation)
        if result.condition_comparison.empty:
            st.info("At least two storage conditions are required for direct comparison.")
        else:
            st.dataframe(display_condition_comparison, width="stretch")

        st.subheader("Potential Stability Outliers")
        if result.outlier_table.empty:
            st.info("No potential stability outliers were available for review.")
        else:
            st.dataframe(display_outliers, width="stretch")

        st.subheader("Risk Assessment")
        st.info(risk_assessment)

        st.subheader("Interpretation")
        st.info(interpretation)

        st.subheader("Visualizations")
        trend_plot = create_stability_trend_plot(
            result.timepoint_summary, result.overall_summary["Baseline Mean"]
        )
        percent_change_plot = create_stability_percent_change_plot(
            result.timepoint_summary,
            criteria["Maximum Percent Change"],
            criteria["Borderline Zone"],
        )
        stability_recovery_plot = create_stability_recovery_plot(
            result.timepoint_summary,
            criteria["Minimum Recovery"],
            criteria["Borderline Zone"],
        )
        stability_bias_plot = create_stability_bias_plot(
            result.timepoint_summary,
            criteria["Maximum Absolute Bias"],
            criteria["Borderline Zone"],
        )
        individual_plot = create_individual_stability_plot(result.analyzed_data)
        condition_difference_plot = (
            create_condition_difference_plot(result.condition_comparison)
            if not result.condition_comparison.empty
            else None
        )

        st.plotly_chart(trend_plot, width="stretch")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(percent_change_plot, width="stretch")
        with chart_right:
            st.plotly_chart(stability_recovery_plot, width="stretch")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(stability_bias_plot, width="stretch")
        with chart_right:
            if condition_difference_plot is not None:
                st.plotly_chart(condition_difference_plot, width="stretch")
        st.plotly_chart(individual_plot, width="stretch")

        st.subheader("Analyzed Data")
        st.dataframe(
            result.analyzed_data.drop(columns=["Timepoint Sort"], errors="ignore"),
            width="stretch",
        )
        render_submit_for_review("Stability Study", metadata)

        html_report = build_stability_html_report(
            stability_summary=result.stability_summary,
            timepoint_summary=result.timepoint_summary,
            recovery_summary=result.recovery_summary,
            bias_summary=result.bias_summary,
            condition_comparison=result.condition_comparison,
            outlier_table=result.outlier_table,
            overall_summary=result.overall_summary,
            criteria_table=criteria_table,
            risk_assessment=risk_assessment,
            condition_interpretation=condition_interpretation,
            interpretation=interpretation,
            metadata=metadata,
            criteria=criteria,
            decision=decision,
            visualization_html={
                "Stability Trend Plot": trend_plot.to_html(
                    full_html=False, include_plotlyjs="cdn"
                ),
                "Percent Change Plot": percent_change_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Recovery Plot": stability_recovery_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Bias Plot": stability_bias_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                **(
                    {
                        "Condition Difference Plot": condition_difference_plot.to_html(
                            full_html=False, include_plotlyjs=False
                        )
                    }
                    if condition_difference_plot is not None
                    else {}
                ),
                "Individual Sample Stability Plot": individual_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
            },
        )

        pdf_report = build_stability_pdf_report(
            stability_summary=result.stability_summary,
            timepoint_summary=result.timepoint_summary,
            recovery_summary=result.recovery_summary,
            bias_summary=result.bias_summary,
            condition_comparison=result.condition_comparison,
            outlier_table=result.outlier_table,
            overall_summary=result.overall_summary,
            criteria_table=criteria_table,
            risk_assessment=risk_assessment,
            interpretation=interpretation,
            metadata=metadata,
            criteria=criteria,
            decision=decision,
        )

        export_left, export_middle, export_right = st.columns(3)
        with export_left:
            st.download_button(
                "Download stability summary CSV",
                data=build_stability_summary_csv(
                    metadata,
                    display_criteria,
                    display_overall,
                    display_stability,
                    display_timepoint,
                    display_recovery,
                    display_bias,
                    display_condition_comparison,
                    display_outliers,
                    risk_assessment,
                ).encode("utf-8"),
                file_name="stability_summary.csv",
                mime="text/csv",
            )
        with export_middle:
            st.download_button(
                "Download stability report HTML",
                data=html_report.encode("utf-8"),
                file_name="stability_study_report.html",
                mime="text/html",
            )
        with export_right:
            st.download_button(
                "Download Stability Report PDF",
                data=pdf_report,
                file_name="stability_study_report.pdf",
                mime="application/pdf",
            )


def main() -> None:
    """Render the Streamlit interface and run selected analyses."""

    st.set_page_config(page_title=APP_TITLE, layout="wide")
    initialize_platform_state()
    st.title(APP_TITLE)
    inject_validation_styles()
    st.caption(
        f"{APP_VERSION} {APP_STATUS} · Validation analytics for assay studies and diagnostic laboratory workflows."
    )

    page_options = [
        "Dashboard",
        "Projects",
        "Validation Workspace",
        "Validation Review Center",
        "Reports Library",
        "Platform Settings",
    ]
    demo_mode = st.sidebar.checkbox("Developer / demo mode", value=False)
    if demo_mode:
        page_options.append("Sample Datasets")
    if "pending_page" in st.session_state:
        st.session_state.navigation_choice = st.session_state.pop("pending_page")
    if "navigation_choice" not in st.session_state or st.session_state.navigation_choice not in page_options:
        st.session_state.navigation_choice = "Dashboard"
    page = st.sidebar.radio(
        "Navigation",
        page_options,
        key="navigation_choice",
    )
    with st.sidebar.expander("Version & Release Notes", expanded=False):
        st.markdown(f"### {APP_VERSION} Production Release")
        st.caption("Production navigation is focused on Project → Validation Study → Review → Report Package.")
        st.markdown("**Core Validation Modules**")
        st.write("\n".join(f"- {module}" for module in CORE_VALIDATION_MODULES))
        st.markdown("**Platform Features**")
        st.write(
            "\n".join(
                [
                    "- Validation Program Management",
                    "- Validation Workspace",
                    "- Validation Review Center",
                    "- Reports Library",
                    "- Consolidated Validation Packages",
                    "- PDF Export",
                    "- HTML Export",
                    "- Platform Settings",
                ]
            )
        )
        st.markdown("**Known Limitations**")
        st.write("- Audit Trail planned for v1.1\n- Electronic Signatures planned for v1.2")

    if page == "Dashboard":
        render_dashboard()
        st.stop()

    if page == "Projects":
        render_projects_workspace()
        st.stop()

    if page == "Validation Review Center":
        render_validation_review_center()
        st.stop()

    if page == "Reports Library":
        render_reports_library()
        st.stop()

    if page == "Sample Datasets":
        render_sample_dataset_library()
        st.stop()

    if page == "Platform Settings":
        render_platform_settings()
        st.stop()

    study_type = render_study_type_selector()
    if study_type == "Precision Study":
        metadata = render_study_documentation(study_type)
        render_precision_workspace(metadata)
        st.stop()

    if study_type == "Linearity Study":
        metadata = render_study_documentation(study_type)
        render_linearity_workspace(metadata)
        st.stop()

    if study_type == "Stability Study":
        metadata = render_study_documentation(study_type)
        render_stability_workspace(metadata)
        st.stop()

    if study_type == "Accuracy Study":
        metadata = render_study_documentation(study_type)
        render_accuracy_workspace(metadata)
        st.stop()

    if study_type == "Detection Capability":
        metadata = render_study_documentation(study_type)
        render_detection_workspace(metadata)
        st.stop()

    if study_type == "DBS Validation":
        metadata = render_study_documentation(study_type)
        render_dbs_workspace(metadata)
        st.stop()

    if study_type == "Microtainer Validation":
        metadata = render_study_documentation(study_type)
        render_microtainer_workspace(metadata)
        st.stop()

    if study_type != "Method Comparison":
        st.stop()

    metadata = render_study_documentation(study_type)
    criteria = render_method_comparison_criteria()

    uploaded_file = st.file_uploader(
        "Upload paired method-comparison data",
        type=["csv", "xlsx", "xls"],
        help="Upload a file containing reference and candidate result columns.",
    )

    use_sample_data = st.checkbox(
        "Use included sample HbA1c method-comparison dataset",
        value=uploaded_file is None,
    )

    data = None
    if uploaded_file is not None:
        try:
            data = load_uploaded_file(uploaded_file)
        except Exception as exc:
            st.error(f"Unable to load file: {exc}")
            st.stop()
    elif use_sample_data:
        data = pd.read_csv(SAMPLE_DATA_PATH)

    if data is None:
        st.info("Upload a CSV or Excel file to begin.")
        st.stop()

    metadata["Sample Count"] = len(data)

    st.subheader("Data Preview")
    st.dataframe(data.head(20), width='stretch')

    numeric_columns = get_numeric_columns(data)
    if len(numeric_columns) < 2:
        st.error("At least two numeric result columns are required for analysis.")
        st.stop()

    st.subheader("Column Selection")
    detected_sample_id = detect_sample_id_column(data)
    sample_id_options = ["None"] + list(data.columns)
    sample_id_default = (
        sample_id_options.index(detected_sample_id)
        if detected_sample_id in sample_id_options
        else 0
    )

    left, middle, right = st.columns(3)
    with left:
        reference_column = st.selectbox(
            "Reference result column",
            numeric_columns,
            index=0,
        )
    with middle:
        default_candidate_index = 1 if len(numeric_columns) > 1 else 0
        candidate_column = st.selectbox(
            "Candidate result column",
            numeric_columns,
            index=default_candidate_index,
        )
    with right:
        sample_id_selection = st.selectbox(
            "Sample ID column (optional)",
            sample_id_options,
            index=sample_id_default,
        )
        sample_id_column = None if sample_id_selection == "None" else sample_id_selection

    if reference_column == candidate_column:
        st.warning("Select two different columns before running the analysis.")

    run_analysis = st.button("Run Analysis", type="primary")

    if run_analysis:
        if reference_column == candidate_column:
            st.stop()

        try:
            result = run_method_comparison(
                data, reference_column, candidate_column, sample_id_column
            )
        except Exception as exc:
            st.error(f"Analysis could not be completed: {exc}")
            st.stop()

        summary_table = build_summary_table(result.summary)
        percent_samples_meeting_agreement = calculate_percent_samples_meeting_agreement(
            result.analyzed_data,
            criteria["Sample Agreement Percent Bias Limit"],
        )
        criteria_result = evaluate_acceptance_criteria(
            result.summary,
            min_r_squared=criteria["Minimum R²"],
            min_correlation_r=criteria["Minimum Correlation r"],
            slope_lower_limit=criteria["Slope Lower Limit"],
            slope_upper_limit=criteria["Slope Upper Limit"],
            max_abs_intercept=criteria["Maximum Absolute Intercept"],
            max_abs_mean_bias=criteria["Maximum Absolute Mean Bias"],
            max_abs_mean_percent_bias=criteria[
                "Maximum Absolute Mean Percent Bias"
            ],
            max_abs_mean_difference=criteria["Maximum Absolute Mean Difference"],
            percent_samples_meeting_agreement=percent_samples_meeting_agreement,
            min_percent_samples_meeting_agreement=criteria[
                "Minimum Percent Samples Meeting Agreement"
            ],
        )
        decision = str(criteria_result["decision"])
        criteria_table = build_criteria_table(criteria_result)
        outlier_table = get_top_outliers(result.analyzed_data, count=5)
        interpretation = generate_interpretation(
            result.summary, metadata, criteria, decision
        )

        st.subheader("Summary Table")
        show_decision(decision)
        st.dataframe(summary_table, width='stretch')

        st.subheader("Acceptance Criteria Results")
        render_badged_criteria_table(criteria_table)
        st.metric(
            "Samples meeting agreement criteria",
            f"{percent_samples_meeting_agreement:.1f}%",
        )

        st.subheader("Interpretation")
        st.info(interpretation)

        st.subheader("Bland-Altman Style Difference Statistics")
        ba_columns = st.columns(4)
        ba_columns[0].metric("Mean Difference", f"{result.summary['Mean Difference']:.3f}")
        ba_columns[1].metric("SD Difference", f"{result.summary['SD Difference']:.3f}")
        ba_columns[2].metric(
            "Upper LOA", f"{result.summary['Upper Limit of Agreement']:.3f}"
        )
        ba_columns[3].metric(
            "Lower LOA", f"{result.summary['Lower Limit of Agreement']:.3f}"
        )

        st.subheader("Visualizations")
        scatter_plot = create_scatter_plot(result.analyzed_data, result.summary)
        bias_histogram = create_percent_bias_histogram(result.analyzed_data)
        difference_plot = create_difference_plot(result.analyzed_data)
        st.plotly_chart(
            scatter_plot,
            width='stretch',
        )

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(
                bias_histogram,
                width='stretch',
            )
        with chart_right:
            st.plotly_chart(
                difference_plot,
                width='stretch',
            )

        st.subheader("Top Disagreement Samples")
        st.dataframe(outlier_table, width='stretch')

        st.subheader("Analyzed Data")
        st.dataframe(result.analyzed_data, width='stretch')
        render_submit_for_review("Method Comparison", metadata)

        html_report = build_html_report(
            summary_table,
            criteria_table,
            outlier_table,
            interpretation,
            metadata,
            criteria,
            decision,
            reference_column,
            candidate_column,
            visualization_html={
                "Reference vs Candidate Scatter Plot": scatter_plot.to_html(
                    full_html=False, include_plotlyjs="cdn"
                ),
                "Percent Bias Histogram": bias_histogram.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Difference Plot": difference_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
            },
        )

        export_left, export_right = st.columns(2)
        with export_left:
            st.download_button(
                "Download analysis summary CSV",
                data=summary_table.to_csv(index=False).encode("utf-8"),
                file_name="analysis_summary.csv",
                mime="text/csv",
            )
        with export_right:
            st.download_button(
                "Download summary report HTML",
                data=html_report.encode("utf-8"),
                file_name="summary_report.html",
                mime="text/html",
            )


if __name__ == "__main__":
    main()
