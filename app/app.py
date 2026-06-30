"""Streamlit application for the Scientific Validation Analytics Platform."""

from __future__ import annotations

import base64
from datetime import date, datetime
import hashlib
from html import escape
from io import BytesIO, StringIO
import json
from pathlib import Path
import pickle
import sys
from urllib.parse import quote
import zipfile

import pandas as pd
import streamlit as st
from fpdf import FPDF

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
REVIEW_PACKAGE_DIR = PROJECT_ROOT / "data" / "review_packages"
EXECUTION_STATE_DIR = PROJECT_ROOT / "data" / "execution_state"
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
    create_method_comparison_residual_plot,
    create_percent_bias_histogram,
    create_percent_recovery_plot,
    create_precision_box_plot,
    create_precision_cv_bar_chart,
    create_precision_day_variation_plot,
    create_precision_run_variation_plot,
    create_precision_run_chart,
    create_precision_variance_component_plot,
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


def sanitize(value: object) -> str:
    """Return a filesystem-safe lowercase identifier."""

    text = str(value or "").strip().lower()
    cleaned = "".join(character if character.isalnum() else "_" for character in text)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "validation_package"


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
TRUE_VALIDATION_STUDY_TYPES = (
    "Method Comparison",
    "Precision",
    "Accuracy",
    "Linearity",
    "Stability",
    "Detection Capability",
    "Carryover",
    "Interference",
    "Reference Interval Verification",
)
DEFAULT_PROGRAM_REQUIRED_STUDIES = tuple(TRUE_VALIDATION_STUDY_TYPES[:6])
VALIDATION_CONTEXTS = (
    "Serum vs Microtainer Equivalency",
    "DBS Validation",
    "Specimen Stability",
    "New Assay Launch",
    "Method Transfer",
    "Instrument Comparison",
)
VALIDATION_SCOPE_DEFINITIONS = {
    "Microtainer Serum vs Venous Serum": {
        "Validation Context": "Serum vs Microtainer Equivalency",
        "Validation Template": "Microtainer Equivalency Template",
        "Candidate Specimen": "Microtainer serum",
        "Reference Specimen": "Venous serum",
    },
    "Dried Blood Spot vs Venous Whole Blood": {
        "Validation Context": "DBS Validation",
        "Validation Template": "DBS Equivalency Template",
        "Candidate Specimen": "Dried blood spot",
        "Reference Specimen": "Venous whole blood",
    },
}
VALIDATION_PROGRAM_CATEGORIES = (
    "Specimen Equivalency Validation",
    "Microtainer Validation",
    "DBS Validation",
    "Method Validation",
    "Method Transfer",
    "Instrument Comparison",
)
ANALYZER_OPTIONS = ("Beckman Access 2", "Beckman AU480", "LC-MS/MS")
REAGENT_OPTIONS = (
    "Cortisol Reagent Pack",
    "TSH3 Ultra Reagent Pack",
    "Ferritin Reagent Pack",
    "Vitamin D Total Reagent Pack",
    "HbA1c Variant Analysis Reagent",
    "Vitamin B12 Reagent Pack",
    "Glucose Hexokinase Reagent",
    "CRP Wide Range Reagent",
    "LC-MS Validation Reagent Kit",
)
VALIDATION_TEMPLATES = {
    "Microtainer Equivalency Template": {
        "Program Category": "Specimen Equivalency Validation",
        "Validation Context": "Serum vs Microtainer Equivalency",
        "Purpose": "Demonstrate analytical equivalency between reference and microtainer specimens.",
        "Required Study Types": (
            "Method Comparison",
            "Precision",
            "Accuracy",
            "Linearity",
            "Stability",
            "Detection Capability",
        ),
        "Optional Study Types": ("Carryover", "Interference", "Reference Interval Verification"),
        "Acceptance Criteria": "User-defined analyte and study-specific acceptance criteria.",
        "Sample Requirements": "Paired venous and microtainer specimens by analyte.",
        "Review Workflow": "Analysis completion, scientific review, approval, final report.",
        "Report Structure": "Program summary, analyte sections, study results, review record.",
    },
    "DBS Equivalency Template": {
        "Program Category": "Specimen Equivalency Validation",
        "Validation Context": "DBS Validation",
        "Purpose": "Demonstrate analytical equivalency between DBS-derived and reference specimen results.",
        "Required Study Types": (
            "Method Comparison",
            "Precision",
            "Accuracy",
            "Stability",
            "Detection Capability",
        ),
        "Optional Study Types": ("Interference", "Reference Interval Verification"),
        "Acceptance Criteria": "User-defined DBS equivalency, recovery, and bias criteria.",
        "Sample Requirements": "Matched DBS and reference specimens by analyte.",
        "Review Workflow": "Analysis completion, scientific review, approval, final report.",
        "Report Structure": "DBS equivalency summary, analyte sections, study results.",
    },
    "Method Validation Template": {
        "Program Category": "Method Validation",
        "Validation Context": "New Assay Launch",
        "Purpose": "Establish analytical performance for a new or transferred assay method.",
        "Required Study Types": (
            "Method Comparison",
            "Precision",
            "Accuracy",
            "Linearity",
            "Stability",
            "Detection Capability",
            "Reference Interval Verification",
        ),
        "Optional Study Types": ("Carryover", "Interference"),
        "Acceptance Criteria": "User-defined analytical performance criteria by study type.",
        "Sample Requirements": "Representative specimens across analytical range.",
        "Review Workflow": "Analysis completion, scientific review, approval, final report.",
        "Report Structure": "Analytical validation package with study appendices.",
    },
}
STUDY_LIFECYCLE_STATES = (
    "Draft",
    "Data Uploaded",
    "Analysis Complete",
    "Ready For Review",
    "Pending Review",
    "Scientific Review",
    "Submitted for Review",
    "Under Review",
    "Approved",
    "Rejected",
    "Requires Revision",
    "Follow-Up Required",
    "Report Generated",
    "Locked",
    "Archived",
)
REPORT_ELIGIBLE_STATES = {"Approved", "Locked"}
EXECUTION_COMPLETE_STATES = {"Approved", "Report Generated", "Locked", "Archived"}
EXECUTION_PROGRESS_COMPLETE_STATES = EXECUTION_COMPLETE_STATES | {
    "Analysis Complete",
    "Ready For Review",
    "Pending Review",
    "Scientific Review",
    "Submitted for Review",
    "Under Review",
}
EXECUTION_ANALYZED_STATES = EXECUTION_PROGRESS_COMPLETE_STATES
EXECUTION_FINISHED_STATES = EXECUTION_PROGRESS_COMPLETE_STATES
EXECUTION_REVIEW_STATES = {"Ready For Review", "Pending Review", "Scientific Review", "Submitted for Review", "Under Review"}
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
        "Project Name": "Core 1 Microtainer Validation Program",
        "Program Name": "Core 1 Microtainer Validation Program",
        "Assay": "Multi-Analyte Panel",
        "Assay / Biomarker": "Core 1 Microtainer Panel",
        "Program Owner": "Validation Team",
        "Candidate Specimen": "Microtainer serum",
        "Reference Specimen": "Venous serum",
        "Study Status": "In Progress",
        "Status": "In Progress",
        "Start Date": "2026-06-18",
        "Target Completion Date": "2026-07-18",
        "Reviewer": "",
        "Notes": "Microtainer equivalency validation program for core chemistry and immunoassay analytes.",
        "Required Studies": list(DEFAULT_PROGRAM_REQUIRED_STUDIES),
        "Completed Studies": [],
        "Analytes": [
            {
                "Analyte": "Ferritin",
                "Assigned Analyzer": "Atellica CI 1900",
                "Assigned Reagent": "Ferritin Reagent Pack",
                "Required Studies": list(DEFAULT_PROGRAM_REQUIRED_STUDIES),
                "Completed Studies": [],
                "In Progress Studies": [],
                "Last Updated": "2026-06-18",
                "Required Samples": 80,
                "Received Samples": 0,
                "Processed Samples": 0,
            },
            {
                "Analyte": "Vitamin D",
                "Assigned Analyzer": "Atellica IM 1600",
                "Assigned Reagent": "Vitamin D Total Reagent Pack",
                "Required Studies": list(DEFAULT_PROGRAM_REQUIRED_STUDIES),
                "Completed Studies": ["Method Comparison", "Precision", "Accuracy", "Linearity"],
                "In Progress Studies": ["Stability"],
                "Last Updated": "2026-06-18",
                "Required Samples": 80,
                "Received Samples": 52,
                "Processed Samples": 40,
            },
            {
                "Analyte": "HbA1c",
                "Assigned Analyzer": "Tosoh G8",
                "Assigned Reagent": "HbA1c Variant Analysis Reagent",
                "Required Studies": list(DEFAULT_PROGRAM_REQUIRED_STUDIES),
                "Completed Studies": list(DEFAULT_PROGRAM_REQUIRED_STUDIES),
                "In Progress Studies": [],
                "Last Updated": "2026-06-18",
                "Required Samples": 80,
                "Received Samples": 80,
                "Processed Samples": 80,
            },
            {
                "Analyte": "TSH",
                "Assigned Analyzer": "Atellica IM 1600",
                "Assigned Reagent": "TSH3-Ultra Reagent Pack",
                "Required Studies": list(DEFAULT_PROGRAM_REQUIRED_STUDIES),
                "Completed Studies": ["Method Comparison", "Precision", "Accuracy"],
                "In Progress Studies": ["Linearity"],
                "Last Updated": "2026-06-18",
                "Required Samples": 80,
                "Received Samples": 46,
                "Processed Samples": 32,
            },
            {
                "Analyte": "Cortisol",
                "Assigned Analyzer": "Atellica IM 1600",
                "Assigned Reagent": "Cortisol Reagent Pack",
                "Required Studies": list(DEFAULT_PROGRAM_REQUIRED_STUDIES),
                "Completed Studies": [],
                "In Progress Studies": [],
                "Last Updated": "2026-06-18",
                "Required Samples": 80,
                "Received Samples": 0,
                "Processed Samples": 0,
            },
            {
                "Analyte": "Vitamin B12",
                "Assigned Analyzer": "Atellica IM 1600",
                "Assigned Reagent": "Vitamin B12 Reagent Pack",
                "Required Studies": list(DEFAULT_PROGRAM_REQUIRED_STUDIES),
                "Completed Studies": ["Method Comparison", "Precision"],
                "In Progress Studies": ["Accuracy"],
                "Last Updated": "2026-06-18",
                "Required Samples": 80,
                "Received Samples": 36,
                "Processed Samples": 24,
            },
        ],
        "Last Updated": "2026-06-18",
        "Overall Status": "In Progress",
        "Final Package Generated": False,
    },
    {
        "Project Name": "DBS Equivalency Validation Program",
        "Program Name": "DBS Equivalency Validation Program",
        "Assay": "DBS Panel",
        "Assay / Biomarker": "DBS Equivalency Panel",
        "Program Owner": "Validation Team",
        "Candidate Specimen": "Dried blood spot",
        "Reference Specimen": "Venous whole blood",
        "Study Status": "In Progress",
        "Status": "In Progress",
        "Start Date": "2026-06-18",
        "Target Completion Date": "2026-07-18",
        "Reviewer": "",
        "Notes": "",
        "Required Studies": list(DEFAULT_PROGRAM_REQUIRED_STUDIES),
        "Completed Studies": [],
        "Analytes": [
            {
                "Analyte": "HbA1c",
                "Assigned Analyzer": "Tosoh G8",
                "Assigned Reagent": "HbA1c Variant Analysis Reagent",
                "Required Studies": list(DEFAULT_PROGRAM_REQUIRED_STUDIES),
                "Completed Studies": ["Method Comparison", "Precision", "Accuracy"],
                "In Progress Studies": ["Stability"],
                "Last Updated": "2026-06-18",
                "Required Samples": 60,
                "Received Samples": 42,
                "Processed Samples": 35,
            },
            {
                "Analyte": "Cortisol",
                "Assigned Analyzer": "Atellica IM 1600",
                "Assigned Reagent": "Cortisol Reagent Pack",
                "Required Studies": list(DEFAULT_PROGRAM_REQUIRED_STUDIES),
                "Completed Studies": [],
                "In Progress Studies": [],
                "Last Updated": "2026-06-18",
                "Required Samples": 60,
                "Received Samples": 0,
                "Processed Samples": 0,
            },
        ],
        "Last Updated": "2026-06-18",
        "Overall Status": "In Progress",
        "Final Package Generated": False,
    },
    {
        "Project Name": "Core 2 Microtainer Validation Program",
        "Program Name": "Core 2 Microtainer Validation Program",
        "Assay": "Expanded Microtainer Panel",
        "Assay / Biomarker": "Core 2 Microtainer Panel",
        "Program Owner": "Validation Team",
        "Candidate Specimen": "Microtainer serum",
        "Reference Specimen": "Venous serum",
        "Study Status": "Planned",
        "Status": "Not Started",
        "Start Date": "2026-06-18",
        "Target Completion Date": "2026-08-18",
        "Reviewer": "",
        "Notes": "",
        "Required Studies": list(DEFAULT_PROGRAM_REQUIRED_STUDIES),
        "Completed Studies": [],
        "Analytes": [
            {
                "Analyte": "Glucose",
                "Assigned Analyzer": "Atellica CH 930",
                "Assigned Reagent": "Glucose Hexokinase Reagent",
                "Required Studies": list(DEFAULT_PROGRAM_REQUIRED_STUDIES),
                "Completed Studies": [],
                "In Progress Studies": [],
                "Last Updated": "2026-06-18",
                "Required Samples": 80,
                "Received Samples": 0,
                "Processed Samples": 0,
            },
            {
                "Analyte": "CRP",
                "Assigned Analyzer": "Atellica CH 930",
                "Assigned Reagent": "CRP Wide Range Reagent",
                "Required Studies": list(DEFAULT_PROGRAM_REQUIRED_STUDIES),
                "Completed Studies": [],
                "In Progress Studies": [],
                "Last Updated": "2026-06-18",
                "Required Samples": 80,
                "Received Samples": 0,
                "Processed Samples": 0,
            },
        ],
        "Last Updated": "2026-06-18",
        "Overall Status": "Not Started",
        "Final Package Generated": False,
    },
]
DEFAULT_PLATFORM_SETTINGS = {
    "Organization Name": "",
    "Laboratory Name": "",
    "Department": "",
    "Organization Logo": "",
    "Analyst Name": "",
    "Reviewer Name": "",
    "Approver Name": "",
    "Report Footer": "Generated by the Scientific Validation Analytics Platform.",
    "Organization Branding": "",
    "Default Export Format": "PDF + HTML",
    "Logo Inclusion": "Include",
}


def inject_validation_styles() -> None:
    """Add shared styling for validation result cards and status badges."""

    st.markdown(
        """
        <style>
          :root {
            --svap-ink: var(--text-color, #202939);
            --svap-ink-strong: var(--text-color, #102a43);
            --svap-surface: var(--background-color, #ffffff);
            --svap-surface-muted: var(--secondary-background-color, #f8fafc);
            --svap-muted: color-mix(in srgb, var(--svap-ink) 70%, var(--svap-surface) 30%);
            --svap-soft: color-mix(in srgb, var(--svap-ink) 55%, var(--svap-surface) 45%);
            --svap-border: color-mix(in srgb, var(--svap-ink) 14%, var(--svap-surface) 86%);
            --svap-border-strong: color-mix(in srgb, var(--svap-ink) 24%, var(--svap-surface) 76%);
            --svap-primary: #102a43;
            --svap-success: #1f7a1f;
            --svap-success-bg: #e3f9e5;
            --svap-success-text: #1f7a1f;
            --svap-warning-bg: #fff8c5;
            --svap-warning-text: #7c4d00;
            --svap-followup-bg: #fff4e6;
            --svap-followup-text: #9a4d00;
            --svap-neutral-bg: #f0f4f8;
            --svap-neutral-text: #52606d;
            --svap-neutral-fill: #cbd5e1;
            --svap-radius: 8px;
            --svap-page-gap: 28px;
            --svap-section-gap: 34px;
          }
          .stApp,
          .block-container,
          div[data-testid="stMarkdownContainer"],
          div[data-testid="stMarkdownContainer"] p,
          div[data-testid="stMarkdownContainer"] li,
          div[data-testid="stMarkdownContainer"] label {
            color: var(--svap-ink);
          }
          .block-container {
            max-width: 1180px;
            padding-top: 3.1rem;
          }
          h1 {
            color: var(--svap-ink);
            font-size: 2.45rem !important;
            font-weight: 760 !important;
            letter-spacing: 0 !important;
            line-height: 1.08 !important;
            margin-bottom: 2.1rem !important;
          }
          h2 {
            color: var(--svap-ink);
            font-size: 1.55rem !important;
            font-weight: 760 !important;
            letter-spacing: 0 !important;
            line-height: 1.18 !important;
            margin-top: var(--svap-section-gap) !important;
            margin-bottom: 0.75rem !important;
          }
          h3 {
            color: var(--svap-ink);
            font-size: 1.08rem !important;
            font-weight: 760 !important;
            letter-spacing: 0 !important;
            line-height: 1.25 !important;
            margin-top: 1.35rem !important;
            margin-bottom: 0.55rem !important;
          }
          p, li, div[data-testid="stMarkdownContainer"] {
            letter-spacing: 0;
          }
          hr {
            border-color: var(--svap-border) !important;
            margin: var(--svap-page-gap) 0 !important;
          }
          div[data-testid="stExpander"] {
            border: 1px solid var(--svap-border-strong);
            border-radius: var(--svap-radius);
            box-shadow: none;
          }
          div[data-testid="stExpander"] details summary {
            min-height: 38px;
          }
          div[data-testid="stButton"] button,
          div[data-testid="stDownloadButton"] button {
            border-radius: var(--svap-radius);
            box-shadow: none;
            font-weight: 700;
            min-height: 38px;
          }
          div[data-testid="stButton"] button p,
          div[data-testid="stDownloadButton"] button p {
            color: inherit !important;
          }
          div[data-testid="stDataFrame"] {
            border-radius: var(--svap-radius);
          }
          .svap-card {
            border: 1px solid var(--svap-border-strong);
            border-radius: var(--svap-radius);
            background: var(--svap-surface);
            padding: 8px 10px;
            min-height: 72px;
            box-shadow: 0 1px 2px rgba(16, 42, 67, 0.04);
          }
          .svap-card-label {
            color: var(--svap-muted);
            font-size: 0.74rem;
            font-weight: 700;
            margin-bottom: 4px;
            text-transform: uppercase;
          }
          .svap-card-value {
            color: var(--svap-ink-strong);
            font-size: 1.05rem;
            font-weight: 700;
            line-height: 1.18;
            overflow-wrap: anywhere;
          }
          .svap-card-subtext {
            color: var(--svap-muted);
            font-size: 0.82rem;
            margin-top: 4px;
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
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            overflow: hidden;
            width: 100%;
          }
          .svap-status-table th, .svap-status-table td {
            border: 0;
            border-bottom: 1px solid #e5e7eb;
            padding: 10px 12px;
          }
          .svap-status-table th {
            background: #f8fafc;
            color: #102a43;
            font-size: 0.84rem;
            font-weight: 800;
          }
          .svap-status-table td {
            color: #1f2933;
            font-size: 0.9rem;
          }
          .svap-status-table tbody tr:last-child td {
            border-bottom: 0;
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
            background: var(--svap-success-bg);
            color: var(--svap-success-text);
          }
          .status-borderline {
            background: var(--svap-warning-bg);
            color: var(--svap-warning-text);
          }
          .status-fail {
            background: #ffe3e3;
            color: #c92a2a;
          }
          .status-neutral {
            background: var(--svap-neutral-bg);
            color: var(--svap-neutral-text);
          }
          .svap-page-header {
            border-bottom: 1px solid var(--svap-border-strong);
            margin: 0 0 var(--svap-page-gap);
            padding-bottom: 14px;
          }
          .svap-shell-brand {
            color: var(--svap-ink-strong);
            font-size: 0.95rem;
            font-weight: 820;
            line-height: 1.25;
            margin: 0 0 24px;
          }
          .svap-shell-brand-subtitle {
            color: var(--svap-muted);
            font-size: 0.76rem;
            font-weight: 650;
            margin-top: 3px;
          }
          .svap-page-kicker {
            color: var(--svap-muted);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
          }
          .svap-page-title {
            color: var(--svap-ink-strong);
            font-size: 1.45rem;
            font-weight: 750;
            line-height: 1.15;
            margin-top: 2px;
          }
          .svap-page-subtitle {
            color: var(--svap-muted);
            font-size: 0.9rem;
            margin-top: 3px;
          }
          .svap-workspace-header {
            margin: 0 0 6px;
          }
          .svap-workspace-title {
            color: #102a43;
            font-size: 1.35rem;
            font-weight: 750;
            line-height: 1.15;
          }
          .svap-workspace-meta {
            color: #52606d;
            font-size: 0.88rem;
            margin-top: 2px;
          }
          .svap-overview-context {
            color: #102a43;
            font-size: 0.96rem;
            font-weight: 650;
            margin-top: 8px;
          }
          .svap-workspace-lab-line {
            color: #52606d;
            font-size: 0.88rem;
            margin-top: 3px;
          }
          .svap-workspace-lab-line strong {
            color: #334155;
            font-weight: 760;
          }
          .svap-workspace-lab-separator {
            color: #9aa5b1;
            font-weight: 760;
          }
          .svap-execution-study-header {
            margin: 0 0 12px;
          }
          .svap-execution-study-title {
            color: var(--svap-ink-strong, #102a43);
            font-size: 1.55rem;
            font-weight: 820;
            letter-spacing: 0;
            line-height: 1.1;
          }
          .svap-execution-study-program {
            color: var(--svap-muted, #64748b);
            font-size: 0.95rem;
            font-weight: 560;
            line-height: 1.3;
            margin-top: 4px;
          }
          .svap-execution-context-grid {
            display: grid;
            gap: 22px;
            grid-template-columns: minmax(260px, 1.35fr) minmax(160px, 0.75fr) minmax(220px, 1fr);
            margin-top: 20px;
          }
          .svap-execution-meta-label {
            color: var(--svap-muted, #64748b);
            font-size: 0.72rem;
            font-weight: 780;
            letter-spacing: 0;
            line-height: 1.2;
            margin-bottom: 5px;
            text-transform: uppercase;
          }
          .svap-execution-scope-value {
            color: var(--svap-ink-strong, #102a43);
            font-size: 1.03rem;
            font-weight: 780;
            line-height: 1.25;
          }
          .svap-execution-meta-value {
            color: var(--svap-muted, #64748b);
            font-size: 0.9rem;
            font-weight: 650;
            line-height: 1.25;
          }
          .svap-study-progress-label {
            color: var(--svap-muted, #64748b);
            font-size: 0.72rem;
            font-weight: 780;
            letter-spacing: 0;
            margin: 18px 0 6px;
            text-transform: uppercase;
          }
          div[class*="st-key-execution_back_to_queue"] button {
            background: transparent;
            border-color: var(--svap-border, #d9dee7);
            box-shadow: none;
            color: var(--svap-muted, #64748b);
            font-size: 0.86rem;
            font-weight: 650;
            min-height: 36px;
            padding: 0 14px;
          }
          div[class*="st-key-execution_back_to_queue"] button:hover {
            background: var(--svap-surface-muted, #f6f8fb);
            border-color: var(--svap-border-strong, #cbd5e1);
            color: var(--svap-ink-strong, #102a43);
          }
          @media (max-width: 900px) {
            .svap-execution-context-grid {
              gap: 12px;
              grid-template-columns: 1fr;
            }
          }
          .svap-workspace-status-line {
            color: #102a43;
            font-size: 1rem;
            font-weight: 760;
            margin-top: 14px;
          }
          .svap-workspace-progress-line {
            color: #52606d;
            font-size: 0.9rem;
            margin-top: 2px;
          }
          .svap-overview-fields {
            display: grid;
            gap: 18px;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            margin: 20px 0 20px;
            max-width: 720px;
          }
          .svap-overview-label {
            color: #52606d;
            font-size: 0.82rem;
            font-weight: 750;
            margin-bottom: 4px;
          }
          .svap-overview-value {
            color: #102a43;
            font-size: 0.98rem;
            font-weight: 650;
          }
          .svap-overview-status {
            border-top: 1px solid #e5e7eb;
            border-bottom: 1px solid #e5e7eb;
            margin: 8px 0 22px;
            max-width: 720px;
            padding: 18px 0;
          }
          .svap-overview-status-title {
            color: #52606d;
            font-size: 0.82rem;
            font-weight: 750;
            margin-bottom: 6px;
          }
          .svap-overview-status-main {
            color: #102a43;
            font-size: 1.02rem;
            font-weight: 750;
            margin-bottom: 3px;
          }
          .svap-overview-status-sub {
            color: #52606d;
            font-size: 0.9rem;
          }
          .svap-workspace-summary {
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            background: #ffffff;
            padding: 10px 12px;
            margin: 2px 0 8px;
          }
          .svap-workspace-summary-title {
            color: #102a43;
            font-size: 0.92rem;
            font-weight: 750;
            margin-bottom: 4px;
          }
          .svap-workspace-summary-line {
            color: #52606d;
            font-size: 0.84rem;
            margin-top: 3px;
          }
          .svap-review-summary {
            margin: 2px 0 18px;
            padding: 0;
          }
          .svap-review-summary-top {
            align-items: center;
            display: flex;
            gap: 12px;
            justify-content: flex-start;
            margin-bottom: 12px;
          }
          .svap-review-summary-label {
            color: var(--svap-muted);
            font-size: 0.76rem;
            font-weight: 760;
            margin-bottom: 4px;
          }
          .svap-review-summary-value {
            color: var(--svap-ink-strong);
            font-size: 0.98rem;
            font-weight: 760;
            line-height: 1.3;
          }
          .svap-review-summary-title {
            color: var(--svap-ink-strong);
            font-size: 1.08rem;
            font-weight: 800;
            line-height: 1.25;
          }
          .svap-review-summary-question {
            margin-top: 10px;
            max-width: 900px;
          }
          .svap-review-audit-line {
            color: var(--svap-soft);
            font-size: 0.8rem;
            margin-top: 4px;
          }
          .svap-review-meta-line {
            color: var(--svap-muted);
            font-size: 0.85rem;
            margin-top: 4px;
          }
          .svap-review-decision-panel {
            border: 1px solid var(--svap-border-strong);
            border-radius: 8px;
            margin-top: 8px;
            padding: 14px 16px 16px;
          }
          .svap-review-decision-divider {
            border-top: 1px solid var(--svap-border);
            margin: 12px 0;
          }
          .svap-review-interpretation {
            color: var(--svap-ink);
            font-size: 0.96rem;
            line-height: 1.55;
            margin: 4px 0 16px;
            max-width: 980px;
          }
          .svap-review-interpretation p {
            margin: 0 0 8px;
          }
          .svap-review-interpretation p:last-child {
            margin-bottom: 0;
          }
          .svap-review-tabs {
            align-items: center;
            display: inline-flex;
            gap: 22px;
            margin: 14px 0 14px;
          }
          .svap-review-tab {
            border-bottom: 2px solid transparent;
            color: var(--svap-muted) !important;
            font-size: 0.86rem;
            font-weight: 720;
            padding: 4px 0 7px;
            text-decoration: none !important;
          }
          .svap-review-tab:hover {
            color: var(--svap-ink-strong) !important;
            text-decoration: none !important;
          }
          .svap-review-tab-active {
            border-bottom-color: var(--svap-ink-strong);
            color: var(--svap-ink-strong) !important;
            font-weight: 820;
          }
          .svap-review-list {
            border-top: 1px solid var(--svap-border);
            margin-top: 6px;
          }
.svap-review-list-header,
.svap-review-row {
  align-items: center;
  display: grid;
  gap: 22px;
  grid-template-columns: minmax(220px, 1.2fr) minmax(190px, 1fr) minmax(110px, 0.55fr) minmax(150px, 0.7fr) 18px;
}
.svap-review-list-header > *,
.svap-review-row > * {
  min-width: 0;
}
.svap-review-list-header {
  color: var(--svap-muted);
  font-size: 0.78rem;
  font-weight: 760;
            padding: 8px 4px;
          }
          .svap-review-row {
            border-bottom: 1px solid var(--svap-border);
            color: var(--svap-ink);
            cursor: pointer;
            min-height: 52px;
            padding: 8px 4px;
            text-decoration: none !important;
            transition: background 120ms ease, color 120ms ease, margin 120ms ease, padding 120ms ease;
          }
          .svap-review-row *,
          .svap-review-row:visited,
          .svap-review-row:hover,
          .svap-review-row:active {
            color: inherit;
            text-decoration: none !important;
          }
          .svap-review-row:hover {
            background: var(--svap-surface-muted);
            border-radius: 6px;
            box-shadow: inset 2px 0 0 var(--svap-ink-strong);
            color: var(--svap-ink-strong);
            margin-left: -12px;
            margin-right: -12px;
            padding-left: 16px;
            padding-right: 16px;
          }
          .svap-review-row:hover .svap-review-row-analyte {
            color: var(--svap-ink-strong);
          }
.svap-review-row:hover .svap-review-chevron {
  color: var(--svap-ink-strong);
  transform: translateX(2px);
}
.svap-review-row-identity {
  display: block;
}
.svap-review-row-analyte {
  color: var(--svap-ink-strong);
  display: block;
  font-size: 0.98rem;
  font-weight: 800;
}
.svap-review-row-secondary {
  color: var(--svap-muted);
  display: block;
  font-size: 0.84rem;
  margin-top: 2px;
}
.svap-review-row-study,
.svap-review-row-submitted {
            color: var(--svap-ink);
            font-size: 0.94rem;
            font-weight: 650;
          }
          .svap-review-chevron {
            color: var(--svap-soft);
            font-size: 1.1rem;
            text-align: right;
            transition: color 120ms ease, transform 120ms ease;
          }
.svap-review-status-pending {
  background: var(--svap-neutral-bg);
  color: var(--svap-neutral-text);
}
.svap-review-status-followup {
  background: var(--svap-warning-bg);
  color: var(--svap-warning-text);
}
.svap-review-status-approved {
  background: var(--svap-success-bg);
  color: var(--svap-success-text);
}
.svap-review-row .svap-review-status-pending {
  background: var(--svap-neutral-bg) !important;
  color: var(--svap-neutral-text) !important;
}
.svap-review-row .svap-review-status-followup {
  background: var(--svap-warning-bg) !important;
  color: var(--svap-warning-text) !important;
}
.svap-review-row .svap-review-status-approved {
  background: var(--svap-success-bg) !important;
  color: var(--svap-success-text) !important;
}
.svap-review-row .status-badge {
  display: inline-block;
  min-width: 132px;
  text-align: center;
}
          div[class*="st-key-review_rowcell_"] button {
            background: #ffffff;
            border: 0;
            border-bottom: 1px solid #e5e7eb;
            border-radius: 0;
            box-shadow: none;
            color: #1f2933;
            display: flex !important;
            justify-content: flex-start !important;
            min-height: 52px;
            padding: 8px 4px;
            text-align: left;
            width: 100%;
          }
          div[class*="st-key-review_rowcell_"] button:hover {
            background: #f8fafc;
            border-bottom: 1px solid #e5e7eb;
            color: #102a43;
          }
          div[class*="st-key-review_rowcell_status_"] button {
            color: #946200;
            font-weight: 760;
          }
          div[class*="st-key-review_queue_row_"] button {
            background: #ffffff;
            border: 0;
            border-bottom: 1px solid #e5e7eb;
            border-radius: 0;
            box-shadow: none;
            color: #1f2933;
            min-height: 52px;
            padding: 8px 4px;
            text-align: left;
            width: 100%;
          }
          div[class*="st-key-review_queue_row_"] button:hover {
            background: #f8fafc;
            border-bottom: 1px solid #e5e7eb;
            color: #102a43;
          }
          .svap-status-table tr.svap-criteria-fail td {
            background: #fff5f5;
          }
          .svap-status-strip {
            align-items: center;
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            display: grid;
            gap: 8px;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            padding: 8px 10px;
          }
          .svap-strip-label {
            color: #52606d;
            font-size: 0.72rem;
            font-weight: 750;
            text-transform: uppercase;
          }
          .svap-strip-value {
            color: #102a43;
            font-size: 0.92rem;
            font-weight: 700;
            margin-top: 2px;
            overflow-wrap: anywhere;
          }
          .svap-queue-row {
            align-items: center;
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            display: grid;
            gap: 6px;
            grid-template-columns: minmax(0, 1fr);
            margin-bottom: 3px;
            padding: 5px 8px;
          }
          .svap-queue-title {
            color: #102a43;
            font-size: 0.9rem;
            font-weight: 750;
          }
          .svap-queue-subtext {
            color: #52606d;
            font-size: 0.78rem;
            margin-top: 1px;
          }
          .svap-report-ready-item {
            margin: 0;
            padding: 4px 0;
          }
          .svap-report-ready-title {
            color: var(--svap-ink-strong);
            font-size: 1.08rem;
            font-weight: 820;
            margin-bottom: 6px;
          }
          .svap-report-ready-source-label {
            color: var(--svap-muted);
            font-size: 0.78rem;
            font-weight: 760;
            letter-spacing: 0;
            margin: 10px 0 2px;
            text-transform: none;
          }
          .svap-report-ready-meta {
            color: var(--svap-muted);
            font-size: 0.88rem;
            line-height: 1.5;
          }
          .svap-report-ready-status {
            color: var(--svap-success);
            font-weight: 780;
          }
          .svap-report-ready-check {
            color: var(--svap-success);
            font-weight: 780;
          }
          .svap-report-publication-summary {
            align-items: stretch;
            color: var(--svap-muted);
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin: 18px 0 36px;
            padding-bottom: 18px;
            border-bottom: 1px solid var(--svap-border);
          }
          .svap-report-publication-metric {
            align-items: baseline;
            border: 1px solid var(--svap-border);
            border-radius: 999px;
            display: inline-flex;
            gap: 10px;
            padding: 7px 14px;
          }
          .svap-report-publication-metric-label {
            color: var(--svap-muted);
            font-size: 0.84rem;
            font-weight: 720;
          }
          .svap-report-publication-metric-value {
            color: var(--svap-ink-strong);
            font-size: 0.98rem;
            font-weight: 820;
          }
          .svap-report-empty-state {
            color: var(--svap-muted);
            font-size: 0.94rem;
            padding: 6px 0 24px;
          }
          .svap-report-item-divider {
            border-bottom: 1px solid var(--svap-border);
            margin: 18px 0 22px;
          }
          .svap-report-section-divider {
            margin: 14px 0 18px;
          }
          .svap-published-report {
            border-bottom: 1px solid var(--svap-border);
            margin: 0 0 12px;
            padding: 4px 0 14px;
          }
          .svap-published-report-heading {
            align-items: baseline;
            display: flex;
            flex-wrap: wrap;
            gap: 14px;
            margin-bottom: 5px;
          }
          .svap-published-report-title {
            color: var(--svap-ink-strong);
            font-size: 1.06rem;
            font-weight: 820;
            margin-bottom: 0;
          }
          .svap-published-report-type {
            color: var(--svap-muted);
            font-size: 0.78rem;
            font-weight: 720;
            margin-bottom: 8px;
          }
          .svap-published-report-meta {
            color: var(--svap-muted);
            font-size: 0.86rem;
            line-height: 1.45;
            margin-top: 2px;
          }
          .svap-published-report-status-row {
            font-size: 0.86rem;
            margin-top: 8px;
          }
          .svap-report-status-published {
            color: var(--svap-success);
            font-size: 0.86rem;
            font-weight: 780;
          }
          .svap-report-status-superseded {
            color: var(--svap-muted);
            font-size: 0.86rem;
            font-weight: 760;
          }
          .svap-report-summary {
            color: #1f2933;
            font-size: 0.98rem;
            line-height: 1.55;
            margin: 10px 0 18px;
            max-width: 920px;
          }
          .svap-report-study-list {
            columns: 2;
            color: #1f2933;
            margin: 4px 0 20px;
          }
          .svap-report-study-list div {
            break-inside: avoid;
            padding: 5px 0;
          }
          .svap-report-content-section {
            border-top: 1px solid #e5e7eb;
            margin-top: 18px;
            padding-top: 18px;
          }
          .svap-report-content-title {
            color: #102a43;
            font-size: 1.05rem;
            font-weight: 820;
            margin-bottom: 8px;
          }
          .svap-report-content-meta {
            color: #52606d;
            font-size: 0.9rem;
            line-height: 1.5;
            margin-bottom: 10px;
          }
          .svap-report-decision {
            color: #1f2933;
            font-size: 0.92rem;
            line-height: 1.5;
            margin: 8px 0 14px;
          }
          .svap-portfolio-row {
            align-items: center;
            border-bottom: 1px solid #e5e7eb;
            display: grid;
            gap: 8px;
            grid-template-columns: 2.2fr 1fr 1fr 1.1fr 0.8fr;
            padding: 6px 4px;
          }
          .svap-portfolio-row:hover {
            background: #f8fafc;
          }
.svap-program-directory {
  border-top: 1px solid var(--svap-border);
  margin-top: 8px;
}
.svap-program-directory-header,
.svap-program-directory-row {
  align-items: center;
  display: grid !important;
  gap: 18px;
  grid-template-columns: minmax(260px, 1.45fr) minmax(280px, 1.35fr) minmax(260px, 1.25fr) minmax(120px, 0.65fr);
  width: 100%;
}
          .svap-program-directory-header {
            color: var(--svap-muted);
            font-size: 0.78rem;
  font-weight: 780;
  padding: 10px 4px;
}
.svap-program-directory-row {
  border-bottom: 1px solid var(--svap-border);
  box-sizing: border-box;
  color: var(--svap-ink) !important;
  cursor: pointer;
  min-height: 64px;
            padding: 12px 4px;
            position: relative;
  text-decoration: none !important;
  transition: background 0.12s ease, box-shadow 0.12s ease, color 0.12s ease;
}
.svap-program-directory-row > span {
  display: block;
  min-width: 0;
  pointer-events: none;
  position: relative;
  z-index: 1;
  width: 100%;
}
.svap-program-row-link {
  border-radius: 6px;
  inset: 0;
  position: absolute;
  z-index: 2;
}
.svap-program-row-link,
.svap-program-row-link:visited,
.svap-program-row-link:active,
.svap-program-row-link:hover {
  color: inherit !important;
  text-decoration: none !important;
}
.svap-program-row-link-label {
  clip: rect(0 0 0 0);
  height: 1px;
  overflow: hidden;
  position: absolute;
  white-space: nowrap;
  width: 1px;
}
.svap-program-directory-row:hover {
  background: var(--svap-surface-muted);
  border-radius: 6px;
  box-shadow: inset 2px 0 0 var(--svap-ink-strong);
  color: var(--svap-ink-strong);
  cursor: pointer;
          }
.svap-program-name {
  color: var(--svap-ink-strong);
  font-weight: 820;
}
          .svap-program-scope {
            color: var(--svap-ink);
            font-weight: 520;
          }
.svap-program-muted {
  color: var(--svap-muted);
}
.svap-program-directory-row > .svap-program-progress-cell {
  align-items: flex-start;
  display: flex !important;
  flex-direction: column;
  gap: 8px;
}
          .svap-progress-bar-row {
            align-items: center;
            display: flex;
            gap: 12px;
            max-width: 360px;
            width: 100%;
          }
          .svap-progress-bar-row .svap-execution-program-progress-bar,
          .svap-progress-bar-row .svap-execution-progress-bar {
            flex: 1;
          }
          .svap-program-progress-cell .svap-progress-bar-row {
            max-width: 340px;
          }
          .svap-program-progress-summary {
            align-items: baseline;
            display: flex;
            gap: 12px;
            white-space: nowrap;
          }
          .svap-program-status {
            border-radius: 999px;
            display: inline-block;
            font-size: 0.74rem;
            font-weight: 800;
            min-width: 92px;
            padding: 4px 9px;
            text-align: center;
          }
          .svap-program-status-active {
            background: var(--svap-warning-bg);
            color: var(--svap-warning-text);
          }
          .svap-program-status-not-started {
            background: #f0f4f8;
            color: #52606d;
          }
          .svap-program-status-completed {
            background: #e3f9e5;
            color: #1f7a1f;
          }
          .svap-program-status-review {
            background: #fff8c5;
            color: #946200;
          }
          .svap-card-button {
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            background: #ffffff;
            min-height: 96px;
            padding: 10px 12px;
          }
          .svap-check {
            color: #1f7a1f;
            font-weight: 800;
          }
          .svap-template-note {
            color: #697586;
            font-size: 0.9rem;
            margin: -4px 0 8px;
          }
          .svap-definition-section {
            border-bottom: 1px solid #e5e7eb;
            margin: 0 0 34px;
            padding: 0 0 30px;
          }
          .svap-definition-grid {
            display: grid;
            gap: 18px 48px;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            margin-top: 14px;
          }
          .svap-definition-label {
            color: #52606d;
            font-size: 0.84rem;
            font-weight: 700;
            margin-bottom: 5px;
          }
          .svap-definition-value {
            color: #1f2933;
            font-size: 0.98rem;
            font-weight: 500;
          }
          .svap-definition-heading {
            color: #202939;
            font-size: 1.08rem;
            font-weight: 760;
            margin: 28px 0 14px;
          }
          .svap-comparison-hero {
            align-items: end;
            display: grid;
            gap: 18px;
            grid-template-columns: minmax(0, 1fr) 34px minmax(0, 1fr);
            margin: 10px 0 4px;
          }
          .svap-comparison-label {
            color: #52606d;
            font-size: 0.84rem;
            font-weight: 700;
            margin-bottom: 6px;
          }
          .svap-comparison-value {
            color: #102a43;
            font-size: 1.12rem;
            font-weight: 750;
            line-height: 1.25;
          }
          .svap-comparison-arrow {
            color: #52606d;
            font-size: 1.35rem;
            font-weight: 700;
            padding-bottom: 2px;
            text-align: center;
          }
          .svap-validation-objective {
            color: #52606d;
            font-size: 0.92rem;
            line-height: 1.5;
            margin: 14px 0 4px;
            max-width: 760px;
          }
          .svap-validation-objective-label {
            color: #202939;
            font-size: 0.86rem;
            font-weight: 760;
            margin-bottom: 2px;
          }
          .svap-generated-work {
            color: #334155;
            font-size: 0.98rem;
            line-height: 1.9;
            margin: 2px 0 24px;
            max-width: 680px;
          }
          .svap-generated-work-line strong {
            color: #a86216;
            font-weight: 760;
          }
          .svap-next-step {
            border-top: 1px solid #e5e7eb;
            margin-top: 8px;
            max-width: 680px;
            padding-top: 24px;
          }
          .svap-next-step-label {
            color: #52606d;
            font-size: 0.86rem;
            font-weight: 760;
            letter-spacing: 0.01em;
            margin-bottom: 10px;
          }
          .svap-compact-summary {
            align-items: center;
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            display: flex;
            gap: 12px;
            justify-content: space-between;
            margin: 4px 0 8px;
            padding: 8px 10px;
          }
          .svap-compact-summary-main {
            color: #102a43;
            font-size: 0.9rem;
            font-weight: 750;
          }
          .svap-compact-summary-sub {
            color: #52606d;
            font-size: 0.82rem;
            margin-top: 2px;
          }
          .svap-decision-banner {
            border-radius: 6px;
            font-size: 0.95rem;
            font-weight: 800;
            line-height: 1;
            margin: 2px 0 4px;
            padding: 7px 10px;
          }
          .svap-decision-pass {
            background: #e3f9e5;
            color: #1f7a1f;
          }
          .svap-decision-incomplete {
            background: #e8f4fd;
            color: #1f4e79;
          }
          .svap-decision-warning {
            background: #fff8c5;
            color: #946200;
          }
          .svap-decision-fail {
            background: #ffe3e3;
            color: #c92a2a;
          }
          .svap-study-strip {
            color: #52606d;
            display: flex;
            flex-wrap: wrap;
            font-size: 0.82rem;
            gap: 8px;
            margin: 18px 0 14px;
          }
          .svap-study-strip span {
            white-space: nowrap;
          }
          .svap-study-complete {
            color: #1f7a1f;
            font-weight: 750;
          }
          .svap-study-current {
            color: #102a43;
            font-weight: 800;
          }
          .svap-study-pending {
            color: #7b8794;
          }
          div[class*="st-key-study_nav_"] button {
            background: transparent;
            border: 0;
            border-bottom: 2px solid transparent;
            border-radius: 0;
            box-shadow: none;
            color: var(--svap-muted, #64748b);
            min-height: 34px;
            padding: 0 2px 7px;
            text-align: center;
          }
          div[class*="st-key-study_nav_"] button:hover {
            background: transparent;
            border-bottom-color: var(--svap-border-strong, #cbd5e1);
            color: var(--svap-ink-strong, #102a43);
          }
          div[class*="st-key-study_nav_"] button:hover p {
            color: var(--svap-ink-strong, #102a43) !important;
          }
          div[class*="st-key-study_nav_active_"] button {
            border-bottom-color: var(--svap-ink-strong, #102a43);
            color: var(--svap-ink-strong, #102a43);
          }
          div[class*="st-key-study_nav_active_"] button p {
            color: var(--svap-ink-strong, #102a43) !important;
            font-weight: 850;
          }
          div[class*="st-key-study_nav_complete_"] button p {
            color: #2f7d45 !important;
            font-weight: 760;
          }
          div[class*="st-key-study_nav_pending_"] button p {
            color: var(--svap-muted, #64748b) !important;
            font-weight: 720;
          }
          div[class*="st-key-study_nav_"] button p {
            font-size: 0.86rem;
            font-weight: 760;
            letter-spacing: 0;
            line-height: 1.2;
            margin: 0;
            text-align: center;
            white-space: nowrap;
          }
          div.stButton > button[kind="primary"] {
            background-color: #102a43;
            border-color: #102a43;
            color: #ffffff;
          }
          div[data-testid="stButton"] button[kind="primary"],
          div[data-testid="stDownloadButton"] button[kind="primary"] {
            background-color: #102a43;
            border-color: #102a43;
            color: #ffffff !important;
          }
          div[data-testid="stButton"] button[kind="primary"] *,
          div[data-testid="stDownloadButton"] button[kind="primary"] * {
            color: #ffffff !important;
          }
          div.stButton > button[kind="primary"]:hover {
            background-color: #1f3f5b;
            border-color: #1f3f5b;
            color: #ffffff;
          }
          .svap-directory-table {
            border-top: 1px solid #d9e2ec;
            margin-top: 4px;
          }
          .svap-directory-header,
          .svap-directory-row {
            align-items: center;
            display: grid;
            gap: 16px;
            grid-template-columns: minmax(180px, 0.9fr) minmax(280px, 1.4fr) 24px;
          }
          .svap-directory-header {
            color: #1f2933;
            font-size: 0.82rem;
            font-weight: 750;
            padding: 8px 0;
          }
          .svap-directory-row {
            border-bottom: 1px solid #e5e7eb;
            color: #1f2933;
            min-height: 46px;
            padding: 9px 0;
            text-decoration: none;
          }
          .svap-directory-row:hover {
            background: #f8fafc;
            color: #102a43;
          }
          .svap-directory-analyte {
            font-weight: 700;
          }
          .svap-directory-status {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 7px;
          }
          .svap-status-pill {
            border-radius: 999px;
            display: inline-block;
            font-size: 0.76rem;
            font-weight: 750;
            line-height: 1;
            padding: 5px 8px;
          }
          .svap-status-ready {
            background: #e3f9e5;
            color: #1f7a1f;
          }
          .svap-status-progress {
            background: #fff8c5;
            color: #946200;
          }
          .svap-status-blocked {
            background: #ffe3e3;
            color: #c92a2a;
          }
          .svap-status-neutral {
            background: #f0f4f8;
            color: #52606d;
          }
          .svap-status-context {
            color: #52606d;
            font-size: 0.86rem;
          }
          .svap-protocol-summary {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            margin: 10px 0 12px;
            padding: 14px 16px;
          }
          .svap-protocol-summary-success {
            border-left: 4px solid #2f9e44;
          }
          .svap-protocol-eyebrow {
            color: #1f7a1f;
            font-size: 0.88rem;
            font-weight: 800;
            margin-bottom: 7px;
          }
          .svap-protocol-title {
            color: #202939;
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.35;
          }
          .svap-protocol-subtext {
            color: #697586;
            font-size: 0.88rem;
            margin-top: 5px;
          }
          .svap-results-header {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            margin: 8px 0 16px;
            overflow: hidden;
          }
          .svap-results-decision {
            background: #e3f9e5;
            color: #1f7a1f;
            font-size: 1rem;
            font-weight: 850;
            padding: 10px 14px;
          }
          .svap-results-decision-fail {
            background: #ffe3e3;
            color: #c92a2a;
          }
          .svap-results-decision-incomplete {
            background: #e8f4fd;
            color: #1f4e79;
          }
          .svap-results-body {
            padding: 14px;
          }
          .svap-results-label {
            color: #52606d;
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
          }
          .svap-results-finding {
            color: #202939;
            font-size: 0.98rem;
            font-weight: 650;
            margin-top: 4px;
          }
          .svap-study-result {
            border-bottom: 1px solid #e5e7eb;
            margin: 8px 0 18px;
            padding: 0 0 16px;
          }
          .svap-study-result-status {
            color: #1f7a1f;
            font-size: 1.45rem;
            font-weight: 850;
            letter-spacing: 0;
            line-height: 1.15;
            margin-bottom: 8px;
          }
          .svap-study-result-status-fail {
            color: #c92a2a;
          }
          .svap-study-result-status-incomplete {
            color: #1f4e79;
          }
          .svap-study-result-finding {
            color: #202939;
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.45;
            margin-bottom: 5px;
          }
          .svap-study-result-meta {
            color: #697586;
            font-size: 0.92rem;
            line-height: 1.4;
          }
          .svap-study-interpretation {
            color: #202939;
            font-size: 1rem;
            line-height: 1.65;
            margin: 4px 0 18px;
            max-width: 980px;
          }
          .svap-study-interpretation p {
            margin: 0 0 10px;
          }
          .svap-study-interpretation p:last-child {
            margin-bottom: 0;
          }
          .svap-study-complete {
            border-top: 1px solid #d9e2ec;
            margin-top: 22px;
            padding-top: 22px;
          }
          .svap-study-next-label {
            color: #697586;
            font-size: 0.82rem;
            font-weight: 800;
            margin-top: 14px;
            text-transform: uppercase;
          }
          .svap-study-next-stage {
            color: #202939;
            font-size: 1.05rem;
            font-weight: 800;
            margin: 2px 0 12px;
          }
          .svap-metric-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            margin: 6px 0 18px;
          }
          .svap-metric-cell {
            border-bottom: 1px solid #d9e2ec;
            padding: 8px 14px 10px 0;
          }
          .svap-metric-label {
            color: #52606d;
            font-size: 0.76rem;
            font-weight: 800;
            margin-bottom: 5px;
          }
          .svap-metric-value {
            color: #202939;
            font-size: 0.98rem;
            font-weight: 800;
          }
          .svap-evidence-note {
            color: #52606d;
            font-size: 0.88rem;
            margin: -4px 0 20px;
          }
          .svap-figure-heading {
            border-top: 1px solid #e5e7eb;
            color: #202939;
            font-size: 0.98rem;
            font-weight: 800;
            margin: 24px 0 8px;
            padding-top: 18px;
          }
          .svap-interpretation-panel {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            color: #202939;
            font-size: 0.96rem;
            line-height: 1.55;
            margin: 8px 0 16px;
            padding: 16px 18px;
          }
          .svap-interpretation-panel p {
            margin: 0 0 10px;
          }
          .svap-interpretation-panel p:last-child {
            margin-bottom: 0;
          }
          div[class*="st-key-execution_rowcell_"] button,
          div[class*="st-key-execution_chevron_"] button {
            background: #ffffff;
            border: 0;
            border-bottom: 1px solid #e5e7eb;
            border-radius: 0;
            box-shadow: none;
            color: #1f2933;
            display: flex !important;
            justify-content: flex-start !important;
            min-height: 58px;
            padding: 10px 14px;
            text-align: left !important;
          }
          div[class*="st-key-execution_rowcell_"] button:hover,
          div[class*="st-key-execution_chevron_"] button:hover {
            background: #f8fafc;
            border-bottom-color: #d9e2ec;
            color: #102a43;
          }
          div[class*="st-key-execution_rowcell_"] button p,
          div[class*="st-key-execution_chevron_"] button p {
            font-size: 0.95rem;
            line-height: 1.25;
            margin: 0 !important;
            text-align: left !important;
            width: 100% !important;
          }
          div[class*="st-key-execution_rowcell_"] button div,
          div[class*="st-key-execution_rowcell_"] button span {
            justify-content: flex-start !important;
            text-align: left !important;
          }
          div[class*="st-key-execution_chevron_"] button {
            color: #52606d;
            font-size: 1.1rem;
            justify-content: flex-end !important;
            padding-right: 10px;
            text-align: right;
          }
          div[class*="st-key-execution_chevron_"] button p {
            text-align: right !important;
          }
          .svap-execution-directory {
            border-top: 1px solid var(--svap-border-strong);
            margin-top: 18px;
          }
          .svap-execution-program-progress {
            margin: 18px 0 26px;
            max-width: 360px;
          }
          .svap-execution-program-progress-row {
            display: block;
            margin-bottom: 7px;
          }
          .svap-execution-program-progress-value {
            color: var(--svap-ink-strong);
            font-size: 1rem;
            font-weight: 800;
          }
          .svap-execution-program-progress-percent {
            color: var(--svap-muted);
            display: inline-block;
            font-size: 0.88rem;
            font-weight: 720;
            margin-top: 0;
            white-space: nowrap;
          }
          .svap-execution-program-progress-bar {
            background: var(--svap-border);
            border-radius: 999px;
            height: 4px;
            overflow: hidden;
            width: 100%;
          }
          .svap-execution-program-progress-fill {
            background: var(--svap-ink-strong);
            border-radius: inherit;
            height: 100%;
          }
          .svap-execution-directory-header,
          .svap-execution-directory-row {
            align-items: center;
            display: grid;
            gap: 28px;
            grid-template-columns: minmax(240px, 1.2fr) minmax(340px, 1.65fr) 18px;
          }
          .svap-execution-directory-header {
            color: var(--svap-muted);
            font-size: 0.82rem;
            font-weight: 800;
            padding: 12px 0 10px;
          }
          .svap-execution-directory-row {
            border-bottom: 1px solid var(--svap-border);
            color: var(--svap-ink) !important;
            min-height: 72px;
            padding: 16px 0;
            text-decoration: none !important;
            transition: background 120ms ease, color 120ms ease, margin 120ms ease, padding 120ms ease;
          }
          .svap-execution-directory-row:visited,
          .svap-execution-directory-row:active,
          .svap-execution-directory-row:hover {
            color: var(--svap-ink) !important;
            text-decoration: none !important;
          }
          .svap-execution-directory-row * {
            text-decoration: none !important;
          }
          .svap-execution-directory-row:hover {
            background: var(--svap-surface-muted);
            border-radius: 6px;
            box-shadow: inset 2px 0 0 var(--svap-ink-strong);
            cursor: pointer;
            margin-left: -12px;
            margin-right: -12px;
            padding-left: 16px;
            padding-right: 16px;
          }
          .svap-execution-directory-row:hover .svap-execution-chevron {
            color: var(--svap-ink-strong);
            transform: translateX(2px);
          }
          .svap-execution-directory-row:hover .svap-execution-analyte {
            color: var(--svap-ink-strong);
          }
          .svap-execution-analyte {
            color: var(--svap-ink-strong);
            font-size: 0.98rem;
            font-weight: 760;
            letter-spacing: 0;
          }
          .svap-execution-artifact-subtext {
            color: var(--svap-muted);
            font-size: 0.84rem;
            font-weight: 620;
            margin-top: 3px;
          }
          .svap-execution-progress {
            display: flex;
            flex-direction: column;
            gap: 8px;
          }
          .svap-execution-progress-line {
            align-items: center;
            display: flex;
            gap: 10px;
          }
          .svap-execution-status {
            border-radius: 999px;
            display: inline-flex;
            font-size: 0.73rem;
            font-weight: 850;
            line-height: 1;
            padding: 5px 9px;
          }
          .svap-execution-status-progress {
            background: var(--svap-warning-bg);
            color: var(--svap-warning-text);
          }
          .svap-execution-status-start {
            background: var(--svap-neutral-bg);
            color: var(--svap-neutral-text);
          }
          .svap-execution-status-completed {
            background: var(--svap-success-bg);
            color: var(--svap-success-text);
          }
          .svap-execution-progress-count {
            color: var(--svap-ink-strong);
            font-size: 1.02rem;
            font-weight: 850;
          }
          .svap-execution-status-text {
            color: var(--svap-muted);
            font-size: 0.86rem;
            font-weight: 760;
          }
          .svap-execution-status-text-progress {
            color: var(--svap-warning-text);
          }
          .svap-execution-status-text-start {
            color: var(--svap-neutral-text);
          }
          .svap-execution-status-text-completed {
            color: var(--svap-success-text);
          }
          .svap-execution-progress-bar {
            background: var(--svap-border);
            border-radius: 999px;
            display: block;
            height: 4px;
            max-width: 280px;
            overflow: hidden;
            width: 100%;
          }
          .svap-execution-progress-fill {
            background: var(--svap-ink-strong);
            border-radius: inherit;
            display: block;
            height: 100%;
          }
          .svap-execution-progress-fill-completed {
            background: var(--svap-success-text);
          }
          .svap-execution-progress-fill-start {
            background: var(--svap-neutral-fill);
          }
          .svap-execution-analyzer {
            color: var(--svap-muted);
            font-size: 0.92rem;
            font-weight: 600;
          }
          .svap-execution-chevron {
            color: var(--svap-soft);
            font-size: 1.05rem;
            text-align: right;
            transition: transform 120ms ease, color 120ms ease;
          }
          div[class*="st-key-program_directory_select_"] button {
            background: transparent;
            border: 0;
            box-shadow: none;
            color: #1f2933;
            font-weight: 750;
            min-height: 28px;
            padding: 0;
            text-align: left;
          }
          div[class*="st-key-program_directory_select_"] button:hover {
            background: transparent;
            border: 0;
            color: #102a43;
            text-decoration: underline;
          }
          div[class*="st-key-program_directory_select_"] button p {
            font-size: 0.95rem;
            text-align: left;
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
    legacy_program_names = {
        str(project.get("Program Name") or project.get("Project Name"))
        for project in st.session_state.validation_projects
    }
    if not any("Analytes" in project for project in st.session_state.validation_projects) and legacy_program_names <= {
        "HbA1c Validation Program",
        "Ferritin Validation Program",
        "Vitamin D Validation Program",
    }:
        st.session_state.validation_projects = [project.copy() for project in DEFAULT_PROJECTS]
    st.session_state.validation_projects = [
        normalize_program(project) for project in st.session_state.validation_projects
    ]
    if "reports_library" not in st.session_state:
        st.session_state.reports_library = [
            {
                "Report Name": "HbA1c Full Validation Package",
                "Project": "Core 1 Microtainer Validation Program",
                "Study Type": "Validation Reports",
                "Date": "2026-06-18",
                "Analyst": st.session_state.platform_settings.get("Analyst Name", ""),
                "Format": "HTML / PDF",
            },
            {
                "Report Name": "HbA1c Microtainer Validation Report",
                "Project": "Core 1 Microtainer Validation Program",
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
                "Project": "Core 1 Microtainer Validation Program",
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
    if "review_packages" not in st.session_state:
        st.session_state.review_packages = {}
    if "study_execution_state" not in st.session_state:
        st.session_state.study_execution_state = {}
    if not st.session_state.get("_execution_state_loaded"):
        load_persisted_study_execution_states()
        st.session_state._execution_state_loaded = True
    if "approval_history" not in st.session_state:
        st.session_state.approval_history = []
    if "program_timeline" not in st.session_state:
        st.session_state.program_timeline = [
            {
                "Timestamp": "2026-06-18 00:00",
                "Program": "Core 1 Microtainer Validation Program",
                "Study": "Validation Package",
                "Activity": "Package generated",
                "Details": "HbA1c analyte validation package generated and submitted for scientific review.",
            }
        ]


def platform_settings() -> dict[str, str]:
    """Return current platform settings."""

    initialize_platform_state()
    settings = st.session_state.platform_settings
    for key, value in DEFAULT_PLATFORM_SETTINGS.items():
        settings.setdefault(key, value)
    settings["Report Logo"] = settings.get("Organization Logo", settings.get("Report Logo", ""))
    settings["Organization Logo"] = settings.get("Organization Logo", settings.get("Report Logo", ""))
    settings["Default Report Format"] = settings.get("Default Export Format", settings.get("Default Report Format", "PDF + HTML"))
    settings["Default Export Format"] = settings.get("Default Export Format", settings.get("Default Report Format", "PDF + HTML"))
    settings.setdefault("Address", "")
    return st.session_state.platform_settings


def render_page_header(title: str, subtitle: str = "", kicker: str = "") -> None:
    """Render a standardized platform page header."""

    kicker_html = f'<div class="svap-page-kicker">{escape(kicker)} · {escape(APP_VERSION)}</div>' if kicker else ""
    st.markdown(
        f"""
        <div class="svap-page-header">
          {kicker_html}
          <div class="svap-page-title">{escape(title)}</div>
          <div class="svap-page-subtitle">{escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shell_brand() -> None:
    """Render persistent product identity in the application shell."""

    st.sidebar.markdown(
        f"""
        <div class="svap-shell-brand">
          {escape(APP_TITLE)}
          <div class="svap-shell-brand-subtitle">Validation lifecycle</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, status: str | None = None) -> None:
    """Render a compact validation metric card."""

    value_html = status_badge_html(value) if status else escape(value).replace("\n", "<br>")
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

    required = list(project.get("Required Studies") or DEFAULT_PROGRAM_REQUIRED_STUDIES)
    completed = len(project.get("Completed Studies", []))
    total = len(required)
    percent = (completed / total) * 100 if total else 0.0
    return completed, total, percent


def normalize_analyte(analyte: dict[str, object], fallback_name: str = "") -> dict[str, object]:
    """Backfill analyte-level validation workstream fields without overriding configured metadata."""

    analyte_name = str(analyte.get("Analyte") or analyte.get("Assay") or fallback_name or "Analyte")
    instrument_defaults = {
        "Ferritin": ("Beckman Access 2", "Ferritin Reagent Pack"),
        "Vitamin D": ("Beckman Access 2", "Vitamin D Total Reagent Pack"),
        "HbA1c": ("LC-MS/MS", "HbA1c Variant Analysis Reagent"),
        "TSH": ("Beckman Access 2", "TSH3 Ultra Reagent Pack"),
        "Cortisol": ("Beckman Access 2", "Cortisol Reagent Pack"),
        "Vitamin B12": ("Beckman Access 2", "Vitamin B12 Reagent Pack"),
        "Glucose": ("Beckman AU480", "Glucose Hexokinase Reagent"),
        "CRP": ("Beckman AU480", "CRP Wide Range Reagent"),
    }
    default_analyzer, default_reagent = instrument_defaults.get(analyte_name, ("Analyzer not assigned", "Reagent not assigned"))
    analyte.setdefault("Analyte", analyte_name)
    if not str(analyte.get("Assigned Analyzer") or "").strip():
        analyte["Assigned Analyzer"] = default_analyzer
    if not str(analyte.get("Assigned Reagent") or "").strip():
        analyte["Assigned Reagent"] = default_reagent
    analyte.setdefault("Required Studies", list(DEFAULT_PROGRAM_REQUIRED_STUDIES))
    analyte.setdefault("Completed Studies", [])
    analyte.setdefault("Analyzed Studies", [])
    analyte.setdefault("In Progress Studies", [])
    analyte.setdefault("Review Studies", [])
    analyte.setdefault("Blocked Studies", [])
    analyte.setdefault("Last Updated", str(date.today()))
    analyte.setdefault("Required Samples", 80)
    analyte.setdefault("Received Samples", 0)
    analyte.setdefault("Processed Samples", 0)
    return analyte


def normalize_program_definition(program: dict[str, object]) -> dict[str, object]:
    """Return the formal validation program definition."""

    analytes = [
        str(analyte.get("Analyte", "")).strip()
        for analyte in list(program.get("Analytes") or [])
        if str(analyte.get("Analyte", "")).strip()
    ]
    raw_required = list(program.get("Required Studies") or DEFAULT_PROGRAM_REQUIRED_STUDIES)
    required_studies = [study for study in raw_required if study in TRUE_VALIDATION_STUDY_TYPES]
    if not required_studies:
        required_studies = list(DEFAULT_PROGRAM_REQUIRED_STUDIES)
    definition = dict(program.get("Program Definition") or {})
    template_name = str(definition.get("Validation Template") or program.get("Validation Template") or "")
    template = VALIDATION_TEMPLATES.get(template_name, {})
    definition.setdefault("Program Name", program.get("Program Name", "Validation Program"))
    definition.setdefault("Validation Template", template_name or "Microtainer Equivalency Template")
    definition.setdefault("Program Category", program.get("Program Category", template.get("Program Category", "Specimen Equivalency Validation")))
    definition.setdefault("Validation Context", program.get("Validation Context", template.get("Validation Context", "Serum vs Microtainer Equivalency")))
    definition.setdefault("Candidate Specimen", program.get("Candidate Specimen", "Candidate specimen not specified"))
    definition.setdefault("Reference Specimen", program.get("Reference Specimen", "Reference specimen not specified"))
    definition.setdefault("Description", program.get("Notes", ""))
    definition.setdefault("Analytes", analytes)
    definition.setdefault("Required Study Types", list(template.get("Required Study Types", required_studies)))
    definition["Required Study Types"] = [
        study for study in list(definition.get("Required Study Types") or required_studies)
        if study in TRUE_VALIDATION_STUDY_TYPES
    ]
    definition.setdefault("Optional Study Types", list(template.get("Optional Study Types", [])))
    definition["Optional Study Types"] = [
        study for study in list(definition.get("Optional Study Types") or [])
        if study in TRUE_VALIDATION_STUDY_TYPES and study not in definition["Required Study Types"]
    ]
    definition.setdefault("Sample Requirements", template.get("Sample Requirements", ""))
    definition.setdefault("Review Workflow", template.get("Review Workflow", ""))
    definition.setdefault("Report Structure", template.get("Report Structure", ""))
    definition.setdefault("Analyte Study Requirements", {})
    return definition


def analyte_required_study_types(program: dict[str, object], analyte_name: str) -> list[str]:
    """Return required study types for an analyte from the program definition."""

    definition = normalize_program_definition(program)
    overrides = dict(definition.get("Analyte Study Requirements") or {})
    return list(overrides.get(analyte_name) or definition.get("Required Study Types") or DEFAULT_PROGRAM_REQUIRED_STUDIES)


def generated_study_instances(program: dict[str, object]) -> list[dict[str, str]]:
    """Generate study instances from the formal program definition."""

    definition = normalize_program_definition(program)
    instances: list[dict[str, str]] = []
    for analyte_name in list(definition.get("Analytes") or []):
        for study_type in analyte_required_study_types(program, str(analyte_name)):
            instances.append(
                {
                    "Program": str(definition.get("Program Name", "")),
                    "Analyte": str(analyte_name),
                    "Study Type": str(study_type),
                    "Study Instance": f"{analyte_name} – {study_type}",
                }
            )
    return instances


def apply_program_definition(program: dict[str, object]) -> dict[str, object]:
    """Synchronize analyte workstreams with the formal program definition."""

    definition = normalize_program_definition(program)
    existing = {
        str(analyte.get("Analyte")): normalize_analyte(dict(analyte), fallback_name=str(analyte.get("Analyte", "")))
        for analyte in list(program.get("Analytes") or [])
    }
    analytes: list[dict[str, object]] = []
    for analyte_name in list(definition.get("Analytes") or []):
        analyte = existing.get(str(analyte_name), {"Analyte": str(analyte_name)})
        required = analyte_required_study_types(program, str(analyte_name))
        analyte["Required Studies"] = required
        for key in ("Completed Studies", "Analyzed Studies", "In Progress Studies", "Review Studies", "Blocked Studies"):
            analyte[key] = [study for study in list(analyte.get(key) or []) if study in required]
        analytes.append(normalize_analyte(analyte, fallback_name=str(analyte_name)))
    program["Analytes"] = analytes
    program["Program Definition"] = definition
    program["Required Studies"] = list(definition.get("Required Study Types") or DEFAULT_PROGRAM_REQUIRED_STUDIES)
    program["Generated Studies"] = generated_study_instances(program)
    return program


def normalize_program(program: dict[str, object]) -> dict[str, object]:
    """Backfill validation program fields, including analyte workstreams."""

    program_name = str(program.get("Program Name") or program.get("Project Name") or "Validation Program")
    assay = str(program.get("Assay / Biomarker") or program.get("Assay") or "")
    program.setdefault("Program Name", program_name)
    program.setdefault("Project Name", program_name)
    program.setdefault("Assay / Biomarker", assay)
    program.setdefault("Assay", assay)
    program.setdefault("Program Owner", "")
    program.setdefault("Validation Template", "Microtainer Equivalency Template")
    program.setdefault("Program Category", "Specimen Equivalency Validation")
    program.setdefault("Validation Context", "Serum vs Microtainer Equivalency")
    if "microtainer" in str(program.get("Validation Context", "")).lower() or "microtainer" in program_name.lower():
        program.setdefault("Candidate Specimen", "Microtainer serum")
        program.setdefault("Reference Specimen", "Venous serum")
    elif "dbs" in str(program.get("Validation Context", "")).lower() or "dbs" in program_name.lower():
        program.setdefault("Candidate Specimen", "Dried blood spot")
        program.setdefault("Reference Specimen", "Venous whole blood")
    else:
        program.setdefault("Candidate Specimen", "Candidate specimen not specified")
        program.setdefault("Reference Specimen", "Reference specimen not specified")
    program.setdefault("Status", program.get("Study Status", "Not Started"))
    program.setdefault("Start Date", str(date.today()))
    program.setdefault("Target Completion Date", str(date.today()))
    program.setdefault("Reviewer", "")
    program.setdefault("Notes", "")
    program.setdefault("Required Studies", list(DEFAULT_PROGRAM_REQUIRED_STUDIES))
    program.setdefault("Completed Studies", [])
    program.setdefault("Final Package Generated", False)
    program.setdefault("Last Updated", str(date.today()))
    if not program.get("Analytes"):
        program["Analytes"] = [
            normalize_analyte(
                {
                    "Analyte": assay or program_name.replace(" Validation Program", ""),
                    "Required Studies": [
                        study for study in list(program.get("Required Studies") or DEFAULT_PROGRAM_REQUIRED_STUDIES)
                        if study in TRUE_VALIDATION_STUDY_TYPES
                    ],
                    "Completed Studies": list(program.get("Completed Studies") or []),
                    "In Progress Studies": list(program.get("In Progress Studies") or []),
                    "Last Updated": program.get("Last Updated", str(date.today())),
                    "Required Samples": program.get("Required Samples", 80),
                    "Received Samples": program.get("Received Samples", 0),
                    "Processed Samples": program.get("Processed Samples", 0),
                },
                fallback_name=assay or program_name,
            )
        ]
    else:
        program["Analytes"] = [
            normalize_analyte(dict(analyte), fallback_name=assay)
            for analyte in list(program.get("Analytes") or [])
        ]
    program["Program Definition"] = normalize_program_definition(program)
    return apply_program_definition(program)


def program_analytes(program: dict[str, object]) -> list[dict[str, object]]:
    """Return analyte workstreams assigned to a validation program."""

    return list(normalize_program(program).get("Analytes") or [])


def scientific_title_case(value: object) -> str:
    """Return lab-facing title case without changing missing values."""

    text = str(value or "").strip()
    if not text:
        return ""
    return " ".join(part.capitalize() for part in text.split())


def specimen_context_text(program: dict[str, object]) -> str:
    """Return concise candidate/reference specimen context."""

    program = normalize_program(program)
    candidate = scientific_title_case(program.get("Candidate Specimen") or "Candidate specimen not specified")
    reference = scientific_title_case(program.get("Reference Specimen") or "Reference specimen not specified")
    return f"{candidate} vs {reference}"


def relative_activity_text(value: object) -> str:
    """Return relative activity text for directory rows."""

    parsed = pd.to_datetime(str(value or ""), errors="coerce")
    if pd.isna(parsed):
        return str(value or "")
    activity_date = parsed.date()
    days = (date.today() - activity_date).days
    if days == 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    if days > 1:
        return f"{days} days ago"
    return activity_date.strftime("%b %d, %Y")


def validation_type_label(program: dict[str, object]) -> str:
    """Return a concise validation type for directory views."""

    category = str(normalize_program(program).get("Program Category") or "")
    if "specimen" in category.lower() or "equivalency" in category.lower():
        return "Specimen Equivalency"
    return category or "Validation"


def program_directory_status(program: dict[str, object], metrics: dict[str, object]) -> str:
    """Return lifecycle-oriented program status for the directory."""

    status = str(normalize_program(program).get("Status") or metrics.get("Readiness") or "")
    if status in {"Not Started", "Planning", "Planned"}:
        return "Not Started"
    if status in {"Completed", "Complete", "Archived"}:
        return status
    if str(metrics.get("Readiness")) in {"Review Pending", "Ready for Approval"}:
        return "Review"
    return "Active" if int(metrics.get("Completed", 0)) or status in {"In Progress", "Active"} else status or "Planning"


def program_status_badge_html(status: str) -> str:
    """Return compact status badge markup for the program directory."""

    normalized = str(status or "Active")
    status_class = {
        "Active": "active",
        "Not Started": "not-started",
        "Completed": "completed",
        "Complete": "completed",
        "Review": "review",
    }.get(normalized, "active")
    return f'<span class="svap-program-status svap-program-status-{status_class}">{escape(normalized)}</span>'


def analyte_scientific_context(analyte: dict[str, object]) -> str:
    """Return concise analyzer/reagent assignment text."""

    analyte = normalize_analyte(dict(analyte), fallback_name=str(analyte.get("Analyte", "")))
    analyzer = str(analyte.get("Assigned Analyzer") or "Analyzer not assigned")
    reagent = str(analyte.get("Assigned Reagent") or "Reagent not assigned")
    return f"Analyzer: {analyzer} · Reagent: {reagent}"


def find_program_and_analyte(program_name: str, analyte_name: str) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Find normalized program and analyte records by name."""

    projects = [normalize_program(project) for project in st.session_state.validation_projects]
    program = next((project for project in projects if str(project.get("Program Name")) == program_name), None)
    if not program:
        return None, None
    analyte = next((item for item in program_analytes(program) if str(item.get("Analyte")) == analyte_name), None)
    return program, analyte


def package_analyte_assignment(item: dict[str, object]) -> tuple[str, str]:
    """Return analyzer/reagent assignment for a package item."""

    program = normalize_program(dict(item.get("Program Record") or {}))
    analyte_name = str(item.get("Analyte", ""))
    analyte = next((record for record in program_analytes(program) if str(record.get("Analyte")) == analyte_name), None)
    if analyte:
        return str(analyte.get("Assigned Analyzer", "")), str(analyte.get("Assigned Reagent", ""))
    return "", ""


def program_required_studies(program: dict[str, object]) -> list[str]:
    """Return assigned studies for a validation program."""

    return list(normalize_program(program).get("Required Studies") or DEFAULT_PROGRAM_REQUIRED_STUDIES)


def study_type_to_workspace_name(study_type: str) -> str:
    """Map program definition study types to implemented workspace names."""

    return {
        "Precision": "Precision Study",
        "Accuracy": "Accuracy Study",
        "Linearity": "Linearity Study",
        "Stability": "Stability Study",
    }.get(study_type, study_type)


def workspace_name_to_study_type(workspace_name: str) -> str:
    """Map implemented workspace labels back to program definition study types."""

    return {
        "Precision Study": "Precision",
        "Accuracy Study": "Accuracy",
        "Linearity Study": "Linearity",
        "Stability Study": "Stability",
    }.get(workspace_name, workspace_name)


def find_program_index(program_name: str) -> int | None:
    """Return the session-state index for a validation program."""

    for index, program in enumerate(st.session_state.validation_projects):
        normalized = normalize_program(program)
        if str(normalized.get("Program Name")) == program_name:
            return index
    return None


def find_study_execution_record(program_name: str, analyte_name: str, study_type: str) -> dict[str, object] | None:
    """Return an analyte-scoped execution record for a generated study."""

    canonical_type = workspace_name_to_study_type(study_type)
    for record in get_lifecycle_records():
        if (
            str(record.get("Project")) == program_name
            and str(record.get("Analyte")) == analyte_name
            and str(record.get("Study Type")) == canonical_type
        ):
            return record
    return None


def execution_status(program: dict[str, object], analyte: dict[str, object], study_type: str) -> str:
    """Return the execution status for a generated validation study."""

    program_name = str(program.get("Program Name", ""))
    analyte_name = str(analyte.get("Analyte", ""))
    canonical_type = workspace_name_to_study_type(study_type)
    record = find_study_execution_record(program_name, analyte_name, canonical_type)
    if record:
        return normalize_lifecycle_status(record.get("Status"))
    if canonical_type in set(analyte.get("Completed Studies") or []):
        return "Approved"
    if canonical_type in set(analyte.get("Analyzed Studies") or []):
        return "Analysis Complete"
    if canonical_type in set(analyte.get("Review Studies") or []):
        return "Scientific Review"
    if canonical_type in set(analyte.get("In Progress Studies") or []):
        return "Draft"
    if canonical_type in set(analyte.get("Blocked Studies") or []):
        return "Rejected"
    return "Not Started"


def execution_status_label(status: str) -> str:
    """Return a user-facing status label for generated work."""

    normalized = normalize_lifecycle_status(status)
    if status == "Not Started" or normalized == "Not Started":
        return "Not Started"
    if normalized in {"Draft", "Data Uploaded"}:
        return "In Progress"
    if normalized in EXECUTION_FINISHED_STATES:
        return "Completed"
    if normalized in {"Rejected", "Requires Revision", "Follow-Up Required"}:
        return "In Progress"
    return normalized


def execution_action_label(status: str) -> str:
    """Return the primary action for a study at its current execution state."""

    normalized = execution_status_label(status)
    if normalized == "Not Started":
        return "Upload Dataset"
    if normalized == "In Progress":
        return "Continue Analysis"
    if normalized == "Completed":
        return "Review Results"
    return "View Results"


def update_analyte_execution_state(
    program_name: str,
    analyte_name: str,
    study_type: str,
    status: str,
) -> None:
    """Synchronize analyte progress lists after a study execution status changes."""

    program_index = find_program_index(program_name)
    if program_index is None:
        return
    program = normalize_program(st.session_state.validation_projects[program_index])
    canonical_type = workspace_name_to_study_type(study_type)
    for analyte in program.get("Analytes", []):
        if str(analyte.get("Analyte")) != analyte_name:
            continue
        for key in ("Completed Studies", "Analyzed Studies", "In Progress Studies", "Review Studies", "Blocked Studies"):
            analyte[key] = [study for study in list(analyte.get(key) or []) if study != canonical_type]
        normalized = normalize_lifecycle_status(status)
        if normalized in EXECUTION_COMPLETE_STATES:
            analyte.setdefault("Completed Studies", []).append(canonical_type)
            analyte.setdefault("Analyzed Studies", []).append(canonical_type)
        elif normalized in EXECUTION_REVIEW_STATES:
            analyte.setdefault("Review Studies", []).append(canonical_type)
            analyte.setdefault("Analyzed Studies", []).append(canonical_type)
        elif normalized == "Analysis Complete":
            analyte.setdefault("Analyzed Studies", []).append(canonical_type)
        elif normalized == "Rejected":
            analyte.setdefault("Blocked Studies", []).append(canonical_type)
        else:
            analyte.setdefault("In Progress Studies", []).append(canonical_type)
        analyte["Last Updated"] = date.today().isoformat()
    program["Last Updated"] = date.today().isoformat()
    st.session_state.validation_projects[program_index] = normalize_program(program)


def ensure_study_execution_record(
    program: dict[str, object],
    analyte_name: str,
    study_type: str,
    status: str = "Draft",
) -> dict[str, object]:
    """Create or update the lifecycle record for generated validation work."""

    initialize_platform_state()
    program_name = str(program.get("Program Name", "Validation Program"))
    canonical_type = workspace_name_to_study_type(study_type)
    existing = find_study_execution_record(program_name, analyte_name, canonical_type)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    study_name = f"{analyte_name} {canonical_type}"
    study_id = f"{program_name}-{analyte_name}-{canonical_type}".lower().replace(" ", "-").replace("/", "-")
    record = {
        "Study ID": study_id,
        "Project": program_name,
        "Analyte": analyte_name,
        "Study Name": study_name,
        "Study Type": canonical_type,
        "Assay": analyte_name,
        "Analyst": str(platform_settings().get("Analyst Name", "")),
        "Completion Date": date.today().isoformat() if normalize_lifecycle_status(status) in EXECUTION_PROGRESS_COMPLETE_STATES else "",
        "Status": normalize_lifecycle_status(status),
        "Version": int(existing.get("Version", 1)) if existing else 1,
        "Reviewer Comments": str(existing.get("Reviewer Comments", "")) if existing else "",
        "Submitted Date": str(existing.get("Submitted Date", "")) if existing else "",
        "Last Updated": timestamp,
        "Locked": normalize_lifecycle_status(status) in {"Locked", "Archived"},
    }
    if existing:
        existing.update(record)
    else:
        st.session_state.study_lifecycle_records.append(record)
        record_program_activity(program_name, study_name, "Study workspace opened", f"{canonical_type} execution workspace created.")
    update_analyte_execution_state(program_name, analyte_name, canonical_type, str(record["Status"]))
    return record


def analyte_execution_metrics(program: dict[str, object], analyte: dict[str, object]) -> dict[str, object]:
    """Return execution progress metrics for an analyte workstream."""

    required = list(analyte.get("Required Studies") or program_required_studies(program))
    program_name = str(program.get("Program Name", ""))
    analyte_name = str(analyte.get("Analyte", ""))
    statuses = {study: execution_status(program, analyte, study) for study in required}
    completed = sum(
        1
        for study in required
        if study_execution_has_completed_analysis(program_name, analyte_name, study)
    )
    approved = sum(1 for status in statuses.values() if normalize_lifecycle_status(status) in EXECUTION_COMPLETE_STATES)
    active = sum(
        1
        for status in statuses.values()
        if execution_status_label(status) == "In Progress"
    )
    submitted = sum(1 for status in statuses.values() if normalize_lifecycle_status(status) in EXECUTION_REVIEW_STATES)
    if completed == len(required) and required:
        workstream_status = "Completed"
    elif completed or submitted:
        workstream_status = "In Progress"
    elif active or any(execution_status_label(status) != "Not Started" for status in statuses.values()):
        workstream_status = "In Progress"
    else:
        workstream_status = "Not Started"
    return {
        "Required": len(required),
        "Completed": completed,
        "Approved": approved,
        "Analyzed": completed,
        "Submitted": submitted,
        "In Progress": active,
        "Remaining": max(0, len(required) - completed),
        "Status": workstream_status,
        "Statuses": statuses,
    }


def study_work_item(program: dict[str, object], analyte: dict[str, object], study_type: str) -> dict[str, object]:
    """Return action-oriented workflow guidance for a generated study."""

    status = execution_status(program, analyte, study_type)
    normalized = execution_status_label(status)
    study_button_name = f"{study_type} Study" if study_type in {"Stability", "Precision", "Accuracy", "Linearity"} else study_type
    if normalized == "Not Started":
        return {
            "Study": study_type,
            "Status": "Awaiting Data",
            "Next Action": study_type,
            "Action Label": "Upload Dataset",
            "Primary Blocker": "No study data uploaded",
            "Priority": "High Priority",
            "Route": "study",
        }
    if normalized == "In Progress":
        return {
            "Study": study_type,
            "Status": "Analysis In Progress",
            "Next Action": study_type,
            "Action Label": f"Continue {study_button_name}",
            "Primary Blocker": "Analysis incomplete",
            "Priority": "Medium Priority",
            "Route": "study",
        }
    if normalized == "Analyzed":
        return {
            "Study": study_type,
            "Status": "Analyzed",
            "Next Action": study_type,
            "Action Label": "Submit For Review",
            "Primary Blocker": "Awaiting review submission",
            "Priority": "Ready For Review",
            "Route": "study",
        }
    if normalized == "Submitted For Review":
        return {
            "Study": study_type,
            "Status": "Submitted For Review",
            "Next Action": study_type,
            "Action Label": "Awaiting Review",
            "Primary Blocker": "Awaiting reviewer decision",
            "Priority": "Awaiting Review",
            "Route": "study",
        }
    if normalized == "Approved":
        return {
            "Study": study_type,
            "Status": "Approved",
            "Next Action": study_type,
            "Action Label": "View Approved Study",
            "Primary Blocker": "No Blockers",
            "Priority": "Approved",
            "Route": "study",
        }
    return {
        "Study": study_type,
        "Status": "Approved",
        "Next Action": study_type,
        "Action Label": "View Approved Study",
        "Primary Blocker": "No Blockers",
        "Priority": "Approved",
        "Route": "study",
    }


def analyte_work_item(program: dict[str, object], analyte: dict[str, object]) -> dict[str, object]:
    """Return the next best action for an analyte workstream."""

    metrics = analyte_execution_metrics(program, analyte)
    required = list(analyte.get("Required Studies") or program_required_studies(program))
    required_samples = int(analyte.get("Required Samples", 0))
    received_samples = int(analyte.get("Received Samples", 0))
    samples_missing = max(0, required_samples - received_samples)
    if samples_missing > 0:
        sample_blocker = "Samples Missing"
        sample_blocker_short = f"{samples_missing} Samples Missing"
        sample_blocker_detail = (
            f"Samples Required: {required_samples}<br>"
            f"Samples Received: {received_samples}<br>"
            f"Samples Missing: {samples_missing}"
        )
    else:
        sample_blocker = ""
        sample_blocker_short = ""
        sample_blocker_detail = "No Blockers"
    for priority in ["Ready For Review", "High Priority", "Medium Priority", "Awaiting Review"]:
        for study_type in required:
            item = study_work_item(program, analyte, study_type)
            if item["Priority"] == priority:
                blocker = sample_blocker or str(item["Primary Blocker"])
                blocker_detail = sample_blocker_detail
                if not sample_blocker:
                    record = find_study_execution_record(str(program.get("Program Name", "")), str(analyte["Analyte"]), str(item["Study"]))
                    if item["Priority"] == "Awaiting Review" and record:
                        reviewer = str(record.get("Reviewer") or platform_settings().get("Reviewer Name", "") or "Unassigned")
                        submitted = str(record.get("Submitted Date") or record.get("Last Updated") or "")
                        submitted_date = pd.to_datetime(submitted, errors="coerce")
                        pending_days = (pd.Timestamp(date.today()) - submitted_date.normalize()).days if not pd.isna(submitted_date) else 0
                        blocker_detail = (
                            f"Reviewer: {escape(reviewer)}<br>"
                            f"Review Assigned: {escape(submitted or 'Not assigned')}<br>"
                            f"Pending: {max(0, pending_days)} Days"
                        )
                    else:
                        blocker_detail = str(item["Primary Blocker"])
                return {
                    **item,
                    "Analyte": str(analyte["Analyte"]),
                    "Progress": f"{metrics['Analyzed']} / {metrics['Required']} Analyzed",
                    "Completed": metrics["Completed"],
                    "Analyzed": metrics["Analyzed"],
                    "Required": metrics["Required"],
                    "Primary Blocker": blocker,
                    "Blocker Summary": sample_blocker_short or str(item["Primary Blocker"]),
                    "Blocker Detail": blocker_detail,
                    "Samples Required": required_samples,
                    "Samples Received": received_samples,
                    "Samples Missing": samples_missing,
                }
    return {
        "Analyte": str(analyte["Analyte"]),
        "Study": "Complete" if metrics["Required"] and metrics["Approved"] == metrics["Required"] else "Validation Work",
        "Status": "Complete" if metrics["Required"] and metrics["Approved"] == metrics["Required"] else "No Work Defined",
        "Next Action": "None" if metrics["Required"] and metrics["Approved"] == metrics["Required"] else "Define Validation Work",
        "Action Label": "Open Workspace" if metrics["Required"] else "Define Program",
        "Primary Blocker": "No Blockers" if metrics["Required"] else "No studies defined",
        "Blocker Summary": "No Blockers" if metrics["Required"] else "No studies defined",
        "Blocker Detail": "No Blockers" if metrics["Required"] else "No studies defined",
        "Priority": "Approved" if metrics["Required"] else "High Priority",
        "Route": "study",
        "Progress": f"{metrics['Analyzed']} / {metrics['Required']} Analyzed",
        "Completed": metrics["Completed"],
        "Analyzed": metrics["Analyzed"],
        "Required": metrics["Required"],
        "Samples Required": required_samples,
        "Samples Received": received_samples,
        "Samples Missing": samples_missing,
    }


def execution_analyte_status(metrics: dict[str, object]) -> str:
    """Return the simplified analyte status used by Execution."""

    required = int(metrics.get("Required", 0))
    completed = int(metrics.get("Completed", 0))
    active = int(metrics.get("In Progress", 0))
    if required and completed == required:
        return "Completed"
    if completed or active:
        return "In Progress"
    return "Not Started"


def execution_progress_text(metrics: dict[str, object]) -> str:
    """Return compact study completion progress for an analyte."""

    required = max(0, int(metrics.get("Required", 0)))
    completed = max(0, int(metrics.get("Completed", 0)))
    return f"{completed}/{required} studies"


def execution_progress_indicator(metrics: dict[str, object]) -> str:
    """Return a compact text progress indicator for execution completion."""

    required = max(0, int(metrics.get("Required", 0)))
    completed = max(0, int(metrics.get("Completed", 0)))
    if required <= 0:
        return "□□□□□□"
    filled = max(0, min(6, round((completed / required) * 6)))
    return f"{'■' * filled}{'□' * (6 - filled)}"


def execution_landing_status(analyte: dict[str, object], metrics: dict[str, object]) -> str:
    """Return the analyte-level status shown on the Execution landing page."""

    return execution_analyte_status(metrics)


def next_execution_study(program: dict[str, object], analyte: dict[str, object]) -> str:
    """Return the next study that still has scientific work remaining."""

    required = list(analyte.get("Required Studies") or program_required_studies(program))
    program_name = str(program.get("Program Name", ""))
    analyte_name = str(analyte.get("Analyte", ""))
    for study in required:
        if not study_execution_has_completed_analysis(program_name, analyte_name, study):
            return study
    return "Complete"


def execution_queue_blocker(program: dict[str, object], analyte: dict[str, object], active_study: str, metrics: dict[str, object]) -> str:
    """Return the most useful operational blocker text for the queue."""

    if active_study == "Complete" or (metrics.get("Required") and metrics.get("Approved") == metrics.get("Required")):
        return "Ready for reporting"
    status = execution_status_label(execution_status(program, analyte, active_study))
    required_samples = int(analyte.get("Required Samples", 0))
    received_samples = int(analyte.get("Received Samples", 0))
    missing_samples = max(0, required_samples - received_samples)
    if status == "Not Started":
        return f"{missing_samples} samples missing" if missing_samples else "Awaiting dataset"
    if status == "In Progress":
        return "Running"
    if status == "Analyzed":
        return "Ready to submit"
    if status == "Submitted For Review":
        return "Awaiting review"
    return status


def normalize_lifecycle_status(status: object) -> str:
    """Normalize legacy and current lifecycle status labels."""

    value = str(status or "Draft")
    return {
        "Analyzed": "Analysis Complete",
        "In Progress": "Draft",
        "Complete": "Approved",
        "Review Pending": "Submitted for Review",
        "Ready for Review": "Ready For Review",
        "Scientific Review": "Scientific Review",
        "Returned for Revision": "Requires Revision",
        "Request Follow-Up": "Follow-Up Required",
        "Follow-Up Required": "Follow-Up Required",
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
    statuses: dict[str, str] = {}
    required_count = 0
    approved = 0
    pending = 0
    under_review = 0
    active = 0
    for analyte in list(program.get("Analytes") or []):
        analyte = normalize_analyte(dict(analyte), fallback_name=str(program.get("Assay / Biomarker", "")))
        completed_studies = set(analyte.get("Completed Studies") or [])
        in_progress_studies = set(analyte.get("In Progress Studies") or [])
        review_studies = set(analyte.get("Review Studies") or [])
        blocked_studies = set(analyte.get("Blocked Studies") or [])
        for study in list(analyte.get("Required Studies") or DEFAULT_PROGRAM_REQUIRED_STUDIES):
            required_count += 1
            key = f"{analyte['Analyte']} | {study}"
            if study in completed_studies:
                statuses[key] = "Approved"
                approved += 1
            elif study in review_studies:
                statuses[key] = "Submitted for Review"
                pending += 1
                active += 1
            elif study in blocked_studies:
                statuses[key] = "Rejected"
                active += 1
            elif study in in_progress_studies:
                statuses[key] = "Draft"
                active += 1
            else:
                statuses[key] = "Draft"
    completed = approved
    remaining = max(0, required_count - approved)
    percent = (approved / required_count) * 100 if required_count else 0.0
    if bool(program.get("Final Package Generated")):
        readiness = "Completed"
    elif required_count > 0 and approved == required_count:
        readiness = "Ready for Final Package"
    elif under_review > 0:
        readiness = "Ready for Approval"
    elif pending > 0:
        readiness = "Review Pending"
    elif active > 0 or approved > 0:
        readiness = "In Progress"
    else:
        readiness = "Not Started"
    return {
        "Required": required_count,
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
        "Not Started": "status-borderline",
        "In Progress": "status-borderline",
        "Complete": "status-pass",
        "Draft": "status-borderline",
        "Data Uploaded": "status-borderline",
        "Analysis Complete": "status-borderline",
        "Ready For Review": "status-borderline",
        "Pending Review": "status-borderline",
        "Scientific Review": "status-borderline",
        "Submitted for Review": "status-borderline",
        "Under Review": "status-borderline",
        "Approved": "status-pass",
        "Report Generated": "status-pass",
        "Generated": "status-pass",
        "Ready For Generation": "status-pass",
        "Not Ready": "status-borderline",
        "Locked": "status-pass",
        "Archived": "status-pass",
        "Rejected": "status-fail",
        "Requires Revision": "status-fail",
        "Follow-Up Required": "status-borderline",
        "APPROVED": "status-pass",
        "REJECTED": "status-fail",
        "FOLLOW-UP REQUIRED": "status-borderline",
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


def study_state_key(program_name: str, analyte_name: str, study_type: str) -> str:
    """Return a stable key for persisted analyte-study execution state."""

    canonical_type = workspace_name_to_study_type(study_type)
    return f"{program_name}::{analyte_name}::{canonical_type}"


def execution_state_path(state_key: str) -> Path:
    """Return the durable storage path for a study execution state."""

    return EXECUTION_STATE_DIR / f"{hashlib.sha256(state_key.encode('utf-8')).hexdigest()}.pkl"


def save_study_execution_state_key(state_key: str, state: dict[str, object]) -> None:
    """Persist one study execution state to disk."""

    EXECUTION_STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"State Key": state_key, "State": state}
    try:
        with execution_state_path(state_key).open("wb") as handle:
            pickle.dump(payload, handle)
    except Exception:
        # UI state should never crash the scientific workflow; keep in-memory state if disk persistence fails.
        return


def load_persisted_study_execution_states() -> None:
    """Restore persisted execution datasets, mappings, quality results, and analyses."""

    if not EXECUTION_STATE_DIR.exists():
        return
    for path in EXECUTION_STATE_DIR.glob("*.pkl"):
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        state_key = str(payload.get("State Key", ""))
        state = payload.get("State")
        if state_key and isinstance(state, dict):
            st.session_state.study_execution_state.setdefault(state_key, state)


def get_study_execution_state(program_name: str, analyte_name: str, study_type: str) -> dict[str, object]:
    """Return persisted execution state for a program/analyte/study combination."""

    initialize_platform_state()
    key = study_state_key(program_name, analyte_name, study_type)
    return st.session_state.study_execution_state.setdefault(key, {})


def persist_study_execution_value(
    program_name: str,
    analyte_name: str,
    study_type: str,
    key: str,
    value: object,
) -> None:
    """Persist one execution value for a program/analyte/study combination."""

    state = get_study_execution_state(program_name, analyte_name, study_type)
    state[key] = value
    save_study_execution_state_key(study_state_key(program_name, analyte_name, study_type), state)


def reset_analysis_for_new_dataset(program_name: str, analyte_name: str, study_type: str, source_name: str) -> bool:
    """Clear stale analysis when a different dataset is loaded for a study."""

    state = get_study_execution_state(program_name, analyte_name, study_type)
    previous_source = str(state.get("Dataset Source", ""))
    if previous_source and source_name and previous_source != source_name:
        for key in ("Analysis Result", "Decision", "Analysis Complete", "Execution Complete", "Submitted For Review", "Review Package"):
            state.pop(key, None)
        st.session_state.pop(persisted_analysis_state_key(program_name, analyte_name, study_type), None)
        st.session_state.pop(f"execution_result_{program_name}_{analyte_name}_{study_type}", None)
        state["Dataset Changed"] = True
        save_study_execution_state_key(study_state_key(program_name, analyte_name, study_type), state)
        return True
    return False


def reset_execution_for_uploaded_dataset(program: dict[str, object], analyte_name: str, study_type: str) -> None:
    """Reset downstream execution/review artifacts when a dataset is newly uploaded."""

    program_name = str(program.get("Program Name", ""))
    state = get_study_execution_state(program_name, analyte_name, study_type)
    for key in ("Analysis Result", "Decision", "Analysis Complete", "Execution Complete", "Submitted For Review", "Review Package"):
        state.pop(key, None)
    save_study_execution_state_key(study_state_key(program_name, analyte_name, study_type), state)
    st.session_state.pop(persisted_analysis_state_key(program_name, analyte_name, study_type), None)
    st.session_state.pop(f"execution_result_{program_name}_{analyte_name}_{study_type}", None)
    st.session_state.pop(f"execution_results_reviewed_{program_name}_{analyte_name}_{study_type}", None)
    ensure_study_execution_record(program, analyte_name, study_type, "Data Uploaded")


def mark_data_uploaded_if_needed(program: dict[str, object], analyte_name: str, study_type: str) -> None:
    """Mark a study as data uploaded without downgrading completed analysis/review state."""

    current = normalize_lifecycle_status(execution_status(program, {"Analyte": analyte_name}, study_type))
    if current in {"Not Started", "Draft", "Data Uploaded"}:
        ensure_study_execution_record(program, analyte_name, study_type, "Data Uploaded")


def persisted_analysis_state_key(program_name: str, analyte_name: str, study_type: str) -> str:
    """Return the legacy-compatible analysis result key for a study."""

    return f"study_result_{program_name}_{analyte_name}_{study_type}"


def study_execution_has_completed_analysis(program_name: str, analyte_name: str, study_type: str) -> bool:
    """Return True only when a study has a persisted completed analysis artifact."""

    state = get_study_execution_state(program_name, analyte_name, study_type)
    has_result = isinstance(state.get("Analysis Result"), dict)
    if workspace_name_to_study_type(study_type) == "Method Comparison":
        return bool(state.get("Execution Complete")) and has_result
    return bool(state.get("Execution Complete") or state.get("Analysis Complete")) and has_result


def next_incomplete_study(program: dict[str, object], analyte_name: str) -> str | None:
    """Return the next required study that is not complete within Execution."""

    program_name = str(program["Program Name"])
    analyte = next((item for item in program_analytes(program) if str(item.get("Analyte")) == analyte_name), {})
    for required_study in list(analyte.get("Required Studies") or program_required_studies(program)):
        if not study_execution_has_completed_analysis(program_name, analyte_name, str(required_study)):
            return str(required_study)
    return None


def analyte_execution_complete(program: dict[str, object], analyte_name: str) -> bool:
    """Return True when every required study for an analyte is complete in Execution."""

    return next_incomplete_study(program, analyte_name) is None


def program_execution_complete(program: dict[str, object]) -> bool:
    """Return True when every analyte in a program has completed execution."""

    analytes = program_analytes(program)
    return bool(analytes) and all(analyte_execution_complete(program, str(analyte.get("Analyte", ""))) for analyte in analytes)


def safe_review_package_id(study_id: str) -> str:
    """Return a filesystem-safe review package identifier."""

    return hashlib.sha256(study_id.encode("utf-8")).hexdigest()


def review_package_path(study_id: str) -> Path:
    """Return the persisted package path for a lifecycle study id."""

    return REVIEW_PACKAGE_DIR / f"{safe_review_package_id(study_id)}.pkl"


def save_review_package_record(study_record: dict[str, object], package: dict[str, object]) -> None:
    """Persist a frozen review package and queue record to disk."""

    study_id = str(study_record.get("Study ID", ""))
    if not study_id:
        return
    REVIEW_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "Study ID": study_id,
        "Study Record": dict(study_record),
        "Review Package": package,
    }
    with review_package_path(study_id).open("wb") as handle:
        pickle.dump(payload, handle)


def load_review_package_record(study_id: str) -> dict[str, object] | None:
    """Load a frozen review package payload from disk."""

    path = review_package_path(study_id)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def load_persisted_review_packages() -> None:
    """Restore submitted review packages and queue records after app restart."""

    initialize_platform_state()
    if not REVIEW_PACKAGE_DIR.exists():
        return
    records_by_id = {
        str(record.get("Study ID", "")): record
        for record in st.session_state.study_lifecycle_records
        if str(record.get("Study ID", ""))
    }
    for path in REVIEW_PACKAGE_DIR.glob("*.pkl"):
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        study_id = str(payload.get("Study ID", ""))
        package = payload.get("Review Package")
        record = payload.get("Study Record")
        if study_id and isinstance(package, dict):
            st.session_state.review_packages.setdefault(study_id, package)
        if study_id and isinstance(record, dict):
            record["Review Package"] = package
            if study_id in records_by_id:
                records_by_id[study_id].update(record)
            else:
                st.session_state.study_lifecycle_records.append(record)
                records_by_id[study_id] = record


def package_audit_events(study: dict[str, object], package: dict[str, object]) -> list[dict[str, str]]:
    """Build a concise review audit timeline."""

    metadata = package.get("Study Metadata", {}) if isinstance(package, dict) else {}
    reviewer = str(study.get("Reviewer") or platform_settings().get("Reviewer Name", "") or "")
    events = [
        ("Created", study.get("Created Date") or study.get("Completion Date") or study.get("Last Updated"), study.get("Study Name", "")),
        ("Dataset Uploaded", metadata.get("Dataset Uploaded") or study.get("Completion Date"), metadata.get("Input Dataset Hash", "")),
        ("Analysis Executed", metadata.get("Analysis Timestamp") or study.get("Completion Date"), metadata.get("Analysis Run ID", "")),
        ("Submitted For Review", study.get("Submitted Date") or metadata.get("Submitted Date"), metadata.get("Analyst", "")),
        ("Assigned Reviewer", study.get("Last Updated"), reviewer),
        ("Reviewer Decision", study.get("Review Date"), normalize_lifecycle_status(study.get("Status"))),
        (
            "Approved / Rejected",
            study.get("Review Date") if normalize_lifecycle_status(study.get("Status")) in {"Approved", "Rejected", "Follow-Up Required"} else "",
            normalize_lifecycle_status(study.get("Status")),
        ),
    ]
    return [
        {
            "Event": event,
            "Timestamp": format_review_date(timestamp) if timestamp else "",
            "Details": str(details or ""),
        }
        for event, timestamp, details in events
        if timestamp
    ]


def build_submitted_review_package(
    *,
    program: dict[str, object],
    analyte: dict[str, object],
    study_type: str,
    decision: str,
    criteria_table: pd.DataFrame,
    key_results: dict[str, str],
    interpretation: str,
    figures: dict[str, object],
    supporting_tables: dict[str, pd.DataFrame],
) -> dict[str, object]:
    """Snapshot submitted analyte study evidence for read-only review."""

    program_name = str(program.get("Program Name", ""))
    analyte_name = str(analyte.get("Analyte", ""))
    analysis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    study_id = f"{program_name}-{analyte_name}-{study_type}".lower().replace(" ", "-").replace("/", "-")
    analysis_run_id = f"{study_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    dataset_hash = ""
    for table_name, table in supporting_tables.items():
        if isinstance(table, pd.DataFrame) and ("Data" in table_name or "Dataset" in table_name):
            dataset_hash = hashlib.sha256(table.to_csv(index=False).encode("utf-8")).hexdigest()[:16]
            break
    review_artifacts = pd.DataFrame(
        [
            {"Artifact": "Uploaded Dataset Metadata", "Value": dataset_hash or "Captured in submitted package"},
            {"Artifact": "Analysis Timestamp", "Value": analysis_timestamp},
            {"Artifact": "Study Version", "Value": "1"},
            {"Artifact": "Acceptance Criteria Version", "Value": "Current at submission"},
            {"Artifact": "Software Version", "Value": "LabOS Execution Workspace"},
            {"Artifact": "Analysis Run ID", "Value": analysis_run_id},
            {"Artifact": "Input Dataset Hash", "Value": dataset_hash or "Not available"},
            {"Artifact": "Audit Trail", "Value": "Execution submitted frozen review package"},
        ]
    )
    supporting_evidence = {
        title: table.copy() if isinstance(table, pd.DataFrame) else table
        for title, table in supporting_tables.items()
    }
    supporting_evidence["Review Artifacts"] = review_artifacts
    return {
        "Study Metadata": {
            "Program": program_name,
            "Analyte": analyte_name,
            "Study Type": study_type,
            "Submitted Date": date.today().isoformat(),
            "Analyst": str(platform_settings().get("Analyst Name", "")),
            "Analysis Timestamp": analysis_timestamp,
            "Analysis Run ID": analysis_run_id,
            "Input Dataset Hash": dataset_hash,
        },
        "Decision": decision,
        "Key Results": dict(key_results),
        "Acceptance Criteria": criteria_table.copy() if isinstance(criteria_table, pd.DataFrame) else pd.DataFrame(),
        "Interpretation": interpretation,
        "Visualizations": dict(figures),
        "Supporting Evidence": supporting_evidence,
    }


def persist_review_package(study_record: dict[str, object], package: dict[str, object]) -> None:
    """Persist submitted review artifacts on the lifecycle record and package index."""

    initialize_platform_state()
    study_id = str(study_record.get("Study ID", ""))
    if not study_id:
        return
    study_record["Review Package"] = package
    st.session_state.review_packages[study_id] = package
    save_review_package_record(study_record, package)


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
    execution_program = str(st.session_state.get("execution_program") or "")
    execution_analyte = str(st.session_state.get("execution_analyte") or "")
    execution_study = workspace_name_to_study_type(str(st.session_state.get("execution_study") or ""))
    if execution_program == str(record["Project"]) and execution_analyte and execution_study == canonical_type:
        program_index = find_program_index(execution_program)
        if program_index is not None:
            ensure_study_execution_record(
                normalize_program(st.session_state.validation_projects[program_index]),
                execution_analyte,
                canonical_type,
                "Scientific Review",
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
    if existing and normalize_lifecycle_status(existing.get("Status")) in {"Submitted for Review", "Under Review", "Approved", "Report Generated", "Locked", "Archived"}:
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
        "Status": "Analysis Complete",
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
    execution_program = str(st.session_state.get("execution_program") or "")
    execution_analyte = str(st.session_state.get("execution_analyte") or "")
    execution_study = workspace_name_to_study_type(str(st.session_state.get("execution_study") or ""))
    if execution_program == str(record["Project"]) and execution_analyte and execution_study == canonical_type:
        program_index = find_program_index(execution_program)
        if program_index is not None:
            ensure_study_execution_record(
                normalize_program(st.session_state.validation_projects[program_index]),
                execution_analyte,
                canonical_type,
                "Analysis Complete",
            )


def render_submit_for_review(study_type: str, metadata: dict[str, object]) -> None:
    """Render the post-analysis submit-for-review action."""

    record_study_analyzed(study_type, metadata)
    st.subheader("Validation Lifecycle")
    st.caption("Completed analyses can be submitted to Validation Review.")
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
    if current_status in {"Approved", "Report Generated", "Locked", "Archived"}:
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
    """Render the operational validation program command center."""

    initialize_platform_state()
    projects = [normalize_program(project) for project in st.session_state.validation_projects]
    settings = platform_settings()

    def display_date(value: object, long: bool = False) -> str:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return str(value or "")
        return parsed.strftime("%B %d, %Y") if long else parsed.strftime("%b %d")

    def analyte_required_items(project: dict[str, object], analyte: dict[str, object]) -> list[dict[str, object]]:
        program_name = str(project["Program Name"])
        analyte_name = str(analyte["Analyte"])
        items: list[dict[str, object]] = []
        completed = set(analyte.get("Completed Studies") or [])
        in_progress = set(analyte.get("In Progress Studies") or [])
        review = set(analyte.get("Review Studies") or [])
        blocked = set(analyte.get("Blocked Studies") or [])
        for study_type in list(analyte.get("Required Studies") or program_required_studies(project)):
            if study_type in completed:
                status = "Approved"
                has_record = True
            elif study_type in review:
                status = "Submitted for Review"
                has_record = True
            elif study_type in blocked:
                status = "Rejected"
                has_record = True
            elif study_type in in_progress:
                status = "Draft"
                has_record = True
            else:
                status = "Draft"
                has_record = False
            items.append(
                {
                    "Program": program_name,
                    "Analyte": analyte_name,
                    "Study Type": study_type,
                    "Status": normalize_lifecycle_status(status),
                    "Has Record": has_record,
                    "Record": None,
                }
            )
        return items

    def program_required_items(project: dict[str, object]) -> list[dict[str, object]]:
        return [
            item
            for analyte in program_analytes(project)
            for item in analyte_required_items(project, analyte)
        ]

    def complete_status(status: object) -> bool:
        return normalize_lifecycle_status(status) in EXECUTION_PROGRESS_COMPLETE_STATES

    def timeline_status(item: dict[str, object]) -> str:
        status = normalize_lifecycle_status(item["Status"])
        if status in EXECUTION_PROGRESS_COMPLETE_STATES:
            return "Complete"
        if status in {"Ready For Review", "Pending Review", "Scientific Review", "Submitted for Review", "Under Review"}:
            return "In Review"
        if status == "Rejected":
            return "Blocked"
        if bool(item["Has Record"]):
            return "In Progress"
        return "Not Started"

    def sample_readiness(project: dict[str, object]) -> dict[str, int | str]:
        analytes = program_analytes(project)
        required = int(project.get("Required Samples") or sum(int(analyte.get("Required Samples", 0)) for analyte in analytes) or 80)
        if bool(project.get("Final Package Generated")):
            return {
                "Required": required,
                "Received": required,
                "Processed": required,
                "Pending": 0,
                "Status": "Ready",
            }
        received = int(project.get("Received Samples") or sum(int(analyte.get("Received Samples", 0)) for analyte in analytes))
        processed = int(project.get("Processed Samples") or sum(int(analyte.get("Processed Samples", 0)) for analyte in analytes))
        pending = max(0, required - received)
        if received >= required and processed >= required:
            status = "Ready"
        elif received > 0:
            status = "Partial"
        else:
            status = "Awaiting Samples"
        return {
            "Required": required,
            "Received": received,
            "Processed": processed,
            "Pending": pending,
            "Status": status,
        }

    def analyte_metrics(project: dict[str, object], analyte: dict[str, object]) -> dict[str, object]:
        items = analyte_required_items(project, analyte)
        required = len(items)
        completed = sum(1 for item in items if complete_status(item["Status"]))
        active = sum(1 for item in items if timeline_status(item) in {"In Progress", "In Review", "Blocked"})
        remaining = max(0, required - completed)
        next_item = next((item for item in items if normalize_lifecycle_status(item["Status"]) == "Rejected"), None)
        next_item = next_item or next((item for item in items if normalize_lifecycle_status(item["Status"]) in {"Ready For Review", "Pending Review", "Scientific Review", "Submitted for Review", "Under Review"}), None)
        next_item = next_item or next((item for item in items if normalize_lifecycle_status(item["Status"]) == "Draft" and bool(item["Has Record"])), None)
        next_item = next_item or next((item for item in items if not complete_status(item["Status"])), None)
        if completed == required and required:
            status = "Complete"
        elif active:
            status = "In Progress"
        else:
            status = "Not Started"
        blocker = "None"
        blocked_item = next((item for item in items if normalize_lifecycle_status(item["Status"]) == "Rejected"), None)
        if blocked_item:
            blocker = f"{blocked_item['Study Type']} blocked"
        elif int(analyte.get("Received Samples", 0)) < int(analyte.get("Required Samples", 0)):
            blocker = "Samples pending"
        return {
            "Analyte": str(analyte["Analyte"]),
            "Required": required,
            "Completed": completed,
            "In Progress": active,
            "Remaining": remaining,
            "Status": status,
            "Next Study": str(next_item["Study Type"]) if next_item else "Validation Report",
            "Blockers": blocker,
            "Last Updated": analyte.get("Last Updated", project.get("Last Updated", "")),
        }

    if not projects:
        st.info("Create a validation program to begin tracking study operations.")
        return

    program_names = [str(project["Program Name"]) for project in projects]
    selected_program_name = str(st.session_state.get("dashboard_current_program", program_names[0]))
    if selected_program_name not in program_names:
        selected_program_name = program_names[0]
    selected_program_name = st.selectbox(
        "Program",
        program_names,
        index=program_names.index(selected_program_name),
        key="dashboard_current_program",
    )
    current_program = next(project for project in projects if str(project["Program Name"]) == selected_program_name)
    current_items = program_required_items(current_program)
    current_samples = sample_readiness(current_program)
    pending_review = sum(
        1 for item in current_items
        if normalize_lifecycle_status(item["Status"]) in {"Ready For Review", "Pending Review", "Scientific Review", "Submitted for Review", "Under Review"}
    )
    rejected_count = sum(1 for item in current_items if normalize_lifecycle_status(item["Status"]) in {"Rejected", "Requires Revision"})
    remaining = sum(1 for item in current_items if not complete_status(item["Status"]))
    samples_remaining = int(current_samples.get("Pending", 0))
    if samples_remaining:
        health_status = "Awaiting Samples"
        health_class = "status-borderline"
        primary_blocker = f"{samples_remaining} Samples Remaining"
    elif rejected_count:
        health_status = "Review Blocked"
        health_class = "status-fail"
        primary_blocker = f"{rejected_count} Studies Require Revision"
    elif pending_review:
        health_status = "Awaiting Review"
        health_class = "status-borderline"
        primary_blocker = f"{pending_review} Studies Awaiting Review"
    elif remaining:
        health_status = "In Progress"
        health_class = "status-borderline"
        primary_blocker = f"{remaining} Studies Remaining"
    else:
        health_status = "On Track"
        health_class = "status-pass"
        primary_blocker = "No Active Blockers"
    review_status = "Not Ready" if pending_review or remaining else "Ready"
    report_status = "Generated" if bool(current_program.get("Final Package Generated")) else "Not Generated"

    st.subheader("Program Health")
    st.markdown(
        f"""
        <div class="svap-status-strip">
          <div>
            <div class="svap-strip-label">Status</div>
            <div class="svap-strip-value">{escape(health_status)}</div>
          </div>
          <div>
            <div class="svap-strip-label">Blocker</div>
            <div class="svap-strip-value">{escape(primary_blocker)}</div>
          </div>
          <div>
            <div class="svap-strip-label">Review</div>
            <div class="svap-strip-value">{escape(review_status)}</div>
          </div>
          <div>
            <div class="svap-strip-label">Reports</div>
            <div class="svap-strip-value">{escape(report_status)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Validation Context: {specimen_context_text(current_program)}")

    st.subheader("Program Portfolio")
    portfolio_header = st.columns([2.4, 1, 1.4, 0.6])
    for column, label in zip(portfolio_header, ["Program", "Progress", "Primary Blocker", "Action"]):
        column.markdown(f"**{label}**")
    for index, project in enumerate(projects):
        analyte_rows = [analyte_metrics(project, analyte) for analyte in program_analytes(project)]
        required = sum(int(metric["Required"]) for metric in analyte_rows)
        completed = sum(int(metric["Completed"]) for metric in analyte_rows)
        if required and completed == required and bool(project.get("Final Package Generated")):
            continue
        next_analyte = next((metric for metric in analyte_rows if metric["Blockers"] != "None"), None)
        next_analyte = next_analyte or next((metric for metric in analyte_rows if metric["Status"] == "In Progress"), None)
        next_analyte = next_analyte or next((metric for metric in analyte_rows if metric["Status"] == "Not Started"), None)
        program_samples = sample_readiness(project)
        blocker = str(next_analyte["Blockers"]) if next_analyte and str(next_analyte["Blockers"]) != "None" else (
            f"{program_samples['Pending']} Samples Remaining" if int(program_samples.get("Pending", 0)) else "None"
        )
        program_name = str(project["Program Name"])
        row_cols = st.columns([2.4, 1, 1.4, 0.6])
        row_cols[0].write(program_name)
        row_cols[1].write(f"{completed}/{required}")
        row_cols[2].write(blocker)
        with row_cols[3]:
            if st.button("Open", key=f"dashboard_select_program_{index}", use_container_width=True):
                st.session_state.dashboard_current_program = program_name
                st.session_state.execution_program = program_name
                st.session_state.execution_mode = ""
                st.session_state.pending_page = "Execution"
                st.rerun()

def render_validation_program_definition(embedded: bool = False) -> None:
    """Render the validation program definition workspace."""

    initialize_platform_state()
    if not embedded:
        render_page_header(
            "Program Definition",
            "Define validation programs.",
            kicker="",
        )
    projects = [normalize_program(project) for project in st.session_state.validation_projects]
    if not projects:
        st.info("Create a validation program before defining study requirements.")
        return

    program_names = [str(project["Program Name"]) for project in projects]
    selected_name = str(st.session_state.get("definition_program_selector") or program_names[0])
    if selected_name not in program_names:
        selected_name = program_names[0]
    program_index = program_names.index(selected_name)
    program = normalize_program(projects[program_index])
    definition = normalize_program_definition(program)

    default_analytes = ["Ferritin", "Vitamin D", "HbA1c", "TSH", "Cortisol", "Vitamin B12"]
    existing_analytes = list(definition.get("Analytes") or [])
    initial_template_name = str(definition.get("Validation Template"))
    if initial_template_name not in VALIDATION_TEMPLATES:
        initial_template_name = "Microtainer Equivalency Template"
    program_name = selected_name
    st.markdown('<div class="svap-definition-heading">Validation Scope</div>', unsafe_allow_html=True)
    context_cols = st.columns(2)
    with context_cols[0]:
        selected_name = st.selectbox(
            "Program",
            program_names,
            index=program_names.index(selected_name),
            key="definition_program_selector",
        )
        if selected_name != program_name:
            st.rerun()
        program_name = selected_name
    with context_cols[1]:
        template_name = st.selectbox(
            "Template",
            list(VALIDATION_TEMPLATES.keys()),
            index=list(VALIDATION_TEMPLATES.keys()).index(initial_template_name),
        )
    template = VALIDATION_TEMPLATES[template_name]
    template_context = str(template.get("Validation Context", program.get("Validation Context", "")))
    candidate_specimen = str(program.get("Candidate Specimen", ""))
    reference_specimen = str(program.get("Reference Specimen", ""))
    st.markdown('<div class="svap-definition-heading">Scientific Context</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="svap-comparison-hero">
          <div>
            <div class="svap-comparison-label">Candidate Specimen</div>
            <div class="svap-comparison-value">{escape(candidate_specimen)}</div>
          </div>
          <div class="svap-comparison-arrow">→</div>
          <div>
            <div class="svap-comparison-label">Reference Specimen</div>
            <div class="svap-comparison-value">{escape(reference_specimen)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="svap-validation-objective">
          <div class="svap-validation-objective-label">Validation Objective</div>
          <div>Validate analytical equivalency between the candidate specimen and the reference specimen according to the selected validation template.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="svap-definition-section"></div>', unsafe_allow_html=True)
    template_required_studies = list(template.get("Required Study Types", DEFAULT_PROGRAM_REQUIRED_STUDIES))
    study_state_key = f"definition_required_studies_{selected_name}"
    study_template_state_key = f"definition_required_studies_template_{selected_name}"
    study_editor_version_key = f"definition_required_studies_editor_version_{selected_name}"
    if (
        study_state_key not in st.session_state
        or st.session_state.get(study_template_state_key) != template_name
    ):
        st.session_state[study_state_key] = list(definition.get("Required Study Types") or template_required_studies)
        st.session_state[study_template_state_key] = template_name
        st.session_state[study_editor_version_key] = 0
    st.session_state.setdefault(study_editor_version_key, 0)
    required_studies = [
        str(study)
        for study in list(st.session_state.get(study_state_key, template_required_studies))
        if str(study) in TRUE_VALIDATION_STUDY_TYPES
    ]
    if not required_studies:
        required_studies = list(template_required_studies)
        st.session_state[study_state_key] = required_studies
    program_category = str(template.get("Program Category", definition.get("Program Category", "")))
    description = str(definition.get("Description", ""))
    existing_status = str(program.get("Status") or "Not Started")

    st.subheader("Analytes")
    analyte_state_key = f"definition_analyte_rows_{selected_name}"
    analyte_editor_version_key = f"definition_analyte_editor_version_{selected_name}"
    analyte_metadata_rows = []
    existing_analyte_records = {
        str(existing.get("Analyte")): normalize_analyte(dict(existing), fallback_name=str(existing.get("Analyte", "")))
        for existing in program_analytes(program)
    }
    if analyte_state_key not in st.session_state:
        st.session_state[analyte_state_key] = list(existing_analytes or default_analytes)
        st.session_state[analyte_editor_version_key] = 0
    st.session_state.setdefault(analyte_editor_version_key, 0)
    for analyte_name in list(st.session_state.get(analyte_state_key, existing_analytes or default_analytes)):
        existing = existing_analyte_records.get(str(analyte_name), {"Analyte": str(analyte_name)})
        analyte_metadata_rows.append(
            {
                "Analyte": str(analyte_name),
                "Assigned Analyzer": str(existing.get("Assigned Analyzer", "")),
                "Assigned Reagent": str(existing.get("Assigned Reagent", "")),
            }
        )
    if analyte_metadata_rows:
        analyzer_options = list(dict.fromkeys([*ANALYZER_OPTIONS, *[str(row.get("Assigned Analyzer", "")) for row in analyte_metadata_rows if str(row.get("Assigned Analyzer", "")).strip()]]))
        reagent_options = list(dict.fromkeys([*REAGENT_OPTIONS, *[str(row.get("Assigned Reagent", "")) for row in analyte_metadata_rows if str(row.get("Assigned Reagent", "")).strip()]]))
        analyte_metadata = st.data_editor(
            pd.DataFrame(analyte_metadata_rows),
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            column_config={
                "Analyte": st.column_config.TextColumn("Analyte", width="medium"),
                "Assigned Analyzer": st.column_config.SelectboxColumn(
                    "Analyzer",
                    options=analyzer_options,
                    required=True,
                    width="medium",
                ),
                "Assigned Reagent": st.column_config.SelectboxColumn(
                    "Reagent",
                    options=reagent_options,
                    required=True,
                    width="large",
                ),
            },
            key=f"analyte_metadata_editor_{selected_name}_{st.session_state[analyte_editor_version_key]}",
        )
        edited_analyte_metadata_rows = [
            {
                "Analyte": str(row.get("Analyte") or "").strip(),
                "Assigned Analyzer": str(row.get("Assigned Analyzer") or analyzer_options[0]),
                "Assigned Reagent": str(row.get("Assigned Reagent") or reagent_options[0]),
            }
            for row in analyte_metadata.to_dict("records")
            if str(row.get("Analyte") or "").strip()
        ]
        if edited_analyte_metadata_rows != analyte_metadata_rows:
            st.toast("✓ Saved")
            st.session_state[analyte_state_key] = [row["Analyte"] for row in edited_analyte_metadata_rows]
        analyte_metadata_by_name = {
            str(row.get("Analyte")): {
                "Assigned Analyzer": str(row.get("Assigned Analyzer") or analyzer_options[0]),
                "Assigned Reagent": str(row.get("Assigned Reagent") or reagent_options[0]),
            }
            for row in edited_analyte_metadata_rows
        }
        analytes = list(analyte_metadata_by_name.keys())
    else:
        analyte_metadata_by_name = {}
        analytes = []
    if not analytes:
        st.info("Add analytes included in this validation program.")
    if st.button("+ Add Analyte", key=f"add_analyte_{selected_name}", type="primary"):
        next_index = len(analytes) + 1
        new_analyte = f"New Analyte {next_index}"
        while new_analyte in set(analytes):
            next_index += 1
            new_analyte = f"New Analyte {next_index}"
        st.session_state[analyte_state_key] = [*analytes, new_analyte]
        st.session_state[analyte_editor_version_key] = int(st.session_state[analyte_editor_version_key]) + 1
        st.toast("✓ Saved")
        st.rerun()
    st.markdown('<div class="svap-definition-section"></div>', unsafe_allow_html=True)

    st.subheader("Template Studies")
    st.markdown(
        '<div class="svap-template-note">Required validation studies automatically generated for every analyte.</div>',
        unsafe_allow_html=True,
    )
    study_rows = [{"Study": study, "Required": "✓"} for study in required_studies]
    edited_study_rows = st.data_editor(
        pd.DataFrame(study_rows),
        width="stretch",
        hide_index=True,
        num_rows="fixed",
        disabled=["Required"],
        column_config={
            "Study": st.column_config.SelectboxColumn(
                "Study",
                options=list(TRUE_VALIDATION_STUDY_TYPES),
                required=True,
                width="large",
            ),
            "Required": st.column_config.TextColumn(
                "Required",
                width="small",
                help=None,
            ),
        },
        key=f"template_studies_editor_{selected_name}_{template_name}_{st.session_state[study_editor_version_key]}",
    )
    edited_required_studies = []
    for row in edited_study_rows.to_dict("records"):
        study = str(row.get("Study") or "").strip()
        if study in TRUE_VALIDATION_STUDY_TYPES and study not in edited_required_studies:
            edited_required_studies.append(study)
    if edited_required_studies and edited_required_studies != required_studies:
        required_studies = edited_required_studies
        st.session_state[study_state_key] = required_studies
        st.toast("✓ Saved")
    add_study_options = [study for study in TRUE_VALIDATION_STUDY_TYPES if study not in required_studies]
    if st.button("Manage Template Studies", key=f"add_template_study_{selected_name}", type="primary"):
        if add_study_options:
            st.session_state[study_state_key] = [*required_studies, add_study_options[0]]
            st.session_state[study_editor_version_key] = int(st.session_state[study_editor_version_key]) + 1
            st.toast("✓ Saved")
            st.rerun()
    st.markdown('<div class="svap-definition-section"></div>', unsafe_allow_html=True)

    existing_overrides = {
        str(analyte): [study for study in list(studies) if study in required_studies]
        for analyte, studies in dict(definition.get("Analyte Study Requirements") or {}).items()
        if str(analyte) in set(str(item) for item in analytes)
    }
    effective_overrides = dict(existing_overrides)

    def persist_program_definition() -> None:
        """Persist the current program definition and generated work plan."""

        existing_by_analyte = {
            str(existing.get("Analyte")): normalize_analyte(dict(existing), fallback_name=str(existing.get("Analyte", "")))
            for existing in program_analytes(program)
        }
        program["Program Name"] = program_name
        program["Project Name"] = program_name
        program["Validation Template"] = template_name
        program["Program Category"] = program_category
        program["Validation Context"] = template_context
        program["Candidate Specimen"] = candidate_specimen
        program["Reference Specimen"] = reference_specimen
        program["Assay / Biomarker"] = template_context
        program["Program Owner"] = str(program.get("Program Owner", platform_settings().get("Analyst Name", "")))
        program["Status"] = existing_status
        program["Notes"] = description
        program["Last Updated"] = date.today().isoformat()
        program["Program Definition"] = {
            "Program Name": program_name,
            "Validation Template": template_name,
            "Program Category": program_category,
            "Validation Context": template_context,
            "Candidate Specimen": candidate_specimen,
            "Reference Specimen": reference_specimen,
            "Description": description,
            "Analytes": analytes,
            "Required Study Types": required_studies,
            "Analyte Study Requirements": effective_overrides,
        }
        program["Analytes"] = [
            {
                **existing_by_analyte.get(analyte, {"Analyte": analyte}),
                "Analyte": analyte,
                "Assigned Analyzer": analyte_metadata_by_name.get(str(analyte), {}).get("Assigned Analyzer", "Analyzer not assigned"),
                "Assigned Reagent": analyte_metadata_by_name.get(str(analyte), {}).get("Assigned Reagent", "Reagent not assigned"),
                "Required Studies": effective_overrides.get(analyte, required_studies),
            }
            for analyte in analytes
        ]
        st.session_state.validation_projects[program_index] = normalize_program(program)

    persist_program_definition()

    total_required_studies = sum(
        len(effective_overrides.get(str(analyte), required_studies))
        for analyte in analytes
    )
    st.subheader("Generated Study Plan")
    st.markdown(
        f"""
        <div class="svap-generated-work">
          <div class="svap-generated-work-line"><strong>{len(analytes)}</strong> analytes</div>
          <div class="svap-generated-work-line"><strong>{len(required_studies)}</strong> required validation studies per analyte</div>
          <div class="svap-generated-work-line"><strong>{total_required_studies}</strong> execution workspaces</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="svap-next-step"><div class="svap-next-step-label">Next Step</div></div>', unsafe_allow_html=True)
    action_cols = st.columns([0.85, 3.15])
    with action_cols[0]:
        if st.button("Open Execution", type="primary", use_container_width=True):
            persist_program_definition()
            st.session_state.execution_program = program_name
            if analytes:
                st.session_state.execution_analyte = str(analytes[0])
            st.session_state.pending_page = "Execution"
            st.rerun()


def render_projects_workspace(embedded: bool = False) -> None:
    """Render the validation program directory."""

    initialize_platform_state()
    if not embedded:
        render_page_header(
            "Programs",
            "Define validation programs.",
            kicker="",
        )
    settings = platform_settings()

    create_expanded = bool(st.session_state.get("show_create_program", False))
    if st.button("+ New Program", type="primary", key="show_create_program_button"):
        st.session_state.show_create_program = not create_expanded
        st.rerun()

    if create_expanded:
        with st.expander("Program Details", expanded=True):
            default_program_lead = str(settings.get("Analyst Name") or "").strip()
            if not default_program_lead:
                home_user = Path.home().name.replace(".", " ").replace("_", " ").replace("-", " ")
                default_program_lead = " ".join(part.capitalize() for part in home_user.split()) or "Program Lead"
                if default_program_lead.lower().replace(" ", "") == "ankitapuri":
                    default_program_lead = "Ankita Puri"
            row = st.columns([1.2, 1.25, 0.95])
            with row[0]:
                program_name = st.text_input("Program Name", value="New Validation Program")
            with row[1]:
                validation_scope = st.selectbox(
                    "Validation Scope",
                    list(VALIDATION_SCOPE_DEFINITIONS.keys()),
                    key="new_program_validation_scope",
                )
            with row[2]:
                owner = st.text_input("Program Lead", value=default_program_lead)
            scope_definition = VALIDATION_SCOPE_DEFINITIONS[validation_scope]
            validation_context = str(scope_definition["Validation Context"])
            template_name = str(scope_definition["Validation Template"])
            candidate_specimen = str(scope_definition["Candidate Specimen"])
            reference_specimen = str(scope_definition["Reference Specimen"])
            st.markdown(
                f"""
                <div class="svap-definition-heading">Validation Comparison</div>
                <div class="svap-comparison-hero">
                  <div>
                    <div class="svap-comparison-label">Candidate Specimen</div>
                    <div class="svap-comparison-value">{escape(scientific_title_case(candidate_specimen))}</div>
                  </div>
                  <div class="svap-comparison-arrow">→</div>
                  <div>
                    <div class="svap-comparison-label">Reference Specimen</div>
                    <div class="svap-comparison-value">{escape(scientific_title_case(reference_specimen))}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            status = "Not Started"
            if st.button("Save Program", type="primary"):
                template = VALIDATION_TEMPLATES[template_name]
                st.session_state.validation_projects.append(
                    {
                        "Project Name": program_name,
                        "Program Name": program_name,
                        "Assay": validation_context,
                        "Assay / Biomarker": validation_context,
                        "Program Owner": owner,
                        "Validation Template": template_name,
                        "Program Category": template["Program Category"],
                        "Validation Context": validation_context,
                        "Candidate Specimen": candidate_specimen,
                        "Reference Specimen": reference_specimen,
                        "Status": status,
                        "Study Status": status,
                        "Start Date": date.today().isoformat(),
                        "Target Completion Date": date.today().isoformat(),
                        "Reviewer": settings.get("Reviewer Name", ""),
                        "Notes": "",
                        "Required Studies": list(template.get("Required Study Types", DEFAULT_PROGRAM_REQUIRED_STUDIES)),
                        "Program Definition": {
                            "Program Name": program_name,
                            "Validation Template": template_name,
                            "Program Category": template["Program Category"],
                            "Validation Context": validation_context,
                            "Candidate Specimen": candidate_specimen,
                            "Reference Specimen": reference_specimen,
                            "Description": "",
                            "Status": status,
                            "Analytes": [],
                            "Required Study Types": list(template.get("Required Study Types", DEFAULT_PROGRAM_REQUIRED_STUDIES)),
                            "Optional Study Types": list(template.get("Optional Study Types", [])),
                            "Sample Requirements": template.get("Sample Requirements", ""),
                            "Review Workflow": template.get("Review Workflow", ""),
                            "Report Structure": template.get("Report Structure", ""),
                            "Analyte Study Requirements": {},
                        },
                        "Analytes": [],
                        "Completed Studies": [],
                        "Last Updated": date.today().isoformat(),
                        "Overall Status": status,
                        "Final Package Generated": False,
                    }
                )
                record_program_activity(program_name, "Program", "Program saved", "Validation program created.")
                st.success(f"Saved program: {program_name}")
                st.session_state.show_create_program = False
                st.rerun()

    st.subheader("Program Directory")
    projects = [normalize_program(project) for project in st.session_state.validation_projects]
    search = st.text_input("Search", placeholder="Search Programs")

    program_rows: list[dict[str, object]] = []
    for project in projects:
        project = normalize_program(project)
        metrics = program_metrics(project)
        completed = int(metrics.get("Completed", 0))
        required = int(metrics.get("Required", 0))
        percent = min(100, max(0, round((completed / required) * 100))) if required else 0
        if required and completed >= required:
            status_label = "Completed"
            status_class = "completed"
            fill_class = "completed"
        elif completed == 0:
            status_label = "Not Started"
            status_class = "start"
            fill_class = "start"
        else:
            status_label = "In Progress"
            status_class = "progress"
            fill_class = "progress"
        progress_text = f"{completed}/{required} studies" if required else "No studies configured"
        row = {
            "Program": project["Program Name"],
            "Validation Scope": specimen_context_text(project),
            "Progress": progress_text,
            "Progress Percent": percent,
            "Progress Status": status_label,
            "Progress Status Class": status_class,
            "Progress Fill Class": fill_class,
            "Last Activity": relative_activity_text(project.get("Last Updated", "")),
        }
        row_text = " ".join(str(value) for value in row.values()).lower()
        if search and search.lower() not in row_text:
            continue
        program_rows.append(row)

    if not program_rows:
        st.info("No programs match the current filters.")
        return

    rows_html = [
        """
        <div class="svap-program-directory">
          <div class="svap-program-directory-header">
            <div>Program</div>
            <div>Validation Scope</div>
            <div>Progress</div>
            <div>Last Activity</div>
          </div>
        """
    ]
    for row in program_rows:
        program_name = str(row["Program"])
        href = escape(f"?page=Programs&program_open={quote(program_name)}", quote=True)
        rows_html.append(
            f'<a class="svap-program-directory-row" href="{href}" target="_self" aria-label="Open {escape(program_name)}">'
            f'<span class="svap-program-name">{escape(program_name)}</span>'
            f'<span class="svap-program-scope">{escape(str(row["Validation Scope"]))}</span>'
            '<span class="svap-program-progress-cell">'
            '<span class="svap-program-progress-summary">'
            f'<span class="svap-execution-progress-count">{escape(str(row["Progress"]))}</span>'
            f'<span class="svap-execution-status-text svap-execution-status-text-{escape(str(row["Progress Status Class"]))}">{escape(str(row["Progress Status"]))}</span>'
            "</span>"
            '<span class="svap-progress-bar-row">'
            '<span class="svap-execution-progress-bar">'
            f'<span class="svap-execution-progress-fill svap-execution-progress-fill-{escape(str(row["Progress Fill Class"]))}" style="width: {int(row["Progress Percent"])}%;"></span>'
            "</span>"
            f'<span class="svap-execution-program-progress-percent">{int(row["Progress Percent"])}%</span>'
            "</span>"
            "</span>"
            f'<span class="svap-program-muted svap-program-last-activity">{escape(str(row["Last Activity"]))}</span>'
            "</a>"
        )
    rows_html.append("</div>")
    st.markdown("\n".join(rows_html), unsafe_allow_html=True)


def render_programs_page() -> None:
    """Render program directory and definition under one Programs page."""

    initialize_platform_state()
    query_program = st.query_params.get("program_open", "")
    if isinstance(query_program, list):
        query_program = query_program[0] if query_program else ""
    if query_program:
        st.session_state.definition_program_selector = str(query_program)
        st.session_state.programs_tab = "Definition"
        try:
            del st.query_params["program_open"]
        except KeyError:
            pass
    render_page_header(
        "Programs",
        "Define validation programs.",
        kicker="",
    )
    if "programs_tab" not in st.session_state or st.session_state.programs_tab not in {"Directory", "Definition"}:
        st.session_state.programs_tab = "Directory"
    if st.session_state.programs_tab == "Directory":
        render_projects_workspace(embedded=True)
    else:
        if st.button("Back to Program Directory"):
            st.session_state.programs_tab = "Directory"
            st.rerun()
        render_validation_program_definition(embedded=True)


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


def review_status_label(status: object) -> str:
    """Return review-facing lifecycle status labels."""

    normalized = normalize_lifecycle_status(status)
    return "In Review" if normalized == "Under Review" else normalized


def format_review_date(value: object) -> str:
    """Return compact review date text."""

    parsed = pd.to_datetime(str(value or ""), errors="coerce")
    if pd.isna(parsed):
        return str(value or "")
    return parsed.strftime("%b %d, %Y")


def _format_review_cell(value: object) -> object:
    """Format review-table values without long floating-point noise."""

    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        numeric = float(value)
        if abs(numeric) <= 1 and numeric != 0:
            return f"{numeric:.4f}"
        return f"{numeric:.2f}"
    text = str(value)
    numeric_text = text.strip().replace("%", "")
    try:
        numeric = float(numeric_text)
    except ValueError:
        return text
    if "%" in text:
        return f"{numeric:.2f}%"
    if any(token in text.lower() for token in ("r²", "correlation")):
        return f"{numeric:.4f}"
    if "." in text and len(text.split(".")[-1]) > 4:
        return f"{numeric:.4f}" if abs(numeric) <= 1 else f"{numeric:.2f}"
    return text


def review_criteria_display_table(criteria_table: pd.DataFrame) -> pd.DataFrame:
    """Return the read-only review criteria table."""

    display = _compact_criteria_table(criteria_table).drop(columns=["Acceptance Source"], errors="ignore")
    preferred = [column for column in ["Criterion", "Observed Value", "Observed", "Acceptance Limit", "Status", "Pass/Fail Status"] if column in display.columns]
    display = display[preferred].copy() if preferred else display.copy()
    display = display.rename(columns={"Observed Value": "Observed", "Acceptance Limit": "Required", "Pass/Fail Status": "Status"})
    for column in display.columns:
        if column != "Status":
            display[column] = display[column].map(_format_review_cell)
    return display


def review_criteria_table_html(criteria_table: pd.DataFrame) -> str:
    """Render review criteria with subtle emphasis only for failed rows."""

    if criteria_table.empty:
        return "<table></table>"
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in criteria_table.columns)
    rows: list[str] = []
    for _, row in criteria_table.iterrows():
        status = str(row.get("Status", "") or row.get("Pass/Fail Status", "")).strip().upper()
        row_class = ' class="svap-criteria-fail"' if status in {"FAIL", "FAILED", "REJECTED"} else ""
        cells = []
        for column in criteria_table.columns:
            value = row[column]
            if column in {"Pass/Fail Status", "Status"}:
                status_value = str(value)
                status_class = "status-fail" if status_value.strip().upper() in {"FAIL", "FAILED", "REJECTED"} else "status-neutral"
                cells.append(f'<td><span class="status-badge {status_class}">{escape(status_value)}</span></td>')
            else:
                cells.append(f"<td>{escape(str(value))}</td>")
        rows.append(f"<tr{row_class}>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def render_review_summary_card(study_name: str, execution_result: str, scientific_question: str, key_finding: str) -> None:
    """Render a compact reviewer-facing executive summary without card chrome."""

    result_class = "status-pass" if str(execution_result).upper() == "PASS" else "status-fail"
    st.markdown(
        f"""
        <div class="svap-review-summary">
          <div class="svap-review-summary-top">
            <div class="svap-review-summary-title">{escape(study_name)}</div>
            <span class="status-badge {result_class}">{escape(str(execution_result))}</span>
          </div>
          <div class="svap-review-summary-question">
            <div class="svap-review-summary-label">Scientific Question</div>
            <div class="svap-review-summary-value">{escape(scientific_question)}</div>
          </div>
          <div class="svap-review-summary-question">
            <div class="svap-review-summary-label">Key Finding</div>
            <div class="svap-review-summary-value">{escape(key_finding)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_review_interpretation(text: str) -> None:
    """Render a frozen scientific interpretation as plain review text."""

    lines = [part.strip() for part in text.split("\n\n") if part.strip()]
    paragraphs = "".join(f"<p>{escape(line)}</p>" for line in lines)
    st.markdown(
        f"""
        <div class="svap-review-interpretation">
          {paragraphs}
        </div>
        """,
        unsafe_allow_html=True,
    )


def review_artifact_value(package: dict[str, object], artifact_name: str) -> str:
    """Read one frozen review artifact value from the submitted package."""

    supporting = package.get("Supporting Evidence", {}) if isinstance(package, dict) else {}
    artifacts = supporting.get("Review Artifacts") if isinstance(supporting, dict) else None
    if not isinstance(artifacts, pd.DataFrame) or artifacts.empty:
        return ""
    artifact_rows = artifacts[artifacts["Artifact"].astype(str).str.lower() == artifact_name.lower()]
    if artifact_rows.empty:
        return ""
    return str(artifact_rows.iloc[0].get("Value", "") or "")


def _artifact_matches(title: str, study_type: str, critical: bool = True) -> bool:
    """Return whether an artifact belongs in critical evidence for a study type."""

    text = title.lower()
    study = study_type.lower()
    if "method comparison" in study:
        tokens = ("scatter", "regression", "bland", "difference", "disagreement", "outlier")
    elif "precision" in study:
        tokens = ("cv", "precision distribution", "qc level")
    elif "accuracy" in study:
        tokens = ("recovery", "bias", "accuracy summary")
    elif "linearity" in study:
        tokens = ("regression", "residual", "deviation", "linearity summary")
    elif "stability" in study:
        tokens = ("recovery", "drift", "condition", "timepoint")
    elif "detection" in study:
        tokens = ("lob", "lod", "loq", "cv vs concentration", "distribution")
    else:
        tokens = ("summary",)
    return any(token in text for token in tokens)


def _precision_critical_figure(title: str) -> bool:
    """Return whether a precision plot belongs on the primary review page."""

    text = title.lower()
    return "precision distribution" in text


def _precision_critical_table(title: str) -> bool:
    """Return whether a precision table belongs on the primary review page."""

    text = title.lower()
    return (
        ("qc" in text or "level" in text)
        and "summary" in text
        and not any(token in text for token in ("raw", "analyzed", "artifact", "audit"))
    )


def _precision_summary_table_columns(table: object) -> bool:
    """Return whether a table looks like a precision QC summary."""

    if not isinstance(table, pd.DataFrame):
        return False
    columns = {str(column).lower() for column in table.columns}
    has_level = any("level" in column or "qc" in column for column in columns)
    has_cv = any("cv" in column for column in columns)
    has_summary_stat = any("mean" in column for column in columns) and any("sd" in column or "standard deviation" in column for column in columns)
    return has_level and has_cv and has_summary_stat


def review_key_finding(study_type: str, execution_result: str) -> str:
    """Return one reviewer-facing scientific finding for the summary."""

    passed = str(execution_result).upper() == "PASS"
    study = study_type.lower()
    outcome = "met" if passed else "did not meet"
    if "method comparison" in study:
        return f"Method comparison {outcome} predefined agreement criteria."
    if "precision" in study:
        return f"Precision {outcome} predefined acceptance limits."
    if "accuracy" in study:
        return f"Accuracy {outcome} predefined recovery and bias limits."
    if "linearity" in study:
        return f"Linearity {outcome} predefined regression and deviation limits."
    if "stability" in study:
        return f"Stability {outcome} predefined recovery and drift limits."
    if "detection" in study:
        return f"Detection capability {outcome} predefined LoB, LoD, and LoQ criteria."
    return f"Study {outcome} predefined acceptance criteria."


def review_scientific_question(study_type: str, analyte: str) -> str:
    """Return the scientific question being reviewed for a study type."""

    study = study_type.lower()
    analyte_name = analyte or "the analyte"
    if "method comparison" in study:
        return f"Do candidate and reference results agree for {analyte_name}?"
    if "precision" in study:
        return f"Is {analyte_name} measurement precision acceptable across runs, days, and replicates?"
    if "accuracy" in study:
        return f"Do observed {analyte_name} results recover expected values within predefined limits?"
    if "linearity" in study:
        return f"Is {analyte_name} response linear across the evaluated measurement range?"
    if "stability" in study:
        return f"Does {analyte_name} remain stable across evaluated timepoints and storage conditions?"
    if "detection" in study:
        return f"Do LoB, LoD, and LoQ support the required detection capability for {analyte_name}?"
    return f"Does the completed {study_type} evidence support the scientific conclusion?"


def review_critical_artifact_titles(study_type: str) -> tuple[set[str], set[str]]:
    """Return critical figure and table names for the primary review surface."""

    study = study_type.lower()
    if "method comparison" in study:
        return (
            {"regression plot", "bland-altman plot", "residual plot"},
            {"flagged samples"},
        )
    if "precision" in study:
        return (
            {"precision distribution plot", "precision component summary", "cv% summary plot"},
            {"variance components", "qc level summary"},
        )
    if "accuracy" in study:
        return (
            {"expected vs observed plot", "recovery plot", "bias plot"},
            {"accuracy summary", "worst case summary"},
        )
    if "linearity" in study:
        return (
            {"regression plot", "recovery / deviation plot", "residual plot"},
            {"linearity summary", "residual / deviation summary"},
        )
    if "stability" in study:
        return (
            {"recovery vs timepoint", "drift vs timepoint", "storage condition comparison"},
            {"stability summary", "timepoint summary", "potential instability flags"},
        )
    if "detection" in study:
        return (
            {"loq decision plot", "blank replicate distribution", "low-level replicate distribution"},
            {"lob / lod calculation", "loq determination", "decision matrix"},
        )
    return set(), set()


def render_review_critical_evidence(study: dict[str, object], study_name: str, package: dict[str, object], study_type: str) -> None:
    """Render the smallest evidence set needed to support the review decision."""

    figures = package.get("Visualizations", {}) if isinstance(package, dict) else {}
    supporting = package.get("Supporting Evidence", {}) if isinstance(package, dict) else {}
    figure_titles, table_titles = review_critical_artifact_titles(study_type)
    critical_figures = {
        title: figure
        for title, figure in figures.items()
        if str(title).lower() in figure_titles
    }
    critical_tables = {
        title: table
        for title, table in supporting.items()
        if str(title).lower() in table_titles and isinstance(table, pd.DataFrame)
    }

    st.markdown("### Critical Scientific Evidence")
    if not critical_figures and not critical_tables:
        st.caption("No critical evidence artifacts were submitted for this study.")
        return
    chart_base = f"review_critical_{study.get('Study ID', study_name)}".replace(" ", "_").replace("/", "_").lower()
    for index, (title, figure) in enumerate(critical_figures.items()):
        st.markdown(f"**{title}**")
        st.plotly_chart(
            figure,
            width="stretch",
            key=f"{chart_base}_{index}",
            config={"displayModeBar": False, "responsive": True},
        )
    for title, table in critical_tables.items():
        st.markdown(f"**{title}**")
        if table.empty:
            if "flagged" in str(title).lower():
                st.caption("No flagged samples.")
        else:
            st.dataframe(_sanitize_table(table), width="stretch", hide_index=True)


def render_review_supporting_evidence(study: dict[str, object], study_name: str, package: dict[str, object], study_type: str) -> None:
    """Render all secondary artifacts inside one collapsed evidence package."""

    figures = package.get("Visualizations", {}) if isinstance(package, dict) else {}
    supporting = package.get("Supporting Evidence", {}) if isinstance(package, dict) else {}
    critical_figure_titles, critical_table_titles = review_critical_artifact_titles(study_type)
    data_tables: dict[str, object] = {}
    secondary_figures: dict[str, object] = {}
    for title, figure in figures.items():
        if str(title).lower() not in critical_figure_titles:
            secondary_figures[str(title)] = figure
    for title, artifact in supporting.items():
        title_text = str(title)
        if any(token in title_text.lower() for token in ("artifact", "audit", "version", "hash", "run id")):
            continue
        if title_text.lower() in critical_table_titles:
            continue
        if isinstance(artifact, pd.DataFrame) and artifact.empty:
            continue
        data_tables[title_text] = artifact

    if secondary_figures or data_tables:
        with st.expander("Supporting Evidence", expanded=False):
            if secondary_figures:
                st.markdown("**Additional Visualizations**")
                chart_base = f"review_{study.get('Study ID', study_name)}".replace(" ", "_").replace("/", "_").lower()
                for index, (title, figure) in enumerate(secondary_figures.items()):
                    st.markdown(f"**{title}**")
                    st.plotly_chart(figure, width="stretch", key=f"{chart_base}_{index}")
            if data_tables:
                st.markdown("**Data Tables**")
                for title, table in data_tables.items():
                    if isinstance(table, pd.DataFrame) and table.empty:
                        continue
                    st.markdown(f"**{title}**")
                    if isinstance(table, pd.DataFrame):
                        st.dataframe(_sanitize_table(table), width="stretch", hide_index=True)
                    else:
                        st.write(table)


def render_review_workspace(study: dict[str, object]) -> None:
    """Render a compact scientific approval workspace for one submitted study."""

    settings = platform_settings()
    study_name = str(study.get("Study Name", "Study"))
    analyte = str(study.get("Analyte") or study.get("Assay") or "")
    study_type = str(study.get("Study Type", ""))
    program = str(study.get("Project", ""))
    current_status = normalize_lifecycle_status(study.get("Status"))
    persisted_payload = load_review_package_record(str(study.get("Study ID", "")))
    package = (
        study.get("Review Package")
        or st.session_state.get("review_packages", {}).get(str(study.get("Study ID", "")))
        or (persisted_payload or {}).get("Review Package")
        or {}
    )
    metadata = package.get("Study Metadata", {}) if isinstance(package, dict) else {}
    execution_result = str(package.get("Decision", "Pending")) if isinstance(package, dict) else "Pending"
    submitted_date = study.get("Submitted Date") or metadata.get("Submitted Date") or study.get("Completion Date")
    program_record, analyte_record = find_program_and_analyte(program, analyte)

    st.subheader(f"{analyte or study_name} · {study_type or 'Study Review'}")
    submitted_by = str(metadata.get("Analyst") or study.get("Analyst") or "").strip()
    study_version = review_artifact_value(package if isinstance(package, dict) else {}, "Study Version") or "1"
    dataset_hash = str(metadata.get("Input Dataset Hash") or review_artifact_value(package if isinstance(package, dict) else {}, "Input Dataset Hash") or "")
    meta_parts = [
        f"Program: {program}" if program else "",
        analyte_scientific_context(analyte_record) if analyte_record else "",
        f"Submitted: {format_review_date(submitted_date)}" if submitted_date else "",
        f"Study v{study_version}",
    ]
    st.markdown(
        f'<div class="svap-review-meta-line">{" · ".join(escape(part) for part in meta_parts if part)}</div>',
        unsafe_allow_html=True,
    )
    with st.expander("View Metadata", expanded=False):
        metadata_rows = []
        if program:
            metadata_rows.append({"Field": "Program", "Value": program})
        if program_record:
            metadata_rows.append({"Field": "Specimens", "Value": specimen_context_text(program_record)})
        if submitted_by:
            metadata_rows.append({"Field": "Submitted By", "Value": submitted_by})
            metadata_rows.append({"Field": "Execution Completed By", "Value": submitted_by})
        if submitted_date:
            metadata_rows.append({"Field": "Submitted", "Value": format_review_date(submitted_date)})
        metadata_rows.append({"Field": "Study Version", "Value": study_version})
        if dataset_hash:
            metadata_rows.append({"Field": "Dataset Hash", "Value": dataset_hash})
        if metadata_rows:
            st.dataframe(pd.DataFrame(metadata_rows), width="stretch", hide_index=True)

    st.markdown("### Executive Summary")
    criteria_table = package.get("Acceptance Criteria", pd.DataFrame()) if isinstance(package, dict) else pd.DataFrame()
    criteria_missing = not isinstance(criteria_table, pd.DataFrame) or criteria_table.empty
    render_review_summary_card(
        study_name,
        execution_result,
        review_scientific_question(study_type, analyte),
        review_key_finding(study_type, execution_result),
    )

    st.divider()
    st.markdown("### Acceptance Criteria")
    if isinstance(criteria_table, pd.DataFrame) and not criteria_table.empty:
        review_criteria = review_criteria_display_table(criteria_table)
        st.markdown(
            f'<div class="svap-status-table">{review_criteria_table_html(_sanitize_table(review_criteria))}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Acceptance criteria package missing. Review cannot be submitted.")

    st.divider()
    st.markdown("### Scientific Interpretation")
    interpretation = str(package.get("Interpretation", "") if isinstance(package, dict) else "").strip()
    if interpretation:
        render_review_interpretation(interpretation)
    else:
        st.warning("Scientific interpretation package missing. Review cannot be submitted.")
        criteria_missing = True

    st.divider()
    render_review_critical_evidence(study, study_name, package if isinstance(package, dict) else {}, study_type)

    render_review_supporting_evidence(study, study_name, package if isinstance(package, dict) else {}, study_type)

    if current_status in {"Approved", "Rejected", "Follow-Up Required"}:
        st.divider()
        st.markdown("### Reviewer Assessment")
        reviewer_name = str(study.get("Reviewer") or settings.get("Reviewer Name", "") or "Reviewer")
        review_date = format_review_date(study.get("Review Date"))
        st.write(f"Decision: {review_status_label(current_status)}")
        st.caption(f"Recorded by {reviewer_name}{f' on {review_date}' if review_date else ''}.")
        comments = str(study.get("Reviewer Comments", "")).strip()
        follow_up = str(study.get("Required Follow-Up Actions", "")).strip()
        if comments:
            st.markdown("**Reviewer Comments**")
            st.write(comments)
        if follow_up:
            st.markdown("**Required Follow-Up Actions**")
            st.write(follow_up)
        return

    st.divider()
    st.markdown("### Reviewer Assessment")
    reviewer = str(study.get("Reviewer") or settings.get("Reviewer Name", "") or "Current User")
    with st.container(border=True):
        decision = st.radio(
            "Reviewer Decision",
            ["Approve Study", "Request Follow-up"],
            key=f"review_decision_{study.get('Study ID')}",
            horizontal=True,
        )
        st.markdown('<div class="svap-review-decision-divider"></div>', unsafe_allow_html=True)
        reviewer_comments = st.text_area(
            "Reviewer Comments",
            value=str(study.get("Reviewer Comments", "")),
            key=f"review_comments_{study.get('Study ID')}",
            height=100,
            placeholder="Optional comments for the study record.",
        )
        follow_up_actions = ""
        if decision == "Request Follow-up":
            follow_up_actions = st.text_area(
                "Required Follow-up",
                value=str(study.get("Required Follow-Up Actions", "")),
                key=f"review_follow_up_{study.get('Study ID')}",
                height=100,
                placeholder="Describe what must be resolved before approval.",
            )

        if criteria_missing:
            st.caption("Review cannot be submitted until the frozen scientific review package is complete.")

        submit_decision = st.button(
            "Submit Review Decision",
            type="primary",
            use_container_width=True,
            key=f"submit_review_{study.get('Study ID')}",
            disabled=criteria_missing,
        )

    def record_review_decision(new_status: str, history_decision: str, comments: str, required_follow_up: str = "") -> None:
        previous_status = normalize_lifecycle_status(study.get("Status"))
        study["Status"] = new_status
        study["Reviewer"] = reviewer
        study["Reviewer Comments"] = comments
        study["Required Follow-Up Actions"] = required_follow_up
        study["Review Date"] = date.today().isoformat()
        study["Last Updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        study["Locked"] = new_status == "Approved"
        update_analyte_execution_state(program, analyte, study_type, new_status)
        if isinstance(package, dict):
            persist_review_package(study, package)
        record_approval_history(study, previous_status, new_status, history_decision, reviewer, comments)
        record_program_activity(program, study_name, history_decision, comments)
        st.session_state.review_selected_study_id = ""
        st.success(f"Review decision recorded: {history_decision}.")
        st.rerun()

    if submit_decision:
        if decision == "Request Follow-up" and not follow_up_actions.strip():
            st.error("Enter the required follow-up before submitting the review decision.")
            st.stop()
        if decision == "Approve Study":
            comments = reviewer_comments.strip()
            record_review_decision("Approved", "Approved", comments)
        else:
            comments = reviewer_comments.strip()
            record_review_decision("Follow-Up Required", "Follow-Up Required", comments, follow_up_actions.strip())


def review_queue_status(record: dict[str, object]) -> str:
    """Return the compact queue status label."""

    normalized = normalize_lifecycle_status(record.get("Status"))
    if normalized in {"Follow-Up Required", "Requires Revision"}:
        return "Follow-up Requested"
    if normalized == "Approved":
        return "Approved"
    if normalized in {"Under Review", "Scientific Review"}:
        return "In Review"
    return "Pending Review"


def review_queue_status_badge(status: str) -> str:
    """Render a simple review queue status badge."""

    css_class = {
        "Pending Review": "svap-review-status-pending",
        "In Review": "svap-review-status-pending",
        "Follow-up Requested": "svap-review-status-followup",
        "Approved": "svap-review-status-approved",
    }.get(status, "status-neutral")
    return f'<span class="status-badge {css_class}">{escape(status)}</span>'


def review_submitted_text(submitted_value: object, submitted_by: object) -> str:
    """Return compact submitted date and submitter text for the queue."""

    parsed = pd.to_datetime(str(submitted_value or ""), errors="coerce")
    if pd.isna(parsed):
        date_text = str(submitted_value or "")
    else:
        submitted_date = parsed.date()
        today = date.today()
        delta_days = (today - submitted_date).days
        if delta_days == 0:
            date_text = "Today"
        elif delta_days == 1:
            date_text = "Yesterday"
        elif delta_days > 1:
            date_text = f"{delta_days} days ago"
        else:
            date_text = parsed.strftime("%b %d")
    submitter = str(submitted_by or "").strip()
    if date_text and submitter:
        return f"{escape(date_text)}<br><span class=\"svap-review-row-secondary\">{escape(submitter)}</span>"
    return escape(date_text or submitter)


def review_queue_sort_key(record: dict[str, object]) -> tuple[int, int]:
    """Sort review tasks by priority, then most recently submitted."""

    priority = {
        "Pending Review": 0,
        "In Review": 0,
        "Follow-up Requested": 1,
        "Approved": 2,
    }.get(review_queue_status(record), 3)
    submitted = pd.to_datetime(str(record.get("Submitted Date") or record.get("Completion Date") or ""), errors="coerce")
    submitted_sort = 0 if pd.isna(submitted) else -int(submitted.value)
    return priority, submitted_sort


def render_validation_review_center() -> None:
    """Render the scientific review queue."""

    initialize_platform_state()
    load_persisted_review_packages()
    query_review_id = st.query_params.get("review_open", "")
    if isinstance(query_review_id, list):
        query_review_id = query_review_id[0] if query_review_id else ""
    if query_review_id:
        st.session_state.review_selected_study_id = str(query_review_id)
    selected_id = str(st.session_state.get("review_selected_study_id", ""))
    selected_study = next((record for record in get_lifecycle_records() if str(record.get("Study ID")) == selected_id), None)
    if selected_id and selected_study:
        previous_status = normalize_lifecycle_status(selected_study.get("Status"))
        if previous_status in {"Ready For Review", "Pending Review", "Submitted for Review", "Scientific Review"}:
            selected_study["Status"] = "Under Review"
            selected_study["Last Updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        render_page_header(
            "Review",
            "Review validation studies.",
            kicker="",
        )
        if st.button("Back to Review Queue", use_container_width=False):
            st.session_state.review_selected_study_id = ""
            st.query_params.clear()
            st.rerun()
        render_review_workspace(selected_study)
        return
    if selected_id and not selected_study:
        st.session_state.review_selected_study_id = ""

    pending_records = [
        record for record in get_lifecycle_records()
        if normalize_lifecycle_status(record.get("Status")) in {
            "Ready For Review",
            "Pending Review",
            "Submitted for Review",
            "Under Review",
            "Scientific Review",
            "Follow-Up Required",
            "Approved",
        }
    ]
    pending_count = sum(1 for record in pending_records if review_queue_status(record) in {"Pending Review", "In Review"})
    followup_count = sum(1 for record in pending_records if review_queue_status(record) == "Follow-up Requested")
    approved_count = sum(1 for record in pending_records if review_queue_status(record) == "Approved")

    render_page_header("Review", "Review validation studies.", kicker="")

    if not pending_records:
        st.info("✓ No studies are currently waiting for review.")
    else:
        query_filter = st.query_params.get("review_filter", "")
        if isinstance(query_filter, list):
            query_filter = query_filter[0] if query_filter else ""
        review_filter = str(query_filter or st.session_state.get("review_queue_filter", "Pending"))
        if review_filter not in {"All", "Pending", "Follow-up", "Approved"}:
            review_filter = "Pending"
        st.session_state.review_queue_filter = review_filter
        tab_html = ['<div class="svap-review-tabs">']
        tab_counts = {
            "Pending": pending_count,
            "Follow-up": followup_count,
            "Approved": approved_count,
        }
        for option in ["Pending", "Follow-up", "Approved", "All"]:
            active_class = " svap-review-tab-active" if option == review_filter else ""
            label = f"{option} ({tab_counts[option]})" if option in tab_counts else option
            tab_html.append(
                f'<a class="svap-review-tab{active_class}" href="?page=Review&review_filter={quote(option)}" target="_self">'
                f"{escape(label)}</a>"
            )
        tab_html.append("</div>")
        st.markdown("".join(tab_html), unsafe_allow_html=True)
        filter_map = {
            "Pending": {"Pending Review", "In Review"},
            "Follow-up": {"Follow-up Requested"},
            "Approved": {"Approved"},
        }
        visible_records = [
            record
            for record in pending_records
            if review_filter == "All" or review_queue_status(record) in filter_map.get(str(review_filter), set())
        ]
        visible_records = sorted(visible_records, key=review_queue_sort_key)
        if not visible_records:
            empty_messages = {
                "Pending": "✓ No studies are currently waiting for review.",
                "Follow-up": "✓ No follow-up requests are open.",
                "Approved": "No approved studies match this view.",
            }
            st.info(empty_messages.get(str(review_filter), "No studies match this review view."))
            return

        show_status_column = review_filter == "All"
        list_class = "svap-review-list svap-review-list-all" if show_status_column else "svap-review-list svap-review-list-filtered"
        row_html = [
            f'<div class="{list_class}">'
        ]
        for record in visible_records:
            study_id = str(record.get("Study ID", ""))
            payload = load_review_package_record(study_id)
            package = (
                record.get("Review Package")
                or st.session_state.get("review_packages", {}).get(study_id)
                or (payload or {}).get("Review Package")
                or {}
            )
            metadata = package.get("Study Metadata", {}) if isinstance(package, dict) else {}
            analyte_name = str(record.get("Analyte") or record.get("Assay") or "")
            study_type = str(record.get("Study Type", ""))
            program_name = str(record.get("Program") or metadata.get("Program") or "")
            _, analyte_record = find_program_and_analyte(program_name, analyte_name)
            analyzer = str(
                (analyte_record or {}).get("Assigned Analyzer")
                or metadata.get("Assigned Analyzer")
                or ""
            ).strip()
            submitter = str(metadata.get("Analyst") or record.get("Analyst") or "")
            submitted = record.get("Submitted Date") or record.get("Completion Date")
            queue_status = review_queue_status(record)
            event_date = submitted
            event_person = submitter
            if review_filter == "Approved":
                event_date = record.get("Review Date") or record.get("Last Updated") or submitted
                event_person = str(record.get("Reviewer") or metadata.get("Reviewer") or submitter or "")
            event_text = review_submitted_text(event_date, event_person)
            analyzer_line = (
                f'<div class="svap-review-row-secondary">{escape(analyzer)}</div>'
                if analyzer
                else ""
            )
            row_html.append(
                f'<a class="svap-review-row" href="?page=Review&review_open={quote(study_id)}" target="_self">'
                "<div>"
                f'<div class="svap-review-row-analyte">{escape(analyte_name)}</div>'
                f"{analyzer_line}"
                "</div>"
                f'<div class="svap-review-row-study">{escape(study_type)}</div>'
                f'<div class="svap-review-row-submitted">{event_text}</div>'
            )
            if show_status_column:
                row_html.append(f"<div>{review_queue_status_badge(queue_status)}</div>")
            row_html.append('<div class="svap-review-chevron">›</div></a>')
        row_html.append("</div>")
        st.markdown("".join(row_html), unsafe_allow_html=True)


def render_reports_library() -> None:
    """Render the final output stage for approved scientific work."""

    initialize_platform_state()
    load_persisted_review_packages()
    render_page_header(
        "Reports",
        "Publish and access controlled validation reports.",
        kicker="",
    )
    selected_preview = str(st.session_state.get("report_package_preview_id", ""))
    selected_generated = str(st.session_state.get("report_generated_package_id", ""))
    package_items = build_report_package_items()
    if selected_preview:
        st.session_state.report_package_preview_id = ""
    if selected_generated:
        selected_record = next(
            (record for record in generated_report_packages() if str(record.get("Package ID")) == selected_generated),
            None,
        )
        if selected_record:
            render_generated_package_view(selected_record)
            return
        st.session_state.report_generated_package_id = ""
    ready_items = [item for item in package_items if item["Ready"]]
    render_ready_for_generation_table(ready_items)
    st.markdown('<div class="svap-report-item-divider svap-report-section-divider"></div>', unsafe_allow_html=True)
    render_generated_packages_table()


def generated_report_packages() -> list[dict[str, object]]:
    """Return generated immutable validation packages only."""

    initialize_platform_state()
    return [
        record for record in st.session_state.reports_library
        if str(record.get("Record Type", "")) == "Validation Package"
    ]


def display_report_version(version: object) -> str:
    """Return a reader-facing report version without implementation prefixes."""

    text = str(version or "").strip()
    return text[1:] if text.lower().startswith("v") else text


def report_version_parts(version: object) -> tuple[int, ...]:
    """Return numeric version parts for controlled report ordering."""

    text = display_report_version(version)
    parts: list[int] = []
    for part in text.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts or [0])


def report_record_sort_key(record: dict[str, object]) -> tuple[tuple[int, ...], pd.Timestamp, str]:
    """Return newest-first sortable key for generated report records."""

    generated = pd.to_datetime(str(record.get("Generated Date") or ""), errors="coerce")
    if pd.isna(generated):
        generated = pd.Timestamp.min
    return (
        report_version_parts(record.get("Version")),
        generated,
        str(record.get("Package ID", "")),
    )


def sorted_generated_report_packages() -> list[dict[str, object]]:
    """Return generated reports with newest versions first."""

    return sorted(generated_report_packages(), key=report_record_sort_key, reverse=True)


def generated_reports_for_source(source_package_id: str) -> list[dict[str, object]]:
    """Return generated reports for a single approved validation package source."""

    return [
        record for record in generated_report_packages()
        if str(record.get("Source Package ID", "")) == source_package_id
    ]


def latest_generated_report_for_source(source_package_id: str) -> dict[str, object] | None:
    """Return the current controlled report for a source package."""

    records = generated_reports_for_source(source_package_id)
    if not records:
        return None
    return max(records, key=report_record_sort_key)


def is_current_controlled_report(record: dict[str, object]) -> bool:
    """Return whether the record is the latest controlled version for its source."""

    latest = latest_generated_report_for_source(str(record.get("Source Package ID", "")))
    return bool(latest) and str(latest.get("Package ID", "")) == str(record.get("Package ID", ""))


def package_approval_signature(records: list[dict[str, object]]) -> str:
    """Return a stable signature for the approved study set used to publish a report."""

    payload = []
    for record in records:
        package = dict(record.get("Review Package") or {})
        payload.append(
            {
                "Study ID": str(record.get("Study ID", "")),
                "Study Type": str(record.get("Study Type", "")),
                "Version": str(record.get("Version", "1.0")),
                "Status": normalize_lifecycle_status(record.get("Status")),
                "Review Date": str(record.get("Review Date") or record.get("Last Updated") or ""),
                "Dataset Hash": str(package.get("Dataset Hash") or ""),
            }
        )
    ordered = sorted(payload, key=lambda item: (item["Study Type"], item["Study ID"]))
    return hashlib.sha256(json.dumps(ordered, sort_keys=True).encode("utf-8")).hexdigest()


def report_approval_signature(record: dict[str, object]) -> str:
    """Return the approval signature stored with a published report."""

    metadata = dict(record.get("Package Metadata") or {})
    return str(record.get("Approval Signature") or metadata.get("Approval Signature") or "")


def report_package_metadata(program: dict[str, object]) -> dict[str, object]:
    """Build read-only package metadata from approved platform records."""

    settings = platform_settings()
    program = normalize_program(program)
    owner = str(program.get("Program Owner") or settings.get("Analyst Name", ""))
    program_name = str(program.get("Program Name") or "")
    validation_context = str(program.get("Validation Context") or "")
    return {
        "Program": program_name,
        "Validation Context": validation_context,
        "Organization": str(settings.get("Organization Name", "")),
        "Laboratory": str(settings.get("Laboratory Name", "")),
        "Department": str(settings.get("Department", "")),
        "Generated By": owner,
        "Validation Project Name": program_name,
        "Assay / Biomarker": str(program.get("Assay / Biomarker") or ""),
        "Specimen Type": validation_context,
        "Analyst": owner,
        "Reviewer": str(program.get("Reviewer") or settings.get("Reviewer Name", "")),
        "Study Date": date.today().isoformat(),
        "Laboratory Name": settings.get("Laboratory Name", ""),
        "Organization Name": settings.get("Organization Name", ""),
        "Report Footer": settings.get("Report Footer", ""),
    }


def build_program_validation_package(program_name: str):
    """Compatibility wrapper for legacy export helpers."""

    eligible = eligible_study_types(program_name)
    if not eligible:
        return None
    program = next(
        (
            normalize_program(item)
            for item in st.session_state.validation_projects
            if normalize_program(item)["Program Name"] == program_name
        ),
        None,
    )
    if program is None:
        return None
    package = generate_validation_package(
        selected_studies=eligible,
        root_dir=ROOT_DIR,
        project_metadata=report_package_metadata(program),
    )
    for study in package.studies:
        study.status = program_study_status(program_name, study.study_type)
    return package


def build_report_package_items() -> list[dict[str, object]]:
    """Build package readiness records from approved lifecycle artifacts."""

    items: list[dict[str, object]] = []
    programs = [normalize_program(program) for program in st.session_state.validation_projects]
    for program in programs:
        program_name = str(program.get("Program Name", ""))
        for analyte in list(program.get("Analytes") or []):
            analyte_record = normalize_analyte(dict(analyte), fallback_name="")
            analyte_name = str(analyte_record.get("Analyte") or "").strip()
            if not analyte_name:
                continue
            required_studies = list(analyte_record.get("Required Studies") or DEFAULT_PROGRAM_REQUIRED_STUDIES)
            approved_records: list[dict[str, object]] = []
            missing: list[str] = []
            for study_type in required_studies:
                record = find_study_execution_record(program_name, analyte_name, str(study_type))
                status = normalize_lifecycle_status(record.get("Status") if record else "")
                if record and status in REPORT_ELIGIBLE_STATES:
                    approved_records.append(record)
                else:
                    missing.append(str(study_type))
            approved_count = len(approved_records)
            required_count = len(required_studies)
            package_name = f"{analyte_name} Validation Package"
            package_id = sanitize(f"{program_name}-{analyte_name}-validation-package")
            approval_signature = package_approval_signature(approved_records)
            latest_report = latest_generated_report_for_source(package_id)
            latest_signature = report_approval_signature(latest_report or {})
            has_current_published_report = bool(
                latest_report
                and approval_signature
                and latest_signature
                and latest_signature == approval_signature
            )
            ready = (
                required_count > 0
                and approved_count == required_count
                and not has_current_published_report
            )
            items.append(
                {
                    "Package ID": package_id,
                    "Package": package_name,
                    "Program": program_name,
                    "Analyte": analyte_name,
                    "Scope": f"{analyte_name} analyte package",
                    "Approved Studies": approved_count,
                    "Required Studies": required_count,
                    "Pending Studies": max(0, required_count - approved_count),
                    "Ready": ready,
                    "Status": "Ready" if ready else ("Published" if has_current_published_report else "Blocked"),
                    "Missing Studies": missing,
                    "Blocking Item": f"{missing[0]} Review" if missing else "",
                    "Approved Records": approved_records,
                    "Program Record": program,
                    "Approval Signature": approval_signature,
                }
            )
    return items


def report_package_contents() -> list[str]:
    """Return the standard contents generated for a validation package."""

    return [
        "Executive Summary",
        "Method Comparison Report",
        "Precision Report",
        "Accuracy Report",
        "Linearity Report",
        "Stability Report",
        "Detection Capability Report",
        "Reviewer Decisions",
        "Acceptance Criteria Summary",
        "Audit Metadata",
        "Supporting Appendices",
    ]


def next_package_version(package_id: str) -> str:
    """Return the next immutable package version for a package id."""

    existing = [
        record for record in generated_report_packages()
        if str(record.get("Source Package ID")) == package_id
    ]
    return f"v1.{len(existing)}"


def package_study_rows(item: dict[str, object]) -> list[dict[str, object]]:
    """Return approved study rows for preview and generated package views."""

    rows = []
    for record in list(item.get("Approved Records") or []):
        rows.append(
            {
                "Study": str(record.get("Study Type", "")),
                "Status": normalize_lifecycle_status(record.get("Status")),
                "Approved Date": format_review_date(record.get("Review Date") or record.get("Last Updated")),
                "Reviewer": str(record.get("Reviewer") or "Recorded reviewer"),
            }
        )
    return rows


def package_traceability_rows(item: dict[str, object], version: str) -> list[dict[str, object]]:
    """Return traceability rows for package preview and generated records."""

    rows = [
        {"Traceability Item": "Package Version", "Value": version},
        {"Traceability Item": "Study Versions", "Value": ", ".join(
            f"{record.get('Study Type')}: {record.get('Version', '1.0')}"
            for record in list(item.get("Approved Records") or [])
        ) or "Not available"},
        {"Traceability Item": "Reviewer Decisions", "Value": ", ".join(
            f"{record.get('Study Type')}: {normalize_lifecycle_status(record.get('Status'))}"
            for record in list(item.get("Approved Records") or [])
        ) or "Not available"},
        {"Traceability Item": "Dataset Hashes", "Value": ", ".join(
            sorted(
                {
                    str((record.get("Review Package") or {}).get("Dataset Hash") or "")
                    for record in list(item.get("Approved Records") or [])
                    if str((record.get("Review Package") or {}).get("Dataset Hash") or "")
                }
            )
        ) or "Stored with study artifacts"},
        {"Traceability Item": "Approval Timestamps", "Value": ", ".join(
            filter(
                None,
                [
                    format_review_date(record.get("Review Date") or record.get("Last Updated"))
                    for record in list(item.get("Approved Records") or [])
                ],
            )
        ) or "Not available"},
        {"Traceability Item": "Audit References", "Value": str(item.get("Package ID", ""))},
    ]
    return rows


def approved_review_artifacts(item: dict[str, object]) -> list[dict[str, object]]:
    """Return frozen approved review packages for report assembly."""

    artifacts: list[dict[str, object]] = []
    for record in list(item.get("Approved Records") or []):
        study_id = str(record.get("Study ID", ""))
        payload = load_review_package_record(study_id) if study_id else None
        package = (
            record.get("Review Package")
            or st.session_state.get("review_packages", {}).get(study_id)
            or (payload or {}).get("Review Package")
            or {}
        )
        artifacts.append(
            {
                "Study Record": dict(record),
                "Review Package": package if isinstance(package, dict) else {},
            }
        )
    return artifacts


def build_immutable_report_record(item: dict[str, object]) -> dict[str, object]:
    """Create an immutable generated validation package record."""

    generated_at = datetime.now()
    version = next_package_version(str(item.get("Package ID", "")))
    package_id = sanitize(f"{item.get('Package ID')}-{version}-{generated_at.strftime('%Y%m%d%H%M%S')}")
    audit_id = hashlib.sha256(package_id.encode("utf-8")).hexdigest()[:16].upper()
    study_rows = package_study_rows(item)
    approved_artifacts = approved_review_artifacts(item)
    traceability_rows = package_traceability_rows(item, version)
    package_metadata = report_package_metadata(dict(item.get("Program Record") or {}))
    assigned_analyzer, assigned_reagent = package_analyte_assignment(item)
    approval_signature = str(item.get("Approval Signature") or package_approval_signature(list(item.get("Approved Records") or [])))
    package_metadata.update(
        {
            "Analyte": str(item.get("Analyte", "")),
            "Assigned Analyzer": assigned_analyzer,
            "Assigned Reagent": assigned_reagent,
            "Package Version": version,
            "Generation Timestamp": generated_at.isoformat(timespec="seconds"),
            "Approval Timestamp": latest_package_approval_timestamp(list(item.get("Approved Records") or [])),
            "Approval Signature": approval_signature,
            "Package ID": package_id,
            "Audit Identifier": audit_id,
            "Study Versions": ", ".join(
                f"{record.get('Study Type')}: {record.get('Version', '1.0')}"
                for record in list(item.get("Approved Records") or [])
            ),
            "Reviewer Names": ", ".join(
                sorted({str(record.get("Reviewer") or "Recorded reviewer") for record in list(item.get("Approved Records") or [])})
            ),
            "Dataset Hashes": ", ".join(
                sorted(
                    {
                        str((record.get("Review Package") or {}).get("Dataset Hash") or "")
                        for record in list(item.get("Approved Records") or [])
                        if str((record.get("Review Package") or {}).get("Dataset Hash") or "")
                    }
                )
                or "Stored with study artifacts",
            ),
        }
    )
    return {
        "Record Type": "Validation Package",
        "Package ID": package_id,
        "Source Package ID": str(item.get("Package ID", "")),
        "Package Name": str(item.get("Package", "")),
        "Report Name": str(item.get("Package", "")),
        "Program": str(item.get("Program", "")),
        "Project": str(item.get("Program", "")),
        "Analyte": str(item.get("Analyte", "")),
        "Scope": str(item.get("Scope", "")),
        "Version": version,
        "Generated Date": generated_at.strftime("%Y-%m-%d %H:%M"),
        "Status": "Generated",
        "Approval Signature": approval_signature,
        "Included Studies": study_rows,
        "Approved Review Artifacts": approved_artifacts,
        "Traceability": traceability_rows,
        "Package Metadata": package_metadata,
    }


def latest_package_approval_timestamp(records: list[dict[str, object]]) -> str:
    """Return latest approval timestamp for a generated package."""

    dates = [
        pd.to_datetime(str(record.get("Review Date") or record.get("Last Updated") or ""), errors="coerce")
        for record in records
    ]
    valid_dates = [value for value in dates if not pd.isna(value)]
    if not valid_dates:
        return ""
    return max(valid_dates).isoformat()


def validation_report_name(value: object) -> str:
    """Return a document-facing report name from an internal package name."""

    name = str(value or "Validation Report").strip()
    if "Package" in name:
        return name.replace("Package", "Report")
    if "Report" not in name:
        return f"{name} Report"
    return name


def render_ready_for_generation_table(items: list[dict[str, object]]) -> None:
    """Render publication candidates whose source packages are approved."""

    st.subheader("Ready to Publish")
    if not items:
        st.markdown(
            '<div class="svap-report-empty-state">No validation reports are ready to publish.</div>',
            unsafe_allow_html=True,
        )
        return
    for index, item in enumerate(items):
        package_name_raw = str(item.get("Package", "") or "Validation Package")
        report_name = escape(validation_report_name(package_name_raw))
        version = escape(display_report_version(next_package_version(str(item.get("Package ID", "")))))
        approved = int(item.get("Approved Studies", 0))
        approved_date = format_review_date(latest_package_approval_timestamp(list(item.get("Approved Records") or [])))
        meta_parts = [package_name_raw, f"Version {version}", f"{approved} approved studies"]
        if approved_date:
            meta_parts.append(f"Approved {approved_date}")
        item_cols = st.columns([0.78, 0.09, 0.13], vertical_alignment="center")
        with item_cols[0]:
            st.markdown(
                f"""
                <div class="svap-report-ready-item">
                  <div class="svap-report-ready-title">{report_name}</div>
                  <div class="svap-report-ready-source-label">Source Validation Package</div>
                  <div class="svap-report-ready-meta">{escape(" • ".join(meta_parts))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with item_cols[1]:
            if st.button("Publish", key=f"preview_package_{item['Package ID']}", type="primary", use_container_width=True):
                record = build_immutable_report_record(item)
                st.session_state.reports_library.append(record)
                record_program_activity(
                    str(item.get("Program", "")),
                    str(item.get("Package", "")),
                    "Report published",
                    f"{validation_report_name(record['Package Name'])} {record['Version']} published from approved studies.",
                )
                st.session_state.report_package_preview_id = ""
                st.session_state.report_generated_package_id = str(record["Package ID"])
                st.rerun()
        if index < len(items) - 1:
            st.markdown('<div class="svap-report-item-divider"></div>', unsafe_allow_html=True)


def render_validation_package_preview(item: dict[str, object]) -> None:
    """Render a read-only package preview before immutable generation."""

    if st.button("Back to Reports", use_container_width=False):
        st.session_state.report_package_preview_id = ""
        st.rerun()
    version = next_package_version(str(item.get("Package ID", "")))
    st.subheader("Validation Package Preview")
    header = st.columns([1.8, 1.8, 1])
    header[0].markdown(f"**Package Name**  \n{escape(str(item.get('Package', '')))}")
    header[1].markdown(f"**Program**  \n{escape(str(item.get('Program', '')))}")
    header[2].markdown(f"**Version To Generate**  \n{version}")

    program_record = dict(item.get("Program Record") or {})
    st.markdown("**Package Status**")
    if item.get("Ready"):
        st.success(
            f"Ready for Report Generation · {item.get('Approved Studies', 0)} / {item.get('Required Studies', 0)} Studies Approved · "
            f"{item.get('Pending Studies', 0)} Pending"
        )
    else:
        st.error(
            f"Blocked · {item.get('Approved Studies', 0)} / {item.get('Required Studies', 0)} Studies Approved · "
            f"{item.get('Pending Studies', 0)} Pending"
        )
        st.write("Blocking studies: " + (", ".join(list(item.get("Missing Studies") or [])) or "None"))

    st.markdown("**Included Studies**")
    st.dataframe(pd.DataFrame(package_study_rows(item)), width="stretch", hide_index=True)

    st.markdown("**Package Summary**")
    assigned_analyzer, assigned_reagent = package_analyte_assignment(item)
    summary_rows = [
        {"Field": "Validation Context", "Value": str(program_record.get("Validation Context", ""))},
        {"Field": "Candidate Specimen", "Value": str(program_record.get("Candidate Specimen", ""))},
        {"Field": "Reference Specimen", "Value": str(program_record.get("Reference Specimen", ""))},
        {"Field": "Included Analytes", "Value": str(item.get("Analyte", ""))},
        {"Field": "Assigned Analyzer", "Value": assigned_analyzer},
        {"Field": "Assigned Reagent", "Value": assigned_reagent},
        {"Field": "Reviewer Approval Status", "Value": "All required studies approved" if item.get("Ready") else "Approval incomplete"},
    ]
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    if st.button("Publish Validation Report", type="primary", use_container_width=True, disabled=not bool(item.get("Ready"))):
        record = build_immutable_report_record(item)
        st.session_state.reports_library.append(record)
        record_program_activity(
            str(item.get("Program", "")),
            str(item.get("Package", "")),
            "Report published",
            f"{validation_report_name(record['Package Name'])} {record['Version']} published from approved studies.",
        )
        st.session_state.report_package_preview_id = ""
        st.session_state.report_generated_package_id = str(record["Package ID"])
        st.success("Validation report published.")
        st.rerun()


def report_value(value: object) -> str:
    """Return text safe for PDF/HTML report output."""

    text = str(value or "").replace("≥", ">=").replace("≤", "<=").replace("²", "2")
    return text.encode("latin-1", errors="replace").decode("latin-1")


def report_html_table(rows: list[dict[str, object]] | pd.DataFrame, columns: list[str] | None = None) -> str:
    """Render a compact HTML table for generated reports."""

    if isinstance(rows, pd.DataFrame):
        data = rows.copy()
    else:
        data = pd.DataFrame(rows)
    if data.empty:
        return "<p>No records available.</p>"
    if columns:
        visible = [column for column in columns if column in data.columns]
        data = data[visible] if visible else data
    header = "".join(f"<th>{escape(str(column))}</th>" for column in data.columns)
    body = ""
    for _, row in data.iterrows():
        body += "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row.tolist()) + "</tr>"
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def report_artifact_study_name(artifact: dict[str, object]) -> str:
    """Return a readable study label for a report artifact."""

    study = dict(artifact.get("Study Record") or {})
    package = dict(artifact.get("Review Package") or {})
    metadata = dict(package.get("Study Metadata") or {})
    return str(study.get("Study Type") or metadata.get("Study Type") or "Validation Study")


def report_figure_html(figure: object, title: str, include_plotlyjs: bool = False) -> str:
    """Render a frozen review figure for a standalone report document."""

    image_bytes = report_figure_png_bytes(figure, width=1100, height=620)
    if image_bytes:
        image_data = base64.b64encode(image_bytes).decode("ascii")
        return f'<img class="report-figure" alt="{escape(title)}" src="data:image/png;base64,{image_data}" />'
    if hasattr(figure, "to_html"):
        try:
            return figure.to_html(
                full_html=False,
                include_plotlyjs=True if include_plotlyjs else False,
                config={"displayModeBar": False, "staticPlot": True, "responsive": True},
            )
        except Exception:
            pass
    return ""


def report_numeric_pairs(x_values: object, y_values: object) -> list[tuple[float, float]]:
    """Return finite numeric x/y pairs from plot-like values."""

    try:
        x_list = list([] if x_values is None else x_values)
        y_list = list([] if y_values is None else y_values)
    except TypeError:
        return []
    pairs: list[tuple[float, float]] = []
    for x_value, y_value in zip(x_list, y_list):
        try:
            x_float = float(x_value)
            y_float = float(y_value)
        except (TypeError, ValueError):
            continue
        if pd.notna(x_float) and pd.notna(y_float):
            pairs.append((x_float, y_float))
    return pairs


def report_axis_title(axis: object, fallback: str) -> str:
    """Return a Plotly axis title as plain text."""

    axis_dict = dict(axis or {}) if isinstance(axis, dict) else {}
    title = axis_dict.get("title", "")
    if isinstance(title, dict):
        return str(title.get("text") or fallback)
    if title:
        return str(title)
    return fallback


def report_trace_color(trace: dict[str, object], index: int) -> tuple[int, int, int]:
    """Return a conservative RGB color for a report plot trace."""

    palette = [
        (37, 99, 235),
        (220, 38, 38),
        (75, 85, 99),
        (124, 58, 237),
        (5, 150, 105),
        (217, 119, 6),
    ]
    color = ""
    for source in (trace.get("line"), trace.get("marker")):
        if isinstance(source, dict) and source.get("color"):
            color = str(source["color"]).strip()
            break
    if color.startswith("#") and len(color) == 7:
        try:
            return tuple(int(color[position : position + 2], 16) for position in (1, 3, 5))  # type: ignore[return-value]
        except ValueError:
            pass
    simple_colors = {
        "red": (220, 38, 38),
        "blue": (37, 99, 235),
        "gray": (75, 85, 99),
        "grey": (75, 85, 99),
        "green": (22, 163, 74),
        "purple": (124, 58, 237),
        "orange": (217, 119, 6),
    }
    return simple_colors.get(color.lower(), palette[index % len(palette)])


def report_figure_png_bytes(figure: object, width: int = 980, height: int = 560) -> bytes:
    """Return a static PNG for a report figure without relying on app UI controls."""

    if hasattr(figure, "to_image"):
        try:
            return bytes(figure.to_image(format="png", width=width, height=height, scale=1))
        except Exception:
            pass
    if not hasattr(figure, "to_plotly_json"):
        return b""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return b""
    try:
        payload = figure.to_plotly_json()
    except Exception:
        return b""
    traces = list(payload.get("data") or []) if isinstance(payload, dict) else []
    layout = dict(payload.get("layout") or {}) if isinstance(payload, dict) else {}
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    margin_left, margin_top = 82, 54
    margin_right, margin_bottom = 176, 70
    plot_left, plot_top = margin_left, margin_top
    plot_right, plot_bottom = width - margin_right, height - margin_bottom
    grid_color = (226, 232, 240)
    axis_color = (148, 163, 184)
    text_color = (71, 85, 105)

    def values(source: object) -> list[object]:
        try:
            return list([] if source is None else source)
        except TypeError:
            return []

    def number(value: object) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if pd.notna(parsed) else None

    def numbers(source: object) -> list[float]:
        return [parsed for parsed in (number(item) for item in values(source)) if parsed is not None]

    def label(value: object) -> str:
        text = str(value if value is not None else "").strip()
        return text[:18] + "..." if len(text) > 21 else text

    title = layout.get("title", "")
    title_text = str(title.get("text") if isinstance(title, dict) else title or "")
    x_title = report_axis_title(layout.get("xaxis"), "Result")
    y_title = report_axis_title(layout.get("yaxis"), "Result")

    def y_domain(y_values: list[float]) -> tuple[float, float]:
        if not y_values:
            return 0.0, 1.0
        y_minimum, y_maximum = min(y_values), max(y_values)
        if y_minimum == y_maximum:
            y_minimum -= 1
            y_maximum += 1
        if y_minimum > 0:
            y_minimum = min(0.0, y_minimum)
        pad = (y_maximum - y_minimum) * 0.12
        return y_minimum - pad, y_maximum + pad

    def draw_frame(y_minimum: float, y_maximum: float, x_labels: list[str] | None = None) -> None:
        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=axis_color)
        for tick in range(6):
            ratio = tick / 5
            y_pixel = int(plot_bottom - ratio * (plot_bottom - plot_top))
            y_value = y_minimum + ratio * (y_maximum - y_minimum)
            draw.line((plot_left, y_pixel, plot_right, y_pixel), fill=grid_color)
            draw.text((plot_left - 52, y_pixel - 5), f"{y_value:.2f}", fill=text_color, font=font)
        if x_labels:
            for index, tick_label in enumerate(x_labels[:10]):
                if len(x_labels) == 1:
                    x_pixel = int((plot_left + plot_right) / 2)
                else:
                    x_pixel = int(plot_left + (index / (len(x_labels) - 1)) * (plot_right - plot_left))
                draw.line((x_pixel, plot_top, x_pixel, plot_bottom), fill=grid_color)
                draw.text((x_pixel - 18, plot_bottom + 12), label(tick_label), fill=text_color, font=font)
        else:
            for tick in range(6):
                ratio = tick / 5
                x_pixel = int(plot_left + ratio * (plot_right - plot_left))
                draw.line((x_pixel, plot_top, x_pixel, plot_bottom), fill=grid_color)
        if title_text:
            draw.text((plot_left, 22), title_text.replace("<br>", " "), fill=(31, 41, 55), font=font)
        draw.text((int((plot_left + plot_right) / 2) - 38, height - 34), x_title, fill=text_color, font=font)
        draw.text((18, int((plot_top + plot_bottom) / 2)), y_title, fill=text_color, font=font)

    def y_project(value: float, y_minimum: float, y_maximum: float) -> int:
        return int(plot_bottom - ((value - y_minimum) / (y_maximum - y_minimum)) * (plot_bottom - plot_top))

    def draw_horizontal_shapes(y_minimum: float, y_maximum: float) -> None:
        for shape in list(layout.get("shapes") or []):
            shape_dict = dict(shape or {})
            if str(shape_dict.get("type", "")).lower() != "line":
                continue
            y0 = number(shape_dict.get("y0"))
            y1 = number(shape_dict.get("y1"))
            if y0 is None or y1 is None or abs(y0 - y1) > 1e-9:
                continue
            y_pixel = y_project(y0, y_minimum, y_maximum)
            line = dict(shape_dict.get("line") or {})
            color = report_trace_color({"line": line}, 2)
            draw.line((plot_left, y_pixel, plot_right, y_pixel), fill=color, width=2)

    trace_dicts = [dict(trace or {}) for trace in traces]
    trace_types = {str(trace.get("type") or "scatter").lower() for trace in trace_dicts}

    def render_histogram() -> bool:
        histogram_values = [value for trace in trace_dicts for value in numbers(trace.get("x", []))]
        if not histogram_values:
            return False
        bin_count = min(12, max(5, int(len(histogram_values) ** 0.5)))
        x_minimum, x_maximum = min(histogram_values), max(histogram_values)
        if x_minimum == x_maximum:
            x_minimum -= 0.5
            x_maximum += 0.5
        bin_width = (x_maximum - x_minimum) / bin_count
        counts = [0] * bin_count
        for value in histogram_values:
            index = min(bin_count - 1, int((value - x_minimum) / bin_width))
            counts[index] += 1
        y_minimum, y_maximum = y_domain([float(item) for item in counts])
        draw_frame(y_minimum, y_maximum, [f"{x_minimum + i * bin_width:.1f}" for i in range(bin_count)])
        bar_width = (plot_right - plot_left) / bin_count * 0.72
        for index, count in enumerate(counts):
            x_center = plot_left + (index + 0.5) * (plot_right - plot_left) / bin_count
            y_pixel = y_project(float(count), y_minimum, y_maximum)
            color = report_trace_color(trace_dicts[0], 0)
            draw.rectangle((x_center - bar_width / 2, y_pixel, x_center + bar_width / 2, plot_bottom), fill=color)
        return True

    def render_box() -> bool:
        boxes: list[tuple[str, list[float], dict[str, object]]] = []
        for trace in trace_dicts:
            y_values = numbers(trace.get("y", []))
            if not y_values:
                continue
            name = str(trace.get("name") or "").strip()
            x_values = [label(item) for item in values(trace.get("x", [])) if str(item).strip()]
            box_label = name or (x_values[0] if x_values else "Results")
            boxes.append((box_label, sorted(y_values), trace))
        if not boxes:
            return False
        all_y = [value for _, y_values, _ in boxes for value in y_values]
        y_minimum, y_maximum = y_domain(all_y)
        draw_frame(y_minimum, y_maximum, [item[0] for item in boxes])
        draw_horizontal_shapes(y_minimum, y_maximum)
        for index, (_, y_values, trace) in enumerate(boxes):
            x_pixel = int(plot_left + ((index + 0.5) / len(boxes)) * (plot_right - plot_left))
            q1, median, q3 = np.percentile(y_values, [25, 50, 75])
            low, high = min(y_values), max(y_values)
            color = report_trace_color(trace, index)
            box_half = max(18, int((plot_right - plot_left) / max(8, len(boxes) * 5)))
            y_q1, y_q3 = y_project(float(q1), y_minimum, y_maximum), y_project(float(q3), y_minimum, y_maximum)
            y_median = y_project(float(median), y_minimum, y_maximum)
            y_low, y_high = y_project(float(low), y_minimum, y_maximum), y_project(float(high), y_minimum, y_maximum)
            draw.line((x_pixel, y_high, x_pixel, y_q3), fill=color, width=2)
            draw.line((x_pixel, y_q1, x_pixel, y_low), fill=color, width=2)
            draw.rectangle((x_pixel - box_half, y_q3, x_pixel + box_half, y_q1), outline=color, width=2)
            draw.line((x_pixel - box_half, y_median, x_pixel + box_half, y_median), fill=color, width=2)
            for value in y_values[:80]:
                y_point = y_project(value, y_minimum, y_maximum)
                draw.ellipse((x_pixel - 2, y_point - 2, x_pixel + 2, y_point + 2), fill=color)
        return True

    def render_bar() -> bool:
        categories: list[str] = []
        bars: list[tuple[dict[str, object], list[tuple[str, float]]]] = []
        for trace in trace_dicts:
            if str(trace.get("type") or "").lower() != "bar":
                continue
            x_values = values(trace.get("x", []))
            y_values = values(trace.get("y", []))
            pairs: list[tuple[str, float]] = []
            for index, y_value in enumerate(y_values):
                parsed = number(y_value)
                if parsed is None:
                    continue
                category = label(x_values[index] if index < len(x_values) else index + 1)
                if category not in categories:
                    categories.append(category)
                pairs.append((category, parsed))
            if pairs:
                bars.append((trace, pairs))
        if not bars:
            return False
        all_y = [value for _, pairs in bars for _, value in pairs]
        y_minimum, y_maximum = y_domain(all_y)
        draw_frame(y_minimum, y_maximum, categories)
        draw_horizontal_shapes(y_minimum, y_maximum)
        group_width = (plot_right - plot_left) / max(1, len(categories))
        bar_width = group_width / max(2, len(bars) + 1) * 0.72
        for trace_index, (trace, pairs) in enumerate(bars):
            color = report_trace_color(trace, trace_index)
            pair_map = dict(pairs)
            for category_index, category in enumerate(categories):
                if category not in pair_map:
                    continue
                x_center = plot_left + (category_index + 0.5) * group_width
                x_offset = (trace_index - (len(bars) - 1) / 2) * bar_width
                y_pixel = y_project(pair_map[category], y_minimum, y_maximum)
                baseline = y_project(0.0, y_minimum, y_maximum)
                draw.rectangle((x_center + x_offset - bar_width / 2, min(y_pixel, baseline), x_center + x_offset + bar_width / 2, max(y_pixel, baseline)), fill=color)
        return True

    def render_xy() -> bool:
        categories: list[str] = []
        plotted: list[tuple[dict[str, object], list[tuple[float, float]]]] = []
        x_numeric = True
        for trace in trace_dicts:
            y_values = values(trace.get("y", []))
            x_values = values(trace.get("x", []))
            if not x_values:
                x_values = list(range(1, len(y_values) + 1))
            pairs: list[tuple[float, float]] = []
            for index, y_value in enumerate(y_values):
                y_number = number(y_value)
                if y_number is None:
                    continue
                raw_x = x_values[index] if index < len(x_values) else index + 1
                x_number = number(raw_x)
                if x_number is None:
                    x_numeric = False
                    category = label(raw_x)
                    if category not in categories:
                        categories.append(category)
                    x_number = float(categories.index(category))
                pairs.append((x_number, y_number))
            if pairs:
                plotted.append((trace, pairs))
        if not plotted:
            return False
        all_x = [pair[0] for _, pairs in plotted for pair in pairs]
        all_y = [pair[1] for _, pairs in plotted for pair in pairs]
        x_minimum, x_maximum = min(all_x), max(all_x)
        y_minimum, y_maximum = y_domain(all_y)
        if x_minimum == x_maximum:
            x_minimum -= 1
            x_maximum += 1
        x_pad = (x_maximum - x_minimum) * 0.06
        x_minimum -= x_pad
        x_maximum += x_pad

        def project(point: tuple[float, float]) -> tuple[int, int]:
            x_value, y_value = point
            x_position = plot_left + ((x_value - x_minimum) / (x_maximum - x_minimum)) * (plot_right - plot_left)
            return int(x_position), y_project(y_value, y_minimum, y_maximum)

        draw_frame(y_minimum, y_maximum, categories if not x_numeric and categories else None)
        draw_horizontal_shapes(y_minimum, y_maximum)
        for index, (trace, pairs) in enumerate(plotted):
            color = report_trace_color(trace, index)
            mode = str(trace.get("mode") or "lines").lower()
            points = [project(pair) for pair in pairs]
            if "lines" in mode or mode == "lines":
                if len(points) > 1:
                    draw.line(points, fill=color, width=2)
            if "markers" in mode or mode == "markers" or "lines" not in mode:
                for x_pixel, y_pixel in points:
                    draw.ellipse((x_pixel - 3, y_pixel - 3, x_pixel + 3, y_pixel + 3), fill=color)
            name = str(trace.get("name") or "").strip()
            if name:
                legend_y = plot_top + index * 18
                draw.line((plot_right + 20, legend_y + 5, plot_right + 42, legend_y + 5), fill=color, width=2)
                draw.text((plot_right + 48, legend_y), name[:24], fill=text_color, font=font)
        return True

    rendered = False
    if "histogram" in trace_types:
        rendered = render_histogram()
    if not rendered and "box" in trace_types:
        rendered = render_box()
    if not rendered and "bar" in trace_types:
        rendered = render_bar()
    if not rendered:
        rendered = render_xy()
    if not rendered:
        return b""

    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def report_visualization_html(figures: dict[str, object]) -> str:
    """Render frozen review figures in report order without app controls."""

    fragments: list[str] = []
    plotlyjs_included = False
    for title, figure in figures.items():
        figure_html = report_figure_html(figure, str(title), include_plotlyjs=not plotlyjs_included)
        if not str(figure_html or "").strip():
            continue
        if "Plotly.newPlot" in figure_html or "plotly.js" in figure_html:
            plotlyjs_included = True
        fragments.append(
            f"""
            <figure>
              <figcaption>{escape(str(title))}</figcaption>
              {figure_html}
            </figure>
            """
        )
    return "".join(fragments)


REPORT_PRIMARY_FIGURE_PRIORITY = (
    "Regression Plot",
    "Precision Distribution Plot",
    "Expected vs Observed",
    "Recovery Plot",
    "Linearity Plot",
    "Recovery vs Time",
    "LoQ Decision Plot",
    "Bland-Altman Plot",
    "Residual Plot",
    "CV% Summary Plot",
    "Drift vs Time",
)


def report_clean_sentence(value: object) -> str:
    """Return report text as one readable sentence block."""

    return " ".join(str(value or "").replace("\n", " ").split())


def report_primary_figure(figures: dict[str, object]) -> tuple[str, object | None]:
    """Return the single figure that should represent a study in the report body."""

    available = [(str(title), figure) for title, figure in figures.items() if figure is not None]
    if not available:
        return "", None
    for preferred in REPORT_PRIMARY_FIGURE_PRIORITY:
        for title, figure in available:
            if preferred.lower() in title.lower():
                return title, figure
    return available[0]


def report_study_summary_text(section: dict[str, object]) -> str:
    """Return the concise report summary for one approved study."""

    finding = report_clean_sentence(section.get("Key Finding"))
    criteria_summary = report_clean_sentence(section.get("Acceptance Criteria Summary"))
    if finding and criteria_summary:
        return f"{finding} {criteria_summary}"
    return finding or criteria_summary or "Study findings were retained in the approved review artifact."


def report_study_conclusion_text(section: dict[str, object]) -> str:
    """Return the approved scientific conclusion for one study."""

    interpretation = report_clean_sentence(section.get("Scientific Interpretation"))
    if interpretation:
        return interpretation
    study_type = report_clean_sentence(section.get("Study Type")) or "Validation study"
    decision = report_clean_sentence(section.get("Decision")) or "approved"
    return f"{study_type} conclusion was retained in the approved review artifact. Final result: {decision}."


def report_study_key_results_table(section: dict[str, object], max_rows: int = 6) -> pd.DataFrame:
    """Return a compact, report-facing summary table for one study."""

    key_results = section.get("Key Results")
    if isinstance(key_results, pd.DataFrame) and not key_results.empty:
        data = key_results.copy()
    elif isinstance(key_results, list):
        data = pd.DataFrame(key_results)
    elif isinstance(key_results, dict):
        data = pd.DataFrame([{"Metric": metric, "Value": value} for metric, value in key_results.items()])
    else:
        data = pd.DataFrame()
    if not data.empty:
        if "Metric" not in data.columns or "Value" not in data.columns:
            data = pd.DataFrame([{"Metric": column, "Value": data.iloc[0][column]} for column in data.columns])
        return data[["Metric", "Value"]].head(max_rows)

    criteria = section.get("Acceptance Criteria")
    if isinstance(criteria, pd.DataFrame) and not criteria.empty:
        return pd.DataFrame(
            [
                {"Metric": "Acceptance Criteria", "Value": report_clean_sentence(section.get("Acceptance Criteria Summary"))},
            ]
        )
    return pd.DataFrame()


def report_study_section_html(section: dict[str, object]) -> str:
    """Render one approved study section for the report view and HTML export."""

    key_results = report_study_key_results_table(section)
    key_results_html = ""
    if isinstance(key_results, pd.DataFrame) and not key_results.empty:
        key_results_html = f"<h4>Key Results</h4>{report_html_table(key_results, ['Metric', 'Value'])}"
    primary_figure_title, primary_figure = report_primary_figure(dict(section["Critical Scientific Evidence"]))
    evidence_html = ""
    if primary_figure is not None:
        figure_html = report_figure_html(primary_figure, primary_figure_title, include_plotlyjs=True)
        if str(figure_html or "").strip():
            evidence_html = f"""
            <h4>Scientific Evidence</h4>
            <figure>
              <figcaption>{escape(primary_figure_title)}</figcaption>
              {figure_html}
            </figure>
            """
    return f"""
    <section class="study-section">
      <h3>{escape(str(section["Study Name"]))} <span class="badge">{escape(str(section["Decision"]))}</span></h3>
      <p class="muted">{escape(str(section["Reviewed Text"]))}</p>
      <h4>Objective</h4>
      <p>{escape(str(section["Scientific Question"]))}</p>
      <h4>Summary</h4>
      <p>{escape(report_study_summary_text(section))}</p>
      {key_results_html}
      {evidence_html}
      <h4>Conclusion</h4>
      <p>{escape(report_study_conclusion_text(section))}</p>
    </section>
    """


def report_artifact_study_context(artifact: dict[str, object]) -> dict[str, str]:
    """Return study identity fields from a frozen approved review artifact."""

    study = dict(artifact.get("Study Record") or {})
    package = dict(artifact.get("Review Package") or {})
    metadata = dict(package.get("Study Metadata") or {})
    study_type = str(study.get("Study Type") or metadata.get("Study Type") or "Validation Study")
    analyte = str(study.get("Analyte") or metadata.get("Analyte") or "").strip()
    study_name = f"{analyte} · {study_type}" if analyte else study_type
    decision = str(package.get("Decision") or study.get("Execution Result") or study.get("Status") or "").strip()
    return {
        "Analyte": analyte,
        "Study Type": study_type,
        "Study Name": study_name,
        "Decision": decision,
    }


def report_artifact_key_results_table(package: dict[str, object]) -> pd.DataFrame:
    """Return approved key result rows from a review package."""

    key_results = package.get("Key Results") if isinstance(package, dict) else {}
    if isinstance(key_results, pd.DataFrame):
        return key_results.copy()
    if isinstance(key_results, dict):
        return pd.DataFrame([{"Metric": metric, "Value": value} for metric, value in key_results.items()])
    if isinstance(key_results, list):
        return pd.DataFrame(key_results)
    return pd.DataFrame()


def report_artifact_criteria_table(package: dict[str, object]) -> pd.DataFrame:
    """Return acceptance criteria exactly in the reviewer-facing column shape."""

    criteria = package.get("Acceptance Criteria", pd.DataFrame()) if isinstance(package, dict) else pd.DataFrame()
    if isinstance(criteria, pd.DataFrame) and not criteria.empty:
        return review_criteria_display_table(criteria)
    if isinstance(criteria, list):
        data = pd.DataFrame(criteria)
        return review_criteria_display_table(data) if not data.empty else data
    return pd.DataFrame()


def criteria_status_summary(criteria: pd.DataFrame) -> str:
    """Return a compact acceptance criteria summary."""

    if criteria.empty or "Status" not in criteria.columns:
        return "Acceptance criteria retained in approved review artifact."
    statuses = criteria["Status"].astype(str).str.upper()
    passed = int(statuses.isin({"PASS", "PASSED", "✓"}).sum())
    failed = int(statuses.isin({"FAIL", "FAILED", "REJECTED"}).sum())
    total = len(criteria.index)
    if failed:
        return f"{passed} of {total} acceptance criteria passed; {failed} criterion requires attention."
    return f"All {total} acceptance criteria passed."


def report_primary_criteria_table(criteria: pd.DataFrame) -> pd.DataFrame:
    """Return only criteria rows that need primary report attention."""

    if criteria.empty or "Status" not in criteria.columns:
        return pd.DataFrame()
    statuses = criteria["Status"].astype(str).str.upper()
    failures = criteria[statuses.isin({"FAIL", "FAILED", "REJECTED"})].copy()
    return failures


def report_artifact_reviewer_rows(study: dict[str, object]) -> list[dict[str, object]]:
    """Return reviewer decision fields approved in Review."""

    rows = [
        {"Field": "Reviewer Decision", "Value": normalize_lifecycle_status(study.get("Status"))},
    ]
    reviewer = str(study.get("Reviewer") or "").strip()
    review_date = format_review_date(study.get("Review Date") or study.get("Last Updated"))
    comments = str(study.get("Reviewer Comments") or "").strip()
    follow_up = str(study.get("Required Follow-Up Actions") or "").strip()
    if reviewer:
        rows.append({"Field": "Reviewer", "Value": reviewer})
    if review_date:
        rows.append({"Field": "Review Date", "Value": review_date})
    if comments:
        rows.append({"Field": "Reviewer Comments", "Value": comments})
    if follow_up:
        rows.append({"Field": "Required Follow-Up", "Value": follow_up})
    return rows


def report_artifact_reviewed_text(study: dict[str, object]) -> str:
    """Return compact reviewer metadata for report study headings."""

    review_date = format_review_date(study.get("Review Date") or study.get("Last Updated"))
    reviewer = str(study.get("Reviewer") or "").strip()
    if review_date and reviewer:
        return f"Reviewed {review_date} by {reviewer}"
    if review_date:
        return f"Reviewed {review_date}"
    if reviewer:
        return f"Reviewed by {reviewer}"
    return "Reviewed"


def report_overall_validation_statement(record: dict[str, object], included_studies: list[dict[str, object]]) -> str:
    """Return the final validation statement for a published validation report."""

    metadata = dict(record.get("Package Metadata") or {})
    package_name = str(record.get("Package Name") or validation_report_name(record.get("Package Name")) or "Validation Package")
    context = str(metadata.get("Validation Context") or "the defined validation scope")
    candidate = str(metadata.get("Candidate Specimen") or "").strip()
    reference = str(metadata.get("Reference Specimen") or "").strip()
    comparison = f"{candidate} and {reference}" if candidate and reference else context
    study_count = len(included_studies)
    return (
        f"All {study_count} required validation studies were successfully completed and independently reviewed. "
        f"Based on the collected scientific evidence, {comparison} satisfies the predefined validation requirements "
        f"for {package_name}. Validation Package: APPROVED."
    )


def build_report_document(record: dict[str, object]) -> dict[str, object]:
    """Build the canonical report document consumed by view, PDF, HTML, and bundle exports."""

    metadata = dict(record.get("Package Metadata") or {})
    report_name = validation_report_name(record.get("Package Name") or "Validation Report")
    included_studies = list(record.get("Included Studies") or [])
    artifacts = list(record.get("Approved Review Artifacts") or [])
    if not artifacts:
        artifacts = [{"Study Record": row, "Review Package": {}} for row in included_studies]

    study_sections: list[dict[str, object]] = []
    for artifact in artifacts:
        study = dict(artifact.get("Study Record") or {})
        package = dict(artifact.get("Review Package") or {})
        context = report_artifact_study_context(artifact)
        study_type = context["Study Type"]
        analyte = context["Analyte"]
        criteria = report_artifact_criteria_table(package)
        key_results = report_artifact_key_results_table(package)
        supporting = package.get("Supporting Evidence", {}) if isinstance(package, dict) else {}
        supporting_tables = {
            str(title): table.copy() if isinstance(table, pd.DataFrame) else table
            for title, table in dict(supporting or {}).items()
        }
        study_sections.append(
            {
                "Study Record": study,
                "Review Package": package,
                "Study Name": context["Study Name"],
                "Study Type": study_type,
                "Analyte": analyte,
                "Decision": context["Decision"],
                "Reviewed Text": report_artifact_reviewed_text(study),
                "Scientific Question": review_scientific_question(study_type, analyte),
                "Key Finding": review_key_finding(study_type, context["Decision"]),
                "Scientific Interpretation": str(package.get("Interpretation") or "").strip(),
                "Key Results": key_results,
                "Acceptance Criteria": criteria,
                "Acceptance Criteria Summary": criteria_status_summary(criteria),
                "Primary Acceptance Criteria": report_primary_criteria_table(criteria),
                "Critical Scientific Evidence": dict(package.get("Visualizations") or {}),
                "Supporting Evidence": supporting_tables,
            }
        )

    if not included_studies and study_sections:
        included_studies = [
            {
                "Study": str(section.get("Study Type", "")),
                "Status": str(section.get("Decision", "")),
                "Approval Date": report_artifact_reviewed_text(dict(section.get("Study Record") or {})),
            }
            for section in study_sections
        ]

    validation_scope = str(metadata.get("Validation Context", "the defined validation scope"))
    executive_summary = {
        "Validation Objective": f"Evaluate {validation_scope} using approved validation studies.",
        "Validation Scope": str(metadata.get("Validation Context", "the defined validation scope")),
        "Approved Studies": f"{len(included_studies)} approved validation studies.",
        "Overall Validation Conclusion": "The approved validation evidence supports the final validation conclusion.",
    }
    validation_context = [
        {"Field": "Validation Program", "Value": str(record.get("Program", ""))},
        {"Field": "Validation Package", "Value": str(record.get("Package Name", ""))},
        {"Field": "Candidate Specimen", "Value": str(metadata.get("Candidate Specimen", ""))},
        {"Field": "Reference Specimen", "Value": str(metadata.get("Reference Specimen", ""))},
        {"Field": "Analyzer", "Value": str(metadata.get("Assigned Analyzer", ""))},
        {"Field": "Reagent", "Value": str(metadata.get("Assigned Reagent", ""))},
        {"Field": "Validation Protocol", "Value": str(metadata.get("Validation Context", ""))},
    ]
    appendix = [
        {"Field": "Report Version", "Value": str(record.get("Version", ""))},
        {"Field": "Publication Date", "Value": str(record.get("Generated Date", ""))},
        {"Field": "Approval Date", "Value": format_review_date(metadata.get("Approval Timestamp"))},
        {"Field": "Package Identifier", "Value": str(record.get("Package ID", ""))},
        {"Field": "Audit Identifier", "Value": str(metadata.get("Audit Identifier", ""))},
        {"Field": "Study Versions", "Value": str(metadata.get("Study Versions", ""))},
        {"Field": "Dataset Hashes", "Value": str(metadata.get("Dataset Hashes", ""))},
        {"Field": "Approval Signature", "Value": str(record.get("Approval Signature") or metadata.get("Approval Signature") or "")},
    ]
    return {
        "Report Name": report_name,
        "Version": str(record.get("Version", "")),
        "Publication Date": str(record.get("Generated Date", "")),
        "Approval Date": format_review_date(metadata.get("Approval Timestamp")),
        "Metadata": metadata,
        "Executive Summary": executive_summary,
        "Validation Context": validation_context,
        "Included Studies": included_studies,
        "Study Sections": study_sections,
        "Overall Validation Statement": report_overall_validation_statement(record, included_studies),
        "Appendix": appendix,
        "Version History": package_version_history(record),
        "Traceability": list(record.get("Traceability") or []),
    }


def report_safe_csv(data: object) -> str:
    """Return CSV text for a dataframe-like report artifact."""

    if isinstance(data, pd.DataFrame):
        return data.to_csv(index=False)
    if isinstance(data, list):
        return pd.DataFrame(data).to_csv(index=False)
    if isinstance(data, dict):
        return pd.DataFrame([{"Field": key, "Value": value} for key, value in data.items()]).to_csv(index=False)
    return pd.DataFrame([{"Value": str(data)}]).to_csv(index=False)


def pdf_heading(pdf: FPDF, text: str, size: int = 13) -> None:
    """Write a report heading."""

    pdf.set_font("Arial", "B", size)
    pdf.multi_cell(0, 7, report_value(text))
    pdf.ln(1)


def pdf_paragraph(pdf: FPDF, text: str, size: int = 9) -> None:
    """Write wrapped report body text."""

    if not str(text or "").strip():
        return
    pdf.set_font("Arial", "", size)
    for paragraph in str(text).split("\n"):
        if paragraph.strip():
            pdf.multi_cell(0, 5.5, report_value(paragraph.strip()))
    pdf.ln(1)


def pdf_text_fit(pdf: FPDF, text: object, width: float) -> str:
    """Return text truncated to fit a single PDF table cell."""

    value = report_value(text).replace("\n", " ").strip()
    if pdf.get_string_width(value) <= width - 3:
        return value
    ellipsis = "..."
    while value and pdf.get_string_width(value + ellipsis) > width - 3:
        value = value[:-1]
    return f"{value}{ellipsis}" if value else ellipsis


def pdf_ensure_space(pdf: FPDF, height_needed: float) -> None:
    """Add a page before writing a block that would overflow."""

    if pdf.get_y() + height_needed > pdf.h - pdf.b_margin:
        pdf.add_page()


def pdf_key_value(pdf: FPDF, label: str, value: object) -> None:
    """Write one key-value report row."""

    label_width = 58
    label_text = report_value(label)
    value_text = report_value(value)
    pdf.set_font("Arial", "B", 9)
    if pdf.get_string_width(label_text) > label_width - 2:
        pdf.multi_cell(0, 5.5, label_text)
        pdf.set_font("Arial", "", 9)
        pdf.multi_cell(0, 5.5, value_text)
        pdf.ln(1)
        return
    pdf.cell(label_width, 5.5, label_text, border=0)
    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(0, 5.5, value_text)


def pdf_table(pdf: FPDF, rows: list[dict[str, object]] | pd.DataFrame, columns: list[str], max_rows: int = 16) -> None:
    """Write a professional report table to PDF."""

    data = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if data.empty:
        pdf_paragraph(pdf, "No records available.")
        return
    visible = [column for column in columns if column in data.columns]
    if not visible:
        visible = list(data.columns)[:4]
    data = data[visible].head(max_rows)
    available_width = pdf.w - pdf.l_margin - pdf.r_margin
    if visible == ["Criterion", "Observed", "Required", "Status"]:
        widths = [available_width * 0.46, available_width * 0.22, available_width * 0.20, available_width * 0.12]
    elif visible == ["Metric", "Value"]:
        widths = [available_width * 0.40, available_width * 0.60]
    elif visible == ["Field", "Value"]:
        widths = [available_width * 0.34, available_width * 0.66]
    else:
        widths = [available_width / len(visible)] * len(visible)
    row_height = 7

    pdf_ensure_space(pdf, row_height * 2)
    pdf.set_fill_color(244, 247, 251)
    pdf.set_draw_color(214, 222, 232)
    pdf.set_text_color(16, 42, 67)
    pdf.set_font("Arial", "B", 8)
    for column, width in zip(visible, widths):
        align = "C" if column == "Status" else "L"
        pdf.cell(width, row_height, pdf_text_fit(pdf, column, width), border=1, align=align, fill=True)
    pdf.ln(row_height)

    pdf.set_font("Arial", "", 8)
    for _, row in data.iterrows():
        pdf_ensure_space(pdf, row_height)
        status = str(row.get("Status", "")).upper() if isinstance(row, pd.Series) else ""
        failed = status in {"FAIL", "FAILED", "REJECTED"}
        for column, width in zip(visible, widths):
            value = row.get(column, "")
            align = "C" if column == "Status" else "L"
            fill = bool(failed)
            if fill:
                pdf.set_fill_color(254, 242, 242)
            else:
                pdf.set_fill_color(255, 255, 255)
            if column == "Status" and "PASS" in str(value).upper():
                pdf.set_text_color(31, 122, 31)
                pdf.set_font("Arial", "B", 8)
            elif column == "Status" and failed:
                pdf.set_text_color(185, 28, 28)
                pdf.set_font("Arial", "B", 8)
            else:
                pdf.set_text_color(31, 41, 55)
                pdf.set_font("Arial", "", 8)
            pdf.cell(width, row_height, pdf_text_fit(pdf, value, width), border=1, align=align, fill=fill)
        pdf.ln(row_height)
    pdf.set_text_color(0, 0, 0)
    pdf.set_draw_color(0, 0, 0)
    if len(data.index) == max_rows:
        pdf_paragraph(pdf, "Additional rows are retained in the export bundle.", size=8)
    pdf.ln(2)


def pdf_add_plot(pdf: FPDF, figure: object, title: str) -> bool:
    """Add a Plotly figure to the PDF when static image export is available."""

    image_bytes = report_figure_png_bytes(figure, width=980, height=560)
    if not image_bytes:
        return False
    pdf_add_plot_image_bytes(pdf, image_bytes, title)
    return True


def pdf_add_plot_image_bytes(pdf: FPDF, image_bytes: bytes, title: str) -> None:
    """Add already-rendered PNG bytes to the PDF."""

    image_path = Path("/tmp") / f"{sanitize(title)}_{datetime.now().strftime('%H%M%S%f')}.png"
    image_path.write_bytes(bytes(image_bytes))
    try:
        pdf.image(str(image_path), w=180)
        pdf.ln(2)
    finally:
        try:
            image_path.unlink()
        except OSError:
            pass


def package_record_to_html(record: dict[str, object]) -> str:
    """Render the approved validation report as standalone print-friendly HTML."""

    document = build_report_document(record)
    report_name = str(document["Report Name"])
    study_sections = [report_study_section_html(section) for section in list(document["Study Sections"])]
    summary = dict(document["Executive Summary"])
    included_html = "".join(
        f"<li>{escape(str(row.get('Study', '')))}</li>"
        for row in list(document["Included Studies"])
        if str(row.get("Study", "")).strip()
    )
    return f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>{escape(report_name)}</title>
        <style>
          @page {{ margin: 0.7in; }}
          body {{ font-family: Arial, sans-serif; margin: 40px; color: #1f2933; line-height: 1.45; }}
          h1 {{ margin-bottom: 4px; font-size: 30px; }}
          h2 {{ border-top: 1px solid #d9e2ec; margin-top: 30px; padding-top: 18px; font-size: 22px; }}
          h3 {{ margin-top: 24px; font-size: 18px; }}
          h4 {{ margin: 18px 0 6px; font-size: 14px; }}
          table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
          th, td {{ border: 1px solid #d9e2ec; padding: 8px; text-align: left; vertical-align: top; }}
          th {{ background: #f0f4f8; color: #102a43; }}
          section.study-section {{ page-break-inside: avoid; margin-top: 20px; }}
          figure {{ margin: 16px 0 26px; page-break-inside: avoid; }}
          figcaption {{ font-weight: 700; margin-bottom: 8px; }}
          .report-figure {{ width: 100%; max-width: 980px; height: auto; border: 1px solid #d9e2ec; }}
          .muted {{ color: #52606d; }}
          .badge {{ background: #e3f9e5; border-radius: 999px; color: #1f7a1f; font-size: 12px; padding: 3px 8px; }}
          @media print {{
            body {{ margin: 0; }}
            a {{ color: inherit; text-decoration: none; }}
          }}
        </style>
      </head>
      <body>
        <h1>{escape(report_name)}</h1>
        <p class="muted">Version {escape(str(document["Version"]))} · Published {escape(str(document["Publication Date"]))}</p>
        <h2>Executive Summary</h2>
        <p><strong>Validation Objective</strong><br>{escape(str(summary["Validation Objective"]))}</p>
        <p><strong>Validation Scope</strong><br>{escape(str(summary["Validation Scope"]))}</p>
        <p><strong>Approved Studies</strong><br>{escape(str(summary["Approved Studies"]))}</p>
        <p><strong>Overall Validation Conclusion</strong><br>{escape(str(summary["Overall Validation Conclusion"]))}</p>
        <h2>Validation Context</h2>
        {report_html_table(list(document["Validation Context"]), ["Field", "Value"])}
        <h2>Included Studies</h2>
        <ul>{included_html}</ul>
        <h2>Approved Validation Studies</h2>
        {"".join(study_sections)}
        <h2>Overall Validation Conclusion</h2>
        <p>{escape(str(document["Overall Validation Statement"]))}</p>
        <h2>Appendix</h2>
        {report_html_table(list(document["Appendix"]), ["Field", "Value"])}
      </body>
    </html>
    """


def package_record_to_pdf(record: dict[str, object]) -> bytes:
    """Render a generated package record as the final validation report PDF."""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    document = build_report_document(record)
    report_name = str(document["Report Name"])
    summary = dict(document["Executive Summary"])

    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.multi_cell(0, 10, report_value(report_name))
    pdf.ln(6)
    pdf_key_value(pdf, "Validation Program", record.get("Program", ""))
    pdf_key_value(pdf, "Validation Package", record.get("Package Name", ""))
    pdf_key_value(pdf, "Report Version", document["Version"])
    pdf_key_value(pdf, "Publication Date", document["Publication Date"])
    pdf_key_value(pdf, "Approval Date", document["Approval Date"])

    pdf.add_page()
    pdf_heading(pdf, "Executive Summary")
    for label, value in summary.items():
        pdf_key_value(pdf, label, value)

    pdf_heading(pdf, "Validation Context")
    for row in list(document["Validation Context"]):
        pdf_key_value(pdf, str(row.get("Field", "")), row.get("Value", ""))

    pdf_heading(pdf, "Included Studies")
    for row in list(document["Included Studies"]):
        study_name = str(row.get("Study", "")).strip()
        if study_name:
            pdf_paragraph(pdf, f"- {study_name}", size=9)

    pdf_heading(pdf, "Approved Validation Studies")
    for section in list(document["Study Sections"]):
        pdf_heading(pdf, str(section["Study Name"]), size=12)
        pdf_paragraph(pdf, f"{section['Decision']} | {section['Reviewed Text']}", size=8)
        pdf.set_font("Arial", "B", 9)
        pdf.cell(0, 6, "Objective", ln=True)
        pdf_paragraph(pdf, str(section["Scientific Question"]), size=8)
        pdf.set_font("Arial", "B", 9)
        pdf.cell(0, 6, "Summary", ln=True)
        pdf_paragraph(pdf, report_study_summary_text(section), size=8)
        key_results = report_study_key_results_table(section)
        if isinstance(key_results, pd.DataFrame) and not key_results.empty:
            pdf.set_font("Arial", "B", 9)
            pdf.cell(0, 6, "Key Results", ln=True)
            pdf_table(pdf, key_results, ["Metric", "Value"], max_rows=6)
        pdf.set_font("Arial", "B", 9)
        primary_figure_title, primary_figure = report_primary_figure(dict(section["Critical Scientific Evidence"]))
        if primary_figure is not None:
            image_bytes = report_figure_png_bytes(primary_figure, width=980, height=560)
        else:
            image_bytes = b""
        if image_bytes:
            pdf_ensure_space(pdf, 115)
            pdf.set_font("Arial", "B", 9)
            pdf.cell(0, 6, "Scientific Evidence", ln=True)
            pdf.set_font("Arial", "B", 8)
            pdf.cell(0, 5, report_value(primary_figure_title), ln=True)
            pdf_add_plot_image_bytes(pdf, image_bytes, primary_figure_title)
        pdf.set_font("Arial", "B", 9)
        pdf.cell(0, 6, "Conclusion", ln=True)
        pdf_paragraph(pdf, report_study_conclusion_text(section), size=8)
        pdf.ln(3)

    pdf_heading(pdf, "Overall Validation Conclusion")
    pdf_paragraph(pdf, str(document["Overall Validation Statement"]))

    pdf_heading(pdf, "Appendix")
    pdf_table(pdf, list(document["Appendix"]), ["Field", "Value"], max_rows=20)
    output = pdf.output(dest="S")
    if isinstance(output, str):
        return output.encode("latin-1")
    return bytes(output)


def package_record_to_bundle(record: dict[str, object]) -> bytes:
    """Build a regulatory export bundle from an immutable package record."""

    document = build_report_document(record)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report/validation_report.html", package_record_to_html(record))
        archive.writestr("report/validation_report.pdf", package_record_to_pdf(record))

        manifest = {
            "Report Name": document["Report Name"],
            "Version": document["Version"],
            "Publication Date": document["Publication Date"],
            "Approval Date": document["Approval Date"],
            "Package ID": record.get("Package ID", ""),
            "Source Package ID": record.get("Source Package ID", ""),
            "Approval Signature": record.get("Approval Signature", ""),
            "Included Studies": [
                {
                    "Study": section["Study Name"],
                    "Decision": section["Decision"],
                    "Reviewed": section["Reviewed Text"],
                }
                for section in list(document["Study Sections"])
            ],
        }
        archive.writestr("manifest/manifest.json", json.dumps(manifest, indent=2, default=str))
        archive.writestr("manifest/source_record.json", json.dumps(record, indent=2, default=str))
        archive.writestr("manifest/hash_manifest.json", json.dumps({
            "Package ID": record.get("Package ID", ""),
            "Audit Identifier": (record.get("Package Metadata") or {}).get("Audit Identifier", ""),
            "Dataset Hashes": (record.get("Package Metadata") or {}).get("Dataset Hashes", ""),
            "Approval Signature": record.get("Approval Signature", ""),
        }, indent=2, default=str))

        archive.writestr("audit/appendix_metadata.csv", report_safe_csv(list(document["Appendix"])))
        archive.writestr("audit/package_traceability.csv", report_safe_csv(list(document["Traceability"])))
        archive.writestr("audit/version_history.csv", report_safe_csv(list(document["Version History"])))
        archive.writestr("appendices/included_studies.csv", report_safe_csv(list(document["Included Studies"])))
        archive.writestr("appendices/validation_context.csv", report_safe_csv(list(document["Validation Context"])))
        archive.writestr(
            "appendices/overall_validation_statement.txt",
            str(document["Overall Validation Statement"]),
        )

        for section in list(document["Study Sections"]):
            study_folder = f"studies/{sanitize(section['Study Name'])}"
            archive.writestr(
                f"{study_folder}/study_manifest.json",
                json.dumps(
                    {
                        "Study": section["Study Name"],
                        "Study Type": section["Study Type"],
                        "Analyte": section["Analyte"],
                        "Decision": section["Decision"],
                        "Reviewed": section["Reviewed Text"],
                        "Scientific Question": section["Scientific Question"],
                        "Key Finding": section["Key Finding"],
                        "Acceptance Criteria Summary": section["Acceptance Criteria Summary"],
                    },
                    indent=2,
                    default=str,
                ),
            )
            archive.writestr(f"{study_folder}/scientific_interpretation.txt", str(section["Scientific Interpretation"]))
            archive.writestr(f"{study_folder}/key_results.csv", report_safe_csv(section["Key Results"]))
            archive.writestr(f"{study_folder}/acceptance_criteria.csv", report_safe_csv(section["Acceptance Criteria"]))
            for title, table in dict(section["Supporting Evidence"]).items():
                archive.writestr(f"{study_folder}/supporting_evidence/{sanitize(title)}.csv", report_safe_csv(table))
            for title, figure in dict(section["Critical Scientific Evidence"]).items():
                if hasattr(figure, "to_html"):
                    try:
                        archive.writestr(
                            f"{study_folder}/figures/{sanitize(title)}.html",
                            figure.to_html(full_html=True, include_plotlyjs="cdn"),
                        )
                        continue
                    except Exception:
                        pass
                archive.writestr(f"{study_folder}/figures/{sanitize(title)}.txt", "Figure retained in approved review artifact.")
    buffer.seek(0)
    return buffer.getvalue()


def render_generated_packages_table() -> None:
    """Render immutable generated validation packages."""

    st.subheader("Controlled Validation Reports")
    records = sorted_generated_report_packages()
    if not records:
        st.markdown(
            '<div class="svap-report-empty-state">No controlled validation reports have been published.</div>',
            unsafe_allow_html=True,
        )
        return
    for index, record in enumerate(records):
        package_name = validation_report_name(record.get("Package Name") or record.get("Report Name") or "Validation Report")
        package_id = str(record.get("Package ID", f"package-{index}"))
        if is_current_controlled_report(record):
            status_html = '<span class="svap-report-status-published">Current Controlled Report</span>'
        else:
            status_html = '<span class="svap-report-status-superseded">Superseded</span>'
        raw_version = str(record.get("Version", "")).replace("Version ", "").strip()
        version_text = raw_version[1:] if raw_version.lower().startswith("v") else raw_version
        published_date = format_review_date(record.get("Generated Date"))
        st.markdown(
            f"""
            <div class="svap-published-report">
              <div class="svap-published-report-heading">
                <span class="svap-published-report-title">{escape(package_name)}</span>
                {status_html}
              </div>
              <div class="svap-published-report-meta">Version {escape(version_text)} &bull; Published {escape(published_date)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        action_cols = st.columns([0.16, 0.13, 0.13, 0.17, 0.41])
        with action_cols[0]:
            if st.button("View Report", key=f"open_generated_package_{package_id}", type="primary", use_container_width=True):
                st.session_state.report_generated_package_id = package_id
                st.session_state.report_package_preview_id = ""
                st.rerun()
        with action_cols[1]:
            st.download_button(
                "Download PDF",
                data=package_record_to_pdf(record),
                file_name=f"{sanitize(package_name)}_{record.get('Version', 'v1')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"table_pdf_{package_id}",
            )
        with action_cols[2]:
            st.download_button(
                "Download HTML",
                data=package_record_to_html(record).encode("utf-8"),
                file_name=f"{sanitize(package_name)}_{record.get('Version', 'v1')}.html",
                mime="text/html",
                use_container_width=True,
                key=f"table_html_{package_id}",
            )
        with action_cols[3]:
            st.download_button(
                "Download Package",
                data=package_record_to_bundle(record),
                file_name=f"{sanitize(package_name)}_{record.get('Version', 'v1')}_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"table_bundle_{package_id}",
            )
        st.write("")


def render_published_report_content(record: dict[str, object]) -> None:
    """Render the scientific report content from approved Review artifacts."""

    document = build_report_document(record)
    sections = list(document["Study Sections"])
    if not sections:
        st.write("No approved review artifacts are available for this report.")
        return
    for section in sections:
        decision = str(section["Decision"])
        result_class = "status-pass" if decision.upper() in {"PASS", "APPROVED"} else "status-neutral"
        st.markdown(
            f"""
            <div class="svap-report-content-section">
              <div class="svap-report-content-title">{escape(str(section["Study Name"]))} <span class="status-badge {result_class}">{escape(decision)}</span></div>
              <div class="svap-report-content-meta">{escape(str(section["Reviewed Text"]))}</div>
              <div class="svap-report-content-meta"><strong>Objective</strong><br>{escape(str(section["Scientific Question"]))}</div>
              <div class="svap-report-content-meta"><strong>Summary</strong><br>{escape(report_study_summary_text(section))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        criteria = section["Acceptance Criteria"]
        key_results = section["Key Results"]
        key_results_summary = report_study_key_results_table(section)
        if isinstance(key_results_summary, pd.DataFrame) and not key_results_summary.empty:
            st.markdown("**Key Results**")
            st.dataframe(key_results_summary, width="stretch", hide_index=True)

        primary_figure_title, primary_figure = report_primary_figure(dict(section["Critical Scientific Evidence"]))
        if primary_figure is not None:
            st.markdown("**Scientific Evidence**")
            st.markdown(f"**{escape(primary_figure_title)}**")
            if hasattr(primary_figure, "to_html"):
                st.plotly_chart(primary_figure, width="stretch")
            else:
                st.caption("Figure retained in approved review artifact.")

        st.markdown("**Conclusion**")
        st.write(report_study_conclusion_text(section))

        if isinstance(criteria, pd.DataFrame) and not criteria.empty:
            criteria_count = len(criteria.index)
            with st.expander(f"Acceptance Criteria ({criteria_count} criteria)", expanded=False):
                st.write(str(section["Acceptance Criteria Summary"]))
                st.dataframe(criteria, width="stretch", hide_index=True)

        figures = dict(section["Critical Scientific Evidence"])
        secondary_figures = {
            title: figure
            for title, figure in figures.items()
            if str(title) != primary_figure_title
        }
        if secondary_figures:
            figure_count = len(secondary_figures)
            plot_label = "plot" if figure_count == 1 else "plots"
            with st.expander(f"Additional Scientific Evidence ({figure_count} {plot_label})", expanded=False):
                for title, figure in secondary_figures.items():
                    st.markdown(f"**{escape(str(title))}**")
                    if hasattr(figure, "to_html"):
                        st.plotly_chart(figure, width="stretch")
                    else:
                        st.caption("Figure retained in approved review artifact.")

        supporting = dict(section["Supporting Evidence"])
        if (isinstance(key_results, pd.DataFrame) and not key_results.empty) or supporting:
            with st.expander("Supporting Evidence", expanded=False):
                if isinstance(key_results, pd.DataFrame) and not key_results.empty:
                    st.markdown("**Additional Statistics**")
                    st.dataframe(key_results, width="stretch", hide_index=True)
                for title, table in supporting.items():
                    st.markdown(f"**{escape(str(title))}**")
                    if isinstance(table, pd.DataFrame):
                        if table.empty:
                            st.caption("No records available.")
                        else:
                            st.dataframe(table, width="stretch", hide_index=True)
                    else:
                        st.write(table)


def package_version_history(record: dict[str, object]) -> list[dict[str, object]]:
    """Return generated package versions for the same source package."""

    source_id = str(record.get("Source Package ID", ""))
    versions: list[dict[str, object]] = []
    for item in sorted(generated_reports_for_source(source_id), key=report_record_sort_key, reverse=True):
        versions.append(
            {
                "Version": str(item.get("Version", "")),
                "Generated Date": format_review_date(item.get("Generated Date")),
                "Status": "Current controlled version" if is_current_controlled_report(item) else "Superseded",
                "Package ID": str(item.get("Package ID", "")),
            }
        )
    return versions


def render_generated_package_view(record: dict[str, object]) -> None:
    """Render a generated immutable validation package."""

    if st.button("Back to Reports", use_container_width=False):
        st.session_state.report_generated_package_id = ""
        st.rerun()
    metadata = dict(record.get("Package Metadata") or {})
    document = build_report_document(record)
    package_name = validation_report_name(record.get("Package Name") or "Validation Report")
    package_id = str(record.get("Package ID", "generated_package"))
    st.subheader(package_name)
    header = st.columns([0.8, 1, 1, 2.0])
    header[0].markdown(f"**Version**  \n{escape(str(record.get('Version', '')))}")
    header[1].markdown(f"**Publication Date**  \n{escape(format_review_date(record.get('Generated Date')))}")
    header[2].markdown(f"**Approval Date**  \n{escape(format_review_date(metadata.get('Approval Timestamp')))}")
    with header[3]:
        action_cols = st.columns([0.85, 0.9, 1.15])
        with action_cols[0]:
            st.download_button(
                "Download PDF",
                data=package_record_to_pdf(record),
                file_name=f"{sanitize(package_name)}_{record.get('Version', 'v1')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"view_pdf_{package_id}",
            )
        with action_cols[1]:
            st.download_button(
                "Download HTML",
                data=package_record_to_html(record).encode("utf-8"),
                file_name=f"{sanitize(package_name)}_{record.get('Version', 'v1')}.html",
                mime="text/html",
                use_container_width=True,
                key=f"view_html_{package_id}",
            )
        with action_cols[2]:
            st.download_button(
                "Download Validation Package",
                data=package_record_to_bundle(record),
                file_name=f"{sanitize(package_name)}_{record.get('Version', 'v1')}_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"view_bundle_{package_id}",
            )

    included_studies = list(document["Included Studies"])
    st.markdown("### Executive Summary")
    summary = dict(document["Executive Summary"])
    st.markdown(
        f"""
        <div class="svap-report-summary">
          <strong>Validation Objective</strong><br>{escape(str(summary["Validation Objective"]))}<br><br>
          <strong>Validation Scope</strong><br>{escape(str(summary["Validation Scope"]))}<br><br>
          <strong>Approved Studies</strong><br>{escape(str(summary["Approved Studies"]))}<br><br>
          <strong>Overall Validation Conclusion</strong><br>{escape(str(summary["Overall Validation Conclusion"]))}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Included Studies")
    study_names = [escape(str(row.get("Study", ""))) for row in included_studies if str(row.get("Study", "")).strip()]
    if study_names:
        st.markdown(
            '<div class="svap-report-study-list">'
            + "".join(f"<div>✓ {name}</div>" for name in study_names)
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.write("No studies included.")

    st.markdown("### Approved Validation Studies")
    render_published_report_content(record)

    st.markdown("### Overall Validation Statement")
    st.markdown(
        f'<div class="svap-report-summary">{escape(str(document["Overall Validation Statement"]))}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Audit & Traceability", expanded=False):
        st.markdown("**Version History**")
        version_history = pd.DataFrame(package_version_history(record))
        if not version_history.empty:
            st.dataframe(version_history[["Version", "Generated Date", "Status"]], width="stretch", hide_index=True)
        else:
            st.write("No prior versions.")
        st.markdown("**Traceability Information**")
        st.dataframe(pd.DataFrame(list(record.get("Traceability") or [])), width="stretch", hide_index=True)
        st.markdown("**Audit Information**")
        audit_rows = [
            {"Field": "Package Version", "Value": metadata.get("Package Version", record.get("Version", ""))},
            {"Field": "Study Versions", "Value": metadata.get("Study Versions", "")},
            {"Field": "Reviewer Decisions", "Value": ", ".join(str(row.get("Status", "")) for row in included_studies)},
            {"Field": "Approval Timestamps", "Value": metadata.get("Approval Timestamp", "")},
            {"Field": "Dataset Hashes", "Value": metadata.get("Dataset Hashes", "")},
            {"Field": "Package Identifier", "Value": record.get("Package ID", "")},
            {"Field": "Audit Identifier", "Value": metadata.get("Audit Identifier", "")},
            {"Field": "Generated By", "Value": metadata.get("Generated By", "")},
            {"Field": "Internal References", "Value": str(record.get("Source Package ID", ""))},
        ]
        st.dataframe(pd.DataFrame(audit_rows), width="stretch", hide_index=True)


def render_blocked_packages_table(items: list[dict[str, object]]) -> None:
    """Render blocked packages as secondary information."""

    with st.expander("Blocked Packages", expanded=False):
        if not items:
            st.write("No blocked packages.")
            return
        header = st.columns([1.4, 1.1, 2])
        for column, label in zip(header, ["Package", "Progress", "Blocker"]):
            column.markdown(f"**{label}**")
        for item in items:
            row = st.columns([1.4, 1.1, 2])
            row[0].write(str(item.get("Package", "")))
            row[1].write(f"{item.get('Approved Studies', 0)} / {item.get('Required Studies', 0)}")
            row[2].write(str(item.get("Blocking Item") or "Review pending"))


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
            st.session_state.pending_page = "Execution"
            st.rerun()


def render_platform_settings() -> None:
    """Render centralized platform settings."""

    initialize_platform_state()
    render_page_header(
        "Settings",
        "Organization setup and reporting defaults.",
        kicker="",
    )
    settings = platform_settings().copy()

    st.subheader("Organization")
    org_cols = st.columns(4)
    with org_cols[0]:
        settings["Organization Name"] = st.text_input("Organization Name", value=settings.get("Organization Name", ""))
    with org_cols[1]:
        settings["Laboratory Name"] = st.text_input("Laboratory Name", value=settings.get("Laboratory Name", ""))
    with org_cols[2]:
        settings["Department"] = st.text_input("Department", value=settings.get("Department", ""))
    with org_cols[3]:
        settings["Organization Logo"] = st.text_input("Organization Logo", value=settings.get("Organization Logo", settings.get("Report Logo", "")))

    st.subheader("Users")
    user_cols = st.columns(3)
    with user_cols[0]:
        settings["Analyst Name"] = st.text_input("Default Analyst", value=settings.get("Analyst Name", ""))
    with user_cols[1]:
        settings["Reviewer Name"] = st.text_input("Default Reviewer", value=settings.get("Reviewer Name", ""))
    with user_cols[2]:
        settings["Approver Name"] = st.text_input("Default Approver", value=settings.get("Approver Name", ""))

    st.subheader("Reports")
    report_cols = st.columns(3)
    with report_cols[0]:
        settings["Default Export Format"] = st.selectbox(
            "Default Export Format",
            ["PDF", "HTML", "PDF + HTML"],
            index=["PDF", "HTML", "PDF + HTML"].index(settings.get("Default Export Format", "PDF + HTML"))
            if settings.get("Default Export Format", "PDF + HTML") in ["PDF", "HTML", "PDF + HTML"]
            else 2,
        )
    with report_cols[1]:
        settings["Report Footer"] = st.text_input("Report Footer", value=settings.get("Report Footer", ""))
    with report_cols[2]:
        settings["Logo Inclusion"] = st.selectbox(
            "Logo Inclusion",
            ["Include", "Exclude"],
            index=0 if settings.get("Logo Inclusion", "Include") == "Include" else 1,
        )
    settings["Organization Branding"] = st.text_input("Organization Branding", value=settings.get("Organization Branding", ""))

    if st.button("Save Settings", type="primary"):
        settings["Report Logo"] = settings.get("Organization Logo", "")
        settings["Default Report Format"] = settings.get("Default Export Format", "PDF + HTML")
        for removed_key in ["Address", "PDF Settings"]:
            settings.pop(removed_key, None)
        st.session_state.platform_settings = settings
        st.success("Settings saved.")


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
        render_metric_card("Purpose", "Validation Execution")
    with cols[2]:
        render_metric_card("Reports", "HTML / PDF")

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
        st.subheader("Generate")

    program_names = [str(normalize_program(program)["Program Name"]) for program in st.session_state.validation_projects]
    selected_program = st.selectbox("Validation Program", program_names)
    program = next(
        normalize_program(item)
        for item in st.session_state.validation_projects
        if normalize_program(item)["Program Name"] == selected_program
    )
    analyst = str(program.get("Program Owner") or settings.get("Analyst Name", ""))
    reviewer = str(program.get("Reviewer") or settings.get("Reviewer Name", ""))
    st.caption(f"{selected_program} · Approved or locked studies only")
    project_metadata = {
        "Validation Project Name": selected_program,
        "Analyst": analyst,
        "Study Date": date.today().isoformat(),
        "Instrument": "",
        "Assay / Biomarker": str(program.get("Assay / Biomarker") or ""),
        "Specimen Type": str(program.get("Validation Context") or ""),
        "Protocol Number": "",
        "Reviewer": reviewer,
        "Laboratory Name": settings.get("Laboratory Name", ""),
        "Report Version": "",
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
    st.markdown("### Include")
    selected_studies: list[str] = []
    cols = st.columns(3)
    for index, study_name in enumerate(SUPPORTED_STUDIES):
        if study_name not in eligible:
            continue
        with cols[index % 3]:
            if st.checkbox(study_name, value=True):
                selected_studies.append(study_name)

    if st.button("Generate Package", type="primary"):
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
            "Report Name": selected_program,
            "Project": selected_program,
            "Study Type": "Validation Reports",
            "Date": date.today().isoformat(),
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

    study_names = get_study_type_names()
    requested = st.session_state.pop("preselected_study_type", None)
    if requested in study_names:
        render_page_header(
            requested,
            "Upload data, run analysis, and review scientific results.",
            kicker="",
        )
        return requested

    render_page_header(
        "Execution",
        "Execute validation studies.",
    )
    if "selected_sample_dataset" in st.session_state:
        st.info(f"Selected sample dataset: {st.session_state.selected_sample_dataset}")
    st.subheader("Study Type")
    study_type = st.selectbox("Select validation study type", study_names, index=0)
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


def resolve_execution_context() -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Return the selected program and analyte for generated work execution."""

    projects = [normalize_program(project) for project in st.session_state.validation_projects]
    selected_program = str(st.session_state.get("execution_program") or (projects[0]["Program Name"] if projects else ""))
    if selected_program not in {str(project["Program Name"]) for project in projects} and projects:
        selected_program = str(projects[0]["Program Name"])
    program = next((project for project in projects if str(project["Program Name"]) == selected_program), None)
    if not program:
        return None, None
    analytes = program_analytes(program)
    selected_analyte = str(st.session_state.get("execution_analyte") or (analytes[0]["Analyte"] if analytes else ""))
    analyte = next((item for item in analytes if str(item.get("Analyte")) == selected_analyte), None)
    return program, analyte


def render_validation_workspace_landing() -> None:
    """Render the execution landing page for generated validation work."""

    render_page_header(
        "Execution",
        "Execute validation studies.",
        kicker="",
    )
    projects = [normalize_program(project) for project in st.session_state.validation_projects]
    if not projects:
        st.info("Create a validation program before starting validation work.")
        return
    selected_program = str(st.session_state.get("execution_program") or projects[0]["Program Name"])
    program_names = [str(project["Program Name"]) for project in projects]
    if selected_program not in program_names:
        selected_program = program_names[0]
    if len(program_names) > 1:
        selected_program = st.selectbox(
            "Validation Program",
            program_names,
            index=program_names.index(selected_program),
            key="workspace_program_selector",
        )
    else:
        st.caption("Validation Program")
        st.markdown(f"**{selected_program}**")
    st.session_state.execution_program = selected_program
    program = next(project for project in projects if str(project["Program Name"]) == selected_program)

    analytes = program_analytes(program)
    if not analytes:
        st.info("No analytes have been defined for this validation program.")
        return
    queue_rows = []
    for analyte in analytes:
        metrics = analyte_execution_metrics(program, analyte)
        queue_rows.append(
            {
                "Analyte": str(analyte.get("Analyte", "")),
                "Analyzer": str(analyte.get("Assigned Analyzer") or "Analyzer not assigned"),
                "Progress": execution_progress_text(metrics),
                "Status": execution_landing_status(analyte, metrics),
                "Metrics": metrics,
                "Analyte Record": analyte,
            }
        )
    total_required = sum(int(item["Metrics"].get("Required", 0)) for item in queue_rows)
    total_completed = sum(int(item["Metrics"].get("Completed", 0)) for item in queue_rows)
    completed_analytes = sum(1 for item in queue_rows if item["Status"] == "Completed")
    total_percent = min(100, max(0, round((total_completed / total_required) * 100))) if total_required else 0
    st.markdown(
        f"""
        <div class="svap-execution-program-progress">
          <div class="svap-execution-program-progress-row">
            <span class="svap-execution-program-progress-value">{total_completed} of {total_required} studies completed</span>
          </div>
          <div class="svap-progress-bar-row">
            <div class="svap-execution-program-progress-bar">
              <div class="svap-execution-program-progress-fill" style="width: {total_percent}%;"></div>
            </div>
            <span class="svap-execution-program-progress-percent">{total_percent}%</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    status_order = {
        "In Progress": 0,
        "Not Started": 1,
        "Completed": 2,
    }
    sorted_items = sorted(queue_rows, key=lambda item: (status_order.get(str(item["Status"]), 9), str(item["Analyte"])))

    if completed_analytes == len(queue_rows) and queue_rows:
        st.markdown("No analytes require scientific execution.")
        st.caption("All execution work has been completed.")
        return

    visible_items = sorted_items
    rows_html = [
        '<div class="svap-execution-directory">'
        '<div class="svap-execution-directory-header">'
        "<div>Analyte</div>"
        "<div>Progress</div>"
        "<div></div>"
        "</div>"
    ]
    for item in visible_items:
        status = str(item["Status"])
        metrics = dict(item["Metrics"])
        required = max(0, int(metrics.get("Required", 0)))
        completed = max(0, int(metrics.get("Completed", 0)))
        percent = 0 if required <= 0 else min(100, max(0, round((completed / required) * 100)))
        if status == "Completed":
            status_label = "Completed"
            status_class = "completed"
            fill_class = "completed"
        elif status == "Not Started":
            status_label = "Not Started"
            status_class = "start"
            fill_class = "start"
        else:
            status_label = "In Progress"
            status_class = "progress"
            fill_class = "progress"
        analyte_name = str(item["Analyte"])
        row_href = (
            f"?page=Execution&execution_open_program={quote(selected_program)}"
            f"&execution_open_analyte={quote(analyte_name)}"
        )
        row_href = escape(row_href, quote=True)
        rows_html.append(
            f'<a class="svap-execution-directory-row" href="{row_href}" target="_self">'
            "<div>"
            f'<div class="svap-execution-analyte">{escape(analyte_name)}</div>'
            f'<div class="svap-execution-artifact-subtext">{escape(str(item["Analyzer"]))}</div>'
            "</div>"
            '<div class="svap-execution-progress">'
            '<div class="svap-execution-progress-line">'
            f'<span class="svap-execution-progress-count">{completed}/{required} studies</span>'
            f'<span class="svap-execution-status-text svap-execution-status-text-{status_class}">{escape(status_label)}</span>'
            "</div>"
            '<div class="svap-execution-progress-bar">'
            f'<div class="svap-execution-progress-fill svap-execution-progress-fill-{fill_class}" style="width: {percent}%;"></div>'
            "</div>"
            "</div>"
            '<div class="svap-execution-chevron">›</div>'
            "</a>"
        )
    rows_html.append("</div>")
    st.markdown("\n".join(rows_html), unsafe_allow_html=True)


def render_generic_study_workstream_tab(program: dict[str, object], analyte: dict[str, object], study_type: str) -> None:
    """Render a generated study tab as an in-place workspace view."""

    render_implemented_study_workspace(program, analyte, study_type)


def render_execution_study_progress(program: dict[str, object], analyte: dict[str, object], required_studies: list[str]) -> None:
    """Render compact study-level execution status."""

    st.subheader("Study Progress")
    header = st.columns([2, 1])
    header[0].markdown("**Study**")
    header[1].markdown("**Status**")
    for study in required_studies:
        status = normalize_lifecycle_status(execution_status(program, analyte, study))
        label = execution_status_label(status)
        row = st.columns([2, 1])
        row[0].write(study)
        row[1].write(label)


def render_analyte_validation_workspace() -> None:
    """Render an analyte-centered validation workspace with a workflow navigator."""

    program, analyte = resolve_execution_context()
    if not program or not analyte:
        st.info("Select a validation program and analyte before opening a workstream.")
        return
    program_name = str(program["Program Name"])
    analyte_name = str(analyte["Analyte"])
    required_studies = list(analyte.get("Required Studies") or program_required_studies(program))
    analyzer = str(analyte.get("Assigned Analyzer") or "Analyzer not assigned")
    reagent = str(analyte.get("Assigned Reagent") or "Reagent not assigned")
    suggested_study = next_execution_study(program, analyte)
    selected_study_key = f"execution_selected_study_{program_name}_{analyte_name}"
    selected_study = str(st.session_state.get(selected_study_key) or "")
    if selected_study not in required_studies:
        selected_study = suggested_study if suggested_study in required_studies else required_studies[0]
        st.session_state[selected_study_key] = selected_study

    top_cols = st.columns([0.72, 5.28])
    with top_cols[0]:
        if st.button("Back to Queue", key=f"execution_back_to_queue_{program_name}_{analyte_name}", use_container_width=True):
            st.session_state.execution_mode = ""
            st.rerun()
    with top_cols[1]:
        st.markdown(
            f"""
            <div class="svap-execution-study-header">
              <div class="svap-execution-study-title">{escape(analyte_name)}</div>
              <div class="svap-execution-study-program">{escape(program_name)}</div>
              <div class="svap-execution-context-grid">
                <div>
                  <div class="svap-execution-meta-label">Validation Scope</div>
                  <div class="svap-execution-scope-value">{escape(specimen_context_text(program))}</div>
                </div>
                <div>
                  <div class="svap-execution-meta-label">Analyzer</div>
                  <div class="svap-execution-meta-value">{escape(analyzer)}</div>
                </div>
                <div>
                  <div class="svap-execution-meta-label">Reagent</div>
                  <div class="svap-execution-meta-value">{escape(reagent)}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="svap-study-progress-label">Study Progress</div>', unsafe_allow_html=True)
    timeline_columns = st.columns([1 for _ in required_studies])
    for index, (column, study) in enumerate(zip(timeline_columns, required_studies)):
        if study == selected_study:
            timeline_label = study
            timeline_state = "active"
        elif study_execution_has_completed_analysis(program_name, analyte_name, study):
            timeline_label = f"✓ {study}"
            timeline_state = "complete"
        else:
            timeline_label = f"○ {study}"
            timeline_state = "pending"
        with column:
            if st.button(
                timeline_label,
                key=f"study_nav_{timeline_state}_{program_name}_{analyte_name}_{index}_{study}",
                use_container_width=True,
            ):
                st.session_state[selected_study_key] = study
                st.rerun()

    st.markdown('<div class="svap-definition-section"></div>', unsafe_allow_html=True)
    if selected_study == "Method Comparison":
        record = find_study_execution_record(program_name, analyte_name, selected_study) or {
            "Status": execution_status(program, analyte, selected_study),
            "Last Updated": "",
        }
        render_method_comparison_execution_workspace(program, analyte, selected_study, record)
    else:
        render_generic_study_workstream_tab(program, analyte, selected_study)


def build_execution_study_pdf(
    title: str,
    metadata: dict[str, object],
    summary_table: pd.DataFrame,
    criteria_table: pd.DataFrame,
    decision: str,
    interpretation: str,
    data_quality: dict[str, object] | None = None,
) -> bytes:
    """Build a compact PDF report for execution workspace study outputs."""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 7, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 7, f"Decision: {decision}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Study Metadata", ln=True)
    pdf.set_font("Arial", "", 9)
    for key, value in metadata.items():
        pdf.multi_cell(0, 5, f"{key}: {value or 'Not specified'}")
    pdf.ln(3)
    if data_quality:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Data Quality Assessment", ln=True)
        pdf.set_font("Arial", "", 9)
        quality_fields = [
            "Sample Count",
            "Missing Values",
            "Duplicate IDs",
            "Outlier Count",
            "Range Violations",
            "Invalid Numeric Values",
            "Required Columns Detected",
            "Data Quality Status",
        ]
        for field in quality_fields:
            pdf.multi_cell(0, 5, f"{field}: {data_quality.get(field, 'Not assessed')}")
        warnings = data_quality.get("Warnings") or []
        warning_text = "; ".join(str(warning) for warning in warnings) if warnings else "No Critical Issues Detected"
        pdf.multi_cell(0, 5, f"Warnings: {warning_text}")
        pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Summary Statistics", ln=True)
    pdf.set_font("Arial", "", 8)
    for _, row in summary_table.iterrows():
        pdf.multi_cell(0, 5, " | ".join(f"{column}: {row[column]}" for column in summary_table.columns))
    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Acceptance Criteria", ln=True)
    pdf.set_font("Arial", "", 8)
    for _, row in criteria_table.iterrows():
        pdf.multi_cell(0, 5, " | ".join(f"{column}: {row[column]}" for column in criteria_table.columns))
    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Interpretation", ln=True)
    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(0, 5, interpretation)
    output = pdf.output(dest="S")
    if isinstance(output, bytes):
        return output
    if isinstance(output, bytearray):
        return bytes(output)
    return str(output).encode("latin-1", errors="replace")


def method_comparison_data_quality(
    data: pd.DataFrame,
    reference_column: str | None = None,
    candidate_column: str | None = None,
    sample_id_column: str | None = None,
) -> dict[str, object]:
    """Assess method-comparison data quality before analysis."""

    required_columns = [column for column in [reference_column, candidate_column] if column]
    missing_values = int(data[required_columns].isna().sum().sum()) if required_columns else int(data.isna().sum().sum())
    duplicate_ids = int(data[sample_id_column].duplicated().sum()) if sample_id_column and sample_id_column in data.columns else 0
    outlier_count = 0
    range_violations = 0
    if reference_column and candidate_column and reference_column in data.columns and candidate_column in data.columns:
        numeric = data[[reference_column, candidate_column]].apply(pd.to_numeric, errors="coerce")
        range_violations = int((numeric <= 0).sum().sum())
        for column in [reference_column, candidate_column]:
            series = numeric[column][numeric[column] > 0].dropna()
            if len(series) >= 4:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    outlier_count += int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
    warnings: list[str] = []
    if missing_values:
        warnings.append(f"{missing_values} missing required values detected.")
    if duplicate_ids:
        warnings.append(f"{duplicate_ids} duplicate sample IDs detected.")
    if range_violations:
        warnings.append(f"{range_violations} zero or negative result values detected.")
    invalid_numeric_values = 0
    if reference_column and candidate_column and reference_column in data.columns and candidate_column in data.columns:
        raw_required = data[[reference_column, candidate_column]]
        numeric_required = raw_required.apply(pd.to_numeric, errors="coerce")
        invalid_numeric_values = int((numeric_required.isna() & raw_required.notna()).sum().sum())
    if invalid_numeric_values:
        warnings.append(f"{invalid_numeric_values} invalid numeric values detected.")
    if outlier_count:
        warnings.append(f"{outlier_count} potential outliers detected by IQR screening.")
    return {
        "Sample Count": int(len(data)),
        "Missing Values": missing_values,
        "Duplicate IDs": duplicate_ids,
        "Outlier Count": outlier_count,
        "Range Violations": range_violations,
        "Invalid Numeric Values": invalid_numeric_values,
        "Warnings": warnings,
        "Required Columns Detected": len(required_columns),
        "Columns Detected": int(len(data.columns)),
    }


def build_method_comparison_quality_issues(
    data: pd.DataFrame,
    reference_column: str,
    candidate_column: str,
    sample_id_column: str | None,
) -> list[dict[str, object]]:
    """Return traceable method-comparison data quality issues by affected record."""

    issues: list[dict[str, object]] = []

    def sample_value(row_index: int) -> str:
        if sample_id_column and sample_id_column in data.columns:
            value = data.iloc[row_index][sample_id_column]
            if pd.notna(value):
                return str(value)
        return f"Row {row_index + 1}"

    def add_issue(
        row_index: int,
        column: str,
        value: object,
        issue: str,
        severity: str,
        suggested_action: str,
        issue_type: str,
    ) -> None:
        sample = sample_value(row_index)
        issues.append(
            {
                "Issue ID": f"{issue_type}|{row_index}|{column}|{sample}",
                "Sample ID": sample,
                "Row": row_index + 1,
                "Column": column,
                "Value": "" if pd.isna(value) else value,
                "Issue": issue,
                "Severity": severity,
                "Suggested Action": suggested_action,
                "Detected": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Detection Method": "Detected Automatically",
            }
        )

    required_columns = [reference_column, candidate_column]
    for column in required_columns:
        if column not in data.columns:
            continue
        raw_series = data[column]
        numeric_series = pd.to_numeric(raw_series, errors="coerce")
        missing_mask = raw_series.isna()
        invalid_mask = numeric_series.isna() & raw_series.notna()
        range_mask = numeric_series.notna() & (numeric_series <= 0)
        for row_index in raw_series[missing_mask].index:
            add_issue(
                int(row_index),
                column,
                raw_series.loc[row_index],
                "Required result value is missing",
                "Critical",
                "Populate the missing measurement or remove the paired sample from analysis with scientific justification.",
                "missing",
            )
        for row_index in raw_series[invalid_mask].index:
            add_issue(
                int(row_index),
                column,
                raw_series.loc[row_index],
                "Result value is not numeric",
                "Critical",
                "Review the source instrument data and correct the imported value if a transcription or formatting error occurred.",
                "invalid_numeric",
            )
        for row_index in raw_series[range_mask].index:
            value = numeric_series.loc[row_index]
            issue = "Negative result value" if value < 0 else "Result must be greater than zero"
            add_issue(
                int(row_index),
                column,
                raw_series.loc[row_index],
                issue,
                "Critical",
                "Review source instrument data, confirm whether the result is valid, correct the imported value if needed, or exclude the sample with scientific justification.",
                "range",
            )
        clean_series = numeric_series[numeric_series > 0].dropna()
        if len(clean_series) >= 4:
            q1 = clean_series.quantile(0.25)
            q3 = clean_series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outlier_mask = (numeric_series > 0) & ((numeric_series < q1 - 1.5 * iqr) | (numeric_series > q3 + 1.5 * iqr))
                for row_index in raw_series[outlier_mask.fillna(False)].index:
                    add_issue(
                        int(row_index),
                        column,
                        raw_series.loc[row_index],
                        "Potential statistical outlier",
                        "Warning",
                        "Review the specimen, source result, and analytical context before final submission.",
                        "outlier",
                    )

    if sample_id_column and sample_id_column in data.columns:
        duplicate_mask = data[sample_id_column].duplicated(keep=False) & data[sample_id_column].notna()
        for row_index in data[duplicate_mask].index:
            add_issue(
                int(row_index),
                sample_id_column,
                data.loc[row_index, sample_id_column],
                "Duplicate sample ID",
                "Warning",
                "Verify specimen identity and remove or reconcile duplicate entries before final submission.",
                "duplicate_id",
            )

    return issues


def render_method_lifecycle_tracker(status: str) -> None:
    """Render scientific lifecycle progress for method-comparison execution."""

    normalized = normalize_lifecycle_status(status)
    steps = [
        ("Data Upload", {"Data Uploaded", "Analysis Complete", "Ready For Review", "Scientific Review", "Submitted for Review", "Under Review", "Approved", "Report Generated", "Locked", "Archived"}),
        ("Analysis", {"Analysis Complete", "Ready For Review", "Scientific Review", "Submitted for Review", "Under Review", "Approved", "Report Generated", "Locked", "Archived"}),
        ("Scientific Review", {"Ready For Review", "Scientific Review", "Submitted for Review", "Under Review", "Approved", "Report Generated", "Locked", "Archived"}),
        ("Approval", {"Approved", "Report Generated", "Locked", "Archived"}),
        ("Report", {"Report Generated", "Locked", "Archived"}),
    ]
    st.subheader("Study Lifecycle")
    cols = st.columns(len(steps))
    current_index = 0
    for index, (_, completed_states) in enumerate(steps):
        if normalized in completed_states:
            current_index = min(index + 1, len(steps) - 1)
    if normalized in {"Draft", "Not Started"}:
        current_index = 0
    for index, (label, completed_states) in enumerate(steps):
        marker = "✓" if normalized in completed_states else "●" if index == current_index else "○"
        with cols[index]:
            st.markdown(f"**{marker} {label}**")


def method_comparison_workflow_message(
    status: str,
    sample_requirement: int,
    uploaded_count: int = 0,
    has_result: bool = False,
) -> dict[str, str]:
    """Return workflow-oriented empty-state messaging for method comparison."""

    normalized = normalize_lifecycle_status(status)
    remaining_count = max(sample_requirement - uploaded_count, 0)
    sample_progress = f"{sample_requirement} Paired Samples Required\nUploaded: {uploaded_count}\nRemaining: {remaining_count}"
    if normalized in {"Draft", "Not Started"}:
        return {
            "Study Status": "Awaiting Data Upload",
            "Next Step": "Upload paired serum and microtainer dataset.",
            "Sample Requirement": sample_progress,
            "Analysis Status": "Waiting For Data",
        }
    if normalized == "Data Uploaded":
        return {
            "Study Status": "Dataset Uploaded",
            "Next Step": "Validate dataset quality and run analysis.",
            "Sample Requirement": sample_progress,
            "Analysis Status": "Analysis Complete" if has_result else "Ready To Run",
        }
    if normalized == "Analysis Complete":
        return {
            "Study Status": "Analysis Complete",
            "Next Step": "Review plots, interpretation, and submit for review.",
            "Sample Requirement": sample_progress,
            "Analysis Status": "Analysis Complete",
        }
    if normalized in {"Ready For Review", "Scientific Review", "Submitted for Review", "Under Review"}:
        return {
            "Study Status": "Scientific Review Pending",
            "Next Step": "Reviewer decision required.",
            "Sample Requirement": sample_progress,
            "Analysis Status": "Ready For Review",
        }
    if normalized == "Approved":
        return {
            "Study Status": "Approved",
            "Next Step": "Generate final study report.",
            "Sample Requirement": sample_progress,
            "Analysis Status": "Ready For Review",
        }
    return {
        "Study Status": normalized,
        "Next Step": "View or export completed study record.",
        "Sample Requirement": sample_progress,
        "Analysis Status": "Analysis Complete",
    }


def detect_required_method_columns(data: pd.DataFrame) -> dict[str, str | None]:
    """Detect common method-comparison columns by name."""

    lowered = {str(column).strip().lower(): str(column) for column in data.columns}
    sample_id = detect_sample_id_column(data)
    reference = next(
        (original for key, original in lowered.items() if any(term in key for term in ["serum", "reference", "venous"])),
        None,
    )
    candidate = next(
        (original for key, original in lowered.items() if any(term in key for term in ["microtainer", "candidate", "capillary"])),
        None,
    )
    return {
        "Sample ID": sample_id,
        "Reference Result": reference,
        "Candidate Result": candidate,
    }


def add_identity_line_to_method_plot(figure, analyzed_data: pd.DataFrame):
    """Add an identity line to a method comparison scatter plot."""

    min_value = min(float(analyzed_data["Reference"].min()), float(analyzed_data["Candidate"].min()))
    max_value = max(float(analyzed_data["Reference"].max()), float(analyzed_data["Candidate"].max()))
    figure.add_trace(
        {
            "type": "scatter",
            "x": [min_value, max_value],
            "y": [min_value, max_value],
            "mode": "lines",
            "name": "Identity line",
            "line": {"color": "#64748b", "width": 2, "dash": "dash"},
        }
    )
    return figure


STUDY_WORKSPACE_FRAMEWORK = {
    "Method Comparison": {
        "criteria": ("R²", "Bias", "Agreement Rate"),
        "metrics": ("R²", "Mean Bias", "Agreement Rate"),
        "columns": ("Sample ID", "Reference Result", "Candidate Result", "Collection Date"),
    },
    "Precision": {
        "criteria": ("CV%",),
        "metrics": ("Within-run CV%", "Between-run CV%", "Total CV%"),
        "columns": ("Run ID", "Replicate", "Sample Level", "Result"),
    },
    "Accuracy": {
        "criteria": ("Bias", "Recovery"),
        "metrics": ("Mean Bias", "Recovery", "Worst Level"),
        "columns": ("Sample ID", "Level", "Expected Result", "Observed Result"),
    },
    "Linearity": {
        "criteria": ("Recovery", "Linearity Deviation"),
        "metrics": ("Regression", "Recovery", "Maximum Deviation"),
        "columns": ("Level", "Expected Value", "Observed Value"),
    },
    "Stability": {
        "criteria": ("Timepoint Drift", "Recovery"),
        "metrics": ("Percent Recovery", "Drift", "Worst Timepoint"),
        "columns": ("Sample ID", "Timepoint", "Result"),
    },
    "Detection Capability": {
        "criteria": ("LoB", "LoD", "LoQ"),
        "metrics": ("LoB", "LoD", "LoQ"),
        "columns": ("Sample ID", "Sample Type", "Concentration Level", "Observed Result"),
    },
}


def render_study_workspace_header(analyte_name: str, study_type: str) -> None:
    """Render the shared study workspace header."""

    st.markdown(
        f"""
        <div class="svap-workspace-header">
          <div class="svap-workspace-title">{escape(analyte_name)}</div>
          <div class="svap-workspace-meta">{escape(study_type)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _column_index(columns: list[str], preferred: str | None, fallback: int = 0) -> int:
    """Return a stable selectbox index for a preferred column."""

    if preferred in columns:
        return columns.index(preferred)
    return min(fallback, max(len(columns) - 1, 0))


def _optional_column_select(label: str, columns: list[str], preferred: str | None, key: str) -> str | None:
    """Render a reusable optional column selector."""

    options = ["None"] + columns
    index = options.index(preferred) if preferred in options else 0
    selected = st.selectbox(label, options, index=index, key=key)
    return None if selected == "None" else selected


def _required_column_select(label: str, columns: list[str], preferred: str | None, key: str) -> str | None:
    """Render a required column selector that can expose unmapped required fields."""

    if preferred in columns:
        selected = st.selectbox(label, columns, index=columns.index(preferred), key=key)
        return selected
    options = ["Select column"] + columns
    selected = st.selectbox(label, options, index=0, key=key)
    return None if selected == "Select column" else selected


def _build_quality_summary(data: pd.DataFrame, selected_columns: list[str], sample_id_column: str | None = None) -> dict[str, int | str]:
    """Build a compact cross-study data quality summary."""

    available = [column for column in selected_columns if column in data.columns]
    quality_data = data[available].copy() if available else data.copy()
    missing_values = int(quality_data.isna().sum().sum())
    duplicate_ids = int(quality_data[sample_id_column].duplicated().sum()) if sample_id_column and sample_id_column in quality_data.columns else 0
    return {
        "Sample Count": int(len(data)),
        "Missing Values": missing_values,
        "Duplicate IDs": duplicate_ids,
        "Required Columns Found": len(available),
        "Status": "PASS" if missing_values == 0 else "FAIL",
    }


def _mapping_failures(required_mapping: dict[str, str | None]) -> list[str]:
    """Return required scientific fields that are not mapped to uploaded columns."""

    return [label for label, column in required_mapping.items() if not column]


def _display_value(value: object, decimals: int = 2, empty_label: str = "N/A") -> str:
    """Return a user-safe display value without NaN/inf leakage."""

    if value is None or pd.isna(value):
        return empty_label
    if isinstance(value, (int, float)):
        if pd.isna(value) or value in (float("inf"), float("-inf")):
            return empty_label
        return f"{float(value):.{decimals}f}"
    text = str(value)
    return empty_label if text.lower() in {"nan", "none", "inf", "-inf"} else text


def _assay_units(analyte_name: str, analyzed_data: pd.DataFrame | None = None) -> str:
    """Return display units from study data or a conservative assay default."""

    if analyzed_data is not None and "Units" in analyzed_data.columns:
        units = analyzed_data["Units"].dropna().astype(str).str.strip()
        units = units[units != ""]
        if not units.empty:
            return str(units.iloc[0])
    normalized = analyte_name.strip().lower()
    if normalized == "hba1c":
        return "%"
    return "units"


def _with_units(value: object, units: str, decimals: int = 3) -> str:
    """Format a numeric value with assay units."""

    return f"{_display_value(value, decimals=decimals)} {units}"


def _sanitize_table(table: pd.DataFrame) -> pd.DataFrame:
    """Replace non-displayable values in a table with scientist-facing text."""

    if table.empty:
        return table
    return table.replace([float("inf"), float("-inf")], pd.NA).fillna("N/A")


def dataframe_content_hash(data: pd.DataFrame) -> str:
    """Return a deterministic hash for a dataframe used as scientific input."""

    return hashlib.sha256(data.to_csv(index=False).encode("utf-8")).hexdigest()


def inline_status_line(text: str) -> None:
    """Render a compact success status without a large alert banner."""

    st.markdown(f"**✓ {escape(text)}**")


def render_protocol_summary(status: str, title: str, subtext: str = "") -> None:
    """Render a contained protocol status summary for execution workspaces."""

    status_html = escape(status)
    subtext_html = f'<div class="svap-protocol-subtext">{escape(subtext)}</div>' if subtext else ""
    st.markdown(
        f"""
        <div class="svap-protocol-summary svap-protocol-summary-success">
          <div class="svap-protocol-eyebrow">✓ {status_html}</div>
          <div class="svap-protocol-title">{escape(title)}</div>
          {subtext_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(metrics: dict[str, str]) -> None:
    """Render compact scientific key metrics with a consistent visual rhythm."""

    st.dataframe(
        pd.DataFrame([metrics]),
        width="stretch",
        hide_index=True,
    )


def render_scientific_interpretation(text: str) -> None:
    """Render a concise scientific conclusion as a report-like panel."""

    lines = [part.strip() for part in text.split("\n\n") if part.strip()]
    paragraphs = "".join(f"<p>{escape(line)}</p>" for line in lines)
    st.markdown(
        f"""
        <div class="svap-interpretation-panel">
          {paragraphs}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_plain_scientific_interpretation(text: str) -> None:
    """Render scientific interpretation as manuscript-style narrative text."""

    lines = [part.strip() for part in text.split("\n\n") if part.strip()]
    paragraphs = "".join(f"<p>{escape(line)}</p>" for line in lines)
    st.markdown(
        f"""
        <div class="svap-study-interpretation">
          {paragraphs}
        </div>
        """,
        unsafe_allow_html=True,
    )


def summarize_criteria_status(criteria_display: pd.DataFrame) -> str:
    """Return a concise acceptance criteria summary for study results."""

    if criteria_display.empty or "Status" not in criteria_display.columns:
        return "Acceptance criteria evaluated"
    statuses = criteria_display["Status"].astype(str).str.upper()
    passed = int(statuses.str.contains("PASS").sum())
    total = int(len(criteria_display))
    if total == 0:
        return "Acceptance criteria evaluated"
    if passed == total:
        return f"{passed} / {total} acceptance criteria satisfied"
    failed = total - passed
    return f"{passed} / {total} acceptance criteria satisfied · {failed} requiring review"


def normalize_criteria_display(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize criteria columns for scientific review/readout sections."""

    if table.empty:
        return table
    display = table.copy()
    rename_map = {
        "Observed Value": "Observed",
        "Acceptance Limit": "Required",
    }
    if "Status" not in display.columns and "Pass/Fail Status" in display.columns:
        rename_map["Pass/Fail Status"] = "Status"
    display = display.rename(columns=rename_map)
    display = display.drop(columns=["Acceptance Source", "Pass/Fail Status"], errors="ignore")
    return display


def _format_method_comparison_criteria_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return Method Comparison criteria with professional numeric formatting."""

    if table.empty:
        return table
    criteria_columns = [
        column
        for column in ["Criterion", "Observed Value", "Acceptance Limit", "Status", "Pass/Fail Status"]
        if column in table.columns
    ]
    display = table[criteria_columns].copy() if criteria_columns else table.copy()
    display = display.rename(
        columns={
            "Observed Value": "Observed",
            "Acceptance Limit": "Required",
            "Pass/Fail Status": "Status",
        }
    )
    if "Observed" in display.columns:
        display["Observed"] = display["Observed"].apply(lambda value: _display_value(value, decimals=4))
    return display


def build_method_comparison_analysis_summary(summary: dict[str, float]) -> pd.DataFrame:
    """Return the method-comparison detailed summary scientists need first."""

    rows = [
        ("Samples", summary.get("N"), 0),
        ("Mean Reference", summary.get("Mean Reference"), 3),
        ("Mean Candidate", summary.get("Mean Candidate"), 3),
        ("Mean Bias", summary.get("Mean Difference"), 3),
        ("SD Bias", summary.get("SD Difference"), 3),
        ("Lower LOA", summary.get("Lower Limit of Agreement"), 3),
        ("Upper LOA", summary.get("Upper Limit of Agreement"), 3),
        ("Correlation", summary.get("Correlation r"), 4),
        ("R²", summary.get("R²"), 4),
    ]
    return pd.DataFrame(
        [
            {"Metric": metric, "Value": _display_value(value, decimals=decimals)}
            for metric, value, decimals in rows
        ]
    )


def build_method_comparison_flagged_samples(analyzed_data: pd.DataFrame, percent_bias_limit: float) -> pd.DataFrame:
    """Return only method-comparison samples outside sample-level bias criteria."""

    if analyzed_data.empty or "Percent Bias" not in analyzed_data.columns:
        return pd.DataFrame()
    flagged = analyzed_data.dropna(subset=["Percent Bias"]).copy()
    flagged = flagged[flagged["Percent Bias"].abs() > percent_bias_limit].copy()
    if flagged.empty:
        return pd.DataFrame()
    flagged["Reason"] = f"Outside ±{percent_bias_limit:g}% bias limit"
    columns = ["Reference", "Candidate", "Difference", "Percent Bias", "Reason"]
    if "Sample ID" in flagged.columns:
        columns.insert(0, "Sample ID")
    display = flagged[columns].copy()
    for column in ["Reference", "Candidate", "Difference"]:
        if column in display.columns:
            display[column] = display[column].apply(lambda value: _display_value(value, decimals=3))
    if "Percent Bias" in display.columns:
        display["Percent Bias"] = display["Percent Bias"].apply(lambda value: f"{_display_value(value, decimals=2)}%")
    return display.reset_index(drop=True)


def _compact_criteria_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return a reviewer-readable criteria table without statistical detail sprawl."""

    if table.empty:
        return table
    status_column = "Status" if "Status" in table.columns else "Pass/Fail Status" if "Pass/Fail Status" in table.columns else None
    if (
        status_column
        and {"Level", "Expected Result", "Mean Observed Result", "Recovery %", "Percent Bias"}.issubset(table.columns)
    ):
        compact = table[
            [
                "Level",
                "Expected Result",
                "Mean Observed Result",
                "Recovery %",
                "Percent Bias",
                status_column,
            ]
        ].copy()
        compact = compact.rename(
            columns={
                "Expected Result": "Expected",
                "Mean Observed Result": "Observed",
                "Percent Bias": "Bias %",
                status_column: "Status",
            }
        )
        return compact
    preferred = [
        column
        for column in ["Criterion", "Observed Value", "Observed", "Acceptance Limit", "Acceptance Source", "Status", "Pass/Fail Status"]
        if column in table.columns
    ]
    return table[preferred].copy() if preferred else table


def _detect_column_by_alias(columns: list[str], aliases: list[str]) -> str | None:
    """Detect a column from common scientific aliases."""

    normalized = {str(column).strip().lower().replace("_", " "): column for column in columns}
    for alias in aliases:
        alias_key = alias.lower().replace("_", " ")
        if alias_key in normalized:
            return str(normalized[alias_key])
    for alias in aliases:
        alias_key = alias.lower().replace("_", " ")
        for key, column in normalized.items():
            if alias_key in key:
                return str(column)
    return None


def _precision_dataset_structure_issues(data: pd.DataFrame) -> list[str]:
    """Return precision-specific dataset structure failures."""

    columns = list(data.columns)
    numeric_columns = get_numeric_columns(data)
    issues = []
    if _detect_column_by_alias(columns, ["Level", "Sample Level", "Control Level"]) is None:
        issues.append("Missing precision level column.")
    result_column = _detect_column_by_alias(columns, ["Result", "Observed Result", "Measurement"])
    if result_column is None or result_column not in numeric_columns:
        issues.append("Missing numeric precision result column.")
    if _detect_column_by_alias(columns, ["Replicate", "Replicate ID"]) is None:
        issues.append("Missing replicate column.")
    if _detect_column_by_alias(columns, ["Run", "Run ID"]) is None:
        issues.append("Missing run column.")
    return issues


def _checks_to_criteria_table(criteria_result: dict[str, object]) -> pd.DataFrame:
    """Convert analysis-engine checks into a consistent criteria table."""

    rows = []
    for check in list(criteria_result.get("checks", [])):
        status = "PASS" if check.get("Met") else "BORDERLINE" if check.get("Borderline") else "FAIL"
        acceptance_limit = str(check.get("Acceptance Limit", ""))
        observed = check.get("Observed", "")
        observed_value = _display_value(observed)
        if "%" in acceptance_limit and observed_value not in {"N/A", "Not Calculated", "Insufficient Data"}:
            observed_value = f"{observed_value}%"
        rows.append(
            {
                "Criterion": check.get("Criterion", ""),
                "Observed Value": observed_value,
                "Acceptance Limit": acceptance_limit,
                **({"Acceptance Source": check.get("Acceptance Source")} if check.get("Acceptance Source") else {}),
                "Status": status,
            }
        )
    return pd.DataFrame(rows)


def _render_execution_results(
    *,
    program: dict[str, object],
    analyte: dict[str, object],
    study_type: str,
    decision: str,
    criteria_table: pd.DataFrame,
    key_results: dict[str, str],
    interpretation: str,
    figures: dict[str, object],
    supporting_tables: dict[str, pd.DataFrame],
    failure_type: str | None = None,
    failure_reason: str | None = None,
    show_review_action: bool = True,
) -> None:
    """Render the shared Results section for all study workspaces."""

    program_name = str(program["Program Name"])
    analyte_name = str(analyte["Analyte"])

    st.subheader("Results")
    finding = review_key_finding(study_type, decision)
    decision_class = "svap-study-result-status"
    if decision == "FAIL":
        decision_class = "svap-study-result-status svap-study-result-status-fail"
    elif decision == "INCOMPLETE":
        decision_class = "svap-study-result-status svap-study-result-status-incomplete"
    criteria_display = normalize_criteria_display(_sanitize_table(_compact_criteria_table(criteria_table)))
    criteria_summary = summarize_criteria_status(criteria_display)
    st.markdown(
        f"""
        <div class="svap-study-result">
          <div class="{decision_class}">{escape(decision)}</div>
          <div class="svap-study-result-finding">{escape(finding)}</div>
          <div class="svap-study-result-meta">{escape(criteria_summary)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if failure_type:
        st.caption(failure_type)

    st.markdown("**Key Metrics**")
    render_metric_grid({label: _display_value(value) for label, value in key_results.items()})

    st.markdown("**Acceptance Criteria**")
    render_badged_criteria_table(criteria_display)

    chart_base = f"{program_name}_{analyte_name}_{study_type}".replace(" ", "_").replace("/", "_").lower()
    if figures:
        st.subheader("Scientific Evidence")
        for index, (title, figure) in enumerate(figures.items()):
            st.markdown(f'<div class="svap-figure-heading">{escape(title)}</div>', unsafe_allow_html=True)
            st.plotly_chart(
                figure,
                width="stretch",
                key=f"{chart_base}_{index}",
                config={"displayModeBar": False, "responsive": True},
            )

    st.markdown("### Scientific Interpretation")
    render_plain_scientific_interpretation(interpretation)

    with st.expander("Detailed Results", expanded=False):
        for title, table in supporting_tables.items():
            st.markdown(f"**{title}**")
            st.dataframe(_sanitize_table(table), width="stretch", hide_index=True)

    return


def _render_execution_completion_step(
    program: dict[str, object],
    analyte: dict[str, object],
    study_type: str,
    decision: str,
    criteria_table: pd.DataFrame,
    key_results: dict[str, str],
    interpretation: str,
    figures: dict[str, object],
    supporting_tables: dict[str, pd.DataFrame],
) -> None:
    """Render the explicit final execution action for a study workspace."""

    program_name = str(program["Program Name"])
    analyte_name = str(analyte["Analyte"])
    execution_state = get_study_execution_state(program_name, analyte_name, study_type)

    def send_to_review() -> None:
        updated = ensure_study_execution_record(program, analyte_name, study_type, "Pending Review")
        updated["Submitted Date"] = date.today().isoformat()
        review_package = build_submitted_review_package(
            program=program,
            analyte=analyte,
            study_type=study_type,
            decision=decision,
            criteria_table=criteria_table,
            key_results=key_results,
            interpretation=interpretation,
            figures=figures,
            supporting_tables=supporting_tables,
        )
        persist_review_package(updated, review_package)
        persist_study_execution_value(program_name, analyte_name, study_type, "Submitted For Review", True)
        persist_study_execution_value(program_name, analyte_name, study_type, "Review Package", review_package)
        record_program_activity(
            program_name,
            f"{analyte_name} {study_type}",
            "Pending review",
            f"{study_type} execution package sent to Review.",
        )
        st.session_state.execution_mode = ""
        st.toast(f"{study_type} sent to Review.")
        st.rerun()

    def back_to_execution() -> None:
        st.session_state.execution_mode = ""
        st.rerun()

    st.subheader("Conclusion")
    if decision == "INCOMPLETE":
        st.markdown("**Next Action**")
        st.write("Upload a valid dataset and rerun analysis before completing this study.")
        return

    submitted_to_review = bool(execution_state.get("Submitted For Review")) and isinstance(execution_state.get("Review Package"), dict)
    if submitted_to_review:
        st.markdown(
            """
            <div class="svap-study-complete">
              <h3>Study Completed</h3>
              <p><strong>✓ Submitted to Review</strong></p>
              <div class="svap-study-next-label">Next Stage</div>
              <div class="svap-study-next-stage">Review</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "Open Review Queue",
            type="primary",
            key=f"back_to_execution_submitted_{program_name}_{analyte_name}_{study_type}",
            use_container_width=True,
        ):
            st.session_state.execution_mode = ""
            st.session_state.pending_page = "Review"
            st.rerun()
        return

    if execution_state.get("Execution Complete"):
        st.write("Execution is complete.")
        send_col, back_col = st.columns([1.4, 0.8])
        with send_col:
            if st.button(
                "Submit for Review",
                type="primary",
                key=f"send_to_review_{program_name}_{analyte_name}_{study_type}",
                use_container_width=True,
            ):
                send_to_review()
        with back_col:
            if st.button(
                "Return to Queue",
                key=f"back_to_execution_{program_name}_{analyte_name}_{study_type}",
                use_container_width=True,
            ):
                back_to_execution()
        return

    st.markdown("**Complete Study**")
    st.write(f"This marks {study_type} as complete within Execution.")
    st.caption("The study can then be submitted for scientific review.")
    if st.button(
        "Complete Study",
        type="primary",
        key=f"complete_execution_{program_name}_{analyte_name}_{study_type}",
        use_container_width=True,
    ):
        persist_study_execution_value(program_name, analyte_name, study_type, "Execution Complete", True)
        persist_study_execution_value(program_name, analyte_name, study_type, "Execution Completed At", datetime.now().strftime("%Y-%m-%d %H:%M"))
        persist_study_execution_value(program_name, analyte_name, study_type, "Study Version", str(execution_state.get("Study Version") or "1.0"))
        ensure_study_execution_record(program, analyte_name, study_type, "Analysis Complete")
        record_program_activity(
            program_name,
            f"{analyte_name} {study_type}",
            "Study completed",
            f"{study_type} execution completed.",
        )
        st.toast(f"{study_type} completed.")
        st.rerun()


def _render_method_comparison_scientific_results(
    *,
    program_name: str,
    analyte_name: str,
    decision: str,
    criteria_table: pd.DataFrame,
    key_results: dict[str, str],
    interpretation: str,
    figures: dict[str, object],
    supporting_tables: dict[str, pd.DataFrame],
) -> None:
    """Render Method Comparison results as a scientific workspace package."""

    st.subheader("Results")
    finding = (
        "Method comparison met predefined agreement and bias requirements."
        if decision == "PASS"
        else "Method comparison did not meet all predefined agreement and bias requirements."
    )
    decision_class = "svap-study-result-status"
    if decision == "FAIL":
        decision_class = "svap-study-result-status svap-study-result-status-fail"
    elif decision == "INCOMPLETE":
        decision_class = "svap-study-result-status svap-study-result-status-incomplete"
    criteria_display = _format_method_comparison_criteria_table(criteria_table)
    criteria_summary = summarize_criteria_status(criteria_display)
    st.markdown(
        f"""
        <div class="svap-study-result">
          <div class="{decision_class}">{escape(decision)}</div>
          <div class="svap-study-result-finding">{escape(finding)}</div>
          <div class="svap-study-result-meta">{escape(criteria_summary)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Key Metrics**")
    render_metric_grid(key_results)

    st.markdown("**Acceptance Criteria**")
    render_badged_criteria_table(_sanitize_table(criteria_display))

    chart_base = f"{program_name}_{analyte_name}_method_comparison".replace(" ", "_").replace("/", "_").lower()
    plot_interpretations = {
        "Regression Plot": "Strong linear agreement between candidate and reference measurements.",
        "Bland-Altman Plot": "Mean bias is minimal. No proportional bias observed.",
        "Residual Plot": "Residuals are randomly distributed with no systematic trend.",
    }
    st.subheader("Scientific Evidence")
    for index, (title, figure) in enumerate(figures.items()):
        st.markdown(f'<div class="svap-figure-heading">{escape(title)}</div>', unsafe_allow_html=True)
        st.plotly_chart(
            figure,
            width="stretch",
            key=f"{chart_base}_{index}",
            config={"displayModeBar": False, "responsive": True},
        )
        if title in plot_interpretations:
            st.markdown(
                f'<div class="svap-evidence-note">{escape(plot_interpretations[title])}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Scientific Interpretation")
    render_plain_scientific_interpretation(interpretation)

    with st.expander("Detailed Results", expanded=False):
        analysis_summary = supporting_tables.get("Analysis Summary", pd.DataFrame())
        flagged_samples = supporting_tables.get("Flagged Samples", pd.DataFrame())
        complete_dataset = supporting_tables.get("Complete Dataset", pd.DataFrame())
        summary_tab, flagged_tab, dataset_tab = st.tabs(["Analysis Summary", "Flagged Samples", "Complete Dataset"])
        with summary_tab:
            st.dataframe(_sanitize_table(analysis_summary), width="stretch", hide_index=True)
        with flagged_tab:
            if flagged_samples.empty:
                st.caption("No flagged samples.")
            else:
                st.dataframe(_sanitize_table(flagged_samples), width="stretch", hide_index=True)
        with dataset_tab:
            st.dataframe(_sanitize_table(complete_dataset), width="stretch", hide_index=True)


def _render_study_data_upload(
    program_name: str,
    analyte_name: str,
    study_type: str,
    sample_path: Path,
) -> pd.DataFrame | None:
    """Render the shared dataset upload pattern."""

    persisted = get_study_execution_state(program_name, analyte_name, study_type)
    st.subheader(f"{study_type} Dataset")
    expected_columns = STUDY_WORKSPACE_FRAMEWORK.get(study_type, {}).get("columns", ())
    if expected_columns:
        with st.expander("Dataset Structure", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Column": column,
                            "Requirement": "Required" if index < min(3, len(expected_columns)) else "Optional",
                        }
                        for index, column in enumerate(expected_columns)
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx", "xls"],
        key=f"study_upload_{program_name}_{analyte_name}_{study_type}",
    )
    use_sample_data = st.checkbox(
        "Load Demo Dataset",
        value=False,
        key=f"study_sample_{program_name}_{analyte_name}_{study_type}",
    )
    data = None
    source_name = str(persisted.get("Dataset Source", ""))
    if uploaded_file is not None:
        try:
            data = load_uploaded_file(uploaded_file)
            source_name = uploaded_file.name
            reset_analysis_for_new_dataset(program_name, analyte_name, study_type, source_name)
        except Exception as exc:
            st.error(f"Unable to load file: {exc}")
            return None
    elif use_sample_data:
        data = pd.read_csv(sample_path)
        source_name = sample_path.name
        reset_analysis_for_new_dataset(program_name, analyte_name, study_type, source_name)
    elif isinstance(persisted.get("Dataset"), pd.DataFrame):
        data = persisted["Dataset"].copy()
    if data is None:
        st.info("Upload a dataset or load the demo dataset to begin.")
        return None
    persist_study_execution_value(program_name, analyte_name, study_type, "Dataset", data.copy())
    if source_name:
        persist_study_execution_value(program_name, analyte_name, study_type, "Dataset Source", source_name)
    persist_study_execution_value(program_name, analyte_name, study_type, "Dataset Uploaded", True)
    st.markdown("**✓ Dataset Uploaded**")
    st.caption(f"{len(data)} samples · {len(data.columns)} columns found")
    with st.expander("View Dataset Details", expanded=False):
        if source_name:
            st.write(f"Source: {source_name}")
        st.dataframe(data.head(20), width="stretch")
    return data


def render_implemented_study_workspace(program: dict[str, object], analyte: dict[str, object], study_type: str) -> None:
    """Render implemented non-method study workspaces using the shared execution pattern."""

    program_name = str(program["Program Name"])
    analyte_name = str(analyte["Analyte"])
    sample_paths = {
        "Precision": PRECISION_SAMPLE_DATA_PATH,
        "Accuracy": ACCURACY_SAMPLE_DATA_PATH,
        "Linearity": LINEARITY_SAMPLE_DATA_PATH,
        "Stability": STABILITY_SAMPLE_DATA_PATH,
        "Detection Capability": DETECTION_SAMPLE_DATA_PATH,
    }
    data = _render_study_data_upload(program_name, analyte_name, study_type, sample_paths[study_type])
    if data is None:
        return
    persisted_state = get_study_execution_state(program_name, analyte_name, study_type)
    if persisted_state.pop("Dataset Changed", False):
        save_study_execution_state_key(study_state_key(program_name, analyte_name, study_type), persisted_state)
        ensure_study_execution_record(program, analyte_name, study_type, "Data Uploaded")
    else:
        mark_data_uploaded_if_needed(program, analyte_name, study_type)

    all_columns = list(data.columns)
    numeric_columns = get_numeric_columns(data)
    if not numeric_columns:
        st.error("At least one numeric result column is required.")
        return
    precision_completeness_issues: list[str] = []
    if study_type == "Precision":
        structure_issues = _precision_dataset_structure_issues(data)
        if structure_issues:
            persist_study_execution_value(
                program_name,
                analyte_name,
                study_type,
                "Data Quality",
                {
                    "Status": "FAIL",
                    "Failure Type": "Dataset Structure Failure",
                    "Issues": structure_issues,
                },
            )
            st.subheader("Data Quality")
            st.error("Dataset Structure Failure")
            st.write("Precision analysis requires precision-study data. Method Comparison or other paired-result datasets cannot be analyzed as Precision studies.")
            for issue in structure_issues:
                st.write(f"- {issue}")
            st.markdown("**Next Action:** Upload a valid precision dataset.")
            return
        detected_day = _detect_column_by_alias(all_columns, ["Day"])
        detected_run = _detect_column_by_alias(all_columns, ["Run", "Run ID"])
        detected_rep = _detect_column_by_alias(all_columns, ["Replicate", "Replicate ID"])
        completeness = {
            "Replicates Found": int(data[detected_rep].nunique()) if detected_rep else 0,
            "Runs Found": int(data[detected_run].nunique()) if detected_run else 0,
            "Days Found": int(data[detected_day].nunique()) if detected_day else 0,
        }
        if completeness["Replicates Found"] < 2:
            precision_completeness_issues.append("at least two replicates are required")
        if completeness["Runs Found"] < 2:
            precision_completeness_issues.append("at least two runs are required")
        if completeness["Days Found"] < 2:
            precision_completeness_issues.append("at least two days are required")

    st.subheader("Data Quality")
    required_mapping: dict[str, str | None] = {}
    with st.expander("View Column Mapping", expanded=False):
        if study_type == "Precision":
            detected_result = _detect_column_by_alias(all_columns, ["Result", "Observed Result", "Measurement"])
            detected_level = _detect_column_by_alias(all_columns, ["Level", "Sample Level", "Control Level"])
            detected_run = _detect_column_by_alias(all_columns, ["Run", "Run ID"])
            result_column = st.selectbox("Result", numeric_columns, index=_column_index(numeric_columns, detected_result), key=f"map_result_{program_name}_{analyte_name}_{study_type}")
            level_column = st.selectbox("Level", all_columns, index=_column_index(all_columns, detected_level), key=f"map_level_{program_name}_{analyte_name}_{study_type}")
            day_column = _optional_column_select("Day", all_columns, _detect_column_by_alias(all_columns, ["Day"]), f"map_day_{program_name}_{analyte_name}_{study_type}")
            run_column = _optional_column_select("Run", all_columns, detected_run, f"map_run_{program_name}_{analyte_name}_{study_type}")
            replicate_column = _optional_column_select("Replicate", all_columns, "Replicate", f"map_rep_{program_name}_{analyte_name}_{study_type}")
            sample_id_column = _optional_column_select("Sample ID", all_columns, "Sample ID", f"map_sample_{program_name}_{analyte_name}_{study_type}")
            selected_columns = [result_column, level_column, day_column, run_column, replicate_column, sample_id_column]
        elif study_type == "Accuracy":
            sample_id_column = st.selectbox("Sample ID", all_columns, index=_column_index(all_columns, "Sample ID"), key=f"map_sample_{program_name}_{analyte_name}_{study_type}")
            level_column = st.selectbox("Level", all_columns, index=_column_index(all_columns, "Level"), key=f"map_level_{program_name}_{analyte_name}_{study_type}")
            expected_column = st.selectbox("Expected Result", numeric_columns, index=_column_index(numeric_columns, "Expected Result"), key=f"map_expected_{program_name}_{analyte_name}_{study_type}")
            observed_column = st.selectbox("Observed Result", numeric_columns, index=_column_index(numeric_columns, "Observed Result", 1), key=f"map_observed_{program_name}_{analyte_name}_{study_type}")
            replicate_column = _optional_column_select("Replicate", all_columns, "Replicate", f"map_rep_{program_name}_{analyte_name}_{study_type}")
            include_column = _optional_column_select("Include in Analysis", all_columns, "Include in Analysis", f"map_include_{program_name}_{analyte_name}_{study_type}")
            selected_columns = [sample_id_column, level_column, expected_column, observed_column, replicate_column, include_column]
        elif study_type == "Linearity":
            level_column = st.selectbox("Level", all_columns, index=_column_index(all_columns, "Level"), key=f"map_level_{program_name}_{analyte_name}_{study_type}")
            expected_column = st.selectbox("Expected Result", numeric_columns, index=_column_index(numeric_columns, "Expected Result"), key=f"map_expected_{program_name}_{analyte_name}_{study_type}")
            observed_column = st.selectbox("Observed Result", numeric_columns, index=_column_index(numeric_columns, "Observed Result", 1), key=f"map_observed_{program_name}_{analyte_name}_{study_type}")
            replicate_column = _optional_column_select("Replicate", all_columns, "Replicate", f"map_rep_{program_name}_{analyte_name}_{study_type}")
            include_column = _optional_column_select("Include in Analysis", all_columns, "Include in Analysis", f"map_include_{program_name}_{analyte_name}_{study_type}")
            selected_columns = [level_column, expected_column, observed_column, replicate_column, include_column]
        elif study_type == "Stability":
            detected_sample = _detect_column_by_alias(all_columns, ["Sample ID", "Sample_ID"])
            detected_timepoint = _detect_column_by_alias(all_columns, ["Timepoint", "Timepoint Hours", "Timepoint_Hours", "Hours"])
            detected_baseline = _detect_column_by_alias(numeric_columns, ["Baseline Result", "Baseline_Result"])
            detected_result = _detect_column_by_alias(numeric_columns, ["Current Result", "Current_Result", "Result", "Observed Result"])
            detected_storage = _detect_column_by_alias(all_columns, ["Storage Condition", "Storage_Condition", "Condition"])
            sample_id_column = _required_column_select("Sample ID", all_columns, detected_sample, key=f"map_sample_{program_name}_{analyte_name}_{study_type}")
            timepoint_column = _required_column_select("Timepoint", all_columns, detected_timepoint, key=f"map_timepoint_{program_name}_{analyte_name}_{study_type}")
            result_column = _required_column_select("Result", numeric_columns, detected_result, key=f"map_result_{program_name}_{analyte_name}_{study_type}")
            storage_condition_column = _required_column_select("Storage Condition", all_columns, detected_storage, key=f"map_condition_{program_name}_{analyte_name}_{study_type}")
            replicate_column = _optional_column_select("Replicate", all_columns, "Replicate", f"map_rep_{program_name}_{analyte_name}_{study_type}")
            include_column = _optional_column_select("Include in Analysis", all_columns, "Include in Analysis", f"map_include_{program_name}_{analyte_name}_{study_type}")
            selected_columns = [sample_id_column, timepoint_column, detected_baseline, result_column, storage_condition_column, replicate_column, include_column]
            baseline_timepoint_available = False
            if timepoint_column:
                timepoint_values = data[timepoint_column].astype(str).str.strip().str.lower()
                numeric_timepoints = pd.to_numeric(data[timepoint_column], errors="coerce")
                baseline_timepoint_available = bool(
                    timepoint_values.isin(["baseline", "base", "day 0", "d0", "t0", "0"]).any()
                    or numeric_timepoints.eq(0).any()
                )
            required_mapping = {
                "Sample_ID": sample_id_column,
                "Timepoint_Hours": timepoint_column,
                "Baseline_Result or baseline timepoint": detected_baseline or ("Baseline timepoint" if baseline_timepoint_available else None),
                "Current_Result": result_column,
                "Storage_Condition": storage_condition_column,
            }
        else:
            detected_sample = _detect_column_by_alias(all_columns, ["Sample ID", "Sample_ID"])
            detected_type = _detect_column_by_alias(all_columns, ["Sample Type", "Sample_Type", "Blank Flag", "Blank_Flag"])
            detected_concentration = _detect_column_by_alias(numeric_columns, ["Concentration Level", "Concentration_Level", "Concentration"])
            detected_result = _detect_column_by_alias(numeric_columns, ["Observed Result", "Observed_Result", "Result"])
            sample_id_column = _required_column_select("Sample ID", all_columns, detected_sample, key=f"map_sample_{program_name}_{analyte_name}_{study_type}")
            sample_type_column = _required_column_select("Sample Type", all_columns, detected_type, key=f"map_type_{program_name}_{analyte_name}_{study_type}")
            concentration_column = _required_column_select("Concentration Level", numeric_columns, detected_concentration, key=f"map_conc_{program_name}_{analyte_name}_{study_type}")
            result_column = _required_column_select("Observed Result", numeric_columns, detected_result, key=f"map_result_{program_name}_{analyte_name}_{study_type}")
            replicate_column = _optional_column_select("Replicate", all_columns, "Replicate", f"map_rep_{program_name}_{analyte_name}_{study_type}")
            units_column = _optional_column_select("Units", all_columns, "Units", f"map_units_{program_name}_{analyte_name}_{study_type}")
            include_column = _optional_column_select("Include in Analysis", all_columns, "Include in Analysis", f"map_include_{program_name}_{analyte_name}_{study_type}")
            selected_columns = [sample_id_column, sample_type_column, concentration_column, result_column, replicate_column, units_column, include_column]
            required_mapping = {
                "Sample_ID": sample_id_column,
                "Blank_Flag / Sample Type": sample_type_column,
                "Concentration": concentration_column,
                "Result": result_column,
            }
    mapping_values = {
        key: value
        for key, value in locals().items()
        if key.endswith("_column") and isinstance(value, (str, type(None)))
    }
    persist_study_execution_value(program_name, analyte_name, study_type, "Column Mappings", mapping_values)

    missing_required = _mapping_failures(required_mapping)
    if missing_required:
        persist_study_execution_value(
            program_name,
            analyte_name,
            study_type,
            "Data Quality",
            {
                "Status": "FAIL",
                "Failure Type": "Dataset Structure Failure",
                "Missing Required Fields": missing_required,
            },
        )
        st.error("Data Quality FAIL")
        st.markdown("**Missing Required Fields**")
        for field in missing_required:
            st.write(f"- {field}")
        st.markdown("**Next Action:** Map the required fields or upload a dataset with the required study columns.")
        return

    quality = _build_quality_summary(data, [column for column in selected_columns if column], sample_id_column if "sample_id_column" in locals() else None)
    persist_study_execution_value(program_name, analyte_name, study_type, "Data Quality", quality.copy())
    if quality["Status"] == "PASS":
        if study_type == "Precision" and precision_completeness_issues:
            st.info("Data Quality Incomplete")
            st.write(
                f"{completeness['Replicates Found']} Replicates Found · "
                f"{completeness['Runs Found']} Runs Found · "
                f"{completeness['Days Found']} Days Found"
            )
            st.write("Precision calculations require at least two replicates, two runs, and two days.")
        else:
            duplicate_text = (
                "Repeated Sample IDs Detected · ✓ Required For Stability Analysis"
                if study_type == "Stability"
                else "No duplicate IDs"
                if quality["Duplicate IDs"] == 0
                else f"{quality['Duplicate IDs']} duplicate IDs"
            )
            missing_text = (
                "No missing values"
                if quality["Missing Values"] == 0
                else f"{quality['Missing Values']} missing values"
            )
            st.markdown("**✓ Data Quality Passed**")
            st.caption(f"{quality['Sample Count']} samples · {missing_text} · {duplicate_text}")
            if study_type == "Precision":
                st.write(
                    f"{completeness['Replicates Found']} Replicates Found · "
                    f"{completeness['Runs Found']} Runs Found · "
                    f"{completeness['Days Found']} Days Found"
                )
    else:
        st.error(f"{quality['Missing Values']} missing values detected. Resolve data quality issues before analysis.")
        return
    with st.expander("View Quality Details", expanded=False):
        st.dataframe(pd.DataFrame([quality]), width="stretch", hide_index=True)

    criteria_prefix = f"{program_name}_{analyte_name}_{study_type}"
    if study_type == "Precision":
        st.subheader("Acceptance Criteria Settings")
        criteria_cols = st.columns(3)
        with criteria_cols[0]:
            max_within_cv = st.number_input("Maximum Within-Run CV%", min_value=0.0, value=5.0, step=0.5, key=f"criteria_within_cv_{criteria_prefix}")
        with criteria_cols[1]:
            max_between_cv = st.number_input("Maximum Between-Run CV%", min_value=0.0, value=5.0, step=0.5, key=f"criteria_between_cv_{criteria_prefix}")
        with criteria_cols[2]:
            max_total_cv = st.number_input("Maximum Total CV%", min_value=0.0, value=5.0, step=0.5, key=f"criteria_total_cv_{criteria_prefix}")
        criteria = {"Maximum Within CV%": max_within_cv, "Maximum Between CV%": max_between_cv, "Maximum Total CV%": max_total_cv}
    else:
        with st.expander("Acceptance Criteria Settings", expanded=False):
            if study_type == "Accuracy":
                min_recovery = st.number_input("Minimum Recovery %", min_value=0.0, value=90.0, step=0.5, key=f"criteria_minrec_{criteria_prefix}")
                max_recovery = st.number_input("Maximum Recovery %", min_value=0.0, value=110.0, step=0.5, key=f"criteria_maxrec_{criteria_prefix}")
                max_bias = st.number_input("Maximum Bias %", min_value=0.0, value=10.0, step=0.5, key=f"criteria_bias_{criteria_prefix}")
                criteria = {"Minimum Recovery": min_recovery, "Maximum Recovery": max_recovery, "Maximum Bias": max_bias}
            elif study_type == "Linearity":
                min_r2 = st.number_input("Minimum R²", min_value=0.0, max_value=1.0, value=0.99, step=0.001, format="%.3f", key=f"criteria_r2_{criteria_prefix}")
                max_dev = st.number_input("Maximum Deviation %", min_value=0.0, value=10.0, step=0.5, key=f"criteria_dev_{criteria_prefix}")
                criteria = {"Minimum R²": min_r2, "Maximum Deviation": max_dev, "Recovery Lower": 90.0, "Recovery Upper": 110.0}
            elif study_type == "Stability":
                max_drift = st.number_input("Maximum Allowed Drift %", min_value=0.0, value=10.0, step=0.5, key=f"criteria_drift_{criteria_prefix}")
                min_recovery = st.number_input("Minimum Recovery %", min_value=0.0, value=90.0, step=0.5, key=f"criteria_rec_{criteria_prefix}")
                criteria = {"Maximum Drift": max_drift, "Minimum Recovery": min_recovery}
            else:
                max_lob = st.number_input("Maximum LoB", min_value=0.0, value=0.15, step=0.01, key=f"criteria_lob_{criteria_prefix}")
                max_lod = st.number_input("Maximum LoD", min_value=0.0, value=0.30, step=0.01, key=f"criteria_lod_{criteria_prefix}")
                target_cv = st.number_input("Target LoQ CV%", min_value=0.0, value=20.0, step=1.0, key=f"criteria_loqcv_{criteria_prefix}")
                max_loq = st.number_input("Maximum LoQ", min_value=0.0, value=1.0, step=0.05, key=f"criteria_loq_{criteria_prefix}")
                criteria = {"Maximum LoB": max_lob, "Maximum LoD": max_lod, "Target LoQ CV%": target_cv, "Maximum LoQ": max_loq}
    persist_study_execution_value(program_name, analyte_name, study_type, "Acceptance Criteria", criteria.copy())

    st.subheader(f"{study_type} Analysis")
    persisted_state = get_study_execution_state(program_name, analyte_name, study_type)
    result_key = persisted_analysis_state_key(program_name, analyte_name, study_type)
    if result_key not in st.session_state and isinstance(persisted_state.get("Analysis Result"), dict):
        st.session_state[result_key] = persisted_state["Analysis Result"]
    if result_key not in st.session_state:
        st.markdown("**Next Action**")
        if not st.button("Run Analysis", key=f"run_study_{program_name}_{analyte_name}_{study_type}", type="primary", use_container_width=True):
            return
        try:
            failure_type = None
            failure_reason = None
            if study_type == "Precision":
                if precision_completeness_issues:
                    reason = (
                        "Precision calculations require at least 2 replicates, at least 2 runs, and at least 2 days. "
                        f"Detected: {completeness['Replicates Found']} replicates, "
                        f"{completeness['Runs Found']} runs, {completeness['Days Found']} days."
                    )
                    decision = "INCOMPLETE"
                    failure_type = "Calculation Incomplete"
                    failure_reason = reason
                    criteria_table = pd.DataFrame(
                        [
                            {"Criterion": "Dataset Completeness", "Observed Value": "Insufficient Data", "Acceptance Limit": "Replicates, runs, and days required", "Status": "INCOMPLETE"}
                        ]
                    )
                    key_results = {
                        "Replicates": str(completeness["Replicates Found"]),
                        "Runs": str(completeness["Runs Found"]),
                        "Days": str(completeness["Days Found"]),
                    }
                    interpretation = reason
                    figures = {}
                    supporting = {"Data Quality": pd.DataFrame([quality])}
                else:
                    result = run_precision_study(data, result_column, level_column, day_column, run_column, replicate_column, sample_id_column)
                    total_cv = result.level_summary["CV%"].max() if not result.level_summary.empty else pd.NA
                    within_cv = result.run_summary["CV%"].max() if not result.run_summary.empty else pd.NA
                    between_cv = result.day_summary["CV%"].max() if not result.day_summary.empty else pd.NA
                    cv_min = result.level_summary["CV%"].min() if not result.level_summary.empty else pd.NA
                    cv_max = result.level_summary["CV%"].max() if not result.level_summary.empty else pd.NA

                    def cv_status(value: object, limit: float) -> str:
                        if value is pd.NA or pd.isna(value):
                            return "INCOMPLETE"
                        return "PASS" if float(value) <= limit else "FAIL"

                    criteria_table = pd.DataFrame(
                        [
                            {"Criterion": "Within-Run CV%", "Observed Value": _display_value(within_cv), "Acceptance Limit": f"<= {criteria['Maximum Within CV%']:g}%", "Status": cv_status(within_cv, criteria["Maximum Within CV%"])},
                            {"Criterion": "Between-Run CV%", "Observed Value": _display_value(between_cv, empty_label="Not Calculated"), "Acceptance Limit": f"<= {criteria['Maximum Between CV%']:g}%", "Status": cv_status(between_cv, criteria["Maximum Between CV%"])},
                            {"Criterion": "Total CV%", "Observed Value": _display_value(total_cv), "Acceptance Limit": f"<= {criteria['Maximum Total CV%']:g}%", "Status": cv_status(total_cv, criteria["Maximum Total CV%"])},
                        ]
                    )
                    calculation_incomplete = any(str(value) in {"N/A", "Not Calculated"} for value in criteria_table["Observed Value"])
                    decision = "INCOMPLETE" if calculation_incomplete else "PASS" if (criteria_table["Status"] == "PASS").all() else "FAIL"
                    failed_criteria = criteria_table[criteria_table["Status"] != "PASS"]["Criterion"].astype(str).tolist()
                    failure_type = "Calculation Incomplete" if decision == "INCOMPLETE" else ("Acceptance Criteria Failure" if decision == "FAIL" else None)
                    failure_reason = (
                        "Insufficient precision grouping data to calculate all required CV metrics."
                        if decision == "INCOMPLETE"
                        else f"Failed criteria: {', '.join(failed_criteria)}."
                        if failed_criteria
                        else None
                    )
                    key_results = {
                        "Within-Run CV%": _display_value(within_cv),
                        "Between-Run CV%": _display_value(between_cv, empty_label="Not Calculated"),
                        "Total CV%": _display_value(total_cv),
                    }
                    interpretation = (
                        f"Precision was evaluated using {len(result.analyzed_data)} measurements across "
                        f"{completeness['Days Found']} days and {completeness['Runs Found']} runs. "
                        "Within-run, between-run, and total precision met predefined acceptance criteria. "
                        "Precision study PASS."
                        if decision == "PASS"
                        else f"Precision study FAIL. {failure_reason}"
                        if decision == "FAIL"
                        else str(failure_reason)
                    )
                    figures = {
                        "Precision Distribution Plot": create_precision_box_plot(result.analyzed_data),
                        "CV% Summary Plot": create_precision_cv_bar_chart(result.level_summary),
                        "Precision Component Summary": create_precision_variance_component_plot(result.variance_components),
                        "Run-to-Run Variation Plot": create_precision_run_variation_plot(result.run_summary),
                        "Day-to-Day Variation Plot": create_precision_day_variation_plot(result.day_summary),
                        "Levey-Jennings Style Trend Plot": create_precision_run_chart(result.analyzed_data),
                    }
                    design_summary = pd.DataFrame(
                        [
                            {
                                "Samples": len(result.analyzed_data),
                                "Runs": result.analyzed_data["Run"].nunique() if "Run" in result.analyzed_data.columns else "N/A",
                                "Days": result.analyzed_data["Day"].nunique() if "Day" in result.analyzed_data.columns else "N/A",
                            }
                        ]
                    )
                    supporting = {
                        "Study Design": design_summary,
                        "Variance Components": result.variance_components,
                        "QC Level Summary": result.level_summary,
                        "Run Summary": result.run_summary,
                        "Day Summary": result.day_summary,
                        "Raw Study Data": result.analyzed_data,
                    }
            elif study_type == "Accuracy":
                result = run_accuracy_study(data, sample_id_column, level_column, expected_column, observed_column, replicate_column=replicate_column, include_column=include_column, max_abs_percent_bias=criteria["Maximum Bias"], min_recovery=criteria["Minimum Recovery"], max_recovery=criteria["Maximum Recovery"])
                criteria_table = result.level_decision_table.rename(columns={"Pass/Fail Status": "Status"})
                decision = "PASS" if not (criteria_table["Status"].astype(str).str.upper() == "FAIL").any() else "FAIL"
                mean_recovery = float(result.accuracy_summary["Percent Recovery"].mean())
                mean_bias = float(result.accuracy_summary["Percent Bias"].mean())
                key_results = {
                    "Recovery": f"{mean_recovery:.2f}%",
                    "Bias": f"{mean_bias:.2f}%",
                    "Mean Recovery": f"{mean_recovery:.2f}%",
                }
                interpretation = f"Observed recovery and bias were {'within' if decision == 'PASS' else 'outside'} predefined limits. Accuracy study {decision}."
                figures = {
                    "Expected vs Observed Plot": create_accuracy_expected_observed_plot(result.accuracy_summary),
                    "Recovery Plot": create_accuracy_recovery_plot(result.accuracy_summary, criteria["Minimum Recovery"], criteria["Maximum Recovery"]),
                    "Bias Plot": create_accuracy_percent_bias_plot(result.accuracy_summary, criteria["Maximum Bias"]),
                    "Replicate Distribution Plot": create_accuracy_replicate_distribution_plot(result.analyzed_data),
                }
                accuracy_summary = result.accuracy_summary.copy()
                compact_accuracy_summary = accuracy_summary[
                    ["Expected Result", "Mean Observed Result", "Percent Recovery", "Percent Bias"]
                ].rename(
                    columns={
                        "Expected Result": "Expected",
                        "Mean Observed Result": "Observed",
                        "Percent Recovery": "Recovery %",
                        "Percent Bias": "Bias %",
                    }
                )
                statistical_details = accuracy_summary.drop(
                    columns=["Expected Result", "Mean Observed Result", "Percent Recovery", "Percent Bias"],
                    errors="ignore",
                )
                supporting = {
                    "Accuracy Summary": compact_accuracy_summary,
                    "Recovery Summary": result.recovery_summary,
                    "Bias Summary": result.bias_summary,
                    "Worst Case Summary": pd.DataFrame([result.worst_case_summary]),
                    "Statistical Details": statistical_details,
                    "Analyzed Data": result.analyzed_data,
                }
                if result.data_quality_warnings:
                    supporting["Data Quality Warnings"] = pd.DataFrame(
                        {"Warning": result.data_quality_warnings}
                    )
            elif study_type == "Linearity":
                result = run_linearity_study(data, expected_column, observed_column, level_column, replicate_column=replicate_column, include_column=include_column)
                criteria_result = evaluate_linearity_criteria(result.level_summary, result.regression_summary, criteria["Minimum R²"], 0.90, 1.10, criteria["Maximum Deviation"], criteria["Recovery Lower"], criteria["Recovery Upper"])
                criteria_table = _checks_to_criteria_table(criteria_result)
                decision = str(criteria_result["decision"])
                max_dev = float(result.level_summary["Percent Bias"].abs().max())
                key_results = {
                    "R²": f"{float(result.regression_summary['R²']):.4f}",
                    "Slope": f"{float(result.regression_summary['Slope']):.3f}",
                    "Intercept": f"{float(result.regression_summary['Intercept']):.3f}",
                    "Maximum Deviation": f"{max_dev:.2f}% / <= {criteria['Maximum Deviation']:g}% allowed",
                }
                interpretation = (
                    "Observed response remained linear across the evaluated range. "
                    "R², slope, intercept, and maximum deviation met predefined acceptance criteria. "
                    "Linearity study PASS."
                    if decision == "PASS"
                    else (
                        "Observed response did not meet predefined linearity criteria across the evaluated range. "
                        "One or more regression or deviation criteria were not satisfied. "
                        f"Maximum observed deviation was {max_dev:.2f}%. Linearity study {decision}."
                    )
                )
                figures = {
                    "Regression Plot": create_linearity_plot(result.level_summary, result.regression_summary),
                    "Residual Plot": create_linearity_residual_plot(result.level_summary, result.regression_summary),
                    "Recovery / Deviation Plot": create_percent_recovery_plot(result.level_summary),
                }
                linearity_summary = result.level_summary.copy()
                compact_linearity_summary = linearity_summary[
                    ["Expected Result", "Mean Observed Result", "Percent Recovery", "Percent Bias"]
                ].rename(
                    columns={
                        "Expected Result": "Expected",
                        "Mean Observed Result": "Observed",
                        "Percent Recovery": "Recovery %",
                        "Percent Bias": "Bias %",
                    }
                )
                statistical_details = linearity_summary.drop(columns=["Expected Result", "Mean Observed Result", "Percent Recovery", "Percent Bias"], errors="ignore")
                supporting = {
                    "Linearity Summary": compact_linearity_summary,
                    "Residual / Deviation Summary": result.level_summary[
                        [
                            "Level",
                            "Expected Result",
                            "Mean Observed Result",
                            "Difference",
                            "Percent Recovery",
                            "Percent Bias",
                        ]
                    ],
                    "Statistical Details": statistical_details,
                    "Analyzed Data": result.analyzed_data,
                }
            elif study_type == "Stability":
                result = run_stability_study(data, sample_id_column, timepoint_column, result_column, storage_condition_column=storage_condition_column, replicate_column=replicate_column, include_column=include_column)
                criteria_result = evaluate_stability_criteria(result.overall_summary, criteria["Maximum Drift"], criteria["Minimum Recovery"], 0.50, 2.0)
                criteria_table = _checks_to_criteria_table(criteria_result)
                decision = str(criteria_result["decision"])
                max_drift = float(result.overall_summary.get("Maximum Observed Change", result.timepoint_summary["Mean Percent Change"].abs().max()))
                key_results = {
                    "Percent Recovery": f"{float(result.overall_summary.get('Minimum Recovery', result.timepoint_summary['Percent Recovery'].min())):.2f}%",
                    "Mean Drift": f"{float(result.timepoint_summary['Mean Percent Change'].mean()):.2f}%",
                    "Maximum Drift": f"{max_drift:.2f}% / <= {criteria['Maximum Drift']:g}% allowed",
                }
                interpretation = (
                    "Recovery remained above the required threshold. "
                    "Maximum drift remained within allowable limits. "
                    "Stability study PASS."
                    if decision == "PASS"
                    else (
                        "Analyte did not remain within predefined stability criteria across all evaluated timepoints. "
                        "Review recovery and drift results before submission. "
                        f"Stability study {decision}."
                    )
                )
                figures = {
                    "Recovery vs Timepoint": create_stability_recovery_plot(result.timepoint_summary, criteria["Minimum Recovery"]),
                    "Drift vs Timepoint": create_stability_percent_change_plot(result.timepoint_summary, criteria["Maximum Drift"]),
                    "Individual Sample Trajectories": create_individual_stability_plot(result.analyzed_data),
                }
                if not result.condition_comparison.empty:
                    figures["Storage Condition Comparison"] = create_condition_difference_plot(
                        result.condition_comparison,
                        acceptance_limit=0.50,
                    )
                supporting = {
                    "Stability Summary": result.stability_summary,
                    "Timepoint Summary": result.timepoint_summary,
                    "Bias Summary": result.bias_summary,
                    "Recovery Summary": result.recovery_summary,
                    "Analyzed Data": result.analyzed_data.drop(columns=["Timepoint Sort"], errors="ignore"),
                }
                if not result.condition_comparison.empty:
                    supporting["Storage Condition Comparison"] = result.condition_comparison
                if not result.outlier_table.empty:
                    supporting["Potential Instability Flags"] = result.outlier_table
            else:
                result = run_detection_capability_study(data, sample_id_column, sample_type_column, concentration_column, result_column, replicate_column=replicate_column, units_column=units_column, include_column=include_column, target_loq_cv=criteria["Target LoQ CV%"])
                criteria_result = evaluate_detection_capability_criteria(result.overall_summary, criteria["Maximum LoB"], criteria["Maximum LoD"], criteria["Target LoQ CV%"], criteria["Maximum LoQ"], 2.0)
                criteria_table = _checks_to_criteria_table(criteria_result)
                criteria_table["Acceptance Source"] = "User Defined"
                decision = str(criteria_result["decision"])
                units = _assay_units(analyte_name, result.analyzed_data)
                loq_cv = float(result.overall_summary.get("Operational LoQ CV%", float("nan")))
                key_results = {
                    "LoB": _with_units(result.overall_summary["LoB"], units),
                    "LoD": _with_units(result.overall_summary["LoD"], units),
                    "LoQ": f"{_with_units(result.overall_summary['LoQ'], units)} · CV {_display_value(loq_cv)}%",
                }
                interpretation = (
                    "Observed LoB, LoD, and LoQ met predefined acceptance criteria. "
                    f"LoQ was established at {_with_units(result.overall_summary['LoQ'], units)} with CV {_display_value(loq_cv)}%. "
                    "Detection Capability study PASS."
                    if decision == "PASS"
                    else (
                        "Observed detection capability did not meet all predefined acceptance criteria. "
                        "Review LoB, LoD, and LoQ determination details before submission. "
                        f"Detection Capability study {decision}."
                    )
                )
                figures = {
                    "CV vs Concentration": create_loq_precision_curve(result.loq_summary, criteria["Target LoQ CV%"], float(result.overall_summary["LoQ"])),
                    "LoB / LoD Summary": create_detection_density_plot(result.analyzed_data, float(result.overall_summary["LoB"]), float(result.overall_summary["LoD"])),
                    "Blank Replicate Distribution": create_blank_replicate_boxplot(result.analyzed_data),
                    "Low-Level Replicate Distribution": create_low_level_distribution_plot(result.analyzed_data),
                    "LoQ Decision Plot": create_loq_decision_plot(result.loq_summary, criteria["Target LoQ CV%"]),
                }
                lob_lod_calculations = pd.DataFrame(
                    [
                        {
                            "Calculation": "LoB",
                            "Replicates": int(result.lob_summary["N"].iloc[0]),
                            "Mean": _with_units(result.lob_summary["Mean Blank"].iloc[0], units),
                            "SD": _with_units(result.lob_summary["SD Blank"].iloc[0], units),
                            "Result": _with_units(result.overall_summary["LoB"], units),
                        },
                        {
                            "Calculation": "LoD",
                            "Replicates": int(result.lod_summary["N"].iloc[0]),
                            "Mean": _with_units(result.lod_summary["Mean Low Sample"].iloc[0], units),
                            "SD": _with_units(result.lod_summary["SD Low Sample"].iloc[0], units),
                            "Result": _with_units(result.overall_summary["LoD"], units),
                        },
                    ]
                )
                loq_determination = result.loq_summary.copy()
                loq_determination["Target CV%"] = f"<= {criteria['Target LoQ CV%']:g}%"
                loq_determination["Operational LoQ"] = loq_determination["Concentration Level"].apply(
                    lambda value: "Yes" if pd.notna(value) and float(value) == float(result.overall_summary["LoQ"]) else ""
                )
                supporting = {
                    "Detection Data Quality": result.data_quality_summary,
                    "Methodology": result.methodology_table,
                    "LoB / LoD Calculation": lob_lod_calculations,
                    "LoQ Determination": loq_determination,
                    "Decision Matrix": result.decision_matrix,
                    "Blank Distribution": result.analyzed_data[result.analyzed_data["Sample Type"].str.lower() == "blank"],
                    "Low Concentration Distribution": result.analyzed_data[result.analyzed_data["Sample Type"].str.lower() == "low concentration"],
                    "Analyzed Data": result.analyzed_data,
                }
                if not result.outlier_table.empty:
                    supporting["Replicate Outliers"] = result.outlier_table
        except Exception as exc:
            st.error(f"Calculation Failure: {study_type} analysis could not be completed. {exc}")
            return
        st.session_state[result_key] = {
            "decision": decision,
            "criteria_table": criteria_table,
            "key_results": key_results,
            "interpretation": interpretation,
            "figures": figures,
            "supporting": supporting,
            "failure_type": failure_type,
            "failure_reason": failure_reason,
        }
        persist_study_execution_value(program_name, analyte_name, study_type, "Analysis Result", st.session_state[result_key])
        persist_study_execution_value(program_name, analyte_name, study_type, "Decision", decision)
        persist_study_execution_value(program_name, analyte_name, study_type, "Analysis Complete", decision != "INCOMPLETE")
        if decision == "INCOMPLETE":
            mark_data_uploaded_if_needed(program, analyte_name, study_type)
            record_program_activity(program_name, f"{analyte_name} {study_type}", "Analysis incomplete", failure_reason or f"{study_type} analysis could not be completed.")
        else:
            ensure_study_execution_record(program, analyte_name, study_type, "Analysis Complete")
            record_program_activity(program_name, f"{analyte_name} {study_type}", "Analysis completed", f"{study_type} analysis completed in Execution.")

    state = st.session_state[result_key]
    if state["decision"] == "INCOMPLETE":
        st.info("Analysis Incomplete")
    else:
        st.markdown("**✓ Analysis Completed**")
        st.caption(f"{study_type} calculations, scientific plots, and acceptance criteria were generated.")
    _render_execution_results(
        program=program,
        analyte=analyte,
        study_type=study_type,
        decision=state["decision"],
        criteria_table=state["criteria_table"],
        key_results=state["key_results"],
        interpretation=state["interpretation"],
        figures=state["figures"],
        supporting_tables=state["supporting"],
        failure_type=state.get("failure_type"),
        failure_reason=state.get("failure_reason"),
    )
    _render_execution_completion_step(
        program,
        analyte,
        study_type,
        state["decision"],
        state["criteria_table"],
        state["key_results"],
        state["interpretation"],
        state["figures"],
        state["supporting"],
    )


def render_method_comparison_execution_workspace(
    program: dict[str, object],
    analyte: dict[str, object],
    study_type: str,
    record: dict[str, object],
) -> None:
    """Render method comparison execution inside a generated study workspace."""

    program_name = str(program["Program Name"])
    analyte_name = str(analyte["Analyte"])
    settings = platform_settings()
    result_key = f"execution_result_{program_name}_{analyte_name}_{study_type}"
    upload_key = f"execution_upload_meta_{program_name}_{analyte_name}_{study_type}"
    review_key = f"execution_results_reviewed_{program_name}_{analyte_name}_{study_type}"
    persisted_state = get_study_execution_state(program_name, analyte_name, study_type)
    if result_key not in st.session_state and isinstance(persisted_state.get("Analysis Result"), dict):
        st.session_state[result_key] = persisted_state["Analysis Result"]
    execution_locked = bool(persisted_state.get("Execution Complete"))
    sample_requirement = int(analyte.get("Required Samples", 80))
    metadata = {
        "Validation Project Name": program_name,
        "Study Name": f"{analyte_name} Method Comparison",
        "Study Objective": f"Evaluate agreement for {analyte_name} results under {program.get('Validation Context', 'the defined validation context')}.",
        "Study Design": "Paired reference and candidate results analyzed for correlation, regression, bias, and agreement.",
        "Assay / Biomarker": analyte_name,
        "Specimen Type": str(program.get("Validation Context", "")),
        "Reference Method": "Reference Method",
        "Candidate Method": "Candidate Method",
        "Analyst Name": settings.get("Analyst Name", ""),
        "Study Date": date.today().isoformat(),
        "Units": "",
    }
    expected_cols = pd.DataFrame(
        [
            {"Column": "Sample ID", "Requirement": "Required", "Description": "Unique paired specimen identifier"},
            {"Column": "Reference Result", "Requirement": "Required", "Description": "Serum or reference specimen result"},
            {"Column": "Candidate Result", "Requirement": "Required", "Description": "Microtainer or candidate specimen result"},
            {"Column": "Collection Date", "Requirement": "Optional", "Description": "Specimen collection date"},
        ]
    )
    st.subheader("Method Comparison Dataset")
    with st.expander("Dataset Structure", expanded=False):
        st.dataframe(expected_cols, use_container_width=True, hide_index=True)
    uploaded_file = None
    use_sample_data = False
    if execution_locked:
        st.caption("Completed study is locked. Dataset, mappings, criteria, and results are read-only.")
    else:
        uploaded_file = st.file_uploader(
            "Upload Dataset",
            type=["csv", "xlsx", "xls"],
            key=f"execution_upload_{program_name}_{analyte_name}_{study_type}",
        )
        use_sample_data = st.checkbox(
            "Load Demo Dataset",
            value=False,
            key=f"execution_sample_{program_name}_{analyte_name}_{study_type}",
        )
    data = None
    uploaded_file_name = str(persisted_state.get("Dataset Source", ""))
    if uploaded_file is not None:
        try:
            data = load_uploaded_file(uploaded_file)
            uploaded_file_name = uploaded_file.name
            reset_analysis_for_new_dataset(program_name, analyte_name, study_type, uploaded_file_name)
            persist_study_execution_value(program_name, analyte_name, study_type, "Dataset", data.copy())
            persist_study_execution_value(program_name, analyte_name, study_type, "Dataset Source", uploaded_file_name)
            persist_study_execution_value(program_name, analyte_name, study_type, "Dataset Hash", dataframe_content_hash(data))
            persist_study_execution_value(program_name, analyte_name, study_type, "Dataset Uploaded At", datetime.now().strftime("%Y-%m-%d %H:%M"))
            if upload_key not in st.session_state or st.session_state[upload_key].get("Uploaded File") != uploaded_file.name:
                reset_execution_for_uploaded_dataset(program, analyte_name, study_type)
                st.session_state[upload_key] = {
                    "Uploaded File": uploaded_file.name,
                    "Upload Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                st.rerun()
        except Exception as exc:
            st.error(f"Unable to load file: {exc}")
            return
    elif use_sample_data:
        data = pd.read_csv(SAMPLE_DATA_PATH)
        uploaded_file_name = SAMPLE_DATA_PATH.name
        reset_analysis_for_new_dataset(program_name, analyte_name, study_type, uploaded_file_name)
        persist_study_execution_value(program_name, analyte_name, study_type, "Dataset", data.copy())
        persist_study_execution_value(program_name, analyte_name, study_type, "Dataset Source", uploaded_file_name)
        persist_study_execution_value(program_name, analyte_name, study_type, "Dataset Hash", dataframe_content_hash(data))
        persist_study_execution_value(program_name, analyte_name, study_type, "Dataset Uploaded At", datetime.now().strftime("%Y-%m-%d %H:%M"))
        if upload_key not in st.session_state or st.session_state[upload_key].get("Uploaded File") != SAMPLE_DATA_PATH.name:
            reset_execution_for_uploaded_dataset(program, analyte_name, study_type)
            st.session_state[upload_key] = {
                "Uploaded File": SAMPLE_DATA_PATH.name,
                "Upload Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            st.rerun()
    elif isinstance(persisted_state.get("Dataset"), pd.DataFrame):
        data = persisted_state["Dataset"].copy()

    if data is None:
        st.info("Upload a CSV or Excel file to begin method comparison analysis.")
        return
    if not persisted_state.get("Dataset Hash"):
        persist_study_execution_value(program_name, analyte_name, study_type, "Dataset Hash", dataframe_content_hash(data))
    if persisted_state.pop("Dataset Changed", False):
        save_study_execution_state_key(study_state_key(program_name, analyte_name, study_type), persisted_state)
        ensure_study_execution_record(program, analyte_name, study_type, "Data Uploaded")
    persist_study_execution_value(program_name, analyte_name, study_type, "Dataset Uploaded", True)

    metadata["Sample Count"] = len(data)
    upload_meta = st.session_state.get(upload_key, {})
    detected_columns = detect_required_method_columns(data)
    numeric_columns = get_numeric_columns(data)
    if len(numeric_columns) < 2:
        st.error("At least two numeric result columns are required for method comparison.")
        return

    detected_sample_id = detect_sample_id_column(data)
    sample_id_options = ["None"] + list(data.columns)
    sample_id_default = sample_id_options.index(detected_sample_id) if detected_sample_id in sample_id_options else 0
    detected_required_count = sum(
        1 for label in ["Sample ID", "Reference Result", "Candidate Result"]
        if detected_columns.get(label)
    )
    stored_mapping = persisted_state.get("Column Mappings") if isinstance(persisted_state.get("Column Mappings"), dict) else {}
    reference_column = str(stored_mapping.get("reference_column") or numeric_columns[0])
    candidate_column = str(stored_mapping.get("candidate_column") or (numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0]))
    stored_sample_id = stored_mapping.get("sample_id_column")
    sample_id_column = str(stored_sample_id) if stored_sample_id else None
    st.markdown("**✓ Dataset Uploaded**")
    st.caption(f"{len(data)} paired samples · {detected_required_count} required columns validated")
    with st.expander("View Dataset Details", expanded=False):
        detected_cols = st.columns(3)
        for index, label in enumerate(["Sample ID", "Reference Result", "Candidate Result"]):
            detected_value = detected_columns.get(label)
            with detected_cols[index]:
                if detected_value:
                    st.success(f"✓ {detected_value}")
                else:
                    st.warning(f"Missing: {label}")
        st.markdown("**Column Mapping**")
        if execution_locked:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Field": "Reference Result", "Mapped Column": reference_column},
                        {"Field": "Candidate Result", "Mapped Column": candidate_column},
                        {"Field": "Sample ID", "Mapped Column": sample_id_column or "None"},
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            map_cols = st.columns(3)
            with map_cols[0]:
                reference_default = numeric_columns.index(reference_column) if reference_column in numeric_columns else 0
                reference_column = st.selectbox("Reference Result", numeric_columns, index=reference_default, key=f"exec_ref_{program_name}_{analyte_name}_{study_type}")
            with map_cols[1]:
                candidate_default = numeric_columns.index(candidate_column) if candidate_column in numeric_columns else (1 if len(numeric_columns) > 1 else 0)
                candidate_column = st.selectbox("Candidate Result", numeric_columns, index=candidate_default, key=f"exec_cand_{program_name}_{analyte_name}_{study_type}")
            with map_cols[2]:
                sample_id_default = sample_id_options.index(sample_id_column) if sample_id_column in sample_id_options else sample_id_default
                sample_id_selection = st.selectbox("Sample ID", sample_id_options, index=sample_id_default, key=f"exec_sample_id_{program_name}_{analyte_name}_{study_type}")
                sample_id_column = None if sample_id_selection == "None" else sample_id_selection
            persist_study_execution_value(
                program_name,
                analyte_name,
                study_type,
                "Column Mappings",
                {
                    "reference_column": reference_column,
                    "candidate_column": candidate_column,
                    "sample_id_column": sample_id_column,
                },
            )
        st.write(f"Uploaded file: {upload_meta.get('Uploaded File') or uploaded_file_name}")
        st.write(f"Upload timestamp: {persisted_state.get('Dataset Uploaded At') or upload_meta.get('Upload Timestamp', '')}")
        st.write(f"Dataset hash: {str(persisted_state.get('Dataset Hash', ''))[:16]}")
        st.write(f"Columns detected: {len(data.columns)}")
        st.dataframe(data.head(20), width="stretch")

    quality = method_comparison_data_quality(data, reference_column, candidate_column, sample_id_column)
    quality_issues = build_method_comparison_quality_issues(data, reference_column, candidate_column, sample_id_column)
    quality["Required Columns Detected"] = detected_required_count
    critical_issues = []
    if reference_column == candidate_column:
        critical_issues.append("Reference and candidate result columns must be different.")
    if quality["Missing Values"]:
        critical_issues.append(f"{quality['Missing Values']} missing required result values.")
    if quality["Invalid Numeric Values"]:
        critical_issues.append(f"{quality['Invalid Numeric Values']} invalid numeric values.")
    if quality["Range Violations"]:
        critical_issues.append(f"{quality['Range Violations']} zero or negative result values.")
    warnings = []
    if quality["Duplicate IDs"]:
        warnings.append(f"{quality['Duplicate IDs']} duplicate sample IDs detected.")
    if quality["Outlier Count"]:
        warnings.append(f"{quality['Outlier Count']} potential outliers identified.")
    if detected_required_count < 3:
        warnings.append("One or more expected columns were not automatically detected. Confirm manual column mapping before analysis.")
    critical_quality_issues = [issue for issue in quality_issues if issue["Severity"] == "Critical"]
    warning_quality_issues = [issue for issue in quality_issues if issue["Severity"] == "Warning"]
    has_critical_gate_issue = bool(critical_quality_issues or critical_issues)
    if has_critical_gate_issue:
        data_quality_status = "FAIL"
        data_quality_status_text = "Critical issues detected. Analysis blocked until corrected source data is uploaded."
    elif warning_quality_issues:
        data_quality_status = "WARNING"
        data_quality_status_text = "Minor issues detected. Analysis may proceed with review."
    else:
        data_quality_status = "PASS"
        data_quality_status_text = "Ready For Analysis"
    quality["Data Quality Status"] = data_quality_status
    quality["Critical Issues"] = len(critical_quality_issues)
    quality["Warnings"] = warnings
    persist_study_execution_value(program_name, analyte_name, study_type, "Data Quality", quality.copy())

    st.subheader("Data Quality")
    if data_quality_status == "PASS" and not quality_issues:
        st.markdown("**✓ Data Quality Passed**")
        st.caption(f"{quality['Sample Count']} paired samples · No missing values · No duplicate IDs")
        with st.expander("View Quality Details", expanded=False):
            st.write(f"Required Columns: {quality['Required Columns Detected']}")
            st.write(f"Potential Outliers: {quality['Outlier Count']}")
            st.write(f"Range Violations: {quality['Range Violations']}")
            st.write(f"Invalid Numeric Values: {quality['Invalid Numeric Values']}")
    else:
        st.caption(data_quality_status_text)
        gate_cols = st.columns(3)
        with gate_cols[0]:
            st.markdown("**Critical Issues**")
            if critical_issues:
                for issue in critical_issues:
                    st.error(issue)
            if critical_quality_issues:
                for issue in critical_quality_issues:
                    st.error(f"{issue['Sample ID']} · {issue['Column']} = {issue['Value']} · {issue['Issue']}")
            if not critical_issues and not critical_quality_issues:
                st.success("No Critical Issues Found")
        with gate_cols[1]:
            st.markdown("**Warnings**")
            if warnings:
                for warning in warnings:
                    st.warning(warning)
            else:
                st.success("No Warnings")
        with gate_cols[2]:
            st.markdown("**Recommendations**")
            if has_critical_gate_issue:
                st.write("Correct the source dataset and upload the revised file before running analysis.")
            elif warnings:
                st.write("Review warnings before final submission.")
            else:
                st.write("Dataset is ready for analysis.")

        if quality_issues:
            st.markdown("**Data Quality Issues**")
            issue_table = pd.DataFrame(quality_issues)
            st.dataframe(
                issue_table[
                    [
                        "Sample ID",
                        "Column",
                        "Value",
                        "Issue",
                        "Severity",
                        "Suggested Action",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
            with st.expander("View Problem Records", expanded=bool(critical_quality_issues)):
                for issue in quality_issues:
                    row_number = int(issue["Row"]) - 1
                    st.markdown(
                        f"**{issue['Severity']} Issue:** {issue['Sample ID']} · "
                        f"{issue['Column']} = {issue['Value']} · {issue['Issue']}"
                    )
                    st.write(f"Suggested Action: {issue['Suggested Action']}")
                    st.dataframe(data.iloc[[row_number]], width="stretch")
                    st.divider()

    with st.expander("Acceptance Criteria Settings", expanded=False):
        if execution_locked and isinstance(persisted_state.get("Acceptance Criteria"), dict):
            criteria = dict(persisted_state["Acceptance Criteria"])
            st.dataframe(
                pd.DataFrame([{"Criterion": key, "Value": value} for key, value in criteria.items()]),
                width="stretch",
                hide_index=True,
            )
        else:
            criteria = render_method_comparison_criteria()
    persist_study_execution_value(program_name, analyte_name, study_type, "Acceptance Criteria", criteria.copy())

    st.subheader("Analysis")
    has_result = result_key in st.session_state
    run_analysis = False

    if has_critical_gate_issue:
        first_issue = critical_quality_issues[0] if critical_quality_issues else None
        affected_sample = first_issue["Sample ID"] if first_issue else "Column mapping"
        affected_detail = f"{first_issue['Column']} = {first_issue['Value']}" if first_issue else critical_issues[0]
        st.error(
            "Analysis Blocked\n\n"
            f"Reason: {len(critical_quality_issues) + len(critical_issues)} critical data quality issue(s) detected.\n\n"
            f"Affected Sample: {affected_sample}\n\n"
            f"{affected_detail}\n\n"
            "Correct the source dataset and upload the revised file before analysis can proceed."
        )
    analysis_complete = result_key in st.session_state
    if execution_locked:
        st.caption("Analysis is locked because this study has been completed.")
    elif analysis_complete:
        st.markdown("**✓ Analysis Completed**")
        st.caption("Regression statistics, agreement metrics, scientific plots, and acceptance criteria were generated.")
    elif not has_result:
        st.markdown("**Next Action**")
        run_analysis = st.button(
            "Run Analysis",
            type="primary",
            key=f"exec_primary_action_{program_name}_{analyte_name}_{study_type}_run_analysis",
            use_container_width=True,
            disabled=has_critical_gate_issue,
        )

    if not run_analysis and result_key not in st.session_state:
        return

    if run_analysis and has_critical_gate_issue:
        st.error("Analysis cannot run until critical data quality issues are resolved.")
        return

    if run_analysis:
        if reference_column == candidate_column:
            st.warning("Select different reference and candidate result columns.")
            return
        with st.spinner("Running method comparison analysis..."):
            try:
                result = run_method_comparison(data, reference_column, candidate_column, sample_id_column)
            except Exception as exc:
                st.error(f"Analysis could not be completed: {exc}")
                return
        percent_samples_meeting_agreement = calculate_percent_samples_meeting_agreement(
            result.analyzed_data,
            criteria["Sample Agreement Percent Bias Limit"],
        )
        agreement_limit = criteria["Sample Agreement Percent Bias Limit"]
        passing_samples = int((result.analyzed_data["Percent Bias"].abs() <= agreement_limit).sum())
        failing_samples = int(len(result.analyzed_data) - passing_samples)
        criteria_result = evaluate_acceptance_criteria(
            result.summary,
            min_r_squared=criteria["Minimum R²"],
            min_correlation_r=criteria["Minimum Correlation r"],
            slope_lower_limit=criteria["Slope Lower Limit"],
            slope_upper_limit=criteria["Slope Upper Limit"],
            max_abs_intercept=criteria["Maximum Absolute Intercept"],
            max_abs_mean_bias=criteria["Maximum Absolute Mean Bias"],
            max_abs_mean_percent_bias=criteria["Maximum Absolute Mean Percent Bias"],
            max_abs_mean_difference=criteria["Maximum Absolute Mean Difference"],
            percent_samples_meeting_agreement=percent_samples_meeting_agreement,
            min_percent_samples_meeting_agreement=criteria["Minimum Percent Samples Meeting Agreement"],
        )
        decision = str(criteria_result["decision"])
        criteria_table = build_criteria_table(criteria_result)
        summary_table = build_summary_table(result.summary)
        outlier_table = get_top_outliers(result.analyzed_data, count=5)
        interpretation = generate_interpretation(result.summary, metadata, criteria, decision)
        st.session_state[result_key] = {
            "result": result,
            "summary_table": summary_table,
            "criteria_table": criteria_table,
            "outlier_table": outlier_table,
            "interpretation": interpretation,
            "decision": decision,
            "reference_column": reference_column,
            "candidate_column": candidate_column,
            "agreement_rate": percent_samples_meeting_agreement,
            "passing_samples": passing_samples,
            "failing_samples": failing_samples,
            "data_quality": quality,
        }
        persist_study_execution_value(program_name, analyte_name, study_type, "Analysis Result", st.session_state[result_key])
        persist_study_execution_value(program_name, analyte_name, study_type, "Decision", decision)
        persist_study_execution_value(program_name, analyte_name, study_type, "Analysis Complete", True)
        persist_study_execution_value(program_name, analyte_name, study_type, "Execution Complete", False)
        analysis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        analysis_run_id = f"{study_state_key(program_name, analyte_name, study_type)}::{datetime.now().strftime('%Y%m%d%H%M%S')}"
        persist_study_execution_value(program_name, analyte_name, study_type, "Analysis Timestamp", analysis_timestamp)
        persist_study_execution_value(program_name, analyte_name, study_type, "Analysis Run ID", analysis_run_id)
        persist_study_execution_value(program_name, analyte_name, study_type, "Software Version", APP_VERSION)
        ensure_study_execution_record(program, analyte_name, study_type, "Analysis Complete")
        record_program_activity(program_name, f"{analyte_name} {study_type}", "Analysis completed", "Method comparison analysis completed in Validation Execution.")
        st.session_state[review_key] = False
        st.rerun()

    analysis_state = st.session_state[result_key]
    result = analysis_state["result"]
    summary_table = analysis_state["summary_table"]
    criteria_table = analysis_state["criteria_table"]
    outlier_table = analysis_state["outlier_table"]
    interpretation = str(analysis_state["interpretation"])
    agreement_rate = float(analysis_state.get("agreement_rate", 0.0))
    passing_samples = int(analysis_state.get("passing_samples", 0))
    failing_samples = int(analysis_state.get("failing_samples", 0))

    if "Acceptance Source" not in criteria_table.columns:
        criteria_table = criteria_table.copy()
        insert_at = list(criteria_table.columns).index("Acceptance Limit") + 1 if "Acceptance Limit" in criteria_table.columns else len(criteria_table.columns)
        criteria_table.insert(insert_at, "Acceptance Source", "User Defined")
        analysis_state["criteria_table"] = criteria_table

    scatter_plot = add_identity_line_to_method_plot(create_scatter_plot(result.analyzed_data, result.summary), result.analyzed_data)
    difference_plot = create_difference_plot(result.analyzed_data)
    residual_plot = create_method_comparison_residual_plot(result.analyzed_data, result.summary)
    decision = str(analysis_state["decision"])
    method_interpretation = (
        "Strong agreement was observed between candidate and reference methods.\n\n"
        "All predefined acceptance criteria were satisfied.\n\n"
        "Conclusion: Method Comparison PASSED."
        if decision == "PASS"
        else (
            "Candidate and reference methods did not satisfy all predefined acceptance criteria.\n\n"
            "Review criteria details and disagreement samples before completing the study.\n\n"
            "Conclusion: Method Comparison FAILED."
        )
    )
    key_results = {
        "Agreement Rate": f"{agreement_rate:.1f}%",
        "R²": f"{float(result.summary['R²']):.4f}",
        "Correlation": f"{float(result.summary['Correlation r']):.4f}",
        "Slope": f"{float(result.summary['Slope']):.3f}",
        "Intercept": f"{float(result.summary['Intercept']):.3f}",
        "Mean Bias": f"{float(result.summary['Mean Percent Bias']):.2f}%",
    }
    figures = {
        "Regression Plot": scatter_plot,
        "Bland-Altman Plot": difference_plot,
        "Residual Plot": residual_plot,
    }
    flagged_samples = build_method_comparison_flagged_samples(
        result.analyzed_data,
        float(criteria["Sample Agreement Percent Bias Limit"]),
    )
    supporting = {
        "Analysis Summary": build_method_comparison_analysis_summary(result.summary),
        "Flagged Samples": flagged_samples,
        "Complete Dataset": result.analyzed_data,
    }
    _render_method_comparison_scientific_results(
        program_name=program_name,
        analyte_name=analyte_name,
        decision=decision,
        criteria_table=criteria_table,
        key_results=key_results,
        interpretation=method_interpretation,
        figures=figures,
        supporting_tables=supporting,
    )
    _render_execution_completion_step(
        program,
        analyte,
        study_type,
        decision,
        criteria_table,
        key_results,
        method_interpretation,
        figures,
        supporting,
    )

def render_validation_execution_workspace() -> bool:
    """Render generated-work execution views when selected."""

    if st.session_state.get("preselected_study_type"):
        return False
    query_program = st.query_params.get("execution_open_program", "")
    query_analyte = st.query_params.get("execution_open_analyte", "")
    if isinstance(query_program, list):
        query_program = query_program[0] if query_program else ""
    if isinstance(query_analyte, list):
        query_analyte = query_analyte[0] if query_analyte else ""
    if query_program and query_analyte:
        st.session_state.execution_program = str(query_program)
        st.session_state.execution_analyte = str(query_analyte)
        st.session_state.execution_mode = "analyte"
        st.session_state.pop("preselected_study_type", None)
        st.session_state.pop("review_selected_study_id", None)
        for execution_param in ("execution_open_program", "execution_open_analyte"):
            try:
                del st.query_params[execution_param]
            except KeyError:
                pass
    mode = str(st.session_state.get("execution_mode") or "")
    if mode == "analyte":
        render_analyte_validation_workspace()
        return True
    if mode == "study":
        st.session_state.execution_mode = "analyte"
        st.rerun()
    render_validation_workspace_landing()
    return True


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
    """Display the study decision with simple visual emphasis."""

    escaped_decision = escape(str(decision))
    if decision == "PASS":
        banner_class = "svap-decision-pass"
    elif decision == "INCOMPLETE":
        banner_class = "svap-decision-incomplete"
    elif decision in {"BORDERLINE", "REVIEW", "PASS WITH CAUTION"}:
        banner_class = "svap-decision-warning"
    else:
        escaped_decision = "INVESTIGATE" if decision == "INVESTIGATE" else "FAIL"
        banner_class = "svap-decision-fail"
    st.markdown(
        f'<div class="svap-decision-banner {banner_class}">{escaped_decision}</div>',
        unsafe_allow_html=True,
    )


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
    inject_validation_styles()
    render_shell_brand()

    page_options = [
        "Programs",
        "Execution",
        "Review",
        "Reports",
    ]
    legacy_pages = {
        "Dashboard": "Programs",
        "Validation Program Definition": "Programs",
        "Program Definition": "Programs",
        "Projects": "Programs",
        "Validation Workspace": "Execution",
        "Validation Execution": "Execution",
        "Validation Review Center": "Review",
        "Validation Review": "Review",
        "Reports Library": "Reports",
        "Platform Settings": "Programs",
        "Settings": "Programs",
        "Sample Datasets": "Programs",
    }
    if "pending_page" in st.session_state:
        pending_page = st.session_state.pop("pending_page")
        st.session_state.navigation_choice = legacy_pages.get(str(pending_page), str(pending_page))
    query_page = st.query_params.get("page", "")
    if isinstance(query_page, list):
        query_page = query_page[0] if query_page else ""
    query_page = legacy_pages.get(str(query_page), str(query_page))
    if query_page in page_options:
        st.session_state.navigation_choice = str(query_page)
        try:
            del st.query_params["page"]
        except KeyError:
            pass
    if "navigation_choice" not in st.session_state or st.session_state.navigation_choice not in page_options:
        st.session_state.navigation_choice = "Programs"
    page = st.sidebar.radio(
        "Navigation",
        page_options,
        key="navigation_choice",
    )
    previous_page = st.session_state.get("_last_navigation_choice")
    if page != "Review":
        st.session_state.review_selected_study_id = ""
        for review_param in ("review_open", "review_filter"):
            try:
                del st.query_params[review_param]
            except KeyError:
                pass
    if page == "Programs" and previous_page != "Programs":
        st.session_state.programs_tab = "Directory"
    st.session_state["_last_navigation_choice"] = page
    if page == "Programs":
        render_programs_page()
        st.stop()

    if page == "Review":
        render_validation_review_center()
        st.stop()

    if page == "Reports":
        render_reports_library()
        st.stop()

    if page == "Execution":
        if render_validation_execution_workspace():
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
