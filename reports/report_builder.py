"""HTML report assembly for consolidated validation packages."""

from __future__ import annotations

from datetime import datetime
from html import escape

import pandas as pd

from .executive_summary import create_executive_summary_table, study_counts
from .models import StudyReport, ValidationPackage
from .validation_matrix import coverage_table


def slugify(value: str) -> str:
    """Create stable HTML anchors for report navigation."""

    return (
        str(value)
        .strip()
        .lower()
        .replace("&", "and")
        .replace("/", "-")
        .replace(" ", "-")
    )


def dataframe_to_html(table: pd.DataFrame) -> str:
    """Render a DataFrame to HTML with a stable empty-state fallback."""

    if table is None or table.empty:
        return "<p>Not available.</p>"
    return table.to_html(index=False, escape=True)


def metadata_table(metadata: dict[str, object]) -> str:
    """Render metadata key-value pairs."""

    rows = [
        {"Field": key, "Value": value if value not in (None, "") else "Not specified"}
        for key, value in metadata.items()
    ]
    return dataframe_to_html(pd.DataFrame(rows))


def status_badge(status: str) -> str:
    """Render a colored status badge."""

    normalized = str(status).strip().upper()
    css_class = (
        "pass"
        if normalized == "PASS"
        else "borderline"
        if normalized in {"BORDERLINE", "PASS WITH CAUTION", "REVIEW"}
        else "fail"
    )
    return f'<span class="status-badge status-{css_class}">{escape(normalized)}</span>'


def included_study_cards(studies: list[StudyReport]) -> str:
    """Render included study cards."""

    cards = []
    for study in studies:
        cards.append(
            f"""
            <div class="summary-card">
              <div class="summary-label">{escape(study.study_type)}</div>
              <div class="summary-value">{status_badge(study.decision)}</div>
              <div class="summary-subtext">{escape(study.study_name)}<br>{escape(study.date)}<br>Lifecycle: {escape(study.status)}</div>
            </div>
            """
        )
    return '<div class="summary-grid">' + "".join(cards) + "</div>"


def render_visualizations(visualizations: dict[str, str]) -> str:
    """Render figure HTML blocks."""

    if not visualizations:
        return "<p>No visualizations available.</p>"
    return "".join(
        f"<h4>{escape(title)}</h4>{figure_html}"
        for title, figure_html in visualizations.items()
    )


def render_badged_dataframe(table: pd.DataFrame) -> str:
    """Render DataFrame with badge formatting for status-like columns."""

    if table is None or table.empty:
        return "<p>Not available.</p>"
    status_columns = {
        column
        for column in table.columns
        if any(token in str(column).lower() for token in ["status", "decision", "borderline"])
    }
    rows = []
    for _, row in table.iterrows():
        cells = []
        for column in table.columns:
            value = row[column]
            if column in status_columns and str(value).strip().upper() in {
                "PASS",
                "FAIL",
                "BORDERLINE",
                "PASS WITH CAUTION",
                "REVIEW",
            }:
                cells.append(f"<td>{status_badge(str(value))}</td>")
            else:
                cells.append(f"<td>{escape(str(value))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "".join(f"<th>{escape(str(column))}</th>" for column in table.columns)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def traceability_table(study: StudyReport) -> str:
    """Render audit-trail and traceability metadata for one study."""

    criteria_text = "; ".join(study.acceptance_criteria.astype(str).agg(" | ".join, axis=1).tolist())
    rows = [
        {"Traceability Element": "Study Name", "Value": study.study_name},
        {"Traceability Element": "Lifecycle Status", "Value": study.status},
        {"Traceability Element": "Analysis Version", "Value": study.analysis_version},
        {"Traceability Element": "Source Dataset", "Value": study.source_dataset},
        {"Traceability Element": "Analysis Timestamp", "Value": study.analysis_timestamp},
        {"Traceability Element": "Number of Samples / Records", "Value": study.sample_count},
        {"Traceability Element": "Number of Excluded Samples / Records", "Value": study.excluded_samples},
        {"Traceability Element": "User-Defined Acceptance Criteria", "Value": criteria_text or "Not specified"},
    ]
    return dataframe_to_html(pd.DataFrame(rows))


def render_study_section(study: StudyReport) -> str:
    """Render one individual study section."""

    interpretation = "<br><br>".join(
        escape(paragraph) for paragraph in str(study.interpretation).split("\n\n")
    )
    raw_outputs = "".join(
        f"<h4>{escape(name)}</h4>{dataframe_to_html(table)}"
        for name, table in study.raw_outputs.items()
    )
    anchor = slugify(study.study_type)
    return f"""
    <section class="page-break report-section" id="study-{anchor}">
      <h2>{escape(study.study_type)} Study Report</h2>
      <h3>Study Overview</h3>
      {metadata_table(study.metadata)}
      <h3>Study Traceability</h3>
      {traceability_table(study)}
      <h3>Objective</h3>
      <p>{escape(study.objective)}</p>
      <h3>Design</h3>
      <p>{escape(study.design)}</p>
      <h3>Acceptance Criteria</h3>
      {render_badged_dataframe(study.acceptance_criteria)}
      <h3>Statistical Methods</h3>
      <p>{escape(study.statistical_methods or "Standardized module-specific statistical methods were applied.")}</p>
      <h3>Key Findings</h3>
      {dataframe_to_html(study.key_findings)}
      <h3>Acceptance Criteria Assessment</h3>
      {render_badged_dataframe(study.acceptance_criteria)}
      <h3>Scientific Interpretation</h3>
      <p class="note">{interpretation}</p>
      <h3>Conclusion</h3>
      <p>{escape(study.conclusion or f"Preliminary study decision: {study.decision}.")}</p>
      <h3>Visualizations</h3>
      {render_visualizations(study.visualizations)}
      <h3>Technical Appendix</h3>
      {raw_outputs}
    </section>
    """


def build_study_section_html(study: StudyReport) -> str:
    """Build a standalone HTML document for one study section."""

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(study.study_type)} Study Report</title>
  <style>
    body {{ color:#1f2933; font-family:Arial,sans-serif; line-height:1.5; margin:40px; }}
    h1,h2,h3,h4 {{ color:#102a43; }}
    table {{ border-collapse:collapse; margin:12px 0 28px; width:100%; }}
    th,td {{ border:1px solid #d9e2ec; padding:8px; text-align:right; }}
    th {{ background:#f0f4f8; }}
    td:first-child,th:first-child {{ text-align:left; }}
    .note {{ background:#f7f9fb; border-left:4px solid #2a6f97; padding:12px 16px; }}
    .status-badge {{ border-radius:999px; display:inline-block; font-size:.78rem; font-weight:bold; padding:6px 10px; }}
    .status-pass {{ background:#e3f9e5; color:#1f7a1f; }}
    .status-borderline {{ background:#fff8c5; color:#946200; }}
    .status-fail {{ background:#ffe3e3; color:#c92a2a; }}
  </style>
</head>
<body>
  <h1>{escape(study.study_type)} Study Report</h1>
  <p><strong>Generated:</strong> {generated_at}</p>
  {render_study_section(study)}
</body>
</html>
"""


def build_full_report_html(
    package: ValidationPackage,
    supported_studies: tuple[str, ...],
) -> str:
    """Build full validation package HTML."""

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    counts = study_counts(package.studies)
    coverage = package.coverage_matrix if not package.coverage_matrix.empty else coverage_table(package.studies, supported_studies)
    toc_items = "".join(
        f'<li><a href="#study-{slugify(study.study_type)}">{escape(study.study_type)} Study Report</a></li>'
        for study in package.studies
    )
    study_sections = "".join(render_study_section(study) for study in package.studies)
    report_version = package.project_metadata.get("Report Version", "v1.0.0")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Validation Report Package</title>
  <link rel="stylesheet" href="validation_report.css">
  <style>
    @page {{ margin: 0.75in; }}
    body {{ color:#1f2933; font-family:Arial,sans-serif; line-height:1.5; margin:40px; }}
    .report-header,.report-footer {{ background:#fff; color:#52606d; font-size:.78rem; left:0; right:0; padding:8px 40px; position:fixed; }}
    .report-header {{ border-bottom:1px solid #d9e2ec; top:0; }}
    .report-footer {{ border-top:1px solid #d9e2ec; bottom:0; }}
    .content {{ margin-top:52px; }}
    .cover-page {{ min-height:78vh; display:flex; flex-direction:column; justify-content:center; text-align:center; }}
    .cover-title {{ color:#102a43; font-size:2rem; font-weight:700; margin-bottom:12px; }}
    .cover-subtitle {{ color:#52606d; font-size:1rem; margin:4px 0; }}
    a {{ color:#2a6f97; text-decoration:none; }}
    h1,h2,h3,h4 {{ color:#102a43; }}
    table {{ border-collapse:collapse; margin:12px 0 28px; width:100%; }}
    th,td {{ border:1px solid #d9e2ec; padding:8px; text-align:right; }}
    th {{ background:#f0f4f8; }}
    td:first-child,th:first-child {{ text-align:left; }}
    .summary-grid {{ display:grid; gap:12px; grid-template-columns:repeat(3,minmax(0,1fr)); margin:14px 0 28px; }}
    .summary-card {{ border:1px solid #d9e2ec; border-radius:8px; padding:14px 16px; background:#fff; }}
    .summary-label {{ color:#52606d; font-size:.78rem; font-weight:bold; text-transform:uppercase; }}
    .summary-value {{ color:#102a43; font-size:1.25rem; font-weight:bold; margin-top:8px; }}
    .summary-subtext {{ color:#52606d; font-size:.9rem; margin-top:8px; }}
    .note {{ background:#f7f9fb; border-left:4px solid #2a6f97; padding:12px 16px; }}
    .status-badge {{ border-radius:999px; display:inline-block; font-size:.78rem; font-weight:bold; padding:6px 10px; }}
    .status-pass {{ background:#e3f9e5; color:#1f7a1f; }}
    .status-borderline {{ background:#fff8c5; color:#946200; }}
    .status-fail {{ background:#ffe3e3; color:#c92a2a; }}
    .page-break {{ page-break-before:always; }}
    .signature-grid {{ display:grid; gap:28px; grid-template-columns:1fr; margin-top:24px; }}
    .signature-line {{ border-bottom:1px solid #52606d; height:32px; margin-bottom:4px; }}
    @media print {{
      body {{ margin:0.75in; }}
      .report-header,.report-footer {{ position:fixed; }}
      .page-break {{ break-before:page; }}
    }}
  </style>
</head>
<body>
  <div class="report-header">{escape(str(package.project_metadata.get("Validation Project Name", "Validation Report Package")))} | {escape(str(report_version))}</div>
  <div class="report-footer">Scientific Validation Analytics Platform | Generated {generated_at}</div>
  <main class="content">
  <section class="cover-page" id="cover-page">
    <div class="cover-title">{escape(str(package.project_metadata.get("Validation Project Name", "Validation Report Package")))}</div>
    <div class="cover-subtitle">Analytical Validation Report Package</div>
    <div class="cover-subtitle">Assay / Biomarker: {escape(str(package.project_metadata.get("Assay / Biomarker", "Not specified")))}</div>
    <div class="cover-subtitle">Study Date: {escape(str(package.project_metadata.get("Study Date", "Not specified")))}</div>
    <div class="cover-subtitle">Analyst: {escape(str(package.project_metadata.get("Analyst", "Not specified") or "Not specified"))}</div>
    <div class="cover-subtitle">Reviewer: {escape(str(package.project_metadata.get("Reviewer", "Not specified") or "Not specified"))}</div>
    <div class="cover-subtitle">Laboratory: {escape(str(package.project_metadata.get("Laboratory Name", "Not specified") or "Not specified"))}</div>
    <div class="cover-subtitle">Protocol Number: {escape(str(package.project_metadata.get("Protocol Number", "Not specified") or "Not specified"))}</div>
    <div class="cover-subtitle">Report Version: {escape(str(report_version))}</div>
  </section>
  <section class="page-break" id="report-information">
  <h2>Report Information</h2>
  {metadata_table(package.project_metadata)}
  </section>
  <section class="page-break" id="table-of-contents">
  <h2>Table of Contents</h2>
  <ol>
    <li><a href="#executive-summary">Executive Summary</a></li>
    <li><a href="#risk-summary">Validation Risk Summary</a></li>
    <li><a href="#coverage-matrix">Validation Coverage Matrix</a></li>
    <li><a href="#validation-matrix">Cross-Study Validation Matrix</a></li>
    <li><a href="#qa-section">Quality Assurance Section</a></li>
    {toc_items}
    <li><a href="#final-conclusion">Final Validation Conclusion</a></li>
    <li><a href="#sign-off">Reviewer Sign-Off</a></li>
  </ol>
  </section>
  <section class="page-break" id="executive-summary">
  <h2>Executive Summary</h2>
  <p class="note">{escape(package.executive_narrative)}</p>
  {dataframe_to_html(create_executive_summary_table(package))}
  <p><strong>Overall Validation Status:</strong> {status_badge(package.overall_status)}</p>
  <p>Studies completed: {counts["Number completed"]}; passed: {counts["Number passed"]}; borderline: {counts["Number borderline"]}; failed: {counts["Number failed"]}.</p>
  <h3>Included Studies</h3>
  {included_study_cards(package.studies)}
  </section>
  <section class="page-break" id="risk-summary">
  <h2>Validation Risk Summary</h2>
  {render_badged_dataframe(package.risk_summary)}
  </section>
  <section class="page-break" id="coverage-matrix">
  <h2>Validation Coverage Matrix</h2>
  <p><strong>Overall validation completeness:</strong> {package.completeness_percent:.1f}%</p>
  {dataframe_to_html(coverage)}
  </section>
  <section class="page-break" id="validation-matrix">
  <h2>Cross-Study Validation Summary</h2>
  {render_badged_dataframe(package.validation_matrix)}
  </section>
  <section class="page-break" id="qa-section">
  <h2>Quality Assurance Section</h2>
  {render_badged_dataframe(package.qa_findings)}
  </section>
  {study_sections}
  <section class="page-break" id="final-conclusion">
    <h2>Final Validation Conclusion</h2>
    <h3>Decision Logic</h3>
    <p>{escape(package.decision_justification)}</p>
    <p class="note">{escape(package.final_conclusion)}</p>
  </section>
  <section class="page-break" id="sign-off">
    <h2>Reviewer Sign-Off</h2>
    <div class="signature-grid">
      <div><div class="signature-line"></div><strong>Prepared By:</strong> {escape(str(package.project_metadata.get("Package Generated By", package.project_metadata.get("Analyst", "")) or ""))}</div>
      <div><div class="signature-line"></div><strong>Reviewed By:</strong> {escape(str(package.project_metadata.get("Reviewer", "") or ""))}</div>
      <div><div class="signature-line"></div><strong>Approved By:</strong></div>
      <div><div class="signature-line"></div><strong>Approval Date:</strong></div>
    </div>
  </section>
  </main>
</body>
</html>
"""


def build_executive_summary_html(package: ValidationPackage) -> str:
    """Build compact executive summary HTML."""

    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Executive Summary</title></head>
<body>
  <h1>Executive Summary</h1>
  <p>{escape(package.executive_narrative)}</p>
  {dataframe_to_html(create_executive_summary_table(package))}
  <h2>Validation Risk Summary</h2>
  {render_badged_dataframe(package.risk_summary)}
  <h2>Validation Matrix</h2>
  {render_badged_dataframe(package.validation_matrix)}
  <h2>Final Conclusion</h2>
  <p>{escape(package.decision_justification)}</p>
  <p>{escape(package.final_conclusion)}</p>
</body>
</html>
"""
