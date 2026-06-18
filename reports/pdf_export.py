"""PDF export helpers for consolidated validation report packages."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from fpdf import FPDF

from .executive_summary import create_executive_summary_table
from .models import ValidationPackage


def _clean(value: object) -> str:
    """Return PDF-safe text."""

    return (
        str(value)
        .replace("²", "2")
        .replace("≥", ">=")
        .replace("≤", "<=")
        .replace("±", "+/-")
        .replace("×", "x")
        .replace("–", "-")
        .replace("—", "-")
    )


class ValidationPDF(FPDF):
    """Validation report PDF with repeatable header/footer."""

    def __init__(self, report_title: str, report_version: str) -> None:
        super().__init__()
        self.report_title = _clean(report_title)
        self.report_version = _clean(report_version)

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Arial", "", 8)
        self.set_text_color(82, 96, 109)
        self.cell(0, 6, f"{self.report_title} | {self.report_version}", ln=True)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        if self.page_no() == 1:
            return
        self.set_y(-14)
        self.set_font("Arial", "", 8)
        self.set_text_color(82, 96, 109)
        self.cell(0, 8, f"Scientific Validation Analytics Platform | Page {self.page_no()} of {{nb}}", align="C")
        self.set_text_color(0, 0, 0)


def _add_table(pdf: FPDF, title: str, table: pd.DataFrame, max_rows: int = 18) -> None:
    """Add a compact table to a PDF."""

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _clean(title), ln=True)
    if table.empty:
        pdf.set_font("Arial", "", 9)
        pdf.cell(0, 6, "Not available.", ln=True)
        pdf.ln(4)
        return
    pdf.set_font("Arial", "B", 7)
    columns = list(table.columns)
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    column_width = usable_width / max(1, len(columns))
    for column in columns:
        pdf.cell(column_width, 6, _clean(column)[:24], border=1)
    pdf.ln()
    pdf.set_font("Arial", "", 7)
    for _, row in table.head(max_rows).iterrows():
        for column in columns:
            pdf.cell(column_width, 6, _clean(row[column])[:24], border=1)
        pdf.ln()
    pdf.ln(4)


def _add_paragraph(pdf: FPDF, title: str, text: object) -> None:
    """Add a titled paragraph block."""

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _clean(title), ln=True)
    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(0, 5, _clean(text))
    pdf.ln(3)


def _metadata_table(metadata: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Field": key, "Value": value or "Not specified"} for key, value in metadata.items()]
    )


def build_full_pdf(package: ValidationPackage) -> bytes:
    """Build full validation package PDF."""

    title = str(package.project_metadata.get("Validation Project Name", "Validation Report Package"))
    report_version = str(package.project_metadata.get("Report Version", "v0.7.1"))
    pdf = ValidationPDF(title, report_version)
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.ln(35)
    pdf.multi_cell(0, 10, _clean(title), align="C")
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, "Analytical Validation Report Package", ln=True, align="C")
    pdf.cell(0, 8, f"Report Version: {_clean(report_version)}", ln=True, align="C")
    pdf.cell(0, 8, f"Generated: {generated_at}", ln=True, align="C")
    pdf.ln(12)
    _add_table(pdf, "Validation Report Information", _metadata_table(package.project_metadata), max_rows=30)

    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "1. Executive Summary", ln=True)
    _add_paragraph(pdf, "Scientific Objective and Scope", package.executive_narrative)
    _add_table(pdf, "Executive Summary", create_executive_summary_table(package))
    _add_paragraph(pdf, "Decision Logic", package.decision_justification)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "2. Validation Risk Summary", ln=True)
    _add_table(pdf, "Validation Risk Summary", package.risk_summary)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "3. Validation Coverage Matrix", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 7, f"Overall validation completeness: {package.completeness_percent:.1f}%", ln=True)
    _add_table(pdf, "Coverage Matrix", package.coverage_matrix)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "4. Cross-Study Validation Matrix", ln=True)
    _add_table(pdf, "Cross-Study Validation Matrix", package.validation_matrix)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "5. Quality Assurance Section", ln=True)
    _add_table(pdf, "Quality Assurance Findings", package.qa_findings)

    for index, study in enumerate(package.studies, start=6):
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 9, f"{index}. {_clean(study.study_type)} Study Report", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 6, f"Decision: {_clean(study.decision)}")
        pdf.multi_cell(0, 6, f"Objective: {_clean(study.objective)}")
        pdf.multi_cell(0, 6, f"Design: {_clean(study.design)}")
        traceability = pd.DataFrame(
            [
                {"Element": "Study Name", "Value": study.study_name},
                {"Element": "Analysis Version", "Value": study.analysis_version},
                {"Element": "Source Dataset", "Value": study.source_dataset},
                {"Element": "Analysis Timestamp", "Value": study.analysis_timestamp},
                {"Element": "Sample Count", "Value": study.sample_count},
                {"Element": "Excluded Samples", "Value": study.excluded_samples},
            ]
        )
        _add_table(pdf, "Study Traceability", traceability)
        _add_paragraph(pdf, "Statistical Methods", study.statistical_methods)
        _add_table(pdf, "Key Findings", study.key_findings)
        _add_table(pdf, "Acceptance Criteria", study.acceptance_criteria)
        _add_paragraph(pdf, "Scientific Interpretation", study.interpretation)
        _add_paragraph(pdf, "Conclusion", study.conclusion or f"Preliminary study decision: {study.decision}.")
        for name, table in list(study.raw_outputs.items())[:3]:
            _add_table(pdf, f"Appendix - {name}", table, max_rows=10)

    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Final Validation Conclusion", ln=True)
    _add_paragraph(pdf, "Decision Justification", package.decision_justification)
    _add_paragraph(pdf, "Final Conclusion", package.final_conclusion)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Reviewer Sign-Off", ln=True)
    pdf.set_font("Arial", "", 11)
    for label in ["Prepared By:", "Reviewed By:", "Approved By:", "Approval Date:"]:
        pdf.ln(8)
        pdf.cell(0, 8, label, ln=True)
        pdf.cell(0, 8, "_" * 72, ln=True)

    output = pdf.output(dest="S")
    if isinstance(output, bytes):
        return output
    return output.encode("latin-1")


def build_executive_summary_pdf(package: ValidationPackage) -> bytes:
    """Build a compact executive summary PDF."""

    title = str(package.project_metadata.get("Validation Project Name", "Validation Report Package"))
    report_version = str(package.project_metadata.get("Report Version", "v0.7.1"))
    pdf = ValidationPDF("Validation Executive Summary", report_version)
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "Validation Executive Summary", ln=True, align="C")
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7, _clean(title), align="C")
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, f"Overall Decision: {_clean(package.overall_status)}", ln=True)
    _add_paragraph(pdf, "Executive Narrative", package.executive_narrative)
    _add_table(pdf, "Executive Summary", create_executive_summary_table(package))
    _add_table(pdf, "Risk Summary", package.risk_summary, max_rows=8)
    _add_table(pdf, "Validation Matrix", package.validation_matrix)
    _add_paragraph(pdf, "Final Conclusion", package.final_conclusion)
    output = pdf.output(dest="S")
    if isinstance(output, bytes):
        return output
    return output.encode("latin-1")
