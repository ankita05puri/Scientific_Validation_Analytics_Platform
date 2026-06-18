"""HTML export helpers for validation report packages."""

from __future__ import annotations

from .models import ValidationPackage
from .report_builder import build_executive_summary_html, build_full_report_html


def export_full_html(package: ValidationPackage, supported_studies: tuple[str, ...]) -> str:
    """Export full validation package as HTML."""

    return build_full_report_html(package, supported_studies)


def export_executive_html(package: ValidationPackage) -> str:
    """Export executive summary as HTML."""

    return build_executive_summary_html(package)

