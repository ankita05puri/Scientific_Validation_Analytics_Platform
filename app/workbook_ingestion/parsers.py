"""Placeholder parser implementations for future laboratory workbook ingestion."""

from __future__ import annotations

from pathlib import Path

from workbook_ingestion.interfaces import ParsedWorkbook


class LaboratoryWorkbookParser:
    """Placeholder parser for raw laboratory Excel workbooks.

    Future implementations should map workbook tabs such as Plate Layout,
    Controls, Patient Results, DBS Corrections, and Grading into a ParsedWorkbook.
    """

    parser_name = "Generic Laboratory Workbook Parser"

    def parse(self, workbook_path: Path) -> ParsedWorkbook:
        """Return an empty parsed workbook shell until parsing rules are defined."""

        return ParsedWorkbook(
            source_path=workbook_path,
            metadata={
                "parser": self.parser_name,
                "status": "placeholder",
                "supported_sections": [
                    "Plate Layout",
                    "Controls",
                    "Patient Results",
                    "DBS Corrections",
                    "Grading",
                ],
            },
        )
