"""Interfaces for future raw laboratory workbook ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd


@dataclass
class ParsedWorkbook:
    """Structured representation of workbook sections used by validation workflows."""

    source_path: Path | None = None
    plate_layout: pd.DataFrame | None = None
    controls: pd.DataFrame | None = None
    patient_results: pd.DataFrame | None = None
    dbs_corrections: pd.DataFrame | None = None
    grading: pd.DataFrame | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class WorkbookParser(Protocol):
    """Parser contract for laboratory workbook formats."""

    parser_name: str

    def parse(self, workbook_path: Path) -> ParsedWorkbook:
        """Parse a workbook into structured validation sections."""
        ...
