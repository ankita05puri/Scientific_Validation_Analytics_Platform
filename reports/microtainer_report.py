"""Reusable Microtainer validation report entry points."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

spec = importlib.util.spec_from_file_location("_sva_app_report_microtainer", APP_DIR / "report.py")
if spec is None or spec.loader is None:
    raise ImportError("Unable to load app report module.")
_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _module
spec.loader.exec_module(_module)

format_microtainer_table = _module.format_microtainer_table
format_microtainer_criteria_table = _module.format_microtainer_criteria_table
format_microtainer_overall_summary = _module.format_microtainer_overall_summary
build_microtainer_executive_summary = _module.build_microtainer_executive_summary
build_microtainer_html_report = _module.build_microtainer_html_report
build_microtainer_pdf_report = _module.build_microtainer_pdf_report
generate_microtainer_interpretation = _module.generate_microtainer_interpretation

__all__ = [
    "format_microtainer_table",
    "format_microtainer_criteria_table",
    "format_microtainer_overall_summary",
    "build_microtainer_executive_summary",
    "build_microtainer_html_report",
    "build_microtainer_pdf_report",
    "generate_microtainer_interpretation",
]
