"""Reusable DBS validation report entry points."""

from __future__ import annotations

from pathlib import Path
import sys
import importlib.util

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

spec = importlib.util.spec_from_file_location("_sva_app_report", APP_DIR / "report.py")
if spec is None or spec.loader is None:
    raise ImportError("Unable to load app report module.")
_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _module
spec.loader.exec_module(_module)

build_dbs_executive_summary = _module.build_dbs_executive_summary
build_dbs_html_report = _module.build_dbs_html_report
build_dbs_pdf_report = _module.build_dbs_pdf_report
dbs_executive_summary_html = _module.dbs_executive_summary_html
format_dbs_criteria_table = _module.format_dbs_criteria_table
format_dbs_overall_summary = _module.format_dbs_overall_summary
format_dbs_table = _module.format_dbs_table
generate_dbs_interpretation = _module.generate_dbs_interpretation


__all__ = [
    "build_dbs_executive_summary",
    "build_dbs_html_report",
    "build_dbs_pdf_report",
    "dbs_executive_summary_html",
    "format_dbs_criteria_table",
    "format_dbs_overall_summary",
    "format_dbs_table",
    "generate_dbs_interpretation",
]
