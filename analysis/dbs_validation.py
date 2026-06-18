"""Reusable DBS validation analysis entry points.

The Streamlit application currently hosts the production implementations in
``app.analysis``. This module provides the requested architecture path for
future package-level reuse without duplicating calculations.
"""

from __future__ import annotations

from pathlib import Path
import sys
import importlib.util

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

spec = importlib.util.spec_from_file_location("_sva_app_analysis", APP_DIR / "analysis.py")
if spec is None or spec.loader is None:
    raise ImportError("Unable to load app analysis module.")
_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _module
spec.loader.exec_module(_module)

DBSValidationResult = _module.DBSValidationResult
calculate_dbs_validation_summary = _module.calculate_dbs_validation_summary
calculate_dbs_enhancement_assessments = _module.calculate_dbs_enhancement_assessments
evaluate_dbs_criteria = _module.evaluate_dbs_criteria
prepare_dbs_validation_data = _module.prepare_dbs_validation_data
run_dbs_validation_study = _module.run_dbs_validation_study


__all__ = [
    "DBSValidationResult",
    "calculate_dbs_validation_summary",
    "calculate_dbs_enhancement_assessments",
    "evaluate_dbs_criteria",
    "prepare_dbs_validation_data",
    "run_dbs_validation_study",
]
