"""Reusable Microtainer validation analysis entry points."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

spec = importlib.util.spec_from_file_location("_sva_app_analysis_microtainer", APP_DIR / "analysis.py")
if spec is None or spec.loader is None:
    raise ImportError("Unable to load app analysis module.")
_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _module
spec.loader.exec_module(_module)

MicrotainerValidationResult = _module.MicrotainerValidationResult
prepare_microtainer_validation_data = _module.prepare_microtainer_validation_data
calculate_microtainer_validation_summary = _module.calculate_microtainer_validation_summary
calculate_microtainer_assessments = _module.calculate_microtainer_assessments
evaluate_microtainer_criteria = _module.evaluate_microtainer_criteria
run_microtainer_validation_study = _module.run_microtainer_validation_study

__all__ = [
    "MicrotainerValidationResult",
    "prepare_microtainer_validation_data",
    "calculate_microtainer_validation_summary",
    "calculate_microtainer_assessments",
    "evaluate_microtainer_criteria",
    "run_microtainer_validation_study",
]
