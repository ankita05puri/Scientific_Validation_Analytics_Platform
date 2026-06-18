"""Reusable Microtainer validation Plotly figure entry points."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

spec = importlib.util.spec_from_file_location("_sva_app_plots_microtainer", APP_DIR / "plots.py")
if spec is None or spec.loader is None:
    raise ImportError("Unable to load app plots module.")
_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _module
spec.loader.exec_module(_module)

create_microtainer_scatter_plot = _module.create_microtainer_scatter_plot
create_microtainer_bland_altman_plot = _module.create_microtainer_bland_altman_plot
create_microtainer_recovery_plot = _module.create_microtainer_recovery_plot
create_microtainer_percent_bias_plot = _module.create_microtainer_percent_bias_plot
create_microtainer_distribution_comparison = _module.create_microtainer_distribution_comparison

__all__ = [
    "create_microtainer_scatter_plot",
    "create_microtainer_bland_altman_plot",
    "create_microtainer_recovery_plot",
    "create_microtainer_percent_bias_plot",
    "create_microtainer_distribution_comparison",
]
