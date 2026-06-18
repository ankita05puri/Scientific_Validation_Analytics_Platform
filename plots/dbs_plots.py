"""Reusable DBS validation Plotly figure entry points."""

from __future__ import annotations

from pathlib import Path
import sys
import importlib.util

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

spec = importlib.util.spec_from_file_location("_sva_app_plots", APP_DIR / "plots.py")
if spec is None or spec.loader is None:
    raise ImportError("Unable to load app plots module.")
_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _module
spec.loader.exec_module(_module)

create_dbs_bland_altman_plot = _module.create_dbs_bland_altman_plot
create_dbs_delay_bias_plot = _module.create_dbs_delay_bias_plot
create_dbs_delay_category_bias_plot = _module.create_dbs_delay_category_bias_plot
create_dbs_delay_distribution_plot = _module.create_dbs_delay_distribution_plot
create_dbs_distribution_comparison = _module.create_dbs_distribution_comparison
create_dbs_hematocrit_bias_plot = _module.create_dbs_hematocrit_bias_plot
create_dbs_hematocrit_percent_bias_plot = _module.create_dbs_hematocrit_percent_bias_plot
create_dbs_instrument_bias_plot = _module.create_dbs_instrument_bias_plot
create_dbs_instrument_recovery_plot = _module.create_dbs_instrument_recovery_plot
create_dbs_percent_bias_plot = _module.create_dbs_percent_bias_plot
create_dbs_recovery_plot = _module.create_dbs_recovery_plot
create_dbs_scatter_plot = _module.create_dbs_scatter_plot


__all__ = [
    "create_dbs_bland_altman_plot",
    "create_dbs_delay_bias_plot",
    "create_dbs_delay_category_bias_plot",
    "create_dbs_delay_distribution_plot",
    "create_dbs_distribution_comparison",
    "create_dbs_hematocrit_bias_plot",
    "create_dbs_hematocrit_percent_bias_plot",
    "create_dbs_instrument_bias_plot",
    "create_dbs_instrument_recovery_plot",
    "create_dbs_percent_bias_plot",
    "create_dbs_recovery_plot",
    "create_dbs_scatter_plot",
]
