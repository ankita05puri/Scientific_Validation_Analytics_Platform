"""Plot wrappers for scientific validation modules.

This package re-exports the current Streamlit plotting implementations so
package-style imports and legacy ``app/plots.py`` imports both remain usable.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

spec = importlib.util.spec_from_file_location("_sva_app_plots_package", APP_DIR / "plots.py")
if spec is None or spec.loader is None:
    raise ImportError("Unable to load app plots module.")
_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _module
spec.loader.exec_module(_module)

for _name in dir(_module):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_module, _name)

__all__ = [_name for _name in globals() if _name.startswith("create_")]
