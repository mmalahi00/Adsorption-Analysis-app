# tabs/__init__.py
"""
AdsorbLab Pro v2.0.0 - Tab Modules (Production Edition)
=======================================================

Publication-ready analysis modules with:
- Confidence intervals and bootstrap resampling
- Residual diagnostics
- Model comparison statistics

Version: 2.0.0

ARCHITECTURE: True Lazy Loading at Render Time
----------------------------------------------
This package implements genuine lazy loading via two mechanisms:

1. PEP 562 __getattr__: Provides deferred module resolution when accessing
   `tabs.foo` - the module is only loaded when first accessed.

2. Render-time imports (adsorption_app.py): The main app uses `_lazy_render()`
   which imports tab modules only when their content is actually rendered,
   NOT at application startup. This ensures:

   - Graceful degradation when optional dependencies are unavailable
   - Each tab can fail independently without breaking the entire app

Usage in adsorption_app.py:
    from tabs import home_tab  # Only immediate import (lightweight)

    with some_tab:
        _lazy_render("isotherm_tab")  # Imports only when tab renders
"""

import importlib
from typing import Any

__all__ = [
    "home_tab",
    "calibration_tab",
    "isotherm_tab",
    "kinetic_tab",
    "dosage_tab",
    "ph_effect_tab",
    "temperature_tab",
    "thermodynamics_tab",
    "threed_explorer_tab",
    "statistical_summary_tab",
    "comparison_tab",
    "report_tab",
    "competitive_tab",
]


def __getattr__(name: str) -> Any:
    """Lazy-load tab modules on first access."""
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    """List available modules for IDE autocompletion."""
    return __all__
