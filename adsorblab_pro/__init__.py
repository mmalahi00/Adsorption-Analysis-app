# adsorblab_pro/__init__.py
"""
AdsorbLab Pro - Advanced Adsorption Data Analysis Platform
==========================================================

A comprehensive Python package for analyzing adsorption experiments with:
- Advanced statistical analysis with confidence intervals
- Bootstrap parameter estimation
- Comprehensive model comparison (R², Adj-R², AIC, AICc, BIC)
- Residual diagnostics and outlier detection

Usage:
    # Run the Streamlit app
    adsorblab  # CLI command after pip install

    # Or via Python
    python -m adsorblab_pro

    # Or programmatic access
    from adsorblab_pro import models, utils, validation

Example:
    >>> from adsorblab_pro.models import langmuir_model, pso_model
    >>> from adsorblab_pro.utils import calculate_adsorption_capacity, bootstrap_confidence_intervals
"""


def _read_version() -> str:
    """
    Resolve the package version, preferring installed distribution metadata.

    Source checkouts — including Streamlit Community Cloud, which runs the repo
    directly rather than installing it — have no distribution metadata, so the
    version is read from ``pyproject.toml`` instead of being hardcoded here.
    Hardcoding it created a second copy that silently went stale after a release
    bump, and a stale version in the app footer is worse than an obviously
    unreleased one, because it is indistinguishable from a correct one.
    """
    import re
    from importlib.metadata import PackageNotFoundError, version
    from pathlib import Path

    try:
        return version("adsorblab-pro")
    except PackageNotFoundError:
        pass

    # Read the [project] version line directly rather than pulling in a TOML
    # parser: tomllib is 3.11+, and adding a tomli dependency for one fallback
    # path is not worth it.
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return "unknown"

    match = re.search(r'^\s*version\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    # Neither installed nor next to a readable pyproject.toml: say so rather
    # than assert a version number that may be wrong.
    return match.group(1) if match else "unknown"


__version__ = _read_version()
__author__ = "Mohamed EL MALLAHI"
__license__ = "MIT"

# Public API - lazy imports for fast startup
__all__ = [
    "__version__",
    "main",
    "models",
    "utils",
    "validation",
    "config",
]


def main() -> None:
    """
    Launch the AdsorbLab Pro Streamlit application.

    This is the entry point for the `adsorblab` console script.
    Can also be called programmatically.

    Example:
        >>> import adsorblab_pro
        >>> adsorblab_pro.main()  # Launches Streamlit app
    """
    import subprocess
    import sys
    from pathlib import Path

    # Get the path to the Streamlit entrypoint within the installed package
    app_path = Path(__file__).parent / "app.py"

    # Launch Streamlit with the app
    sys.exit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.headless=true",
                "--browser.gatherUsageStats=false",
                *sys.argv[1:],  # Pass through any additional arguments
            ]
        )
    )


import types as types_module


def _lazy_import(name: str) -> types_module.ModuleType:
    """Lazy import submodules for faster startup."""
    import importlib

    return importlib.import_module(f".{name}", __package__)


def __getattr__(name: str) -> types_module.ModuleType:
    """Enable lazy loading of submodules."""
    if name in ("models", "utils", "validation", "config"):
        return _lazy_import(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
