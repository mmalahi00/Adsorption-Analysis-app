"""Backward-compatible Streamlit launcher (repo root).

Run:
    streamlit run adsorption_app.py

We must execute the real app on EVERY Streamlit rerun.
Using runpy avoids import caching and avoids importlib.reload side effects.
"""

from __future__ import annotations

import sys
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "adsorblab_pro" / "app_main.py"), run_name="__main__")
