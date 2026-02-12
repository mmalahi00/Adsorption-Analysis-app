"""AdsorbLab Pro Streamlit entrypoint.

Streamlit reruns this file on every interaction.
We must execute the real UI script every time (avoid import caching).
"""

from __future__ import annotations

import sys
from pathlib import Path
import runpy

PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(PKG_DIR / "app_main.py"), run_name="__main__")
