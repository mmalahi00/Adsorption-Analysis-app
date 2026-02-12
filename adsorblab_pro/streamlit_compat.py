"""Streamlit compatibility layer.

AdsorbLab Pro uses Streamlit for the UI. At the same time, we want unit tests and
other tooling (docs builds, static analysis) to be able to import UI modules in
environments where Streamlit isn't installed.

This module attempts to import Streamlit and, if unavailable, falls back to a
minimal stub:
- Provides ``cache_data`` decorator compatible with Streamlit's use pattern.
- Raises a clear error if any other Streamlit API is accessed.

UI modules should prefer:
    from adsorblab_pro.streamlit_compat import st

and then use ``@st.cache_data`` / ``st.*`` normally.
"""

from __future__ import annotations

from typing import Any


class _StreamlitStub:
    """Very small Streamlit stand-in.

    The purpose is to let UI modules be imported without Streamlit.
    """

    def cache_data(self, *dargs: Any, **dkwargs: Any) -> Any:
        """Decorator compatible with ``st.cache_data``.

        Supports both:
            @st.cache_data
            @st.cache_data(show_spinner=False, ...)

        When Streamlit is not installed, this is an identity decorator.
        """

        # Case: used without parentheses
        if dargs and callable(dargs[0]) and len(dargs) == 1 and not dkwargs:
            return dargs[0]

        def decorator(func: Any) -> Any:
            return func

        return decorator

    def __getattr__(self, name: str) -> Any:  # pragma: no cover
        raise RuntimeError(
            "Streamlit is required to use the UI components. Install it with: pip install streamlit"
        )


st: Any
try:
    import streamlit as _st

    st = _st
    STREAMLIT_AVAILABLE = True
except Exception:  # pragma: no cover
    st = _StreamlitStub()
    STREAMLIT_AVAILABLE = False
