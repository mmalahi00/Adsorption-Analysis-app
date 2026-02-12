# adsorblab_pro/__main__.py
"""
Enable running the package as a module: python -m adsorblab_pro

This provides an alternative way to launch the Streamlit app:
    python -m adsorblab_pro [streamlit args...]

Equivalent to:
    adsorblab [streamlit args...]
"""

from adsorblab_pro import main

if __name__ == "__main__":
    main()
