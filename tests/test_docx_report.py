# tests/test_docx_report.py
"""Tests for DOCX report generation."""

from io import BytesIO

import pandas as pd
import pytest
from PIL import Image

from adsorblab_pro.docx_report import DOCX_AVAILABLE, DocxReportConfig, create_docx_report


@pytest.mark.skipif(not DOCX_AVAILABLE, reason="python-docx not installed")
def test_create_docx_report_produces_valid_docx():
    # Dummy PNG bytes (no kaleido dependency)
    img = Image.new("RGB", (120, 80))
    bio = BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    png_bytes = bio.getvalue()

    def fig_gen(fig_id, _state):
        assert fig_id == "fig_demo"
        return png_bytes

    def tbl_gen(tbl_id, _state):
        assert tbl_id == "tbl_demo"
        return pd.DataFrame({"A": [1, 2], "B": [3.14, 2.72]})

    docx_bytes, warnings = create_docx_report(
        study_title="AdsorbLab Pro — Unit Test Report",
        study_state={"isotherm_models_fitted": {}},
        selected_figures=["fig_demo"],
        selected_tables=["tbl_demo"],
        figure_generator=fig_gen,
        table_generator=tbl_gen,
        figure_meta={"fig_demo": ("Demo figure", "Synthetic image for testing")},
        table_meta={"tbl_demo": ("Demo table", "Synthetic table for testing")},
        config=DocxReportConfig(max_table_rows=10, max_table_cols=10),
    )

    assert isinstance(docx_bytes, bytes | bytearray)
    assert len(docx_bytes) > 1000
    assert isinstance(warnings, list)

    # Verify the resulting file opens as a Word document
    from docx import Document  # type: ignore

    doc = Document(BytesIO(docx_bytes))
    full_text = "\n".join(p.text for p in doc.paragraphs)
    assert "Unit Test Report" in full_text
    assert "Study overview" in full_text
    assert "Figures" in full_text
    assert "Tables" in full_text
