# report_generator.py
from fpdf import FPDF
from datetime import datetime
import os
from PIL import Image
import io

class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 14)
        self.cell(0, 10, "Rapport d'Analyse d'Adsorption", border=False, ln=1, align='C')
        self.set_font("Arial", '', 10)
        self.cell(0, 10, f"Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align='C')
        self.ln(5)

    def section_title(self, title):
        self.set_font("Arial", 'B', 12)
        self.set_text_color(0)
        self.cell(0, 10, title, ln=True)
        self.set_draw_color(100, 100, 100)
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(4)

    def add_table(self, dataframe, col_widths=None):
        self.set_font("Arial", '', 10)
        cols = dataframe.columns.tolist()
        if col_widths is None:
            col_widths = [190 / len(cols)] * len(cols)

        for i, col in enumerate(cols):
            self.cell(col_widths[i], 8, col, 1, 0, 'C')
        self.ln()

        for _, row in dataframe.iterrows():
            for i, col in enumerate(cols):
                self.cell(col_widths[i], 8, str(round(row[col], 4)), 1)
            self.ln()

        self.ln(4)

    def add_image_from_fig(self, fig, width=170):
        img_bytes = io.BytesIO()
        fig.write_image(img_bytes, format="png", width=1000, height=800, scale=2)
        img_bytes.seek(0)
        self.image(img_bytes, w=width, type='PNG')       
        self.ln(5)
