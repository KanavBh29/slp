"""Stamps the page border used by the course template onto every page."""
import sys
from io import BytesIO

from pypdf import PdfReader, PdfWriter
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

path = sys.argv[1]
reader = PdfReader(path)
writer = PdfWriter()
for page in reader.pages:
    w, h = float(page.mediabox.width), float(page.mediabox.height)
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(w, h))
    c.setLineWidth(1)
    c.rect(12 * mm, 12 * mm, w - 24 * mm, h - 24 * mm)
    c.save()
    buf.seek(0)
    page.merge_page(PdfReader(buf).pages[0])
    writer.add_page(page)
writer.add_metadata({"/Title": "Speech and Language Processing - Project Report"})
with open(path, "wb") as f:
    writer.write(f)
print("bordered", path)
