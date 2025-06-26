# placeholder
import pdfplumber
from docx import Document as DocxDoc

def extract_text_from_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def extract_text_from_docx(path):
    doc = DocxDoc(path)
    return "\n".join([p.text for p in doc.paragraphs])
