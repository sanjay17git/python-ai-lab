
import pdfplumber
from pathlib import Path

pdf_path = Path("data/docs/skcet_code_of_conduct.pdf")
txt_path = Path("data/docs/skcet_code_of_conduct.txt")

with pdfplumber.open(pdf_path) as pdf:
    text = "\n\n".join(
        page.extract_text() or ""
        for page in pdf.pages
    )

if not text.strip():
    raise ValueError("No readable text found in the PDF.")

txt_path.write_text(text, encoding="utf-8")

print("PDF converted successfully!")
print("Total characters:", len(text))
print("Preview:")
print(text[:500])
