import pdfplumber

pdf_path = "data/docs/skcet_code_of_conduct.pdf"

with pdfplumber.open(pdf_path) as pdf:
    print("Total pages:", len(pdf.pages))

    for i, page in enumerate(pdf.pages[:5]):
        text = page.extract_text() or ""
        print(f"\n--- Page {i + 1} ---")
        print(text[:1500])