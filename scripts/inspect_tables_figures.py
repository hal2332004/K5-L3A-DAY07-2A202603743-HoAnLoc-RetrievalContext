import pypdf
from pathlib import Path

pdf_path = Path("data/Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf")
reader = pypdf.PdfReader(str(pdf_path))

pages_to_inspect = [31, 41, 42, 76, 81, 88, 93, 106, 109, 27, 29, 37, 39, 43, 45, 58, 62]

for p in pages_to_inspect:
    txt = reader.pages[p-1].extract_text() or ""
    print(f"\n{'='*30} PAGE {p} {'='*30}")
    print(txt[:1000])
