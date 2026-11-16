import pypdf
from pathlib import Path

pdf_path = Path("data/Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf")
reader = pypdf.PdfReader(str(pdf_path))

for p in [88, 93, 94, 98, 102, 106, 107, 109, 110, 115, 120, 122, 125, 130]:
    txt = reader.pages[p-1].extract_text() or ""
    print(f"\n{'='*30} PAGE {p} {'='*30}")
    print(txt[:800])
