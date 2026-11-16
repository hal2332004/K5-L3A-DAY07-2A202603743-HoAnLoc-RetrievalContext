import pypdf
import re
from pathlib import Path

pdf_path = Path("data/Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf")
reader = pypdf.PdfReader(str(pdf_path))

figures = []
tables = []

for idx in range(24, 136):
    page_num = idx + 1
    page = reader.pages[idx]
    text = page.extract_text() or ""
    
    # Look for Figure patterns
    fig_matches = re.findall(r'(Figure\s+[12]-\d+[^.\n]*\.[^\n]*)', text, re.IGNORECASE)
    for fm in fig_matches:
        figures.append({"page": page_num, "caption": fm.strip(), "text_snippet": text[:200].replace("\n", " ")})
        
    # Look for Table patterns
    tab_matches = re.findall(r'(Table\s+[12]-\d+[^.\n]*\.[^\n]*)', text, re.IGNORECASE)
    for tm in tab_matches:
        tables.append({"page": page_num, "caption": tm.strip(), "text_snippet": text[:200].replace("\n", " ")})

print(f"Found {len(figures)} figure captions and {len(tables)} table captions across pages 25-136:")
print("\n--- TABLES ---")
for t in tables:
    print(f"Page {t['page']}: {t['caption']}")

print("\n--- FIGURES ---")
for f in figures[:25]:
    print(f"Page {f['page']}: {f['caption']}")
if len(figures) > 25:
    print(f"... and {len(figures)-25} more figures.")
