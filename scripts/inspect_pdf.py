import pypdf
from pathlib import Path

pdf_path = Path("data/Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf")
reader = pypdf.PdfReader(str(pdf_path))
print("File exists:", pdf_path.exists())
print("Total pages in PDF:", len(reader.pages))

# Check images and tables across pages 25 to 136
image_count = 0
for idx in range(24, min(136, len(reader.pages))):
    page = reader.pages[idx]
    if len(page.images) > 0:
        image_count += len(page.images)

print(f"Total images found in Ch1-2 (pages 25-136): {image_count}")
