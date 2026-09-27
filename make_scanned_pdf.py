import pymupdf
import io
from PIL import Image

INPUT_PDF  = "studentHandbook.pdf"
OUTPUT_PDF = "studentHandbook_scanned.pdf"
DPI        = 150

src = pymupdf.open(INPUT_PDF)
out = pymupdf.open()

for i, page in enumerate(src):
    pix = page.get_pixmap(dpi=DPI)

    # pixmap -> PIL image -> jpeg bytes
    img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf  = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    jpeg = buf.getvalue()

    # create a blank pdf page the same size, insert the jpeg
    out_page = out.new_page(width=pix.width, height=pix.height)
    out_page.insert_image(out_page.rect, stream=jpeg)
    print(f"page {i+1}/{len(src)}")

out.save(OUTPUT_PDF, garbage=4, deflate=True)
print(f"saved to {OUTPUT_PDF}")
