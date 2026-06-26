# OpenCortex
# src/ingestion/pdf.py — PDF text and embedded-image extraction.
# Uses PyMuPDF (fitz) to walk each page in reading order, collecting text
# spans and visible images, then passes images through the vision pipeline.

import fitz

from src.ingestion.image import process_image_vision
from utils.logger import setup_logger

logger = setup_logger("pdf_extractor")


def extract_pdf_text_and_images(file_bytes, username):
    """
    Extract all textual and visual content from a PDF.

    Strategy per page:
      1. Group content into (y, x, type, data) items, sorted top-to-bottom.
      2. Text blocks are appended directly.
      3. Embedded images are sent to the vision model with a positional tag.
      4. Pages with zero text blocks are treated as scanned images.
    """
    combined_text = ""
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    for page_index, page in enumerate(doc):
        combined_text += f"\n--- Page {page_index + 1} ---\n"

        blocks = page.get_text("dict", sort=True)["blocks"]
        text_blocks = [b for b in blocks if b["type"] == 0]

        # Scanned page — rasterise and send the whole page to the vision model
        if not text_blocks:
            pix = page.get_pixmap(dpi=200)
            combined_text += process_image_vision(
                pix.tobytes("png"),
                f"{username}_P{page_index + 1}_fullpage",
                position="full-page",
            )
            continue

        # Collect text items with their vertical (y0) and horizontal (x0) positions
        items = []
        for block in text_blocks:
            bbox = block.get("bbox")
            text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text += span.get("text", "") + " "
                text += "\n"
            items.append((bbox[1], bbox[0], "text", text))

        # Collect visible embedded images, deduplicated by xref
        processed_xrefs = set()
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in processed_xrefs:
                continue
            processed_xrefs.add(xref)

            rects = page.get_image_rects(xref)
            if not rects:
                continue

            rect = rects[0]
            base_image = doc.extract_image(xref)
            items.append((rect.y0, rect.x0, "image", base_image["image"]))

        # Sort items in reading order (top-to-bottom, left-to-right)
        items.sort(key=lambda x: (x[0], x[1]))

        img_counter = 0
        page_rect = page.rect
        for y, x, item_type, data in items:
            if item_type == "text":
                combined_text += data
            else:
                source = f"{username}_P{page_index + 1}_img{img_counter}"
                img_counter += 1

                # Classify the image's rough position on the page
                y_pos = (
                    "top"
                    if y < page_rect.height / 3
                    else "bottom"
                    if y >= page_rect.height * 2 / 3
                    else "middle"
                )
                x_pos = (
                    "left"
                    if x < page_rect.width / 3
                    else "right"
                    if x >= page_rect.width * 2 / 3
                    else "center"
                )
                combined_text += process_image_vision(
                    data, source, position=f"{y_pos}-{x_pos}"
                )

    logger.info(f"PDF processed: {len(doc)} pages for user {username}")
    return combined_text
