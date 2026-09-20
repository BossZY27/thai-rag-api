import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import chunk_text, extract_pdf_pages, save_document_chunks


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/load_pdf.py <pdf-path>")
        return 1

    pdf_path = Path(sys.argv[1]).resolve()
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        print(f"PDF not found: {pdf_path}")
        return 1

    pages = extract_pdf_pages(pdf_path)
    if not pages:
        print("No selectable text found. This PDF probably needs OCR first.")
        return 2

    page_chunks = []
    for page in pages:
        for text in chunk_text(page["text"]):
            page_chunks.append({"page": page["page"], "text": text})

    result = save_document_chunks(
        document_id=pdf_path.stem[:64],
        title=pdf_path.name,
        source_type="database",
        source_ref=f"file:{pdf_path}",
        page_chunks=page_chunks,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
