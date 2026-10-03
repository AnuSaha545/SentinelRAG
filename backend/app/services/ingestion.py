from pathlib import Path
import re

from pypdf import PdfReader


def extract_text(file_path: str) -> str:
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)
        pages = []

        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

        text = "\n".join(pages)

        text = re.sub(
            r"(?<=\b[A-Za-z])\s+(?=[a-z]{1,3}\b)",
            "",
            text,
        )

    elif path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8")
    else:
        raise ValueError("Unsupported file type")

    if not text or not text.strip():
        raise ValueError("Document is empty or contains no readable text")

    return text


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step

    return chunks