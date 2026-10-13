from pathlib import Path
import re

from pypdf import PdfReader

def extract_text(file_path: str) -> str:
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)
        pages = [
            page.extract_text() or ""
            for page in reader.pages
        ]
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
        raise ValueError(
            "Document is empty or contains no readable text"
        )

    return text


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "chunk_size must be positive and overlap must be "
            "non-negative and smaller than chunk_size"
        )

    text = text.strip()
    if not text:
        return []

    # Split at sentence endings, keeping each sentence intact.
    sentences = re.split(
        r'(?<=[.!?])\s+|\n{2,}',
        text,
    )
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if (
            current_sentences
            and current_length + sentence_length + 1 > chunk_size
        ):
            chunks.append(" ".join(current_sentences))

            # Retain complete sentences for overlap.
            overlap_sentences = []
            overlap_length = 0

            for previous in reversed(current_sentences):
                if overlap_length + len(previous) > overlap:
                    break
                overlap_sentences.insert(0, previous)
                overlap_length += len(previous) + 1

            current_sentences = overlap_sentences
            current_length = sum(
                len(s) + 1 for s in current_sentences
            )

        current_sentences.append(sentence)
        current_length += sentence_length + 1

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
