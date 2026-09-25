from app.services.embeddings import generate_embedding
from app.services.ingestion import chunk_text, extract_text


def process_document(file_path: str) -> list[dict]:
    text = extract_text(file_path)
    chunks = chunk_text(text)

    processed_chunks = []

    for index, chunk in enumerate(chunks):
        processed_chunks.append(
            {
                "chunk_index": index,
                "text": chunk,
                "embedding": generate_embedding(chunk),
            }
        )

    return processed_chunks