from sentence_transformers import CrossEncoder
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.document import DocumentChunk
from app.services.embeddings import generate_embedding


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(
    RERANKER_MODEL,
    local_files_only=True,
)


def retrieve_chunks(
    db: Session,
    query: str,
    document_id: str,
    limit: int = 5,
) -> list[dict]:
    query_embedding = generate_embedding(query)

    distance = DocumentChunk.embedding.cosine_distance(
        query_embedding
    ).label("distance")

    statement = (
        select(DocumentChunk, distance)
        .where(DocumentChunk.document_id == document_id)
        .order_by(distance)
        .limit(limit)
    )

    results = db.execute(statement).all()

    return [
        {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "similarity": round(1 - float(distance_value), 4),
        }
        for chunk, distance_value in results
    ]


def keyword_retrieve_chunks(
    db: Session,
    query: str,
    document_id: str,
    limit: int = 5,
) -> list[dict]:
    statement = (
        select(DocumentChunk)
        .where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.search_vector.op("@@")(
                func.websearch_to_tsquery("english", query)
            ),
        )
        .limit(limit)
    )

    results = db.execute(statement).scalars().all()

    return [
        {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
        }
        for chunk in results
    ]


def hybrid_retrieve_chunks(
    db: Session,
    query: str,
    document_id: str,
    limit: int = 5,
) -> list[dict]:
    vector_results = retrieve_chunks(
        db=db,
        query=query,
        document_id=document_id,
        limit=limit,
    )

    keyword_results = keyword_retrieve_chunks(
        db=db,
        query=query,
        document_id=document_id,
        limit=limit,
    )

    combined = {}

    for result in vector_results:
        combined[result["chunk_id"]] = {
            **result,
            "retrieval_methods": ["vector"],
        }

    for result in keyword_results:
        chunk_id = result["chunk_id"]

        if chunk_id in combined:
            combined[chunk_id]["retrieval_methods"].append("keyword")
        else:
            combined[chunk_id] = {
                **result,
                "retrieval_methods": ["keyword"],
            }

    return list(combined.values())


def rerank_chunks(
    query: str,
    chunks: list[dict],
    limit: int = 5,
) -> list[dict]:
    if not chunks:
        return []

    pairs = [
        (query, chunk["content"])
        for chunk in chunks
    ]

    scores = reranker.predict(pairs)

    reranked = []

    for chunk, score in zip(chunks, scores):
        reranked.append({
            **chunk,
            "rerank_score": round(float(score), 4),
        })

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return reranked[:limit]