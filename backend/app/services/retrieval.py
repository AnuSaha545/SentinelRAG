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
    import re

    terms = re.findall(r"[A-Za-z0-9]+", query.lower())

    stop_words = {
        "what",
        "are",
        "the",
        "is",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "how",
        "does",
        "do",
        "can",
        "be",
    }

    terms = [
        term
        for term in terms
        if term not in stop_words and len(term) >= 3
    ]

    if not terms:
        return []

    tsquery = " | ".join(terms)

    rank = func.ts_rank(
        DocumentChunk.search_vector,
        func.to_tsquery("english", tsquery),
    ).label("rank")

    statement = (
        select(DocumentChunk, rank)
        .where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.search_vector.op("@@")(
                func.to_tsquery("english", tsquery)
            ),
        )
        .order_by(rank.desc())
        .limit(limit)
    )

    results = db.execute(statement).all()

    return [
        {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "keyword_score": round(float(rank_value), 4),
        }
        for chunk, rank_value in results
    ]


def hybrid_retrieve_chunks(
    db: Session,
    query: str,
    document_id: str,
    limit: int = 10,
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
            "keyword_score": 0.0,
        }

    for result in keyword_results:
        chunk_id = result["chunk_id"]

        if chunk_id in combined:
            combined[chunk_id]["retrieval_methods"].append("keyword")
            combined[chunk_id]["keyword_score"] = result.get(
                "keyword_score",
                0.0,
            )
        else:
            combined[chunk_id] = {
                **result,
                "similarity": 0.0,
                "retrieval_methods": ["keyword"],
            }

    results = list(combined.values())

    # Normalize scores before combining them.
    max_similarity = max(
        (item.get("similarity", 0.0) for item in results),
        default=0.0,
    )

    max_keyword_score = max(
        (item.get("keyword_score", 0.0) for item in results),
        default=0.0,
    )

    for item in results:
        vector_score = (
            item.get("similarity", 0.0) / max_similarity
            if max_similarity > 0
            else 0.0
        )

        keyword_score = (
            item.get("keyword_score", 0.0) / max_keyword_score
            if max_keyword_score > 0
            else 0.0
        )

        item["hybrid_score"] = round(
            0.6 * vector_score + 0.4 * keyword_score,
            4,
        )

    results.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )

    return results[:limit]

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