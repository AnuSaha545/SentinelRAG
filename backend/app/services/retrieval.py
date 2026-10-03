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
    limit: int = 20,
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
    limit: int = 20,
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


def expand_context(
    db: Session,
    chunks: list[dict],
    window: int = 1,
) -> list[dict]:
    if not chunks:
        return []

    expanded = {}

    chunk_indices = {
        chunk["chunk_index"]
        for chunk in chunks
    }

    for index in chunk_indices:
        statement = (
            select(DocumentChunk)
            .where(
                DocumentChunk.document_id
                == chunks[0]["document_id"],
                DocumentChunk.chunk_index >= index - window,
                DocumentChunk.chunk_index <= index + window,
            )
            .order_by(DocumentChunk.chunk_index)
        )

        neighbors = db.execute(statement).scalars().all()

        for neighbor in neighbors:
            if neighbor.id not in expanded:
                expanded[neighbor.id] = {
                    "chunk_id": neighbor.id,
                    "document_id": neighbor.document_id,
                    "chunk_index": neighbor.chunk_index,
                    "content": neighbor.content,
                    "similarity": 0.0,
                    "keyword_score": 0.0,
                    "retrieval_methods": ["context"],
                }

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]

        if chunk_id in expanded:
            expanded[chunk_id] = {
                **expanded[chunk_id],
                **chunk,
            }
        else:
            expanded[chunk_id] = chunk

    return list(expanded.values())


def hybrid_retrieve_chunks(
    db: Session,
    query: str,
    document_id: str,
    limit: int = 20,
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
            "keyword_score": 0.0,
            "retrieval_methods": ["vector"],
        }

    for result in keyword_results:
        chunk_id = result["chunk_id"]

        if chunk_id in combined:
            combined[chunk_id]["keyword_score"] = result.get(
                "keyword_score",
                0.0,
            )
            combined[chunk_id]["retrieval_methods"].append("keyword")
        else:
            combined[chunk_id] = {
                **result,
                "similarity": 0.0,
                "keyword_score": result.get(
                    "keyword_score",
                    0.0,
                ),
                "retrieval_methods": ["keyword"],
            }

    candidates = list(combined.values())

    return expand_context(
        db=db,
        chunks=candidates,
        window=1,
    )


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