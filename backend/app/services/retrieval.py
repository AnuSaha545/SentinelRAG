import re

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


def is_list_question(query: str) -> bool:
    return bool(re.match(
        r"\s*(?:what\s+are\b|what\s+types?\s+of\b|which\b|list\b|name\b)",
        query,
        re.IGNORECASE,
    ))


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

    chunk_pairs = [
        (query, str(chunk.get("content", "") or ""))
        for chunk in chunks
    ]
    passage_pairs = []
    passage_chunk_indices = []

    for chunk_index, chunk in enumerate(chunks):
        words = str(chunk.get("content", "") or "").split()
        if not words:
            continue

        window_size = 80
        stride = 40
        for start in range(0, len(words), stride):
            passage = " ".join(words[start:start + window_size])
            passage_pairs.append((query, passage))
            passage_chunk_indices.append(chunk_index)
            if start + window_size >= len(words):
                break

    scores = reranker.predict(chunk_pairs + passage_pairs)
    chunk_scores = scores[:len(chunk_pairs)]
    passage_scores = scores[len(chunk_pairs):]
    best_scores = {}
    best_passages = {}
    list_answer_matches = {}

    list_anchor = None
    expected_count = None
    if is_list_question(query):
        normalized_query = query.lower()
        if normalized_query.lstrip().startswith("which "):
            query_terms = re.findall(r"[a-z0-9]+", normalized_query)
            list_anchor = next(
                (
                    term
                    for term in query_terms[1:]
                    if term.endswith("s") and len(term) > 3
                ),
                None,
            )
        else:
            query_prefix = re.split(
                r"\b(?:in|of|from|for|within)\b",
                normalized_query,
                maxsplit=1,
            )[0]
            query_terms = re.findall(r"[a-z0-9]+", query_prefix)
            list_anchor = next(
                (
                    term
                    for term in reversed(query_terms)
                    if term.endswith("s") and len(term) > 3
                ),
                None,
            )

        number_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        expected_count = next(
            (
                int(term) if term.isdigit() else number_words[term]
                for term in query_terms
                if term.isdigit() or term in number_words
            ),
            None,
        )

    def matches_requested_list(passage: str) -> bool:
        if not list_anchor:
            return False

        label = re.search(
            rf"\b{re.escape(list_anchor)}\s*:\s*([^.!?]+)",
            passage,
            re.IGNORECASE,
        )
        if not label:
            return False

        items = [
            item.strip()
            for item in re.split(r",\s*|\s+and\s+", label.group(1))
            if item.strip()
        ]
        return len(items) >= 3 and (
            expected_count is None or len(items) == expected_count
        )

    for chunk_index, pair, score in zip(
        passage_chunk_indices,
        passage_pairs,
        passage_scores,
    ):
        score = float(score)
        if score > best_scores.get(chunk_index, float("-inf")):
            best_scores[chunk_index] = score
            best_passages[chunk_index] = pair[1]
        if matches_requested_list(pair[1]):
            list_answer_matches[chunk_index] = True

    reranked = [
        {
            **chunk,
            "rerank_score": round(
                best_scores.get(index, float("-inf")),
                4,
            ),
            "_confidence_rerank_score": round(
                float(chunk_scores[index]),
                4,
            ),
            "_rerank_evidence": best_passages.get(
                index,
                str(chunk.get("content", "") or ""),
            ),
            "_list_answer_match": list_answer_matches.get(index, False),
        }
        for index, chunk in enumerate(chunks)
    ]

    reranked.sort(
        key=lambda item: (
            item["_list_answer_match"],
            item["rerank_score"],
        ),
        reverse=True,
    )

    return [
        {
            key: value
            for key, value in item.items()
            if key != "_list_answer_match"
        }
        for item in reranked[:limit]
    ]


def prepare_generation_context(
    query: str,
    chunks: list[dict],
    max_chunks: int = 3,
) -> list[dict]:
    if not chunks:
        return []

    if all("_rerank_evidence" in chunk for chunk in chunks[:max_chunks]):
        return [
            {
                **{
                    key: value
                    for key, value in chunk.items()
                    if key != "_rerank_evidence"
                },
                "content": chunk["_rerank_evidence"],
            }
            for chunk in chunks[:max_chunks]
        ]

    stop_words = {
        "what",
        "when",
        "where",
        "who",
        "why",
        "how",
        "the",
        "this",
        "that",
        "these",
        "those",
        "with",
        "from",
        "into",
        "about",
        "are",
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
        "do",
        "does",
        "can",
        "be",
    }

    query_terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9]+", query.lower())
        if len(term) >= 3 and term not in stop_words
    }

    chunk_map = {
        chunk.get("chunk_id"): chunk
        for chunk in chunks
        if chunk.get("chunk_id") is not None
    }

    sentence_candidates = []

    for chunk in chunks[: max_chunks * 3]:
        content = str(chunk.get("content", "") or "").strip()
        if not content:
            continue

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", content)
            if sentence.strip() and len(sentence.strip()) >= 40
        ]

        if not sentences:
            sentences = [content[:500]]

        for sentence in sentences:
            sentence_overlap = sum(
                1
                for term in query_terms
                if term in sentence.lower()
            )
            sentence_candidates.append({
                "chunk_id": chunk.get("chunk_id"),
                "sentence": sentence,
                "overlap": sentence_overlap,
            })

    if not sentence_candidates:
        return chunks[:max_chunks]

    sentence_pairs = [
        (query, candidate["sentence"])
        for candidate in sentence_candidates
    ]
    sentence_scores = reranker.predict(sentence_pairs)

    ranked_sentences = []
    for candidate, score in zip(sentence_candidates, sentence_scores):
        ranked_sentences.append({
            **candidate,
            "score": float(score),
        })

    ranked_sentences.sort(
        key=lambda item: (
            item["score"],
            item["overlap"],
        ),
        reverse=True,
    )

    selected = []
    seen_chunk_ids = set()

    for candidate in ranked_sentences:
        chunk_id = candidate.get("chunk_id")
        if chunk_id in seen_chunk_ids:
            continue

        chunk = chunk_map.get(chunk_id)
        if not chunk:
            continue

        selected.append({
            **chunk,
            "content": candidate["sentence"],
        })
        seen_chunk_ids.add(chunk_id)

        if len(selected) >= max_chunks:
            break

    if selected:
        return selected

    return chunks[:max_chunks]