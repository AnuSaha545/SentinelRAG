def calculate_confidence(chunks: list[dict]) -> dict:
    if not chunks:
        return {
            "score": 0.0,
            "top_similarity": 0.0,
            "top_rerank_score": 0.0,
        }

    similarities = [
        chunk["similarity"]
        for chunk in chunks
        if "similarity" in chunk
    ]

    rerank_scores = [
        chunk["rerank_score"]
        for chunk in chunks
        if "rerank_score" in chunk
    ]

    top_similarity = max(similarities) if similarities else 0.0
    top_rerank_score = max(rerank_scores) if rerank_scores else 0.0

    return {
        "score": 0.0,
        "top_similarity": round(top_similarity, 4),
        "top_rerank_score": round(top_rerank_score, 4),
    }