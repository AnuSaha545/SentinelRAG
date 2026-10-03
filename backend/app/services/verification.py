from sentence_transformers import CrossEncoder, SentenceTransformer


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    local_files_only=True,
)

nli_model = CrossEncoder(
    NLI_MODEL_NAME,
    local_files_only=True,
)


def calculate_verification_features(
    answer: str,
    chunks: list[dict],
) -> dict:
    if not answer or not chunks:
        return {
            "context_answer_similarity": 0.0,
            "max_answer_chunk_similarity": 0.0,
            "max_nli_entailment": 0.0,
            "max_nli_contradiction": 0.0,
        }

    answer_embedding = embedding_model.encode(
        answer,
        normalize_embeddings=True,
    )

    context = "\n".join(
        chunk.get("content", "")
        for chunk in chunks
    )

    context_embedding = embedding_model.encode(
        context,
        normalize_embeddings=True,
    )

    context_answer_similarity = float(
        answer_embedding @ context_embedding
    )

    answer_chunk_similarities = []

    for chunk in chunks:
        chunk_embedding = embedding_model.encode(
            chunk.get("content", ""),
            normalize_embeddings=True,
        )

        similarity = float(
            answer_embedding @ chunk_embedding
        )

        answer_chunk_similarities.append(similarity)

    max_answer_chunk_similarity = max(
        answer_chunk_similarities,
        default=0.0,
    )

    nli_pairs = [
        (chunk.get("content", ""), answer)
        for chunk in chunks
    ]

    nli_scores = nli_model.predict(
        nli_pairs,
        apply_softmax=True,
    )

    max_nli_entailment = float(
        nli_scores[:, 1].max()
    )

    max_nli_contradiction = float(
        nli_scores[:, 0].max()
    )

    return {
        "context_answer_similarity": round(
            context_answer_similarity,
            4,
        ),
        "max_answer_chunk_similarity": round(
            max_answer_chunk_similarity,
            4,
        ),
        "max_nli_entailment": round(
            max_nli_entailment,
            4,
        ),
        "max_nli_contradiction": round(
            max_nli_contradiction,
            4,
        ),
    }