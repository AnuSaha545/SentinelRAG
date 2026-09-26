import json
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

QUESTIONS_FILE = Path(__file__).parent / "questions.json"
RESULTS_FILE = Path(__file__).parent / "results.json"
LABELS_FILE = Path(__file__).parent / "labels.json"
DATASET_FILE = Path(__file__).parent / "dataset.json"

EMBEDDING_MODEL = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True,
)

NLI_MODEL = CrossEncoder(
    "cross-encoder/nli-deberta-v3-small",
    local_files_only=True,
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


questions = load_json(QUESTIONS_FILE)
results = load_json(RESULTS_FILE)
labels = load_json(LABELS_FILE)

label_map = {
    item["question_id"]: item["correct"]
    for item in labels
}

dataset = []

for index, result in enumerate(results, start=1):
    chunks = result.get("results", [])
    retrieval_methods = result.get("retrieval_methods", [])

    vector_matches = sum(
        1
        for item in chunks
        if "vector" in item.get("retrieval_methods", [])
    )

    keyword_matches = sum(
        1
        for item in chunks
        if "keyword" in item.get("retrieval_methods", [])
    )

    rerank_scores = [
        item["rerank_score"]
        for item in chunks
        if "rerank_score" in item
    ]

    max_rerank_score = max(rerank_scores) if rerank_scores else 0.0
    mean_rerank_score = (
        sum(rerank_scores) / len(rerank_scores)
        if rerank_scores
        else 0.0
    )

    rerank_score_spread = (
        max(rerank_scores) - min(rerank_scores)
        if rerank_scores
        else 0.0
    )

    answer = result.get("generated_answer", "")

    context = "\n".join(
        item.get("content", "")
        for item in chunks
    )

    answer_embedding = EMBEDDING_MODEL.encode(
        answer,
        normalize_embeddings=True,
    )

    context_embedding = EMBEDDING_MODEL.encode(
        context,
        normalize_embeddings=True,
    )

    context_answer_similarity = float(
        answer_embedding @ context_embedding
    )

    answer_chunk_similarities = [
        float(
            answer_embedding
            @ EMBEDDING_MODEL.encode(
                item.get("content", ""),
                normalize_embeddings=True,
            )
        )
        for item in chunks
    ]

    max_answer_chunk_similarity = (
        max(answer_chunk_similarities)
        if answer_chunk_similarities
        else 0.0
    )

    if chunks and answer:
        nli_pairs = [
            (item["content"], answer)
            for item in chunks
        ]

        nli_scores = NLI_MODEL.predict(
            nli_pairs,
            apply_softmax=True,
        )

        max_nli_entailment = float(nli_scores[:, 1].max())
        max_nli_contradiction = float(nli_scores[:, 0].max())
    else:
        max_nli_entailment = 0.0
        max_nli_contradiction = 0.0

    is_refusal = int(
        "could not find the answer" in answer.lower()
    )

    dataset.append({
        "question_id": index,
        "top_similarity": result.get("top_similarity", 0.0),
        "max_rerank_score": max_rerank_score,
        "mean_rerank_score": mean_rerank_score,
        "rerank_score_spread": rerank_score_spread,
        "vector_matches": vector_matches,
        "keyword_matches": keyword_matches,
        "hybrid_match": int(
            "vector" in retrieval_methods
            and "keyword" in retrieval_methods
        ),
        "answer_length": len(answer),
        "is_refusal": is_refusal,
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
        "correct": label_map[index],
    })


with DATASET_FILE.open("w", encoding="utf-8") as file:
    json.dump(
        dataset,
        file,
        indent=2,
    )

print(f"Dataset created: {DATASET_FILE}")
print(f"Samples: {len(dataset)}")