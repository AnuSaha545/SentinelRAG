import json
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

RESULTS_FILE = Path(__file__).parent / "results.json"
LABELS_FILE = Path(__file__).parent / "labels.json"

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True,
)

nli_model = CrossEncoder(
    "cross-encoder/nli-deberta-v3-small",
    local_files_only=True,
)

with RESULTS_FILE.open("r", encoding="utf-8") as file:
    results = json.load(file)

with LABELS_FILE.open("r", encoding="utf-8") as file:
    labels = json.load(file)

label_map = {
    item["question_id"]: item["correct"]
    for item in labels
}

for index, result in enumerate(results, start=1):
    chunks = result.get("results", [])
    answer = result.get("generated_answer", "")

    if not chunks or not answer:
        print(f"Q{index:02d} | no data")
        continue

    answer_embedding = embedding_model.encode(
        answer,
        normalize_embeddings=True,
    )

    chunk_embeddings = embedding_model.encode(
        [chunk["content"] for chunk in chunks],
        normalize_embeddings=True,
    )

    similarities = [
        float(answer_embedding @ chunk_embedding)
        for chunk_embedding in chunk_embeddings
    ]

    max_similarity = max(similarities)

    pairs = [
        (chunk["content"], answer)
        for chunk in chunks
    ]

    nli_scores = nli_model.predict(
        pairs,
        apply_softmax=True,
    )

    max_entailment = float(nli_scores[:, 1].max())
    max_contradiction = float(nli_scores[:, 0].max())

    verification_score = (
        0.5 * max_entailment
        + 0.3 * max_similarity
        - 0.2 * max_contradiction
    )

    print(
        f"Q{index:02d} | "
        f"correct={label_map[index]} | "
        f"similarity={max_similarity:.3f} | "
        f"entailment={max_entailment:.3f} | "
        f"contradiction={max_contradiction:.3f} | "
        f"verification={verification_score:.3f}"
    )