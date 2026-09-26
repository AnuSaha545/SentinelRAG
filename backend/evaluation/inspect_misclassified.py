import json
from pathlib import Path


DATASET_FILE = Path(__file__).parent / "dataset.json"

with DATASET_FILE.open("r", encoding="utf-8") as file:
    data = json.load(file)


question_ids = {6, 23, 31, 36, 37, 38, 40, 41, 44}

features = [
    "top_similarity",
    "max_rerank_score",
    "mean_rerank_score",
    "rerank_score_spread",
    "vector_matches",
    "keyword_matches",
    "hybrid_match",
    "answer_length",
    "is_refusal",
    "context_answer_similarity",
]


for row in data:
    if row["question_id"] in question_ids:
        print(f"\nQ{row['question_id']} | actual={row['correct']}")

        for feature in features:
            print(f"  {feature}: {row[feature]}")