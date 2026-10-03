import json
from pathlib import Path

import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_FILE = Path(__file__).parent / "dataset.json"
MODEL_FILE = Path(__file__).parent / "confidence_model.joblib"


with DATASET_FILE.open("r", encoding="utf-8") as file:
    data = json.load(file)


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
    "max_answer_chunk_similarity",
    "max_nli_entailment",
    "max_nli_contradiction",
]


X = [
    [row[feature] for feature in features]
    for row in data
]

y = [
    row["correct"]
    for row in data
]


model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(class_weight="balanced"),
    ),
])


model.fit(X, y)


joblib.dump(
    {
        "model": model,
        "features": features,
    },
    MODEL_FILE,
)


probabilities = model.predict_proba(X)[:, 1]

for row, probability in zip(data, probabilities):
    print(
        f"Q{row['question_id']:02d} | "
        f"correct={row['correct']} | "
        f"confidence={probability:.4f}"
    )


print(f"\nModel saved to: {MODEL_FILE}")
print("Model trained successfully.")