import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_FILE = Path(__file__).parent / "dataset.json"


with DATASET_FILE.open("r", encoding="utf-8") as file:
    data = json.load(file)


features = [
    "top_similarity",
    "max_rerank_score",
    "answer_length",
    "is_refusal",
    "context_answer_similarity",
]


X = [
    [
        0.0 if row[feature] is None else row[feature]
        for feature in features
    ]
    for row in data
]


y = [
    row["correct"]
    for row in data
]


model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(class_weight="balanced"),
    ),
])


cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42,
)


predictions = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
)


print("Features used:")
for feature in features:
    print(f"- {feature}")


print(f"\nAccuracy: {accuracy_score(y, predictions):.4f}")


print("\nClassification report:")
print(
    classification_report(
        y,
        predictions,
        zero_division=0,
    )
)


print("Confusion matrix:")
print(confusion_matrix(y, predictions))