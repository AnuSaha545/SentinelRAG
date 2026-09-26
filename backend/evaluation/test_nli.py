import json
from pathlib import Path

from sentence_transformers import CrossEncoder

RESULTS_FILE = Path(__file__).parent / "results.json"

model = CrossEncoder(
    "cross-encoder/nli-deberta-v3-small",
    local_files_only=True,
)

with RESULTS_FILE.open("r", encoding="utf-8") as file:
    results = json.load(file)

for index, result in enumerate(results, start=1):
    chunks = result.get("results", [])
    answer = result.get("generated_answer", "")

    if not chunks or not answer:
        print(f"Q{index:02d} | no data")
        continue

    pairs = [
        (chunk["content"], answer)
        for chunk in chunks
    ]

    scores = model.predict(
        pairs,
        apply_softmax=True,
    )

    best = scores.max(axis=0)

    print(
        f"Q{index:02d} | "
        f"contradiction={best[0]:.3f} | "
        f"entailment={best[1]:.3f} | "
        f"neutral={best[2]:.3f}"
    )