import json
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / "results.json"
LABELS_FILE = Path(__file__).parent / "labels.json"

with RESULTS_FILE.open("r", encoding="utf-8") as file:
    results = json.load(file)

with LABELS_FILE.open("r", encoding="utf-8") as file:
    labels = json.load(file)

label_map = {
    item["question_id"]: item["correct"]
    for item in labels
}

for index, result in enumerate(results, start=1):
    correct = label_map[index]
    answer = result.get("generated_answer", "")

    is_refusal = (
        "could not find the answer"
        in answer.lower()
    )

    if correct == 0 or (correct == 1 and is_refusal):
        print(
            f"Q{index:02d} | "
            f"correct={correct} | "
            f"refusal={is_refusal} | "
            f"answer={answer[:180]}"
        )