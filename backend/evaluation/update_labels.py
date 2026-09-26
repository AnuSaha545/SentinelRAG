import json
from pathlib import Path

evaluation_dir = Path(__file__).parent

with open(evaluation_dir / "questions.json", encoding="utf-8") as file:
    questions = json.load(file)

labels = [
    {
        "question_id": question["question_id"],
        "correct": question["correct"],
    }
    for question in questions
]

with open(evaluation_dir / "labels.json", "w", encoding="utf-8") as file:
    json.dump(labels, file, indent=2)

print(f"Updated labels.json with {len(labels)} labels.")