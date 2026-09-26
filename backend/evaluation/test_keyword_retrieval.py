import json
from pathlib import Path

from app.core.database import SessionLocal
from app.services.retrieval import keyword_retrieve_chunks

QUESTIONS_FILE = Path(__file__).parent / "questions.json"

QUESTION_IDS = {
    6, 24, 28, 29, 31,
    37, 38, 40, 42, 44,
}

with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
    questions = json.load(file)

db = SessionLocal()

try:
    for index, question in enumerate(questions, start=1):
        if index not in QUESTION_IDS:
            continue

        results = keyword_retrieve_chunks(
            db=db,
            query=question["question"],
            document_id=question["document_id"],
            limit=5,
        )

        print(f"\nQ{index:02d}")
        print(f"Query: {question['question']}")
        print(f"Keyword matches: {len(results)}")

        for result in results:
            print(
                f"  Chunk {result['chunk_index']}: "
                f"{result['content'][:200]}"
            )
finally:
    db.close()