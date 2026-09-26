import json
from pathlib import Path

from app.core.database import SessionLocal
from app.services.retrieval import hybrid_retrieve_chunks, rerank_chunks
from app.services.generation import generate_answer


QUESTIONS_FILE = Path(__file__).parent / "questions.json"
RESULTS_FILE = Path(__file__).parent / "results.json"


def load_questions() -> list[dict]:
    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_question(question: dict) -> dict:
    db = SessionLocal()

    try:
        candidates = hybrid_retrieve_chunks(
            db=db,
            query=question["question"],
            document_id=question["document_id"],
            limit=5,
        )

        results = rerank_chunks(
            query=question["question"],
            chunks=candidates,
            limit=5,
        )

        answer = generate_answer(
            query=question["question"],
            chunks=results,
        )

        top_result = results[0] if results else {}

        return {
    "question_id": question["question_id"],
    "question": question["question"],
    "correct": question["correct"],
    "generated_answer": answer,
    "top_similarity": top_result.get("similarity"),
    "top_rerank_score": top_result.get("rerank_score"),
    "retrieval_methods": top_result.get("retrieval_methods", []),
    "results": results,
}

    finally:
        db.close()


if __name__ == "__main__":
    questions = load_questions()
    evaluation_results = []

    for index, question in enumerate(questions, start=1):
        print(f"\nEvaluating {index}/{len(questions)}:")
        print(question["question"])

        result = evaluate_question(question)
        evaluation_results.append(result)

        print(f"Answer: {result['generated_answer']}")
        print(f"Top similarity: {result['top_similarity']}")
        print(f"Top rerank score: {result['top_rerank_score']}")
        print(f"Methods: {result['retrieval_methods']}")

    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            evaluation_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nEvaluation complete.")
    print(f"Saved results to: {RESULTS_FILE}")