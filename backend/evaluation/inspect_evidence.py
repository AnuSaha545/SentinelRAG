import json
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / "results.json"

QUESTION_IDS = {"Q02", "Q12", "Q17", "Q18", "Q24", "Q40", "Q56", "Q65"}

with RESULTS_FILE.open("r", encoding="utf-8") as file:
    results = json.load(file)

for result in results:
    if result["question_id"] not in QUESTION_IDS:
        continue

    print("\n" + "=" * 80)
    print(result["question_id"])
    print("=" * 80)

    print("\nQUESTION:")
    print(result["question"])

    print("\nGENERATED ANSWER:")
    print(result["generated_answer"])

    print("\nRETRIEVED CHUNKS:")

    for index, chunk in enumerate(result.get("results", []), start=1):
        print(f"\n--- Chunk {index} ---")
        print(f"Similarity: {chunk.get('similarity')}")
        print(f"Rerank score: {chunk.get('rerank_score')}")
        print(f"Retrieval methods: {chunk.get('retrieval_methods')}")
        print(chunk.get("content", ""))