import json
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / "results.json"

QUESTION_IDS = {
    6, 24, 28, 29, 31,
    37, 38, 40, 42, 44,
}

with RESULTS_FILE.open("r", encoding="utf-8") as file:
    results = json.load(file)

for index, result in enumerate(results, start=1):
    if index not in QUESTION_IDS:
        continue

    print(f"\n{'=' * 70}")
    print(f"Q{index:02d}")
    print(f"ANSWER: {result['generated_answer']}")
    print(f"{'=' * 70}")

    for chunk in result.get("results", []):
        print(
            f"\nChunk {chunk['chunk_index']}"
            f" | similarity={chunk.get('similarity')}"
            f" | rerank={chunk.get('rerank_score')}"
            f" | methods={chunk.get('retrieval_methods')}"
        )
        print(chunk["content"][:1000])