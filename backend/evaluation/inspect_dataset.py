import json

with open("evaluation/dataset.json", encoding="utf-8") as file:
    data = json.load(file)

print("ID | SIM | RERANK | ANSWER_SIM | CHUNK_SIM | ENTAIL | CONTRA | REFUSAL | CORRECT")

for x in data:
    def fmt(value):
        return "None" if value is None else f"{value:.3f}"

    print(
        f"{x['question_id']:02d} | "
        f"{fmt(x['top_similarity'])} | "
        f"{fmt(x['max_rerank_score'])} | "
        f"{fmt(x['context_answer_similarity'])} | "
        f"{fmt(x['max_answer_chunk_similarity'])} | "
        f"{fmt(x['max_nli_entailment'])} | "
        f"{fmt(x['max_nli_contradiction'])} | "
        f"{x['is_refusal']} | "
        f"{x['correct']}"
    )