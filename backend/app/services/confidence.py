from pathlib import Path

import joblib


MODEL_FILE = (
    Path(__file__).resolve().parents[2]
    / "evaluation"
    / "confidence_model.joblib"
)

model_data = joblib.load(MODEL_FILE)

model = model_data["model"]
features = model_data["features"]


def calculate_confidence(
    top_similarity: float,
    max_rerank_score: float,
    mean_rerank_score: float,
    rerank_score_spread: float,
    vector_matches: int,
    keyword_matches: int,
    hybrid_match: int,
    answer_length: int,
    is_refusal: int,
    context_answer_similarity: float,
    max_answer_chunk_similarity: float,
    max_nli_entailment: float,
    max_nli_contradiction: float,
) -> float:

    values = [[
        top_similarity,
        max_rerank_score,
        mean_rerank_score,
        rerank_score_spread,
        vector_matches,
        keyword_matches,
        hybrid_match,
        answer_length,
        is_refusal,
        context_answer_similarity,
        max_answer_chunk_similarity,
        max_nli_entailment,
        max_nli_contradiction,
    ]]

    confidence = model.predict_proba(values)[0][1]

    return round(float(confidence), 4)