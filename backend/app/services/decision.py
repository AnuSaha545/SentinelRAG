CONFIDENCE_THRESHOLD = 0.70


def make_decision(confidence: float) -> str:
    if confidence >= CONFIDENCE_THRESHOLD:
        return "accept"

    return "retry"


def needs_human_review(confidence: float) -> bool:
    return confidence < CONFIDENCE_THRESHOLD