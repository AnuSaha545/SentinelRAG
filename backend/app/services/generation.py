import ollama


MODEL_NAME = "llama3.2:latest"


def generate_answer(
    query: str,
    chunks: list[dict],
) -> str:
    if not chunks:
        return "I could not find the answer in the provided document."

    context = "\n\n".join(
        f"[Source {index + 1}]\n{chunk['content']}"
        for index, chunk in enumerate(chunks)
    )

    prompt = f"""
You are answering a question using a retrieved document.

Rules:
1. Answer the question directly.
2. Use only information contained in the context.
3. Do not mention source numbers unless explicitly asked.
4. Do not say where the answer is located; give the actual answer.
5. If the context does not contain the answer, say:
   "I could not find the answer in the provided document."
6. Do not invent or add information.

Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    answer = response["message"]["content"]
    if not answer or not answer.strip():
        return "I could not find the answer in the provided document."

    return answer