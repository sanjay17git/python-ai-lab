
from sentence_transformers import CrossEncoder


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(
    query: str,
    candidates: list[tuple[int, float, str]],
    top_n: int = 5
) -> list[tuple[int, float, str]]:
    if not 1 <= top_n <= len(candidates):
        raise ValueError(
            "top_n must be between 1 and the number of candidates."
        )

    pairs = [(query, text) for _, _, text in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    return [
        (doc_id, float(score), text)
        for (doc_id, _, text), score in ranked[:top_n]
    ]
