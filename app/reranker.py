_reranker = None


def get_reranker():
    global _reranker

    if _reranker is None:
        print("Loading cross-encoder model...")

        # Import only when reranking is actually requested
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    return _reranker


def rerank(
    query: str,
    candidates: list[tuple[int, float, str]],
    top_n: int = 5
) -> list[tuple[int, float, str]]:

    if not candidates:
        return []

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    reranker = get_reranker()
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