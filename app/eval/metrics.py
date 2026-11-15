def hit_at_k(
    ranked_ids: list[int],
    relevant_ids: set[int],
    k: int
) -> float:
    if not relevant_ids:
        return 0.0
    return float(bool(set(ranked_ids[:k]) & relevant_ids))

def recall_at_k(
    ranked_ids: list[int],
    relevant_ids: set[int],
    k: int
) -> float:
    if not relevant_ids:
        return 0.0
    found = set(ranked_ids[:k]) & relevant_ids
    return len(found) / len(relevant_ids)

def reciprocal_rank(
    ranked_ids: list[int],
    relevant_ids: set[int]
) -> float:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def average_precision(
    ranked_ids: list[int],
    relevant_ids: set[int]
) -> float:
    if not relevant_ids:
        return 0.0
    hits, total = 0, 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            total += hits / rank
    return total / len(relevant_ids)

def compute_metrics(
    ranked_ids: list[int],
    relevant_ids: set[int],
    k: int = 5
) -> dict:
    return {
        f"Hit@{k}": hit_at_k(ranked_ids, relevant_ids, k),
        f"Recall@{k}": recall_at_k(ranked_ids, relevant_ids, k),
        "MRR": reciprocal_rank(ranked_ids, relevant_ids),
        "AP": average_precision(ranked_ids, relevant_ids)
    }