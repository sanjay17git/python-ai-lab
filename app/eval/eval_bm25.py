
import json
from pathlib import Path

from app.pipeline import read_docs, chunk_text
from app.bm25 import BM25

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = PROJECT_ROOT / "eval" / "eval_set.json"
DOC_DIR = PROJECT_ROOT / "data" / "eval_docs"
RESULT_FILE = PROJECT_ROOT / "eval" / "bm25_metrics.json"

K = 5


def main():
    with EVAL_FILE.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    documents = read_docs(str(DOC_DIR))

    chunks = []
    for filename, content in documents.items():
        chunks.extend(chunk_text(content, size=100, overlap=20))

    bm25 = BM25(chunks)

    hits = []
    reciprocal_ranks = []
    details = []

    answerable = [
        item for item in questions
        if item["type"] != "negative"
    ]

    for item in answerable:
        relevant = set(item["relevant_chunks"])

        if not relevant:
            print(f"Skipping {item['id']}: no relevant chunks labelled.")
            continue

        # BM25 returns (chunk_id, score, text) tuples.
        results = bm25.search(item["question"], K)
        ranked_ids = [chunk_id for chunk_id, score, text in results]

        hit = any(chunk_id in relevant for chunk_id in ranked_ids)
        reciprocal_rank = next(
            (
                1.0 / rank
                for rank, chunk_id in enumerate(ranked_ids, start=1)
                if chunk_id in relevant
            ),
            0.0,
        )

        hits.append(int(hit))
        reciprocal_ranks.append(reciprocal_rank)

        details.append({
            "id": item["id"],
            "question": item["question"],
            "relevant_chunks": sorted(relevant),
            "retrieved_chunks": ranked_ids,
            "hit_at_5": hit,
            "reciprocal_rank": reciprocal_rank,
        })

        print(f"\n{item['id']}: {item['question']}")
        print("Relevant:", sorted(relevant))
        print("BM25 retrieved:", ranked_ids)
        print("Hit@5:", hit)
        print("Reciprocal rank:", reciprocal_rank)

    evaluated = len(hits)

    metrics = {
        "retrieval_method": "BM25-only",
        "k": K,
        "questions_evaluated": evaluated,
        "questions_skipped": len(answerable) - evaluated,
        "hit_at_5": round(sum(hits) / evaluated, 4) if evaluated else None,
        "mrr": round(
            sum(reciprocal_ranks) / evaluated, 4
        ) if evaluated else None,
        "details": details,
    }

    with RESULT_FILE.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n--- BM25-only Summary ---")
    print("Questions evaluated:", evaluated)
    print("Hit@5:", metrics["hit_at_5"])
    print("MRR:", metrics["mrr"])
    print("Saved to:", RESULT_FILE)


if __name__ == "__main__":
    main()
