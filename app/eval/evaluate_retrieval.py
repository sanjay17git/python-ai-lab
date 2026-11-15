
import json
from pathlib import Path

from app.pipeline import RetrievalPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = PROJECT_ROOT / "eval" / "eval_set.json"
DOC_DIR = PROJECT_ROOT / "data" / "eval_docs"
RESULT_FILE = PROJECT_ROOT / "eval" / "retrieval_metrics.json"

K = 5
CANDIDATE_K = 10


def main():
    with EVAL_FILE.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    pipeline = RetrievalPipeline()
    index_info = pipeline.index_documents(str(DOC_DIR))
    
    print("\n--- Chunk ID verification ---")

    for chunk_id in [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15,16,17,18]:
        if chunk_id < len(pipeline.chunks):
            print(f"\nChunk {chunk_id}:")
            print(pipeline.chunks[chunk_id][:500])

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
            print(
                f"WARNING: {item['id']} has no relevant_chunks. "
                "Skipping it."
            )
            continue

        results = pipeline.search(
            item["question"],
            k=K,
            candidate_k=CANDIDATE_K,
        )

        # Search results contain (chunk_id, score, text).
        ranked_ids = [chunk_id for chunk_id, score, text in results]

        hit = any(chunk_id in relevant for chunk_id in ranked_ids)
        hits.append(int(hit))

        reciprocal_rank = 0.0
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            if chunk_id in relevant:
                reciprocal_rank = 1.0 / rank
                break

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
        print("Relevant chunks:", sorted(relevant))
        print("Retrieved chunks:", ranked_ids)
        print("Hit@5:", hit)
        print("Reciprocal rank:", reciprocal_rank)

    evaluated = len(hits)

    metrics = {
        "documents_loaded": index_info["documents_loaded"],
        "chunks_indexed": index_info["chunks_indexed"],
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

    print("\n--- Retrieval Evaluation Summary ---")
    print(f"Questions evaluated: {evaluated}")
    print(f"Hit@5: {metrics['hit_at_5']}")
    print(f"MRR: {metrics['mrr']}")
    print(f"Results saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()
