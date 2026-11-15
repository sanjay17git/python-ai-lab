
import json
from pathlib import Path

from app.pipeline import read_docs, chunk_text
from app.retriever import Retriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = PROJECT_ROOT / "data" / "eval_docs"
EVAL_FILE = PROJECT_ROOT / "eval" / "eval_set.json"

K = 5


def main():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    documents = read_docs(str(DOC_DIR))

    chunks = []
    for filename, content in documents.items():
        chunks.extend(chunk_text(content, size=100, overlap=20))

    retriever = Retriever()
    retriever.index(chunks)

    reciprocal_ranks = []
    hits = 0
    evaluated = 0

    for item in eval_data:
        relevant = item.get("relevant_chunks", [])

        # Skip unanswerable questions and questions without relevance labels.
        if item.get("type") == "negative" or not relevant:
            continue

        results = retriever.search(item["question"],K)

        retrieved_ids = [
        result[0] for result in results
        ]

        hit = any(chunk_id in relevant for chunk_id in retrieved_ids)

        rank = next(
            (
                i
                for i, chunk_id in enumerate(retrieved_ids, start=1)
                if chunk_id in relevant
            ),
            None,
        )

        reciprocal_rank = 1 / rank if rank else 0

        hits += int(hit)
        reciprocal_ranks.append(reciprocal_rank)
        evaluated += 1

        print(f"\n{item['id']}: {item['question']}")
        print(f"Relevant: {relevant}")
        print(f"Dense retrieved: {retrieved_ids}")
        print(f"Hit@{K}: {hit}")
        print(f"Reciprocal rank: {reciprocal_rank}")

    if evaluated == 0:
        print("No answerable questions with relevance labels were found.")
        return

    metrics = {
        "questions_evaluated": evaluated,
        "hit_at_5": hits / evaluated,
        "mrr": sum(reciprocal_ranks) / evaluated,
    }

    output_file = PROJECT_ROOT / "eval" / "dense_metrics.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\n--- Dense-only Summary ---")
    print(f"Questions evaluated: {evaluated}")
    print(f"Hit@{K}: {metrics['hit_at_5']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
