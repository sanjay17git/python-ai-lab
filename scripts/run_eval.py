import json
import time
from pathlib import Path
from app.pipeline import RAGPipeline
from app.eval.metrics import compute_metrics

def run_eval(
    pipeline: RAGPipeline,
    eval_path: Path,
    k: int = 5,
    candidate_k: int = 20,
    mode: str = "hybrid"   # "bm25", "dense", "hybrid", "hybrid+rerank"
) -> dict:
    with open(eval_path) as f:
        questions = json.load(f)

    # filter out negative questions for retrieval eval
    answerable = [q for q in questions if q["relevant_chunks"]]
    negative   = [q for q in questions if not q["relevant_chunks"]]

    all_metrics = []
    per_type    = {}
    per_difficulty = {}

    for q in answerable:
        relevant = set(q["relevant_chunks"])

        if mode == "bm25":
            results = pipeline.bm25_retriever.search(q["question"], k=candidate_k)
        elif mode == "dense":
            results = pipeline.embedding_retriever.search(q["question"], k=candidate_k)
        elif mode == "hybrid":
            results = pipeline.hybrid_search(q["question"], k=candidate_k, rerank=False)
        elif mode == "hybrid+rerank":
            results = pipeline.search(q["question"], k=k, candidate_k=candidate_k)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        ranked_ids = [doc_id for doc_id, _, _ in results]
        metrics = compute_metrics(ranked_ids, relevant, k=k)
        metrics["id"] = q["id"]
        metrics["type"] = q["type"]
        metrics["difficulty"] = q.get("difficulty", "unspecified")
        all_metrics.append(metrics)

        # group by type
        t = q["type"]
        per_type.setdefault(t, []).append(metrics)

        # group by difficulty
        d = q.get("difficulty", "unspecified")
        per_difficulty.setdefault(d, []).append(metrics)

    def avg(lst, key):
        return round(sum(m[key] for m in lst) / len(lst), 4) if lst else 0.0

    k_label = f"Hit@{k}"
    summary = {
        "mode": mode,
        "questions_evaluated": len(answerable),
        "negative_questions_skipped": len(negative),
        k_label: avg(all_metrics, k_label),
        f"Recall@{k}": avg(all_metrics, f"Recall@{k}"),
        "MRR": avg(all_metrics, "MRR"),
        "MAP": avg(all_metrics, "AP"),
        "by_type": {
            t: {
                k_label: avg(ms, k_label),
                "MRR": avg(ms, "MRR")
            }
            for t, ms in per_type.items()
        },
        "by_difficulty": {
            d: {
                k_label: avg(ms, k_label),
                "MRR": avg(ms, "MRR")
            }
            for d, ms in per_difficulty.items()
        }
    }

    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent

    pipeline = RAGPipeline(root / "data" / "docs")
    pipeline.build_index()

    # Step 2: Locate tuning dataset inside app/eval
    eval_path = root / "app" / "eval" / "tuning.json"

    modes = ["bm25", "dense", "hybrid", "hybrid+rerank"]

    print(f"\n{'Mode':<20} {'Hit@5':<10} {'MRR':<10} {'MAP':<10}")
    print("-" * 50)

    results = {}

    for mode in modes:
        summary = run_eval(pipeline, eval_path, k=5, mode=mode)
        results[mode] = summary

        print(
            f"{mode:<20} "
            f"{summary['Hit@5']:<10} "
            f"{summary['MRR']:<10} "
            f"{summary['MAP']:<10}"
        )

    # Step 3: Save results inside app/eval
    results_path = root / "app" / "eval" / "tuning_results.json"

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nFull results saved to {results_path}")
