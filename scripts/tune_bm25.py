import json
from pathlib import Path
from app.pipeline import RAGPipeline
from scripts.run_eval import run_eval

pipeline = RAGPipeline(Path("data/docs"))
pipeline.build_index()

k1_values = [0.5, 1.2, 1.5, 2.0]
b_values  = [0.5, 0.75, 1.0]

results = []

print(f"{'k1':<8} {'b':<8} {'Hit@5':<10} {'MRR':<10} {'MAP'}")
print("-" * 46)

for k1 in k1_values:
    for b in b_values:
        # rebuild BM25 index with new parameters
        pipeline.bm25_retriever.k1 = k1
        pipeline.bm25_retriever.b  = b
        pipeline.bm25_retriever.index(pipeline.chunks)

        root = Path(__file__).resolve().parent.parent

        eval_path = root / "app" / "eval" / "tuning.json"

        summary = run_eval(
            pipeline,
            eval_path,
            k=5,
            mode="bm25"
        )

        results.append({"k1": k1, "b": b, **summary})
        print(
            f"{k1:<8} {b:<8} "
            f"{summary['Hit@5']:<10} "
            f"{summary['MRR']:<10} "
            f"{summary['MAP']}"
        )

# find best by MRR
best = max(results, key=lambda x: x["MRR"])
print(f"\nBest: k1={best['k1']}, b={best['b']}, MRR={best['MRR']}")

root = Path(__file__).resolve().parent.parent
output_path = root / "app" / "eval" / "bm25_tuning.json"

with output_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)