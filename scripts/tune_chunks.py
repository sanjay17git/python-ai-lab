
import json
from pathlib import Path

from app.pipeline import RAGPipeline
from app.ingestion.loader import load_documents
from app.ingestion.cleaner import clean_document
from app.ingestion.chunker import chunk_documents
from scripts.run_eval import run_eval


# Project paths
ROOT = Path(__file__).resolve().parent.parent
DOCS_PATH = ROOT / "data" / "docs"
EVAL_PATH = ROOT / "app" / "eval" / "tuning.json"
OUTPUT_PATH = ROOT / "app" / "eval" / "chunk_tuning.json"

docs = load_documents(DOCS_PATH)
docs = [clean_document(d) for d in docs]

configs = [
    {"size": 75, "overlap": 15},
    {"size": 100, "overlap": 20},
    {"size": 150, "overlap": 30},
    {"size": 200, "overlap": 40},
    {"size": 250, "overlap": 50},
]

pipeline = RAGPipeline(DOCS_PATH)
pipeline.build_index()

print(
    f"{'Size':<8} {'Overlap':<10} {'Chunks':<10} "
    f"{'Hit@5':<10} {'MRR':<10} {'MAP'}"
)
print("-" * 58)

results = []

for cfg in configs:
    chunks = chunk_documents(docs, **cfg)

    pipeline.chunks = chunks

    # Rebuild dense embeddings for this configuration
    pipeline.embedding_retriever.index(chunks, force=True)

    # Best BM25 parameters from the previous experiment
    pipeline.bm25_retriever.k1 = 1.2
    pipeline.bm25_retriever.b = 0.5
    pipeline.bm25_retriever.index(chunks)

    summary = run_eval(
        pipeline,
        EVAL_PATH,
        k=5,
        mode="hybrid+rerank",
    )

    results.append({
        "config": cfg,
        "chunks": len(chunks),
        **summary,
    })

    print(
        f"{cfg['size']:<8} {cfg['overlap']:<10} "
        f"{len(chunks):<10} "
        f"{summary['Hit@5']:<10.4f} "
        f"{summary['MRR']:<10.4f} "
        f"{summary['MAP']:.4f}"
    )

best = max(results, key=lambda x: x["MRR"])

print(
    f"\nBest: size={best['config']['size']}, "
    f"overlap={best['config']['overlap']}, "
    f"MRR={best['MRR']:.4f}"
)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {OUTPUT_PATH}")
