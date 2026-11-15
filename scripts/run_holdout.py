
import json
from pathlib import Path

from app.pipeline import RAGPipeline
from scripts.run_eval import run_eval

ROOT = Path(__file__).resolve().parent.parent

HOLDOUT_PATH = ROOT / "app" / "eval" / "holdout.json"
RESULTS_PATH = ROOT / "app" / "eval" / "holdout_results.json"

if __name__ == "__main__":
    pipeline = RAGPipeline(ROOT / "data" / "docs")
    pipeline.build_index()

    summary = run_eval(
        pipeline,
        HOLDOUT_PATH,
        k=5,
        mode="hybrid+rerank"
    )

    print(json.dumps(summary, indent=2))

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {RESULTS_PATH}")
