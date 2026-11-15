
import json
from pathlib import Path

from app.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "app" / "eval" / "tuning.json"
DOC_PATH = ROOT / "data" / "docs"

pipeline = RAGPipeline(DOC_PATH)
pipeline.build_index()

with open(EVAL_PATH, encoding="utf-8") as f:
    questions = json.load(f)

answerable = [
    q for q in questions
    if q["relevant_chunks"]
][:10]

for q in answerable:
    result = pipeline.ask(q["question"], k=3)

    print(f"\n=== {q['id']} ===")
    print(f"Q: {q['question']}")
    print(f"A: {result['answer'][:200]}")
    print(f"Expected chunks: {q['relevant_chunks']}")
    print(f"Citations returned: {len(result['citations'])}")
