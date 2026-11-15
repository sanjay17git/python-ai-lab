
import json
import time
from pathlib import Path

from app.pipeline import RetrievalPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = PROJECT_ROOT / "eval" / "eval_set.json"
DOC_DIR = PROJECT_ROOT / "data" / "docs"
RESULT_FILE = PROJECT_ROOT / "eval" / "eval_results.json"


def main():
    with EVAL_FILE.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    pipeline = RetrievalPipeline()
    index_info = pipeline.index_documents(str(DOC_DIR))

    print(f"Indexed chunks: {index_info['chunks_indexed']}")
    print(f"Questions to evaluate: {len(questions)}\n")

    results = []

    for item in questions:
        question = item["question"]
        expected = item["answer"]
        start = time.perf_counter()

        try:
            output = pipeline.ask(question)
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            result = {
                "id": item["id"],
                "question": question,
                "expected_answer": expected,
                "generated_answer": output["answer"],
                "expected_source": item["source_file"],
                "retrieved_citations": output["citations"],
                "latency_ms": latency_ms,
            }

            results.append(result)

            print(f"--- {item['id']} ---")
            print(f"Question: {question}")
            print(f"Expected: {expected}")
            print(f"Generated: {output['answer']}")
            print(f"Citations: {output['citations']}")
            print(f"Latency: {latency_ms} ms\n")

        except Exception as exc:
            print(f"ERROR on {item['id']}: {exc}\n")
            results.append({
                "id": item["id"],
                "question": question,
                "error": str(exc),
            })

    with RESULT_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()
