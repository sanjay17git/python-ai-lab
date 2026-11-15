import json
import sys
from pathlib import Path

# make sure app/ is importable
sys.path.append(str(Path(__file__).parent.parent))
ROOT = Path(__file__).resolve().parent.parent
from app.pipeline import RAGPipeline

REFUSAL_PHRASES = [
    "I cannot answer this from the provided documents",
    "I don't have enough information",
    "not enough information",
    "cannot answer this question",
    "context does not contain",
]


def is_refusal(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(phrase.lower() in answer_lower
               for phrase in REFUSAL_PHRASES)

def eval_refusal(pipeline: RAGPipeline, eval_path: Path) -> dict:
    with open(eval_path) as f:
        questions = json.load(f)

    negative = [q for q in questions if not q["relevant_chunks"]]

    if not negative:
        print("No negative questions found in eval set.")
        return {}

    refusals = 0
    hallucinations = []

    for q in negative:
        result = pipeline.ask(q["question"], k=3)
        answer = result["answer"]

        if is_refusal(answer):
            refusals += 1
            print(f"✓ REFUSED  [{q['id']}]: {q['question'][:60]}")
        else:
            hallucinations.append(q["id"])
            print(f"✗ HALLUCINATED [{q['id']}]: {q['question'][:60]}")
            print(f"  Answer: {answer[:120]}")

    summary = {
        "negative_questions": len(negative),
        "correct_refusals": refusals,
        "hallucinations": hallucinations,
        "refusal_accuracy": round(refusals / len(negative), 4)
    }

    print(f"\n--- Refusal Accuracy ---")
    print(f"Correct refusals : {refusals}/{len(negative)}")
    print(f"Hallucinations   : {len(hallucinations)}")
    print(f"Refusal accuracy : {summary['refusal_accuracy']}")

    RESULTS_PATH = ROOT / "app" / "eval" / "refusal_results.json"

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved to eval/refusal_results.json")
    return summary


if __name__ == "__main__":
    pipeline = RAGPipeline(Path("data/docs"))
    pipeline.build_index()
    ROOT = Path(__file__).resolve().parent.parent
    EVAL_PATH = ROOT / "app" / "eval" / "tuning.json"

    eval_refusal(pipeline, EVAL_PATH)