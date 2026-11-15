
import json
import re
from pathlib import Path

from app.pipeline import RAGPipeline


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def find_matching_chunks(passage, chunks, threshold=0.75):
    passage = normalize(passage)

    if len(passage) < 20:
        return []

    passage_tokens = set(re.findall(r"\b\w+\b", passage))
    if not passage_tokens:
        return []

    matches = []

    for chunk in chunks:
        chunk_tokens = set(
            re.findall(r"\b\w+\b", normalize(chunk.text))
        )

        overlap = len(passage_tokens & chunk_tokens) / len(passage_tokens)

        if overlap >= threshold:
            matches.append(chunk.chunk_id)

    return matches


def label_chunks(eval_path, pipeline):
    eval_path = Path(eval_path)

    with eval_path.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    for q in questions:
        passage = q.get("source_passage", "").strip()

        # Negative questions should have no relevant chunks.
        if q.get("type") == "negative":
            q["relevant_chunks"] = []
            print(f"NEGATIVE: {q['id']} -> []")
            continue

        # Preserve existing labels if no source passage is available.
        if not passage:
            print(
                f"PRESERVED: {q['id']} -> "
                f"{q.get('relevant_chunks', [])} "
                "(no source passage)"
            )
            continue

        matched = find_matching_chunks(passage, pipeline.chunks)

        if matched:
            q["relevant_chunks"] = matched
            print(f"CANDIDATE: {q['id']} -> {matched}")
        else:
            print(
                f"REVIEW NEEDED: {q['id']} - "
                "no matching chunk found; existing labels preserved"
            )

    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print("Finished. Candidate labels still require manual verification.")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent

    pipeline = RAGPipeline(root / "data" / "docs")
    print(pipeline.build_index())

    label_chunks(root / "eval" / "tuning.json", pipeline)
