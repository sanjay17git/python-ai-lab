
from pathlib import Path

from app.pipeline import RAGPipeline


def test_pipeline_builds_index():
    pipeline = RAGPipeline(Path("data/docs"))

    result = pipeline.build_index()

    assert result["documents"] == 2
    assert result["chunks"] > 0


def test_pipeline_search_returns_valid_chunks():
    pipeline = RAGPipeline(Path("data/docs"))
    pipeline.build_index()

    results = pipeline.search(
        "What is the minimum attendance percentage required?",
        k=5,
    )

    assert results, "Search returned no results"

    for chunk_id, score, text in results:
        assert 0 <= chunk_id < len(pipeline.chunks)
        assert isinstance(score, (int, float))
        assert isinstance(text, str)
        assert text.strip()
