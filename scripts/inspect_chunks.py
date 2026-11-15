
from pathlib import Path
from app.pipeline import RAGPipeline

root = Path(__file__).resolve().parent.parent

pipeline = RAGPipeline(root / "data" / "docs")
print(pipeline.build_index())

for chunk in pipeline.chunks[:10]:
    print(f"\nChunk ID: {chunk.chunk_id}")
    print(f"Source: {getattr(chunk, 'source', 'unknown')}")
    print(chunk.text)
