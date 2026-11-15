import numpy as np
import hashlib
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from app.ingestion.chunker import Chunk

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

class EmbeddingRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def _cache_key(self, chunks: list[Chunk]) -> str:
        content = json.dumps(
            [c.text for c in chunks],
            ensure_ascii=False
        )
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / f"embeddings_{key}.npy"

    
    def index(self, chunks, force: bool = False) -> None:
        key = self._cache_key(chunks)
        cache_path = self._cache_path(key)

        if cache_path.exists() and not force:
            print(f"Loading cached embeddings ({key})")
            self.matrix = np.load(cache_path)
            self.chunks = chunks
            return

        print(f"Embedding {len(chunks)} chunks...")
        texts = [c.text for c in chunks]

        self.matrix = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        )

        # Critical: keep embedding rows aligned with their chunks
        self.chunks = chunks

        np.save(cache_path, self.matrix)
        print(f"Cached to {cache_path}")

        assert self.matrix.shape[0] == len(self.chunks), (
            f"Embedding/chunk mismatch: "
            f"{self.matrix.shape[0]} embeddings vs "
            f"{len(self.chunks)} chunks"
        )


    def search(
        self, query: str, k: int = 5
    ) -> list[tuple[int, float, str]]:
        if self.matrix is None:
            raise RuntimeError("Call index() before search()")

        query_vec = self.model.encode(
            query, normalize_embeddings=True
        )
        scores = self.matrix @ query_vec
        indices = np.argsort(-scores)[:k]

        return [
            (int(i), float(scores[i]), self.chunks[i].text)
            for i in indices
        ]